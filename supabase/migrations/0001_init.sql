-- 0001_init.sql — 「知选」AI 导购数据基础
-- 约定：
--   * 金额一律 integer cents，字段后缀 _cents，禁止 float
--   * link_status ∈ {ok, broken, pending}：broken 硬过滤，pending 可进推荐但前端标"待核验"
--   * 价格快照 24h 有效期，以 price_snapshot_at 计算；seed 用 now()，生产由人工核验刷新
--   * 推荐 trace 统一表 recommendation_traces，stage ∈ recall/price/filter/rank/rerank/final_check
--   * 语义召回：skus.embedding vector(1536) 可空，为空时引擎侧降级

create extension if not exists vector;

-- ============ 用户 ============

create table user_profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  role text not null default 'user' check (role in ('user', 'admin')),
  is_anonymous boolean not null default false,
  display_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 账户合并审计（merge_anonymous_data 写入）
create table account_merges (
  id bigint generated always as identity primary key,
  old_user_id uuid not null,
  new_user_id uuid not null,
  migrated jsonb not null default '{}', -- {table: row_count}
  merged_at timestamptz not null default now(),
  unique (old_user_id)
);

-- ============ 对话与约束 ============

create table conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text,
  status text not null default 'active' check (status in ('active', 'archived', 'deleted')),
  -- 原始对话保留 30 天：purge_after 之后仅保留脱敏约束与统计
  purge_after timestamptz not null default (now() + interval '30 days'),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index conversations_user_idx on conversations (user_id, created_at desc);

create table messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  created_at timestamptz not null default now()
);
create index messages_conversation_idx on messages (conversation_id, created_at);

-- 约束版本：每次约束修改生成新版本；superseded_at 非空即失效
create table constraint_versions (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  version integer not null check (version > 0),
  constraints jsonb not null, -- {category_slug, budget_max_cents, quantity, preferences{...}}
  superseded_at timestamptz,
  created_at timestamptz not null default now(),
  unique (conversation_id, version)
);
create index constraint_versions_user_idx on constraint_versions (user_id, created_at desc);

-- ============ 品类与动态属性 Schema ============

create table categories (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  name text not null,
  sort_order integer not null default 0,
  created_at timestamptz not null default now()
);

-- 动态品类属性定义（配置驱动，新增品类不改代码）
create table category_attribute_defs (
  id uuid primary key default gen_random_uuid(),
  category_id uuid not null references categories(id) on delete cascade,
  key text not null,
  label text not null,
  type text not null check (type in ('number', 'text', 'boolean', 'enum')),
  unit text,                 -- 如 mAh / dB / g / bar
  options jsonb,             -- type=enum 时的合法取值数组
  is_hard_filter boolean not null default false, -- 数值硬约束标记
  evidence_required boolean not null default true,
  sort_order integer not null default 0,
  unique (category_id, key)
);

-- ============ 商品 ============

create table skus (
  id uuid primary key default gen_random_uuid(),
  category_id uuid not null references categories(id),
  name text not null,
  brand text not null,
  model text,
  price_cents integer not null check (price_cents >= 0),
  price_snapshot_at timestamptz not null, -- 24h 有效期由此计算；过期不得进入最终推荐
  source_url text not null,
  link_status text not null default 'pending' check (link_status in ('ok', 'broken', 'pending')),
  embedding vector(1536), -- 可空；为空时语义召回降级（引擎侧处理）
  verified_at timestamptz,           -- 最近人工核验时间
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index skus_category_link_idx on skus (category_id, link_status);

-- 参数事实 + 证据（参数缺失不等于满足；AI 推断不能作为参数事实）
create table sku_attribute_values (
  id uuid primary key default gen_random_uuid(),
  sku_id uuid not null references skus(id) on delete cascade,
  attribute_def_id uuid not null references category_attribute_defs(id) on delete cascade,
  value jsonb not null, -- number/string/boolean，与 def.type 一致
  evidence text,        -- 证据文本（如"商品详情页标注 4000mAh"）
  source_url text,      -- 证据来源链接
  evidence_level text not null default 'listing_page'
    check (evidence_level in ('official_spec', 'listing_page', 'manual_review')),
  created_at timestamptz not null default now(),
  unique (sku_id, attribute_def_id)
);
create index sku_attribute_values_sku_idx on sku_attribute_values (sku_id);

-- 优惠券：通用券可计入预算；条件券默认不计入，用户确认资格后才重算
create table coupons (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  type text not null check (type in ('universal', 'conditional')),
  condition_desc text, -- type=conditional 时的条件说明（会员/新人/地区等）
  amount_cents integer not null check (amount_cents >= 0),
  min_spend_cents integer not null default 0 check (min_spend_cents >= 0), -- 0 = 无门槛
  valid_from timestamptz not null,
  valid_to timestamptz not null,
  applies_to text not null default 'sku' check (applies_to in ('all', 'category', 'sku')),
  category_id uuid references categories(id),
  sku_id uuid references skus(id),
  check (valid_to > valid_from),
  check (
    (applies_to = 'all' and category_id is null and sku_id is null)
    or (applies_to = 'category' and category_id is not null and sku_id is null)
    or (applies_to = 'sku' and sku_id is not null)
  ),
  created_at timestamptz not null default now()
);
create index coupons_sku_idx on coupons (sku_id);
create index coupons_valid_idx on coupons (valid_from, valid_to);

-- 运费规则：0 = 包邮
create table sku_shipping (
  sku_id uuid primary key references skus(id) on delete cascade,
  shipping_cents integer not null default 0 check (shipping_cents >= 0),
  free_over_cents integer check (free_over_cents >= 0), -- 满额包邮门槛，null = 无
  note text
);

-- ============ 用户行为 ============

create table favorites (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  sku_id uuid not null references skus(id) on delete cascade,
  created_at timestamptz not null default now(),
  unique (user_id, sku_id)
);

-- 访客与注册用户统一事件模型；session_id 承载匿名会话标识
create table events (
  id bigint generated always as identity primary key,
  user_id uuid references auth.users(id) on delete cascade, -- 匿名事件可为 null
  session_id text not null,
  event_type text not null,
  payload jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create index events_user_idx on events (user_id, created_at desc);
create index events_session_idx on events (session_id, created_at desc);

-- ============ 推荐 ============

create table recommendations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade, -- 匿名推荐可为 null
  session_id text,
  conversation_id uuid references conversations(id) on delete set null,
  constraint_version_id uuid references constraint_versions(id) on delete set null,
  status text not null default 'active' check (status in ('active', 'stale', 'superseded')),
  created_at timestamptz not null default now()
);
create index recommendations_user_idx on recommendations (user_id, created_at desc);

create table recommendation_items (
  id uuid primary key default gen_random_uuid(),
  recommendation_id uuid not null references recommendations(id) on delete cascade,
  sku_id uuid not null references skus(id),
  rank integer not null check (rank > 0),
  badge text check (badge in ('best_match', 'best_value', 'feature_pick')), -- 最匹配/性价比/特色选择
  explanation text,
  -- 价格快照（推荐时的计算结果，单位分）
  price_cents integer check (price_cents >= 0),
  coupon_cents integer check (coupon_cents >= 0),
  shipping_cents integer check (shipping_cents >= 0),
  payable_cents integer check (payable_cents >= 0),
  unique (recommendation_id, sku_id)
);
create index recommendation_items_rec_idx on recommendation_items (recommendation_id, rank);

-- 推荐全链路 trace（一张统一表）
create table recommendation_traces (
  id bigint generated always as identity primary key,
  recommendation_id uuid not null references recommendations(id) on delete cascade,
  stage text not null check (stage in ('recall', 'price', 'filter', 'rank', 'rerank', 'final_check')),
  payload jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create index recommendation_traces_rec_stage_idx on recommendation_traces (recommendation_id, stage);

-- ============ 模拟交易（不真实扣款发货，明确标注） ============

create table mock_orders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  idempotency_key text not null unique,
  recommendation_id uuid references recommendations(id) on delete set null,
  constraint_version_id uuid references constraint_versions(id) on delete set null,
  status text not null default 'created' check (status in ('created', 'paid', 'cancelled')),
  -- 订单级金额快照（单位分）
  original_total_cents integer not null check (original_total_cents >= 0),
  coupon_total_cents integer not null default 0 check (coupon_total_cents >= 0),
  shipping_total_cents integer not null default 0 check (shipping_total_cents >= 0),
  payable_total_cents integer not null check (payable_total_cents >= 0),
  created_at timestamptz not null default now()
);
create index mock_orders_user_idx on mock_orders (user_id, created_at desc);

create table mock_order_items (
  id uuid primary key default gen_random_uuid(),
  order_id uuid not null references mock_orders(id) on delete cascade,
  sku_id uuid not null references skus(id),
  quantity integer not null check (quantity > 0),
  -- 下单时价格快照（单位分），与当时价格引擎输出一致
  unit_price_cents integer not null check (unit_price_cents >= 0),
  coupon_cents integer not null default 0 check (coupon_cents >= 0),
  shipping_cents integer not null default 0 check (shipping_cents >= 0),
  item_total_cents integer not null check (item_total_cents >= 0),
  price_snapshot_at timestamptz not null,
  created_at timestamptz not null default now()
);
create index mock_order_items_order_idx on mock_order_items (order_id);

-- ============ 策略、实验与冷启动 ============

-- 策略配置：key + 递增 version；当前生效 = 该 key 最大 version
create table strategy_configs (
  id bigint generated always as identity primary key,
  key text not null,
  value jsonb not null,
  version integer not null check (version > 0),
  updated_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  unique (key, version)
);

create table strategy_audit_logs (
  id bigint generated always as identity primary key,
  config_key text not null,
  action text not null check (action in ('create', 'update', 'rollback')),
  old_value jsonb,
  new_value jsonb,
  actor uuid references auth.users(id),
  created_at timestamptz not null default now()
);

create table experiments (
  id uuid primary key default gen_random_uuid(),
  key text not null unique,
  name text not null,
  status text not null default 'draft' check (status in ('draft', 'running', 'paused', 'ended')),
  variants jsonb not null, -- [{name, weight, config_overrides}]
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table experiment_assignments (
  id bigint generated always as identity primary key,
  experiment_id uuid not null references experiments(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  session_id text,
  variant text not null,
  assigned_at timestamptz not null default now(),
  check (user_id is not null or session_id is not null)
);
create unique index experiment_assignments_user_uniq
  on experiment_assignments (experiment_id, user_id) where user_id is not null;
create unique index experiment_assignments_session_uniq
  on experiment_assignments (experiment_id, session_id) where session_id is not null;

-- 冷启动状态机：cold -> exploring -> established
create table cold_start_states (
  id bigint generated always as identity primary key,
  entity_type text not null check (entity_type in ('user', 'sku')),
  entity_id uuid not null,
  state text not null default 'cold' check (state in ('cold', 'exploring', 'established')),
  signals jsonb not null default '{}', -- 曝光/点击/转化计数等
  entered_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (entity_type, entity_id)
);
