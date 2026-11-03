-- 0004：策略默认配置种子 + 模型调用日志表（admin 只读）
-- 当前生效 = 该 key 最大 version；此处以 version=1 落默认值。

insert into strategy_configs (key, value, version) values
  ('new_exploration_quota', '1', 1),
  ('price_ttl_hours', '24', 1),
  ('rerank_enabled', 'true', 1),
  ('max_results', '6', 1),
  ('featured_results', '3', 1),
  ('baseline_experiment_enabled', 'false', 1)
on conflict (key, version) do nothing;

-- 模型调用日志：与引擎 ModelCallLog（lib/engine/types.ts）对齐。
-- 写入方：/api/chat（service role，绕过 RLS）；admin 只读。
create table if not exists model_calls (
  id bigint generated always as identity primary key,
  recommendation_id uuid references recommendations(id),
  stage text not null,                    -- extract / clarify / rerank / explain 等链路环节
  provider text not null,                 -- kimi / openrouter
  model text,                             -- 可为 null（ModelCallLog.model: string | null）
  prompt_version text not null,           -- ModelCallLog.promptVersion
  latency_ms integer check (latency_ms >= 0),
  ok boolean not null default true,
  degraded boolean not null default false, -- 是否降级调用（ModelCallLog.degraded）
  error text,
  created_at timestamptz not null default now()
);
create index if not exists model_calls_created_idx on model_calls (created_at desc);

alter table model_calls enable row level security;

-- 读：仅 admin；写：无客户端策略（service role 绕过 RLS 写入）
drop policy if exists "model_calls admin read" on model_calls;
create policy "model_calls admin read" on model_calls
  for select using (is_admin());
