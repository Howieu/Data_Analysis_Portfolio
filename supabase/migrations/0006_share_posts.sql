-- 0006_share_posts.sql — 频道：编辑精选购物分享帖
-- 仅编辑精选，不做开放 UGC，帖子挂真实 SKU
-- 字段与导入数据一一对应，便于核对

create table if not exists posts (
  id uuid primary key default gen_random_uuid(),
  slug text unique, -- 与导入数据的 id 一致（如 post-001），便于跨环境核对，可空
  title text not null,
  body text not null,
  cover_url text, -- 落盘后的封面图 URL；未爬取阶段可为空，前端按 cover_hue 显示底色
  gallery_urls jsonb not null default '[]',
  tags jsonb not null default '[]', -- text[] 用 jsonb 存储，便于跟前端 tag 数组直接对应
  budget_hint_cents integer check (budget_hint_cents >= 0), -- 帖内标注的预算参考，可空
  source text not null default 'editorial' check (source in ('editorial')), -- 首版仅编辑精选，禁止伪装第三方转载
  author_label text not null default '编辑精选',
  status text not null default 'published' check (status in ('draft', 'published', 'archived')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists posts_status_idx on posts (status, created_at desc);

-- 帖子关联 SKU：一帖可挂 1-3 个真实 SKU，保留展示顺序
create table if not exists post_skus (
  post_id uuid not null references posts(id) on delete cascade,
  sku_id uuid not null references skus(id) on delete cascade,
  sort_order integer not null default 0,
  primary key (post_id, sku_id)
);
create index if not exists post_skus_sku_idx on post_skus (sku_id);

comment on table posts is '频道购物分享帖（编辑精选，非开放社区）；帖 → 买通过 post_skus 关联真实 SKU';
comment on column posts.source is '固定为 editorial：不接入第三方内容，避免伪装转载的合规风险';

-- RLS：与商品域一致——所有人可读已发布内容，仅 admin 可写（沿用 0002_rls.sql 的 is_admin()）
alter table posts enable row level security;
alter table post_skus enable row level security;

create policy "posts readable when published" on posts for select
  using (status = 'published' or public.is_admin());
create policy "posts admin write" on posts for all
  using (public.is_admin()) with check (public.is_admin());

create policy "post_skus readable by all" on post_skus for select
  using (exists (select 1 from posts p where p.id = post_id and (p.status = 'published' or public.is_admin())));
create policy "post_skus admin write" on post_skus for all
  using (public.is_admin()) with check (public.is_admin());
