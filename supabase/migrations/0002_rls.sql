-- 0002_rls.sql — RLS 与权限 helper
-- 原则：
--   * 商品域（categories/defs/skus/attrs/coupons/shipping）：所有人（含 anon）可读，仅 admin 可写
--   * 用户数据表：用户只能读写自己的行
--   * strategy/experiment/audit/cold_start/merge：仅 admin（业务写路径走 service role，天然绕过 RLS）
--   * 服务端（service_role key）绕过 RLS，承担引擎与后台写入

-- is_admin()：security definer，避免策略递归
create or replace function public.is_admin()
returns boolean
language sql
security definer
set search_path = public
stable
as $$
  select exists (
    select 1 from user_profiles
    where id = auth.uid() and role = 'admin'
  );
$$;

-- 新用户注册时自动建 profile（含匿名用户）
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.user_profiles (id, is_anonymous)
  values (new.id, coalesce((new.raw_app_meta_data->>'provider') = 'anonymous', false))
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- 防止普通用户自提权限：role 变化仅 admin 可执行
create or replace function public.prevent_role_escalation()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if new.role is distinct from old.role and auth.uid() is not null and not public.is_admin() then
    raise exception 'only admin can change role';
  end if;
  return new;
end;
$$;

create trigger user_profiles_role_guard
  before update on user_profiles
  for each row execute function public.prevent_role_escalation();

-- ============ 商品域：公开读，admin 写 ============

alter table categories enable row level security;
alter table category_attribute_defs enable row level security;
alter table skus enable row level security;
alter table sku_attribute_values enable row level security;
alter table coupons enable row level security;
alter table sku_shipping enable row level security;

create policy "categories readable by all" on categories for select using (true);
create policy "categories admin write" on categories for all using (is_admin()) with check (is_admin());

create policy "attribute defs readable by all" on category_attribute_defs for select using (true);
create policy "attribute defs admin write" on category_attribute_defs for all using (is_admin()) with check (is_admin());

create policy "skus readable by all" on skus for select using (true);
create policy "skus admin write" on skus for all using (is_admin()) with check (is_admin());

create policy "attribute values readable by all" on sku_attribute_values for select using (true);
create policy "attribute values admin write" on sku_attribute_values for all using (is_admin()) with check (is_admin());

create policy "coupons readable by all" on coupons for select using (true);
create policy "coupons admin write" on coupons for all using (is_admin()) with check (is_admin());

create policy "shipping readable by all" on sku_shipping for select using (true);
create policy "shipping admin write" on sku_shipping for all using (is_admin()) with check (is_admin());

-- ============ 用户数据：仅本人 ============

alter table user_profiles enable row level security;
create policy "profiles read own or admin" on user_profiles for select
  using (auth.uid() = id or is_admin());
create policy "profiles update own" on user_profiles for update
  using (auth.uid() = id) with check (auth.uid() = id);
-- insert 由 handle_new_user trigger（security definer）完成

alter table conversations enable row level security;
create policy "conversations owner all" on conversations for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "conversations admin read" on conversations for select using (is_admin());

alter table messages enable row level security;
create policy "messages owner all" on messages for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "messages admin read" on messages for select using (is_admin());

alter table constraint_versions enable row level security;
create policy "constraint_versions owner all" on constraint_versions for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "constraint_versions admin read" on constraint_versions for select using (is_admin());

alter table favorites enable row level security;
create policy "favorites owner all" on favorites for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

alter table events enable row level security;
create policy "events read own or admin" on events for select
  using (auth.uid() = user_id or is_admin());
create policy "events insert own or anonymous" on events for insert
  with check (user_id is null or auth.uid() = user_id);
-- 事件不允许 update/delete（审计完整性），service role 例外

alter table recommendations enable row level security;
create policy "recommendations read own or admin" on recommendations for select
  using (auth.uid() = user_id or is_admin());
create policy "recommendations insert own" on recommendations for insert
  with check (user_id is null or auth.uid() = user_id);
create policy "recommendations update own" on recommendations for update
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

alter table recommendation_items enable row level security;
create policy "recommendation_items read own or admin" on recommendation_items for select
  using (is_admin() or exists (
    select 1 from recommendations r
    where r.id = recommendation_id and r.user_id = auth.uid()
  ));
-- 写入走 service role

alter table recommendation_traces enable row level security;
create policy "recommendation_traces read own or admin" on recommendation_traces for select
  using (is_admin() or exists (
    select 1 from recommendations r
    where r.id = recommendation_id and r.user_id = auth.uid()
  ));
-- 写入走 service role

alter table mock_orders enable row level security;
create policy "mock_orders owner read" on mock_orders for select
  using (auth.uid() = user_id or is_admin());
create policy "mock_orders owner insert" on mock_orders for insert
  with check (auth.uid() = user_id);
create policy "mock_orders owner update" on mock_orders for update
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

alter table mock_order_items enable row level security;
create policy "mock_order_items read own or admin" on mock_order_items for select
  using (is_admin() or exists (
    select 1 from mock_orders o
    where o.id = order_id and o.user_id = auth.uid()
  ));
create policy "mock_order_items insert own" on mock_order_items for insert
  with check (exists (
    select 1 from mock_orders o
    where o.id = order_id and o.user_id = auth.uid()
  ));

-- ============ 策略 / 实验 / 冷启动 / 合并审计：仅 admin ============

alter table strategy_configs enable row level security;
create policy "strategy_configs admin only" on strategy_configs for all
  using (is_admin()) with check (is_admin());

alter table strategy_audit_logs enable row level security;
create policy "strategy_audit_logs admin only" on strategy_audit_logs for all
  using (is_admin()) with check (is_admin());

alter table experiments enable row level security;
create policy "experiments admin only" on experiments for all
  using (is_admin()) with check (is_admin());

alter table experiment_assignments enable row level security;
create policy "experiment_assignments admin only" on experiment_assignments for all
  using (is_admin()) with check (is_admin());

alter table cold_start_states enable row level security;
create policy "cold_start_states admin only" on cold_start_states for all
  using (is_admin()) with check (is_admin());

alter table account_merges enable row level security;
create policy "account_merges admin only" on account_merges for all
  using (is_admin()) with check (is_admin());
