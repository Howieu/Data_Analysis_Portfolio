-- 0003_functions.sql — 匿名身份合并兜底函数
-- 主路径：Supabase anonymous sign-in 后 updateUser({email, password}) 转正式账号，
-- user_id 不变即天然合并。本函数仅作为跨账号兜底：
-- 事务内迁移 conversations/favorites/events/mock_orders/recommendations，
-- 幂等（同一 old_user_id 只执行一次），并写 account_merges 审计。

create or replace function public.merge_anonymous_data(
  old_user_id uuid,
  new_user_id uuid
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  counts jsonb;
  n integer;
begin
  -- 幂等：已合并过直接返回历史记录
  select migrated into counts from account_merges where account_merges.old_user_id = merge_anonymous_data.old_user_id;
  if found then
    return jsonb_build_object('status', 'already_merged', 'migrated', counts);
  end if;

  if old_user_id = new_user_id then
    raise exception 'old_user_id and new_user_id must differ';
  end if;

  counts := '{}'::jsonb;

  -- conversations（messages / constraint_versions 通过 conversation_id 级联跟随，
  -- 但 messages/constraint_versions 上冗余的 user_id 需同步）
  update conversations set user_id = new_user_id where conversations.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('conversations', n);

  update messages set user_id = new_user_id where messages.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('messages', n);

  update constraint_versions set user_id = new_user_id where constraint_versions.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('constraint_versions', n);

  -- favorites：冲突（新账号已收藏同 SKU）时删旧留新，保证 unique(user_id, sku_id)
  delete from favorites f
  using favorites kept
  where f.user_id = old_user_id
    and kept.user_id = new_user_id
    and kept.sku_id = f.sku_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('favorites_deduped', n);

  update favorites set user_id = new_user_id where favorites.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('favorites', n);

  -- events：整体迁移，合并后同一 session 的事件只归属一个 user_id，不重复计数
  update events set user_id = new_user_id where events.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('events', n);

  update mock_orders set user_id = new_user_id where mock_orders.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('mock_orders', n);

  update recommendations set user_id = new_user_id where recommendations.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('recommendations', n);

  -- 冷启动与实验分组跟随新身份；与新账号已有行冲突时删旧留新，避免唯一约束回滚整个事务
  delete from cold_start_states old_s
  using cold_start_states kept
  where old_s.entity_type = 'user' and old_s.entity_id = old_user_id
    and kept.entity_type = 'user' and kept.entity_id = new_user_id;
  update cold_start_states set entity_id = new_user_id
  where entity_type = 'user' and entity_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('cold_start_states', n);

  delete from experiment_assignments old_a
  using experiment_assignments kept
  where old_a.user_id = old_user_id
    and kept.user_id = new_user_id
    and kept.experiment_id = old_a.experiment_id;
  update experiment_assignments set user_id = new_user_id
  where experiment_assignments.user_id = old_user_id;
  get diagnostics n = row_count;
  counts := counts || jsonb_build_object('experiment_assignments', n);

  insert into account_merges (old_user_id, new_user_id, migrated)
  values (old_user_id, new_user_id, counts);

  return jsonb_build_object('status', 'merged', 'migrated', counts);
end;
$$;

-- 仅 service role / admin 可调用
revoke execute on function public.merge_anonymous_data(uuid, uuid) from public, anon, authenticated;
grant execute on function public.merge_anonymous_data(uuid, uuid) to service_role;
