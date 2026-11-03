-- 0005_sku_media.sql — 商品图片字段 + 京东真实 SKU 关联标识
-- 推荐卡与详情页展示真实商品图片
-- 幂等：全部使用 if not exists，可重复执行

alter table skus add column if not exists primary_image_url text;
alter table skus add column if not exists gallery_urls jsonb not null default '[]';
alter table skus add column if not exists jd_sku_id text;
-- 允许多条 pending 记录共享 jd_sku_id 为空（人工录入的待核验数据），核验后回填并保证唯一
create unique index if not exists skus_jd_sku_id_uniq on skus (jd_sku_id) where jd_sku_id is not null;

-- 爬取/核验状态：与 link_status（业务侧是否可推荐）分开记录，避免混淆
-- pending   = 尚未抓取或抓取失败，等待重试
-- scraped   = 已抓取，字段齐全但未经人工核验
-- verified  = 人工核验通过（配合 skus.verified_at）
-- stale     = 价格快照超过有效期，等待 refresh-stale 重新抓取
alter table skus add column if not exists scrape_status text not null default 'pending'
  check (scrape_status in ('pending', 'scraped', 'verified', 'stale'));

comment on column skus.primary_image_url is '主图，落盘后的可访问 URL（public/ 或 OSS），禁止仅存京东外链';
comment on column skus.gallery_urls is '轮播图 URL 数组，同上落盘要求';
comment on column skus.jd_sku_id is '京东商品 ID（source_url 中的数字段），用于爬虫增量刷新去重';
comment on column skus.scrape_status is '爬取/核验流水线状态，独立于 link_status（业务可推荐性）';
