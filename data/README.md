# SKU 商品池（catalog.json）

真实商品、人民币标价，全部来自无需登录的公开页面/接口。采集时间见 `catalog.json` 的 `generatedAt` / 每条 `observedAt`（2026-11-01/02 UTC）。

## 来源

| 平台 | 入口 | 方法 | 运费政策（shippingPolicy） |
|---|---|---|---|
| 小米商城 | https://www.mi.com/shop | 先用公开搜索接口 `api2.order.mi.com/search/index` 按关键词列出商品，人工挑选 product_id（排除配件、刷头、表带、滤芯、套装），再逐个调 `api2.order.mi.com/product/view?product_id=…`（商品页本身用的公开 JSON）| 商品页「运费说明」：由小米发货商品单笔满69元免运费 → `freeOverCents=6900`，未写满额以下运费金额 → `feeCents=null` |
| Apple 官网 | https://www.apple.com.cn/shop | 配件列表页内嵌的 `categoryResults` JSON（耳机、键鼠、充电器）；机型购买页内嵌的 `products` JSON（iPad/iPhone 每个配置的 fullPrice）；Apple Watch 购买页 `productSelectionData.prices`（表壳配置“起”价，已写入 specs.价格说明）| 帮助页「送货与取货」：所有线上订单可享免费标准送货 → `freeOverCents=0, feeCents=0` |
| 戴森官网 | https://www.dyson.cn | 从首页及公开类目页收集商品链接，解析服务端渲染的商品页（标题、`data-price-amount` 建议零售价、库存标签、短卖点）| 未找到明确运费说明 → `null` |

## 各品类覆盖（共 299 SKU）

| 品类 id | 名称 | SKU 数 | 来源 | 价格区间（元） |
|---|---|---|---|---|
| portable-fan | 便携/手持风扇 | 3 | 小米 2、戴森 1 | 129–799 |
| desk-fan | 桌面/家用风扇（台扇、落地扇、循环扇、塔扇、冷风扇）| 10 | 小米 | 219–599 |
| earphones | 蓝牙/降噪耳机 | 31 | 小米 17、Apple 13、戴森 1 | 39–3999 |
| keyboard-mouse | 键盘/鼠标 | 25 | 小米 12、Apple 13（含 Logitech）| 39–2799 |
| charging | 充电宝/充电器 | 28 | 小米 21、Apple 7 | 29–999 |
| electric-toothbrush | 电动牙刷 | 9 | 小米 | 89–399 |
| hair-dryer | 吹风机 | 20 | 小米 12、戴森 8 | 79–3999 |
| desk-lamp | 台灯 | 10 | 小米 | 79–940 |
| luggage | 行李箱 | 21 | 小米 | 224–1999 |
| smartwatch | 智能手表/手环 | 26 | 小米 19、Apple 7 | 159–6499 |
| floor-cleaning | 扫地机/吸尘器/洗地机 | 32 | 小米 21、戴森 11 | 199–6299 |
| kitchen-appliance | 咖啡机/小厨电（胶囊/便携/半自动咖啡机、电水壶、空气炸锅）| 11 | 小米 | 69–3884 |
| phone | 手机 | 38 | 小米 28、Apple 10 | 699–14299 |
| tablet | 平板 | 35 | 小米 23、Apple 12 | 999–13299 |

基准品类对照：耳机、键盘、鼠标（合并在 keyboard-mouse）、充电宝（charging）、吹风机、台灯、电动牙刷、便携风扇、行李箱、手表/手环、吸尘器（floor-cleaning）、咖啡机（kitchen-appliance，4 款）、手机、平板——全部有数据。**便携风扇只有 3 款**：小米商城只卖 2 款（米家手持风扇、米家桌面移动风扇），戴森 1 款（HushJet Mini Cool，官网导航归在「便携风扇」）；主流便携风扇卖家（京东/淘宝/拼多多店铺、小米有品）价格需登录或页面为纯前端渲染，按规则未采集。

## 字段约定

- `priceCents`：页面当前标价（整数分）。小米为 `price`，Apple 为 `fullPrice`/`priceCurrent`，戴森为页面“建议零售价”。
- `listPriceCents`：**统一为 null**（不展示划线价）。
- `shippingCents`：全部 null（没有商品页写明单品运费）；平台级规则看 `sources[].shippingPolicy`，`shippingNote` 为页面原话摘要。
- `stock`：小米取 `goods_info.is_stock`（未选收货地址时的默认地区状态；同价多颜色时优先取有货颜色）；戴森取页面库存标签（“补货中”→ out_of_stock）；Apple 列表/购买页不显示库存 → null。
- 变体：同一商品只有**价格不同**时才拆成多个 SKU（如手机/平板的存储档、Apple Watch 表壳尺寸），每个商品最多 3 个价位；同价不同颜色合并为一条。
- `specs`：页面原样展示的参数（小米「关键参数」、Apple 配置名中的存储/网络/尺寸）。值为“-”的项已丢弃。
- `highlights`：页面卖点原句（小米 `product_desc` 按「|」切分，戴森短描述段落，去掉脚注角标），已剔除促销文案（直降/立减/券后/到手价等），<3 字的碎片丢弃。
- `facts`：只用正则从标题/卖点/参数原文抽取，不推断，且按品类限定可抽取的键（如 windSpeed 只给风扇/吹风机，「抗风噪 6m/s」不算风速）。键：`batteryMah`、`batteryHours`、`windSpeed`(m/s)、`weightG`、`noiseDb`（运行噪音）、`ancDepthDb`（耳机降噪深度，越大越好）、`wattage`(W)、`capacityL`、`wireless`、`noiseCancelling`、`ramGB`、`storageGB`、`sizeInch`。每个值的原文证据在 `data/raw/facts-evidence.json`（`{productId: {factKey: 原文}}`）。
- `images`：小米（cdn.cnbj*.fds.api.mi-img.com）和 Apple（store.storeimages.cdn-apple.com）无 Referer 可直接访问，保留远程 URL；戴森图床无 Referer 返回 403，已下载到 `public/img/dyson-<sku>.<ext>`（21 张，约 3.6MB），用 `/img/...` 引用。 build 时对每个商品首图做 HEAD 校验（≤1 次/秒/host，结果缓存在 `data/raw/image-check.json`），失效的图会从 images 移除；采集时 201 个远程首图全部返回 200。

## 复现

```
cd app/scripts/sku
python3 mi.py && python3 apple.py && python3 dyson.py   # 各自写 data/parts/<source>.json，原始响应缓存在 data/raw/<host>/
python3 build.py                                        # 合并 → data/catalog.json + data/raw/facts-evidence.json
node validate.mjs                                       # 校验
```

抓取规范：诚实 UA（`zhixuan-sku-research/0.1 …; no login`）、每个 host ≤1 次/秒、不登录、遇到验证/风控直接跳过。`data/raw/`（约 54MB 原始缓存）已 gitignore；删掉缓存再跑即可刷新价格。

## 已知缺口 / 被跳过的来源

- **华为商城 vmall.com**：页面纯前端渲染，商品数据走 `openapi.vmall.com` 私有接口，没有公开的商品 ID 列表入口；未逆向，跳过。
- **京东**：`item.m.jd.com` 能拿到名称/图片，但价格必须登录，不能用来定价；没有用它补充。
- **荣耀 / OPPO / vivo / realme / 一加 / 小米有品**：首页或搜索页为前端渲染，HTML 里没有价格；跳过。DJI 中国站重定向到新加坡站（非人民币），跳过。
- facts 覆盖不均：小米很多参数只在详情图里（图片无法可靠提取），所以风扇/台灯等的风速、噪音、重量常为空。缺失就是缺失，没有填估计值。
- 小米 `stock` 是默认地区状态，选具体收货地址后可能不同。
- Apple Watch 价格是表壳配置的“起”价（已在 specs.价格说明 标注）；Apple 机型 SKU 的 url 指向机型购买页而不是具体配置页。

## 体验笔记（posts.json）

各品类在小红书上的真实使用体验，用 `scripts/posts/xhs_posts.py` 采集（OpenCLI 复用本地浏览器登录）：每个品类搜索“品类 + 使用体验”，按点赞取前 5 篇，读取正文后只保存标题、作者、点赞与收藏数、前 3 个话题、50 字以内摘录和原文链接；去掉与品类无关和没有正文的笔记。界面上的“大家怎么说”和体验页签只展示这些字段，并链接回原笔记。手机品类暂无可用笔记。

## 语义向量（embeddings.json）

每个 SKU 一条 256 维向量，输入为标题、品牌、品类名、参数、卖点和前 6 项规格，用于导购链路的语义召回。由 `node --env-file=.env scripts/agent/embed_catalog.mjs` 生成；商品池更新后需要重新生成，否则新商品没有语义分（不影响其余排序）。
