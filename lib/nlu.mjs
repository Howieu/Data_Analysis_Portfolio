import { moneyToken, cnInt, price } from './money.mjs';
import { hasKeyword, BRAND_ALIASES, brandHit } from './engine.mjs';

const uid = () => globalThis.crypto.randomUUID().slice(0, 8);
const NUM = '(?:\\d+\\s*[千万]\\s*\\d(?!\\d)|\\d[\\d,]*(?:\\.\\d+)?\\s*(?:k|K|w|W|千|万)?|[零〇一二两三四五六七八九十百千万]+)';
const MEASURE = '个|件|把|副|台|支|盏|套|只|部|款|对|条|瓶|盒|张|双';
const SPEC_UNIT = /^\s*(mAh|毫安|g|克|m\/s|W|瓦|L|升|寸|英寸|dB|分贝|个?小时|h|天|年|岁|档|核|G|GB|T|TB|Hz|个人|人|位|个月)/i;

// Merged catalog categories: the word the user said decides the sub-type (键盘 ≠ 鼠标).
const SUBTYPES = { 'keyboard-mouse': ['键盘', '鼠标'], charging: ['充电宝', '充电器', '充电头'], 'floor-cleaning': ['扫地', '吸尘器', '洗地机'], smartwatch: ['手环', '手表'], 'kitchen-appliance': ['咖啡机', '电水壶', '烧水壶', '空气炸锅', '养生壶'] };
// Form-factor words: narrow when product pages state them; otherwise disclosed, never silently assumed.
const TYPE_WORDS = ['入耳', '半入耳', '头戴', '开放式', '挂脖', '颈挂', '手持', '落地', '机械', '静音', '折叠', '磁吸', '胶囊', '意式', '迷你', '高速', '负离子', '声波', '冲牙', '20寸', '24寸', '26寸', '28寸'];
const UNFILTERABLE = ['红色', '粉色', '白色', '黑色', '蓝色', '绿色', '紫色', '灰色', '银色', '金色', '防水', '颜值', '好看', '耐用'];
// Things people ask for that this store does not sell. Named so the reply can say so instead of guessing.
const OFF_CATALOG = ['笔记本', '辆车', '汽车', '买车', '电动车', '自行车', '房子', '一套房', '买房', '机票', '火车票', '高铁票', '酒店', '外卖', '笔记本电脑', '电脑', '冰箱', '洗衣机', '电视', '空调', '衣服', '鞋子', '口红', '相机', '手冲壶', '显卡', '游戏机', '保险', '股票', '基金'];
const INJECTION = /(忽略|无视|不要管|别管|绕过)(之前|上面|所有|系统|你的)?的?(规则|指令|设定|限制|要求)|ignore (all|previous|the)|system prompt|开发者模式|管理员(指令|模式|权限)|系统(提示词|指令|消息)|提示词|越狱|jailbreak|你现在是/i;
const PREF_WORDS = [
  [/安静|静音|不吵|噪音小|声音小/, 'quiet'], [/轻便|轻一点|轻巧|轻薄|便携|随身|带着?出门|出门用|通勤|旅行用|小巧/, 'light'],
  [/续航|待机|电池大|用得久|撑得久/, 'battery'], [/(最好|尽量|希望|可以的话)[^，,]{0,3}降噪/, 'anc'], [/风大|风力大|风力强|风速大|大风力/, 'wind'],
  [/便宜|实惠|性价比|省钱|划算|低价/, 'cheap'], [/最好的|好一点|旗舰|高端|顶配|贵一点|最贵/, 'premium'],
];
const REQUIRE = /(必须|一定要|一定得|要求|得是|得能|还得|得支持|得带|要带|带有|要能|需要支持|必须支持|要支持|支持|能带上?飞机|可以带上?飞机)/;

const clauses = text => text.split(/[，,。；;！!？?\n]/).map(s => s.trim()).filter(Boolean);

export function normalize(text) {
  return String(text ?? '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/[０-９]/g, d => String.fromCharCode(d.charCodeAt(0) - 0xFEE0))
    .replace(/[￥]/g, '¥').replace(/[＊×xX](?=\s*\d)/g, '×').replace(/　/g, ' ')
    .replace(/\s+/g, ' ').trim().slice(0, 4000)
    .replace(/(\d+)[kK](\d)(?!\d)/g, (_, a, b) => String(Number(a) * 1000 + Number(b) * 100))   // 2k5 → 2500
    .replace(/(\d+)[wW](\d)(?!\d)/g, (_, a, b) => String(Number(a) * 10000 + Number(b) * 1000))
    // "一百五一个" → unit price, so the trailing 一 is not glued onto the number
    .replace(/([零〇一二两三四五六七八九十百千万]*[百千万][零〇一二两三四五六七八九十]*)(?=一(?:个|件|台|把|支|副|盏|只))/g, '$1元')
    // Chinese numerals → digits, except fuzzy ranges like 两三百
    .replace(/[零〇一二两三四五六七八九十百千万]{2,}/g, run => /^[一二两三四五六七八九][一二两三四五六七八九][十百千万]$/.test(run) ? run : (/[十百千万]/.test(run) && cnInt(run) !== null ? String(cnInt(run)) : run))
    .replace(/(\d+)(?:块|元)([一二三四五六七八九\d])(?:毛|角)/g, (_, y, j) => `${y}.${/\d/.test(j) ? j : cnInt(j)}元`)
    .replace(/(加|放|合|算)(?:在)?一块儿?/g, '$1起来')
    .replace(/俩/g, '两个').replace(/仨/g, '三个')
    // every spec comparison becomes "N单位以内" / "N单位以上"
    .replace(/(?:不要低于|别低于|不能低于|不低于|不要小于|别小于|不能小于|不少于|不要少于)\s*(\d+(?:\.\d+)?)\s*(毫安|mAh|克|g|kg|公斤|分贝|dB|m\/s|小时|W|瓦|L|升)/gi, '$1$2以上')
    .replace(/(?:别超过|不要超过|不能超过|不超过|不过|不高于|(?<![不别要能])低于|(?<![不别要能])小于|轻于|不到|最多|控制在)\s*(\d+(?:\.\d+)?)\s*(毫安|mAh|克|g|kg|公斤|分贝|dB|m\/s|小时|W|瓦|L|升)/gi, '$1$2以内')
    .replace(/(?:大于|不低于|高于|超过|至少|起码|最少|不少于)\s*(\d+(?:\.\d+)?)\s*(毫安|mAh|克|g|kg|公斤|分贝|dB|m\/s|小时|W|瓦|L|升)/gi, '$1$2以上')
    .replace(/(\d+(?:\.\d+)?)\s*(毫安|mAh|分贝|dB|m\/s|小时)\s*(?:往上|起步|起|及以上)/gi, '$1$2以上');
}

export function findMentions(text, catalog) {
  let raw = findMentionsRaw(text, catalog);
  if (!raw.length) raw = fuzzyMentions(text, catalog);
  return raw.filter((m, i) => !/(带|支持|有|可以|能)\s*$/.test(text.slice(Math.max(i ? raw[i - 1].end : 0, m.index - 3), m.index)));
}
function fuzzyMentions(text, catalog) {
  for (const c of catalog.categories) for (const alias of c.aliases) {
    if (alias.length < 3 || /[a-z]/i.test(alias)) continue;
    for (let i = 0; i + alias.length <= text.length; i++) {
      const w = text.slice(i, i + alias.length);
      let diff = 0; for (let k = 0; k < w.length; k++) if (w[k] !== alias[k]) diff++;
      if (diff === 1 && /[\u4e00-\u9fa5]{3}/.test(w)) return [{ c, alias, index: i, end: i + alias.length, typed: w, subtype: null }];
    }
  }
  return [];
}
function findMentionsRaw(text, catalog) {
  const hits = [];
  for (const c of catalog.categories) for (const alias of c.aliases) {
    if (!alias || (alias.length < 2 && !/[a-z]/i.test(alias))) continue;
    let i = 0, at;
    while ((at = text.indexOf(alias, i)) >= 0) { hits.push({ c, alias, index: at, end: at + alias.length }); i = at + alias.length; }
  }
  hits.sort((a, b) => a.index - b.index || b.alias.length - a.alias.length);
  const out = [];
  for (const h of hits) if (!out.some(m => h.index < m.end && h.end > m.index)) out.push(h);
  return out.map(m => ({ ...m, subtype: (SUBTYPES[m.c.id] || []).find(w => m.alias.includes(w)) || null }));
}

const NOT_A_PRODUCT = /(清单|购物车|列表|里面|里|共|总|我想|我要|想要|想买|要买|帮我|给我|推荐|准备|打算|需要|买|要|个|件|台|把|副|盏|只|一|两|三|四|五|的|预算|总共|一共|以内|以下|左右|上下|元|块钱|块|再|还|也|吧|呢|啊|哦|都|各|每人|一人|请|看看|挑|选|配|套|送|给|女朋友|男朋友|老婆|老公|爸妈|父母|同事|朋友|孩子|家里|自己|办公|通勤|出门|宿舍|学生|上班|旅行|好用|好一点|就行|\d+|[，,。！？!?\s¥~～])/g;
// "投影仪和台灯": list members that match no category must be named back to the user, never silently dropped
function unknownListMembers(text, catalog) {
  const segs = text.split(/和|、|加上|跟|以及|还有|与|及|\+/);
  if (segs.length < 2) return [];
  const known = new RegExp([...TYPE_WORDS, ...UNFILTERABLE, '无线', '降噪', '续航', '毫安', '安静', '静音', '便宜', '实惠', '轻', '便携', '支持', '必须', '品牌', '比一比', '对比', '比较', ...BRAND_ALIASES.flat(), ...catalog.products.map(p => p.brand).filter(Boolean)].map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'i');
  const out = [];
  for (const seg of segs) {
    if (findMentionsRaw(seg, catalog).length || known.test(seg)) continue;
    const rest = seg.replace(NOT_A_PRODUCT, '');
    if (/^[\u4e00-\u9fa5A-Za-z]{2,8}$/.test(rest)) out.push(rest);
  }
  return out;
}
const qtyOf = t => t === '一对' ? 1 : t === '一打' ? 12 : /^\d+$/.test(t) ? Number(t) : cnInt(t);

// ---------- budget ----------
// Budget reading, default inverted: an amount is a per-item price ONLY when it directly modifies an item
// ("每个/一副/单价 + 金额", "金额 + 的 + 品类"). Every other amount is the total budget — no list of "total words".
const PER_ITEM = new RegExp(`(?:每个|每件|每人|每份|每副|每台|每把|每支|每只|每盏|单价|单个|一个|一件|一台|一把|一支|一副|一盏|一只)\\s*(?:预算)?\\s*([不别超过最多顶控制在以内下封大概差约估计要就是为价格的大致]{0,4})\\s*¥?\\s*(\\d+(?:\\.\\d+)?)\\s*(多|来)?\\s*(?:块钱|块|元|钱)?\\s*(多|来|出头|左右|上下|以内|以下)?`);
export function parseBudget(text) {
  if (/[小大]?几[十百千万]|[小大]几/.test(text) && !/预算\s*[\d一二两三四五六七八九十百千万]/.test(text)) return null; // "小几百" is a feeling, not a number
  if (/(没|没有|不)(想好|定|确定|清楚).{0,3}预算|预算.{0,2}(还)?(没|没有|不)(想好|定|确定|清楚)|预算(待定|随后|之后再说)/.test(text) && !/\d|[一二两三四五六七八九十百千万]\s*[元块]/.test(text)) return null; // no budget yet: ask for one, keep the items
  if (/(美元|美金|刀|\$|usd|日元|欧元|港币|英镑|港元|韩元)/i.test(text) && /\d|[一二两三四五六七八九十百千万]/.test(text)) return { error: 'currency' };
  if (/预算.{0,4}(无限|不限|随便|没上限)|不差钱|不限预算|无上限/.test(text)) return { unlimited: true };
  const blank = (t, m) => t.slice(0, m.index) + ' '.repeat(m[0].length) + t.slice(m.index + m[0].length);
  let work = text, delta = null;
  // relative change in either direction ("预算往上调300", "少一百") — never read as an absolute amount
  const rel = work.match(new RegExp(`(?:预算|总价|总预算)?\\s*(?:再|多)?(往上调|往上加|往上提|上调|调高|提高|增加|多加|多给|多出|加|涨|往下调|往下降|下调|调低|降低|减少|缩水|缩减|削减|压缩|少给|砍掉|砍|减|少|降)(?:了|个|掉)?\\s*(${NUM})(?!\\s*(?:到|${MEASURE}))\\s*(?:元|块)?`));
  if (rel && /预算|总价|再|多加|多给|调|往上|往下/.test(text)) { const v = moneyToken(rel[2]); if (v.cents) { delta = /下|低|减|少|降|砍|缩|削|压/.test(rel[1]) ? -v.cents : v.cents; work = blank(work, rel); } }
  // every per-item amount, each kept with its position so it binds to the product named just before it
  const hints = [];
  for (const per of work.matchAll(new RegExp(PER_ITEM.source, 'g'))) {
    const c = moneyToken(per[2]).cents; if (!c) continue;
    const cap = /不|别|最多|顶多|控制|以内|以下|封顶/.test(per[1] || '') || /以内|以下/.test(per[4] || '');
    const over = !!(per[3] || /多|出头/.test(per[4] || ''));
    hints.push({ cents: c, cap, over, rough: !cap && (over || /来/.test(per[4] || '') || /大概|差不多|约|估计|大致/.test(per[1] || '')), at: per.index });
  }
  for (const h of hints) { const m = work.slice(h.at).match(PER_ITEM); if (m) work = blank(work, { index: h.at, 0: m[0] }); }
  if (delta) return { delta, unitHints: hints };
  const total = budgetCore(work);
  if (total?.error && total.error !== 'unreadable') return total;
  if (total?.cents && total.scope !== 'maybe-unit') return { ...total, scope: 'total', unitHints: hints };
  if (!hints.length) return total;
  if (hints.length === 1 && !hints[0].rough) return { cents: hints[0].cents, scope: 'unit' }; // one per-item price, no total: total = price × quantity
  return { cents: null, unitHints: hints }; // several per-item prices (or a rough one) and no total: bind each, then sum or ask
}

function budgetCore(text) {
  if (/(?:预算|不超过|最多|¥)\s*(?:是|为)?\s*[-−]\s*\d/.test(text)) return { error: 'negative' };
  const fuzzy = text.match(/([一二两三四五六七八九])([一二两三四五六七八九])(十|百|千|万)(?:块|元)?/);
  const range = text.match(new RegExp(`(${NUM})\\s*(?:-|~|到|至)\\s*(${NUM})\\s*(?:元|块|¥)`)) || text.match(new RegExp(`(?:预算|价位|价格)\\s*(${NUM})\\s*(?:-|~|到|至)\\s*(${NUM})`));
  if (fuzzy) {
    const lo = cnInt(fuzzy[1] + fuzzy[3]), hi = cnInt(fuzzy[2] + fuzzy[3]);
    if (lo && hi && hi > lo) return { cents: hi * 100, scope: 'total', upper: true, rangeText: fuzzy[0].replace(/[块元]$/, '') };
  }
  if (range) {
    const a = moneyToken(range[1]), b = moneyToken(range[2]);
    if (a.cents && b.cents && b.cents > a.cents) return { cents: b.cents, scope: 'total', upper: true, rangeText: `${range[1]}到${range[2]}` };
  }
  const unit = text.match(new RegExp(`(${NUM})\\s*(?:块钱|块|元)\\s*(?:一(?:个|件|台|把|支|副|盏|只)|每(?:个|件|台|把|支|副))`));
  if (unit) { const v = moneyToken(unit[1]); if (v.cents) return { cents: v.cents, scope: 'unit' }; }
  const found = [];
  const re = new RegExp(
    `(?:总预算|预算|总共|一共|合计|总价|不超过|别超过|不要超过|不能超过|最多|控制在|上限|顶多|¥)\\s*(?:是|为|在|大概|差不多|改成|改为|改到|提到|提高到|降到|降低到|调到|调成|变成|设为|设成|砍到|压到|加到|涨到|放宽到|控制到|:|：)?\\s*¥?\\s*(${NUM})\\s*(?:元|块|rmb|RMB)?` +
    `|(${NUM})\\s*(?:元|块钱|块|rmb|RMB)` +
    `|(${NUM})\\s*(?=以内|以下|之内|左右|上下)` +
    (/数量|件数|个数|几个|几件/.test(text) ? '' : `|(?:不对|应该是|改成|改为|还是|算了)\\s*[，,]?\\s*(${NUM})(?!\\s*(?:${MEASURE}))`), 'g');
  for (const m of text.matchAll(re)) {
    const tok = (m[1] || m[2] || m[3] || m[4]).trim();
    const tail = text.slice(m.index + m[0].length, m.index + m[0].length + 3);
    if (SPEC_UNIT.test(tail) || /^(以上|起)/.test(tail)) continue;
    found.push({ tok, at: m.index, correction: !!m[4], approx: /左右|上下|大概|差不多/.test(text.slice(Math.max(0, m.index - 2), m.index + m[0].length + 3)) });
  }
  if (/\de\d/i.test(text) && /预算|元|块/.test(text)) return { error: 'unreadable' };
  if (!found.length) {
    // any other plain amount is the total ("吹风机 800", "统共不过200"); quantities, specs and model numbers carry units or letters
    for (const m of text.matchAll(new RegExp(`(?<![A-Za-z\\d.])(\\d{2,7}(?:\\.\\d{1,2})?)(?!\\s*(?:${MEASURE}|${SPEC_UNIT.source.slice(5, -1)}|\\d|以上|起))`, 'gi'))) found.push({ tok: m[1], at: m.index, correction: false, approx: false });
  }
  // a price written right before 的 + product is that item's price, not the total
  const itemPrice = f => new RegExp(`^${f.tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*(块钱|块|元|钱)?\\s*(多|来块|来|出头|左右|上下|以内|以下|不到)?\\s*的`).test(text.slice(f.at ?? text.indexOf(f.tok)).replace(/^[^\d零〇一二两三四五六七八九十百千万]*/, ''));
  const free = found.filter(f => !itemPrice(f));
  if (!free.length) {
    const each = text.match(new RegExp(`(${NUM})\\s*(?:(?:块钱|块|元)\\s*(?:左右|上下)?|左右|上下)\\s*的`));
    if (each) { const v = moneyToken(each[1]); if (v.cents) return { cents: v.cents, scope: 'maybe-unit' }; }
    return /预算/.test(text) && !/预算(改一下|多少|呢|吗)/.test(text) ? { error: 'unreadable' } : null;
  }
  let pick = free[0];
  if (new Set(free.map(f => f.tok)).size > 1) {
    if (free.some(f => f.correction)) pick = free[free.length - 1];
    else {
      const cents = [...new Set(free.map(f => moneyToken(f.tok).cents).filter(Boolean))];
      if (cents.length > 1) return { error: 'multiple', values: cents };
    }
  }
  const v = moneyToken(pick.tok);
  if (v.error) return { error: v.error, token: pick.tok };
  return { cents: v.cents, scope: 'total', approx: pick.approx };
}

function priceFloor(text) {
  const m = text.match(new RegExp(`(?:至少|最少|起码|不低于)\\s*(?:要)?\\s*(${NUM})\\s*(?:元|块)?|(?<![A-Za-z\\d])(${NUM})\\s*(?:元|块)?\\s*(?:以上|起)的?`));
  if (!m) return null;
  const tok = (m[1] || m[2]).trim(), end = text.indexOf(tok, m.index) + tok.length;
  if (SPEC_UNIT.test(text.slice(end, end + 3))) return null;
  const c = moneyToken(tok).cents;
  return c && c >= 1000 ? c : null;
}

// ---------- per-mention conditions ----------
function hardFrom(clause, cat, catalog) {
  const hard = [];
  const has = key => catalog.products.some(p => p.category === cat.id && p.facts && key in p.facts);
  const m1 = clause.match(/(\d{3,5})\s*(?:mAh|毫安)/i);
  if (m1 && has('batteryMah')) hard.push({ key: 'batteryMah', op: 'gte', value: Number(m1[1]), label: `电池 ≥${m1[1]}mAh` });
  const m2 = clause.match(/(\d{1,5}(?:\.\d)?)\s*(?:g|克)\s*(?:以内|以下|之内|内)/i);
  if (m2 && has('weightG')) hard.push({ key: 'weightG', op: 'lte', value: Number(m2[1]), label: `重量 ≤${m2[1]}g` });
  const m3 = clause.match(/(\d{1,2}(?:\.\d)?)\s*m\/s/i);
  if (m3 && has('windSpeed')) hard.push({ key: 'windSpeed', op: 'gte', value: Number(m3[1]), label: `风速 ≥${m3[1]}m/s` });
  const m4 = clause.match(/(\d{2,3})\s*(?:dB|分贝)\s*(?:以内|以下|之内)/i);
  if (m4 && has('noiseDb')) hard.push({ key: 'noiseDb', op: 'lte', value: Number(m4[1]), label: `噪音 ≤${m4[1]}dB` });
  if (/降噪/.test(clause) && !/不(要|用|需要)降噪|(最好|尽量|希望|可以的话|有的话)[^，,]{0,3}降噪/.test(clause) && has('noiseCancelling')) hard.push({ key: 'noiseCancelling', op: 'eq', value: true, label: '主动降噪' });
  if (/无线(?!充)/.test(clause) && !/有线/.test(clause) && has('wireless')) hard.push({ key: 'wireless', op: 'eq', value: true, label: '无线' });
  const m5 = clause.match(/续航.{0,6}?(\d{1,3})\s*(?:个)?(?:小时|h)/i) || clause.match(/(\d{1,3})\s*(?:个)?(?:小时|h)\s*(?:以上)?.{0,3}续航/i);
  if (m5 && has('batteryHours')) hard.push({ key: 'batteryHours', op: 'gte', value: Number(m5[1]), label: `续航 ≥${m5[1]}小时` });
  return hard;
}

// "必须支持HomeKit" etc. — requirements no product page states, so nothing can be confirmed to meet them.
function unsupportedRequirement(clause, hard) {
  // check every requirement phrase on its own: "要带降噪 还得支持无线充" has one we can check and one we can't
  const parts = clause.split(new RegExp(`(?=${REQUIRE.source})`)).filter(x => REQUIRE.test(x));
  for (const tail of parts) {
    const m = tail.match(REQUIRE);
    const covered = hard.length && /无线(?!充)|降噪|mAh|毫安|\d\s*(g|克)|m\/s|dB|分贝|品牌|续航/.test(tail.split(/[，,\s]/)[0] + tail.slice(0, 12));
    if (covered && !/无线充/.test(tail.slice(0, 10))) continue;
    if (m[0].includes('飞机')) return m[0];
    const rest = tail.replace(REQUIRE, '').split(/[，,、\s]/)[0].replace(/^(是|有|支持|能|可以|带)/, '').replace(/的$/, '').replace(/功能$/, '').trim();
    if (rest && !/^\d/.test(rest)) return rest.slice(0, 16);
  }
  return null;
}

// A brand name alone is neutral. It is "wanted" only when asked for (要/只要/选…) or written right before the product;
// any negation in its clause excludes it; anything else is asked back instead of guessed.
const NEG = /不|别|除|非|免|没兴趣|没感觉|排除|以外|之外|算了|pass/i;
function brandsIn(text, catalog) {
  const names = [...new Set([...catalog.products.map(p => p.brand).filter(Boolean), ...BRAND_ALIASES.flat()])].sort((a, b) => b.length - a.length);
  const want = [], avoid = [], unsure = [], low = text.toLowerCase();
  const aliases = catalog.categories.flatMap(c => c.aliases);
  for (const b of names) {
    const at = low.indexOf(b.toLowerCase()); if (at < 0) continue;
    if ([...want, ...avoid, ...unsure].some(x => x.toLowerCase().includes(b.toLowerCase()))) continue;
    const cs = text.slice(0, at).search(/[，,。；;！!？?][^，,。；;！!？?]*$/), clauseStart = cs < 0 ? 0 : cs + 1;
    const ce = text.slice(at).search(/[，,。；;！!？?]/), clause = text.slice(clauseStart, ce < 0 ? text.length : at + ce).replace(/不超过|不低于|不到|不少于|不高于|别超过|不要超过/g, '');
    const after = text.slice(at + b.length, at + b.length + 12);
    if (NEG.test(clause)) avoid.push(b);
    else if (/(要|只要|就要|选|买|想要|来个|挑)\s*$/.test(text.slice(Math.max(0, at - 3), at)) || aliases.some(a => new RegExp(`^(的|家的|牌的|家|牌)?\\s*${a.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')}`).test(after))) want.push(b);
    else unsure.push(b);
  }
  return { want, avoid, unsure };
}

function peopleCount(text) {
  if (!/每人|每个人|一人一|人手一|各(买|来|送|要)?一/.test(text)) return null;
  if (/爸妈|父母|爸爸妈妈|两口子|夫妻/.test(text)) return 2;
  const m = text.match(new RegExp(`(${NUM})\\s*(?:个|位)?\\s*(?:人|同事|朋友|同学|家人|孩子|室友|小朋友)`));
  if (!m) return null;
  const n = qtyOf(m[1]);
  return Number.isSafeInteger(n) && n > 0 ? n : null;
}

function mentionItems(text, mentions, catalog) {
  const brand = brandsIn(text, catalog), people = peopleCount(text);
  const out = [], warnings = [], unsupported = [];
  mentions.forEach((m, n) => {
    const prefix = text.slice(n ? mentions[n - 1].end : 0, m.index), suffix = text.slice(m.end, mentions[n + 1]?.index ?? text.length);
    const FORM = `(?:${TYPE_WORDS.join('|')}|式|款|的|新|好|便宜|小|大|\\s)*`;
    const moneyish = idx => /(预算|¥|元|块|总共|一共|合计|以内|不超过|最多)\s*$/.test(prefix.slice(Math.max(0, idx - 5), idx));
    const PRICE_MOD = '(?:\\d+(?:\\.\\d+)?\\s*(?:(?:块钱|块|元)\\s*(?:左右|上下|以内)?|左右|上下|以内)\\s*(?:的)?)?';
    // Structure first: "数量 量词 … 的 品类" binds the quantity to this product. Whatever sits in between can only set a
    // per-item price (or mark it rough and ask for the total) — it can never rewrite the quantity.
    const bound = prefix.match(new RegExp(`(?:(${NUM}|一打)\\s*(?:${MEASURE})|(一对))([^，,。；;和、]{0,16}?)的${FORM}$`)) || prefix.match(new RegExp(`()()((?:不到|大概|差不多|大约|约莫|约|不超过|最多)?\\s*\\d+(?:\\.\\d+)?[^，,。；;和、的\\d]{0,8})的${FORM}$`));
    let unitCap = null, approxPrice = false;
    const bq = bound && (bound[1] || bound[2]), mid = bound?.[3] || '';
    const midPrice = mid.match(/(\d+(?:\.\d+)?)/);
    const isFragment = !!(bound && midPrice && !SPEC_UNIT.test(mid.slice(mid.indexOf(midPrice[1]) + midPrice[1].length)));
    if (isFragment && !moneyish(bound.index)) {
      if (/多|来|出头|大概|差不多|大约|约|估计|上下浮动/.test(mid)) approxPrice = true; // rough: ask for the total instead
      else unitCap = moneyToken(midPrice[1]).cents || null; // 左右/上下/以内/以下/不到/不超过/最多/plain → per-item ceiling
    }
    let before = bq ? [bound[0], bq] : prefix.match(new RegExp(`(${NUM}|一打)\\s*(?:${MEASURE})${PRICE_MOD}${FORM}$`));
    if (!before) before = prefix.match(/(一打)\s*$/);
    if (!before) { const b2 = prefix.match(/(?<![\d.])(\d{1,2})$/); if (b2 && !moneyish(b2.index)) before = b2; }
    if (before && before.index !== undefined && moneyish(before.index)) before = null;
    const after = suffix.match(/^\s*×\s*(\d+(?:\.\d+)?)/) || suffix.match(new RegExp(`^\\s*(${NUM}|一打)\\s*(?:${MEASURE})`)) || suffix.match(new RegExp(`(?:买|要|来|共|一共)\\s*(${NUM}|一打)\\s*(?:${MEASURE})`));
    let qty = 1;
    // a quantity bound through a price fragment is already known to be a quantity; others must not look like money
    let tok = (before && (bq || !/预算|元|块钱|¥/.test(before[0])) ? before[1] : null) || after?.[1];
    if (tok && /^一$/.test(String(tok).trim()) && after?.[1] && bound && /(元|块|块钱)\s*$/.test(prefix.slice(0, bound.index))) tok = after[1]; // "150元一个…买三个": 一个 = each; the stated count wins
    if (tok) {
      const v = qtyOf(tok.trim());
      if (!Number.isSafeInteger(v) || v <= 0) return out.push({ error: `「${m.alias}」的数量「${tok}」得是正整数` });
      if (v > 50) return out.push({ error: `${m.alias}要 ${v} 件？数量有点多，确认一下` });
      qty = v;
    }
    if (people && qty === 1) qty = people;
    // a mention owns its own list segment (split at 和/、/，…) plus following segments that name no category
    const segs = [...text.matchAll(/[^和、，,。；;！!？?\n]+/g)].map(x => ({ s: x[0], a: x.index, b: x.index + x[0].length }));
    const at = segs.findIndex(g => g.a <= m.index && m.index < g.b);
    let clause = at < 0 ? text : segs[at].s;
    for (let j = at + 1; at >= 0 && j < segs.length && !findMentions(segs[j].s, catalog).length; j++) clause += '，' + segs[j].s;
    const hard = hardFrom(clause, m.c, catalog);
    const keywords = [], soft = [];
    if (m.subtype && catalog.products.some(p => p.category === m.c.id && hasKeyword(p, m.subtype))) keywords.push(m.subtype);
    for (const w of TYPE_WORDS.filter(w => clause.includes(w))) {
      const stated = catalog.products.some(p => p.category === m.c.id && hasKeyword(p, w));
      (stated ? keywords : soft).push(w);
    }
    if (soft.length) warnings.push(`商品页没有标注「${soft.join('、')}」，这里的${m.alias}不能确认是${soft.join('、')}款`);
    const want = brand.want.filter(b => clause.toLowerCase().includes(b.toLowerCase()));
    if (want.length && !catalog.products.some(p => p.category === m.c.id && brandHit(p, want))) warnings.push(`${want.join('、')}的${m.alias}暂时没有`);
    if (want.length) hard.push({ key: 'brand', op: 'in', value: want, label: `${want.join('/')}品牌` });
    const FEATURES = [[/无线(?!充)/, 'wireless', '无线'], [/降噪/, 'noiseCancelling', '降噪']];
    const lacking = FEATURES.find(([re, key]) => re.test(clause) && !/有线/.test(clause) && !catalog.products.some(p => p.category === m.c.id && p.facts && key in p.facts));
    const UNIT_KEY = [[/毫安|mAh/i, 'batteryMah'], [/克|g\b|kg|公斤/i, 'weightG'], [/分贝|dB/i, 'noiseDb'], [/m\/s/i, 'windSpeed'], [/小时/, 'batteryHours'], [/W\b|瓦/, 'wattage'], [/L\b|升/, 'capacityL']];
    const specSpan = [...clause.matchAll(/\d+(?:\.\d+)?\s*(?:毫安|mAh|克|g\b|kg|公斤|分贝|dB|m\/s|小时|W\b|瓦|L\b|升)(?:以内|以上)?/gi)].map(x => x[0]).find(sp => { const k = UNIT_KEY.find(([re]) => re.test(sp))?.[1]; return !hard.some(h => h.key === k); });
    const req = lacking ? `${lacking[2]}的${m.alias}` : specSpan ? `${specSpan}这个要求` : unsupportedRequirement(clause, hard);
    if (req) unsupported.push({ alias: m.alias, label: req });
    out.push({ id: uid(), category: m.c.id, alias: m.alias, qty, ...(unitCap ? { unitCapCents: unitCap, boundPrice: true } : {}), ...(approxPrice ? { approxPrice: true } : {}), hard, keywords, excludeBrands: brand.avoid.filter(b => clause.toLowerCase().includes(b.toLowerCase())), lock: null, exclude: [], selected: null });
  });
  return { items: out, warnings, unsupported };
}

// The same category named twice: a later descriptive mention refines the first; a bare repeat is ambiguous.
function dedupe(items, text) {
  const out = [];
  for (const it of items) {
    const prev = out.find(o => o.category === it.category && JSON.stringify(o.keywords) === JSON.stringify(it.keywords));
    if (!prev) { out.push(it); continue; }
    if (new RegExp(`${it.alias}\\s*(要|得|需要|必须|最好)`).test(text)) { prev.hard = [...prev.hard, ...it.hard.filter(h => !prev.hard.some(x => x.key === h.key))]; continue; }
    if (it.qty === 1 && prev.qty === 1) return { ambiguous: it.alias };
    out.push(it);
  }
  return { items: out };
}

const CREATE = /我想|想买|想要|帮我|给我|推荐|准备|打算|需要|要买|买个|买一|买两|挑|选|求|配一套|配个|送/;
const ADD = /^(再|还|另外|顺便|加|添|补)|再加|再来|还要|还想|另外|顺便|加一|加个|加上|加把|加副|添一|补一|补个/;
const REMOVE = /(去掉|不要了?|删掉|删了|删除|移除|拿掉|拿走|撤了|撤掉|撤下|划掉|不考虑了|不用了|不买了?|取消)/;
const ONLY = /(只要|只买|就买|只留|只需要)/;
const VIEW = /(只看|单独看|分开看|分别看|单品看|按单品|看看单品|整套看|看整套|先看看?|看下|单看|瞅瞅|瞧瞧)/;
const SWAP = /(换一个|换一款|换个|换款|换一下|不喜欢|还有别的|还有其他|别的推荐)/;
const SWITCH_WORDS = '换成|换个|换一个|改成|改为|改买|改要|替换为|替换成|换为|改选|来个|加个';
const REPLACE = new RegExp(`把(.+?)(?:${SWITCH_WORDS})(.+)`);
const LOCK = /(锁定|锁住|就要这个|就要这款|就它了|定了|就这个|我要了|先别动|别动|不动了|就它|留着|留下|保留)/;
const UNLOCK = /(解锁|取消锁定|不锁了?)/;
const QTY_CHANGE = new RegExp(`(?:改成|改为|换成|要|来|变成|买|拿)\\s*(${NUM}|一打)\\s*(?:${MEASURE})|(?:数量|件数|个数)\\s*(?:改成|改为|改到|变成|换成|调成|调到|要|是)?\\s*(${NUM})`);
const QTY_MORE = new RegExp(`(?:再|补)(?:来|要|加|买)?\\s*(${NUM})?\\s*(?:${MEASURE})`);

const blank = () => ({ items: [], budgetCents: null, budgetScope: 'total', prefs: [] });
const clone = s => JSON.parse(JSON.stringify(s || blank()));

export function understand(rawText, prev, catalog) {
  const text = normalize(rawText);
  const state = clone(prev); state.items ||= []; state.prefs ||= [];
  const notes = [], warnings = [];
  const keep = (kind, message, extra = {}) => ({ kind, message, next: clone(prev), keepsPrevious: true, ...extra });
  const nameOf = it => it.alias || catalog.catById.get(it.category)?.name;
  const hasPrev = state.items.length > 0;

  if (!text || !/[\p{L}\p{N}]/u.test(text)) return keep('clarify', '想买点什么？可以直接说用途和预算，比如「通勤用的降噪耳机，500以内」。', { reason: 'empty' });
  if (/^(你好|您好|hi|hello|嗨|在吗|在不在|哈喽)[!！。.~～]*$/i.test(text)) return keep('clarify', '在的～想买点什么？说说用途和预算，我帮你挑。', { reason: 'greet' });
  if (INJECTION.test(text)) return keep('reject', '这个我帮不了。我只按你的购物需求挑东西，你的购物车和预算没有变。', { reason: 'injection' });

  // questions about the current recommendation's price and fees
  if (/运费.{0,4}(按|算|当)\s*(0|零)|不算运费|运费不算|免掉运费/.test(text)) return keep('reject', '运费不能按 0 算。运费以平台为准，没确认之前我不会把方案标成在预算内。', { reason: 'fee-zero' });
  if (/券|会员价|新人价|补贴|红包|满减/.test(text) && /算|用|叠|能不能/.test(text)) return keep('info', null, { topic: 'coupon' });
  if (/运费|包邮/.test(text) && /多少|怎么|咋|吗|呢|？|\?/.test(text)) return keep('info', null, { topic: 'shipping' });
  if (/价格|价钱|这个价|的价|价吗/.test(text) && /现在|实时|准|最新|多久|什么时候|啥时候|哪天|是不是|吗/.test(text) && !/预算/.test(text)) return keep('info', null, { topic: 'price' });

  let budget = parseBudget(text);
  if (budget?.error === 'currency') return keep('clarify', '目前只能按人民币计算价格，换成人民币预算告诉我吧，比如「800 元」。', { reason: 'currency' });
  if (budget?.error === 'negative' || budget?.error === 'zero') return keep('clarify', '预算需要大于 0，告诉我一个具体金额吧，比如「300 以内」。', { reason: 'budget-invalid' });
  if (budget?.error === 'precision') return keep('clarify', `金额最多精确到分，「${budget.token}」没法识别，换个写法试试？`, { reason: 'budget-invalid' });
  if (budget?.error === 'too_large') return keep('clarify', '这个金额太大了，确认一下预算是多少？', { reason: 'budget-invalid' });
  if (budget?.error === 'multiple') return keep('clarify', `你提到了几个金额（${budget.values.map(price).join('、')}），总预算按哪个来？`, { reason: 'budget-multiple', options: budget.values.map(v => `预算${v / 100}元`) });
  if (budget?.error) return keep('clarify', '预算没看明白，能说个具体金额吗？比如「总共 800 元」。', { reason: 'budget-unreadable' });
  if (budget?.delta) {
    if (!Number.isSafeInteger(prev?.budgetCents)) return keep('clarify', '现在还没有总预算，直接告诉我一个总价上限吧。', { reason: 'delta-no-budget' });
    const nb = prev.budgetCents + budget.delta;
    if (nb <= 0) return keep('clarify', '这样预算就不够了，告诉我新的总预算是多少？', { reason: 'budget-invalid' });
    budget = { cents: nb, scope: 'total', unitHints: budget.unitHints, note: `总预算${budget.delta > 0 ? '加' : '减'} ${price(Math.abs(budget.delta))}，现在是 ${price(nb)}` };
  }
  if (budget?.unlimited) return keep('clarify', Number.isSafeInteger(prev?.budgetCents) ? `我还是按 ${price(prev.budgetCents)} 帮你控制总价；想提高预算，直接告诉我新的金额。` : '给我一个总价上限，我才能帮你把钱花在刀刃上。大概想花多少？', { reason: 'budget-unlimited' });

  const mentions = findMentions(text, catalog);
  if (mentions[0]?.typed) return keep('clarify', `你说的「${mentions[0].typed}」是指${mentions[0].c.shortName || mentions[0].alias}吗？`, { reason: 'typo', options: [text.replace(mentions[0].typed, mentions[0].alias)] });
  const off = OFF_CATALOG.filter(w => text.includes(w) && !mentions.some(m => m.alias.includes(w)));
  const offName = w => w.replace(/^(辆|一套|买)/, '');
  if (off.length && !mentions.length && (ADD.test(text) || !hasPrev || CREATE.test(text))) return keep('reject', `${offName(off[0])}这类东西暂时没有。现在可以帮你挑：${catalog.categories.slice(0, 6).map(c => c.name).join('、')}等。`, { reason: 'out-of-catalog' });
  if (!off.length && mentions.length && (!hasPrev || CREATE.test(text) || ADD.test(text))) off.push(...unknownListMembers(text, catalog));
  if (off.length && mentions.length) return keep('clarify', `${off.map(offName).join('、')}暂时没有，要先帮你挑${mentions.map(m => m.alias).join('、')}吗？`, { reason: 'partial-off-catalog', options: [`先挑${mentions.map(m => m.alias).join('、')}`], pendingBudget: budget?.cents ? budget : undefined });
  if (/无线/.test(text) && /有线/.test(text) && !/有线和无线|有线或无线|有线无线都/.test(text)) return keep('clarify', '你说了无线又说必须有线，按哪个来？', { reason: 'conflict', options: ['要无线的', '要有线的'] });
  const floor = priceFloor(text);
  const budgetNow = budget?.cents ?? prev?.budgetCents;
  if (floor && budgetNow && floor > budgetNow) return keep('clarify', `你想要 ${price(floor)} 以上的，但总预算是 ${price(budgetNow)}，以哪个为准？`, { reason: 'conflict-floor', options: [`预算改成${floor / 100}`, `就按${budgetNow / 100}以内`] });

  const findItems = m => state.items.filter(i => i.category === m.c.id && (!m.subtype || !i.keywords?.length || i.keywords.includes(m.subtype)));
  let changed = false;

  // ---------- view only: never edits list or budget ----------
  if (VIEW.test(text) && !ONLY.test(text) && !REMOVE.test(text) && !budget && (hasPrev || !mentions.length)) {
    if (!hasPrev) return keep('clarify', '购物车里还没有东西，先说说想买什么吧。', { reason: 'view-empty' });
    if (/整套/.test(text)) return { kind: 'view', view: { mode: 'bundle' }, next: state, notes: ['切回整套，购物车和预算不变。'] };
    const target = mentions.length ? findItems(mentions[0])[0] : state.items[0];
    if (!target) return keep('clarify', `购物车里没有${mentions[0].alias}，要加进来吗？`, { reason: 'view-missing', options: [`加一个${mentions[0].alias}`] });
    return { kind: 'view', view: { mode: 'single', itemId: target.id }, next: state, notes: ['只是换个方式看，购物车和总预算都没变。'] };
  }
  if (hasPrev && /挑一件|随便一件|选一件|买一件就行|一件就行/.test(text) && !mentions.length) return keep('clarify', '只买其中一件吗？要哪一件？', { reason: 'which-one', options: state.items.map(i => `只要${nameOf(i)}`) });
  if (hasPrev && /^(不要了|算了|不买了|删了吧|都不要了)[。.!！]*$/.test(text)) return keep('clarify', '是整个购物车都不要了，还是去掉其中一件？', { reason: 'remove-which', options: [...state.items.map(i => `去掉${nameOf(i)}`), '清空重新挑'] });

  // ---------- edits ----------
  const rep = hasPrev ? text.match(REPLACE) : null;
  if (rep && findMentions(rep[1], catalog).length && findMentions(rep[2], catalog).length) {
    const from = findMentions(rep[1], catalog)[0];
    const target = findItems(from)[0];
    if (!target) return keep('clarify', `购物车里没有${from.alias}，没有改动。`, { reason: 'replace-missing' });
    if (target.lock) return keep('clarify', `${nameOf(target)}已锁定，先解锁再换？`, { reason: 'replace-locked', options: [`解锁${nameOf(target)}`] });
    const { items: fresh } = mentionItems(rep[2], findMentions(rep[2], catalog), catalog);
    state.items.splice(state.items.indexOf(target), 1, ...fresh.filter(f => !f.error).map(f => ({ ...f, qty: target.qty })));
    notes.push(`把${nameOf(target)}换成了${fresh.map(f => f.alias).join('、')}`); changed = true;
  } else if (hasPrev && UNLOCK.test(text)) {
    const targets = mentions.length ? mentions.flatMap(findItems) : state.items.filter(i => i.lock);
    targets.forEach(i => { i.lock = null; }); changed = targets.length > 0;
    if (changed) notes.push(`已解锁${targets.map(nameOf).join('、')}`);
  } else if (hasPrev && LOCK.test(text)) {
    const targets = mentions.length ? mentions.flatMap(findItems) : state.items.length === 1 ? state.items : [];
    if (!targets.length) return keep('clarify', '要锁定哪一件？', { reason: 'lock-which', options: state.items.map(i => `锁定${nameOf(i)}`) });
    for (const i of targets) { if (!i.selected) return keep('clarify', `${nameOf(i)}还没选好，先选一款再锁定。`, { reason: 'lock-unselected' }); i.lock = i.selected; }
    notes.push(`已锁定${targets.map(nameOf).join('、')}，之后改预算也不会换掉它。`); changed = true;
  } else if (hasPrev && REMOVE.test(text) && mentions.length && !brandsIn(text, catalog).avoid.length) {
    const SWITCH = new RegExp(SWITCH_WORDS);
    const removeText = clauses(text).filter(c => REMOVE.test(c)).map(c => c.split(SWITCH)[0]).join('，');
    for (const m of findMentions(removeText, catalog)) {
      const hit = findItems(m);
      if (!hit.length) return keep('clarify', `购物车里本来就没有${m.alias}，没有改动。`, { reason: 'remove-missing' });
      state.items = state.items.filter(i => !hit.includes(i));
      notes.push(`去掉了${hit.some(i => i.lock) ? '已锁定的' : ''}${m.alias}`); changed = true;
    }
    // in a removal turn, any other clause that names a category is an addition, whatever verb introduces it
    const addText = clauses(text).map(c => SWITCH.test(c) ? c.split(SWITCH).slice(1).join('') : (!REMOVE.test(c) ? c : '')).filter(c => c && findMentions(c, catalog).length).join('，');
    if (addText) { const a = mentionItems(addText, findMentions(addText, catalog), catalog); state.items.push(...a.items.filter(i => !i.error)); if (a.items.length) notes.push(`加上了${a.items.map(i => i.alias).join('、')}`); }
  } else if (hasPrev && ONLY.test(text) && mentions.length) {
    const keepIds = new Set(mentions.flatMap(findItems).map(i => i.id));
    const dropped = state.items.filter(i => !keepIds.has(i.id));
    state.items = state.items.filter(i => keepIds.has(i.id));
    const missing = mentions.filter(m => !findItems(m).length);
    if (missing.length) state.items.push(...mentionItems(text, missing, catalog).items.filter(i => !i.error));
    notes.push(`购物车只保留${mentions.map(m => m.alias).join('、')}${dropped.length ? `，去掉了${dropped.map(nameOf).join('、')}` : ''}`); changed = true;
  } else if (hasPrev && QTY_MORE.test(text) && (mentions.length ? mentions.every(m => findItems(m).length) : state.items.length === 1)) {
    const tok = text.match(QTY_MORE)[1], add = tok ? qtyOf(tok) : 1;
    const target = mentions.length ? findItems(mentions[0])[0] : state.items[0];
    if (!Number.isSafeInteger(add) || add <= 0 || target.qty + add > 50) return keep('clarify', '数量得是 1 到 50 之间的整数。', { reason: 'qty-invalid' });
    target.qty += add; notes.push(`${nameOf(target)}改成 ${target.qty} 件`); changed = true;
  } else if (hasPrev && mentions.length === 1 && /换成|改成|换个|要/.test(text) && TYPE_WORDS.some(w => text.includes(w)) && findItems(mentions[0]).length) {
    const it = findItems(mentions[0])[0];
    if (it.lock) return keep('clarify', `${nameOf(it)}已锁定，先解锁才能换。`, { reason: 'swap-locked', options: [`解锁${nameOf(it)}`] });
    const tw = TYPE_WORDS.filter(w => text.includes(w));
    const stated = tw.filter(w => catalog.products.some(p => p.category === it.category && hasKeyword(p, w)));
    if (!stated.length) return keep('clarify', `商品页没有标注「${tw.join('、')}」，没法确认哪款是${tw.join('、')}的。`, { reason: 'type-unknown' });
    it.keywords = [...new Set([...(it.keywords || []).filter(k => !TYPE_WORDS.includes(k)), ...stated])]; it.selected = null;
    notes.push(`${nameOf(it)}换成${stated.join('、')}款`); changed = true;
  } else if (hasPrev && SWAP.test(text)) {
    const targets = mentions.length ? mentions.flatMap(findItems) : state.items.length === 1 ? state.items : [];
    if (!targets.length) return keep('clarify', '想换哪一件？', { reason: 'swap-which', options: state.items.map(i => `换一款${nameOf(i)}`) });
    for (const i of targets) {
      if (i.lock) return keep('clarify', `${nameOf(i)}已锁定，先解锁才能换。`, { reason: 'swap-locked', options: [`解锁${nameOf(i)}`] });
      if (i.selected) i.exclude = [...new Set([...(i.exclude || []), i.selected])];
      i.selected = null;
    }
    notes.push(`给${targets.map(nameOf).join('、')}换一款，其他不动。`); changed = true;
  } else if (hasPrev && QTY_CHANGE.test(text) && (mentions.length ? mentions.every(m => findItems(m).length) : state.items.length === 1) && !/想买|帮我|推荐/.test(text)) {
    const qm = text.match(QTY_CHANGE), tok = qm[1] || qm[2], v = qtyOf(tok);
    if (!Number.isSafeInteger(v) || v <= 0 || v > 50) return keep('clarify', `数量「${tok}」得是 1 到 50 之间的整数。`, { reason: 'qty-invalid' });
    const target = mentions.length ? findItems(mentions[mentions.length - 1])[0] : state.items[0];
    target.qty = v; notes.push(`${nameOf(target)}改成 ${v} 件`); changed = true;
  } else if (hasPrev && QTY_CHANGE.test(text) && !mentions.length) {
    return keep('clarify', '要改哪一件的数量？', { reason: 'qty-which', options: state.items.map(i => `${nameOf(i)}改成2件`) });
  } else if (mentions.length) {
    const parsed = mentionItems(text, mentions, catalog);
    const bad = parsed.items.find(p => p.error);
    if (bad) return keep('clarify', bad.error + '，再说一次？', { reason: 'qty-invalid' });
    // "两把60左右，带无线的鼠标": a "数量+量词" left over in another clause belongs to the only product named, otherwise ask
    const stray = [...text.matchAll(new RegExp(`(${NUM})\\s*(${MEASURE})(?![\\u4e00-\\u9fa5]{0,6}?(?:${mentions.map(m => m.alias).join('|') || '^$'}))`, 'g'))]
      .filter(x => !/^一$/.test(x[1]) && !SPEC_UNIT.test(text.slice(x.index + x[0].length)) && !/(块钱|块|元)\s*$/.test(text.slice(0, x.index)));
    if (stray.length && parsed.items.every(i => i.qty === 1)) {
      const v = qtyOf(stray[0][1]);
      if (parsed.items.length === 1 && Number.isSafeInteger(v) && v > 1 && v <= 50) {
        parsed.items[0].qty = v;
        const pr = text.slice(stray[0].index + stray[0][0].length).match(/^\s*(\d+(?:\.\d+)?)\s*(块钱|块|元)?\s*(左右|上下|以内|以下|多|来|出头)?/);
        if (pr && (pr[2] || pr[3])) { if (/多|来|出头/.test(pr[3] || '')) parsed.items[0].approxPrice = true; else { parsed.items[0].unitCapCents = moneyToken(pr[1]).cents; parsed.items[0].boundPrice = true; } }
      }
      else if (parsed.items.length > 1) return keep('clarify', `「${stray[0][0]}」是指哪一样？`, { reason: 'qty-which', options: parsed.items.map(i => `${i.alias}要${stray[0][0]}`) });
    }
    const dd = dedupe(parsed.items, text);
    if (dd.ambiguous) return keep('clarify', `你说了两次${dd.ambiguous}，是要 2 件，还是 1 件就够？`, { reason: 'dup', options: [`${dd.ambiguous}要2件`, `${dd.ambiguous}1件就够`] });
    const bu = brandsIn(text, catalog).unsure;
    if (bu.length) {
      const pending = clone(prev); pending.items = dd.items;
      if (budget?.cents && budget.scope !== 'unit') pending.budgetCents = budget.cents;
      return { kind: 'clarify', reason: 'brand-unsure', next: pending, message: `「${bu[0]}」是只看这个牌子，还是不要这个牌子？`, options: [`只要${bu[0]}的`, `不要${bu[0]}的`], notes: [] };
    }
    warnings.push(...parsed.warnings);
    const isAdd = hasPrev && ADD.test(text);
    const isNew = !hasPrev || (!isAdd && (CREATE.test(text) || budget));
    if (isAdd) { state.items.push(...dd.items); notes.push(`加上了${dd.items.map(p => p.alias).join('、')}`); changed = true; }
    else if (isNew) {
      const locked = state.items.filter(i => i.lock && !dd.items.some(p => p.category === i.category));
      if (locked.length) return keep('clarify', `新需求里没有已锁定的${locked.map(nameOf).join('、')}，要解锁并重新挑吗？`, { reason: 'new-drops-locked', options: ['解锁后重新挑', '保留原来的购物车'] });
      state.items = dd.items.map(p => { const old = state.items.find(i => i.category === p.category && i.lock); return old ? { ...p, lock: old.lock, selected: old.selected, id: old.id } : p; });
      state.prefs = []; changed = true;
    } else {
      const inList = mentions.filter(m => findItems(m).length);
      if (inList.length === mentions.length) { const n = inList[0].alias; return keep('clarify', `${n}已经在购物车里了，想怎么处理？`, { reason: 'item-action', options: [`锁定${n}`, `换一款${n}`, `去掉${n}`, `只看${n}`] }); }
      return keep('clarify', `是把${mentions.map(m => m.alias).join('、')}加进当前购物车，还是重新挑？`, { reason: 'add-or-new', options: [`再加${mentions.map(m => m.alias).join('、')}`, `重新挑${mentions.map(m => m.alias).join('、')}`] });
    }
    if (parsed.unsupported.length) {
      const u = parsed.unsupported[0];
      if (budget?.cents && budget.scope === 'total') state.budgetCents = budget.cents;
      return { kind: 'clarify', reason: 'unsupported-requirement', next: state, message: `「${u.label}」商品页都没有写，不能确认哪款满足。要不要先不考虑这一条，按其他要求帮你挑？`, options: ['不考虑这一条，先挑'], notes };
    }
  }

  // ---------- brand-only edits on the current list ----------
  if (hasPrev && !mentions.length) {
    const bb = brandsIn(text, catalog);
    for (const it of state.items) {
      for (const b of bb.avoid) if (!(it.excludeBrands || []).includes(b)) { it.excludeBrands = [...(it.excludeBrands || []), b]; it.hard = it.hard.filter(h => !(h.key === 'brand' && h.value.includes(b))); changed = true; }
      const want = bb.want.filter(b => catalog.products.some(p => p.category === it.category && brandHit(p, [b])));
      if (want.length) { it.hard = [...it.hard.filter(h => h.key !== 'brand'), { key: 'brand', op: 'in', value: want, label: `${want.join('/')}品牌` }]; it.excludeBrands = (it.excludeBrands || []).filter(b => !want.includes(b)); changed = true; }
    }
    if (bb.avoid.length) notes.push(`不考虑${bb.avoid.join('、')}`);
    if (bb.want.length) notes.push(`只看${bb.want.join('、')}`);
    if (bb.unsure.length && !bb.avoid.length && !bb.want.length) return keep('clarify', `「${bb.unsure[0]}」是只看这个牌子，还是不要这个牌子？`, { reason: 'brand-unsure', options: [`只要${bb.unsure[0]}的`, `不要${bb.unsure[0]}的`] });
  }

  // ---------- whole-list conditions ----------
  if (!mentions.length && hasPrev && REQUIRE.test(text)) {
    let applied = false;
    for (const it of state.items) {
      for (const h of hardFrom(text, catalog.catById.get(it.category), catalog)) if (!it.hard.some(x => x.key === h.key)) { it.hard.push(h); applied = changed = true; notes.push(`${nameOf(it)}：${h.label}`); }
    }
    if (!applied) return keep('clarify', `「${unsupportedRequirement(text, []) || text}」商品页都没有写，不能确认哪款满足，购物车和预算没有变。`, { reason: 'unsupported-requirement' });
  }
  if (floor && !(budgetNow && floor > budgetNow) && state.items.length) { for (const it of state.items) it.unitFloorCents = floor; changed = true; }
  if (budget?.unitHints?.length && state.items.length) {
    let need = 0, over = false;
    for (const h of budget.unitHints) {
      const before = mentions.filter(m => m.index < h.at);
      const owner = state.items.length === 1 ? state.items[0] : before.length ? state.items.find(i => i.alias === before[before.length - 1].alias) : null;
      if (!owner) continue;
      if (!h.cap) { need += h.cents * owner.qty; over ||= h.over; }
      if (h.rough) owner.approxPrice = true; else { owner.unitCapCents = h.cents; owner.boundPrice = true; }
    }
    if (Number.isSafeInteger(budget.cents) && need && (need > budget.cents || (over && need >= budget.cents))) {
      return { kind: 'clarify', reason: 'conflict-unit-total', next: { ...state, budgetCents: null }, message: `按你说的单价，这几件${over ? '至少' : '就要'} ${price(need)}，${need > budget.cents ? '超过' : '已经到'}了总共 ${price(budget.cents)}。按哪个来？`, options: [`总共${budget.cents / 100}元`, `总共${Math.ceil(need / 100) + (over ? 50 : 0)}元`], notes: [] };
    }
    if (!Number.isSafeInteger(budget.cents)) budget = null;
  }
  // prices written next to each product ("一对两百左右的耳机，再来俩八十左右的台灯") belong to that product only
  const TOTAL_MARK = /预算|总共|一共|合计|总价|加起来|全部/.test(text);
  const fresh = state.items.filter(i => i.boundPrice || i.approxPrice);
  if (fresh.length && !(budget?.cents && budget.scope === 'total') && (mentions.length || state.items.every(i => i.boundPrice || i.approxPrice))) {
    if (state.items.every(i => i.boundPrice)) budget = { cents: state.items.reduce((a, i) => a + i.unitCapCents * i.qty, 0), scope: 'bound' };
    else budget = null; // a rough price ("四十来块") or an unpriced item: ask for the total instead of guessing
  }
  if (budget?.scope === 'maybe-unit') budget.scope = state.items.some(i => i.qty > 1) ? 'unit' : 'total';
  if (budget?.cents) {
    if (budget.scope === 'bound') {
      state.budgetCents = budget.cents; state.budgetScope = 'unit';
      notes.push(`${state.items.map(i => `${nameOf(i)}每件不超过 ${price(i.unitCapCents)}`).join('，')}，总价不超过 ${price(budget.cents)}`);
    } else if (budget.scope === 'unit') {
      if (!state.items.length) return keep('clarify', `每件 ${price(budget.cents)} 以内，想买什么？`, { reason: 'unit-no-items' });
      for (const it of state.items) it.unitCapCents = budget.cents;
      state.budgetCents = state.items.reduce((a, it) => a + budget.cents * it.qty, 0); state.budgetScope = 'unit';
      notes.push(`每件不超过 ${price(budget.cents)}，总价不超过 ${price(state.budgetCents)}`);
    } else {
      state.budgetCents = budget.cents; state.budgetScope = 'total';
      for (const it of state.items) if (!it.boundPrice) delete it.unitCapCents;
      notes.push(budget.note || (budget.upper ? `「${budget.rangeText}」按不超过 ${price(budget.cents)} 来挑` : `总预算 ${price(budget.cents)}${budget.approx ? '，按不超过这个数来挑' : ''}`));
    }
    changed = true;
  }
  const giftLike = /送|礼物|每人|各买|各一|大家|同事们|朋友们|全家/.test(text);
  if (budget?.cents && budget.scope === 'total' && giftLike && state.items.some(i => i.qty > 1) && !/总|一共|合计|全部|加起来/.test(text)) {
    return { kind: 'clarify', reason: 'budget-scope', next: { ...state, budgetCents: null }, message: `${price(budget.cents)} 是这几件的总预算，还是每件的预算？`, options: [`总共${budget.cents / 100}元`, `每个${budget.cents / 100}元`], notes: [] };
  }
  if (/便宜(一)?点|再便宜|更便宜/.test(text) && hasPrev && Number.isSafeInteger(prev.lastTotalCents)) { state.capCents = prev.lastTotalCents - 1; notes.push(`找比现在 ${price(prev.lastTotalCents)} 更便宜的`); changed = true; }
  for (const [re, k] of PREF_WORDS) if (re.test(text) && !state.prefs.includes(k)) { state.prefs.push(k); changed = true; }
  if (/不(用|需要|要求)(安静|静音)/.test(text)) { state.prefs = state.prefs.filter(p => p !== 'quiet'); changed = true; }
  if (state.prefs.includes('cheap') && state.prefs.includes('premium')) warnings.push('「便宜」和「旗舰」会互相拉扯，我会在你的总价上限里挑参数更均衡的');
  for (const w of UNFILTERABLE) if (text.includes(w)) warnings.push(`「${w}」暂时没法作为筛选条件，先按其他要求挑`);

  if (!changed) {
    if (budget?.cents && !state.items.length) return keep('clarify', `好的，预算 ${price(budget.cents)}。想买什么呢？`, { reason: 'budget-only', pendingBudget: budget });
    if (!hasPrev) return keep('clarify', `没太明白想买什么。现在可以帮你挑：${catalog.categories.slice(0, 6).map(c => c.name).join('、')}等，说说用途和预算吧。`, { reason: 'unknown' });
    return keep('clarify', '这句没太理解。可以改预算、换一款、去掉或加上某样东西，也可以重新说需求。', { reason: 'unknown-edit' });
  }
  if (!state.items.length && !hasPrev) return keep('clarify', `${budget?.cents ? `好的，预算 ${price(budget.cents)}。` : ''}想买哪一类？现在可以帮你挑：${catalog.categories.slice(0, 6).map(c => c.shortName || c.name).join('、')}等。`, { reason: 'budget-only', pendingBudget: budget?.cents ? budget : undefined });
  if (!state.items.length) return keep('clarify', '购物车会变成空的，至少保留一样吧？', { reason: 'empty-list' });
  for (const it of state.items) delete it.approxPrice; // only meaningful in the turn it was said
  return { kind: 'update', next: state, notes, warnings };
}
