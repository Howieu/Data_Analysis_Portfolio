import { understand } from './nlu.mjs';
import { solve, candidatesFor, score, categoryStats, FACT_DIRECTION } from './engine.mjs';
import { guideFor } from './guide.mjs';
import { price } from './money.mjs';
import { isFresh, available } from './catalog.mjs';

export const emptySession = () => ({ state: { items: [], budgetCents: null, budgetScope: 'total', prefs: [] }, view: { mode: 'bundle' }, result: null });

const FACT_TEXT = {
  windSpeed: v => `风速 ${v}m/s`, batteryHours: v => `续航约 ${v} 小时`, batteryMah: v => `电池 ${v}mAh`, weightG: v => `约 ${v}g`, noiseDb: v => `噪音 ${v}dB`,
  noiseCancelling: v => v ? '支持主动降噪' : null, ancDepthDb: v => `降噪深度 ${v}dB`, wireless: v => v ? '无线' : null, capacityL: v => `${v}L`, wattage: v => `${v}W`,
};
export const factLine = p => Object.entries(p.facts || {}).map(([k, v]) => FACT_TEXT[k]?.(v)).filter(Boolean).slice(0, 4);
const dateText = iso => { const d = new Date(iso); return Number.isFinite(d.getTime()) ? `${d.getMonth() + 1}月${d.getDate()}日` : ''; };

export function productView(p, now = Date.now()) {
  if (!p) return null;
  return { id: p.id, title: p.title, brand: p.brand, source: p.source, url: p.url, image: p.images?.[0] || null, priceCents: p.priceCents,
    fresh: isFresh(p, now), observedAt: p.observedAt, observedText: dateText(p.observedAt), facts: factLine(p), highlights: (p.highlights || []).slice(0, 3), category: p.category };
}

// Budget state wording is fixed; "在预算内" only when every fee is known and fresh.
export function budgetState(result) {
  if (!result || !result.lines) return null;
  if (result.withinBudget) return { kind: 'within', text: `合计 ${price(result.totalCents)} · 在预算内` };
  const stale = result.stale, shipUnknown = result.shippingCents === null;
  return { kind: 'pending', text: `商品价合计 ${price(result.subtotalCents)}${shipUnknown ? '，运费以平台为准' : ''}${stale ? '，价格需要刷新确认' : ''}` };
}

const PREF_TEXT = { quiet: '安静', light: '轻便', battery: '续航', wind: '风力', cheap: '价格实惠', premium: '规格好一些' };
function reasonFor(p, item, session) {
  const facts = factLine(p);
  const hard = (item.hard || []).filter(h => h.key !== 'brand').map(h => h.label);
  const prefs = (session.state.prefs || []).map(k => PREF_TEXT[k]).filter(Boolean);
  const parts = [];
  if (facts.length) parts.push(facts.join('，'));
  if (p.highlights?.[0] && !facts.some(f => p.highlights[0].includes(f.replace(/[^\d]/g, '')) && /\d/.test(f))) parts.push(p.highlights[0]);
  let s = parts.join('；');
  if (hard.length) s += `${s ? '。' : ''}满足你要的${hard.join('、')}`;
  if (prefs.length) s += `${s ? '，' : ''}按你在意的${prefs.join('、')}优先挑的`;
  return (s || `${p.brand || ''}这款在你的预算里`) + '。';
}

function altViews(result, item, catalog, now) {
  const cur = catalog.byId.get(result.lines.find(l => l.itemId === item.id).productId);
  return (result.alternatives[item.id] || []).slice(0, 3).map(a => {
    const p = catalog.byId.get(a.productId);
    const diff = p.priceCents - cur.priceCents;
    return { ...productView(p, now), diffCents: diff, diffText: diff === 0 ? '同价' : diff < 0 ? `便宜 ${price(-diff)}` : `贵 ${price(diff)}`, withinBudget: a.withinBudget };
  });
}

// Before a budget is set: what each wanted category looks like at different price levels, so the user can pick a budget.
// Bands are price tertiles of the matching products; nothing here is a recommendation or a budget claim.
const BAND_NAMES = ['入门价位', '主流价位', '进阶价位'];
export function overview(state, catalog, now = Date.now()) {
  return state.items.slice(0, 2).map(item => {
    const c = catalog.catById.get(item.category);
    const list = candidatesFor({ ...item, lock: null }, state, catalog);
    if (!list.length) return null;
    const byPrice = [...list].sort((a, b) => a.p.priceCents - b.p.priceCents);
    const k = byPrice.length >= 6 ? 3 : 1, size = Math.ceil(byPrice.length / k);
    const bands = Array.from({ length: k }, (_, i) => byPrice.slice(i * size, (i + 1) * size)).filter(b => b.length).map((b, i, all) => {
      const brands = Object.entries(b.reduce((m, x) => (x.p.brand && (m[x.p.brand] = (m[x.p.brand] || 0) + 1), m), {})).sort((x, y) => y[1] - x[1]).map(([n]) => n);
      const top = [...b].sort((x, y) => y.s - x.s).filter(x => x.p.images?.length).slice(0, 3);
      return { name: all.length > 1 ? BAND_NAMES[i] : '在售款式', minCents: b[0].p.priceCents, maxCents: b.at(-1).p.priceCents, count: b.length, brands: brands.slice(0, 3), products: top.map(x => productView(x.p, now)) };
    });
    const g = guideFor(c.name);
    return { category: c.id, name: item.alias || c.shortName, count: list.length, bands, guide: g && { title: g.title, points: g.points, caution: g.caution } };
  }).filter(Boolean);
}

function compose(session, result, understood, catalog, now) {
  const s = session.state;
  const names = s.items.map(i => i.alias || catalog.catById.get(i.category)?.shortName);
  const r = { status: 'recommend', notes: understood.notes || [], warnings: understood.warnings || [], blocks: [] };
  if (result.status === 'no-solution' || result.status === 'incomplete') {
    r.status = 'no_solution';
    const d = result.diagnosis || {};
    if (d.emptyItems?.length) {
      for (const e of d.emptyItems) {
        const name = catalog.catById.get(e.category)?.name;
        const b = e.blockers.find(x => x.yes === 0);
        r.blocks.push({ type: 'text', text: b ? `${name}里还没有确认满足「${b.label}」的${b.unknown ? `（有 ${b.unknown} 款没写这项参数，没法算作满足）` : ''}。` : e.unitCapCents ? `${name}没有单价在 ${price(e.unitCapCents)} 以内的。` : `${name}暂时没有符合这些条件的。` });
      }
      r.actions = ['去掉这个条件', '换个品类'];
    } else if (d.minTotalCents) {
      r.blocks.push({ type: 'text', text: result.status === 'incomplete' ? `组合太多还没比完，暂时没找到 ${price(s.budgetCents)} 以内的方案，可以减少一样再试试。` : `按最便宜的挑，${names.join('、')}也要 ${price(d.minTotalCents)} 起，比预算多 ${price(d.minTotalCents - s.budgetCents)}。` });
      for (const u of d.unknownHard || []) r.blocks.push({ type: 'text', text: `另外有 ${u.count} 款${catalog.catById.get(u.category).name}在预算里，但页面没写这项参数（${u.label}），没法算作满足。` });
      r.actions = [`预算改成${Math.ceil(d.minTotalCents / 10000) * 100}`, ...s.items.filter(i => !i.lock).slice(0, 2).map(i => `去掉${catalog.catById.get(i.category).name}`)];
    }
    r.blocks.push({ type: 'text', text: '预算和锁定的商品我都没有动，你决定怎么调整。' });
    return r;
  }
  const bs = budgetState(result);
  r.summary = { budgetCents: s.budgetCents, budgetScope: s.budgetScope, itemCount: s.items.length, ...result, budgetState: bs };
  const single = s.items.length === 1;
  const first = s.items[0], firstCat = catalog.catById.get(first.category);
  r.opening = single ? `好的，帮你挑${first.qty > 1 ? ` ${first.qty} 件` : '一款'}${names[0]}～` : `好的，${names.join('、')}一起按总预算来配～`;
  if (single) {
    const g = guideFor(firstCat.name);
    if (g) r.blocks.push({ type: 'tips', title: g.title, points: g.points, caution: g.caution });
  } else {
    const tips = s.items.map(i => guideFor(catalog.catById.get(i.category).name)).filter(Boolean);
    if (tips.length) r.blocks.push({ type: 'tips', title: '这几样各看一个重点', points: tips.map(g => [g.title.replace('怎么挑', ''), g.points[0][1]]), caution: null });
  }
  const lines = result.lines.map(l => {
    const item = s.items.find(i => i.id === l.itemId), p = catalog.byId.get(l.productId);
    return { ...l, item: { id: item.id, category: item.category, categoryName: item.alias || catalog.catById.get(item.category).shortName, qty: item.qty, locked: !!item.lock, hard: item.hard.map(h => h.label) }, product: productView(p, now), reason: reasonFor(p, item, session), alternatives: altViews(result, item, catalog, now) };
  });
  r.blocks.push({ type: single ? 'pick' : 'bundle', title: single ? '这个可能最适合你' : '这一套可能最适合你', lines, budgetState: bs });
  if (!result.exhausted) r.blocks.push({ type: 'text', text: '组合比较多，这是目前比出来的结果，还有更多组合没比完。' });
  if (result.upgrade?.length) {
    const u = result.upgrade[0], p = catalog.byId.get(u.productId);
    r.upgrade = { ...productView(p, now), overByCents: u.overByCents, text: `如果愿意多花 ${price(u.overByCents)}${u.totalKnown ? '' : ' 起'}，可以换成这款` };
  }
  r.actions = single ? ['便宜一点', '换一款', '预算改一下', '锁定这款'] : ['按单品看', '换一款' + names[0], '预算改一下', `去掉${names[names.length - 1]}`];
  return r;
}

const summaryOf = session => session.result?.lines ? { ...session.result, budgetState: budgetState(session.result), budgetCents: session.state.budgetCents } : null;

// Answers about price freshness, fees and coupons, grounded in the current plan's own data.
function infoAnswer(topic, session, catalog, now) {
  const lines = session.result?.lines || [];
  const prods = lines.map(l => catalog.byId.get(l.productId)).filter(Boolean);
  const sources = [...new Set(prods.map(p => p.source))];
  if (topic === 'price') {
    if (!prods.length) return '价格以平台为准。先告诉我想买什么，我会标出每件商品价格的更新时间。';
    return prods.map(p => `${p.title}：${price(p.priceCents)}，${dateText(p.observedAt)}在${p.source}看到的${isFresh(p, now) ? '' : '（已过期，需要刷新）'}`).join('；') + '。最终以平台为准。';
  }
  if (topic === 'shipping') {
    if (!prods.length) return '运费以平台为准，选好商品后我会按平台公开的包邮规则帮你算。';
    const r = session.result;
    const rules = sources.map(s => catalog.sources.get(s)?.shippingPolicy?.evidenceText).filter(Boolean);
    const per = (r.perSource || []).map(x => `${x.source}${x.shipping === null ? '运费以平台为准' : x.shipping === 0 ? '免运费' : `运费 ${price(x.shipping)}`}`).join('；');
    return `${per}。${rules.length ? `平台规则：${rules.join('；')}。` : ''}实际运费以平台为准。`;
  }
  return '新人券、会员价这类优惠要看账号资格，我先按普通价帮你算，不把它算进预算。能不能用以平台为准。';
}


// ---------- side-by-side comparison (deterministic; only fields the listings state) ----------
export const COMPARE = /对比|比较|比一比|比一下|比比|哪个好|哪款好|哪个更|有什么区别|有啥区别|什么区别|区别大|差别/;
const FACT_LABEL = { batteryHours: '续航', batteryMah: '电池', windSpeed: '风速', noiseDb: '噪音', noiseCancelling: '主动降噪', ancDepthDb: '降噪深度', weightG: '重量', capacityL: '容量', wattage: '功率', wireless: '无线' };
const shipText = p => p.shippingCents === 0 ? '免运费' : (p.shippingNote || '以平台为准').replace(/[（(][^）)]*[）)]/g, '').slice(0, 24);

// pick the product to compare with: same category, available, close in price, best spec score
export function compareTarget(p, catalog) {
  const pool = catalog.products.filter(x => x.category === p.category && x.id !== p.id && available(x));
  if (!pool.length) return null;
  const near = pool.filter(x => x.priceCents >= p.priceCents * 0.5 && x.priceCents <= p.priceCents * 1.6);
  const list = near.length ? near : pool;
  const st = categoryStats(pool);
  return list.map(x => ({ x, s: score(x, [], st) - Math.abs(x.priceCents - p.priceCents) / Math.max(p.priceCents, 1) * 100 })).sort((a, b) => b.s - a.s || a.x.priceCents - b.x.priceCents)[0].x;
}

export function compareReply(ids, session, catalog, now) {
  const ps = ids.map(id => catalog.byId.get(id)).filter(Boolean).slice(0, 2);
  if (ps.length < 2) return { status: 'clarify', message: '想对比哪两款？在商品页点「对比同类」，或者先告诉我想买什么。', reason: 'compare-which' };
  const [a, b] = ps;
  const rows = [['商品价格', ...ps.map(p => price(p.priceCents))], ['品牌', ...ps.map(p => p.brand || '—')], ['购买平台', ...ps.map(p => p.source)]];
  const keys = [...new Set(ps.flatMap(p => Object.keys(p.facts || {})))].filter(k => FACT_LABEL[k]);
  for (const k of keys) rows.push([FACT_LABEL[k], ...ps.map(p => { const v = p.facts?.[k]; return v === undefined || v === null ? '页面未写' : FACT_TEXT[k]?.(v)?.replace(new RegExp(`^${FACT_LABEL[k]}\\s*`), '') || (v === true ? '支持' : v === false ? '不支持' : String(v)); })]);
  const specKeys = [...new Set(ps.flatMap(p => Object.keys(p.specs || {})))].slice(0, 6);
  for (const k of specKeys) rows.push([k, ...ps.map(p => p.specs?.[k] ? String(p.specs[k]).slice(0, 40) : '页面未写')]);
  rows.push(['运费', ...ps.map(shipText)], ['价格日期', ...ps.map(p => `${dateText(p.observedAt)}${isFresh(p, now) ? '' : '（需刷新）'}`)]);
  // which one first: fits the budget (if any), then stated specs, then cheaper
  const budget = session.state.budgetCents;
  const fits = p => !Number.isSafeInteger(budget) || p.priceCents <= budget;
  const st = categoryStats(catalog.products.filter(x => x.category === a.category));
  const sc = p => score(p, session.state.prefs || [], st);
  const [first, second] = [...ps].sort((x, y) => (fits(y) - fits(x)) || sc(y) - sc(x) || x.priceCents - y.priceCents);
  const better = Object.keys(FACT_DIRECTION).filter(k => typeof first.facts?.[k] === 'number' && typeof second.facts?.[k] === 'number' && (first.facts[k] - second.facts[k]) * FACT_DIRECTION[k] > 0).map(k => FACT_TEXT[k](first.facts[k]));
  const diff = first.priceCents - second.priceCents;
  const firstWhy = [better.length ? `${better.join('、')}` : '', diff < 0 ? `便宜 ${price(-diff)}` : diff > 0 ? `贵 ${price(diff)}，规格更好一些` : '', !fits(second) ? '另一款超出你的预算' : ''].filter(Boolean).join('；') || '两款页面写明的参数接近，这款价格更合适';
  const secondWhy = diff > 0 ? `想省 ${price(diff)} 可以选它` : diff < 0 ? `预算多一些时可以考虑，贵 ${price(-diff)}` : '同价位的另一个选择';
  return { status: 'compare', message: `帮你把这两款放在一起对比了，想换别的商品对比也可以。`, compare: { products: ps.map(p => productView(p, now)), rows, first: { ...productView(first, now), why: firstWhy }, second: { ...productView(second, now), why: secondWhy } }, keepsPrevious: true, summary: summaryOf(session) };
}

// One turn. Returns {session (new), reply}. Never mutates the input session.
export function turn(session0, text, catalog, now = Date.now(), opts = {}) {
  const session = JSON.parse(JSON.stringify(session0 || emptySession()));
  const u = understand(text, session.state, catalog);
  const lost = (u.kind === 'clarify' || u.kind === 'reject') && ['unknown', 'unknown-edit', 'budget-only', 'item-action'].includes(u.reason);
  if (COMPARE.test(text) && (opts.productId ? (lost || u.kind === 'clarify' || !u.next?.items?.length || u.next.items.length === session.state.items.length) : lost)) {
    const cur = opts.productId ? catalog.byId.get(opts.productId) : catalog.byId.get(session.result?.lines?.[0]?.productId);
    const other = opts.productId ? cur && compareTarget(cur, catalog) : catalog.byId.get(session.result?.alternatives?.[session.result.lines?.[0]?.itemId]?.[0]?.productId);
    return { session, reply: compareReply([cur?.id, other?.id], session, catalog, now) };
  }
  if (u.kind === 'info') return { session, reply: { status: 'clarify', message: infoAnswer(u.topic, session, catalog, now), keepsPrevious: true, summary: summaryOf(session) } };
  if ((u.kind === 'clarify' || u.kind === 'reject') && u.keepsPrevious === false || (u.kind === 'clarify' && u.next && !u.keepsPrevious)) {
    session.state = u.next;
  if (session.state.budgetScope === 'unit' && session.state.items.length && session.state.items.every(i => Number.isSafeInteger(i.unitCapCents))) session.state.budgetCents = session.state.items.reduce((t, i) => t + i.unitCapCents * i.qty, 0); session.result = null;
    return { session, reply: { status: 'clarify', reason: u.reason, message: u.message, options: u.options || [], notes: u.notes || [], summary: null } };
  }
  if (u.kind === 'clarify' || u.kind === 'reject') {
    if (u.pendingBudget) session.state.pendingBudget = u.pendingBudget.cents;
    return { session, reply: { status: u.kind, reason: u.reason, message: u.message, options: u.options || [], keepsPrevious: true, summary: session.result ? { ...session.result, budgetState: budgetState(session.result), budgetCents: session.state.budgetCents } : null } };
  }
  if (u.kind === 'view') {
    session.view = u.view;
    const reply = session.result?.lines ? compose(session, session.result, u, catalog, now) : { status: 'clarify', message: '还没有方案可以看。' };
    reply.view = session.view; reply.notes = u.notes;
    return { session, reply };
  }
  session.state = u.next;
  if (!Number.isSafeInteger(session.state.budgetCents) && Number.isSafeInteger(session.state.pendingBudget)) session.state.budgetCents = session.state.pendingBudget;
  delete session.state.pendingBudget;
  if (!Number.isSafeInteger(session.state.budgetCents)) {
    const names = session.state.items.map(i => i.alias || catalog.catById.get(i.category).shortName).join('、');
    const browse = overview(session.state, catalog, now), b0 = browse[0], mid = b0?.bands[Math.floor((b0.bands.length - 1) / 2)];
    const message = browse.length === 1 && b0.bands.length > 1
      ? `${b0.name}从 ${price(b0.bands[0].minCents)} 到 ${price(b0.bands.at(-1).maxCents)} 都有，多数人会选 ${price(mid.minCents)}–${price(mid.maxCents)} 这一档。先挑了几款参数比较均衡的，大概想花多少跟我说一声，我按总价帮你定下来。`
      : `${names}想控制在多少钱以内？告诉我总预算，我按总价帮你配。`;
    return { session, reply: { status: 'clarify', message, options: ['200以内', '500以内', '1000以内'], notes: u.notes, summary: null, browse } };
  }
  const result = solve(session.state, catalog, { now, ...opts });
  const cheaperAsked = Number.isSafeInteger(session.state.capCents);
  delete session.state.capCents; // one-shot: "便宜一点" never lowers the user's own budget
  session.result = result;
  if (result.lines) session.state.lastTotalCents = result.totalCents ?? result.subtotalCents;
  if (cheaperAsked && !result.lines) return { session: { ...session, result: session0.result, state: { ...session.state, items: session0.state.items } }, reply: { status: 'clarify', message: '在现在的条件下，已经找不到更便宜的组合了。可以去掉一件，或者放宽某个条件。', keepsPrevious: true } };
  if (result.lines) for (const l of result.lines) { const it = session.state.items.find(i => i.id === l.itemId); it.selected = l.productId; }
  const reply = compose(session, result, u, catalog, now);
  reply.view = session.view;
  return { session, reply };
}

// Re-price a saved session against the current catalog/time: never silently swap, keep items and locks.
export function refresh(session0, catalog, now = Date.now()) {
  const session = JSON.parse(JSON.stringify(session0));
  if (!session.result?.lines) return { session, changed: false };
  const missing = session.result.lines.some(l => !catalog.byId.get(l.productId));
  const res = missing ? null : solve({ ...session.state, items: session.state.items.map(i => ({ ...i, lock: i.selected })) }, catalog, { now });
  const changed = !res || !res.lines || res.totalCents !== session.result.totalCents || res.withinBudget !== session.result.withinBudget;
  if (changed) session.result = { ...session.result, withinBudget: false, stale: true, staleReason: missing ? 'missing' : 'price' };
  return { session, changed };
}

// Flattened text for scanning and benchmark assertions.
export function plainText(reply) {
  const out = [reply.message, reply.opening, ...(reply.notes || []), ...(reply.warnings || []), reply.upgrade?.text, reply.summary?.budgetState?.text];
  for (const b of reply.blocks || []) {
    out.push(b.text, b.title, b.caution, b.budgetState?.text);
    for (const [k, v] of b.points || []) out.push(k, v);
    for (const l of b.lines || []) { out.push(l.product?.title, l.reason, l.item?.categoryName); for (const a of l.alternatives || []) out.push(a.title, a.diffText); }
  }
  out.push(...(reply.options || []), ...(reply.actions || []));
  return out.filter(Boolean).join('\n');
}

// Direct edits from buttons (no language understanding involved). Same solve + compose path as chat.
export function applyOp(session0, op, catalog, now = Date.now()) {
  const session = JSON.parse(JSON.stringify(session0 || emptySession()));
  const s = session.state, item = s.items.find(i => i.id === op.itemId);
  const fail = message => ({ session: session0, reply: { status: 'clarify', message, keepsPrevious: true } });
  switch (op.op) {
    case 'add': {
      const p = catalog.byId.get(op.productId); if (!p) return fail('这件商品已下架或找不到了。');
      // picked by hand → locked to that product
      s.items.push({ id: globalThis.crypto.randomUUID().slice(0, 8), category: p.category, alias: catalog.catById.get(p.category).shortName, qty: 1, hard: [], keywords: [], excludeBrands: [], exclude: [], lock: p.id, selected: p.id });
      break;
    }
    case 'lock': if (!item?.selected) return fail('先选好这一件再锁定。'); item.lock = item.selected; break;
    case 'unlock': if (!item) return fail('没找到这一件。'); item.lock = null; break;
    case 'remove': if (!item) return fail('没找到这一件。'); s.items = s.items.filter(i => i !== item); break;
    case 'qty': { const q = Number(op.qty); if (!item || !Number.isSafeInteger(q) || q < 1 || q > 50) return fail('数量需要是 1 到 50 之间的整数。'); item.qty = q; break; }
    case 'pick': {
      const p = catalog.byId.get(op.productId);
      if (!item || !p || p.category !== item.category) return fail('这款不能替换到这一项。');
      if (item.lock && item.lock !== p.id) return fail('这一件已锁定，先解锁再换。');
      item.lock = p.id; item.selected = p.id; break; // explicit choice of the user
    }
    case 'budget': {
      const c = Math.round(Number(op.yuan) * 100);
      if (!Number.isSafeInteger(c) || c <= 0 || String(op.yuan).split('.')[1]?.length > 2) return fail('预算需要是大于 0 的金额，最多两位小数。');
      s.budgetCents = c; s.budgetScope = 'total'; for (const i of s.items) delete i.unitCapCents; break;
    }
    case 'view': session.view = op.mode === 'single' ? { mode: 'single', itemId: op.itemId || s.items[0]?.id } : { mode: 'bundle' };
      return { session, reply: session.result?.lines ? { ...compose(session, session.result, { notes: [] }, catalog, now), view: session.view } : { status: 'clarify', message: '还没有方案。' } };
    default: return fail('不支持的操作。');
  }
  if (!s.items.length) { session.result = null; return { session, reply: { status: 'clarify', message: '购物车空了，想买点什么？' } }; }
  if (!Number.isSafeInteger(s.budgetCents)) { session.result = null; return { session, reply: { status: 'clarify', message: '设一个总预算，我帮你把整个购物车配好。', options: ['200以内', '500以内', '1000以内'] } }; }
  const result = solve(s, catalog, { now });
  session.result = result;
  if (result.lines) for (const l of result.lines) s.items.find(i => i.id === l.itemId).selected = l.productId;
  return { session, reply: { ...compose(session, result, { notes: [] }, catalog, now), view: session.view } };
}
