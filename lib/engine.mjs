import { isFresh, available, shippingFor } from './catalog.mjs';

// Fact keys: +1 more is better, -1 less is better. Booleans count as 1/0.
export const FACT_DIRECTION = { batteryMah: 1, batteryHours: 1, windSpeed: 1, noiseCancelling: 1, ancDepthDb: 1, wireless: 1, capacityL: 1, weightG: -1, noiseDb: -1 };
export const PREFS = {
  quiet: { label: '安静', fact: 'noiseDb', dir: -1 },
  light: { label: '轻便', fact: 'weightG', dir: -1 },
  battery: { label: '续航长', fact: ['batteryHours', 'batteryMah'], dir: 1 },
  wind: { label: '风力大', fact: 'windSpeed', dir: 1 },
  anc: { label: '降噪', fact: 'noiseCancelling', dir: 1 },
  cheap: { label: '便宜些', price: -1 },
  premium: { label: '好一点', price: 1 },
};

// A sub-type word the user said, and the ways product titles actually write it.
const KEYWORD_ALT = { 键盘: ['键盘', '键鼠'], 鼠标: ['鼠标', '键鼠'], 手表: ['手表', 'Watch', 'watch'], 扫地: ['扫地', '扫拖'], 吸尘器: ['吸尘器', '吸尘'] };
export const hasKeyword = (p, k) => (KEYWORD_ALT[k] || [k]).some(w => p.title.includes(w) || Object.values(p.specs || {}).join(' ').includes(w));
// how people name a brand vs how listings write it
export const BRAND_ALIASES = [['小米', '米家', 'xiaomi', 'redmi'], ['苹果', 'apple', 'beats'], ['戴森', 'dyson'], ['罗技', 'logitech']];
export const brandHit = (p, names) => names.some(n => (BRAND_ALIASES.find(g => g.includes(n.toLowerCase())) || [n.toLowerCase()]).some(a => `${p.brand} ${p.title}`.toLowerCase().includes(a)));
const num = v => typeof v === 'boolean' ? (v ? 1 : 0) : (typeof v === 'number' && Number.isFinite(v) ? v : null);

// 'yes' | 'no' | 'unknown' for one hard condition
export function checkHard(p, h) {
  if (h.key === 'brand') return brandHit(p, h.value) ? 'yes' : 'no';
  const v = p.facts?.[h.key];
  if (v === null || v === undefined) return 'unknown';
  if (h.op === 'eq') return v === h.value ? 'yes' : 'no';
  if (typeof v !== 'number') return 'unknown';
  if (h.op === 'gte') return v >= h.value ? 'yes' : 'no';
  if (h.op === 'lte') return v <= h.value ? 'yes' : 'no';
  return 'unknown';
}

export function matches(p, item) {
  if (p.category !== item.category || !available(p)) return false;
  if (item.exclude?.includes(p.id) && item.lock !== p.id) return false;
  if (item.excludeBrands?.length && brandHit(p, item.excludeBrands)) return false;
  if (item.keywords?.some(k => !hasKeyword(p, k))) return false;
  if (item.unitCapCents && p.priceCents > item.unitCapCents) return false;
  if (item.unitFloorCents && p.priceCents < item.unitFloorCents) return false;
  return (item.hard || []).every(h => checkHard(p, h) === 'yes');
}

export function categoryStats(products) {
  const st = {};
  for (const p of products) {
    for (const [k, v] of Object.entries(p.facts || {})) {
      const n = num(v); if (n === null || !(k in FACT_DIRECTION)) continue;
      const s = st[k] ||= { min: n, max: n }; s.min = Math.min(s.min, n); s.max = Math.max(s.max, n);
    }
    const s = st.price ||= { min: p.priceCents, max: p.priceCents };
    s.min = Math.min(s.min, p.priceCents); s.max = Math.max(s.max, p.priceCents);
  }
  return st;
}
const norm = (v, s, dir) => !s || s.max === s.min ? 0.5 : dir > 0 ? (v - s.min) / (s.max - s.min) : (s.max - v) / (s.max - s.min);

// Integer score 0..~200. Spec quality (known facts only) + preference fit. Price never rewards spending by itself.
export function score(p, prefs, stats) {
  let q = 0, n = 0;
  for (const [k, dir] of Object.entries(FACT_DIRECTION)) {
    const v = num(p.facts?.[k]); if (v === null) continue;
    q += norm(v, stats[k], dir); n++;
  }
  let s = n ? (q / n) * 60 + Math.min(n, 4) * 5 : 30; // stated specs are worth something on their own
  for (const key of prefs || []) {
    const pr = PREFS[key]; if (!pr) continue;
    if (pr.price) s += norm(p.priceCents, stats.price, pr.price) * 50;
    else { const f = [pr.fact].flat().find(k => num(p.facts?.[k]) !== null); if (f) s += norm(num(p.facts[f]), stats[f], pr.dir) * 50; }
  }
  return Math.round(s * 10);
}

// boost: optional {productId: points} from the agent's semantic recall / rerank. It only reorders products
// that already passed every filter; it never adds a product, never touches price, budget or hard conditions.
export function candidatesFor(item, state, catalog, boost = null) {
  const pool = catalog.products.filter(p => p.category === item.category);
  const stats = categoryStats(pool);
  let list = pool.filter(p => matches(p, item));
  if (item.lock) list = list.filter(p => p.id === item.lock);
  return list.map(p => ({ p, s: score(p, state.prefs, stats) + Math.round(Math.max(-400, Math.min(400, boost?.[p.id] || 0))) })).sort((a, b) => b.s - a.s || a.p.priceCents - b.p.priceCents || a.p.id.localeCompare(b.p.id));
}

// Price a full selection: store-level shipping, freshness. Nothing unknown is filled with 0.
export function priceSelection(lines, catalog, now) {
  const subtotal = lines.reduce((a, l) => a + l.product.priceCents * l.qty, 0);
  const ship = shippingFor(lines, catalog);
  const stale = lines.some(l => !isFresh(l.product, now));
  const total = ship.cents === null ? null : subtotal + ship.cents;
  return { subtotalCents: subtotal, shippingCents: ship.cents, totalCents: total, perSource: ship.perSource, stale };
}

const LIMIT_NODES = 400_000, LIMIT_MS = 250, TOP = 40;

export function solve(state, catalog, opts = {}) {
  const now = opts.now ?? Date.now();
  const items = state.items || [];
  if (!items.length) return { status: 'needs-input', reason: 'no-items' };
  if (!Number.isSafeInteger(state.budgetCents) || state.budgetCents <= 0) return { status: 'needs-input', reason: 'no-budget' };
  const B = Number.isSafeInteger(state.capCents) ? Math.min(state.budgetCents, state.capCents) : state.budgetCents;
  const lists = items.map(it => {
    const all = candidatesFor(it, state, catalog, opts.boost);
    // keep best-scored plus cheapest witnesses so a tight budget can still be met
    const cheap = [...all].sort((a, b) => a.p.priceCents - b.p.priceCents).slice(0, 8);
    const keep = new Map([...all.slice(0, TOP), ...cheap].map(c => [c.p.id, c]));
    return { all, use: [...keep.values()].sort((a, b) => b.s - a.s) };
  });
  const empty = items.map((it, i) => lists[i].all.length ? null : it.id).filter(Boolean);
  if (empty.length) return { status: 'no-solution', diagnosis: diagnose(state, catalog, lists, empty), exhausted: true };

  // Search order: most constrained item first
  const order = items.map((_, i) => i).sort((a, b) => lists[a].use.length - lists[b].use.length);
  const minRest = new Array(order.length + 1).fill(0), maxScoreRest = new Array(order.length + 1).fill(0);
  for (let k = order.length - 1; k >= 0; k--) {
    const i = order[k], it = items[i];
    minRest[k] = minRest[k + 1] + Math.min(...lists[i].use.map(c => c.p.priceCents)) * it.qty;
    maxScoreRest[k] = maxScoreRest[k + 1] + Math.max(...lists[i].use.map(c => c.s));
  }
  const started = performance.now();
  let visited = 0, exhausted = true;
  let best = null, bestUnconfirmed = null;
  const pick = new Array(items.length);
  const better = (cand, cur) => !cur || cand.score > cur.score || (cand.score === cur.score && cand.cost < cur.cost);
  const visit = (k, cost, sc) => {
    if (!exhausted) return;
    if (++visited > (opts.maxNodes ?? LIMIT_NODES) || (visited % 2048 === 0 && performance.now() - started > (opts.maxMs ?? LIMIT_MS))) { exhausted = false; return; }
    if (k === order.length) {
      const lines = items.map((it, i) => ({ itemId: it.id, product: pick[i].p, qty: it.qty }));
      const pr = priceSelection(lines, catalog, now);
      const cand = { picks: pick.map(c => c.p.id), score: sc, cost: pr.totalCents ?? pr.subtotalCents, pr };
      if (pr.totalCents !== null && pr.totalCents <= B && !pr.stale) { if (better(cand, best)) best = cand; }
      else if (pr.subtotalCents <= B && (pr.totalCents === null || pr.stale) && better(cand, bestUnconfirmed)) bestUnconfirmed = cand;
      return;
    }
    if (best && sc + maxScoreRest[k] < best.score) return;
    const i = order[k], it = items[i];
    for (const c of lists[i].use) {
      const next = cost + c.p.priceCents * it.qty;
      if (next + minRest[k + 1] > B) continue;
      pick[i] = c; visit(k + 1, next, sc + c.s);
    }
  };
  visit(0, 0, 0);

  const chosen = best || bestUnconfirmed;
  if (!chosen) {
    return { status: exhausted ? 'no-solution' : 'incomplete', exhausted, visited, diagnosis: diagnose(state, catalog, lists, []) };
  }
  const lines = items.map((it, i) => ({ itemId: it.id, productId: chosen.picks[i], qty: it.qty, unitCents: catalog.byId.get(chosen.picks[i]).priceCents, lineCents: catalog.byId.get(chosen.picks[i]).priceCents * it.qty }));
  const status = best ? (exhausted ? 'feasible' : 'partial-feasible') : 'unconfirmed';
  const result = { status, exhausted, visited, budgetCents: B, lines, ...chosen.pr, withinBudget: !!best, alternatives: {} };
  result.alternatives = alternativesFor(state, catalog, lists, chosen.picks, now);
  result.upgrade = upgradeFor(state, catalog, lists, chosen.picks, now);
  return result;
}

// For each item: swaps that keep the rest fixed. Each carries its own priced total, so nothing is promised that wasn't checked.
function alternativesFor(state, catalog, lists, picks, now) {
  const out = {};
  state.items.forEach((it, i) => {
    if (it.lock) { out[it.id] = []; return; }
    const rows = [];
    for (const c of lists[i].all) {
      if (c.p.id === picks[i]) continue;
      const lines = state.items.map((x, j) => ({ itemId: x.id, product: catalog.byId.get(j === i ? c.p.id : picks[j]), qty: x.qty }));
      const pr = priceSelection(lines, catalog, now);
      const within = pr.totalCents !== null && !pr.stale && pr.totalCents <= state.budgetCents;
      const knownWithin = pr.subtotalCents <= state.budgetCents;
      if (!knownWithin) continue;
      rows.push({ productId: c.p.id, score: c.s, totalCents: pr.totalCents, subtotalCents: pr.subtotalCents, withinBudget: within });
      if (rows.length >= 12) break;
    }
    out[it.id] = rows;
  });
  return out;
}

// The best-scored option per item that does NOT fit, so the user can choose to raise the budget knowingly.
function upgradeFor(state, catalog, lists, picks, now) {
  const out = [];
  state.items.forEach((it, i) => {
    if (it.lock) return;
    const cur = lists[i].all.find(c => c.p.id === picks[i]);
    const top = lists[i].all.find(c => c.s > (cur?.s ?? 0) && c.p.id !== picks[i]);
    if (!top) return;
    const lines = state.items.map((x, j) => ({ itemId: x.id, product: catalog.byId.get(j === i ? top.p.id : picks[j]), qty: x.qty }));
    const pr = priceSelection(lines, catalog, now);
    if (pr.subtotalCents > state.budgetCents) out.push({ itemId: it.id, productId: top.p.id, overByCents: (pr.totalCents ?? pr.subtotalCents) - state.budgetCents, totalKnown: pr.totalCents !== null });
  });
  return out;
}

function diagnose(state, catalog, lists, emptyIds) {
  const d = { emptyItems: [], minTotalCents: null, budgetCents: state.budgetCents, cheapest: [] };
  for (const id of emptyIds) {
    const it = state.items.find(x => x.id === id);
    const base = catalog.products.filter(p => p.category === it.category && available(p));
    const blockers = [];
    for (const h of it.hard || []) {
      const res = base.map(p => checkHard(p, h));
      blockers.push({ label: h.label, yes: res.filter(r => r === 'yes').length, unknown: res.filter(r => r === 'unknown').length, no: res.filter(r => r === 'no').length });
    }
    d.emptyItems.push({ itemId: id, category: it.category, inCategory: base.length, blockers, keywords: it.keywords || [], unitCapCents: it.unitCapCents || null });
  }
  if (!emptyIds.length) {
    let min = 0;
    state.items.forEach((it, i) => {
      const c = [...lists[i].all].sort((a, b) => a.p.priceCents - b.p.priceCents)[0];
      d.cheapest.push({ itemId: it.id, productId: c.p.id, unitCents: c.p.priceCents, qty: it.qty });
      min += c.p.priceCents * it.qty;
    });
    d.minTotalCents = min; // price-only lower bound; fees can only add
    // cheaper items left out only because a required spec is not stated
    state.items.forEach(it => {
      for (const h of it.hard || []) {
        const n = catalog.products.filter(p => p.category === it.category && available(p) && p.priceCents * it.qty <= state.budgetCents && checkHard(p, h) === 'unknown').length;
        if (n) d.unknownHard = [...(d.unknownHard || []), { itemId: it.id, category: it.category, label: h.label, count: n }];
      }
    });
  }
  return d;
}
