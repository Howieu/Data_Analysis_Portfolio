import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

// A listed price older than this is shown as reference only and never confirms a budget.
export const PRICE_TTL_MS = Number(process.env.PRICE_TTL_HOURS || 24 * 14) * 3600_000;

export function loadCatalog(dir) {
  const file = join(dir, 'catalog.json');
  const raw = JSON.parse(readFileSync(file, 'utf8'));
  const overrides = existsSync(join(dir, 'catalog.local.json')) ? JSON.parse(readFileSync(join(dir, 'catalog.local.json'), 'utf8')) : {};
  return buildCatalog(raw, overrides);
}

// display names for merged categories until the catalog carries its own shortName
const SHORT = { 'portable-fan': '小风扇', 'desk-fan': '电风扇', earphones: '耳机', 'keyboard-mouse': '键盘鼠标', charging: '充电宝', 'electric-toothbrush': '电动牙刷', 'hair-dryer': '吹风机', 'desk-lamp': '台灯', luggage: '行李箱', smartwatch: '手表手环', 'floor-cleaning': '清洁电器', 'kitchen-appliance': '小厨电', phone: '手机', tablet: '平板' };

export function buildCatalog(raw, overrides = {}) {
  const sources = new Map((raw.sources || []).map(s => [s.name, s]));
  const categories = (raw.categories || []).map(c => ({ ...c, shortName: c.shortName || SHORT[c.id] || c.name, aliases: [...new Set([c.name, ...(c.aliases || [])])] }));
  const catById = new Map(categories.map(c => [c.id, c]));
  const products = (raw.products || [])
    .filter(p => catById.has(p.category) && Number.isSafeInteger(p.priceCents) && p.priceCents > 0)
    .map(p => ({ ...p, ...(overrides[p.id] || {}), facts: p.facts || {}, specs: p.specs || {}, images: p.images || [] }));
  const byId = new Map(products.map(p => [p.id, p]));
  const count = new Map();
  for (const p of products) count.set(p.category, (count.get(p.category) || 0) + 1);
  return {
    generatedAt: raw.generatedAt, sources, categories: categories.filter(c => count.get(c.id)), catById, products, byId,
    countOf: id => count.get(id) || 0,
  };
}

export function isFresh(p, now = Date.now()) {
  const t = Date.parse(p.observedAt);
  return Number.isFinite(t) && now - t <= PRICE_TTL_MS && now - t >= -3600_000;
}

export function available(p) { return p.stock !== 'out_of_stock' && p.valid !== false; }

// Store-level shipping for a set of {product, qty}. Returns {cents|null, perSource:[{source, subtotal, shipping|null}]}.
export function shippingFor(lines, catalog) {
  const groups = new Map();
  for (const l of lines) {
    const g = groups.get(l.product.source) || { source: l.product.source, subtotal: 0, lines: [] };
    g.subtotal += l.product.priceCents * l.qty; g.lines.push(l); groups.set(l.product.source, g);
  }
  let total = 0, unknown = false;
  const perSource = [];
  for (const g of groups.values()) {
    let ship = null;
    const itemShip = g.lines.map(l => l.product.shippingCents);
    if (itemShip.every(s => s === 0)) ship = 0;
    else {
      const pol = catalog.sources.get(g.source)?.shippingPolicy;
      if (pol && Number.isSafeInteger(pol.freeOverCents) && g.subtotal >= pol.freeOverCents) ship = 0;
      else if (pol && Number.isSafeInteger(pol.feeCents) && Number.isSafeInteger(pol.freeOverCents)) ship = pol.feeCents;
      // No published store rule: charge each listed fee per unit. Conservative (never under-counts), so a "within budget" claim stays true.
      else if (itemShip.every(s => Number.isSafeInteger(s))) ship = g.lines.reduce((a, l) => a + l.product.shippingCents * l.qty, 0);
    }
    if (ship === null) unknown = true; else total += ship;
    perSource.push({ source: g.source, subtotal: g.subtotal, shipping: ship });
  }
  return { cents: unknown ? null : total, perSource };
}
