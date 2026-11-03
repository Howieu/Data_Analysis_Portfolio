// node scripts/sku/validate.mjs [path/to/catalog.json] — schema + sanity checks, no deps. Exit 1 on any error.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const path = process.argv[2] ?? fileURLToPath(new URL('../../data/catalog.json', import.meta.url));
const c = JSON.parse(readFileSync(path, 'utf8'));
const errs = [];
const err = (id, m) => errs.push(`${id}: ${m}`);
const isInt = (v) => Number.isInteger(v);
const intOrNull = (v) => v === null || isInt(v);
const str = (v) => typeof v === 'string' && v.trim() !== '';
const url = (v) => { try { const u = new URL(v, 'http://local'); return v.startsWith('/') || /^https?:$/.test(u.protocol); } catch { return false; } };
const PROMO = /直降|立减|券后|历史低价|到手价|划线价|领券|限时/;

if (isNaN(Date.parse(c.generatedAt))) err('root', 'generatedAt not ISO');
const srcNames = new Set();
for (const s of c.sources ?? []) {
  if (!str(s.name) || !str(s.homepage) || isNaN(Date.parse(s.fetchedAt))) err('source', `bad source ${JSON.stringify(s)}`);
  srcNames.add(s.name);
  const sp = s.shippingPolicy;
  if (sp !== null && sp !== undefined) {
    if (!intOrNull(sp.freeOverCents) || !intOrNull(sp.feeCents) || !str(sp.evidenceUrl) || !str(sp.evidenceText)) err(s.name, 'bad shippingPolicy');
  } else if (sp === undefined) err(s.name, 'shippingPolicy missing (use null)');
}
const cats = new Set();
for (const k of c.categories ?? []) {
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(k.id) || !str(k.name) || !Array.isArray(k.aliases) || !Array.isArray(k.specKeys)) err('category', JSON.stringify(k));
  cats.add(k.id);
}
const ids = new Set();
for (const p of c.products ?? []) {
  const id = p.id ?? '?';
  if (!str(p.id)) err(id, 'missing id');
  if (ids.has(p.id)) err(id, 'duplicate id');
  ids.add(p.id);
  if (!str(p.source)) err(id, 'empty source');
  else if (!srcNames.has(p.source)) err(id, `source not in sources[]: ${p.source}`);
  for (const f of ['sourceSku', 'url', 'title', 'brand']) if (!str(p[f])) err(id, `missing ${f}`);
  if (!url(p.url)) err(id, 'bad url');
  if (!cats.has(p.category)) err(id, `unknown category ${p.category}`);
  if (!isInt(p.priceCents) || p.priceCents <= 0) err(id, `priceCents must be positive int, got ${p.priceCents}`);
  if (!intOrNull(p.listPriceCents)) err(id, 'listPriceCents not int|null');
  if (!intOrNull(p.shippingCents)) err(id, 'shippingCents not int|null');
  if (!('stock' in p) || ![null, 'in_stock', 'out_of_stock'].includes(p.stock)) err(id, `bad stock ${p.stock}`);
  if (!Array.isArray(p.images) || !p.images.every(url)) err(id, 'images must be URL array');
  if (typeof p.specs !== 'object' || Array.isArray(p.specs) || !Object.values(p.specs).every((v) => typeof v === 'string')) err(id, 'specs must be {string:string}');
  if (typeof p.facts !== 'object' || Array.isArray(p.facts)) err(id, 'facts must be object');
  if (!Array.isArray(p.highlights)) err(id, 'highlights must be array');
  for (const t of [p.title, ...(p.highlights ?? [])]) if (PROMO.test(t)) err(id, `promo text: ${t}`);
  if (isNaN(Date.parse(p.observedAt))) err(id, 'observedAt not ISO');
}

const count = (key) => Object.entries((c.products ?? []).reduce((a, p) => ((a[p[key]] = (a[p[key]] ?? 0) + 1), a), {})).sort((a, b) => b[1] - a[1]);
console.log(`products: ${c.products?.length ?? 0}, categories: ${cats.size}, sources: ${srcNames.size}`);
console.log('by category:', Object.fromEntries(count('category')));
console.log('by source:', Object.fromEntries(count('source')));
console.log('missing images:', (c.products ?? []).filter((p) => !p.images?.length).length);
for (const k of cats) if (!c.products.some((p) => p.category === k)) console.log(`WARN empty category: ${k}`);
if (errs.length) { console.error(`${errs.length} errors:\n` + errs.slice(0, 50).join('\n')); process.exit(1); }
console.log('OK');
