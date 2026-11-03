// npm run bench [-- --shift-days N] [-- --only id1,id2] [-- --verbose]
import { readFileSync, writeFileSync } from 'node:fs';
import { loadCatalog } from '../lib/catalog.mjs';
import { turn, emptySession, plainText } from '../lib/assistant.mjs';

const HARD_GROUPS = ['budget', 'qty', 'lock', 'view', 'nosol', 'price', 'inject', 'copy', 'edit', 'hard', 'multi', 'extreme']; // 硬性分组，共 12 组
const args = process.argv.slice(2);
const opt = k => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const shiftDays = Number(opt('--shift-days') || 0);
const only = opt('--only')?.split(',');
const verbose = args.includes('--verbose');

const root = new URL('..', import.meta.url).pathname;
const catalog = loadCatalog(root + 'data');
const bench = JSON.parse(readFileSync(root + 'bench/badcase-benchmark.json', 'utf8'));
const waivers = (() => { try { return JSON.parse(readFileSync(root + 'bench/waivers.json', 'utf8')); } catch { return {}; } })();
const NOW = Date.parse(catalog.generatedAt) + 3600_000 + shiftDays * 86400_000;

const catMatch = (want, catId) => {
  if (want === null) return true;
  const c = catalog.catById.get(catId); if (!c) return false;
  return c.name.includes(want) || want.includes(c.name) || c.aliases.some(a => a === want || a.includes(want) || want.includes(a));
};
const snapshot = s => JSON.stringify({ items: s.state.items.map(i => ({ c: i.category, q: i.qty, h: i.hard, k: i.keywords, l: i.lock })), b: s.state.budgetCents });

function check(c) {
  let s = emptySession(), r, prev;
  const lockedAt = {};
  for (const t of c.turns) {
    prev = s;
    ({ session: s, reply: r } = turn(s, t, catalog, NOW));
    for (const i of s.state.items) if (i.lock && !lockedAt[i.id]) lockedAt[i.id] = i.lock;
  }
  const e = c.expect, fails = [], text = plainText(r);
  if (e.status && r.status !== e.status) fails.push(`status ${r.status} ≠ ${e.status}`);
  if ('budgetCents' in e && (s.state.budgetCents ?? null) !== e.budgetCents) fails.push(`budget ${s.state.budgetCents} ≠ ${e.budgetCents}`);
  if (e.itemCount !== undefined && s.state.items.length !== e.itemCount) fails.push(`itemCount ${s.state.items.length} ≠ ${e.itemCount}`);
  if (e.items) {
    const pool = [...s.state.items];
    for (const w of e.items) {
      const k = pool.findIndex(i => i.qty === w.qty && catMatch(w.category, i.category));
      if (k < 0) fails.push(`missing item ${w.category}×${w.qty}`); else pool.splice(k, 1);
    }
  }
  for (const w of e.mustNotContain || []) if (text.includes(w)) fails.push(`contains "${w}"`);
  for (const w of e.mustContain || []) if (!text.includes(w)) fails.push(`lacks "${w}"`);
  if (e.totalWithinBudget && r.status === 'recommend') {
    const res = s.result;
    if (res?.totalCents !== null && res?.totalCents !== undefined && !res.stale) { if (res.totalCents > s.state.budgetCents) fails.push(`total ${res.totalCents} > budget`); }
    else if (res?.withinBudget || /预算内|符合预算/.test(text)) fails.push('claims within budget with unknown fees');
  }
  if (e.keepsPrevious && snapshot(s) !== snapshot(prev)) fails.push('state changed on a kept turn');
  for (const name of e.locked || []) {
    const it = s.state.items.find(i => catMatch(name, i.category));
    if (!it?.lock) fails.push(`${name} not locked`);
    else if (lockedAt[it.id] && it.lock !== lockedAt[it.id]) fails.push(`${name} lock changed`);
  }
  return { fails, text, r, s };
}

const rows = [];
for (const c of bench.cases) {
  if (only && !only.includes(c.id)) continue;
  let out;
  try { out = check(c); } catch (e) { out = { fails: [`crash: ${e.message}`] }; }
  const waived = waivers[c.id];
  rows.push({ id: c.id, group: c.group, hard: HARD_GROUPS.includes(c.group), pass: !out.fails.length, waived: !!waived && out.fails.length > 0, fails: out.fails });
  if (verbose && out.fails.length) console.log(`\n✗ ${c.id} [${c.group}] ${c.turns.join(' → ')}\n  ${out.fails.join('; ')}\n  reply: ${(out.text || '').slice(0, 300).replace(/\n/g, ' | ')}`);
}
const by = {};
for (const r of rows) { const g = by[r.group] ||= { pass: 0, total: 0, hard: r.hard }; g.total++; if (r.pass || r.waived) g.pass++; }
const hardFail = rows.filter(r => r.hard && !r.pass && !r.waived);
const lines = [`知选 bad case 基准 · ${new Date(NOW).toISOString()}${shiftDays ? `（时钟后移 ${shiftDays} 天）` : ''}`, '', '| 组 | 硬规则 | 通过 |', '|---|---|---|',
  ...Object.entries(by).map(([g, v]) => `| ${g} | ${v.hard ? '是' : ''} | ${v.pass}/${v.total} |`),
  '', `总计 ${rows.filter(r => r.pass || r.waived).length}/${rows.length}，硬规则失败 ${hardFail.length}，豁免 ${rows.filter(r => r.waived).length}`,
  ...rows.filter(r => !r.pass).map(r => `- ${r.waived ? '[豁免] ' : ''}${r.id}：${r.fails.join('；')}`)];
writeFileSync(root + `bench/report${shiftDays ? `-shift${shiftDays}` : ''}.md`, lines.join('\n') + '\n');
console.log(lines.join('\n'));
process.exit(hardFail.length ? 1 : 0);
