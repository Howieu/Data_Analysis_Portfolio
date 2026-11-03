import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildCatalog } from '../lib/catalog.mjs';
import { moneyToken, cnInt } from '../lib/money.mjs';
import { turn, emptySession, plainText, refresh } from '../lib/assistant.mjs';

const catalog = buildCatalog(JSON.parse(readFileSync(new URL('./fixture-catalog.json', import.meta.url))));
const NOW = Date.parse('2026-11-02T01:00:00Z');
const run = (turns, opts) => { let s = emptySession(), r; for (const t of turns) ({ session: s, reply: r } = turn(s, t, catalog, NOW, opts)); return { s, r, text: plainText(r) }; };

test('money parsing is exact and strict', () => {
  assert.equal(cnInt('八百'), 800); assert.equal(cnInt('一千二'), 1200); assert.equal(cnInt('两万五'), 25000); assert.equal(cnInt('十五'), 15);
  assert.equal(moneyToken('1k').cents, 100000); assert.equal(moneyToken('1,299').cents, 129900); assert.equal(moneyToken('1.5k').cents, 150000);
  assert.ok(moneyToken('1.234').error); assert.ok(moneyToken('-100').error); assert.ok(moneyToken('0').error);
});

test('known store rule: within budget, integer total', () => {
  const { r, s } = run(['降噪耳机，预算500']);
  assert.equal(r.status, 'recommend'); assert.equal(s.state.budgetCents, 50000);
  assert.equal(r.summary.withinBudget, true); assert.equal(r.summary.totalCents, 19900); // 满99包邮
  assert.match(plainText(r), /在预算内/);
});

test('unknown shipping is never filled with 0 and never claims within budget', () => {
  const { r } = run(['小风扇，预算100']);
  assert.equal(r.status, 'recommend'); assert.equal(r.summary.withinBudget, true);
  // f1 (乙, unknown fees) vs f2 (甲, 59 < 99 → fee 10 → 69 known). Confirmed option exists → prefer it.
  assert.equal(r.summary.lines[0].productId, 'f2'); assert.equal(r.summary.totalCents, 6900);
});

test('only unknown-fee option → pending wording, no 预算内', () => {
  const { r, text } = run(['小风扇，4000mAh以上，预算100']);
  assert.equal(r.summary.lines[0].productId, 'f1'); assert.equal(r.summary.withinBudget, false); assert.equal(r.summary.totalCents, null);
  assert.match(text, /运费以平台为准/); assert.doesNotMatch(text, /预算内/);
});

test('stale price is never confirmed', () => {
  const { r } = run(['键盘，预算100']);
  assert.equal(r.summary.lines[0].productId, 'k3'); assert.equal(r.summary.withinBudget, false); assert.match(plainText(r), /价格需要刷新/);
});

test('hard condition: unknown is not satisfied', () => {
  const { r } = run(['无线耳机，预算150']);
  assert.equal(r.status, 'no_solution'); // only e3 fits the price and its wireless fact is unknown
  assert.match(plainText(r), /没写这项参数/);
});

test('bundle, lock survives budget change, swap touches one item', () => {
  let { s, r } = run(['耳机和键盘，总预算800元']);
  assert.equal(r.summary.lines.length, 2);
  const ear = s.state.items.find(i => i.category === 'earbuds').selected;
  ({ session: s, reply: r } = turn(s, '锁定耳机', catalog, NOW));
  ({ session: s, reply: r } = turn(s, '预算改成600', catalog, NOW));
  assert.equal(s.state.items.find(i => i.category === 'earbuds').selected, ear);
  const kb = s.state.items.find(i => i.category === 'keyboard').selected;
  ({ session: s, reply: r } = turn(s, '换一个键盘', catalog, NOW));
  assert.equal(s.state.items.find(i => i.category === 'earbuds').selected, ear);
  assert.notEqual(s.state.items.find(i => i.category === 'keyboard').selected, kb);
});

test('quantity edit keeps the other item', () => {
  let { s } = run(['耳机和键盘，总预算1000元', '键盘改成两把']);
  assert.deepEqual(s.state.items.map(i => [i.category, i.qty]).sort(), [['earbuds', 1], ['keyboard', 2]]);
});

test('view never edits list or budget', () => {
  const { s, r } = run(['耳机和键盘，总预算800元', '按单品看']);
  assert.equal(s.state.items.length, 2); assert.equal(s.state.budgetCents, 80000); assert.equal(r.view.mode, 'single');
});

test('no solution keeps budget and locks', () => {
  const { s, r, text } = run(['耳机和键盘，总预算800元', '锁定耳机', '预算改成100']);
  assert.equal(r.status, 'no_solution'); assert.equal(s.state.budgetCents, 10000);
  assert.ok(s.state.items.find(i => i.category === 'earbuds').lock); assert.doesNotMatch(text, /预算内/);
});

test('category without guide gets no invented tips', () => {
  const { r } = run(['收纳盒，预算50']);
  assert.equal(r.status, 'recommend'); assert.ok(!r.blocks.some(b => b.type === 'tips'));
});

test('truncated search is reported, not called optimal', () => {
  const { r } = run(['耳机和键盘，总预算800元'], { maxNodes: 2 });
  assert.ok(['no_solution', 'recommend'].includes(r.status));
  assert.match(plainText(r), /没比完/);
});

test('refresh after expiry revokes within-budget but keeps items', () => {
  const { s } = run(['降噪耳机，预算500']);
  const later = refresh(s, catalog, NOW + 30 * 86400_000);
  assert.equal(later.session.result.withinBudget, false); assert.equal(later.session.state.items.length, 1);
});

test('bad input keeps previous state', () => {
  const { s, r } = run(['耳机，预算500', '预算-100']);
  assert.equal(r.status, 'clarify'); assert.equal(s.state.budgetCents, 50000);
});
