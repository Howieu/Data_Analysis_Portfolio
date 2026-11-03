// End-to-end checks of the agent-assisted chain against a running server (needs AGENT_API_KEY on that server).
// BASE=http://127.0.0.1:8790 node scripts/agent/eval.mjs   → exit 1 on any failure
import { BANNED } from '../../lib/banned.mjs';
const BASE = process.env.BASE || 'http://127.0.0.1:8790';

const CASES = [
  { say: '下周出差一周，想配点数码装备，1500以内', budget: 150000, cats: ['charging', 'earphones'], planned: true },
  { say: '宿舍用的吹风机，别太吵，300左右', budget: 30000, cats: ['hair-dryer'] },
  { say: '送给经常跑步的男朋友，一千以内', budget: 100000, cats: ['smartwatch'], planned: true },
  { say: '给我妈整个能刷牙的电动的，三百块钱以内吧', budget: 30000, cats: ['electric-toothbrush'] },
  { say: '开学住宿舍，台灯和小风扇，一共400', budget: 40000, cats: ['desk-lamp', 'portable-fan'] },
  { say: '想在家做咖啡，预算两千', budget: 200000, cats: ['kitchen-appliance'] },
  { say: '经常坐高铁，想要安静点的耳机，八百', budget: 80000, cats: ['earphones'] },
  { say: '今天天气不错', budget: null, cats: [], status: 'clarify' },
];

const sid = async () => (await fetch(`${BASE}/api/me`)).headers.get('set-cookie')?.split(';')[0];
const texts = o => typeof o === 'string' ? [o] : Array.isArray(o) ? o.flatMap(texts) : o && typeof o === 'object' ? Object.values(o).flatMap(texts) : [];
let fail = 0;
for (const c of CASES) {
  const cookie = await sid(), t0 = Date.now();
  const r = await fetch(`${BASE}/api/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: BASE, cookie }, body: JSON.stringify({ text: c.say }) }).then(x => x.json());
  const ms = Date.now() - t0, rep = r.reply, plan = r.plan, errs = [];
  const sum = rep.summary || {};
  if (c.status && rep.status !== c.status) errs.push(`status ${rep.status}`);
  if (c.budget !== null) {
    if (plan.state.budgetCents !== c.budget) errs.push(`budget ${plan.state.budgetCents} ≠ ${c.budget}`);
    const cats = plan.state.items.map(i => i.category);
    for (const k of c.cats) if (!cats.includes(k)) errs.push(`missing ${k} (got ${cats.join(',')})`);
    if (sum.withinBudget && !(sum.totalCents <= plan.state.budgetCents)) errs.push('within but over budget');
    if (rep.status === 'recommend' && !sum.withinBudget && sum.subtotalCents > plan.state.budgetCents) errs.push('picks over budget');
    if (c.planned && !rep.ai?.plan?.length) errs.push('no plan');
  }
  const words = texts([rep.message, rep.opening, rep.agent, rep.ai, rep.notes]).join(' ');
  for (const w of BANNED) if (words.includes(w)) errs.push(`banned ${w}`);
  if (ms > 25000) errs.push(`slow ${ms}ms`);
  console.log(`${errs.length ? '✗' : '✓'} ${c.say} → ${rep.status} ${plan.state.items.map(i => i.alias).join('+') || '-'} ¥${(sum.totalCents ?? sum.subtotalCents ?? 0) / 100} · ${ms}ms${errs.length ? '\n    ' + errs.join('；') : ''}`);
  fail += errs.length ? 1 : 0;
}
console.log(`\n${CASES.length - fail}/${CASES.length} 通过`);
process.exit(fail ? 1 : 0);
