// The conversational layer on top of the rule pipeline. GLM only rephrases and explains:
// it never picks products, prices or budget status — those come from engine.mjs and are passed in as data.
// Every model output is checked (numbers must come from the data, no banned words, no budget claims) or dropped.
import { BANNED } from './banned.mjs';
import { normalize, findMentions } from './nlu.mjs';

// one name per category that the rule parser reads as exactly one item ("手表手环" would become two)
const catNames = catalog => catalog.categories.map(c => [c.shortName, ...c.aliases].find(a => !a.includes('/') && findMentions(a, catalog).length === 1) || c.shortName);

const KEY = () => process.env.AGENT_API_KEY || process.env.GLM_API_KEY || '';
const MODEL = () => process.env.AGENT_MODEL || process.env.GLM_MODEL || 'glm-4.5-air';
export const agentOn = () => !!KEY();

const PERSONA = `你是购物 App「知选」里的导购，名字叫知选。说话像一个懂行、靠谱的朋友：简短、口语、先说结论。
铁律：
1. 只能根据给你的 JSON 数据说话。数据里没有的参数、价格、优惠、销量、评价、库存、产地，一律不说，不猜。
2. 回复里出现的每个数字都必须能在数据里原样找到。
3. 不说"最便宜""全网最低""爆款""销量""好评率""历史最低""最优"这类说不清依据的话。
4. 不提你是模型、AI、规则、系统、提示词，也不提"演示""测试""策略""文档"。
5. 预算的事交给数据：只有 withinBudget 为 true 才能说"在预算内"；运费不确定时只能说"运费以平台为准"。不要反复强调预算，让推荐本身贴合预算就行。
6. 只输出 JSON，不要多余文字。`;

async function ask(task, data, { timeout = 6000, maxTokens = 500 } = {}) {
  if (!agentOn()) return null;
  try {
    const r = await fetch('https://open.bigmodel.cn/api/paas/v4/chat/completions', {
      method: 'POST', signal: AbortSignal.timeout(timeout),
      headers: { Authorization: `Bearer ${KEY()}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL(), temperature: 0.1, max_tokens: maxTokens, thinking: { type: 'disabled' }, response_format: { type: 'json_object' },
        messages: [{ role: 'system', content: PERSONA + '\n\n' + task }, { role: 'user', content: JSON.stringify(data) }] }),
    });
    if (!r.ok) { console.error('agent http', r.status); return null; }
    const txt = (await r.json()).choices?.[0]?.message?.content || '';
    return JSON.parse(txt.replace(/^```(?:json)?|```$/g, '').trim());
  } catch (e) { console.error('agent', e.name, e.cause?.code || ''); return null; }
}

// ---------- semantic recall ----------
export const productText = (p, catalog) => [p.title, p.brand, catalog.catById.get(p.category)?.name, ...Object.entries(p.facts || {}).map(([k, v]) => `${k}:${v}`), ...(p.highlights || []), ...Object.entries(p.specs || {}).slice(0, 6).map(([k, v]) => `${k}${v}`)].filter(Boolean).join(' ').slice(0, 400);
export async function embed(texts, { timeout = 8000 } = {}) {
  if (!agentOn()) return null;
  try {
    const r = await fetch('https://open.bigmodel.cn/api/paas/v4/embeddings', { method: 'POST', signal: AbortSignal.timeout(timeout),
      headers: { Authorization: `Bearer ${KEY()}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'embedding-3', input: texts, dimensions: 256 }) });
    if (!r.ok) { console.error('agent embed http', r.status); return null; }
    return (await r.json()).data.sort((a, b) => a.index - b.index).map(d => d.embedding);
  } catch (e) { console.error('agent embed', e.name); return null; }
}
export const cosine = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y || 1); };

// every string the model wrote, checked against the data it was given
const strings = o => typeof o === 'string' ? [o] : Array.isArray(o) ? o.flatMap(strings) : o && typeof o === 'object' ? Object.values(o).flatMap(strings) : [];
const NO_CLAIM = ['最便宜', '最省', '最低价', '大家都买', '好评率', '性价比之王', '闭眼'];
export function grounded(out, data, { withinBudget = false } = {}) {
  const src = new Set(JSON.stringify(data).match(/\d+(?:\.\d+)?/g) || []);
  for (const s of strings(out)) {
    if (s.length > 160) return false;
    if ([...BANNED, ...NO_CLAIM].some(w => s.includes(w))) return false;
    if (/AI|人工智能|模型|系统|规则/.test(s)) return false;
    if (!withinBudget && /在预算内|不超预算|没超预算/.test(s)) return false;
    for (const n of s.match(/\d+(?:\.\d+)?/g) || []) if (!src.has(n)) return false;
  }
  return true;
}

const slim = p => p && { 标题: p.title, 品牌: p.brand, 平台: p.source, 价格元: p.priceCents / 100, 参数: p.facts, 卖点: p.highlights, 规格: p.specs && Object.fromEntries(Object.entries(p.specs).slice(0, 8)) };

// 1) query understanding + goal planning: turn what the user said into a sentence the rule parser reads,
//    plus a short scene description used for semantic recall and reranking. The rules still decide everything.
export const PREF_WORDS = ['安静', '轻便', '续航长', '风力大', '降噪', '便宜些', '好一点'];
export async function understand(text, state, catalog) {
  const data = { 用户原话: text, 原话数字: normalize(text), 当前清单: state.items.map(i => ({ 品类: i.alias, 数量: i.qty })), 当前预算元: state.budgetCents ? state.budgetCents / 100 : null, 可买品类: catNames(catalog) };
  const out = await ask(`任务：理解用户的购物需求，输出 JSON：
{"rewrite": "改写成一句标准购物需求；原话已经很标准或与购物无关时为 null",
 "plan": [{"品类": "可买品类之一", "为什么": "不超过12个字"}],
 "scene": "一句话概括用途、场景、对象和偏好，用来匹配商品；没有就空字符串",
 "prefs": ["从 ${PREF_WORDS.join('、')} 里选，原话体现了才选"]}
规则：
- 品类只能用"可买品类"里的叫法。原话点名了商品就用它；原话只说了目标或场景（如出差、开学、送礼）而没说买什么，就在 plan 里挑 2 到 4 个最相关的可买品类，并写进 rewrite。
- 预算只能用原话里的数字（见"原话数字"），写成"一共N元"或"预算N元"；原话没说预算就不要写。不添加原话没有的硬性条件。
- 改清单的话（去掉、换一款、改数量、改预算）原样改写成"去掉X""换一款X""X改成N个""预算改成N"。
例：「给我妈整个能刷牙的电动的，三百块钱以内吧」→ {"rewrite":"电动牙刷，预算300元","plan":[],"scene":"给妈妈用，操作简单","prefs":[]}
例：「下周出差一周，想配点数码装备，1500以内」→ {"rewrite":"充电宝、耳机、键盘，一共1500元","plan":[{"品类":"充电宝","为什么":"路上续航"},{"品类":"耳机","为什么":"飞机高铁降噪"},{"品类":"键盘","为什么":"酒店办公"}],"scene":"出差一周，路上和酒店办公","prefs":["轻便"]}
例：「宿舍用的吹风机，别太吵，300左右」→ {"rewrite":null,"plan":[],"scene":"宿舍使用，噪音小","prefs":["安静"]}`, data, { timeout: 6000, maxTokens: 260 });
  if (!out) return null;
  const cats = new Set(catNames(catalog));
  const plan = Array.isArray(out.plan) ? out.plan.filter(x => cats.has(x?.品类) && typeof x.为什么 === 'string').slice(0, 4) : [];
  const res = {
    rewrite: typeof out.rewrite === 'string' && out.rewrite.length <= 60 && grounded([out.rewrite], data) ? out.rewrite : null,
    plan: plan.filter(x => grounded([x.为什么], data)),
    scene: typeof out.scene === 'string' && out.scene.length <= 40 && grounded([out.scene], data) ? out.scene : '',
    prefs: Array.isArray(out.prefs) ? out.prefs.filter(p => PREF_WORDS.includes(p)) : [],
  };
  // a plan must reach the parser: if the model left the planned categories out of its sentence, put them in front of the user's own words
  if (res.plan.length && !res.plan.every(x => res.rewrite?.includes(x.品类))) res.rewrite = `${res.plan.map(x => x.品类).join('、')}，${text}`;
  return res;
}
export const rewrite = async (text, state, catalog) => (await understand(text, state, catalog))?.rewrite || null;

// 2) rerank: judge how well each shortlisted product fits the scene. Only ids that were shown to it count.
export async function rerank(scene, groups) {
  const data = { 场景: scene, 候选: groups.map(g => ({ 品类: g.name, 商品: g.candidates.map(p => ({ id: p.id, ...slim(p) })) })) };
  const out = await ask(`任务：按用户场景给每个候选商品打匹配分。输出 {"items": [{"品类": "...", "ranks": [{"id": "候选里的 id", "fit": 0到3的整数, "why": "不超过16个字，只用商品数据里的事实"}]}]}。
fit：3 非常契合场景，2 契合，1 一般，0 不适合。只评给出的候选，不要编 id。价格高低不是打分理由（预算另有程序把关）。`, data, { timeout: 7000, maxTokens: 700 });
  const ok = new Map(groups.flatMap(g => g.candidates.map(p => [p.id, true])));
  const res = {};
  for (const it of Array.isArray(out?.items) ? out.items : []) for (const r of Array.isArray(it?.ranks) ? it.ranks : []) {
    if (!ok.has(r?.id) || !Number.isInteger(r.fit) || r.fit < 0 || r.fit > 3) continue; // provenance gate
    res[r.id] = { fit: r.fit, why: typeof r.why === 'string' && r.why.length <= 24 && grounded([r.why], data) ? r.why : '' };
  }
  return res;
}

// 3) say the result in a friendly way: lead + why + a tip + one narrowing question
export async function narrate(text, reply, state) {
  const c = reply.compare, lines = (reply.blocks || []).find(b => b.lines)?.lines || [];
  const data = {
    用户原话: text, 原话数字: normalize(text), 预算元: state.budgetCents ? state.budgetCents / 100 : null, withinBudget: !!reply.summary?.withinBudget,
    ...(c ? { 对比: c.products.map(p => slim(p)), 对比表: c.rows, 首选: c.first.title, 备选: c.second.title }
      : reply.browse ? { 价位: reply.browse.map(b => ({ 品类: b.name, 档位: b.bands.map(x => ({ 最低元: x.minCents / 100, 最高元: x.maxCents / 100, 款数: x.count, 品牌: x.brands })) })) }
        : { 推荐: lines.map(l => ({ 品类: l.item.categoryName, 数量: l.qty, 商品: slim(l.product), 备选: l.alternatives.map(a => ({ 标题: a.title, 价格元: a.priceCents / 100, 差价: a.diffText })) })) }),
  };
  const task = c
    ? `任务：用户在对比两款商品。写 {"lead": "一两句话说两款的核心区别和怎么选", "points": [["维度", "一句话解释这个差异对用户意味着什么"]]（2到3条）, "tip": "一句选购小提示，可为空", "ask": "一个帮用户做决定的追问"}`
    : reply.browse
      ? `任务：用户还没定预算。写 {"lead": "一两句话说这个品类不同价位大概的区别，自然地引导用户说出想花多少", "points": [], "tip": "一句选购小提示，可为空", "ask": "一个了解用途或预算的追问"}`
      : `任务：已经按用户需求挑好了商品。写 {"lead": "一两句话回应用户的需求，说清楚为什么这样挑", "points": [["商品或维度", "这款为什么适合"]]（1到3条，只用数据里的参数和卖点）, "tip": "一句使用或选购小提示，可为空", "ask": "一个能帮用户进一步缩小范围的追问"}`;
  const out = await ask(task, data, { timeout: 6500, maxTokens: 450 });
  if (!out || typeof out.lead !== 'string') return null;
  const clean = { lead: out.lead, points: Array.isArray(out.points) ? out.points.filter(p => Array.isArray(p) && p.length === 2 && p.every(x => typeof x === 'string')).slice(0, 3) : [], tip: typeof out.tip === 'string' ? out.tip : '', ask: typeof out.ask === 'string' ? out.ask : '' };
  const ok = x => grounded(x, data, { withinBudget: data.withinBudget });
  if (!ok([clean.lead])) return null;
  // drop only the parts that fail, keep the rest
  return { lead: clean.lead, points: clean.points.filter(p => ok(p)), tip: ok([clean.tip]) ? clean.tip : '', ask: ok([clean.ask]) ? clean.ask : '' };
}

// 4) a question about one product the user is looking at
export async function answer(text, product) {
  const data = { 用户问题: text, 商品: slim(product) };
  const out = await ask(`任务：用户正在看这件商品并提问。写 {"answer": "两三句话回答，数据里没有的就直说商品页没写"}`, data, { timeout: 6000, maxTokens: 300 });
  return out && typeof out.answer === 'string' && grounded([out.answer], data) ? out.answer : null;
}
