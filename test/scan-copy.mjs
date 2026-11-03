// Words that must never reach a user: internal/process vocabulary, not product copy.
// Scans every file under public/, every user-facing string literal in lib/ + server.mjs, and live replies for all benchmark turns.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';
import { loadCatalog } from '../lib/catalog.mjs';
import { turn, emptySession, plainText } from '../lib/assistant.mjs';
import { BANNED } from './banned.mjs';

const root = new URL('..', import.meta.url).pathname;
const hits = [];
const check = (where, text) => { for (const w of BANNED) if (text.includes(w)) hits.push(`${where}: “${w}” … ${text.slice(Math.max(0, text.indexOf(w) - 20), text.indexOf(w) + 20).replace(/\s+/g, ' ')}`); };

function walk(dir) { return readdirSync(dir).flatMap(f => { const p = join(dir, f); return statSync(p).isDirectory() ? walk(p) : [p]; }); }
// public assets: everything except code identifiers (strip JS comments, then check string literals + markup text)
for (const f of walk(join(root, 'public'))) {
  if (!['.html', '.js', '.css', '.svg', '.json'].includes(extname(f))) continue;
  let src = readFileSync(f, 'utf8');
  if (extname(f) === '.js' || extname(f) === '.css') src = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/.*$/gm, '$1');
  if (extname(f) === '.js') { for (const m of src.matchAll(/(['"`])((?:\\.|(?!\1)[^\\])*)\1/g)) check(f.replace(root, ''), m[2]); }
  else check(f.replace(root, ''), src);
}
// server-side user-facing strings: only string literals that contain CJK (identifiers and keys are not shown to users)
// agent.mjs holds model instructions and banned.mjs the word list itself: neither is shown to users
for (const f of [join(root, 'server.mjs'), ...walk(join(root, 'lib')).filter(f => !/(agent|banned)\.mjs$/.test(f))]) {
  const src = readFileSync(f, 'utf8').replace(/\/\/.*$/gm, '');
  for (const m of src.matchAll(/(['"`])((?:\\.|(?!\1)[^\\])*)\1/g)) if (/[一-龥]/.test(m[2])) check(f.replace(root, ''), m[2]);
}
// live replies for every benchmark turn
const catalog = loadCatalog(join(root, 'data'));
const bench = JSON.parse(readFileSync(join(root, 'bench/badcase-benchmark.json'), 'utf8'));
const NOW = Date.parse(catalog.generatedAt) + 3600_000;
for (const c of bench.cases) { let s = emptySession(); for (const t of c.turns) { const r = turn(s, t, catalog, NOW); s = r.session; check(`reply ${c.id}`, plainText(r.reply)); } }
// catalog text that is rendered (titles, highlights, specs)
for (const p of catalog.products) check(`product ${p.id}`, [p.title, ...(p.highlights || []), ...Object.values(p.specs || {})].join(' '));

// the model key must only live in the environment, never in a file that ships
const key = process.env.AGENT_API_KEY || process.env.GLM_API_KEY;
if (key) for (const f of [join(root, 'server.mjs'), join(root, 'vercel.json'), ...walk(join(root, 'lib')), ...walk(join(root, 'public')), ...walk(join(root, 'data')), ...walk(join(root, 'docs'))]) if (readFileSync(f, 'utf8').includes(key)) hits.push(`${f.replace(root, '')}: 含有接口密钥`);
if (hits.length) { console.log(hits.join('\n')); console.log(`\n✗ ${hits.length} 处违禁用词`); process.exit(1); }
console.log('✓ 违禁词扫描通过（public/、服务端文案、全部基准回复、商品文本）');
