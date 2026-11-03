import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { readdirSync, statSync, readFileSync, existsSync } from 'node:fs';
import { join, extname, normalize } from 'node:path';
import { scryptSync, randomBytes, timingSafeEqual, createHash, randomUUID } from 'node:crypto';
import { openStore } from './lib/store.mjs';
import { loadCatalog } from './lib/catalog.mjs';
import { guideFor } from './lib/guide.mjs';
import * as agent from './lib/agent.mjs';
import { candidatesFor } from './lib/engine.mjs';
import { turn, applyOp, refresh, emptySession, productView, budgetState } from './lib/assistant.mjs';

const ROOT = new URL('.', import.meta.url).pathname;
const PORT = Number(process.env.PORT || 8790);
const catalog = loadCatalog(join(ROOT, 'data'));
// real user experience notes per category (collected by scripts/posts); optional
const posts = existsSync(join(ROOT, 'data/posts.json')) ? JSON.parse(readFileSync(join(ROOT, 'data/posts.json'), 'utf8')) : { posts: {} };
const store = await openStore({ dbFile: process.env.DB_FILE || join(ROOT, 'var', 'app.db') });
const ON_PLATFORM = !!process.env.VERCEL;
const DAY = 86400_000;

// ---------- helpers ----------
const json = (res, code, body, headers = {}) => { res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers }); res.end(JSON.stringify(body)); };
const err = (res, code, message) => json(res, code, { error: message });
const sha = t => createHash('sha256').update(t).digest('hex');
const hashPass = p => { const salt = randomBytes(16); return `s1$${salt.toString('hex')}$${scryptSync(p, salt, 32, { N: 16384 }).toString('hex')}`; };
const checkPass = (p, stored) => {
  const [, salt, hash] = stored.split('$');
  const got = scryptSync(p, Buffer.from(salt, 'hex'), 32, { N: 16384 });
  return timingSafeEqual(got, Buffer.from(hash, 'hex'));
};
const cookies = req => Object.fromEntries((req.headers.cookie || '').split(';').map(c => c.trim().split('=')).filter(([k]) => k).map(([k, ...v]) => [k, decodeURIComponent(v.join('='))]));
// Secure whenever the request reached us over HTTPS (directly or via the platform proxy)
const isHttps = req => process.env.COOKIE_SECURE === '1' || req.headers['x-forwarded-proto'] === 'https';
const setCookie = (req, token, maxAge) => `zx_sid=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAge}${isHttps(req) ? '; Secure' : ''}`;
const clientIp = req => (ON_PLATFORM || process.env.TRUST_PROXY === '1') ? String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() || req.socket?.remoteAddress : req.socket?.remoteAddress;

async function body(req) {
  let size = 0; const chunks = [];
  for await (const c of req) { size += c.length; if (size > 32_768) throw Object.assign(new Error('请求内容太大'), { code: 413 }); chunks.push(c); }
  if (!chunks.length) return {};
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { throw Object.assign(new Error('请求格式不正确'), { code: 400 }); }
}

// Every visitor gets a session; logging in attaches a user to a fresh token.
async function session(req, res) {
  const raw = cookies(req).zx_sid;
  if (raw) {
    const row = await store.sessionGet(sha(raw));
    if (row && Number(row.expires_at) > Date.now()) {
      const user = row.user_id ? await store.userById(row.user_id) : null;
      return { token: row.token, user, owner: user ? `u:${user.id}` : `g:${row.guest}`, guest: row.guest };
    }
  }
  const token = randomBytes(24).toString('base64url'), guest = randomUUID();
  await store.sessionCreate({ token: sha(token), user_id: null, guest, created_at: Date.now(), expires_at: Date.now() + 30 * DAY });
  res.setHeader('set-cookie', setCookie(req, token, 30 * 86400));
  return { token: sha(token), user: null, owner: `g:${guest}`, guest };
}

const attempts = new Map();
function limited(key) {
  const now = Date.now(), list = (attempts.get(key) || []).filter(t => now - t < 10 * 60_000);
  list.push(now); attempts.set(key, list);
  return list.length > 8;
}

async function login(req, res, s, user) {
  const token = randomBytes(24).toString('base64url');
  await store.sessionDelete(s.token);
  await store.sessionCreate({ token: sha(token), user_id: user.id, guest: s.guest, created_at: Date.now(), expires_at: Date.now() + 30 * DAY });
  res.setHeader('set-cookie', setCookie(req, token, 30 * 86400));
  // Guest data: favorites union; guest plans become separate plans — the account's own plans are never altered.
  const from = `g:${s.guest}`, to = `u:${user.id}`;
  await store.favMove(from, to);
  const guestActive = await store.prefGet(from), userActive = await store.prefGet(to);
  const moved = await store.convByOwner(from);
  let imported = false;
  for (const c of moved) {
    const hasItems = JSON.parse(c.session).state.items.length > 0;
    const rename = hasItems && c.id === guestActive && userActive;
    if (rename) imported = true;
    await store.convUpdate(c.id, rename ? { owner: to, title: '登录前的购物车' } : { owner: to });
  }
  if (!userActive && guestActive) await store.prefSet(to, guestActive);
  await store.prefDelete(from);
  return { importedPlan: imported };
}

const EMAIL = /^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,24}$/;

// ---------- conversations ----------
function titleOf(sess) { const n = sess.state.items.map(i => catalog.catById.get(i.category)?.name).filter(Boolean); return n.length ? n.join(' + ') : '新的对话'; }
const getConv = (owner, id) => store.convGet(owner, id);
async function activeConv(owner, create = true) {
  const id = await store.prefGet(owner);
  const c = id && await getConv(owner, id);
  if (c || !create) return c || null;
  return newConv(owner);
}
async function newConv(owner) {
  const id = randomUUID(), now = Date.now();
  await store.convCreate({ id, owner, title: '新的对话', session: JSON.stringify(emptySession()), created_at: now, updated_at: now });
  await store.prefSet(owner, id);
  return getConv(owner, id);
}
async function saveConv(c, sess) {
  await store.convUpdate(c.id, { session: JSON.stringify(sess), title: c.title === '登录前的购物车' ? c.title : titleOf(sess), updated_at: Date.now() });
}
function planView(sess) {
  const r = sess.result;
  return { state: sess.state, view: sess.view, result: r ? { ...r, budgetState: budgetState(r) } : null,
    products: Object.fromEntries((r?.lines || []).flatMap(l => [l.productId, ...Object.values(r.alternatives || {}).flat().map(a => a.productId)]).map(id => [id, productView(catalog.byId.get(id))]).filter(([, v]) => v)) };
}

// ---------- product search ----------
// Default browse order: in-stock items with pictures first, categories taken in turn so no single store or category floods the feed.
function interleave(rows) {
  const ok = x => x.p.stock !== 'out_of_stock' && x.p.images?.length ? 0 : 1;
  const byCat = new Map();
  for (const x of [...rows].sort((a, b) => ok(a) - ok(b) || a.p.priceCents - b.p.priceCents)) { if (!byCat.has(x.p.category)) byCat.set(x.p.category, []); byCat.get(x.p.category).push(x); }
  const lists = [...byCat.values()].map(l => { const mid = []; for (let i = 0; i < l.length; i += 2) mid.push(l[i]); for (let i = 1; i < l.length; i += 2) mid.push(l[i]); return mid; });
  const out = [];
  for (let i = 0; out.length < rows.length; i++) for (const l of lists) if (l[i]) out.push(l[i]);
  return out;
}
function search({ q = '', cat = '', sort = '', min = '', max = '', page = '1' }) {
  let list = catalog.products;
  if (cat) list = list.filter(p => p.category === cat);
  const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
  let scored = list.map(p => {
    if (!terms.length) return { p, s: 0 };
    const hay = `${p.title} ${p.brand} ${catalog.catById.get(p.category).aliases.join(' ')} ${(p.highlights || []).join(' ')}`.toLowerCase();
    const s = terms.reduce((a, t) => a + (hay.includes(t) ? 2 : [...t].filter(ch => hay.includes(ch)).length / t.length >= 0.75 ? 1 : 0), 0);
    return { p, s };
  });
  if (terms.length) scored = scored.filter(x => x.s >= terms.length);
  const lo = Number(min) * 100 || 0, hi = Number(max) * 100 || Infinity;
  scored = scored.filter(x => x.p.priceCents >= lo && x.p.priceCents <= hi);
  if (sort === 'price-asc') scored.sort((a, b) => a.p.priceCents - b.p.priceCents);
  else if (sort === 'price-desc') scored.sort((a, b) => b.p.priceCents - a.p.priceCents);
  else if (terms.length) scored.sort((a, b) => b.s - a.s || a.p.priceCents - b.p.priceCents);
  else scored = interleave(scored);
  const size = 20, pg = Math.max(1, Number(page) || 1);
  return { total: scored.length, page: pg, items: scored.slice((pg - 1) * size, pg * size).map(x => productView(x.p)) };
}

// ---------- routes ----------
async function api(req, res, url, s) {
  const p = url.pathname, m = req.method;
  const q = Object.fromEntries(url.searchParams);
  if (m !== 'GET' && m !== 'HEAD') {
    const origin = req.headers.origin;
    if (origin) { let host = null; try { host = new URL(origin).host; } catch {} if (host !== req.headers.host) return err(res, 403, '请求来源不受信任'); }
  }
  if (p === '/api/me' && m === 'GET') return json(res, 200, { user: s.user });
  if (p === '/api/posts' && m === 'GET') { const cat = url.searchParams.get('cat') || ''; return json(res, 200, { source: posts.source || null, items: (posts.posts[cat] || []).slice(0, 20) }); }
  if (p === '/api/meta' && m === 'GET') return json(res, 200, { categories: catalog.categories.map(c => ({ id: c.id, name: c.name, short: c.shortName, count: catalog.countOf(c.id), guide: guideFor(c.name) && (({ title, points, caution }) => ({ title, points, caution }))(guideFor(c.name)), cover: productView(catalog.products.find(x => x.category === c.id && x.images?.length && x.stock !== 'out_of_stock') || catalog.products.find(x => x.category === c.id && x.images?.length))?.image || null })), sources: [...catalog.sources.values()].map(x => x.name), updatedAt: catalog.generatedAt });
  if (p === '/api/products' && m === 'GET') return json(res, 200, search(q));
  let mm;
  if ((mm = p.match(/^\/api\/products\/([\w.-]+)$/)) && m === 'GET') {
    const prod = catalog.byId.get(mm[1]); if (!prod) return err(res, 404, '商品不存在或已下架');
    const fav = await store.favHas(s.owner, prod.id);
    const similar = catalog.products.filter(x => x.category === prod.category && x.id !== prod.id).sort((a, b) => Math.abs(a.priceCents - prod.priceCents) - Math.abs(b.priceCents - prod.priceCents)).slice(0, 6).map(x => productView(x));
    return json(res, 200, { product: { ...productView(prod), images: prod.images, specs: prod.specs, highlights: (prod.highlights || []).filter(h => h && !prod.title.includes(h) && h.length >= 4), stock: prod.stock ?? null, shippingNote: (prod.shippingNote || '').replace(/[（(][^）)]*(页面|说明|来源)[^）)]*[）)]/g, '').trim() || null, categoryName: catalog.catById.get(prod.category).shortName }, favorite: fav, similar });
  }

  if (p === '/api/auth/register' && m === 'POST') {
    const b = await body(req);
    const email = String(b.email || '').trim().toLowerCase(), pass = String(b.password || ''), name = String(b.name || '').trim().slice(0, 20);
    if (limited(`reg:${clientIp(req)}`)) return err(res, 429, '操作太频繁，请稍后再试');
    if (!EMAIL.test(email)) return err(res, 400, '邮箱格式不正确');
    if (pass.length < 8 || pass.length > 72 || !/[a-zA-Z]/.test(pass) || !/\d/.test(pass)) return err(res, 400, '密码需要 8–72 位，同时包含字母和数字');
    if (await store.userByEmail(email)) return err(res, 409, '这个邮箱已经注册过，可以直接登录');
    const user = await store.userCreate({ email, name: name || email.split('@')[0], pass: hashPass(pass), created_at: Date.now() });
    return json(res, 200, { user, ...(await login(req, res, s, user)) });
  }
  if (p === '/api/auth/login' && m === 'POST') {
    const b = await body(req);
    const email = String(b.email || '').trim().toLowerCase();
    const ip = clientIp(req);
    if (limited(`ip:${ip}`) || limited(`em:${email}:${ip}`)) return err(res, 429, '尝试次数太多，请 10 分钟后再试');
    const row = await store.userByEmail(email);
    if (!row || !checkPass(String(b.password || ''), row.pass)) return err(res, 401, '邮箱或密码不正确');
    const user = { id: row.id, email: row.email, name: row.name, created_at: row.created_at };
    return json(res, 200, { user, ...(await login(req, res, s, user)) });
  }
  if (p === '/api/auth/logout' && m === 'POST') {
    await store.sessionDelete(s.token);
    res.setHeader('set-cookie', setCookie(req, '', 0));
    return json(res, 200, { ok: true });
  }
  if (p === '/api/account/password' && m === 'POST') {
    if (!s.user) return err(res, 401, '请先登录');
    const b = await body(req), row = await store.userWithPass(s.user.id);
    if (!checkPass(String(b.current || ''), row.pass)) return err(res, 400, '当前密码不正确');
    const pass = String(b.next || '');
    if (pass.length < 8 || pass.length > 72 || !/[a-zA-Z]/.test(pass) || !/\d/.test(pass)) return err(res, 400, '新密码需要 8–72 位，同时包含字母和数字');
    await store.userSetPass(s.user.id, hashPass(pass));
    await store.sessionDeleteUserExcept(s.user.id, s.token);
    return json(res, 200, { ok: true });
  }
  if (p === '/api/account' && m === 'DELETE') {
    if (!s.user) return err(res, 401, '请先登录');
    const b = await body(req), row = await store.userWithPass(s.user.id);
    if (!checkPass(String(b.password || ''), row.pass)) return err(res, 400, '密码不正确');
    const owner = `u:${s.user.id}`;
    await store.favDeleteOwner(owner); await store.convDeleteOwner(owner); await store.prefDelete(owner); await store.userDelete(s.user.id);
    res.setHeader('set-cookie', setCookie(req, '', 0));
    return json(res, 200, { ok: true });
  }

  if (p === '/api/favorites' && m === 'GET') {
    const ids = await store.favList(s.owner);
    return json(res, 200, { items: ids.map(id => productView(catalog.byId.get(id))).filter(Boolean), missing: ids.filter(id => !catalog.byId.get(id)).length });
  }
  if ((mm = p.match(/^\/api\/favorites\/([\w.-]+)$/))) {
    if (!catalog.byId.get(mm[1])) return err(res, 404, '商品不存在');
    if (m === 'POST') await store.favAdd(s.owner, mm[1]);
    else if (m === 'DELETE') await store.favRemove(s.owner, mm[1]);
    else return err(res, 405, '不支持');
    return json(res, 200, { ok: true });
  }

  if (p === '/api/conversations' && m === 'GET') {
    const active = (await activeConv(s.owner, false))?.id || null;
    return json(res, 200, { active, items: (await store.convList(s.owner)).map(c => ({ ...c, updated_at: Number(c.updated_at) })) });
  }
  if (p === '/api/conversations' && m === 'POST') {
    const cur = await activeConv(s.owner, false);
    const empty = cur && !(await store.msgExists(cur.id)) && !JSON.parse(cur.session).state.items.length;
    return json(res, 200, { id: empty ? cur.id : (await newConv(s.owner)).id });
  }
  if ((mm = p.match(/^\/api\/conversations\/([\w-]+)$/))) {
    const c = mm[1] === 'active' ? await activeConv(s.owner) : await getConv(s.owner, mm[1]);
    if (!c) return err(res, 404, '对话不存在');
    if (m === 'DELETE') { await store.convDelete(c.id); return json(res, 200, { ok: true }); }
    await store.prefSet(s.owner, c.id);
    // reopening re-prices against today's catalog; never swaps silently
    const { session: sess, changed } = refresh(JSON.parse(c.session), catalog);
    if (changed) await saveConv(c, sess);
    const msgs = (await store.msgList(c.id)).map(x => ({ role: x.role, text: x.text, reply: x.reply ? JSON.parse(x.reply) : null, at: Number(x.created_at) }));
    return json(res, 200, { id: c.id, title: c.title, messages: msgs, plan: planView(sess), staleNotice: changed ? '价格有变化或已过期，方案里的商品和锁定都保留了，需要重新确认合计。' : null });
  }
  if ((p === '/api/chat' || p === '/api/chat/stream') && m === 'POST') {
    const b = await body(req);
    const text = String(b.text || '').slice(0, 4000);
    const c = b.conversationId ? await getConv(s.owner, b.conversationId) : await activeConv(s.owner);
    if (!c) return err(res, 404, '对话不存在');
    const prod = b.productId ? catalog.byId.get(String(b.productId)) : null;
    // streaming: one server-sent event per pipeline step, so the page can show the work as it happens
    const live = p === '/api/chat/stream';
    if (live) res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform', 'x-accel-buffering': 'no', connection: 'keep-alive' });
    const send = ev => { if (live) res.write(`data: ${JSON.stringify(ev)}\n\n`); };
    let out;
    try { out = await converse(JSON.parse(c.session), text, prod, send); }
    catch (e) { if (!live) throw e; console.error('chat stream', e); send({ type: 'error', message: '这次没处理好，请再发一次' }); return res.end(); }
    const { session: sess, reply } = out;
    await saveConv(c, sess);
    await store.msgAdd({ conversation_id: c.id, role: 'user', text, reply: null, created_at: Date.now() });
    await store.msgAdd({ conversation_id: c.id, role: 'assistant', text: '', reply: JSON.stringify(reply), created_at: Date.now() });
    if (live) { send({ type: 'done', conversationId: c.id, reply, plan: planView(sess) }); return res.end(); }
    return json(res, 200, { conversationId: c.id, reply, plan: planView(sess) });
  }
  if (p === '/api/plan/op' && m === 'POST') {
    const b = await body(req);
    const c = b.conversationId ? await getConv(s.owner, b.conversationId) : await activeConv(s.owner);
    if (!c) return err(res, 404, '方案不存在');
    const { session: sess, reply } = applyOp(JSON.parse(c.session), b, catalog);
    await saveConv(c, sess);
    return json(res, 200, { conversationId: c.id, reply, plan: planView(sess) });
  }
  return err(res, 404, '没有这个接口');
}

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.ico': 'image/x-icon', '.json': 'application/json', '.woff2': 'font/woff2' };
const SEC = {
  'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer', 'x-frame-options': 'DENY',
  'content-security-policy': "default-src 'self'; img-src 'self' data: https:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
};

// Assets live under /v/<build>/… so a new deploy is never masked by a cached module graph.
// build id = newest asset mtime, so any edit to public/ yields new URLs
const newest = dir => Math.max(0, ...readdirSync(dir).map(f => { const p = join(dir, f), st = statSync(p); return st.isDirectory() ? newest(p) : st.mtimeMs; }));
const buildId = () => Math.floor(newest(join(ROOT, 'public'))).toString(36);

// One chat turn. The agent takes part in understanding, planning, recall and ranking; the deterministic
// engine still owns filtering, budget and the final selection, so every budget rule holds whatever the model says.
const LOST = ['unknown', 'unknown-edit', 'budget-unreadable', 'budget-only'];
const vectors = existsSync(join(ROOT, 'data/embeddings.json')) ? JSON.parse(readFileSync(join(ROOT, 'data/embeddings.json'), 'utf8')).vectors : {};
const ALIASES = catalog.categories.flatMap(c => c.aliases).sort((x, y) => y.length - x.length);
// what is left of a sentence once product names, money and filler are removed: a use, a person, a feeling
const sceneResidue = text => ALIASES.reduce((t, a) => t.split(a).join(''), text)
  .replace(/[\d.]+|预算|以内|以下|左右|上下|一共|总共|合计|加起来|不超过?|元|块|钱|[万千百十两一二三四五六七八九零]|个|件|台|把|支|副|盏|我|想|要|买|帮|给|挑|推荐|的|吧|呢|啊|吗|哦|和|跟|还有|再|也|就|[\s，,。.、！!？?~～]/g, '');
const sameNeed = (a, b) => a.budgetCents === b.budgetCents && a.items.map(i => i.category + i.qty).join() === b.items.map(i => i.category + i.qty).join();

async function converse(prev, text, prod, emit = () => {}) {
  const trace = [], t0 = Date.now();
  const step = (id, label) => { const st = { id, label, at: Date.now() - t0 }; emit({ type: 'step', status: 'running', id, label }); return detail => { const done = { ...st, detail, ms: Date.now() - t0 - st.at }; trace.push(done); emit({ type: 'step', status: 'done', ...done }); }; };
  const opts = prod ? { productId: prod.id } : {};
  const run = (t, extra = {}) => turn(prev, t, catalog, Date.now(), { ...opts, ...extra });
  const yuan = c => `¥${(c / 100).toLocaleString('zh-CN')}`;
  const said = st => st.items.length ? `${st.items.map(i => `${i.alias}${i.qty > 1 ? ` ×${i.qty}` : ''}`).join('、')}${st.budgetCents ? ` · 总预算 ${yuan(st.budgetCents)}` : ' · 还没给预算'}` : '没读出要买什么';

  let done = step('read', '读取需求');
  let out = run(text), used = text, ai = null;
  done(prod ? `带着商品「${prod.title}」提问` : said(out.session.state));
  if (agent.agentOn() && !prod) {
    const lost = LOST.includes(out.reply.reason);
    const isNew = out.session.state.items.map(i => i.category).join() !== prev.state.items.map(i => i.category).join();
    if (lost || (isNew && sceneResidue(text).length >= 2)) {
      done = step('understand', '理解场景与意图');
      const u = await agent.understand(text, prev.state, catalog);
      done(u ? [u.scene && `场景：${u.scene}`, u.prefs.length && `偏好：${u.prefs.join('、')}`].filter(Boolean).join('；') || '没有额外的场景信息' : '这一步没有拿到结果，按原话继续');
      if (u) {
        ai = { scene: u.scene, plan: u.plan };
        const extra = u.prefs.length ? '，' + u.prefs.join('，') : '';
        if (u.rewrite && (lost || u.plan.length)) {
          done = step(u.plan.length ? 'plan' : 'rewrite', u.plan.length ? '拆解目标' : '改写需求');
          const again = run(u.rewrite + extra);
          if (!LOST.includes(again.reply.reason)) { out = again; used = u.rewrite + extra; out.reply.notes = [...(out.reply.notes || []), u.plan.length ? `按“${u.scene || text}”配了：${u.plan.map(x => x.品类).join('、')}` : `按「${u.rewrite}」理解`]; }
          done(u.plan.length ? `${u.plan.map(x => `${x.品类}（${x.为什么}）`).join('、')} → ${said(out.session.state)}` : `「${u.rewrite}」→ ${said(out.session.state)}`);
        } else if (extra) {
          const again = run(text + extra);
          if (sameNeed(again.session.state, out.session.state) && again.reply.status === out.reply.status) { out = again; used = text + extra; }
        }
      }
    }
  }
  const st = out.session.state;
  if (!prod && (out.reply.status === 'recommend' || out.reply.browse?.length || out.reply.status === 'no_solution')) {
    done = step('recall', '召回商品');
    done(st.items.map(it => {
      const pool = catalog.products.filter(p => p.category === it.category).length, ok = candidatesFor(it, st, catalog);
      const fit = st.budgetCents ? ok.filter(c => c.p.priceCents * it.qty <= st.budgetCents).length : null;
      return `${it.alias}：商品池 ${pool} 款 → 在售且符合条件 ${ok.length} 款${fit === null ? '' : ` → 单件不超预算 ${fit} 款`}`;
    }).join('\n'));
  }
  // semantic recall + rerank over what already passed every filter, then let the solver decide again
  if (agent.agentOn() && !prod && ai?.scene && out.reply.status === 'recommend') {
    const boost = {}, groups = [];
    done = step('semantic', '语义匹配');
    const qv = Object.keys(vectors).length ? (await agent.embed([ai.scene]))?.[0] : null;
    const topSem = [];
    for (const it of st.items) {
      if (it.lock) continue;
      const cands = candidatesFor(it, st, catalog).filter(c => c.p.priceCents * it.qty <= st.budgetCents);
      if (qv && cands.length > 1) {
        const sims = cands.map(c => vectors[c.p.id] ? agent.cosine(qv, vectors[c.p.id]) : null).filter(x => x !== null);
        const mean = sims.reduce((a, x) => a + x, 0) / (sims.length || 1), sd = Math.sqrt(sims.reduce((a, x) => a + (x - mean) ** 2, 0) / (sims.length || 1)) || 1;
        for (const c of cands) if (vectors[c.p.id]) boost[c.p.id] = Math.max(-120, Math.min(120, (agent.cosine(qv, vectors[c.p.id]) - mean) / sd * 60));
        topSem.push(`${it.alias}：${[...cands].sort((x, y) => (boost[y.p.id] || 0) - (boost[x.p.id] || 0)).slice(0, 2).map(c => c.p.title).join('、')}`);
      }
      const short = cands.map(c => ({ c, s: c.s + (boost[c.p.id] || 0) })).sort((a, b) => b.s - a.s).slice(0, 4).map(x => x.c.p);
      if (short.length > 1) groups.push({ name: it.alias, candidates: short.map(p => productView(p)) });
    }
    done(qv ? `按“${ai.scene}”和每款商品的语义做比对，最贴近的：\n${topSem.join('\n')}` : '语义服务暂不可用，按参数排序');
    if (groups.length) {
      done = step('rerank', '场景精排');
      // one small request per item, in parallel: faster, and one slow item no longer costs the others their scores
      const ranks = Object.assign({}, ...await Promise.all(groups.map(g => agent.rerank(ai.scene, [g]))));
      for (const [id, r] of Object.entries(ranks)) boost[id] = (boost[id] || 0) + (r.fit - 1.5) * 120;
      done(Object.keys(ranks).length ? groups.flatMap(g => g.candidates.filter(p => ranks[p.id]).map(p => `${p.title}：${ranks[p.id].fit} 分${ranks[p.id].why ? ` · ${ranks[p.id].why}` : ''}`)).join('\n') : '这一步没有拿到打分，保持参数排序');
      ai.reasons = Object.fromEntries(Object.entries(ranks).filter(([, r]) => r.why && r.fit >= 2).map(([id, r]) => [id, r.why]));
    }
    if (Object.keys(boost).length) {
      const again = run(used, { boost });
      if (again.reply.status === 'recommend' && sameNeed(again.session.state, out.session.state)) out = again;
    }
    ai.semantic = !!qv; ai.considered = groups.reduce((n, g) => n + g.candidates.length, 0);
  }
  const r0 = out.session.result;
  if (out.reply.status === 'recommend' && r0?.lines) {
    done = step('solve', '组合求解与核算');
    const picks = r0.lines.map(l => catalog.byId.get(l.productId)?.title).join(' + ');
    done(`比较了 ${r0.visited || 1} 个组合 → ${picks}\n合计 ${yuan(r0.totalCents ?? r0.subtotalCents)}${r0.withinBudget ? `，在预算 ${yuan(st.budgetCents)} 内` : r0.shippingCents === null ? '，运费以平台为准' : ''}`);
  } else if (out.reply.status === 'no_solution') {
    done = step('solve', '组合求解与核算'); done('按现在的预算和条件配不齐，保留购物车，给出可选调整');
  } else if (out.reply.status === 'compare') {
    done = step('compare', '找同类对比'); done(`${out.reply.compare.products.map(p => p.title).join(' vs ')}，对比 ${out.reply.compare.rows.length} 项`);
  }
  if (prod && LOST.includes(out.reply.reason)) {
    done = step('answer', '查看商品信息');
    const said2 = agent.agentOn() ? await agent.answer(text, prod) : null;
    const facts = productView(prod).facts;
    out = { session: out.session, reply: { status: 'answer', message: said2 || `${prod.title}，${productView(prod).observedText}${prod.source}标价 ¥${prod.priceCents / 100}${facts.length ? `，商品页写了：${facts.join('、')}` : '，商品页没写更多参数'}。想和同类比一比吗？`, options: ['帮我对比同类商品', `想买${catalog.catById.get(prod.category).shortName}，帮我挑`] } };
    done(facts.length ? `商品页写明：${facts.join('、')}` : '商品页没有更多参数');
  }
  if (prod) out.reply.contextProduct = productView(prod);
  if (ai && (ai.scene || ai.plan?.length)) out.reply.ai = ai;
  if (agent.agentOn() && (out.reply.status === 'recommend' || out.reply.status === 'compare' || out.reply.browse?.length)) {
    done = step('compose', '组织回复');
    const said3 = await agent.narrate(text, out.reply, out.session.state);
    if (said3) out.reply.agent = said3;
    done(said3 ? '理由、差异和小提示已核对，数字都来自商品数据' : '用标准文案回复');
  }
  out.reply.trace = { steps: trace, ms: Date.now() - t0 };
  return out;
}

async function serveStatic(req, res, url) {
  const path = decodeURIComponent(url.pathname).replace(/^\/v\/[\w]+\//, '/');
  const versioned = path !== decodeURIComponent(url.pathname);
  let file = normalize(join(ROOT, 'public', path));
  if (!file.startsWith(join(ROOT, 'public'))) return err(res, 403, '禁止访问');
  try { if ((await stat(file)).isDirectory()) file = join(file, 'index.html'); }
  catch { file = join(ROOT, 'public', 'index.html'); } // SPA fallback
  let data = await readFile(file);
  if (extname(file) === '.html') data = Buffer.from(data.toString('utf8').replaceAll('/v/BUILD/', `/v/${(process.env.VERCEL_GIT_COMMIT_SHA || '').slice(0, 10) || buildId()}/`));
  res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream', 'cache-control': versioned ? 'public, max-age=31536000, immutable' : 'no-cache' });
  res.end(data);
}

export async function handler(req, res) {
  for (const [k, v] of Object.entries(SEC)) res.setHeader(k, v);
  const url = new URL(req.url, 'http://local');
  try {
    if (url.pathname === '/healthz') return json(res, 200, { ok: true, products: catalog.products.length, store: store.kind, agent: agent.agentOn() });
    if (url.pathname.startsWith('/api/')) return await api(req, res, url, await session(req, res));
    return await serveStatic(req, res, url);
  } catch (e) {
    if (e.code === 413 || e.code === 400) return err(res, e.code, e.message);
    console.error(e);
    return err(res, 500, '服务开小差了，稍后再试');
  }
}
export const server = createServer(handler);

if (process.argv[1] === new URL(import.meta.url).pathname) server.listen(PORT, process.env.HOST || "127.0.0.1", () => console.log(`知选 listening on http://127.0.0.1:${PORT}`));
