// In-memory stand-in for Supabase's PostgREST, covering only the query syntax lib/store.mjs uses.
// node test/mock-postgrest.mjs [port]  → then run the app with SUPABASE_URL=http://127.0.0.1:<port> SUPABASE_SERVICE_ROLE_KEY=local
import { createServer } from 'node:http';

const tables = { zx_users: [], zx_sessions: [], zx_favorites: [], zx_conversations: [], zx_messages: [], zx_prefs: [] };
const keys = { zx_favorites: ['owner', 'product_id'], zx_prefs: ['owner'], zx_users: ['email'], zx_sessions: ['token'], zx_conversations: ['id'] };
let seq = 1;

function parse(url) {
  const u = new URL(url, 'http://x'), table = u.pathname.replace('/rest/v1/', '');
  const filters = [], opts = {};
  for (const [k, v] of u.searchParams) {
    if (k === 'select') opts.select = v.split(',');
    else if (k === 'order') { const [c, d] = v.split('.'); opts.order = { c, d }; }
    else if (k === 'limit') opts.limit = Number(v);
    else if (k === 'on_conflict') opts.onConflict = v.split(',');
    else { const i = v.indexOf('.'); filters.push({ k, op: v.slice(0, i), val: v.slice(i + 1) }); }
  }
  return { table, filters, opts };
}
const match = (row, fs) => fs.every(({ k, op, val }) => {
  const x = row[k] == null ? null : String(row[k]);
  if (op === 'eq') return x === val;
  if (op === 'neq') return x !== val;
  if (op === 'in') return val.replace(/^\(|\)$/g, '').split(',').map(s => s.replace(/^"|"$/g, '')).includes(x);
  throw new Error('unsupported op ' + op);
});
const pick = (row, sel) => !sel || sel[0] === '*' ? { ...row } : Object.fromEntries(sel.map(c => [c, row[c] ?? null]));

const server = createServer(async (req, res) => {
  if (req.headers.apikey !== 'local') { res.writeHead(401); return res.end('{}'); }
  let body = ''; for await (const c of req) body += c;
  const { table, filters, opts } = parse(req.url), rows = tables[table];
  if (!rows) { res.writeHead(404); return res.end('{}'); }
  const prefer = req.headers.prefer || '';
  const send = (code, data) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(data === undefined ? '' : JSON.stringify(data)); };
  try {
    if (req.method === 'GET') {
      let out = rows.filter(r => match(r, filters));
      if (opts.order) out.sort((a, b) => (a[opts.order.c] > b[opts.order.c] ? 1 : -1) * (opts.order.d === 'desc' ? -1 : 1));
      if (opts.limit) out = out.slice(0, opts.limit);
      return send(200, out.map(r => pick(r, opts.select)));
    }
    if (req.method === 'POST') {
      const items = [JSON.parse(body)].flat(), created = [];
      for (const it of items) {
        const conflictCols = opts.onConflict || keys[table] || [];
        const dup = conflictCols.length && rows.find(r => conflictCols.every(c => String(r[c]) === String(it[c])));
        if (dup) {
          if (prefer.includes('ignore-duplicates')) continue;
          if (prefer.includes('merge-duplicates')) { Object.assign(dup, it); created.push(dup); continue; }
          return send(409, { message: 'duplicate key' });
        }
        if (table === 'zx_users' || table === 'zx_messages') it.id = seq++;
        if (table === 'zx_messages' && !tables.zx_conversations.some(c => c.id === it.conversation_id)) return send(409, { message: 'fk' });
        rows.push(it); created.push(it);
      }
      return prefer.includes('return=representation') ? send(201, created.map(r => pick(r, opts.select))) : send(201);
    }
    if (req.method === 'PATCH') { const patch = JSON.parse(body); rows.filter(r => match(r, filters)).forEach(r => Object.assign(r, patch)); return send(204); }
    if (req.method === 'DELETE') {
      const gone = rows.filter(r => match(r, filters));
      tables[table] = rows.filter(r => !gone.includes(r));
      if (table === 'zx_conversations') tables.zx_messages = tables.zx_messages.filter(m => !gone.some(c => c.id === m.conversation_id));
      if (table === 'zx_users') tables.zx_sessions = tables.zx_sessions.filter(s => !gone.some(u => String(u.id) === String(s.user_id)));
      return send(204);
    }
    send(405, {});
  } catch (e) { send(400, { message: e.message }); }
});
server.listen(Number(process.argv[2] || 54321), '127.0.0.1', () => console.log('mock postgrest ready'));
