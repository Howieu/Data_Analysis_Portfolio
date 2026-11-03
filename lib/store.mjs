// Persistence behind one async interface: SQLite on a single machine, Supabase (PostgREST) when deployed.
// Selected by env: NEXT_PUBLIC_SUPABASE_URL (or SUPABASE_URL) + SUPABASE_SERVICE_ROLE_KEY → Supabase, otherwise SQLite.

export async function openStore({ dbFile } = {}) {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL, key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (url && key && process.env.STORE !== 'sqlite') return supabaseStore(url.replace(/\/$/, ''), key);
  const { openDb } = await import('./db.mjs');
  return sqliteStore(openDb(dbFile));
}

function sqliteStore(db) {
  const q = (sql, ...a) => db.prepare(sql);
  return {
    kind: 'sqlite',
    sessionGet: async t => q('select token, user_id, guest, expires_at from sessions where token = ?').get(t) || null,
    sessionCreate: async r => { q('insert into sessions (token, user_id, guest, created_at, expires_at) values (?, ?, ?, ?, ?)').run(r.token, r.user_id ?? null, r.guest, r.created_at, r.expires_at); },
    sessionDelete: async t => { q('delete from sessions where token = ?').run(t); },
    sessionDeleteUserExcept: async (uid, t) => { q('delete from sessions where user_id = ? and token != ?').run(uid, t); },
    userByEmail: async e => q('select * from users where email = ?').get(e) || null,
    userById: async id => q('select id, email, name, created_at from users where id = ?').get(id) || null,
    userWithPass: async id => q('select * from users where id = ?').get(id) || null,
    userCreate: async r => { const i = q('insert into users (email, name, pass, created_at) values (?, ?, ?, ?)').run(r.email, r.name, r.pass, r.created_at); return q('select id, email, name, created_at from users where id = ?').get(i.lastInsertRowid); },
    userSetPass: async (id, pass) => { q('update users set pass = ? where id = ?').run(pass, id); },
    userDelete: async id => { q('delete from users where id = ?').run(id); },
    favHas: async (o, p) => !!q('select 1 from favorites where owner = ? and product_id = ?').get(o, p),
    favList: async o => q('select product_id from favorites where owner = ? order by created_at desc').all(o).map(r => r.product_id),
    favAdd: async (o, p) => { q('insert or ignore into favorites (owner, product_id, created_at) values (?, ?, ?)').run(o, p, Date.now()); },
    favRemove: async (o, p) => { q('delete from favorites where owner = ? and product_id = ?').run(o, p); },
    favMove: async (from, to) => { q('insert or ignore into favorites (owner, product_id, created_at) select ?, product_id, created_at from favorites where owner = ?').run(to, from); q('delete from favorites where owner = ?').run(from); },
    favDeleteOwner: async o => { q('delete from favorites where owner = ?').run(o); },
    convGet: async (o, id) => q('select * from conversations where id = ? and owner = ?').get(id, o) || null,
    convByOwner: async o => q('select id, title, session, updated_at from conversations where owner = ?').all(o),
    convList: async o => q("select id, title, updated_at from conversations c where owner = ? and (exists (select 1 from messages where conversation_id = c.id) or title != '新的对话') order by updated_at desc limit 50").all(o),
    convCreate: async r => { q('insert into conversations (id, owner, title, session, created_at, updated_at) values (?, ?, ?, ?, ?, ?)').run(r.id, r.owner, r.title, r.session, r.created_at, r.updated_at); },
    convUpdate: async (id, f) => { const k = Object.keys(f); q(`update conversations set ${k.map(x => `${x} = ?`).join(', ')} where id = ?`).run(...k.map(x => f[x]), id); },
    convDelete: async id => { q('delete from conversations where id = ?').run(id); },
    convDeleteOwner: async o => { q('delete from conversations where owner = ?').run(o); },
    msgList: async c => q('select role, text, reply, created_at from messages where conversation_id = ? order by id').all(c),
    msgAdd: async r => { q('insert into messages (conversation_id, role, text, reply, created_at) values (?, ?, ?, ?, ?)').run(r.conversation_id, r.role, r.text, r.reply ?? null, r.created_at); },
    msgExists: async c => !!q('select 1 from messages where conversation_id = ?').get(c),
    prefGet: async o => q('select active_conversation from prefs where owner = ?').get(o)?.active_conversation || null,
    prefSet: async (o, id) => { q('insert into prefs (owner, active_conversation) values (?, ?) on conflict(owner) do update set active_conversation = excluded.active_conversation').run(o, id); },
    prefDelete: async o => { q('delete from prefs where owner = ?').run(o); },
  };
}

function supabaseStore(base, key) {
  const H = { apikey: key, authorization: `Bearer ${key}`, 'content-type': 'application/json' };
  const enc = v => encodeURIComponent(String(v));
  async function call(method, path, body, prefer) {
    const res = await fetch(`${base}/rest/v1/${path}`, { method, headers: { ...H, ...(prefer ? { prefer } : {}) }, body: body ? JSON.stringify(body) : undefined });
    if (!res.ok) throw new Error(`store ${method} ${path.split('?')[0]} ${res.status}`);
    return res.status === 204 ? null : res.json().catch(() => null);
  }
  const get = path => call('GET', path);
  const one = async path => (await get(path))?.[0] || null;
  return {
    kind: 'supabase',
    sessionGet: t => one(`zx_sessions?token=eq.${enc(t)}&select=token,user_id,guest,expires_at`),
    sessionCreate: r => call('POST', 'zx_sessions', r, 'return=minimal'),
    sessionDelete: t => call('DELETE', `zx_sessions?token=eq.${enc(t)}`),
    sessionDeleteUserExcept: (uid, t) => call('DELETE', `zx_sessions?user_id=eq.${enc(uid)}&token=neq.${enc(t)}`),
    userByEmail: e => one(`zx_users?email=eq.${enc(e)}&select=*`),
    userById: id => one(`zx_users?id=eq.${enc(id)}&select=id,email,name,created_at`),
    userWithPass: id => one(`zx_users?id=eq.${enc(id)}&select=*`),
    userCreate: async r => { const [u] = await call('POST', 'zx_users?select=id,email,name,created_at', r, 'return=representation'); return u; },
    userSetPass: (id, pass) => call('PATCH', `zx_users?id=eq.${enc(id)}`, { pass }),
    userDelete: id => call('DELETE', `zx_users?id=eq.${enc(id)}`),
    favHas: async (o, p) => !!(await one(`zx_favorites?owner=eq.${enc(o)}&product_id=eq.${enc(p)}&select=product_id`)),
    favList: async o => (await get(`zx_favorites?owner=eq.${enc(o)}&select=product_id&order=created_at.desc`)).map(r => r.product_id),
    favAdd: (o, p) => call('POST', 'zx_favorites?on_conflict=owner,product_id', { owner: o, product_id: p, created_at: Date.now() }, 'resolution=ignore-duplicates,return=minimal'),
    favRemove: (o, p) => call('DELETE', `zx_favorites?owner=eq.${enc(o)}&product_id=eq.${enc(p)}`),
    favMove: async (from, to) => {
      const rows = await get(`zx_favorites?owner=eq.${enc(from)}&select=product_id,created_at`);
      if (rows.length) await call('POST', 'zx_favorites?on_conflict=owner,product_id', rows.map(r => ({ owner: to, ...r })), 'resolution=ignore-duplicates,return=minimal');
      await call('DELETE', `zx_favorites?owner=eq.${enc(from)}`);
    },
    favDeleteOwner: o => call('DELETE', `zx_favorites?owner=eq.${enc(o)}`),
    convGet: (o, id) => one(`zx_conversations?id=eq.${enc(id)}&owner=eq.${enc(o)}&select=*`),
    convByOwner: o => get(`zx_conversations?owner=eq.${enc(o)}&select=id,title,session,updated_at`),
    convList: async o => {
      const rows = await get(`zx_conversations?owner=eq.${enc(o)}&select=id,title,updated_at&order=updated_at.desc&limit=100`);
      if (!rows.length) return [];
      const ids = rows.map(r => `"${r.id}"`).join(',');
      const withMsg = new Set((await get(`zx_messages?conversation_id=in.(${enc(ids)})&select=conversation_id`)).map(r => r.conversation_id));
      return rows.filter(r => withMsg.has(r.id) || r.title !== '新的对话').slice(0, 50);
    },
    convCreate: r => call('POST', 'zx_conversations', r, 'return=minimal'),
    convUpdate: (id, f) => call('PATCH', `zx_conversations?id=eq.${enc(id)}`, f),
    convDelete: id => call('DELETE', `zx_conversations?id=eq.${enc(id)}`),
    convDeleteOwner: o => call('DELETE', `zx_conversations?owner=eq.${enc(o)}`),
    msgList: c => get(`zx_messages?conversation_id=eq.${enc(c)}&select=role,text,reply,created_at&order=id.asc`),
    msgAdd: r => call('POST', 'zx_messages', { ...r, reply: r.reply ?? null }, 'return=minimal'),
    msgExists: async c => !!(await one(`zx_messages?conversation_id=eq.${enc(c)}&select=id&limit=1`)),
    prefGet: async o => (await one(`zx_prefs?owner=eq.${enc(o)}&select=active_conversation`))?.active_conversation || null,
    prefSet: (o, id) => call('POST', 'zx_prefs?on_conflict=owner', { owner: o, active_conversation: id }, 'resolution=merge-duplicates,return=minimal'),
    prefDelete: o => call('DELETE', `zx_prefs?owner=eq.${enc(o)}`),
  };
}
