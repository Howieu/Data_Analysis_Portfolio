export async function api(path, { method = 'GET', body } = {}) {
  let res;
  try {
    res = await fetch(path, { method, headers: body ? { 'content-type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined, credentials: 'same-origin' });
  } catch {
    throw new Error('网络好像断了，检查一下再试');
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data.error || '出了点问题，稍后再试'), { status: res.status });
  return data;
}

// Server-sent events over POST: calls onEvent for each `data:` line, resolves with the final `done` event.
export async function stream(path, body, onEvent) {
  let res;
  try {
    res = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), credentials: 'same-origin' });
  } catch {
    throw new Error('网络好像断了，检查一下再试');
  }
  if (!res.ok || !res.body) { const data = await res.json().catch(() => ({})); throw Object.assign(new Error(data.error || '出了点问题，稍后再试'), { status: res.status }); }
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '', final = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      if (!chunk.startsWith('data: ')) continue;
      const ev = JSON.parse(chunk.slice(6));
      if (ev.type === 'error') throw new Error(ev.message);
      if (ev.type === 'done') final = ev; else onEvent(ev);
    }
  }
  if (!final) throw new Error('连接中断了，再发一次试试');
  return final;
}
