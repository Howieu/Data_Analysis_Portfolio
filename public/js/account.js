import { api } from './api.js';
import { icon } from './icons.js';
import { esc, productCard, toast, topbar, loadingGrid, errorBox } from './ui.js';
import { go, loadMe, store, setPlanCount } from './main.js';

const EMAIL = /^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,24}$/;
const passOk = p => p.length >= 8 && p.length <= 72 && /[a-zA-Z]/.test(p) && /\d/.test(p);
const next = () => { const n = new URLSearchParams(location.search).get('next'); return n && n.startsWith('/') && !n.startsWith('//') ? n : '/me'; };

function passField(id, label, auto) {
  return `<div class="field"><label for="${id}">${label}</label><div class="inp"><input id="${id}" name="${id}" type="password" autocomplete="${auto}" required><button type="button" class="icon-btn" data-peek="${id}" aria-label="显示密码">${icon.eye('ico-s')}</button></div><span class="err" data-err="${id}"></span></div>`;
}
function bindPeek(root) {
  root.addEventListener('click', e => {
    const b = e.target.closest('[data-peek]'); if (!b) return;
    const inp = root.querySelector(`#${b.dataset.peek}`); const show = inp.type === 'password';
    inp.type = show ? 'text' : 'password'; b.innerHTML = show ? icon.eyeOff('ico-s') : icon.eye('ico-s'); b.setAttribute('aria-label', show ? '隐藏密码' : '显示密码');
  });
}
const fieldErr = (root, id, msg) => { const el = root.querySelector(`[data-err="${id}"]`); if (el) el.textContent = msg || ''; };

function authForm(view, mode) {
  const reg = mode === 'register';
  view.innerHTML = topbar(reg ? '注册' : '登录') + `
  <form class="auth" novalidate data-auth>
    <div><h2>${reg ? '注册知选账号' : '欢迎回来'}</h2><p class="sub">${reg ? '收藏、购物车和对话会保存在账号里，换设备也能看到。' : '登录后，收藏和购物车会同步到你的账号。'}</p></div>
    <div class="form-err" data-form-err hidden></div>
    <div class="field"><label for="email">邮箱</label><div class="inp"><input id="email" name="email" type="email" inputmode="email" autocomplete="email" required></div><span class="err" data-err="email"></span></div>
    ${reg ? '<div class="field"><label for="name">昵称（选填）</label><div class="inp"><input id="name" name="name" maxlength="20" autocomplete="nickname"></div></div>' : ''}
    ${passField('password', '密码', reg ? 'new-password' : 'current-password')}
    ${reg ? '<span class="hint small muted">8–72 位，同时包含字母和数字</span><label class="check"><input type="checkbox" name="agree" required><span>我已阅读并同意 <a href="/terms" data-link>用户协议</a> 和 <a href="/privacy" data-link>隐私政策</a></span></label><span class="err" data-err="agree"></span>' : ''}
    <button class="btn btn-dark btn-block" data-submit>${reg ? '注册并登录' : '登录'}</button>
    <p class="alt-link">${reg ? '已有账号？<a href="/login" data-link>去登录</a>' : '还没有账号？<a href="/register" data-link>注册一个</a>'}</p>
    <p class="alt-link"><a href="/" data-link>先随便逛逛</a></p>
  </form>`;
  const form = view.querySelector('[data-auth]');
  bindPeek(form);
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const f = new FormData(form), email = String(f.get('email') || '').trim(), password = String(f.get('password') || '');
    let bad = false;
    fieldErr(form, 'email', EMAIL.test(email) ? '' : (bad = true, '请输入正确的邮箱'));
    fieldErr(form, 'password', reg ? (passOk(password) ? '' : (bad = true, '密码需要 8–72 位，同时包含字母和数字')) : password ? '' : (bad = true, '请输入密码'));
    if (reg) fieldErr(form, 'agree', f.get('agree') ? '' : (bad = true, '需要先同意用户协议和隐私政策'));
    if (bad) return;
    const btn = form.querySelector('[data-submit]'), box = form.querySelector('[data-form-err]');
    btn.disabled = true; box.hidden = true;
    try {
      const r = await api(`/api/auth/${reg ? 'register' : 'login'}`, { method: 'POST', body: { email, password, name: f.get('name') || '' } });
      store.me = r.user;
      toast(r.importedPlan ? '登录成功，登录前的购物车已另存，可以在历史对话里找到' : reg ? '注册成功' : '登录成功');
      api('/api/conversations/active').then(c => setPlanCount(c.plan.state.items.length)).catch(() => {});
      go(next(), { replace: true });
    } catch (err) { box.textContent = err.message; box.hidden = false; btn.disabled = false; }
  });
}
export const login = view => authForm(view, 'login');
export const register = view => authForm(view, 'register');

export async function me(view, alive) {
  const user = await loadMe(); if (!alive()) return;
  const [fav, convs] = await Promise.all([api('/api/favorites').catch(() => ({ items: [] })), api('/api/conversations').catch(() => ({ items: [] }))]);
  if (!alive()) return;
  view.innerHTML = topbar('我的', { back: false }) + `
    <div class="me-head">${user ? `<span class="avatar">${esc((user.name || user.email)[0].toUpperCase())}</span><div class="who"><b>${esc(user.name)}</b><span class="muted small">${esc(user.email)}</span></div>`
      : `<span class="avatar">${icon.user()}</span><div class="who"><b>还没登录</b><span class="muted small">登录后收藏和购物车会保存在账号里</span></div><a class="btn btn-dark" href="/login" data-link>登录 / 注册</a>`}</div>
    <nav class="list">
      <a href="/favorites" data-link>${icon.heart('ico-s')}<span class="grow">我的收藏</span><span class="val num">${fav.items.length}</span>${icon.chev('ico-s chev')}</a>
      <a href="/history" data-link>${icon.clock('ico-s')}<span class="grow">历史对话</span><span class="val num">${convs.items.length}</span>${icon.chev('ico-s chev')}</a>
      <a href="/plan" data-link>${icon.bag('ico-s')}<span class="grow">购物车</span>${icon.chev('ico-s chev')}</a>
    </nav>
    <nav class="list">
      ${user ? `<a href="/account" data-link>${icon.lock('ico-s')}<span class="grow">账号与安全</span>${icon.chev('ico-s chev')}</a>` : ''}
      <a href="/terms" data-link><span class="grow">用户协议</span>${icon.chev('ico-s chev')}</a>
      <a href="/privacy" data-link><span class="grow">隐私政策</span>${icon.chev('ico-s chev')}</a>
    </nav>
    ${user ? '<nav class="list"><button data-logout><span class="grow">退出登录</span></button></nav>' : ''}`;
  view.querySelector('[data-logout]')?.addEventListener('click', async () => {
    await api('/api/auth/logout', { method: 'POST' }); store.me = null; setPlanCount(0); toast('已退出登录'); go('/me', { replace: true });
  });
}

export async function favorites(view, alive) {
  view.innerHTML = topbar('我的收藏') + `<div class="feed" data-f>${loadingGrid(4)}</div>`;
  const el = view.querySelector('[data-f]');
  try {
    const r = await api('/api/favorites'); if (!alive()) return;
    el.innerHTML = r.items.length ? `<div class="grid">${r.items.map(productCard).join('')}</div>${r.missing ? `<p class="section muted small">有 ${r.missing} 件收藏的商品已经下架。</p>` : ''}` : '<div class="empty"><h3>还没有收藏</h3><p>在商品页点心形就能收藏。</p><a class="btn btn-soft" href="/" data-link>去逛逛</a></div>';
  } catch (e) { el.innerHTML = errorBox(e.message); }
}

export async function history(view, alive) {
  view.innerHTML = topbar('历史对话') + '<nav class="list" data-h></nav>';
  const r = await api('/api/conversations'); if (!alive()) return;
  const fmt = t => { const d = new Date(t); return `${d.getMonth() + 1}月${d.getDate()}日 ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`; };
  const el = view.querySelector('[data-h]');
  el.innerHTML = r.items.length ? r.items.map(c => `<a href="/chat/${c.id}" data-link>${icon.chat('ico-s')}<span class="grow">${esc(c.title)}${c.id === r.active ? ' <span class="tag">当前</span>' : ''}</span><span class="val">${fmt(c.updated_at)}</span>${icon.chev('ico-s chev')}</a>`).join('') : '<div class="empty"><h3>还没有对话</h3><a class="btn btn-soft" href="/chat" data-link>问问知选</a></div>';
}

export async function security(view, alive) {
  const user = await loadMe(); if (!alive()) return;
  if (!user) return go('/login?next=/account', { replace: true });
  view.innerHTML = topbar('账号与安全') + `
    <form class="auth" data-pw novalidate><h2>修改密码</h2><div class="form-err" data-form-err hidden></div>
      ${passField('current', '当前密码', 'current-password')}${passField('next', '新密码', 'new-password')}
      <button class="btn btn-dark btn-block">保存新密码</button></form>
    <form class="auth" data-del novalidate><h2>注销账号</h2><p class="sub">注销后，收藏、购物车和对话会被永久删除，无法恢复。</p><div class="form-err" data-form-err hidden></div>
      ${passField('delpass', '输入密码确认', 'current-password')}
      <button class="btn btn-line btn-block danger">注销账号</button></form>`;
  bindPeek(view);
  view.querySelector('[data-pw]').addEventListener('submit', async e => {
    e.preventDefault(); const f = e.target, box = f.querySelector('[data-form-err]'); box.hidden = true;
    const cur = f.current.value, nxt = f.next.value;
    fieldErr(f, 'next', passOk(nxt) ? '' : '新密码需要 8–72 位，同时包含字母和数字'); if (!passOk(nxt)) return;
    try { await api('/api/account/password', { method: 'POST', body: { current: cur, next: nxt } }); toast('密码已修改，其他设备需要重新登录'); f.reset(); }
    catch (err) { box.textContent = err.message; box.hidden = false; }
  });
  view.querySelector('[data-del]').addEventListener('submit', async e => {
    e.preventDefault(); const f = e.target, box = f.querySelector('[data-form-err]'); box.hidden = true;
    const btn = f.querySelector('button.danger');
    if (btn.dataset.confirm !== '1') { btn.dataset.confirm = '1'; btn.textContent = '再点一次，确认永久注销'; return; }
    try { await api('/api/account', { method: 'DELETE', body: { password: f.delpass.value } }); store.me = null; toast('账号已注销'); go('/', { replace: true }); }
    catch (err) { box.textContent = err.message; box.hidden = false; btn.dataset.confirm = ''; btn.textContent = '注销账号'; }
  });
}

export function terms(view) {
  view.innerHTML = topbar('用户协议') + `<article class="doc"><h2>知选用户协议</h2>
  <h3>一、服务内容</h3><p>知选根据你描述的用途、数量和预算，从电商平台公开在售的商品中帮你挑选并比较。知选本身不销售商品，也不处理支付；你点击「去平台查看」后，购买行为在对应平台完成，以该平台的规则为准。</p>
  <h3>二、价格与信息</h3><p>商品价格、库存、运费和优惠信息来自平台公开页面，并标注了获取时间。平台价格可能随时变动，下单时以平台结算页为准。运费或优惠无法确认时，知选不会把方案标注为「在预算内」。</p>
  <h3>三、账号</h3><p>你需要对账号下的操作负责，请妥善保管密码。你可以随时在「账号与安全」中修改密码或注销账号。</p>
  <h3>四、使用规范</h3><p>请勿利用知选从事违法活动，或以自动化方式大量抓取服务内容。</p>
  <h3>五、变更</h3><p>协议调整时，我们会在应用内提示，继续使用视为同意更新后的协议。</p></article>`;
}
export function privacy(view) {
  view.innerHTML = topbar('隐私政策') + `<article class="doc"><h2>知选隐私政策</h2>
  <h3>我们收集什么</h3><p>注册时的邮箱和昵称；你在对话中输入的需求文字；收藏、购物车与对话记录。密码经过加盐哈希保存，我们无法看到原文。</p>
  <h3>用来做什么</h3><p>仅用于为你挑选商品、保存你的购物车和历史对话，以及保障账号安全（例如登录次数限制）。</p>
  <h3>不会做什么</h3><p>不出售你的个人信息；不向商品平台传递你的账号信息。点击外部商品链接时，我们不附带你的身份信息。</p>
  <h3>保存与删除</h3><p>未登录时的数据与浏览器会话关联，最长保留 30 天。注销账号后，收藏、购物车和对话将被删除。</p>
  <h3>Cookie</h3><p>我们只使用一个必要的会话 Cookie 来保持登录状态和你的购物车，不使用广告追踪。</p></article>`;
}
