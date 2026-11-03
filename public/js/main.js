import { api } from './api.js';
import { icon } from './icons.js';
import { toast, esc } from './ui.js';
import * as browse from './browse.js';
import * as chat from './chat.js';
import * as account from './account.js';

export const store = { me: undefined, meta: null, planCount: 0 };
export async function loadMe(force) { if (store.me === undefined || force) store.me = (await api('/api/me')).user; return store.me; }
export async function loadMeta() { return store.meta ||= await api('/api/meta'); }

const TABS = [
  ['/', '首页', 'home'], ['/categories', '分类', 'grid'], ['/chat', '问问知选', 'chat'], ['/plan', '购物车', 'cart'], ['/me', '我的', 'user'],
];
function renderTabs(path) {
  const active = path === '/' ? '/' : TABS.find(([href]) => href !== '/' && path.startsWith(href))?.[0];
  document.querySelector('.tabbar').innerHTML = TABS.map(([href, label, ic]) =>
    `<a href="${href}" data-link ${active === href ? 'aria-current="page"' : ''}>${ic === 'chat' ? '<img class="ico tab-logo" src="/logo.svg" alt="">' : icon[ic]()}<span>${label}</span>${href === '/plan' && store.planCount ? `<span class="badge num">${store.planCount}</span>` : ''}</a>`).join('');
}
export function setPlanCount(n) { store.planCount = n; renderTabs(location.pathname); }

const routes = [
  [/^\/$/, browse.home], [/^\/categories$/, browse.categories], [/^\/c\/([\w.-]+)$/, browse.category], [/^\/search$/, browse.search], [/^\/p\/([\w.-]+)$/, browse.product],
  [/^\/chat$/, chat.chatPage], [/^\/chat\/([\w-]+)$/, chat.chatPage], [/^\/plan$/, chat.planPage],
  [/^\/me$/, account.me], [/^\/login$/, account.login], [/^\/register$/, account.register], [/^\/favorites$/, account.favorites], [/^\/history$/, account.history], [/^\/account$/, account.security],
  [/^\/terms$/, account.terms], [/^\/privacy$/, account.privacy],
];

let seq = 0;
export async function render() {
  const path = location.pathname, my = ++seq;
  renderTabs(path);
  const view = document.getElementById('view');
  const hit = routes.map(([re, fn]) => [path.match(re), fn]).find(([m]) => m);
  document.body.classList.toggle('has-buybar', path.startsWith('/p/'));
  document.body.classList.toggle('has-fab', /^\/($|search|c\/|categories)/.test(path));
  if (!hit) { view.innerHTML = '<div class="empty"><h3>页面不存在</h3><a class="btn btn-soft" href="/" data-link>回到首页</a></div>'; return; }
  try { await hit[1](view, ...hit[0].slice(1).map(decodeURIComponent), () => my === seq); }
  catch (e) {
    if (my !== seq) return;
    // keep the page header so the user can always go back
    const head = view.querySelector('.topbar')?.outerHTML || `<header class="topbar"><div class="side"><button class="icon-btn" data-back aria-label="返回">${icon.back()}</button></div><h1></h1><div class="side"></div></header>`;
    const gone = e.status === 404;
    view.innerHTML = head + `<div class="empty"><h3>${gone ? '内容不存在' : '没加载出来'}</h3><p>${esc(e.message)}</p>${gone ? '<a class="btn btn-soft" href="/" data-link>回首页</a>' : '<button class="btn btn-soft" data-retry>重试</button>'}</div>`;
  }
}

export function go(href, { replace = false } = {}) {
  if (href === location.pathname + location.search) return render();
  history[replace ? 'replaceState' : 'pushState']({}, '', href);
  window.scrollTo(0, 0);
  render();
}

document.addEventListener('click', e => {
  const a = e.target.closest('a[data-link]');
  if (a && !e.metaKey && !e.ctrlKey && a.origin === location.origin) { e.preventDefault(); go(a.pathname + a.search); return; }
  if (e.target.closest('[data-back]')) { history.length > 1 ? history.back() : go('/'); return; }
  if (e.target.closest('[data-retry]')) render();
});
window.addEventListener('popstate', render);
window.addEventListener('unhandledrejection', e => { toast(e.reason?.message || '出了点问题'); });

loadMe().catch(() => {}).finally(render);
api('/api/conversations/active').then(c => setPlanCount(c.plan.state.items.length)).catch(() => {});
