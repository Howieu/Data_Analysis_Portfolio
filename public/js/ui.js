import { icon } from './icons.js';

export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
export const yuan = cents => { const y = Math.floor(cents / 100), f = cents % 100; return y.toLocaleString('zh-CN') + (f ? '.' + String(f).padStart(2, '0').replace(/0$/, '') : ''); };
export const money = (cents, cls = '') => Number.isSafeInteger(cents) ? `<span class="price num ${cls}"><small>¥</small>${yuan(cents)}</span>` : '<span class="price">—</span>';
export const img = (src, alt) => src ? `<img src="${esc(src)}" alt="${esc(alt)}" loading="lazy" referrerpolicy="no-referrer" data-fallback>` : '<div class="noimg">暂无图片</div>';

// one place decides how a broken image looks
document.addEventListener('error', e => {
  const t = e.target;
  if (t.tagName === 'IMG' && t.hasAttribute('data-fallback')) { const d = document.createElement('div'); d.className = 'noimg'; d.textContent = '图片加载失败'; t.replaceWith(d); }
}, true);

export function productCard(p) {
  return `<a class="card" href="/p/${encodeURIComponent(p.id)}" data-link>
    <div class="pic">${img(p.image, p.title)}${p.stock === 'out_of_stock' ? '<span class="tag">暂时缺货</span>' : ''}</div>
    <div class="body"><div class="t">${esc(p.title)}</div>${money(p.priceCents)}<div class="meta"><span>${esc(p.source)}</span><span>${esc(p.observedText)}</span></div></div>
  </a>`;
}

let toastTimer;
export function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg; el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

export function topbar(title, { back = true, right = '' } = {}) {
  return `<header class="topbar"><div class="side">${back ? `<button class="icon-btn" data-back aria-label="返回">${icon.back()}</button>` : ''}</div><h1>${esc(title)}</h1><div class="side">${right}</div></header>`;
}

export function sheet(html) {
  const scrim = document.createElement('div'); scrim.className = 'scrim';
  const box = document.createElement('div'); box.className = 'sheet'; box.setAttribute('role', 'dialog'); box.setAttribute('aria-modal', 'true');
  box.innerHTML = html;
  const close = () => { scrim.remove(); box.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = e => { if (e.key === 'Escape') close(); };
  scrim.addEventListener('click', close); document.addEventListener('keydown', onKey);
  document.body.append(scrim, box);
  box.querySelector('button, a, input')?.focus();
  return { box, close };
}

export const loadingGrid = n => `<div class="grid" aria-busy="true">${Array.from({ length: n }, () => '<div class="card"><div class="pic skeleton"></div><div class="body"><div class="skeleton sk-line"></div><div class="skeleton sk-line sk-short"></div></div></div>').join('')}</div>`;
export const errorBox = (msg, retry = true) => `<div class="empty"><h3>没加载出来</h3><p>${esc(msg)}</p>${retry ? '<button class="btn btn-soft" data-retry>重试</button>' : ''}</div>`;
