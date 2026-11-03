import { api } from './api.js';
import { icon } from './icons.js';
import { esc, money, img, productCard, toast, topbar, loadingGrid, errorBox } from './ui.js';
import { go, loadMeta, store, setPlanCount } from './main.js';

// a query that reads like a need (budget, quantity, "帮我") goes to the assistant instead of keyword search
const looksLikeNeed = q => /预算|以内|以下|左右|一共|总共|\d+\s*(元|块)|帮我|推荐|想买|适合|送/.test(q);

function searchBar(value = '') {
  return `<form class="search" role="search" data-search>${icon.search('ico-s')}<input name="q" type="search" enterkeyhint="search" value="${esc(value)}" placeholder="搜商品，或直接说需求" aria-label="搜索商品"></form>`;
}
function bindSearch(root) {
  root.querySelector('[data-search]')?.addEventListener('submit', e => {
    e.preventDefault();
    const q = new FormData(e.target).get('q').trim();
    if (!q) return;
    if (looksLikeNeed(q)) return go(`/chat?q=${encodeURIComponent(q)}`);
    go(`/search?q=${encodeURIComponent(q)}`);
  });
}

export async function feed(el, params, alive) {
  let page = 1, total = 0;
  const load = async () => {
    const more = el.querySelector('[data-more]');
    if (more) more.innerHTML = '<span class="muted small">加载中…</span>';
    const r = await api(`/api/products?${new URLSearchParams({ ...params, page })}`);
    if (!alive()) return;
    total = r.total;
    let grid = el.querySelector('.grid:not([aria-busy])');
    if (page === 1 || !grid) { grid = Object.assign(document.createElement('div'), { className: 'grid' }); el.innerHTML = ''; el.append(grid); }
    grid.insertAdjacentHTML('beforeend', r.items.map(productCard).join(''));
    el.querySelector('[data-more]')?.remove();
    if (!r.total) el.innerHTML = '<div class="empty"><h3>没有找到相关商品</h3><p>换个说法试试，或者直接告诉知选你的需求和预算。</p><a class="btn btn-soft" href="/chat" data-link>问问知选</a></div>';
    else if (page * 20 < total) el.insertAdjacentHTML('beforeend', '<div class="more" data-more><button class="btn btn-soft">加载更多</button></div>');
    page++;
  };
  el.innerHTML = loadingGrid(6);
  el.addEventListener('click', e => { if (e.target.closest('[data-more] button')) load().catch(err => toast(err.message)); });
  try { await load(); } catch (e) { el.innerHTML = errorBox(e.message); }
}

export async function home(view, alive) {
  view.innerHTML = `
    <div class="home-head">
      <div class="row"><span class="wordmark">知选</span>${searchBar()}</div>
      <a class="ask" href="/chat" data-link><img class="mark" src="/logo.svg" alt=""><div><b>问问知选</b><span>说预算和用途，一件或一整套都能配</span></div>${icon.chev()}</a>
    </div>
    <div class="cats" data-cats></div>
    <div class="tabs" role="tablist" data-tabs></div>
    <div class="feed" data-feed></div>`;
  bindSearch(view);
  const meta = await loadMeta();
  if (!alive()) return;
  view.querySelector('[data-cats]').innerHTML = meta.categories.map(c => `<a href="/c/${c.id}" data-link><span class="c-pic">${img(c.cover, c.name)}</span>${esc(c.short)}</a>`).join('');
  const tabs = [['', '推荐'], ...meta.categories.map(c => [c.id, c.short])];
  const tabEl = view.querySelector('[data-tabs]');
  tabEl.innerHTML = tabs.map(([id, n], i) => `<button role="tab" aria-selected="${i === 0}" data-cat="${id}">${esc(n)}</button>`).join('');
  const feedEl = view.querySelector('[data-feed]');
  tabEl.addEventListener('click', e => {
    const b = e.target.closest('[data-cat]'); if (!b) return;
    tabEl.querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    const fresh = feedEl.cloneNode(false); feedEl.replaceWith(fresh);
    feed(fresh, b.dataset.cat ? { cat: b.dataset.cat } : {}, alive);
  });
  await feed(feedEl, {}, alive);
}

export async function categories(view, alive) {
  view.innerHTML = topbar('分类', { back: false }) + '<div class="cat-grid" data-c></div>';
  const meta = await loadMeta(); if (!alive()) return;
  view.querySelector('[data-c]').innerHTML = meta.categories.map(c => `<a href="/c/${c.id}" data-link><span class="c-pic">${img(c.cover, c.name)}</span><span>${esc(c.name)}</span><small class="num">${c.count} 件</small></a>`).join('');
}

function listPage(view, title, params, alive, extra = '') {
  const sort = params.sort || '';
  view.innerHTML = topbar(title) + `<div class="filters">${searchBar(params.q || '')}<div class="sortbar">${[['', '综合'], ['price-asc', '价格从低到高'], ['price-desc', '价格从高到低']].map(([k, n]) => `<button data-sort="${k}" aria-pressed="${k === sort}">${n}</button>`).join('')}</div></div>${extra}<div class="feed" data-feed></div>`;
  bindSearch(view);
  view.querySelector('.sortbar').addEventListener('click', e => {
    const b = e.target.closest('[data-sort]'); if (!b) return;
    const u = new URL(location.href); b.dataset.sort ? u.searchParams.set('sort', b.dataset.sort) : u.searchParams.delete('sort');
    go(u.pathname + u.search, { replace: true });
  });
  return feed(view.querySelector('[data-feed]'), params, alive);
}

export async function category(view, id, alive) {
  const meta = await loadMeta();
  const c = meta.categories.find(x => x.id === id);
  if (!c) { view.innerHTML = topbar('分类') + '<div class="empty"><h3>这个分类不存在</h3></div>'; return; }
  const sp = new URLSearchParams(location.search);
  const extra = `<div class="need-hint"><p>不知道选哪款？说说预算，知选帮你挑。</p><a class="btn btn-soft" href="/chat?q=${encodeURIComponent(`想买${c.short}`)}" data-link>问问知选</a></div>`;
  return listPage(view, c.name, { cat: id, sort: sp.get('sort') || '' }, alive, extra);
}

export async function search(view, alive) {
  const sp = new URLSearchParams(location.search);
  const q = sp.get('q') || '';
  const extra = q ? `<div class="need-hint"><p>想按预算挑「${esc(q)}」？</p><a class="btn btn-soft" href="/chat?q=${encodeURIComponent(q)}" data-link>让知选帮我挑</a></div>` : '';
  return listPage(view, q ? `“${q}”` : '搜索', { q, sort: sp.get('sort') || '' }, alive, extra);
}

export async function product(view, id, alive) {
  view.innerHTML = topbar('商品详情') + '<div class="pd-skel" aria-busy="true"><div class="skeleton sk-pic"></div><div class="skeleton sk-price"></div><div class="skeleton sk-line"></div><div class="skeleton sk-line sk-short"></div></div>';
  let data;
  try { data = await api(`/api/products/${encodeURIComponent(id)}`); }
  catch (e) {
    if (!alive()) return;
    view.innerHTML = topbar('商品详情') + (e.status === 404
      ? '<div class="empty"><h3>这件商品已下架</h3><p>可以看看同类的其他商品。</p><a class="btn btn-soft" href="/categories" data-link>看看同类</a><a class="link" href="/" data-link>回首页</a></div>'
      : errorBox(e.message));
    return;
  }
  const { product: p, favorite, similar } = data;
  if (!alive()) return;
  const imgs = p.images?.length ? p.images : [null];
  const specs = Object.entries(p.specs || {});
  view.innerHTML = topbar(p.categoryName) + `
    <div class="pd-wrap">
      <div class="gallery-wrap">${imgs.length > 1 ? `<button class="g-nav g-prev" data-g="-1" aria-label="上一张">${icon.back()}</button><button class="g-nav g-next" data-g="1" aria-label="下一张">${icon.chev()}</button>` : ''}<div class="gallery" data-gallery>${imgs.map((s, i) => s ? `<img src="${esc(s)}" alt="${esc(p.title)} 图${i + 1}" referrerpolicy="no-referrer" ${i ? 'loading="lazy"' : ''} data-fallback>` : '<div class="noimg">暂无图片</div>').join('')}</div>${imgs.length > 1 ? `<span class="gallery-count num" data-count>1/${imgs.length}</span>` : ''}</div>
      <div>
        <div class="pd">
          <div class="price-row">${money(p.priceCents)}<span class="muted small">${esc(p.observedText)} ${esc(p.source)}价格${p.fresh ? '' : ' · 需要刷新'}</span></div>
          <h2>${esc(p.title)}</h2>
          <div class="muted small">${esc(p.brand || '')}${p.stock === 'out_of_stock' ? ' · 暂时缺货' : p.stock === 'in_stock' ? ' · 有货' : ''}</div>
          ${p.facts.length ? `<div class="chips">${p.facts.map(f => `<span class="chip">${esc(f)}</span>`).join('')}</div>` : ''}
          <div class="muted small">${esc(p.shippingNote || '运费以平台为准')}</div>
        </div>
        <div class="pd-block pd-ask"><h3><img class="logo-s" src="/logo.svg" alt="">问问知选</h3><div class="chips">${['帮我对比一下同类商品', '这款适合我吗', `${p.categoryName}怎么选`].map(t => `<a class="chip" href="/chat?q=${encodeURIComponent(t)}&p=${encodeURIComponent(p.id)}" data-link>${esc(t)}</a>`).join('')}</div></div>
        ${p.highlights.length ? `<div class="pd-block"><h3>卖点</h3><ul class="hl">${p.highlights.map(h => `<li>${esc(h)}</li>`).join('')}</ul></div>` : ''}
        ${specs.length ? `<div class="pd-block"><h3>规格参数</h3><table class="specs">${specs.map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td>${esc(v)}</td></tr>`).join('')}</table></div>` : ''}
        ${similar.length ? `<div class="pd-block"><h3>价位相近</h3><div class="hscroll">${similar.map(productCard).join('')}</div></div>` : ''}
        <div class="buybar">
          <button class="icon-col" data-fav aria-pressed="${favorite}" aria-label="收藏">${icon.heart('ico')}<span>${favorite ? '已收藏' : '收藏'}</span></button>
          <a class="icon-col" href="/chat?q=${encodeURIComponent('这款适合我吗')}&p=${encodeURIComponent(p.id)}" data-link aria-label="问问知选"><img class="ico" src="/logo.svg" alt=""><span>问知选</span></a>
          <button class="btn btn-soft" data-add>加入购物车</button>
          <a class="btn btn-primary" href="${esc(p.url)}" target="_blank" rel="noopener noreferrer" aria-label="去${esc(p.source)}购买">立即购买</a>
        </div>
      </div>
    </div>`;
  const g = view.querySelector('[data-gallery]'), cnt = view.querySelector('[data-count]');
  view.querySelectorAll('[data-g]').forEach(b => b.addEventListener('click', () => g.scrollBy({ left: g.clientWidth * Number(b.dataset.g), behavior: 'smooth' })));
  g?.addEventListener('scroll', () => { if (cnt) cnt.textContent = `${Math.round(g.scrollLeft / g.clientWidth) + 1}/${imgs.length}`; }, { passive: true });
  const favBtn = view.querySelector('[data-fav]');
  favBtn.addEventListener('click', async () => {
    const on = favBtn.getAttribute('aria-pressed') !== 'true';
    await api(`/api/favorites/${encodeURIComponent(p.id)}`, { method: on ? 'POST' : 'DELETE' });
    favBtn.setAttribute('aria-pressed', String(on)); favBtn.querySelector('span').textContent = on ? '已收藏' : '收藏';
    toast(on ? (store.me ? '已收藏' : '已收藏，登录后会同步到账号') : '已取消收藏');
  });
  view.querySelector('[data-add]').addEventListener('click', async () => {
    const r = await api('/api/plan/op', { method: 'POST', body: { op: 'add', productId: p.id } });
    setPlanCount(r.plan.state.items.length);
    toast(r.plan.state.budgetCents ? '已加入购物车，并按总预算重新核算' : '已加入购物车');
  });
}
