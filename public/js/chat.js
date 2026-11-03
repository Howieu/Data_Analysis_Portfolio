import { api, stream } from './api.js';
import { icon } from './icons.js';
import { esc, money, img, toast, sheet, yuan, topbar } from './ui.js';
import { go, setPlanCount, loadMeta } from './main.js';
import { feed } from './browse.js';

const SUGGEST = ['我想买一个便携小风扇，平时带着出门用', '通勤用的降噪耳机，500以内', '吹风机和电动牙刷，一共600', '给爸妈各买一个电动牙刷'];
const ASK_LIKE = [['一件', '通勤降噪耳机，预算 800'], ['一整套', '吹风机和电动牙刷，一共 600'], ['先看看', '想买台灯，还没想好预算']];
export const LOGO = '/logo.svg';
const logo = (cls = 'logo') => `<img class="${cls}" src="${LOGO}" alt="" aria-hidden="true">`;


// ---------- one assistant reply ----------
const stepLi = (st, live) => `<li class="${live && st.status === 'running' ? 'run' : 'ok'}" data-step="${esc(st.id)}"><span class="dot" aria-hidden="true">${live && st.status === 'running' ? '' : icon.check('ico-s')}</span><div><b>${esc(st.label)}</b>${st.detail ? `<span class="det">${esc(st.detail)}</span>` : ''}</div></li>`;
function traceBox(t) {
  return `<details class="steps trace"><summary>${logo('logo-s')}<span>已思考 · 用时 ${(t.ms / 1000).toFixed(1)} 秒 · ${t.steps.length} 步</span>${icon.chev('ico-s chev')}</summary><ol>${t.steps.map(st => stepLi(st, false)).join('')}</ol></details>`;
}
function steps(r) {
  if (r.trace?.steps?.length) return traceBox(r.trace);
  const s = [];
  if (r.browse?.length) {
    s.push(['理解需求', `想看看${r.browse.map(b => b.name).join('、')}，预算还没定`]);
    s.push(['整理商品', r.browse.map(b => `${b.count} 款在售${b.name}，按价位分成 ${b.bands.length} 档`).join('；')]);
  } else if (r.status === 'recommend') {
    const lines = (r.blocks || []).find(b => b.lines)?.lines || [];
    const notes = (r.notes || []).filter(Boolean);
    s.push(['理解需求', notes.length ? notes.join('；') : lines.map(l => `${l.item.categoryName}${l.item.qty > 1 ? ` ×${l.item.qty}` : ''}`).join('、')]);
    s.push(['挑选商品', `${lines.length} 件按预算和条件挑好，${lines.some(l => l.alternatives.length) ? '每件附备选' : '暂时没有更多备选'}`]);
    if (r.summary?.budgetState) s.push(['核算总价', r.summary.budgetState.text]);
  } else if (r.status === 'no_solution') s.push(['核算总价', '按现在的预算和条件配不齐']);
  const ai = r.ai;
  if (ai?.scene) s.splice(1, 0, ['理解场景', ai.scene]);
  if (ai?.plan?.length) s.splice(ai.scene ? 2 : 1, 0, ['目标规划', ai.plan.map(x => `${x.品类}（${x.为什么}）`).join('、')]);
  if (ai?.considered) s.splice(s.length - (r.summary?.budgetState ? 1 : 0), 0, ['语义召回与精排', `按场景给 ${ai.considered} 款预算内候选打了匹配分，再交给组合求解定稿`]);
  if (!s.length) return '';
  return `<details class="steps"><summary>${logo('logo-s')}<span>搞定，看看下面</span>${icon.chev('ico-s chev')}</summary><ol>${s.map(([k, v]) => `<li>${icon.check('ico-s')}<div><b>${k}</b><span>${esc(v)}</span></div></li>`).join('')}</ol></details>`;
}

let FIT = {};
function row(p, { tag = '', meta = '', add = null, pick = null, qty = 1 } = {}) {
  if (FIT[p.id]) tag += `<span class="tag tag-fit">契合场景：${esc(FIT[p.id])}</span>`;
  const href = `/p/${encodeURIComponent(p.id)}`;
  const btn = pick ? `<button class="mini" data-pick-alt="${esc(p.id)}" data-item="${esc(pick)}">换这款</button>`
    : add === 'in' ? '<span class="tag">已在购物车</span>'
    : `<button class="add" data-add="${esc(p.id)}" aria-label="加入购物车">${icon.cart('ico-s')}</button>`;
  return `<div class="prow">
    <a class="pic" href="${href}" data-link>${img(p.image, p.title)}</a>
    <button class="fav" data-fav="${esc(p.id)}" aria-label="收藏">${icon.star('ico-s')}</button>
    <div class="info">
      <a class="t" href="${href}" data-link><span class="src">${esc(p.source)}</span>${esc(p.title)}</a>
      <div class="tags">${tag}${p.facts.slice(0, 2).map(f => `<span class="tag">${esc(f)}</span>`).join('')}</div>
      <div class="buy">${money(p.priceCents)}${qty > 1 ? `<span class="muted small num">×${qty}</span>` : ''}<span class="muted small">${esc(meta || `${p.observedText}价格${p.fresh ? '' : '·需刷新'}`)}</span>
        <span class="ops">${btn}<a class="go" href="${esc(p.url)}" target="_blank" rel="noopener noreferrer" aria-label="去${esc(p.source)}购买">立即购买</a></span></div>
    </div>
  </div>`;
}
const allBtn = (cat, n) => `<button class="all" data-goods="${esc(cat)}">查看全部${n ? ` ${n} 款` : '商品'}${icon.chev('ico-s')}</button>`;

function guideCards(cats, meta) {
  const gs = cats.map(id => meta?.categories.find(c => c.id === id)).filter(c => c?.guide);
  if (!gs.length) return '';
  const cards = gs.flatMap(c => c.guide.points.map(([k, v], i) => ({ c, k, v, i }))).slice(0, 4);
  return `<div class="sec"><h3>选购攻略</h3><div class="gcards">${cards.map(x => `<button data-guide="${x.c.id}" data-point="${x.i}"><b>${esc(gs.length > 1 ? `${x.c.short}·${x.k}` : `${x.k}怎么看？`)}</b><span>${esc(x.v)}</span></button>`).join('')}</div></div>`;
}

function filterCard(title, chips, skip) {
  if (!chips.length) return '';
  return `<div class="filter"><div class="fh"><h3>${esc(title)}</h3></div><div class="fgrid">${chips.map(c => `<button data-say="${esc(c)}">${esc(c)}</button>`).join('')}</div>${skip ? `<button class="skip" data-goods="${esc(skip)}">先不定预算，看看全部</button>` : ''}</div>`;
}

// the budget stays in the background: one plain sentence, never a badge or a picker
function budgetLine(summary) {
  const bs = summary?.budgetState;
  if (!bs) return '';
  if (bs.kind !== 'within') return `<p class="note">${esc(bs.text)}。</p>`;
  const left = summary.budgetCents - summary.totalCents;
  return `<p class="note">合计 ¥${yuan(summary.totalCents)}，在预算内${left > 0 ? `，还留了 ¥${yuan(left)}` : ''}。</p>`;
}

export function replyCats(r) {
  if (r?.browse?.length) return r.browse.map(b => b.category);
  return [...new Set((r?.blocks || []).flatMap(b => (b.lines || []).map(l => l.item.category)))];
}

export function replyHtml(r, { meta = null, last = false, prevText = '' } = {}) {
  if (!r) return '';
  FIT = r.ai?.reasons || {};
  const parts = [steps(r)], ag = r.agent;
  const agentBits = title => ag ? [ag.points?.length ? `<div class="sec"><h3>${title}</h3><ul class="pts">${ag.points.map(([k, v]) => `<li><b>${esc(k)}</b>：${esc(v)}</li>`).join('')}</ul></div>` : '', ag.tip ? `<p class="tip">${icon.bulb('ico-s')}<span><b>小提示</b>：${esc(ag.tip)}</span></p>` : '', ag.ask ? `<p class="ask"><b>${esc(ag.ask)}</b></p>` : ''].join('') : '';
  const cats = replyCats(r);
  if (r.status === 'compare' && r.compare) {
    const c = r.compare, ctx = r.contextProduct?.id;
    parts.push(`<p>${esc(ag?.lead || r.message)}</p>`);
    parts.push(`<div class="cmp-heads">${c.products.map(p => `<a class="cmp-h" href="/p/${encodeURIComponent(p.id)}" data-link><span class="pic">${img(p.image, p.title)}</span><span class="t">${p.id === ctx ? '<em>当前商品</em>' : ''}${esc(p.title)}</span></a>`).join('')}</div>`);
    parts.push(`<div class="cmp-wrap"><table class="cmp"><thead><tr><th scope="col">维度</th>${c.products.map(p => `<th scope="col">${esc(p.title)}</th>`).join('')}</tr></thead><tbody><tr><th scope="row">对比商品</th>${c.products.map(p => `<td><span class="pic">${img(p.image, p.title)}</span></td>`).join('')}</tr>${c.rows.map(([k, ...v]) => `<tr><th scope="row">${esc(k)}</th>${v.map(x => `<td${k === '商品价格' ? ' class="cmp-price num"' : ''}>${esc(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
    parts.push(agentBits('怎么理解这些差异').replace(/<p class="ask">.*?<\/p>/, ''));
    parts.push(`<div class="sec"><h3>选款建议</h3><p><b>首选</b>：${esc(c.first.why)}</p>${row(c.first)}<p><b>备选</b>：${esc(c.second.why)}</p>${row(c.second)}</div>`);
    if (ag?.ask) parts.push(`<p class="ask"><b>${esc(ag.ask)}</b></p>`);
  } else if (r.status !== 'recommend' || !r.blocks) {
    if (r.message) parts.push(`<p>${esc(ag?.lead || r.message)}</p>`);
    for (const b of (r.blocks || []).filter(b => b.type === 'text')) parts.push(`<p>${esc(b.text)}</p>`);
    if ((r.notes || []).length) parts.push(`<p class="note">${esc(r.notes.join('；'))}</p>`);
    for (const b of r.browse || []) {
      const m = Math.floor((b.bands.length - 1) / 2);
      const order = b.bands.map((band, i) => ({ band, title: b.bands.length === 1 ? b.name : i === m ? `多数人会选的${b.name}` : i < m ? '想省一点' : '想要更好的' })).sort((x, y) => (x.band === b.bands[m] ? -1 : 0) - (y.band === b.bands[m] ? -1 : 0));
      for (const { band, title } of order) {
        const span = band.minCents === band.maxCents ? `¥${yuan(band.minCents)}` : `¥${yuan(band.minCents)}–${yuan(band.maxCents)}`;
        parts.push(`<div class="sec"><h3>${esc(title)}<span class="num">${span}</span></h3>${band.products.slice(0, band === b.bands[m] ? 3 : 2).map(p => row(p)).join('')}</div>`);
      }
      parts.push(allBtn(b.category, b.count));
    }
    if (r.status === 'no_solution') parts.push(filterCard('可以这样调整', [...(r.options || []), ...(r.actions || [])]));
    if (r.browse?.length) parts.push(agentBits('怎么选'));
    parts.push(guideCards(cats, meta));
  } else {
    if (r.opening) parts.push(`<p>${esc(ag?.lead || r.opening)}</p>`);
    parts.push(budgetLine(r.summary));
    const notes = (r.notes || []).filter(n => !/^总预算 ¥[\d,.]+$/.test(n));
    if (notes.length) parts.push(`<p class="note">已按：${esc(notes.join('；'))}</p>`);
    for (const w of r.warnings || []) parts.push(`<p class="warn">${esc(w)}</p>`);
    const single = r.view?.mode === 'single' ? r.view.itemId : null;
    for (const b of r.blocks) {
      if (b.type === 'pick' || b.type === 'bundle') {
        const lines = single ? b.lines.filter(l => l.itemId === single) : b.lines;
        if (b.type === 'bundle' && !single) parts.push(`<h3 class="bundle-h">${esc(b.title)}</h3>`);
        if (single) parts.push(`<p class="note">单独看${esc(lines[0]?.item.categoryName || '')}，购物车里还有 ${b.lines.length - 1} 件，总预算不变。</p>`);
        for (const l of lines) {
          const tag = `<span class="tag tag-best">最适合你</span>${l.item.locked ? '<span class="tag">已锁定</span>' : ''}`;
          parts.push(`<div class="sec"><h3>${esc(b.lines.length > 1 || single ? l.item.categoryName : b.title)}</h3>${row(l.product, { tag, add: 'in', qty: l.qty })}${l.reason ? `<p class="why">${esc(l.reason)}</p>` : ''}${l.alternatives.slice(0, 2).map(a => row(a, { tag: `<span class="tag">${esc(a.diffText)}</span>`, pick: l.item.locked ? null : l.itemId })).join('')}${allBtn(l.item.category)}</div>`);
        }
      } else if (b.type === 'text') parts.push(`<p>${esc(b.text)}</p>`);
    }
    if (r.upgrade) parts.push(`<p class="note">${esc(r.upgrade.text)}：<a href="/p/${encodeURIComponent(r.upgrade.id)}" data-link>${esc(r.upgrade.title)}</a></p>`);
    parts.push(agentBits('为什么这么挑'));
    parts.push(guideCards(cats, meta));
  }
  parts.push(`<div class="fb"><button class="icon-btn" data-copy aria-label="复制回复">${icon.copy('ico-s')}</button>${last && prevText ? `<button class="icon-btn" data-say="${esc(prevText)}" aria-label="重新回答">${icon.redo('ico-s')}</button>` : ''}</div>`);
  const guess = !last ? [] : r.status === 'recommend' ? r.actions || [] : r.status === 'compare' ? [] : r.status === 'no_solution' || r.browse?.length ? [] : [...(r.options || []), ...(r.actions || [])];
  return `<div class="a-msg">${parts.join('')}</div>${guess.length ? `<div class="guess"><h4>猜你想问</h4>${guess.map(o => `<button data-say="${esc(o)}">${esc(o)}</button>`).join('')}</div>` : ''}`;
}

// ---------- shopping list (used on /plan and beside the chat on desktop) ----------
export function planHtml(plan) {
  const s = plan.state, r = plan.result;
  if (!s.items.length) return `<div class="empty"><h3>购物车还是空的</h3><p>在商品页点「加入购物车」，或者告诉知选你想买什么、预算多少。</p><a class="btn btn-dark" href="/chat" data-link>问问知选</a></div>`;
  const lines = r?.lines || [];
  const lineOf = id => lines.find(l => l.itemId === id);
  let status;
  if (!s.budgetCents) status = '<div class="status">还没设总预算。设好预算，整个购物车会一起按总价配。</div>';
  else if (!lines.length) status = '<div class="status bad">按现在的预算和条件配不齐，可以提高预算、去掉一件，或者到对话里调整。</div>';
  else if (r.budgetState.kind === 'within') status = `<div class="status ok">${esc(r.budgetState.text)}</div>`;
  else status = `<div class="status pending">${esc(r.budgetState.text)}${r.staleReason ? '。价格有变化，商品和锁定都保留着' : ''}</div>`;
  const used = r?.lines?.length ? (r.totalCents ?? r.subtotalCents) : 0;
  const single = plan.view?.mode === 'single' ? plan.view.itemId : null;
  const items = single ? s.items.filter(i => i.id === single) : s.items;
  return `
  <div class="plan-head">
    <div class="budget-row"><div><div class="label">总预算</div>${s.budgetCents ? money(s.budgetCents) : '<span class="muted">未设置</span>'}</div><button class="mini" data-edit-budget>${icon.edit('ico-s')}${s.budgetCents ? '改预算' : '设预算'}</button></div>
    <form class="budget-edit" data-budget-form hidden><input name="yuan" inputmode="decimal" placeholder="例如 800" aria-label="总预算（元）" value="${s.budgetCents ? s.budgetCents / 100 : ''}"><button class="btn btn-dark">确定</button><button type="button" class="btn btn-soft" data-cancel-budget>取消</button></form>
    ${s.budgetCents && used ? `<meter min="0" max="${s.budgetCents}" value="${Math.min(used, s.budgetCents)}" aria-label="已用预算"></meter>` : ''}
    ${status}
    ${s.items.length > 1 ? `<div><span class="seg" role="group" aria-label="查看方式"><button data-view="bundle" aria-pressed="${!single}">整套</button><button data-view="single" aria-pressed="${!!single}">单品</button></span>${single ? `<span class="muted small">　只是换个看法，购物车 ${s.items.length} 件和总预算都不变</span>` : ''}</div>` : ''}
    ${single ? `<div class="chips">${s.items.map(i => `<button class="chip" data-view="single" data-item="${i.id}" aria-pressed="${i.id === single}">${esc(i.alias || i.category)}</button>`).join('')}</div>` : ''}
  </div>
  <div class="items">${items.map(i => {
    const l = lineOf(i.id), p = l && plan.products[l.productId];
    const alts = (r?.alternatives?.[i.id] || []).length;
    return `<div class="item" data-item="${i.id}">
      <div class="top">
        <button class="icon-btn rm" data-remove aria-label="从购物车移除">${icon.trash('ico-s')}</button>
        <div class="pic">${p ? img(p.image, p.title) : '<div class="noimg">待选</div>'}</div>
        <div>
          <div class="cat"><span>${esc(i.alias || '')}</span>${i.lock ? '<span class="tag">已锁定</span>' : ''}${(i.hard || []).map(h => `<span class="tag">${esc(h.label)}</span>`).join('')}</div>
          ${p ? `<a class="t" href="/p/${encodeURIComponent(p.id)}" data-link>${esc(p.title)}</a><div>${money(p.priceCents)} <span class="muted small num">× ${i.qty}</span></div>` : '<div class="muted small">这一件按现在的条件还没挑到</div>'}
        </div>
      </div>
      <div class="bottom">
        <span class="stepper"><button data-qty="-1" aria-label="减少数量" ${i.qty <= 1 ? 'disabled' : ''}>${icon.minus('ico-s')}</button><span class="num" aria-label="数量">${i.qty}</span><button data-qty="1" aria-label="增加数量">${icon.plus('ico-s')}</button></span>
        <button class="mini" data-lock aria-pressed="${!!i.lock}" ${!i.selected && !i.lock ? 'disabled' : ''}>${i.lock ? icon.lock('ico-s') + '已锁定' : icon.unlock('ico-s') + '锁定'}</button>
        <button class="mini" data-swap ${i.lock || !alts ? 'disabled' : ''}>${icon.swap('ico-s')}换一款</button>
      </div>
      ${p ? `<a class="extlink" href="${esc(p.url)}" target="_blank" rel="noopener noreferrer">去${esc(p.source)}查看 · ${esc(p.observedText)}价格${icon.ext('ico-s')}</a>` : ''}
    </div>`;
  }).join('')}</div>`;
}

export function bindPlan(el, getPlan, apply) {
  el.addEventListener('click', async e => {
    const plan = getPlan();
    const item = e.target.closest('[data-item]')?.dataset.item;
    const t = e.target.closest('button');
    if (!t || t.disabled) return;
    if (t.matches('[data-edit-budget]')) { el.querySelector('[data-budget-form]').hidden = false; el.querySelector('[data-budget-form] input').focus(); return; }
    if (t.matches('[data-cancel-budget]')) { el.querySelector('[data-budget-form]').hidden = true; return; }
    if (t.matches('[data-view]')) return apply({ op: 'view', mode: t.dataset.view, itemId: t.dataset.item || plan.state.items[0]?.id });
    const it = plan.state.items.find(i => i.id === item);
    if (!it) return;
    if (t.matches('[data-qty]')) return apply({ op: 'qty', itemId: item, qty: it.qty + Number(t.dataset.qty) });
    if (t.matches('[data-lock]')) return apply({ op: it.lock ? 'unlock' : 'lock', itemId: item });
    if (t.matches('[data-remove]')) return apply({ op: 'remove', itemId: item });
    if (t.matches('[data-swap]')) {
      const alts = plan.result?.alternatives?.[item] || [];
      const { box, close } = sheet(`<h3>换一款${esc(it.alias || '')}</h3><p class="sub">其他商品不动，合计按换后的方案重新算。</p>${alts.map(a => {
        const p = plan.products[a.productId]; if (!p) return '';
        return `<div class="opt"><div class="pic">${img(p.image, p.title)}</div><div><div class="t">${esc(p.title)}</div><div>${money(p.priceCents)} <span class="small ${a.withinBudget ? 'muted' : ''}">${a.withinBudget ? `换后合计 ¥${yuan(a.totalCents)}` : '换后运费以平台为准'}</span></div></div><button class="mini" data-pick="${esc(p.id)}">选这款</button></div>`;
      }).join('')}`);
      box.addEventListener('click', ev => { const b = ev.target.closest('[data-pick]'); if (b) { close(); apply({ op: 'pick', itemId: item, productId: b.dataset.pick }); } });
    }
  });
  el.addEventListener('submit', e => {
    if (!e.target.matches('[data-budget-form]')) return;
    e.preventDefault();
    const v = new FormData(e.target).get('yuan').trim();
    if (!/^\d+(\.\d{1,2})?$/.test(v) || Number(v) <= 0) return toast('预算需要是大于 0 的金额，最多两位小数');
    apply({ op: 'budget', yuan: v });
  });
}

// ---------- pages ----------
export async function planPage(view, alive) {
  view.innerHTML = topbar('购物车', { back: false }) + '<div data-plan><div class="empty muted">加载中…</div></div>';
  let conv = await api('/api/conversations/active');
  if (!alive()) return;
  const el = view.querySelector('[data-plan]');
  const paint = () => { el.innerHTML = (conv.staleNotice ? `<p class="warn section">${esc(conv.staleNotice)}</p>` : '') + planHtml(conv.plan); setPlanCount(conv.plan.state.items.length); };
  paint();
  bindPlan(el, () => conv.plan, async op => {
    const r = await api('/api/plan/op', { method: 'POST', body: { conversationId: conv.id, ...op } });
    conv = { ...conv, plan: r.plan, staleNotice: null };
    if (r.reply.status === 'clarify' && r.reply.message) toast(r.reply.message);
    paint();
  });
}

// ---------- the assistant: 知选 | 商品 | 攻略 ----------
function introHtml() {
  return `<div class="intro">
    <img class="hero" src="${LOGO}" alt="知选">
    <h2>你好，我是知选</h2>
    <p class="muted">说说想买什么、预算多少。一件或者一整套，都按总价帮你配好，价格和运费算清楚再告诉你。</p>
    <div class="likes">${ASK_LIKE.map(([k, v]) => `<button data-say="${esc(v)}"><b>${k}</b><span>${esc(v)}</span></button>`).join('')}</div>
    <div class="guess"><h4>猜你想问</h4>${SUGGEST.map(s => `<button data-say="${esc(s)}">${esc(s)}</button>`).join('')}</div>
  </div>`;
}

function goodsPane(el, cats, meta, active, alive) {
  const list = cats.length ? cats : meta.categories.map(c => c.id);
  const cur = list.includes(active) ? active : list[0];
  el.innerHTML = `<div class="subtabs" role="tablist">${list.map(id => `<button role="tab" data-goods="${id}" aria-selected="${id === cur}">${esc(meta.categories.find(c => c.id === id)?.short || id)}</button>`).join('')}</div><div class="feed" data-feed></div>`;
  return feed(el.querySelector('[data-feed]'), { cat: cur }, alive);
}

function guidePane(el, cats, meta, active, point) {
  const withGuide = (cats.length ? cats : meta.categories.map(c => c.id)).map(id => meta.categories.find(c => c.id === id)).filter(c => c?.guide);
  if (!withGuide.length) { el.innerHTML = '<div class="empty"><h3>这类商品还没有整理攻略</h3><p>直接问知选，我按商品页信息帮你比。</p></div>'; return; }
  const c = withGuide.find(x => x.id === active) || withGuide[0], g = c.guide;
  el.innerHTML = `${withGuide.length > 1 ? `<div class="subtabs" role="tablist">${withGuide.map(x => `<button role="tab" data-guide="${x.id}" aria-selected="${x === c}">${esc(x.short)}</button>`).join('')}</div>` : ''}
    <div class="subtabs topics">${g.points.map(([k], i) => `<button data-topic="${i}" aria-selected="${String(i) === String(point ?? 0)}">${esc(k)}</button>`).join('')}</div>
    <article class="guide"><h2>${esc(g.title)}</h2>${g.points.map(([k, v], i) => `<section id="topic-${i}"><h3>${i + 1}. ${esc(k)}</h3><p>${esc(v)}</p></section>`).join('')}${g.caution ? `<section><h3>避坑提醒</h3><p>${esc(g.caution)}</p></section>` : ''}
    <button class="btn btn-dark" data-say="${esc(`想买${c.short}`)}">按我的预算挑${esc(c.short)}</button></article>`;
  if (point != null) requestAnimationFrame(() => el.querySelector(`#topic-${point}`)?.scrollIntoView({ block: 'start' }));
}

const prodChip = p => p ? `<a class="u-prod" href="/p/${encodeURIComponent(p.id)}" data-link><span class="pic">${img(p.image, p.title)}<span class="num">¥${yuan(p.priceCents)}</span></span><span class="muted small">发送1款商品</span></a>` : '';
const postCard = p => `<a class="post" href="${esc(p.url)}" target="_blank" rel="noopener noreferrer"><b>${esc(p.title)}</b><p>${esc(p.excerpt)}</p><div class="pm"><span>${esc(p.author || '')}</span><span class="num">${[p.likes ? `赞 ${p.likes}` : '', p.collects ? `收藏 ${p.collects}` : ''].filter(Boolean).join(' · ')}</span></div>${p.tags.length ? `<div class="tags">${p.tags.map(t => `<span class="tag">#${esc(t)}</span>`).join('')}</div>` : ''}</a>`;
const postsCache = new Map();
const postsOf = cat => postsCache.get(cat) || postsCache.set(cat, api(`/api/posts?cat=${encodeURIComponent(cat)}`).catch(() => ({ items: [] }))).get(cat);

async function postsPane(el, cats, meta, active, alive) {
  const list = cats.length ? cats : meta.categories.map(c => c.id);
  const cur = list.includes(active) ? active : list[0];
  el.innerHTML = `<div class="subtabs" role="tablist">${list.map(id => `<button role="tab" data-posts="${id}" aria-selected="${id === cur}">${esc(meta.categories.find(c => c.id === id)?.short || id)}</button>`).join('')}</div><div class="posts" data-list><div class="empty muted">加载中…</div></div>`;
  const r = await postsOf(cur); if (!alive()) return;
  el.querySelector('[data-list]').innerHTML = r.items.length ? r.items.map(postCard).join('') + `<p class="disclaim">来自${esc(r.source || '')}用户的公开笔记，点开看原文</p>` : '<div class="empty"><h3>这类商品还没有收集到体验笔记</h3></div>';
}

// append "大家怎么说" to the newest reply once its notes arrive
async function addVoices(msgs, r, alive) {
  const cat = replyCats(r)[0]; if (!cat) return;
  const p = await postsOf(cat); if (!alive() || !p.items.length) return;
  const box = [...msgs.querySelectorAll('.a-msg')].pop(); if (!box || box.querySelector('.voices')) return;
  box.querySelector('.fb').insertAdjacentHTML('beforebegin', `<div class="sec voices"><h3>大家怎么说</h3>${p.items.slice(0, 2).map(postCard).join('')}<button class="all" data-posts="${esc(cat)}">看更多真实体验${icon.chev('ico-s')}</button></div>`);
}

export async function chatPage(view, id, alive) {
  if (typeof id === 'function') { alive = id; id = null; }
  const sp = new URLSearchParams(location.search), q = sp.get('q'), qp = sp.get('p');
  view.innerHTML = `<header class="topbar agent-top"><div class="side"><a class="icon-btn" href="/history" data-link aria-label="历史对话">${icon.menu()}</a></div>
      <div class="seg-tabs" role="tablist"><button role="tab" data-pane-btn="chat" aria-selected="true">${logo('logo-s')}知选</button><i aria-hidden="true"></i><button role="tab" data-pane-btn="goods" aria-selected="false">商品</button><button role="tab" data-pane-btn="guide" aria-selected="false">攻略</button><button role="tab" data-pane-btn="posts" aria-selected="false">体验</button></div>
      <div class="side"><button class="icon-btn" data-new aria-label="新对话">${icon.new()}</button></div></header>` +
    `<div class="chat-layout"><section class="chat" aria-label="对话" data-pane="chat"><div class="msgs" data-msgs aria-live="polite"></div>
      <div class="composer"><div class="quick" data-quick></div><form data-form><label class="sr" for="say">说说你的需求</label><textarea id="say" rows="1" maxlength="500" placeholder="有需要的告诉我"></textarea><button class="send" aria-label="发送" disabled>${icon.send('ico-s')}</button></form><p class="disclaim">回复由知选根据商品页信息生成，价格和运费以平台为准</p></div></section>
      <section class="pane" data-pane="goods" hidden></section><section class="pane" data-pane="guide" hidden></section><section class="pane" data-pane="posts" hidden></section>
      <aside class="side-plan" aria-label="当前购物车" data-side></aside></div>`;
  const [c0, meta] = await Promise.all([api(`/api/conversations/${id || 'active'}`), loadMeta()]);
  let conv = c0;
  if (!alive()) return;
  const msgs = view.querySelector('[data-msgs]'), side = view.querySelector('[data-side]'), quick = view.querySelector('[data-quick]');
  const form = view.querySelector('[data-form]'), ta = form.querySelector('textarea'), sendBtn = form.querySelector('.send');
  const panes = Object.fromEntries([...view.querySelectorAll('[data-pane]')].map(p => [p.dataset.pane, p]));
  const ctxCats = () => {
    const lastR = [...conv.messages].reverse().find(m => m.reply && replyCats(m.reply).length)?.reply;
    return lastR ? replyCats(lastR) : [...new Set(conv.plan.state.items.map(i => i.category))];
  };
  const show = (name, opt = {}) => {
    for (const [k, el] of Object.entries(panes)) el.hidden = k !== name;
    view.querySelectorAll('[data-pane-btn]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.paneBtn === name)));
    if (name === 'goods') goodsPane(panes.goods, ctxCats(), meta, opt.cat, alive);
    if (name === 'guide') guidePane(panes.guide, ctxCats(), meta, opt.cat, opt.point);
    if (name === 'posts') postsPane(panes.posts, ctxCats(), meta, opt.cat, alive);
    window.scrollTo(0, 0);
  };
  const paintSide = () => { side.innerHTML = `<div class="section-h section"><h2>当前购物车</h2><a href="/plan" data-link>全部</a></div>` + planHtml(conv.plan); setPlanCount(conv.plan.state.items.length); };
  const paintQuick = () => {
    const has = conv.plan.result?.lines?.length;
    const chips = has ? [['say', '便宜一点'], ['say', '运费怎么算'], ['say', '价格是哪天的'], ['link', '看看购物车', '/plan']] : [['fill', '按预算配一套', '预算 元，想买'], ['say', '通勤用的降噪耳机'], ['say', '吹风机和电动牙刷，一共600']];
    quick.innerHTML = chips.map(([k, t, v]) => k === 'link' ? `<a class="chip" href="${v}" data-link>${t}</a>` : `<button class="chip" ${k === 'say' ? `data-say="${esc(t)}"` : `data-fill="${esc(v)}"`}>${t}</button>`).join('');
  };
  const paint = () => {
    const n = conv.messages.length;
    msgs.innerHTML = n ? conv.messages.map((m, i) => m.role === 'user' ? `<div class="u-msg">${esc(m.text)}</div>${prodChip(m.product || conv.messages[i + 1]?.reply?.contextProduct)}` : replyHtml(m.reply, { meta, last: i === n - 1, prevText: conv.messages[i - 1]?.text || '' })).join('') + (conv.staleNotice ? `<p class="warn">${esc(conv.staleNotice)}</p>` : '')
      : introHtml();
    paintSide(); paintQuick();
    const lastR = conv.messages.at(-1)?.reply; if (lastR) addVoices(msgs, lastR, alive);
  };
  // land on the start of the newest reply (scroll-margin keeps it below the sticky header)
  const scrollEnd = () => { const last = [...msgs.querySelectorAll('.a-msg, .send-err, .typing, .thinking')].pop() || msgs.lastElementChild; last?.scrollIntoView({ block: 'start', behavior: 'smooth' }); };
  let busy = false;
  // the live thinking panel: events are released one by one so a burst still reads as steps happening
  const thinking = () => {
    const box = document.createElement('div');
    box.className = 'a-msg thinking'; box.setAttribute('aria-live', 'polite');
    box.innerHTML = `<div class="th-head">${logo('logo-s')}<span>正在思考</span><span class="num th-time">0.0 秒</span></div><ol class="th-list"></ol>`;
    msgs.append(box);
    const list = box.querySelector('.th-list'), clock = box.querySelector('.th-time'), t0 = performance.now();
    const tick = setInterval(() => { clock.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} 秒`; }, 100);
    const queue = []; let timer = null;
    const drain = () => {
      const ev = queue.shift();
      if (!ev) { timer = null; return; }
      const old = list.querySelector(`[data-step="${CSS.escape(ev.id)}"]`);
      const html = stepLi(ev, true);
      if (old) old.outerHTML = html; else list.insertAdjacentHTML('beforeend', html);
      list.lastElementChild?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      timer = setTimeout(drain, queue.length > 6 ? 80 : 180);
    };
    return {
      push: ev => { queue.push(ev); if (!timer) drain(); },
      settle: () => new Promise(res => { const wait = () => (timer || queue.length ? setTimeout(wait, 60) : res()); wait(); }),
      stop: () => { clearInterval(tick); box.remove(); },
    };
  };
  const send = async (text, product = null) => {
    text = text.trim(); if (!text || busy) return;
    show('chat');
    busy = true; sendBtn.disabled = true; ta.value = ''; ta.rows = 1;
    conv.messages.push({ role: 'user', text, product });
    paint();
    const th = thinking(); scrollEnd();
    try {
      const body = { conversationId: conv.id, text, productId: product?.id };
      const r = window.ReadableStream ? await stream('/api/chat/stream', body, ev => ev.type === 'step' && th.push(ev)) : await api('/api/chat', { method: 'POST', body });
      await th.settle();
      conv.messages.push({ role: 'assistant', reply: r.reply }); conv.plan = r.plan; conv.staleNotice = null;
    } catch (e) {
      th.stop();
      conv.messages.pop(); ta.value = text; busy = false; sendBtn.disabled = false;
      paint();
      msgs.insertAdjacentHTML('beforeend', `<div class="u-msg">${esc(text)}</div><div class="send-err" role="alert"><span>没发出去：${esc(e.message)}</span><button data-resend>重新发送</button></div>`);
      scrollEnd(); return;
    }
    th.stop();
    busy = false; sendBtn.disabled = !ta.value.trim();
    paint(); scrollEnd();
  };
  const planOp = async op => {
    const r = await api('/api/plan/op', { method: 'POST', body: { conversationId: conv.id, ...op } });
    conv.plan = r.plan; if (r.reply.status === 'clarify' && r.reply.message) toast(r.reply.message);
    paintSide(); paintQuick(); return r;
  };
  paint();
  requestAnimationFrame(() => msgs.lastElementChild?.scrollIntoView({ block: 'end' }));
  ta.addEventListener('input', () => { sendBtn.disabled = !ta.value.trim() || busy; ta.rows = Math.min(5, ta.value.split('\n').length); });
  ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(ta.value); } });
  form.addEventListener('submit', e => { e.preventDefault(); send(ta.value); });
  view.addEventListener('click', async e => {
    const t = e.target.closest('button, a');
    if (!t || side.contains(t)) return;
    if (t.matches('[data-resend]')) return send(ta.value);
    if (t.matches('[data-say]')) return send(t.dataset.say);
    if (t.matches('[data-fill]')) { ta.value = t.dataset.fill; ta.focus(); ta.setSelectionRange(3, 3); sendBtn.disabled = false; return; }
    if (t.matches('[data-pane-btn]')) return show(t.dataset.paneBtn);
    if (t.matches('[data-goods]')) return show('goods', { cat: t.dataset.goods });
    if (t.matches('[data-posts]')) return show('posts', { cat: t.dataset.posts });
    if (t.matches('[data-guide]')) return show('guide', { cat: t.dataset.guide, point: t.dataset.point });
    if (t.matches('[data-topic]')) { panes.guide.querySelectorAll('[data-topic]').forEach(b => b.setAttribute('aria-selected', String(b === t))); return panes.guide.querySelector(`#topic-${t.dataset.topic}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' }); }
    if (t.matches('[data-copy]')) { try { await navigator.clipboard.writeText(t.closest('.a-msg').innerText.trim()); toast('已复制'); } catch { toast('复制失败，长按文字手动复制'); } return; }
    if (t.matches('[data-fav]')) { await api(`/api/favorites/${encodeURIComponent(t.dataset.fav)}`, { method: 'POST' }); t.setAttribute('aria-pressed', 'true'); return toast('已收藏'); }
    if (t.matches('[data-add]')) { const r = await planOp({ op: 'add', productId: t.dataset.add }); t.outerHTML = '<span class="tag">已在购物车</span>'; return toast(r.plan.state.budgetCents ? '已加入购物车，并按总预算重新核算' : '已加入购物车'); }
    if (t.matches('[data-pick-alt]')) { await planOp({ op: 'pick', itemId: t.dataset.item, productId: t.dataset.pickAlt }); return toast('已换成这款，购物车合计已重新核算'); }
    if (t.matches('[data-new]')) { const r = await api('/api/conversations', { method: 'POST' }); go(`/chat/${r.id}`); }
  });
  bindPlan(side, () => conv.plan, planOp);
  if (q) { history.replaceState({}, '', id ? `/chat/${id}` : '/chat'); send(q, qp ? await api(`/api/products/${encodeURIComponent(qp)}`).then(d => d.product).catch(() => null) : null); }
}
