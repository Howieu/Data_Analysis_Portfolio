// Browser end-to-end run against a local server. Uses the Playwright already installed on this machine (dev tool only).
// node test/e2e.mjs   (E2E_URL defaults to http://127.0.0.1:8790)
import { writeFileSync, mkdirSync } from 'node:fs';
import { BANNED } from './banned.mjs';

const PW = process.env.PLAYWRIGHT || '/Users/vendredi/Documents/pm_project/zhixuan-ai-shopping-guide/node_modules/playwright/index.mjs';
const BROWSER = process.env.CHROMIUM || '/Users/vendredi/Library/Caches/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-mac-arm64/chrome-headless-shell';
const { chromium } = await import(PW);
const BASE = process.env.E2E_URL || 'http://127.0.0.1:8790';
const root = new URL('..', import.meta.url).pathname;
const shots = root + 'screenshots/'; mkdirSync(shots, { recursive: true });
const account = { email: `e2e-${Date.now().toString(36)}@example.test`, password: `Zx${Math.random().toString(36).slice(2, 10)}9` };
writeFileSync(root + 'test/.e2e-account.json', JSON.stringify(account, null, 2));

const results = [];
const ok = (name, cond, detail = '') => { results.push({ name, pass: !!cond, detail }); console.log(`${cond ? '✓' : '✗'} ${name}${detail ? ' — ' + detail : ''}`); };
const browser = await chromium.launch({ executablePath: BROWSER });
const errors = [];

async function scanDom(page, where) {
  const found = await page.evaluate(words => {
    const texts = [document.title, document.body.innerText];
    for (const el of document.querySelectorAll('*')) for (const a of ['title', 'placeholder', 'aria-label', 'alt', 'value']) { const v = el.getAttribute(a); if (v) texts.push(v); }
    const all = texts.join('\n');
    return words.filter(w => all.includes(w));
  }, BANNED);
  ok(`界面无违禁词：${where}`, !found.length, found.join('、'));
}
async function noOverflow(page, where) {
  const w = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
  ok(`无横向溢出：${where}`, w[0] <= w[1] + 1, `${w[0]} / ${w[1]}`);
}
async function settle(page) { await page.waitForLoadState('networkidle').catch(() => {}); await page.waitForTimeout(300); }

for (const [label, vp] of [['390', { width: 390, height: 844 }], ['1440', { width: 1440, height: 900 }]]) {
  const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: 2, locale: 'zh-CN' });
  const page = await ctx.newPage();
  page.on('pageerror', e => errors.push(`${label} pageerror: ${e.message}`));
  page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(`${label} console: ${m.text()}`); });
  const shot = async name => page.screenshot({ path: `${shots}${label}-${name}.png`, fullPage: false });

  await page.goto(BASE + '/'); await settle(page);
  ok(`${label} 首页有商品`, await page.locator('.grid .card').count() >= 4);
  ok(`${label} 首屏可见商品价格`, await page.evaluate(() => { const p = document.querySelector('.grid .card .price'); return p && p.getBoundingClientRect().bottom <= innerHeight; }));
  await scanDom(page, `${label} 首页`); await noOverflow(page, `${label} 首页`); await shot('home');

  await page.goto(BASE + '/categories'); await settle(page); await shot('categories'); await scanDom(page, `${label} 分类`);
  await page.locator('.cat-grid a').first().click(); await settle(page);
  ok(`${label} 分类页有商品`, await page.locator('.grid .card').count() >= 1); await shot('category');

  await page.goto(BASE + '/search?q=耳机'); await settle(page);
  ok(`${label} 搜索有结果`, await page.locator('.grid .card').count() >= 1);
  await page.locator('.grid .card').first().click(); await settle(page);
  ok(`${label} 商品详情有外链`, await page.locator('a[target=_blank][href^="https://"]').count() >= 1);
  await scanDom(page, `${label} 商品详情`); await noOverflow(page, `${label} 商品详情`); await shot('product');

  if (label === '390') {
    // guest builds a plan, then registers: plan must be kept (not merged into anything)
    await page.goto(BASE + '/chat'); await settle(page);
    await page.fill('#say', '耳机和键盘，总共800，安静点'); await page.keyboard.press('Enter');
    await page.waitForSelector('.a-msg .prow'); await settle(page);
    const t = await page.locator('.a-msg').last().innerText();
    ok('导购回复含选购要点与推荐', /这一套可能最适合你/.test(t) && /选购攻略/.test(t), t.slice(0, 60).replace(/\n/g, ' '));
    ok('组合在预算内或明确运费口径', /在预算内|运费以平台为准/.test(t));
    await scanDom(page, '390 对话'); await shot('chat');
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight)); await shot('chat-bottom');

    await page.fill('#say', '键盘改成两把'); await page.keyboard.press('Enter'); await page.waitForFunction(() => document.querySelectorAll('.a-msg:not(.thinking)').length >= 2 && !document.querySelector('.thinking')); await page.waitForTimeout(300);
    await page.goto(BASE + '/plan'); await settle(page);
    const qtys = await page.locator('.item .stepper .num').allInnerTexts();
    ok('改数量只动键盘、耳机保留', qtys.length === 2 && qtys.includes('2') && qtys.includes('1'), qtys.join(','));
    await page.locator('.item').first().locator('[data-lock]').click(); await settle(page);
    ok('锁定后显示已锁定', (await page.locator('.item').first().innerText()).includes('已锁定'));
    await page.locator('[data-view="single"]').first().click(); await settle(page);
    ok('单品查看不删项', (await page.locator('.plan-head').innerText()).includes('购物车 2 件'));
    await page.locator('[data-view="bundle"]').first().click(); await settle(page);
    await scanDom(page, '390 清单'); await noOverflow(page, '390 清单'); await shot('plan');

    await page.goto(BASE + '/register'); await settle(page);
    await page.click('[data-submit]');
    ok('注册表单本地校验', (await page.locator('[data-err="email"]').innerText()).length > 0);
    await page.fill('#email', account.email); await page.fill('#password', account.password); await page.check('input[name=agree]');
    await page.click('[data-submit]'); await page.waitForURL(/\/me$/); await settle(page);
    ok('注册后进入我的', (await page.locator('.me-head').innerText()).includes(account.email));
    await shot('me');
    await page.goto(BASE + '/plan'); await settle(page);
    ok('登录后清单仍在', (await page.locator('.item').count()) === 2);

    await page.goto(BASE + '/me'); await settle(page);
    await page.click('[data-logout]'); await settle(page);
    ok('退出后为游客', (await page.locator('.me-head').innerText()).includes('还没登录'));
    await page.goto(BASE + '/login'); await settle(page);
    await page.fill('#email', account.email); await page.fill('#password', 'wrong-pass-1');
    await page.click('[data-submit]'); await page.waitForTimeout(500);
    ok('错误密码提示', (await page.locator('[data-form-err]').innerText()).includes('不正确'));
    await page.fill('#password', account.password); await page.click('[data-submit]'); await page.waitForURL(/\/me$/); await settle(page);
    ok('重新登录成功', (await page.locator('.me-head').innerText()).includes(account.email));
    await page.goto(BASE + '/history'); await settle(page);
    ok('历史对话可见', (await page.locator('.list a').count()) >= 1); await shot('history');
    await scanDom(page, '390 历史');
    await page.goto(BASE + '/login'); await settle(page); await shot('login');
  } else {
    await page.goto(BASE + '/chat'); await settle(page);
    await page.fill('#say', '通勤用的降噪耳机，500以内'); await page.keyboard.press('Enter');
    await page.waitForSelector('.a-msg .prow'); await settle(page);
    ok('桌面对话与清单双区', await page.locator('.side-plan .item').count() >= 1);
    await scanDom(page, '1440 对话'); await noOverflow(page, '1440 对话'); await shot('chat');
    await page.goto(BASE + '/plan'); await settle(page); await shot('plan');
  }
  await ctx.close();
}
await browser.close();
ok('无页面脚本错误', !errors.length, errors.slice(0, 5).join(' | '));
const failed = results.filter(r => !r.pass);
writeFileSync(root + 'test/e2e-report.md', `# 端到端结果 ${new Date().toISOString()}\n\n${results.map(r => `- ${r.pass ? '✓' : '✗'} ${r.name}${r.detail ? `（${r.detail}）` : ''}`).join('\n')}\n\n通过 ${results.length - failed.length}/${results.length}\n`);
console.log(`\n通过 ${results.length - failed.length}/${results.length}`);
process.exit(failed.length ? 1 : 0);
