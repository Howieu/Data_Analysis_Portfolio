// All money is integer cents (CNY). Never floats.
export const yuanText = cents => {
  if (!Number.isSafeInteger(cents)) return '—';
  const y = Math.floor(cents / 100), f = cents % 100;
  return y.toLocaleString('zh-CN') + (f ? '.' + String(f).padStart(2, '0').replace(/0$/, '') : '');
};
export const price = cents => '¥' + yuanText(cents);

const DIGIT = { 零: 0, 〇: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
const UNIT = { 十: 10, 百: 100, 千: 1000, 万: 10000 };

// Chinese integer like 八百, 一千二, 两千五百, 一万, 十五. Returns null when not a clean number.
export function cnInt(s) {
  if (!s || !/^[零〇一二两三四五六七八九十百千万]+$/.test(s)) return null;
  let total = 0, section = 0, num = 0, lastUnit = 0, sawDigit = false;
  for (const ch of s) {
    if (ch in DIGIT) { num = DIGIT[ch]; sawDigit = true; continue; }
    const u = UNIT[ch];
    if (u === 10000) { section = (section + num) * u; total += section; section = 0; num = 0; lastUnit = u; continue; }
    if (!sawDigit && u === 10) num = 1; // 十五
    section += num * u; num = 0; lastUnit = u; sawDigit = false;
  }
  // 一千二 → 1200, 两万五 → 25000 (trailing digit takes next lower unit)
  if (num && lastUnit >= 100) num *= lastUnit / 10;
  return total + section + num;
}

// Parse a money token into cents. Accepts "1299", "1,299", "1299.5", "1.5k", "2w", "1.2万", "八百", "一千二".
// Returns {cents} or {error}. Rejects 0, negatives, >2 decimals.
export function moneyToken(raw) {
  let t = String(raw).trim().replace(/[,，](?=\d{3}\b)/g, '').replace(/\s+/g, '');
  const neg = /^[-−]/.test(t);
  if (neg) return { error: 'negative' };
  const mixed = t.match(/^(\d+)([千万])(\d)$/); // 1千5 → 1500, 2万3 → 23000
  if (mixed) { const u = mixed[2] === '千' ? 1000 : 10000; return { cents: (Number(mixed[1]) * u + Number(mixed[3]) * u / 10) * 100 }; }
  let mult = 1;
  const m = t.match(/^(.*?)(k|K|千|w|W|万)$/);
  if (m && m[1]) { mult = { k: 1000, K: 1000, 千: 1000, w: 10000, W: 10000, 万: 10000 }[m[2]]; t = m[1]; }
  let cents = null;
  if (/^\d+(\.\d+)?$/.test(t)) {
    const [w, f = ''] = t.split('.');
    if (mult === 1 && f.length > 2) return { error: 'precision' };
    // scale with multiplier using integer math
    const scaled = BigInt(w + f) * BigInt(mult) * 100n / 10n ** BigInt(f.length);
    if (scaled > BigInt(Number.MAX_SAFE_INTEGER)) return { error: 'too_large' };
    cents = Number(scaled);
  } else {
    const n = cnInt(t + (mult === 1000 && m?.[2] === '千' ? '千' : mult === 10000 && m?.[2] === '万' ? '万' : ''));
    if (n === null) return { error: 'unreadable' };
    cents = n * 100;
  }
  if (!Number.isSafeInteger(cents)) return { error: 'too_large' };
  if (cents <= 0) return { error: 'zero' };
  return { cents };
}
