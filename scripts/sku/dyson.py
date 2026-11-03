"""戴森官网 dyson.cn (Magento, server-rendered): crawl product links from public category pages,
parse each product page (h1 title, data-price-amount 建议零售价, stock label, short description bullets)."""
import html, re
from common import fetch, cents, dump, now, ROOT, UA

BASE = 'https://www.dyson.cn'
SOURCE = {'name': '戴森官网', 'homepage': BASE + '/'}
SEEDS = ['/', '/product/hair-care.html', '/product/vacuum-cleaner.html', '/product/air-treatment.html']
RULES = [  # (category, include, exclude) on the page title
    ('hair-dryer', r'吹风机', r'造型器|直发|卷发|配件|吸嘴|收纳|支架'),
    ('floor-cleaning', r'吸尘器|洗地机|除螨', r'配件|吸头|滤网|电池|收纳|支架|刷头'),
    ('portable-fan', r'HushJet|Mini Cool', r'配件|净化'),  # 官网导航把 HushJet Mini Cool 归在「便携风扇」
    ('desk-fan', r'风扇|凉风', r'净化|配件|滤网'),
    ('desk-lamp', r'台灯|灯', r'配件|灯泡'),
    ('earphones', r'耳机', r'配件|耳罩|耳垫'),
]


def save_img(pid, url):
    """www-img.dyson.cn returns 403 without a dyson.cn Referer -> keep a local copy under public/img."""
    import os, urllib.request, time
    ext = os.path.splitext(url.split('?')[0])[1] or '.jpg'
    rel = f'/img/{pid}{ext}'
    path = os.path.join(ROOT, 'public', rel.lstrip('/'))
    if not os.path.exists(path):
        time.sleep(1.1)
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': BASE + '/'})
        try:
            data = urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            print('img fail', url, e); return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(data)
    return rel


def text(s):
    s = re.sub(r'<sup>.*?</sup>', '', s, flags=re.S)  # footnote markers
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', s))).strip()


def products():
    links = set()
    for p in SEEDS:
        links |= set(re.findall(r'https://www\.dyson\.cn/(?:dyson-)?[a-z0-9-]*\d{6}-\d{2}[a-z0-9-]*\.html|https://www\.dyson\.cn/dyson-[a-z0-9-]+\.html', fetch(BASE + p)))
    out, seen = [], set()
    for url in sorted(links):
        try:
            s = fetch(url)
        except Exception as e:
            print('skip', url, e); continue
        h1 = re.search(r'introduce-title"><h1>(.*?)</h1>', s, re.S) or re.search(r'<title>(.*?)</title>', s, re.S)
        price = re.search(r'data-price-amount="([\d.]+)"', s)
        sku = re.search(r'itemprop="sku">([^<]+)<', s)
        if not (h1 and price and sku): continue
        title = text(h1.group(1))
        cat = next((c for c, inc, exc in RULES if re.search(inc, title) and not re.search(exc, title)), None)
        if not cat or sku.group(1) in seen: continue
        seen.add(sku.group(1))
        stock = re.search(r'class="stock (available|unavailable)"', s)
        sd = re.search(r'class="shortdes-a">(.*?)</div>', s, re.S)
        hl = [text(x) for x in re.findall(r'<p>(.*?)</p>', sd.group(1), re.S)] if sd else []
        hl = [h.strip('，,。 ') for h in hl if h]
        # product's own image from Magento price-box JSON ("images":[{"url":..}]); page <img> tags include nav thumbnails of other products
        m = re.search(r'"images":\[\{"url":"(https:[^"]+)"', s)
        imgs = [save_img('dyson-' + sku.group(1).strip(), m.group(1).replace('\\/', '/'))] if m else []
        imgs = [u for u in imgs if u]
        out.append({'id': 'dyson-' + sku.group(1).strip(), 'source': SOURCE['name'], 'sourceSku': sku.group(1).strip(), 'url': url,
                    'title': title, 'brand': '戴森', 'model': (re.search(r'\b([A-Z]{1,2}\d{2}[A-Za-z]*|V\d{1,2}[A-Za-z ]*?)\b', title) or [None])[0],
                    'category': cat, 'priceCents': cents(price.group(1)), 'listPriceCents': None, 'shippingCents': None, 'shippingNote': None,
                    'stock': {'available': 'in_stock', 'unavailable': 'out_of_stock'}[stock.group(1)] if stock else None,
                    'images': imgs, 'specs': {}, 'facts': {}, 'highlights': hl[:6], 'observedAt': now()})
    # one SKU per (model without color suffix, price); prefer in-stock color
    best = {}
    for p in out:
        k = (re.sub(r'\s*[（(][^）)]*[）)]?\s*$|\s+红色$', '', p['title']), p['priceCents'])
        if k not in best or (p['stock'] == 'in_stock' and best[k]['stock'] != 'in_stock'):
            best[k] = p
    return list(best.values())


if __name__ == '__main__':
    ps = products()
    dump('parts/dyson.json', {'source': dict(SOURCE, fetchedAt=now(), shippingPolicy=None), 'products': ps})
    for p in ps: print(p['category'], p['priceCents'] // 100, p['stock'], p['title'], p['highlights'][:2])
