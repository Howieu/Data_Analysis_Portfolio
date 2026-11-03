"""Apple 中国官网 (apple.com.cn/shop): accessory listing pages (categoryResults JSON) and model buy pages
("products" JSON with per-config fullPrice). Public HTML, no login."""
import html, json, re
from common import fetch, cents, dump, now

BASE = 'https://www.apple.com.cn'
SOURCE = {'name': 'Apple 官网', 'homepage': BASE + '/shop'}
LISTINGS = {  # listing path -> (category, include regex, exclude regex)
 '/shop/accessories/all/headphones-speakers': ('earphones', r'AirPods|EarPods|Beats (Solo|Studio|Flex)|Powerbeats', r'保护壳|耳塞|耳垫|耳罩|清洁|发射器|Pill|HomePod|Beats 360$'),
 '/shop/accessories/all/mice-keyboards': ('keyboard-mouse', r'键盘|鼠标', r'妙控板|保护壳|Folio|双面夹'),
 '/shop/accessories/all/chargers-adapters': ('charging', r'电源适配器|充电器|移动电源|外置电池|充电宝', r'线缆|连接线|转换器|保护壳|延长线'),
}
MODELS = {  # buy page -> category
 '/shop/buy-ipad/ipad': 'tablet', '/shop/buy-ipad/ipad-mini': 'tablet', '/shop/buy-ipad/ipad-air': 'tablet', '/shop/buy-ipad/ipad-pro': 'tablet',
 '/shop/buy-iphone/iphone-17e': 'phone', '/shop/buy-iphone/iphone-17': 'phone', '/shop/buy-iphone/iphone-air': 'phone', '/shop/buy-iphone/iphone-18-pro': 'phone',
 '/shop/buy-watch/apple-watch-se': 'smartwatch', '/shop/buy-watch/apple-watch': 'smartwatch', '/shop/buy-watch/apple-watch-ultra': 'smartwatch',
}
MAX_VARIANTS = 3


def jvar(s, key):
    i = s.find(key)
    return None if i < 0 else json.JSONDecoder().raw_decode(s[i + len(key):].lstrip())[0]


def base(pid, sku, url, title, cat, price, imgs, specs, t):
    return {'id': f'apple-{pid}', 'source': SOURCE['name'], 'sourceSku': sku, 'url': url, 'title': title,
            'brand': 'Beats' if 'beats' in title.lower() else 'Logitech' if 'Logitech' in title else 'Apple',
            'model': sku, 'category': cat, 'priceCents': price, 'listPriceCents': None, 'shippingCents': None,
            'shippingNote': 'Apple 官网线上订单可享免费标准送货（帮助页「送货与取货」）', 'stock': None,
            'images': imgs, 'specs': specs, 'facts': {}, 'highlights': [], 'observedAt': t}


def products():
    out = []
    for path, (cat, inc, exc) in LISTINGS.items():
        s = fetch(BASE + path); t = now()
        d = jvar(s, 'window.pageLevelData.categoryResults =')
        if not d: print('no data', path); continue
        for it in d['results']['accessories']['accessories']['tiles']['items']:
            v = it['value']; title = html.unescape(v['title']).replace(' ', ' ').replace('‑', '-')
            if not re.search(inc, title) or re.search(exc, title): continue
            p = cents(re.sub(r'[^\d.]', '', v['productPrice'].get('priceCurrent') or ''))
            if not p: continue
            imgs = [x['value']['sources'][0]['srcSet'] for x in v.get('productImages', {}).get('items', []) if x['value'].get('sources')][:4]
            url = BASE + v['link']['url'].split('?')[0]
            out.append(base(v['partNumber'].replace('/', '-').lower(), v['partNumber'], url, title, cat, p, imgs, {}, t))
    for path, cat in MODELS.items():
        try:
            s = fetch(BASE + path)
        except Exception as e:
            print('skip', path, e); continue
        t = now()
        prods = sorted(jvar(s, '"products":') or [], key=lambda x: (x.get('price') or {}).get('fullPrice') or 0)
        if cat == 'smartwatch':  # watch pages expose case-config prices ("起") in productSelectionData instead
            name0 = {'apple-watch-se': 'Apple Watch SE', 'apple-watch': 'Apple Watch', 'apple-watch-ultra': 'Apple Watch Ultra'}[path.rsplit('/', 1)[1]]
            pr = (jvar(s, 'productSelectionData:') or {}).get('displayValues', {}).get('prices', {})
            prods = sorted(({'partNumber': k, 'name': name0 + ' ' + k.replace('watch_cases-', '').replace('-gpscell', ' GPS+蜂窝网络').replace('-gps', ' GPS').replace('-', ' ').replace('aluminum', '铝金属').replace('titanium', '钛金属'),
                             'price': {'fullPrice': v.get('amount')}, 'from': True} for k, v in pr.items() if isinstance(v, dict) and v.get('amount')),
                           key=lambda x: x['price']['fullPrice'])
        og = re.search(r'og:image" content="([^"]+)', s)
        img = [html.unescape(og.group(1))] if og else []
        h1 = re.search(r'<title>(?:购买\s*)?([^<-]+)', s)
        seen = set()
        for x in prods:
            p = cents((x.get('price') or {}).get('fullPrice'))
            if not p or p in seen or len(seen) >= MAX_VARIANTS: continue
            seen.add(p)
            name = x['name']
            specs = {}
            m = re.search(r'(\d+)\s*(GB|TB)', name)
            if m: specs['存储容量'] = m.group(0)
            m = re.search(r'(\d+)\s*mm', name)
            if m: specs['表壳尺寸'] = m.group(0)
            if x.get('from'): specs['价格说明'] = '官网标价为该表壳配置起售价（“起”）'
            if 'Wi-Fi + Cellular' in name: specs['网络'] = 'Wi-Fi + 蜂窝网络'
            elif 'Wi-Fi' in name: specs['网络'] = 'Wi-Fi'
            out.append(base(x['partNumber'].replace('/', '-').lower(), x['partNumber'], BASE + path, name, cat, p, img, specs, t))
    return out


if __name__ == '__main__':
    ps = products()
    dump('parts/apple.json', {'source': dict(SOURCE, fetchedAt=now(), shippingPolicy={
        'freeOverCents': 0, 'feeCents': 0, 'evidenceUrl': BASE + '/shop/help/shipping_delivery',
        'evidenceText': '所有线上订单均可享受免费的标准送货服务。你也可额外付费升级送货方式'}), 'products': ps})
    import collections; print(len(ps), collections.Counter(p['category'] for p in ps))
    for p in ps: print(p['category'], p['priceCents'] // 100, p['title'][:60])
