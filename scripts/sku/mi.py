"""小米商城: curated product_ids (picked from api2.order.mi.com/search/index results, accessories excluded)
-> api2.order.mi.com/product/view (public JSON, no login). One SKU per distinct price."""
import json, re
from common import fetch, cents, dump, now

PICKS = {
 'portable-fan': [10050224, 15947],
 'desk-fan': [21039, 19733, 12010, 18219, 18222, 23259, 15863, 19989, 14522, 18463],
 'earphones': [23261, 23260, 21120, 23966, 25431, 22137, 25802, 24645, 22419, 20978, 21418, 21588, 25300, 24647, 20534],
 'keyboard-mouse': [21270, 20195, 20064, 21774, 20063, 21938, 24998, 20443, 24632, 21272, 21630, 20861],
 'charging': [15554, 18499, 20395, 20878, 21468, 20867, 22179, 21895, 22272, 25813, 25801, 21973, 12896, 20003, 16720, 18290, 22305, 19856, 19331, 20188, 22000],
 'electric-toothbrush': [18289, 20448, 25646, 13857, 20799, 21850, 21699, 24583, 23581],
 'hair-dryer': [16640, 12639, 16725, 21680, 19151, 18520, 25789, 21049, 21384, 10050230],
 'desk-lamp': [19927, 18377, 16018, 19858, 25463, 20565, 25627, 20324, 25060, 15186],
 'luggage': [18580, 18577, 21401, 20024, 19718, 20123, 20018, 19235, 12307, 25748],
 'smartwatch': [25768, 25306, 25307, 24629, 25161, 22153, 21405, 23408, 25419, 22418],
 'floor-cleaning': [12498, 20916, 22332, 16594, 20336, 24553, 24615, 24787, 24789, 23998, 18649, 10050129, 25228, 10050176, 10050174, 10050243, 25000],
 'kitchen-appliance': [20995, 25657, 24900, 25497, 19147, 21269, 21465, 21677, 20071, 20293, 20393],
 'phone': [20866, 24884, 24940, 25687, 20425, 22526, 24814, 21931, 21932, 22422],
 'tablet': [24235, 21591, 19820, 20593, 21872, 25446, 25445, 21130],
}
MAX_VARIANTS = 3
SOURCE = {'name': '小米商城', 'homepage': 'https://www.mi.com/shop'}


def brand(name):
    if re.match(r'(?i)redmi', name): return 'REDMI'
    if re.match(r'[米⽶]家', name): return '米家'
    return '小米'


def products():
    out, freight = [], None
    for cat, pids in PICKS.items():
        for pid in pids:
            url = f'https://api2.order.mi.com/product/view?product_id={pid}&version=2'
            d = json.loads(fetch(url, referer='https://www.mi.com/'))
            if d.get('code') != 200 or not d.get('data'):
                print('skip', pid, d.get('msg')); continue
            d = d['data']; t = now()
            info = d['product_info']
            by_price = {}  # one SKU per distinct price; prefer an in-stock variant (colors at same price collapse)
            for g in d['goods_list']:
                p = cents(g['goods_info'].get('price'))
                if p and (p not in by_price or (g['goods_info'].get('is_stock') and not by_price[p]['goods_info'].get('is_stock'))):
                    by_price[p] = g
            for p, g in list(by_price.items())[:MAX_VARIANTS]:
                gi = g['goods_info']
                for x in (g.get('after_sale_info') or {}).get('list') or []:
                    if x.get('mark') == 'freight' and not freight: freight = x['desc']
                specs = {x['name']: x['value'] for x in (gi.get('class_parameters') or {}).get('list') or [] if x.get('name') and x.get('value') and x['value'].strip() not in ('-', '/')}
                imgs = [x['img_url'] for x in gi.get('gallery_v3') or [] if x.get('type') == 'img'] or gi.get('imgs') or []
                imgs = ['https:' + u if u.startswith('//') else u for u in imgs][:5]
                mp = cents(gi.get('market_price'))
                title = gi['name'].strip()
                out.append({
                    'id': f"mi-{gi['goods_id']}", 'source': SOURCE['name'], 'sourceSku': str(gi['goods_id']),
                    'url': f'https://www.mi.com/shop/buy/detail?product_id={pid}',
                    'title': title, 'brand': brand(title), 'model': None, 'category': cat,
                    'priceCents': p, 'listPriceCents': mp if mp and mp > p else None,
                    'shippingCents': None, 'shippingNote': '小米发货商品单笔满69元免运费（页面运费说明）',
                    'stock': 'in_stock' if gi.get('is_stock') is True else 'out_of_stock' if gi.get('is_stock') is False else None,
                    'images': imgs, 'specs': specs, 'facts': {},
                    'highlights': [h.strip() for h in re.split(r'[丨|｜]', info.get('product_desc') or '') if len(h.strip()) >= 3][:6],
                    'observedAt': t,
                })
    return out, freight


if __name__ == '__main__':
    ps, freight = products()
    src = dict(SOURCE, fetchedAt=now(), shippingPolicy={
        'freeOverCents': 6900, 'feeCents': None,
        'evidenceUrl': 'https://api2.order.mi.com/product/view?product_id=19781&version=2 (商品页「运费说明」)',
        'evidenceText': (freight or '').split(';')[0]})
    dump('parts/mi.json', {'source': src, 'products': ps})
    print(len(ps), 'mi products')
