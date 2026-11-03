"""Merge data/parts/*.json -> data/catalog.json. Re-derives facts, strips promo copy."""
import glob, json, os, re, time, urllib.parse, urllib.request
from common import ROOT, UA, RAW, dump, now
from facts import extract

CATEGORIES = [
 ('portable-fan', '便携/手持风扇', ['小风扇', '手持风扇', '便携风扇', '迷你风扇', 'USB风扇', '挂脖风扇', '随身风扇', '桌面移动风扇'], ['续航', '风速档位', '电池容量', '重量']),
 ('desk-fan', '桌面/家用风扇', ['台扇', '桌面风扇', '落地扇', '循环扇', '电风扇', '塔扇', '冷风扇'], ['风速档位', '噪音', '功率', '适用面积']),
 ('earphones', '蓝牙/降噪耳机', ['耳机', '蓝牙耳机', '无线耳机', '降噪耳机', '真无线', 'TWS', '头戴耳机', '耳麦', '开放式耳机'], ['降噪', '续航', '蓝牙版本', '佩戴方式']),
 ('keyboard-mouse', '键盘/鼠标', ['键盘', '鼠标', '键鼠套装', '无线鼠标', '机械键盘', '蓝牙键盘'], ['连接方式', 'DPI', '续航', '按键']),
 ('charging', '充电宝/充电器', ['充电宝', '移动电源', '充电器', '快充头', '氮化镓', 'GaN', '无线充', '充电头'], ['电池容量', '输出功率', '接口', '重量']),
 ('electric-toothbrush', '电动牙刷', ['电动牙刷', '声波牙刷', '牙刷', '儿童电动牙刷'], ['刷头', '清洁模式', '续航', '防水']),
 ('hair-dryer', '吹风机', ['吹风机', '电吹风', '风筒', '高速吹风机', '负离子吹风机'], ['功率', '风速', '负离子', '重量']),
 ('desk-lamp', '台灯', ['台灯', '护眼灯', '学习灯', '阅读灯', '显示器挂灯', '屏幕挂灯'], ['照度', '色温', '显色指数', '功率']),
 ('luggage', '行李箱', ['行李箱', '旅行箱', '拉杆箱', '登机箱', '箱子'], ['尺寸', '容量', '材质', '重量']),
 ('smartwatch', '智能手表/手环', ['手表', '智能手表', '手环', '运动手环', '智能手环'], ['屏幕尺寸', '续航', '防水', 'NFC']),
 ('floor-cleaning', '扫地机/吸尘器', ['扫地机', '扫地机器人', '扫拖机器人', '吸尘器', '无线吸尘器', '洗地机', '手持吸尘器'], ['吸力', '续航', '尘盒容量', '噪音']),
 ('kitchen-appliance', '咖啡机/小厨电', ['咖啡机', '胶囊咖啡机', '电水壶', '烧水壶', '空气炸锅', '小厨电', '养生壶'], ['容量', '功率', '材质']),
 ('phone', '手机', ['手机', '智能手机', '5G手机'], ['处理器', '屏幕尺寸', '电池容量', '存储']),
 ('tablet', '平板', ['平板', '平板电脑', 'iPad', 'Pad'], ['屏幕尺寸', '处理器', '存储', '电池容量']),
]
PROMO = re.compile(r'直降\S*|立减\S*|券后\S*|历史低价|到手价\S*|限时\S*|领券\S*|划线价|优惠\d+元|省\d+元|\d+期免息|以旧换新\S*|补贴\S*|国补\S*')


def clean(s):
    return re.sub(r'\s+', ' ', PROMO.sub('', s or '')).strip(' ,，|丨')


_last = {}


def image_ok(url, cache):
    """HEAD each cover candidate once (<=1 req/s per host); local /img/ paths must exist on disk."""
    if url.startswith('/'):
        return os.path.exists(os.path.join(ROOT, 'public', url.lstrip('/')))
    if url not in cache:
        host = urllib.parse.urlparse(url).netloc
        time.sleep(max(0, 1.1 - (time.time() - _last.get(host, 0))))
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD', headers={'User-Agent': UA}), timeout=15)
            cache[url] = r.status == 200 and (r.headers.get('Content-Type') or '').startswith('image/')
        except Exception:
            cache[url] = False
        _last[host] = time.time()
    return cache[url]


def main():
    sources, products = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'parts', '*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        sources.append(d['source'])
        products += d['products']
    cats = {c[0] for c in CATEGORIES}
    out, seen, evidence = [], set(), {}
    cache_path = os.path.join(RAW, 'image-check.json')  # delete to re-check
    img_cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for p in products:
        assert p['category'] in cats, p['category']
        if p['id'] in seen: continue
        seen.add(p['id'])
        p['title'] = clean(p['title'])
        p['highlights'] = [h for h in (clean(h) for h in p['highlights']) if 3 <= len(h) <= 40]
        p['listPriceCents'] = None  # ponytail: struck-through prices deliberately not shipped (no display of 划线价)
        while p['images'] and not image_ok(p['images'][0], img_cache):
            print('drop dead image', p['id'], p['images'].pop(0))
        p['facts'], ev = extract(p['title'], p['specs'], p['category'], p['highlights'])
        if ev: evidence[p['id']] = ev
        out.append(p)
    dump('catalog.json', {
        'generatedAt': now(), 'sources': sources,
        'categories': [{'id': i, 'name': n, 'aliases': a, 'specKeys': k} for i, n, a, k in CATEGORIES],
        'products': out})
    json.dump(img_cache, open(cache_path, 'w'), indent=0)
    dump('raw/facts-evidence.json', evidence)  # each fact -> the exact page text it came from
    print(len(out), 'products')


if __name__ == '__main__':
    main()
