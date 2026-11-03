"""Typed facts extracted ONLY from text the page states (title, highlights, spec key:value).
Every fact carries its evidence snippet; nothing is inferred."""
import re

NUM = r'(\d+(?:\.\d+)?)'
FANS = {'portable-fan', 'desk-fan'}
ALLOW = {  # fact key -> categories where it may be extracted (others never get it)
    'windSpeed': FANS | {'hair-dryer'},
    'noiseCancelling': {'earphones'},
    'ancDepthDb': {'earphones'},
    'batteryHours': FANS | {'earphones', 'charging', 'smartwatch', 'electric-toothbrush', 'floor-cleaning', 'phone', 'tablet'},
    'batteryMah': FANS | {'earphones', 'charging', 'smartwatch', 'electric-toothbrush', 'floor-cleaning', 'phone', 'tablet'},
    'noiseDb': FANS | {'earphones', 'hair-dryer', 'floor-cleaning', 'keyboard-mouse'},
    'wattage': {'charging', 'hair-dryer', 'kitchen-appliance', 'floor-cleaning'},
    'wireless': FANS | {'earphones', 'keyboard-mouse', 'charging', 'floor-cleaning'},
    'capacityL': {'kitchen-appliance'},
}


def _n(s):
    v = float(s)
    return int(v) if v == int(v) else v


def extract(title, specs, category, highlights=()):
    """Return (facts, evidence). evidence[key] = exact text the value came from."""
    parts = [title, *highlights, *(f'{k}：{v}' for k, v in specs.items())]
    f, ev = {}, {}

    def put(key, val, src):
        if key in ALLOW and category not in ALLOW[key]:
            return
        if key not in f and val is not None:
            f[key], ev[key] = val, src

    for t in parts:
        if m := re.search(NUM + r'\s*mAh', t, re.I):
            put('batteryMah', _n(m.group(1)), t)
        if m := (re.search(NUM + r'\s*(?:小时|h)\S{0,6}?续航', t) or re.search(r'续航[^\d，,。|]{0,8}?' + NUM + r'\s*(?:小时|h\b)', t)):
            put('batteryHours', _n(m.group(1)), t)
        if '风噪' not in t and (m := re.search(NUM + r'\s*m/s', t)):
            put('windSpeed', _n(m.group(1)), t)
        if m := (re.search(r'(?:重量|净重|机身重|仅重|重约|轻至|仅)\S{0,3}?' + NUM + r'\s*(kg|千克|g|克)(?![a-zA-Z])', t)
                 or re.search(NUM + r'\s*(g|克)\s*轻', t)):
            put('weightG', _n(float(m.group(1)) * (1000 if m.group(2) in ('kg', '千克') else 1)), t)
        if m := re.search(NUM + r'\s*dB', t):  # 「42dB 主动降噪」是降噪深度（越大越好），不是噪音值
            put('ancDepthDb' if '降噪' in t else 'noiseDb', _n(m.group(1)), t)
        if (m := re.search(NUM + r'\s*W(?![a-zA-Z\-])', t)):
            put('wattage', _n(m.group(1)), t)
        if (m := re.search(NUM + r'\s*(L|升)(?![a-zA-Z])', t)):
            put('capacityL', _n(m.group(1)), t)
        tw = t if category == 'charging' else re.sub(r'无线充电?盒?', '', t)  # 无线充电 != 无线连接
        if not re.search(r'不支持|不具备|：无$|：否$', t) and re.search(r'无线|无绳|蓝牙|Bluetooth', tw, re.I):
            put('wireless', True, t)
        if re.search(r'主动降噪|ANC|(?<!通话)(?<!通话智能)降噪', t):
            put('noiseCancelling', True, t)
    if category == 'charging' and re.search('充电宝|移动电源', title) and (m := re.search(r'(?<![\d.])(\d{4,5})(?![\d.]|\s*W)', title)):
        put('batteryMah', int(m.group(1)), title)  # 小米商城标题里的容量数字，如「充电宝 20000 165W」
    if category in ('phone', 'tablet') and (m := (re.search(r'(\d+)GB\s*\+\s*(\d+)(GB|TB)', title) or re.search(r'(\d+)GB内存\s*(\d+)(GB|TB)存储', title))):
        put('ramGB', int(m.group(1)), title)
        put('storageGB', int(m.group(2)) * (1024 if m.group(3) == 'TB' else 1), title)
    elif category in ('phone', 'tablet') and (m := re.search(r'(\d+)\s*(GB|TB)', title)):
        put('storageGB', int(m.group(1)) * (1024 if m.group(2) == 'TB' else 1), title)
    if category == 'luggage' and (m := re.search(r'(\d{2})\s*(?:英寸|寸)', title)):
        put('sizeInch', int(m.group(1)), title)
    return f, ev


if __name__ == '__main__':
    f, _ = extract('米家手持风扇', {}, 'portable-fan', ['250m³/h 强劲风量', '40小时超长续航'])
    assert f == {'batteryHours': 40}, f
    assert extract('小米充电宝 20000 165W', {}, 'charging')[0] == {'wattage': 165, 'batteryMah': 20000}
    assert extract('X', {'重量': '1.2kg', '噪音': '≤ 38dB(A)'}, 'desk-fan')[0] == {'weightG': 1200, 'noiseDb': 38}
    assert extract('REDMI 头戴降噪耳机', {}, 'earphones')[0] == {'noiseCancelling': True}
    assert extract('耳机', {}, 'earphones', ['通话降噪'])[0] == {}
    assert extract('Xiaomi Pad 8 黑色 8GB内存 256GB存储', {}, 'tablet')[0] == {'ramGB': 8, 'storageGB': 256}
    assert extract('iPhone 17 256GB Sage', {}, 'phone')[0] == {'storageGB': 256}
    assert extract('X', {}, 'earphones', ['续航长达 30 小时', '蓝牙 5.4'])[0] == {'batteryHours': 30, 'wireless': True}
    assert extract('耳机', {'充电盒无线充电': '不支持'}, 'earphones')[0] == {}
    assert extract('AirPods 5 (配无线充电盒)', {}, 'earphones')[0] == {}
    assert extract('REDMI Buds 8 青春版', {}, 'earphones', ['抗风噪 6m/s'])[0] == {}
    assert extract('台灯', {}, 'desk-lamp', ['功率 12W'])[0] == {}
    assert extract('耳机', {}, 'earphones', ['55dB深度降噪'])[0] == {'ancDepthDb': 55, 'noiseCancelling': True}
    print('ok')
