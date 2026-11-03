"""Shared fetch helper: honest UA, <=1 req/s per host, raw cache in data/raw/."""
import hashlib, json, os, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = os.path.join(ROOT, 'data', 'raw')
UA = 'zhixuan-sku-research/0.1 (catalog research for a shopping-guide prototype; no login)'
_last = {}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def fetch(url, referer=None, cache=True, timeout=20):
    """Return response text. Cached by URL hash so reruns don't re-hit the site."""
    host = urllib.parse.urlparse(url).netloc
    path = os.path.join(RAW, host, hashlib.sha1(url.encode()).hexdigest()[:16])
    if cache and os.path.exists(path):
        return open(path, encoding='utf-8').read()
    wait = 1.1 - (time.time() - _last.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    headers = {'User-Agent': UA, 'Accept-Language': 'zh-CN,zh;q=0.9'}
    if referer:
        headers['Referer'] = referer
    for attempt in range(3):  # transient TLS/network errors only; HTTP errors raise immediately
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                body = r.read().decode('utf-8', 'replace')
            break
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
        finally:
            _last[host] = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    with open(os.path.join(RAW, host, 'index.tsv'), 'a', encoding='utf-8') as f:
        f.write(f'{os.path.basename(path)}\t{url}\n')
    return body


def jsonp(text):
    text = text.strip()
    return json.loads(text[text.index('(') + 1:text.rindex(')')]) if not text.startswith('{') else json.loads(text)


def cents(yuan):
    """'1099' / '1099.00' / 1099 -> 109900. None/'' -> None."""
    if yuan in (None, ''):
        return None
    from decimal import Decimal
    return int(Decimal(str(yuan).replace(',', '').replace('¥', '').replace('元', '').strip()) * 100)


def dump(name, obj):
    with open(os.path.join(ROOT, 'data', name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    assert cents('1099') == 109900 and cents('19.9') == 1990 and cents('¥1,299.00') == 129900 and cents('') is None
    assert jsonp('cb({"a":1});') == {'a': 1}
    print('ok')
