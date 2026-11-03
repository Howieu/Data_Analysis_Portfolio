# Collect real user experience notes per category from Xiaohongshu via opencli (logged-in browser session).
# Usage: python3 scripts/posts/xhs_posts.py data/posts.json   (needs `opencli` logged in to xiaohongshu in the local browser)
# Only title, author, counts, tags, a <=50-char excerpt and the link are kept; the app links back to the original note.
import json, re, subprocess, sys, time, yaml
QUERIES = {'portable-fan': '手持小风扇 使用体验', 'desk-fan': '电风扇 循环扇 使用体验', 'earphones': '降噪耳机 真实使用体验', 'keyboard-mouse': '键盘鼠标 使用体验',
  'charging': '充电宝 使用体验', 'electric-toothbrush': '电动牙刷 真实使用体验', 'hair-dryer': '吹风机 真实使用体验', 'desk-lamp': '护眼台灯 使用体验',
  'luggage': '行李箱 真实使用体验', 'smartwatch': '智能手表 手环 使用体验', 'floor-cleaning': '扫地机器人 吸尘器 使用体验', 'kitchen-appliance': '小厨电 咖啡机 使用体验',
  'phone': '手机 长期使用体验', 'tablet': '平板 使用体验'}
PER = 5
def cli(*a):
    r = subprocess.run(['opencli', 'xiaohongshu', *a, '-f', 'yaml'], capture_output=True, text=True, timeout=120)
    return yaml.safe_load(r.stdout.split('\n\n  Update available')[0]) or []
def excerpt(t):
    t = re.sub(r'#\S+', '', t); t = re.sub(r'[\U0001F000-\U0001FFFF☀-➿️⸻]', '', t); t = re.sub(r'\s+', ' ', t).strip()
    return t[:50] + ('…' if len(t) > 50 else '')
num = lambda v: int(float(str(v).replace('万', '')) * (10000 if '万' in str(v) else 1)) if re.match(r'^[\d.]+万?$', str(v or '')) else 0
out = {}
for cat, q in QUERIES.items():
    try: res = cli('search', q)
    except Exception as e: print('search fail', cat, e, file=sys.stderr); continue
    res = sorted([r for r in res if r.get('url') and r.get('title')], key=lambda r: -num(r.get('likes')))[:PER]
    posts = []
    for r in res:
        time.sleep(3)
        try: d = {x['field']: x['value'] for x in cli('note', r['url'])}
        except Exception as e: print('note fail', r['url'], e, file=sys.stderr); continue
        if not d.get('content'): continue
        nid = re.search(r'/([0-9a-f]{24})', r['url']).group(1)
        tok = re.search(r'xsec_token=([^&]+)', r['url'])
        posts.append({'id': nid, 'title': d.get('title') or r['title'], 'author': d.get('author') or r.get('author'), 'likes': num(d.get('likes') or r.get('likes')), 'collects': num(d.get('collects')),
          'excerpt': excerpt(d['content']), 'tags': [t.strip().lstrip('#') for t in (d.get('tags') or '').split(',') if t.strip()][:3], 'publishedAt': r.get('published_at'),
          'url': f"https://www.xiaohongshu.com/explore/{nid}" + (f"?xsec_token={tok.group(1)}&xsec_source=pc_search" if tok else '')})
    out[cat] = posts; print(cat, len(posts), flush=True)
    time.sleep(4)
json.dump({'source': '小红书', 'collectedAt': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'posts': out}, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
