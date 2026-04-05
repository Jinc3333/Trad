#!/usr/bin/env python3
"""
Trading Dashboard Server
실행: python server.py
접속: http://localhost:8888
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, sys, urllib.parse
import requests

PORT = 8888
DIR = os.path.dirname(os.path.abspath(__file__))

# ── 외부 데이터 fetch ─────────────────────────────────────────────────────────
YF_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36',
    'Accept': 'application/json',
}

def fetch_yahoo(symbol, range_='2y', interval='1d'):
    sym = urllib.parse.quote(symbol)
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
           f'?interval={interval}&range={range_}&includePrePost=false')
    r = requests.get(url, headers=YF_HEADERS, timeout=15)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    times = res['timestamp']
    q = res['indicators']['quote'][0]
    out = []
    for i, t in enumerate(times):
        o, h, l, c = q['open'][i], q['high'][i], q['low'][i], q['close'][i]
        v = (q.get('volume') or [0]*len(times))[i]
        if all(x is not None for x in [o, h, l, c]):
            out.append({'time': t, 'open': round(o, 4), 'high': round(h, 4),
                        'low': round(l, 4), 'close': round(c, 4), 'volume': int(v or 0)})
    return out

def fetch_binance(symbol='BTCUSDT', interval='4h', limit=300):
    url = (f'https://api.binance.com/api/v3/klines'
           f'?symbol={symbol}&interval={interval}&limit={limit}')
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return [{'time': int(k[0])//1000, 'open': float(k[1]), 'high': float(k[2]),
              'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])}
            for k in r.json()]

# ── HTTP 핸들러 ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            p = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if p in ('/', '/dashboard.html'):
                self._file('dashboard.html', 'text/html; charset=utf-8')
            elif p == '/api/qld':
                rng = qs.get('range', ['2y'])[0]
                self._json(fetch_yahoo('QLD', rng, '1d'))
            elif p == '/api/ndx':
                rng = qs.get('range', ['1y'])[0]
                self._json(fetch_yahoo('^NDX', rng, '1d'))
            elif p == '/api/btc':
                self._json(fetch_binance('BTCUSDT', '4h', 300))
            elif p == '/api/seasonality':
                self._file('seasonality_data.json', 'application/json')
            else:
                self.send_error(404)
        except Exception as e:
            print(f'[ERR] {e}')
            self.send_error(500, str(e)[:200])

    def _file(self, name, ctype):
        path = os.path.join(DIR, name)
        with open(path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f'  {self.address_string()} {fmt % args}')

# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    json_path = os.path.join(DIR, 'seasonality_data.json')
    if not os.path.exists(json_path):
        print('seasonality_data.json 생성 중...')
        import subprocess
        subprocess.run([sys.executable, os.path.join(DIR, 'gen_seasonality_json.py')], check=True)

    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'🚀  http://localhost:{PORT}  (Ctrl+C 종료)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n서버 종료')
