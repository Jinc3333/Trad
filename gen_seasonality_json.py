"""계절성 데이터 → seasonality_data.json 생성"""
import json, os
import pandas as pd
import numpy as np

PRESIDENTS = [
    ('Reagan',     '1985-01-20', '1989-01-20'),
    ('G.H.W.Bush', '1989-01-20', '1993-01-20'),
    ('Clinton',    '1993-01-20', '1997-01-20'),
    ('Clinton',    '1997-01-20', '2001-01-20'),
    ('G.W.Bush',   '2001-01-20', '2005-01-20'),
    ('G.W.Bush',   '2005-01-20', '2009-01-20'),
    ('Obama',      '2009-01-20', '2013-01-20'),
    ('Obama',      '2013-01-20', '2017-01-20'),
    ('Trump',      '2017-01-20', '2021-01-20'),
    ('Biden',      '2021-01-20', '2025-01-20'),
    ('Trump',      '2025-01-20', '2029-01-20'),
]
PRES_DF = pd.DataFrame(PRESIDENTS, columns=['name','start','end'])
PRES_DF['start'] = pd.to_datetime(PRES_DF['start'])
PRES_DF['end']   = pd.to_datetime(PRES_DF['end'])
TODAY = pd.Timestamp('2026-03-28')

DIR = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(DIR, 'ndx_historical.csv'), index_col='Date', parse_dates=True)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
raw = raw[['Close']].rename(columns={'Close': 'close'}).dropna()
raw = raw[raw.index <= TODAY]

records = []
for _, prow in PRES_DF.iterrows():
    p_start = prow['start']
    for yr_n in range(1, 5):
        yr_s = p_start + pd.DateOffset(years=yr_n - 1)
        yr_e = min(p_start + pd.DateOffset(years=yr_n), TODAY)
        if yr_s >= TODAY:
            break
        seg = raw[(raw.index > yr_s) & (raw.index < yr_e)].copy()
        for mo in range(1, 13):
            m_data = seg[seg.index.month == mo].sort_index()
            if len(m_data) < 5:
                continue
            base = m_data['close'].iloc[0]
            for ts, row in m_data.iterrows():
                records.append({
                    'yr_n': yr_n, 'month': mo,
                    'day': ts.day,
                    'cum': (row['close'] / base - 1) * 100
                })

df = pd.DataFrame(records)
agg = (df.groupby(['yr_n','month','day'])['cum']
         .agg(mean='mean', std='std', n='count')
         .reset_index())
agg['se']   = agg['std'] / np.sqrt(agg['n'])
agg['ci95'] = agg['se'] * 1.96

result = {}
for yr in range(1, 5):
    result[str(yr)] = {}
    for mo in range(1, 13):
        sub = agg[(agg['yr_n']==yr) & (agg['month']==mo)].sort_values('day')
        if sub.empty:
            continue
        result[str(yr)][str(mo)] = [
            {
                'day':     int(r['day']),
                'mean':    round(float(r['mean']), 4),
                'ci_low':  round(float(r['mean'] - r['ci95']), 4),
                'ci_high': round(float(r['mean'] + r['ci95']), 4),
            }
            for _, r in sub.iterrows()
        ]

out = os.path.join(DIR, 'seasonality_data.json')
with open(out, 'w') as f:
    json.dump(result, f)
print(f'저장: {out}  ({os.path.getsize(out)//1024} KB)')
