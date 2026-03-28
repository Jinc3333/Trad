"""실제 주봉 데이터 + 검증된 월간 데이터 → 일간 데이터 변환
- 2020.01.05 이후: investing.com 실제 주봉 데이터 사용
- 1985.01 ~ 2019.12: ndx_monthly_verified.csv 월간 앵커 포인트 사용
"""
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1) 실제 주봉 데이터 파싱 (2020+) ────────────────────────────────────────
raw_text = open(os.path.join(SCRIPT_DIR, 'ndx_weekly_raw.txt')).read()

rows = []
for line in raw_text.strip().splitlines():
    line = line.strip()
    if not line or line.startswith('Date'):
        continue
    parts = [p.strip() for p in line.split('\t')]
    if len(parts) < 2:
        continue
    try:
        date = pd.to_datetime(parts[0], format='%m/%d/%Y')
        price = float(parts[1].replace(',', ''))
        rows.append({'Date': date, 'Close': price})
    except Exception:
        continue

df_weekly = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)
print(f"주봉 데이터: {len(df_weekly)}개  ({df_weekly['Date'].iloc[0].date()} ~ {df_weekly['Date'].iloc[-1].date()})")

# ── 2) 검증된 월간 데이터 로드 (1985-2019 구간만 사용) ─────────────────────
monthly_csv = os.path.join(SCRIPT_DIR, 'ndx_monthly_verified.csv')
df_monthly = pd.read_csv(monthly_csv, parse_dates=['Date'])
# 주봉 데이터 시작 전까지만 사용
weekly_start = df_weekly['Date'].iloc[0]
df_monthly_pre = df_monthly[df_monthly['Date'] < weekly_start][['Date', 'Close']].copy()
print(f"월간 데이터(사전기간): {len(df_monthly_pre)}개  ({df_monthly_pre['Date'].iloc[0].date()} ~ {df_monthly_pre['Date'].iloc[-1].date()})")

# ── 3) 두 데이터 합치기 ──────────────────────────────────────────────────────
df_combined = pd.concat([
    df_monthly_pre,
    df_weekly[['Date', 'Close']]
], ignore_index=True).sort_values('Date').drop_duplicates('Date').reset_index(drop=True)

print(f"합산 앵커 포인트: {len(df_combined)}개  ({df_combined['Date'].iloc[0].date()} ~ {df_combined['Date'].iloc[-1].date()})")
print(f"가격 범위: {df_combined['Close'].min():.2f} ~ {df_combined['Close'].max():.2f}")

# ── 4) 영업일 일간 시리즈 생성 (로그 보간) ────────────────────────────────
biz = pd.bdate_range(start=df_combined['Date'].iloc[0], end='2026-03-28')

# 주봉 날짜가 주말일 경우 가장 가까운 영업일로 스냅
def snap_to_biz(dates, biz_index):
    snapped = {}
    biz_arr = pd.DatetimeIndex(biz_index)
    for dt, val in dates.items():
        idx = biz_arr.get_indexer([dt], method='nearest')[0]
        if idx >= 0:
            snapped[biz_arr[idx]] = val
    return pd.Series(snapped)

s = snap_to_biz(df_combined.set_index('Date')['Close'], biz).reindex(biz)
log_s = np.log(s.dropna())
full_log = log_s.reindex(biz).interpolate(method='linear')
daily_close = np.exp(full_log)

# 약간의 일별 노이즈 (0.5% σ)
np.random.seed(42)
noise = np.random.normal(0, 0.005, len(daily_close))
daily_noisy = daily_close * (1 + noise)

out = pd.DataFrame({'Close': daily_noisy.values}, index=biz)
out.index.name = 'Date'
out.to_csv(os.path.join(SCRIPT_DIR, 'ndx_historical.csv'))

print(f"\n일간 데이터: {len(out):,}일  ({out.index[0].date()} ~ {out.index[-1].date()})")
print("\n주요 시점 확인:")
checks = {
    '1987-10-19': 292,   # 블랙먼데이 직후
    '2000-03-10': 4704,  # 닷컴 고점
    '2002-10-10': 815,   # 닷컴 저점
    '2009-03-09': 1018,  # 금융위기 저점
    '2020-03-23': 6994,  # COVID 저점
    '2024-12-31': 21209, # 2024 연말
}
for ds, ref in checks.items():
    dt = pd.to_datetime(ds)
    idx = out.index[out.index.get_indexer([dt], method='nearest')[0]]
    print(f"  {ds}: {out.loc[idx,'Close']:.0f}  (참고:{ref})")
