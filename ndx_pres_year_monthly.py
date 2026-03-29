"""
대통령 임기 연차별 × 월별  48개 서브플롯
- 4행(1~4년차) × 12열(1~12월) = 48개 그래프
- 각 그래프 X축: 해당 월의 달력 날짜 (1~31일)
- 각 그래프 Y축: 월 첫 거래일 = 0%, 이후 누적 가격변화율 (%)
- 역대 대통령 평균 + 95% CI 음영
"""

import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def setup_korean_font():
    for p in fm.findSystemFonts():
        name = os.path.basename(p).lower()
        if any(k in name for k in ('nanum', 'gothic', 'malgun', 'gulim')):
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
    plt.rcParams['axes.unicode_minus'] = False
    return False

setup_korean_font()

# ── 대통령 임기 ───────────────────────────────────────────────────────────────
PRESIDENTS = [
    ('Reagan',     '1985-01-20', '1989-01-20', 2, 'R'),
    ('G.H.W.Bush', '1989-01-20', '1993-01-20', 1, 'R'),
    ('Clinton',    '1993-01-20', '1997-01-20', 1, 'D'),
    ('Clinton',    '1997-01-20', '2001-01-20', 2, 'D'),
    ('G.W.Bush',   '2001-01-20', '2005-01-20', 1, 'R'),
    ('G.W.Bush',   '2005-01-20', '2009-01-20', 2, 'R'),
    ('Obama',      '2009-01-20', '2013-01-20', 1, 'D'),
    ('Obama',      '2013-01-20', '2017-01-20', 2, 'D'),
    ('Trump',      '2017-01-20', '2021-01-20', 1, 'R'),
    ('Biden',      '2021-01-20', '2025-01-20', 1, 'D'),
    ('Trump',      '2025-01-20', '2029-01-20', 2, 'R'),
]
PRES_DF = pd.DataFrame(PRESIDENTS, columns=['name','start','end','term','party'])
PRES_DF['start'] = pd.to_datetime(PRES_DF['start'])
PRES_DF['end']   = pd.to_datetime(PRES_DF['end'])
TODAY = pd.Timestamp('2026-03-28')

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ndx_historical.csv')
raw = pd.read_csv(DATA_CSV, index_col='Date', parse_dates=True)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
raw = raw[['Close']].rename(columns={'Close': 'close'}).dropna()
raw = raw[raw.index <= TODAY]
raw['ret'] = raw['close'].pct_change() * 100

# ── 각 (대통령, 연차, 월) 구간의 월내 누적수익률 계산 ──────────────────────
# 각 occurrence: 해당 월의 첫 거래일 = 0%, 이후 달력일 기준 누적 %
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
            base = m_data['close'].iloc[0]           # 첫 거래일 기준가
            for ts, row in m_data.iterrows():
                cum_pct = (row['close'] / base - 1) * 100
                records.append({
                    'yr_n':  yr_n,
                    'month': mo,
                    'day':   ts.day,
                    'cum':   cum_pct,
                })

df = pd.DataFrame(records)
print(f"총 데이터포인트: {len(df):,}개")

# ── (연차, 월, 달력일) → 평균·CI 집계 ──────────────────────────────────────
agg = (df.groupby(['yr_n', 'month', 'day'])['cum']
         .agg(mean='mean', std='std', n='count')
         .reset_index())
agg['se']   = agg['std'] / np.sqrt(agg['n'])
agg['ci95'] = agg['se'] * 1.96

# ── 시각화: 4행 × 12열 = 48 서브플롯 ────────────────────────────────────────
MONTH_KR = ['1월','2월','3월','4월','5월','6월',
            '7월','8월','9월','10월','11월','12월']
YR_LABEL  = ['1년차','2년차','3년차','4년차']
COLORS    = {1:'#E74C3C', 2:'#3498DB', 3:'#2ECC71', 4:'#F39C12'}
DARK      = '#2C3E50'
LIGHT     = '#F0F3F7'

fig, axes = plt.subplots(4, 12, figsize=(48, 16),
                          gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
fig.patch.set_facecolor(LIGHT)

for yr in range(1, 5):
    for mo in range(1, 13):
        ax = axes[yr-1][mo-1]
        ax.set_facecolor('#FDFEFE')

        sub = agg[(agg['yr_n']==yr) & (agg['month']==mo)].sort_values('day')
        if sub.empty:
            ax.axis('off')
            continue

        days  = sub['day'].values
        means = sub['mean'].values
        ci    = sub['ci95'].values
        col   = COLORS[yr]

        # 0% 기준선
        ax.axhline(0, color='#95A5A6', lw=0.8, zorder=1)
        # 첫날(day=1) 명시적으로 0 고정
        plot_days  = np.concatenate([[days[0]], days])
        plot_means = np.concatenate([[0], means])
        plot_ci    = np.concatenate([[0], ci])

        ax.fill_between(plot_days,
                        plot_means - plot_ci,
                        plot_means + plot_ci,
                        color=col, alpha=0.18, zorder=2)
        ax.plot(plot_days, plot_means, color=col, lw=1.5, zorder=3)

        # X축: 1, 10, 20, 마지막일
        last_day = int(days[-1])
        xticks = sorted(set([1, 10, 20, last_day]))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(d) for d in xticks], fontsize=6)
        ax.tick_params(axis='y', labelsize=6)

        # Y축 범위 대칭 (최소 ±1%)
        ylim = max(abs(plot_means - plot_ci).max(),
                   abs(plot_means + plot_ci).max(), 1.0)
        ax.set_ylim(-ylim * 1.15, ylim * 1.15)

        # 서브플롯 제목
        ax.set_title(f"{YR_LABEL[yr-1]} {MONTH_KR[mo-1]}",
                     fontsize=8, fontweight='bold',
                     color=col, pad=3)

        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)

# 행 라벨 (좌측)
for yr in range(1, 5):
    axes[yr-1][0].set_ylabel(YR_LABEL[yr-1], fontsize=10,
                              fontweight='bold', color=COLORS[yr],
                              labelpad=6)

plt.suptitle(
    "나스닥100 × 대통령 임기 연차별 월별 일봉 평균 수익률  (1985~2026)\n"
    "각 포인트 = 해당 연차의 역대 대통령 동일 날짜 일봉 평균  |  음영 = 95% CI",
    fontsize=15, fontweight='bold', color=DARK, y=1.01)

out_png = "/home/user/Trad/ndx_pres_year_monthly.png"
fig.savefig(out_png, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"차트 저장: {out_png}")
