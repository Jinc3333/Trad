"""
대통령 임기 연차별 × 월별 평균 수익률 꺾은선 그래프
- 4개 선: 1년차 / 2년차 / 3년차 / 4년차
- X축: 1~12월 (달력 기준)
- Y축: 해당 (연차, 월) 조합의 평균 월간 수익률 (%)
- 오차범위(95% CI) 포함
"""

import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import subprocess

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

# ── 대통령 임기 정의 ──────────────────────────────────────────────────────────
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

# ── 연차별 월별 수익률 수집 ──────────────────────────────────────────────────
# 각 (대통령, 연차, 월) → 해당 월의 수익률 수집
records = []

for _, prow in PRES_DF.iterrows():
    p_start = prow['start']
    p_end   = min(prow['end'], TODAY)

    for yr_n in range(1, 5):                         # 1~4년차
        yr_s = p_start + pd.DateOffset(years=yr_n - 1)
        yr_e = min(p_start + pd.DateOffset(years=yr_n), TODAY)
        if yr_s >= TODAY:
            break

        # 해당 연차 구간 데이터
        seg = raw[(raw.index >= yr_s) & (raw.index < yr_e)].copy()
        if len(seg) < 10:
            break

        # (달력연도, 달력월) 단위로 완전한 월만 수집
        # → 취임일(1/20)에 의해 1월이 두 토막 나는 문제 방지
        for (cal_yr, cal_mo), m_data in seg.groupby(
                [seg.index.year, seg.index.month]):
            # 10 영업일 미만 제외 (Jan 20-31 ~8일 탈락, Jan 1-19 ~13일 허용)
            if len(m_data) < 10:
                continue
            mret = (m_data['close'].iloc[-1] / m_data['close'].iloc[0] - 1) * 100
            records.append({
                'pres':    prow['name'],
                'term':    prow['term'],
                'party':   prow['party'],
                'yr_n':    yr_n,
                'month':   cal_mo,
                'mret':    mret,
            })

df = pd.DataFrame(records)
print(f"수집된 (연차, 월) 샘플 수: {len(df)}")
print(df.groupby(['yr_n', 'month']).size().unstack(fill_value=0))

# ── 집계: 평균 & 95% CI ──────────────────────────────────────────────────────
agg = df.groupby(['yr_n', 'month'])['mret'].agg(['mean', 'std', 'count']).reset_index()
agg['se']   = agg['std'] / np.sqrt(agg['count'])
agg['ci95'] = agg['se'] * 1.96

# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
MONTH_KR = ['1월','2월','3월','4월','5월','6월',
            '7월','8월','9월','10월','11월','12월']
YR_LABEL = ['1년차','2년차','3년차','4년차']

print("\n[ 대통령 연차별 월평균 수익률 (%) ]")
header = f"{'월':>4}" + "".join(f"{lbl:>10}" for lbl in YR_LABEL)
print(header)
print("-" * 44)
for m in range(1, 13):
    row_str = f"{MONTH_KR[m-1]:>4}"
    for yr in range(1, 5):
        val = agg[(agg['yr_n']==yr) & (agg['month']==m)]['mean']
        row_str += f"  {val.values[0]:+6.2f}%" if len(val) else "       N/A"
    print(row_str)

# ── 시각화 ────────────────────────────────────────────────────────────────────
COLORS = {
    1: '#E74C3C',   # 1년차 — 빨강
    2: '#3498DB',   # 2년차 — 파랑
    3: '#2ECC71',   # 3년차 — 초록
    4: '#F39C12',   # 4년차 — 주황
}
DARK  = '#2C3E50'
LIGHT = '#F0F3F7'

fig, axes = plt.subplots(2, 1, figsize=(16, 16),
                         gridspec_kw={'height_ratios': [3, 1.4]})
fig.patch.set_facecolor(LIGHT)

# ── 메인: 꺾은선 그래프 ───────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#FDFEFE')
ax.axhline(0, color=DARK, lw=1.2, zorder=2)

x_ticks = np.arange(1, 13)

for yr in range(1, 5):
    sub = agg[agg['yr_n'] == yr].set_index('month').reindex(x_ticks)
    means = sub['mean'].values
    ci    = sub['ci95'].values
    col   = COLORS[yr]
    lbl   = YR_LABEL[yr-1]

    # CI 음영
    ax.fill_between(x_ticks, means - ci, means + ci,
                    color=col, alpha=0.12, zorder=1)

    # 선 + 마커
    ax.plot(x_ticks, means, color=col, lw=2.4, marker='o',
            ms=7, label=lbl, zorder=4, markeredgecolor='white', markeredgewidth=1.2)

    # 값 레이블
    for x, y in zip(x_ticks, means):
        if np.isnan(y):
            continue
        offset = 0.45 if y >= 0 else -0.55
        ax.text(x, y + offset, f"{y:+.1f}%", ha='center',
                va='bottom' if y >= 0 else 'top',
                fontsize=8, color=col, fontweight='bold')

ax.set_xticks(x_ticks)
ax.set_xticklabels(MONTH_KR, fontsize=11)
ax.set_ylabel("평균 월간 수익률 (%)", fontsize=11)
ax.set_xlim(0.4, 12.6)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.set_title(
    "대통령 임기 연차별 × 월별  평균 수익률  (나스닥100, 1985~2026)\n"
    "음영 = 95% 신뢰구간  |  n = 역대 대통령 11명 임기 기준",
    fontsize=13, fontweight='bold', color=DARK, pad=14)

# ── 서브: 히트맵 스타일 테이블 ────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#FDFEFE')
ax2.axis('off')

table_data = []
row_labels = []
for yr in range(1, 5):
    row = []
    row_labels.append(YR_LABEL[yr-1])
    for m in range(1, 13):
        val = agg[(agg['yr_n']==yr) & (agg['month']==m)]['mean']
        row.append(f"{val.values[0]:+.1f}%" if len(val) else "N/A")
    table_data.append(row)

tbl = ax2.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=MONTH_KR,
    cellLoc='center', rowLoc='center',
    loc='center',
    bbox=[0.0, 0.05, 1.0, 0.9]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

# 셀 색상
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#BDC3C7')
    if r == 0:                      # 헤더 행
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.22)
    elif c == -1:                   # 행 라벨
        yr_idx = r
        cell.set_facecolor(COLORS.get(yr_idx, '#ECF0F1'))
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.22)
    else:
        # 값에 따라 녹색/빨강 계열 음영
        try:
            val_str = table_data[r-1][c-1].replace('%','').replace('+','')
            val = float(val_str)
            intensity = min(abs(val) / 8.0, 0.6)
            if val >= 0:
                cell.set_facecolor((*[1 - intensity*0.3 * x for x in (0, 0.3, 0)], 1))
                cell.set_facecolor((1 - intensity*0.15,
                                    1,
                                    1 - intensity*0.15, 1))
            else:
                cell.set_facecolor((1,
                                    1 - intensity*0.15,
                                    1 - intensity*0.15, 1))
        except Exception:
            cell.set_facecolor('#FDFEFE')
        cell.set_height(0.22)

ax2.set_title("월별 평균 수익률 요약 테이블", fontsize=11,
              fontweight='bold', color=DARK, pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle("", y=0.99)

out_png = "/home/user/Trad/ndx_pres_year_monthly.png"
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n차트 저장: {out_png}")
