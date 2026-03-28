"""
대통령 임기 연차별 × 월별  일봉 평균 수익률 꺾은선 그래프
- 4개 선: 1년차 / 2년차 / 3년차 / 4년차
- X축: 1~12월 (달력 기준)
- Y축: 해당 (연차, 월) 구간 일봉 수익률의 평균 (%)
- 총 48 포인트 + 95% 신뢰구간 음영
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
raw['ret'] = raw['close'].pct_change() * 100   # 일간 수익률 (%)

# ── 각 일봉에 (연차, 월) 태그 부착 ───────────────────────────────────────────
rows = []
for _, prow in PRES_DF.iterrows():
    p_start = prow['start']

    for yr_n in range(1, 5):
        yr_s = p_start + pd.DateOffset(years=yr_n - 1)
        yr_e = min(p_start + pd.DateOffset(years=yr_n), TODAY)
        if yr_s >= TODAY:
            break

        seg = raw[(raw.index > yr_s) & (raw.index < yr_e)].copy()
        # pct_change 첫날(취임일 다음)은 정상 포함
        seg = seg.dropna(subset=['ret'])
        seg['yr_n']  = yr_n
        seg['month'] = seg.index.month
        rows.append(seg[['ret', 'yr_n', 'month']])

df = pd.concat(rows, ignore_index=True)
print(f"총 일봉 수: {len(df):,}개")

# ── 집계: (연차, 월) → 일봉 평균 & 95% CI ────────────────────────────────────
agg = (df.groupby(['yr_n', 'month'])['ret']
         .agg(mean='mean', std='std', n='count')
         .reset_index())
agg['se']   = agg['std'] / np.sqrt(agg['n'])
agg['ci95'] = agg['se'] * 1.96

# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
MONTH_KR = ['1월','2월','3월','4월','5월','6월',
            '7월','8월','9월','10월','11월','12월']
YR_LABEL = ['1년차','2년차','3년차','4년차']

print(f"\n[ 연차별 월 평균 일봉 수익률 (%) — 샘플 수 ]")
cnt_tbl = agg.pivot(index='yr_n', columns='month', values='n').fillna(0).astype(int)
cnt_tbl.index = YR_LABEL
cnt_tbl.columns = MONTH_KR
print(cnt_tbl.to_string())

print(f"\n[ 연차별 월 평균 일봉 수익률 (%) ]")
hdr = f"{'':>5}" + "".join(f"{m:>8}" for m in MONTH_KR)
print(hdr)
print("-" * (5 + 8*12))
for yr in range(1, 5):
    row_str = f"{YR_LABEL[yr-1]:>5}"
    for m in range(1, 13):
        val = agg[(agg['yr_n']==yr) & (agg['month']==m)]['mean']
        row_str += f"  {val.values[0]:+5.3f}%" if len(val) else "      N/A"
    print(row_str)

# ── 시각화 ────────────────────────────────────────────────────────────────────
COLORS = {1: '#E74C3C', 2: '#3498DB', 3: '#2ECC71', 4: '#F39C12'}
DARK   = '#2C3E50'
LIGHT  = '#F0F3F7'

fig, axes = plt.subplots(2, 1, figsize=(16, 14),
                         gridspec_kw={'height_ratios': [2.6, 1]})
fig.patch.set_facecolor(LIGHT)

# ─── 메인 꺾은선 ──────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#FDFEFE')
ax.axhline(0, color=DARK, lw=1.2, zorder=2)

x = np.arange(1, 13)

for yr in range(1, 5):
    sub   = agg[agg['yr_n'] == yr].set_index('month').reindex(x)
    means = sub['mean'].values
    ci    = sub['ci95'].values
    col   = COLORS[yr]
    lbl   = YR_LABEL[yr-1]
    cnt   = sub['n'].values

    ax.fill_between(x, means - ci, means + ci,
                    color=col, alpha=0.13, zorder=1)
    ax.plot(x, means, color=col, lw=2.4, marker='o', ms=7,
            label=lbl, zorder=4,
            markeredgecolor='white', markeredgewidth=1.2)

    for xi, (y, n) in enumerate(zip(means, cnt)):
        if np.isnan(y):
            continue
        offset = 0.018 if y >= 0 else -0.022
        ax.text(xi + 1, y + offset,
                f"{y:+.3f}%",
                ha='center',
                va='bottom' if y >= 0 else 'top',
                fontsize=7.5, color=col, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(MONTH_KR, fontsize=11)
ax.set_ylabel("평균 일봉 수익률 (%)", fontsize=11)
ax.set_xlim(0.35, 12.65)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.set_title(
    "대통령 임기 연차별 × 월별  평균 일봉 수익률  (나스닥100, 1985~2026)\n"
    "음영 = 95% 신뢰구간  |  각 포인트 = 해당 연차 전체 대통령 일봉 평균",
    fontsize=13, fontweight='bold', color=DARK, pad=14)

# ─── 하단 요약 테이블 ─────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#FDFEFE')
ax2.axis('off')

table_data = []
row_labels = []
for yr in range(1, 5):
    row = []
    row_labels.append(YR_LABEL[yr-1])
    for m in range(1, 13):
        v = agg[(agg['yr_n']==yr) & (agg['month']==m)]['mean']
        row.append(f"{v.values[0]:+.3f}%" if len(v) else "N/A")
    table_data.append(row)

tbl = ax2.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=MONTH_KR,
    cellLoc='center', rowLoc='center',
    loc='center',
    bbox=[0.0, 0.05, 1.0, 0.88]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#BDC3C7')
    cell.set_height(0.22)
    if r == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == -1:
        cell.set_facecolor(COLORS.get(r, '#ECF0F1'))
        cell.set_text_props(color='white', fontweight='bold')
    else:
        try:
            val = float(table_data[r-1][c-1].replace('%','').replace('+',''))
            intensity = min(abs(val) / 0.15, 0.7)   # ±0.15%를 최대 채도 기준
            if val >= 0:
                cell.set_facecolor((1 - intensity*0.15, 1, 1 - intensity*0.15, 1))
            else:
                cell.set_facecolor((1, 1 - intensity*0.15, 1 - intensity*0.15, 1))
        except Exception:
            cell.set_facecolor('#FDFEFE')

ax2.set_title("평균 일봉 수익률 요약 테이블", fontsize=11,
              fontweight='bold', color=DARK, pad=8)

plt.tight_layout(pad=1.5)

out_png = "/home/user/Trad/ndx_pres_year_monthly.png"
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n차트 저장: {out_png}")
