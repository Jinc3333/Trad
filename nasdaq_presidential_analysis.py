"""
나스닥100 대통령별 수익률 및 계절성 분석 (1985 ~ 현재)
데이터: 역사적 실제 가격 기반 (Bloomberg/Yahoo Finance 참고치)
"""

import warnings
warnings.filterwarnings('ignore')

import os, subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from datetime import datetime

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm

def setup_korean_font():
    for p in fm.findSystemFonts():
        name = os.path.basename(p).lower()
        if any(k in name for k in ('nanum', 'gothic', 'malgun', 'gulim')):
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
    try:
        subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-nanum'],
                       capture_output=True, check=True)
        fm._load_fontmanager(try_read_cache=False)
        for p in fm.findSystemFonts():
            if 'nanum' in os.path.basename(p).lower():
                prop = fm.FontProperties(fname=p)
                plt.rcParams['font.family'] = prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                return True
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False
    return False

HAS_KR = setup_korean_font()
print(f"한글 폰트: {'사용 가능' if HAS_KR else '영문 폴백 사용'}")

# ── 대통령 데이터 ───────────────────────────────────────────────────────────────
PRESIDENTS = [
    # (이름,           임기시작,       임기종료,       선수, 당)
    ('Reagan',        '1985-01-20', '1989-01-20', 2, 'R'),   # 2기 (재선)
    ('G.H.W.Bush',    '1989-01-20', '1993-01-20', 1, 'R'),
    ('Clinton',       '1993-01-20', '1997-01-20', 1, 'D'),
    ('Clinton',       '1997-01-20', '2001-01-20', 2, 'D'),
    ('G.W.Bush',      '2001-01-20', '2005-01-20', 1, 'R'),
    ('G.W.Bush',      '2005-01-20', '2009-01-20', 2, 'R'),
    ('Obama',         '2009-01-20', '2013-01-20', 1, 'D'),
    ('Obama',         '2013-01-20', '2017-01-20', 2, 'D'),
    ('Trump',         '2017-01-20', '2021-01-20', 1, 'R'),
    ('Biden',         '2021-01-20', '2025-01-20', 1, 'D'),
    ('Trump',         '2025-01-20', '2029-01-20', 2, 'R'),   # 2기 (진행중)
]

PRES_DF = pd.DataFrame(PRESIDENTS,
    columns=['name', 'start', 'end', 'term', 'party'])
PRES_DF['start'] = pd.to_datetime(PRES_DF['start'])
PRES_DF['end']   = pd.to_datetime(PRES_DF['end'])
TODAY = pd.Timestamp('2026-03-28')

PARTY_COLOR = {'R': '#E74C3C', 'D': '#3498DB'}
PARTY_KR    = {'R': '공화당', 'D': '민주당'}

# ── 데이터 로드 ─────────────────────────────────────────────────────────────────
DATA_CSV = os.path.join(os.path.dirname(__file__), 'ndx_historical.csv')
if not os.path.exists(DATA_CSV):
    print("데이터 파일 없음 → generate_ndx_data.py 실행 중...")
    subprocess.run(['python3', os.path.join(os.path.dirname(__file__),
                   'generate_ndx_data.py')], check=True)

print("나스닥100 데이터 로드 중...")
raw = pd.read_csv(DATA_CSV, index_col='Date', parse_dates=True)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
raw = raw[['Close']].rename(columns={'Close': 'close'}).dropna()
raw = raw[raw.index <= TODAY]
raw['ret'] = raw['close'].pct_change()

print(f"  데이터: {raw.index[0].date()} ~ {raw.index[-1].date()} ({len(raw):,}일)")
print(f"  현재가 (기준): {raw['close'].iloc[-1]:.0f}")

# ── 대통령 라벨 부착 ────────────────────────────────────────────────────────────
def assign_president(date):
    for _, r in PRES_DF.iterrows():
        end = min(r['end'], TODAY)
        if r['start'] <= date < end:
            return r['name'], r['term'], r['party']
    return None, None, None

labels = [assign_president(d) for d in raw.index]
raw['pres']  = [x[0] for x in labels]
raw['term']  = [x[1] for x in labels]
raw['party'] = [x[2] for x in labels]
raw = raw.dropna(subset=['pres'])
raw['year']  = raw.index.year
raw['month'] = raw.index.month

# ── 헬퍼: 기간 수익률 ──────────────────────────────────────────────────────────
def period_return(df):
    if len(df) < 2: return np.nan
    return (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

# ── 연도별 수익률 ───────────────────────────────────────────────────────────────
annual = (raw.groupby('year').apply(period_return)
            .reset_index(name='annual_ret'))
for col in ['pres', 'party', 'term']:
    annual = annual.merge(
        raw.groupby('year')[col].last().reset_index(), on='year')

# ── 임기별 수익률 통계 ──────────────────────────────────────────────────────────
term_stats = []
for _, row in PRES_DF.iterrows():
    end = min(row['end'], TODAY)
    mask = (raw.index >= row['start']) & (raw.index < end)
    sub = raw[mask]
    if len(sub) < 20: continue

    total_ret = period_return(sub)
    ann_vol   = sub['ret'].std() * np.sqrt(252) * 100
    max_dd    = ((sub['close'] / sub['close'].cummax()) - 1).min() * 100
    sharpe    = (sub['ret'].mean() / sub['ret'].std()) * np.sqrt(252)

    year_rets = []
    for yr_offset in range(1, 5):
        yr_s = row['start'] + pd.DateOffset(years=yr_offset - 1)
        yr_e = min(row['start'] + pd.DateOffset(years=yr_offset), TODAY)
        ym = sub[(sub.index >= yr_s) & (sub.index < yr_e)]
        year_rets.append(round(period_return(ym), 2) if len(ym) > 10 else None)

    term_str = '재선(2기)' if row['term'] == 2 else '초선(1기)'
    term_stats.append({
        'label':     f"{row['name']}\n{term_str}",
        'name':      row['name'],
        'term':      row['term'],
        'party':     row['party'],
        'start':     row['start'],
        'term_str':  term_str,
        'total_ret': round(total_ret, 2),
        'ann_vol':   round(ann_vol, 2),
        'max_dd':    round(max_dd, 2),
        'sharpe':    round(sharpe, 3),
        'yr1': year_rets[0], 'yr2': year_rets[1],
        'yr3': year_rets[2], 'yr4': year_rets[3],
    })

ts = pd.DataFrame(term_stats)

# ── 계절성 ──────────────────────────────────────────────────────────────────────
monthly_ret = raw.groupby(['year', 'month'])['close'].apply(
    lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else np.nan
).reset_index(name='mret')

MONTH_KR = ['1월','2월','3월','4월','5월','6월',
            '7월','8월','9월','10월','11월','12월']
seasonality = monthly_ret.groupby('month')['mret'].agg(['mean','median','std','count'])
seasonality.columns = ['평균수익률','중간값','표준편차','횟수']
seasonality.index = MONTH_KR

seasonality_q = raw.groupby(['year','quarter'] if 'quarter' in raw.columns
                              else ['year', raw.index.quarter.rename('quarter')])
raw['quarter'] = raw.index.quarter
q_ret = raw.groupby(['year','quarter'])['close'].apply(
    lambda x: (x.iloc[-1]/x.iloc[0]-1)*100 if len(x)>1 else np.nan
).reset_index(name='qret')
seasonality_q = q_ret.groupby('quarter')['qret'].agg(['mean','median','std'])
seasonality_q.index = ['Q1(1~3월)','Q2(4~6월)','Q3(7~9월)','Q4(10~12월)']

yr_cols   = ['yr1','yr2','yr3','yr4']
yr_labels = ['1년차','2년차','3년차','4년차']
term_compare = ts.groupby('term')['total_ret'].agg(['mean','median','min','max','count'])
term_compare.index = ['초선(1기)','재선(2기)']

# ════════════════════════════════════════════════════════════════════════════════
#  콘솔 출력
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*75)
print("  나스닥100 × 미국 대통령 수익률 분석 (1985 ~ 2026.03)")
print("="*75)

print("\n┌─ 임기별 상세 수익률 ─────────────────────────────────────────────────────┐")
print(f"{'대통령':<14} {'구분':<8} {'당':^4} {'총수익률':>8} {'연변동성':>8} {'최대낙폭':>8} {'샤프':>6}  "
      f"{'1년차':>7} {'2년차':>7} {'3년차':>7} {'4년차':>7}")
print("─"*95)
for _, r in ts.iterrows():
    yr_str = "  ".join([f"{v:+7.1f}%" if v is not None else "    N/A" for v in
                         [r['yr1'],r['yr2'],r['yr3'],r['yr4']]])
    print(f"{r['name']:<14} {r['term_str']:<8} {r['party']:^4} "
          f"{r['total_ret']:>+7.1f}%  {r['ann_vol']:>6.1f}%  {r['max_dd']:>+7.1f}% "
          f"{r['sharpe']:>6.2f}  {yr_str}")
print("└" + "─"*94 + "┘")

print("\n[ 초선 vs 재선 비교 ]")
print(term_compare.round(2).to_string())

print("\n[ 대통령 연차별 평균 수익률 ]")
for col, lbl in zip(yr_cols, yr_labels):
    all_avg  = ts[col].mean()
    init_avg = ts[ts['term']==1][col].mean()
    re_avg   = ts[ts['term']==2][col].mean()
    bar = '█' * max(0, int(abs(all_avg)/4))
    print(f"  {lbl}: 전체 {all_avg:+6.1f}%  초선 {init_avg:+6.1f}%  재선 {re_avg:+6.1f}%  {bar}")

print("\n[ 월별 평균 수익률 (계절성) ]")
for m, row in seasonality.iterrows():
    bar = '█' * max(0, int(abs(row['평균수익률'])/0.5))
    sign = '+' if row['평균수익률'] >= 0 else ''
    print(f"  {m:>4}: {sign}{row['평균수익률']:5.2f}%  ({row['횟수']:.0f}년 데이터)  {bar}")

print("\n[ 분기별 평균 수익률 ]")
print(seasonality_q.round(2).to_string())

# ════════════════════════════════════════════════════════════════════════════════
#  시각화 (6개 서브플롯)
# ════════════════════════════════════════════════════════════════════════════════
RED   = '#E74C3C'; BLUE  = '#3498DB'; GREEN = '#2ECC71'
GOLD  = '#F39C12'; GRAY  = '#BDC3C7'; DARK  = '#2C3E50'
LIGHT = '#ECF0F1'

fig = plt.figure(figsize=(24, 30))
fig.patch.set_facecolor('#F0F3F7')
gs  = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32,
               left=0.06, right=0.97, top=0.94, bottom=0.03)

# ─── [Row0] 나스닥100 가격 차트 + 대통령 배경 ────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor('#FDFEFE')

ax0.semilogy(raw.index, raw['close'], color=DARK, lw=1.0, zorder=3, alpha=0.85)

for _, r in PRES_DF.iterrows():
    end = min(r['end'], TODAY)
    if r['start'] >= TODAY: continue
    col = PARTY_COLOR[r['party']]
    ax0.axvspan(r['start'], end, alpha=0.13, color=col, zorder=1)
    mid = r['start'] + (end - r['start']) / 2
    ymax = raw['close'].max()
    short_name = r['name'].split('.')[-1]
    suffix = '재' if r['term'] == 2 else '초'
    ax0.text(mid, ymax * 0.55, f"{short_name}\n({suffix})",
             ha='center', va='top', fontsize=8, color=col,
             fontweight='bold', zorder=4, linespacing=1.4)

ax0.set_title(f"나스닥100 지수 추이 (1985.01 ~ {TODAY.date()}) — 대통령 임기별 배경",
              fontsize=14, fontweight='bold', color=DARK, pad=12)
ax0.set_ylabel("지수 (로그 스케일)", fontsize=10)
ax0.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax0.grid(axis='y', linestyle='--', alpha=0.35)
rp = mpatches.Patch(color=RED,  alpha=0.5, label='공화당(R)')
dp = mpatches.Patch(color=BLUE, alpha=0.5, label='민주당(D)')
ax0.legend(handles=[rp, dp], loc='upper left', fontsize=10)

# 주요 이벤트 표시
events = [
    ('1987-10', '블랙먼데이', 309),
    ('2000-03', '닷컴버블\n고점', 4700),
    ('2002-10', '닷컴버블\n최저', 820),
    ('2008-10', '금융위기', 1260),
    ('2009-03', '금융위기\n최저', 1020),
    ('2020-03', 'COVID\n최저', 6600),
]
for ev_date, ev_label, ev_y in events:
    edt = pd.to_datetime(ev_date)
    ax0.annotate(ev_label, xy=(edt, ev_y), xytext=(edt, ev_y * 0.45),
                 fontsize=6.5, ha='center', color='#7F8C8D',
                 arrowprops=dict(arrowstyle='->', color='#95A5A6', lw=0.8),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0.5))

# ─── [Row1, L] 임기별 총 수익률 바 차트 ───────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor('#FDFEFE')

colors_bar = [PARTY_COLOR[p] for p in ts['party']]
ypos = np.arange(len(ts))
bars = ax1.barh(ypos, ts['total_ret'].values,
                color=colors_bar, edgecolor='white', linewidth=0.8, height=0.65)
ax1.set_yticks(ypos)
ax1.set_yticklabels(ts['label'].values, fontsize=9)
for i, (bar, val) in enumerate(zip(bars, ts['total_ret'].values)):
    offset = 8 if val >= 0 else -8
    ax1.text(val + offset, bar.get_y() + bar.get_height()/2,
             f"{val:+.0f}%", va='center',
             ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold')

ax1.axvline(0, color=DARK, lw=1.2)
ax1.set_title("임기별 나스닥100 총 수익률", fontsize=12, fontweight='bold', color=DARK)
ax1.set_xlabel("임기 전체 누적 수익률 (%)", fontsize=9)
ax1.grid(axis='x', linestyle='--', alpha=0.3)

# ─── [Row1, R] 연도별 수익률 바 차트 ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor('#FDFEFE')

colors_yr = [PARTY_COLOR[p] for p in annual['party']]
ax2.bar(annual['year'], annual['annual_ret'],
        color=colors_yr, edgecolor='white', linewidth=0.2, width=0.85)
ax2.axhline(0, color=DARK, lw=1.2)
ax2.set_title("연도별 나스닥100 수익률", fontsize=12, fontweight='bold', color=DARK)
ax2.set_xlabel("연도", fontsize=9)
ax2.set_ylabel("연간 수익률 (%)", fontsize=9)
ax2.tick_params(axis='x', rotation=70, labelsize=7)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
# 대통령 라벨
for _, r in PRES_DF.iterrows():
    end = min(r['end'], TODAY)
    if r['start'] >= TODAY: continue
    mid_yr = r['start'].year + (end.year - r['start'].year) / 2
    ax2.text(mid_yr, annual['annual_ret'].min() - 8,
             r['name'].split('.')[-1][:5],
             ha='center', fontsize=6.5,
             color=PARTY_COLOR[r['party']], fontweight='bold')

# ─── [Row2, L] 연차별 평균 수익률 (전체/초선/재선) ───────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor('#FDFEFE')

yr_all  = ts[yr_cols].mean().values
yr_init = ts[ts['term']==1][yr_cols].mean().values
yr_re   = ts[ts['term']==2][yr_cols].mean().values

x = np.arange(4)
w = 0.25
b0 = ax3.bar(x - w,     yr_all,  w*1.8, label='전체 평균', color=GOLD,  alpha=0.88, edgecolor='white')
b1 = ax3.bar(x + w*0.4, yr_init, w,     label='초선(1기)', color=GREEN, alpha=0.88, edgecolor='white')
b2 = ax3.bar(x + w*1.4, yr_re,   w,     label='재선(2기)', color=BLUE,  alpha=0.88, edgecolor='white')

ax3.set_xticks(x)
ax3.set_xticklabels(yr_labels, fontsize=11)
ax3.axhline(0, color=DARK, lw=1.2)
ax3.set_title("대통령 임기 연차별 평균 수익률", fontsize=12, fontweight='bold', color=DARK)
ax3.set_ylabel("평균 수익률 (%)", fontsize=9)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(axis='y', linestyle='--', alpha=0.3)

for group in [b0, b1, b2]:
    for b in group:
        h = b.get_height()
        if not np.isnan(h):
            ax3.text(b.get_x()+b.get_width()/2,
                     h + 1.5 if h >= 0 else h - 3.5,
                     f"{h:+.0f}%", ha='center',
                     va='bottom' if h >= 0 else 'top', fontsize=8)

# ─── [Row2, R] 월별 계절성 ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor('#FDFEFE')

means  = seasonality['평균수익률'].values
stds   = seasonality['표준편차'].values
cnts   = seasonality['횟수'].values
se     = stds / np.sqrt(cnts)
cols4  = [GREEN if v >= 0 else RED for v in means]

ax4.bar(MONTH_KR, means, color=cols4, edgecolor='white', linewidth=0.6, alpha=0.85)
ax4.errorbar(MONTH_KR, means, yerr=se*1.96,
             fmt='none', color=DARK, capsize=5, lw=1.5, zorder=5)
ax4.axhline(0, color=DARK, lw=1.2)

for i, (m, v) in enumerate(zip(MONTH_KR, means)):
    ax4.text(i, v + 0.15 if v >= 0 else v - 0.25,
             f"{v:+.1f}%", ha='center',
             va='bottom' if v >= 0 else 'top', fontsize=8.5, fontweight='bold')

ax4.set_title("월별 평균 수익률 — 계절성 분석 (1985~2026)", fontsize=12, fontweight='bold', color=DARK)
ax4.set_ylabel("평균 월간 수익률 (%)", fontsize=9)
ax4.set_xlabel("월", fontsize=9)
ax4.grid(axis='y', linestyle='--', alpha=0.3)

# ─── [Row3, L] 대통령별 연차 수익률 궤적 ─────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
ax5.set_facecolor('#FDFEFE')

for _, r in ts.iterrows():
    vals  = [r[c] for c in yr_cols]
    valid = [(i+1, v) for i,v in enumerate(vals) if v is not None and not np.isnan(v)]
    if not valid: continue
    xs, ys = zip(*valid)
    col = PARTY_COLOR[r['party']]
    ls  = '--' if r['term'] == 2 else '-'
    ax5.plot(xs, ys, color=col, lw=1.3, ls=ls, alpha=0.55,
             marker='o', ms=4, zorder=2)
    ax5.text(xs[-1] + 0.06, ys[-1],
             r['name'].split('.')[-1][:5], fontsize=7, color=col)

for term_n, ls, lbl, col in [
        (1, '-',  '초선(1기) 평균', DARK),
        (2, '--', '재선(2기) 평균', '#7F8C8D')]:
    sub  = ts[ts['term'] == term_n]
    avgs = [sub[c].mean() for c in yr_cols]
    ax5.plot([1,2,3,4], avgs, color=col, lw=3, ls=ls, label=lbl, zorder=6)

ax5.axhline(0, color=GRAY, lw=1)
ax5.set_xticks([1,2,3,4])
ax5.set_xticklabels(yr_labels, fontsize=10)
ax5.set_title("대통령별 임기 연차 수익률 궤적", fontsize=12, fontweight='bold', color=DARK)
ax5.set_ylabel("해당 연도 수익률 (%)", fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(linestyle='--', alpha=0.3)
rp2 = mpatches.Patch(color=RED,  alpha=0.6, label='공화당')
dp2 = mpatches.Patch(color=BLUE, alpha=0.6, label='민주당')
ax5.legend(handles=[rp2, dp2] + ax5.get_legend_handles_labels()[0], fontsize=8)

# ─── [Row3, R] 당별 월간 계절성 비교 ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
ax6.set_facecolor('#FDFEFE')

raw2 = raw.copy()
raw2['mret_'] = np.nan
for (yr, mn), grp in raw2.groupby(['year','month']):
    if len(grp) > 1:
        r = (grp['close'].iloc[-1] / grp['close'].iloc[0] - 1) * 100
        raw2.loc[grp.index, 'mret_'] = r

month_party = raw2.drop_duplicates(subset=['year','month'])[
    ['year','month','party','mret_']].dropna()
rep_m = month_party[month_party['party']=='R'].groupby('month')['mret_'].mean()
dem_m = month_party[month_party['party']=='D'].groupby('month')['mret_'].mean()

x6 = np.arange(1, 13)
w6 = 0.38
ax6.bar(x6 - w6/2, rep_m.reindex(x6).values, w6,
        label='공화당(R)', color=RED, alpha=0.82, edgecolor='white')
ax6.bar(x6 + w6/2, dem_m.reindex(x6).values, w6,
        label='민주당(D)', color=BLUE, alpha=0.82, edgecolor='white')
ax6.axhline(0, color=DARK, lw=1.2)
ax6.set_xticks(x6)
ax6.set_xticklabels([f"{m}월" for m in range(1,13)], fontsize=8)
ax6.set_title("당별 월간 수익률 계절성 비교", fontsize=12, fontweight='bold', color=DARK)
ax6.set_ylabel("평균 월간 수익률 (%)", fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(axis='y', linestyle='--', alpha=0.3)

# ─── 메인 타이틀 & 저장 ──────────────────────────────────────────────────────
plt.suptitle(
    f"나스닥100 × 미국 대통령  수익률 & 계절성 종합 분석  (1985.01 ~ {TODAY.date()})",
    fontsize=16, fontweight='bold', color=DARK, y=0.972)

out_png = "/home/user/Trad/nasdaq_presidential_analysis.png"
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n차트 저장: {out_png}")

# ── 엑셀 저장 ───────────────────────────────────────────────────────────────────
try:
    out_xlsx = "/home/user/Trad/nasdaq_presidential_analysis.xlsx"
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        ts[['name','term_str','party','start','total_ret','ann_vol',
            'max_dd','sharpe','yr1','yr2','yr3','yr4']].to_excel(
            writer, sheet_name='임기별수익률', index=False)
        annual[['year','annual_ret','pres','party','term']].to_excel(
            writer, sheet_name='연도별수익률', index=False)
        seasonality.to_excel(writer, sheet_name='월별계절성')
        seasonality_q.to_excel(writer, sheet_name='분기별계절성')
        term_compare.to_excel(writer, sheet_name='초선재선비교')
    print(f"엑셀 저장: {out_xlsx}")
except Exception as e:
    print(f"엑셀 저장 실패: {e}")

print("\n✓ 분석 완료!")
