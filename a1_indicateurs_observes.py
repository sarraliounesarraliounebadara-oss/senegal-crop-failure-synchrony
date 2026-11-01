"""Observed seasonal indicators per retained yield series (department x crop), 1986-2016.

Each series uses climate averaged over its own department with MIRCA2000 rainfed harvested-area weights
of its own crop (reduce_to_departments.py). Indicators: JJAS rainfall, Sep-Oct rainfall, onset date
(Sivakumar 1988), longest dry spell after onset (capped at 30 days), mean JAS Tmax, SPEI-3 (JJAS minimum,
s20_spei_from_raw.py). Every indicator is detrended per series with a Theil-Sen line over 1986-2016
(series mean kept), as the yields are, so that climate-yield links are not driven by common trends.

Usage: python a1_indicateurs_observes.py <folder with obs_*_bycrop.csv.gz> [out]
Output: <out>/s21_obs_indicators.csv (raw and *_dt detrended columns)
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import rain_indicators, heat_indicators
from spei_makkink import detrend_theilsen

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out'
Y0, Y1 = 1986, 2016
y = pd.read_csv(f'{OUT}/yields_detrended.csv')
keep = set(map(tuple, y[['dept', 'crop']].drop_duplicates().values))

ind = rain_indicators(os.path.join(D, 'obs_ARC2_pr_bycrop.csv.gz'), Y0, Y1)
ind['maxDS'] = ind.maxDS.clip(upper=30)
ft = os.path.join(D, 'obs_CHIRTS_tasmax_bycrop.csv.gz')
if os.path.exists(ft):
    ind = ind.merge(heat_indicators(ft, Y0, Y1)[['dept', 'crop', 'year', 'Tmax_JAS']], on=['dept', 'crop', 'year'], how='left')
fs = f'{OUT}/s20_obs_spei3_jjasmin.csv'
if os.path.exists(fs):
    S = pd.read_csv(fs, index_col=0).loc[Y0:Y1]
    S.columns.name = 'series'
    s = S.stack().rename('SPEI3').reset_index()
    s[['dept', 'crop']] = s.series.str.rsplit('_', n=1, expand=True)
    ind = ind.merge(s[['dept', 'crop', 'year', 'SPEI3']], on=['dept', 'crop', 'year'], how='left')
ind = ind[[k in keep for k in zip(ind.dept, ind.crop)]].sort_values(['dept', 'crop', 'year']).reset_index(drop=True)
VARS = [v for v in ['P_JJAS', 'P_SO', 'onset_doy', 'maxDS', 'Tmax_JAS', 'SPEI3'] if v in ind]
trend = []
for v in VARS:
    ind[v + '_dt'] = ind.groupby(['dept', 'crop'], group_keys=False)[v].apply(
        lambda s: pd.Series(detrend_theilsen(s.values, ind.loc[s.index, 'year'].values), index=s.index))
    # size of the removed trends, for reporting
    from scipy.stats import theilslopes
    sl = ind.groupby(['dept', 'crop']).apply(lambda g: theilslopes(g[v], g.year)[0] * 10 if g[v].notna().sum() > 9 else np.nan)
    trend.append(dict(indicator=v, median_trend_per_decade=sl.median(), q10=sl.quantile(.1), q90=sl.quantile(.9),
                      share_positive=(sl > 0).mean(), n_series=sl.notna().sum()))
ind.to_csv(f'{OUT}/s21_obs_indicators.csv', index=False, float_format='%.4f')
T = pd.DataFrame(trend)
T.to_csv(f'{OUT}/s21_removed_trends.csv', index=False)
print(f'{ind.groupby(["dept", "crop"]).ngroups} series x {ind.year.nunique()} years; indicators: {VARS}')
print('Theil-Sen trends removed (per decade, across series):')
print(T.round(3).to_string(index=False))
