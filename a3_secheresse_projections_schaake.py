"""Department drought (SPEI-3, a2_spei3_departements.py): observed link with failures and CMIP6 concurrence (manuscript 3.5-3.6).

Observations, 1986-2016: department SPEI-3 detrended (Theil-Sen); drought = value <= department P20 (1986-2016).
CMIP6: raw SPEI-3; drought = value <= the model's own 1985-2014 P20 of the department; each scenario file is
compared with its own historical window (CDF-t was applied separately to each scenario).
phi and mean inter-department r are computed on SPEI-3 detrended within each 30-year window.
Schaake shuffle: model values re-ordered to follow the ranks of the detrended observed 1985-2014 series.
Usage: python a3_secheresse_projections_schaake.py <out>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spei_makkink import detrend_theilsen
from indicators import loreau

OUT = sys.argv[1]
DD = f'{OUT}/dept'
MODELS = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'FGOALS-g3',
          'GFDL-ESM4', 'MIROC6', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']
WIN = {'hist': (1985, 2014), 'near': (2036, 2065), 'far': (2071, 2100)}
rng = np.random.default_rng(7)


def dtr(S):
    yrs = S.index.values.astype(float)
    return S.apply(lambda s: pd.Series(detrend_theilsen(s.values, yrs), index=S.index))


def pair_r(S):
    C = S.corr().values
    return C[np.triu_indices(C.shape[0], 1)].mean()


def stats(X, thr):
    sh = (X <= thr).mean(axis=1)
    Dt = dtr(X)
    return dict(exposure=100 * sh.mean(), ws40=10 * (sh >= .4).mean(), ws60=10 * (sh >= .6).mean(),
                severe=100 * (X <= -1.5).values.mean(), phi=loreau(Dt.values), r=pair_r(Dt)), sh


# ------------------------------------------------------------ observations
y = pd.read_csv(f'{OUT}/yields_detrended.csv')
y = y[y.fail_p20.notna()].copy()
y['fail'] = y.fail_p20.astype(bool).astype(int)
O = pd.read_csv(f'{DD}/dept_spei3_obs.csv', index_col=0)
O = O[sorted(y.dept.unique())]
Od = dtr(O.loc[1986:2016])
Dr = (Od <= Od.quantile(.2)).astype(int)
Dr.index.name = 'year'
Dr.columns.name = 'dept'
dl = Dr.stack().rename('drought').reset_index()
d = y.merge(dl, on=['dept', 'year'])
yr = pd.DataFrame({'drought': Dr.mean(axis=1), 'fail': y.groupby('year').fail.mean()})
yr.to_csv(f'{OUT}/s31_obs_yearly_drought_failure.csv')
r = np.corrcoef(yr.drought, yr.fail)[0, 1]
b1, b0 = np.polyfit(yr.drought, yr.fail, 1)
rows = []
for crop in ['groundnut', 'millet', 'sorghum', 'maize']:
    s = d[d.crop == crop]
    a, b = s[s.drought == 1].fail.mean(), s[s.drought == 0].fail.mean()
    rows.append(dict(crop=crop, fail_given_drought=a, fail_given_not_drought=b, relative_risk=a / b, records=len(s)))
a, b = d[d.drought == 1].fail.mean(), d[d.drought == 0].fail.mean()
rows.append(dict(crop='all crops', fail_given_drought=a, fail_given_not_drought=b, relative_risk=a / b, records=len(d)))
RR = pd.DataFrame(rows)
RR.to_csv(f'{OUT}/s31_risk_by_crop.csv', index=False)
print(f'OBS 1986-2016: {O.shape[1]} departments; r(yearly drought share, failure share) = {r:.2f}; slope {b1:.2f}')
print(RR.round(2).to_string(index=False))
print((100 * yr[(yr.drought >= .4) | (yr.fail >= .4)]).round(1).to_string())
# reference observed coupling 1985-2014 and Schaake template
Oh = O.loc[1985:2014]
so, _ = stats(Oh, Oh.quantile(.2))
Rk = dtr(Oh).rank(method='first').astype(int).values - 1
print('Observed 1985-2014:', {k: round(v, 3) for k, v in so.items()})


def schaake(X):
    return pd.DataFrame(np.take_along_axis(np.sort(X.values, axis=0), Rk, axis=0), index=X.index, columns=X.columns)


# ------------------------------------------------------------ CMIP6
rows, maps, hdiff = [], [], []
for m in MODELS:
    H0 = None
    for ssp in ['ssp245', 'ssp585']:
        S = pd.read_csv(f'{DD}/dept_spei3_{m}_{ssp}.csv', index_col=0)[O.columns]
        H = S.loc[1985:2014]
        if H0 is not None:
            hdiff.append(float(np.abs(H.values - H0.values).max()))
        H0 = H
        thr = H.quantile(.2)
        for w, (a, b) in WIN.items():
            X = S.loc[a:b]
            assert len(X) == 30 and X.notna().all().all()
            st, _ = stats(X, thr)
            st2, _ = stats(schaake(X), thr)
            rows.append(dict(model=m, ssp=ssp, window=w, version='raw', **st))
            rows.append(dict(model=m, ssp=ssp, window=w, version='schaake', **st2))
            if w == 'far':
                for dep, v in (X <= thr).mean().items():
                    maps.append(dict(model=m, ssp=ssp, dept=dep, freq=100 * v))
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/s31_cmip6_by_model.csv', index=False)
print('max |SPEI hist ssp245 - ssp585| per model:', [round(v, 2) for v in hdiff])
# Table 7: reference column = historical window of the SSP2-4.5 files; increases counted against own historical
raw = R[R.version == 'raw'].set_index(['model', 'ssp', 'window'])
met = ['exposure', 'ws40', 'ws60', 'severe', 'phi', 'r']
T = {}
h = raw.xs(('ssp245', 'hist'), level=('ssp', 'window'))[met]
T['Reference'] = {k: (h[k].median(), h[k].quantile(.1), h[k].quantile(.9), None) for k in met}
for ssp in ['ssp245', 'ssp585']:
    hh = raw.xs((ssp, 'hist'), level=('ssp', 'window'))[met]
    for w in ['near', 'far']:
        f = raw.xs((ssp, w), level=('ssp', 'window'))[met]
        T[f'{ssp[3:]} {w}'] = {k: (f[k].median(), f[k].quantile(.1), f[k].quantile(.9), int((f[k] > hh[k]).sum())) for k in met}
rows7 = []
for k in met:
    rows7.append({'metric': k, **{c: v[k] for c, v in T.items()}})
pd.to_pickle(T, f'{OUT}/s31_table7.pkl')
for k in met:
    print(f'{k:9s}', ' | '.join(f'{c}: {v[k][0]:.2f} [{v[k][1]:.2f}-{v[k][2]:.2f}]' + (f' ({v[k][3]}/9)' if v[k][3] is not None else '') for c, v in T.items()))
# Table S4: Schaake summary
S4 = R.groupby(['ssp', 'window', 'version']).agg(median_phi=('phi', 'median'), median_ws40=('ws40', 'median'),
                                                p10_ws40=('ws40', lambda s: s.quantile(.1)), p90_ws40=('ws40', lambda s: s.quantile(.9))).reset_index()
S4.to_csv(f'{OUT}/s31_tableS4_schaake.csv', index=False)
print(S4.round(2).to_string(index=False))
# Fig 10 maps
M = pd.DataFrame(maps)
Mm = M.groupby(['ssp', 'dept']).agg(freq_median=('freq', 'median'), n_increase=('freq', lambda s: int((s > 20).sum()))).reset_index()
Mm.to_csv(f'{OUT}/s31_dept_freq_far.csv', index=False)
print(Mm.groupby('ssp').freq_median.describe().round(1))
pd.Series({'obs_phi': so['phi'], 'obs_r': so['r'], 'obs_ws40': so['ws40'], 'obs_ws60': so['ws60'], 'obs_severe': so['severe'],
           'r_yearly': r, 'slope': b1, 'intercept': b0}).to_csv(f'{OUT}/s31_obs_summary.csv')
