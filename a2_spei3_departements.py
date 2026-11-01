"""Department SPEI-3 (as described in the manuscript, section 2.8), observations and 9 CMIP6 models.

For each retained department x crop series (masque_mirca.py: MIRCA2000 rainfed-area weights):
  daily PET (Makkink; Rs from Hargreaves-Samani for observations, bias-adjusted radiation for models),
  monthly water balance P - PET on a complete calendar (a month with a missing day is missing).
Department monthly balance = mean of its retained crop balances weighted by MIRCA2000 rainfed harvested area.
3-month sum; for each calendar month a log-logistic distribution (generalized-logistic form) fitted by
unbiased L-moments on 1985-2014 (SPEI R package); SPEI bounded to +-3.72. Seasonal value = JJAS minimum.
Same daily procedure for observations and models.

Usage: python a2_spei3_departements.py <folder with *_bycrop.csv.gz and surfaces_mirca_departements.csv> <out>
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spei_makkink import rs_hargreaves, makkink, spei_from_wb3, seasonal_min_jjas

D, OUT = sys.argv[1], sys.argv[2]
GEO = os.environ.get('GEOJSON', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'departements_1980.geojson'))
REF = range(1985, 2015)
MODELS = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'FGOALS-g3',
          'GFDL-ESM4', 'MIROC6', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']
CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
os.makedirs(OUT, exist_ok=True)

gj = json.load(open(GEO, encoding='utf-8'))
LAT = {}
for ft in gj['features']:
    g = ft['geometry']
    polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    LAT[ft['properties']['departement']] = float(np.vstack([np.asarray(q[0])[:, :2] for q in polys])[:, 1].mean())
S = pd.read_csv(os.path.join(D, 'surfaces_mirca_departements.csv'))
S = S[S.traitee]
W = {f'{r.departement}_{r.culture}': r.surface_mirca_ha for r in S.itertuples()}


CAP = float(os.environ.get('CAP_PR', 'inf'))      # sensitivity test only: cap daily model rainfall


def load(name):
    df = pd.read_csv(os.path.join(D, name), index_col=0)
    df.index = df.index.astype(str)
    df = df.astype(float)
    if '_pr_' in name and not name.startswith('obs_'):
        df = df.clip(upper=CAP)
    return df


def monthly_wb(p, tmin, tmax, rs=None):
    idx = p.index.intersection(tmin.index).intersection(tmax.index)
    p, tmin, tmax = p.loc[idx], tmin.loc[idx], tmax.loc[idx]
    if rs is None:
        doy = CUM[idx.str[5:7].astype(int).values - 1] + idx.str[8:10].astype(int).values
        rs = pd.DataFrame({c: rs_hargreaves(tmax[c].values, tmin[c].values, LAT[c.rsplit('_', 1)[0]], doy)
                           for c in p.columns}, index=idx)
    else:
        rs = rs.loc[idx]
    pet = makkink(((tmin + tmax) / 2).values, rs[p.columns].values)
    wb = pd.DataFrame(p.values - pet, index=idx, columns=p.columns)
    key = pd.to_datetime(pd.Index(idx).str[:7] + '-01')
    g = wb.groupby(key)
    m = g.sum()
    m[g.apply(lambda q: q.isna().any())] = np.nan
    m[g.size() < 28] = np.nan
    return m.reindex(pd.date_range(m.index.min(), m.index.max(), freq='MS'))


def department_spei(m, tag):
    cols = [c for c in m.columns if c in W]
    deps = sorted(set(c.rsplit('_', 1)[0] for c in cols))
    dep = pd.DataFrame({d: sum(m[c] * W[c] for c in cols if c.startswith(d + '_')) /
                        sum(W[c] for c in cols if c.startswith(d + '_')) for d in deps})
    wb3 = dep.rolling(3, min_periods=3).sum()
    spei, fit = spei_from_wb3(wb3, REF)
    fit.insert(0, 'source', tag)
    sea = seasonal_min_jjas(spei)
    sea.index.name = 'year'
    return sea, fit, spei


fits = []
m = monthly_wb(load('obs_ARC2_pr_bycrop.csv.gz'), load('obs_CHIRTS_tasmin_bycrop.csv.gz'), load('obs_CHIRTS_tasmax_bycrop.csv.gz'))
sea, fit, _ = department_spei(m, 'observations')
fits.append(fit)
sea.loc[:2016].to_csv(f'{OUT}/dept_spei3_obs.csv', float_format='%.4f')
print('observations', sea.shape, 'mean 1985-2014 %.2f' % np.nanmean(sea.loc[1985:2014].values))
for mo in MODELS:
    for ssp in ['ssp245', 'ssp585']:
        X = {v: load(f'{mo}_{ssp}_{v}_bycrop.csv.gz') for v in ['pr', 'tasmin', 'tasmax', 'srad']}
        m = monthly_wb(X['pr'], X['tasmin'], X['tasmax'], X['srad'])
        sea, fit, _ = department_spei(m.loc['1984-01-01':], f'{mo}_{ssp}')
        fits.append(fit)
        sea.loc[1985:].to_csv(f'{OUT}/dept_spei3_{mo}_{ssp}.csv', float_format='%.4f')
        print(mo, ssp, 'mean 1985-2014 %.2f | 2071-2100 %.2f' % (np.nanmean(sea.loc[1985:2014].values), np.nanmean(sea.loc[2071:2100].values)))
F = pd.concat(fits)
F.to_csv(f'{OUT}/dept_spei3_fits.csv', index=False)
J = F[F.month.isin([6, 7, 8, 9])]
print('fits Jun-Sep:', len(J), J.status.value_counts().to_dict(), 'k range %.2f..%.2f' % (J.k.min(), J.k.max()))
