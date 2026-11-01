"""How much of the synchrony is climatic? (nested leave-one-year-out: standardisation re-estimated without the held-out year)
Panel model per crop: relative yield anomaly ~ standardised indicators of the same series-year, computed from
ARC2 with the MIRCA2000 rainfed-area weights of the crop and detrended (Theil-Sen, 1986-2016; s21):
JJAS rain, its square, Sep-Oct rain, onset date, longest dry spell (capped at 30 days). Climate component = leave-one-year-out
prediction (all departments of the held-out year removed); residual = observed - climate component."""
import sys, numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
rng = np.random.default_rng(11)
CR = ['groundnut', 'millet', 'sorghum', 'maize']
y = pd.read_csv(f'{OUT}/yields_detrended.csv'); y = y[y.rel_anom.notna()].copy()
ind = pd.read_csv(f'{OUT}/s21_obs_indicators.csv')
d = y.merge(ind, on=['dept', 'crop', 'year'])
F = ['P_JJAS', 'P_SO', 'onset_doy', 'maxDS']
for f in F:
    d[f + '_z'] = d.groupby(['dept', 'crop'])[f + '_dt'].transform(lambda s: (s - s.mean()) / s.std())
d['P2_z'] = d.P_JJAS_z ** 2
X_cols = [f + '_z' for f in F] + ['P2_z']
d['clim'] = np.nan
rows = []
def zfit(s, train):
    """z-scores of the detrended indicators using the means and SDs of the training rows of each series."""
    g = s[train].groupby(['dept', 'crop'])
    out = {}
    for f in F:
        mu = s[['dept', 'crop']].merge(g[f + '_dt'].mean().rename('mu').reset_index(), how='left').mu.values
        sd = s[['dept', 'crop']].merge(g[f + '_dt'].std().rename('sd').reset_index(), how='left').sd.values
        out[f + '_z'] = (s[f + '_dt'].values - mu) / sd
    out['P2_z'] = out['P_JJAS_z'] ** 2
    return np.column_stack([np.ones(len(s))] + [out[c] for c in X_cols])


for crop in CR:
    s = d[d.crop == crop]
    X = np.column_stack([np.ones(len(s)), s[X_cols].values]); yv = s.rel_anom.values; yrs = s.year.values
    pred = np.zeros(len(s))
    for t in np.unique(yrs):
        te = yrs == t
        Xn = zfit(s, ~te)                       # standardisation re-estimated without the held-out year
        b, *_ = np.linalg.lstsq(Xn[~te], yv[~te], rcond=None); pred[te] = Xn[te] @ b
    d.loc[s.index, 'clim'] = pred
    bfull, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r2_in = 1 - ((yv - X @ bfull) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    r2_cv = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    rows.append(dict(crop=crop, n=len(s), R2_in=r2_in, R2_cv=r2_cv, **{f'b_{c}': v for c, v in zip(['const'] + X_cols, bfull)}))
M = pd.DataFrame(rows); M.to_csv(f'{OUT}/s32_panel_models.csv', index=False)
print('Panel models (rel. anomaly ~ rainfall indicators):'); print(M.round(3).to_string(index=False))
d['resid'] = d.rel_anom - d.clim
for c in ['rel_anom', 'clim', 'resid']:
    d['shock_' + c] = d[c] * d.trend_kg_ha * d.median_area_ha / 1000
d.to_csv(f'{OUT}/s32_decomposition_panel.csv', index=False)

def loreau(X): return X.sum(1).var(ddof=1) / (X.std(0, ddof=1).sum() ** 2)
out = []
for crop in CR:
    s = d[d.crop == crop]
    P = {c: s.pivot(index='year', columns='dept', values='shock_' + c).drop(index=2004, errors='ignore').dropna(axis=1) for c in ['rel_anom', 'clim', 'resid']}
    cols = P['rel_anom'].columns.intersection(P['clim'].columns)
    T, Cc, Rr = (P[k][cols].sum(1) for k in ['rel_anom', 'clim', 'resid'])
    nul = []
    for _ in range(2000):
        Z = np.column_stack([rng.permutation(P['resid'][c].values) for c in cols]); nul.append(loreau(Z))
    out.append(dict(crop=crop, n_depts=len(cols), phi_total=loreau(P['rel_anom'][cols].values), phi_climate=loreau(P['clim'][cols].values),
                    phi_residual=loreau(P['resid'][cols].values), phi_resid_null95=np.percentile(nul, 95),
                    var_share_climate=Cc.var() / T.var(), var_share_residual=Rr.var() / T.var(), corr_national_total_climate=np.corrcoef(T, Cc)[0, 1]))
O = pd.DataFrame(out); O.to_csv(f'{OUT}/s32_synchrony_decomposition.csv', index=False)
print('\nSynchrony decomposition (production shocks, 2004 removed, complete departments):'); print(O.round(2).to_string(index=False))
# failure synchrony of residuals vs observed (all crops)
def var_ratio(col, q=.2):
    t = d.copy(); t['f'] = (t[col] <= t.groupby(['dept', 'crop'])[col].transform(lambda s: s.quantile(q))).astype(int)
    P = t.pivot_table(index=['dept', 'crop'], columns='year', values='f'); M_ = P.notna().values; Fv = np.nan_to_num(P.values)
    fo = Fv.sum(0) / M_.sum(0); vn = []
    for _ in range(2000):
        Z = np.zeros_like(Fv)
        for i in range(len(Fv)):
            ix = np.where(M_[i])[0]; Z[i, ix] = rng.permutation(Fv[i, ix])
        vn.append(np.var(Z.sum(0) / M_.sum(0)))
    return np.var(fo) / np.mean(vn), pd.Series(fo, index=P.columns)
vr_obs, fo_obs = var_ratio('rel_anom'); vr_res, fo_res = var_ratio('resid'); vr_cli, fo_cli = var_ratio('clim')
print(f'\nFailure-synchrony variance ratio: observed {vr_obs:.1f} | climate component {vr_cli:.1f} | residual {vr_res:.1f}')
yr = pd.DataFrame({'share_fail_observed': fo_obs, 'share_fail_climate': fo_cli, 'share_fail_residual': fo_res})
nat = d.groupby('year')[['shock_rel_anom', 'shock_clim', 'shock_resid']].sum() / 1000
yr = yr.join(nat.rename(columns={'shock_rel_anom': 'national_shock_kt', 'shock_clim': 'climate_part_kt', 'shock_resid': 'residual_part_kt'}))
yr.to_csv(f'{OUT}/s32_yearly_decomposition.csv')
print('\nWidespread years: what climate explains (all crops):')
print(yr[yr.share_fail_observed >= .35].round(2).to_string())
print('\nLargest negative residual (non-climatic) national shocks:'); print(yr.residual_part_kt.nsmallest(6).round(0).to_string())
