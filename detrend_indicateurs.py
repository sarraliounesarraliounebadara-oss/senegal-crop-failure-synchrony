#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Indicateurs climatiques saisonniers observés par département x culture (masque MIRCA2000), puis
détrendage Theil-Sen de chaque série sur 1986-2016 (la moyenne de la série est conservée), comme pour
les rendements. Les projections CMIP6 ne sont pas détrendées : le changement est le signal étudié.

Entrées (dans --dir, sorties de masque_mirca.py) :
    obs_ARC2_pr_bycrop.csv.gz, obs_CHIRTS_tasmin_bycrop.csv.gz, obs_CHIRTS_tasmax_bycrop.csv.gz
    departements_1980.geojson (latitude des départements pour le rayonnement, chemin GEOJSON)

Indicateurs (une valeur par série et par saison, 1986-2016) :
    P_JJAS      pluie juin-septembre (mm)
    P_SO        pluie septembre-octobre (mm)
    onset_doy   démarrage (Sivakumar 1988) : premier jour à partir du 1er mai avec >= 20 mm en 1-3 jours
                (premier jour pluvieux >= 1 mm) sans séquence sèche > 7 jours dans les 30 jours suivants,
                cherché jusqu'au 30 septembre (sinon 274 = 1er octobre)
    maxDS       plus longue séquence sèche (< 1 mm) entre le démarrage et le 30 septembre, plafonnée à 30 j
    Tmax_JAS    Tmax moyenne juillet-septembre (degC)
    SPEI3       minimum du SPEI-3 mensuel de juin à septembre :
                Rs Hargreaves-Samani (kRs = 0,16 ; Ra FAO-56), ETP Makkink (1957),
                bilan P - ETP mensuel sur calendrier complet (mois avec un jour manquant = manquant),
                cumul sur 3 mois, loi log-logistique (forme logistique généralisée) ajustée par mois
                calendaire par L-moments non biaisés sur 1985-2014, comme le package R SPEI.

Sorties (dans --dir) :
    indicateurs_obs_1986-2016.csv   valeurs brutes et détrendées (suffixe _dt)
    spei3_obs_jjasmin.csv           SPEI-3 saisonnier brut, 1983-2016
    tendances_retirees.csv          pente Theil-Sen retirée (par décennie), résumé sur les séries

Lancement :  python detrend_indicateurs.py
Dépendances : numpy, pandas, scipy.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, theilslopes

p = argparse.ArgumentParser()
GEOJSON = '/mnt/disque/departments/climate_data/cmip6/departement_means/departements_1980.geojson'
p.add_argument('--dir', default='/mnt/disque/departments/climate_data/masque_mirca')
p.add_argument('--debut', type=int, default=1986)
p.add_argument('--fin', type=int, default=2016)
args = p.parse_args()
D = args.dir
Y0, Y1 = args.debut, args.fin
REF = range(1985, 2015)          # période de référence du SPEI (commune aux observations et aux modèles)
WET = 1.0
LAMBDA, GAMMA_PSY, KRS, P_BOUND = 2.45, 0.0665, 0.16, 1e-4
CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


# ================================================================ lecture
def load(name):
    df = pd.read_csv(os.path.join(D, name), index_col=0)
    df.index = df.index.astype(str)
    v = df.astype(float)
    if 'pr' in name and np.nanmax(v.values) < 5:
        v = v * 86400
    if 'tas' in name and np.nanmean(v.values) > 200:
        v = v - 273.15
    return v


def dept_latitudes():
    f = GEOJSON if os.path.exists(GEOJSON) else os.path.join(D, 'departements_1980.geojson')
    gj = json.load(open(f, encoding='utf-8'))
    lat = {}
    for ft in gj['features']:
        g = ft['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        pts = np.vstack([np.asarray(q[0])[:, :2] for q in polys])
        lat[ft['properties']['departement']] = float(pts[:, 1].mean())
    return lat


# ================================================================ indicateurs pluie et chaleur
def _runs(dry):
    out, cur = [], 0
    for d in dry:
        if d:
            cur += 1
        else:
            if cur:
                out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def onset(pp):
    n = len(pp)
    for t in range(n):
        if pp[t] < WET or pp[t:t + 3].sum() < 20:
            continue
        r = _runs(pp[t + 1:min(t + 31, n)] < WET)
        if r and max(r) > 7:
            continue
        return t
    return None


def rain_indicators(df):
    yr = df.index.str[:4].astype(int)
    md = df.index.str[5:]
    rows = []
    for y in range(Y0, Y1 + 1):
        sel = (yr == y) & (md >= '05-01') & (md <= '10-31')
        b = df[sel]
        if len(b) < 180:
            continue
        bmd = np.array(md[sel])
        jjas = (bmd >= '06-01') & (bmd <= '09-30')
        so = bmd >= '09-01'
        upto = int((bmd <= '09-30').sum())
        for col in df.columns:
            pall = b[col].values
            if np.isnan(pall).any():
                continue
            pp = pall[:upto]
            o = onset(pp)
            if o is None:
                ds, od = 120, 274
            else:
                r = _runs(pp[o:] < WET)
                ds, od = (max(r) if r else 0), pd.Timestamp(2001, 5, 1).day_of_year + o
            dep, crop = col.rsplit('_', 1)
            rows.append(dict(dept=dep, crop=crop, year=y, P_JJAS=pall[jjas].sum(), P_SO=pall[so].sum(),
                             onset_doy=od, maxDS=min(ds, 30)))
    return pd.DataFrame(rows)


def heat_indicator(df):
    yr = df.index.str[:4].astype(int)
    md = df.index.str[5:]
    sel = (md >= '07-01') & (md <= '09-30') & (yr >= Y0) & (yr <= Y1)
    m = df[sel].groupby(yr[sel]).mean()
    m.index.name = 'year'
    m.columns.name = 'series'
    t = m.stack().rename('Tmax_JAS').reset_index()
    t[['dept', 'crop']] = t.series.str.rsplit('_', n=1, expand=True)
    return t[['dept', 'crop', 'year', 'Tmax_JAS']]


# ================================================================ SPEI-3
def ra_fao56(lat_deg, doy):
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def makkink(tmean, rs):
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    delta = 4098 * es / (tmean + 237.3) ** 2
    return np.clip(0.61 * delta / (delta + GAMMA_PSY) * rs / LAMBDA - 0.12, 0, None)


def glo_fit(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    if not l2 > 0:
        return np.nan, np.nan, np.nan
    k = -l3 / l2
    if abs(k) < 1e-6:
        return l1, l2, 0.0
    kpi = k * np.pi
    alpha = l2 * np.sin(kpi) / kpi
    return l1 - alpha * (1 / k - np.pi / np.sin(kpi)), alpha, k


def glo_cdf(x, xi, alpha, k):
    x = np.asarray(x, float)
    if k == 0:
        y = (x - xi) / alpha
    else:
        arg = 1 - k * (x - xi) / alpha
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.where(arg > 0, -np.log(np.where(arg > 0, arg, 1)) / k, np.inf if k > 0 else -np.inf)
    with np.errstate(over='ignore'):
        f = 1 / (1 + np.exp(-y))
    return np.where(np.isfinite(x), f, np.nan)


def seasonal_spei(pr, tmin, tmax, lat):
    idx = pr.index.intersection(tmin.index).intersection(tmax.index)
    cols = pr.columns.intersection(tmin.columns).intersection(tmax.columns)
    pr, tmin, tmax = pr.loc[idx, cols], tmin.loc[idx, cols], tmax.loc[idx, cols]
    m = idx.str[5:7].astype(int).values
    dd = idx.str[8:10].astype(int).values
    doy = CUM[m - 1] + dd
    rs = np.column_stack([KRS * np.sqrt(np.clip(tmax[c].values - tmin[c].values, 0, None)) * ra_fao56(lat[c.rsplit('_', 1)[0]], doy)
                          for c in cols])
    pet = makkink(((tmin + tmax) / 2).values, rs)
    wb = pd.DataFrame(pr.values - pet, index=idx, columns=cols)
    key = pd.to_datetime(pd.Index(idx).str[:7] + '-01')
    g = wb.groupby(key)
    s = g.sum()
    s[g.apply(lambda q: q.isna().any())] = np.nan
    s[g.size() < 28] = np.nan
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='MS'))
    wb3 = s.rolling(3, min_periods=3).sum()
    spei = pd.DataFrame(np.nan, index=wb3.index, columns=cols)
    in_ref = wb3.index.year.isin(list(REF))
    n_bad = 0
    for mo in [6, 7, 8, 9]:
        sel = wb3.index.month == mo
        for c in cols:
            xr_ = wb3.loc[sel & in_ref, c].dropna().values
            if len(xr_) < 25:
                continue
            xi, al, k = glo_fit(xr_)
            if not np.isfinite([xi, al, k]).all():
                n_bad += 1
                continue
            f = glo_cdf(wb3.loc[sel, c].values, xi, al, k)
            spei.loc[sel, c] = norm.ppf(np.clip(f, P_BOUND, 1 - P_BOUND))
    j = spei[spei.index.month.isin([6, 7, 8, 9])]
    gg = j.groupby(j.index.year)
    out = gg.min()
    out[gg.count() < 4] = np.nan
    out.index.name = 'year'
    print(f'SPEI-3 : {len(cols)} séries, ajustements impossibles = {n_bad}')
    return out


# ================================================================ détrendage
def detrend_theilsen(s, years):
    s = np.asarray(s, float)
    years = np.asarray(years, float)
    ok = np.isfinite(s)
    if ok.sum() < 10:
        return s * np.nan, np.nan
    b, a, *_ = theilslopes(s[ok], years[ok])
    return s - (a + b * years) + np.nanmean(s), b


if __name__ == '__main__':
    pr = load('obs_ARC2_pr_bycrop.csv.gz')
    tn = load('obs_CHIRTS_tasmin_bycrop.csv.gz')
    tx = load('obs_CHIRTS_tasmax_bycrop.csv.gz')
    ind = rain_indicators(pr)
    ind = ind.merge(heat_indicator(tx), on=['dept', 'crop', 'year'], how='left')
    S = seasonal_spei(pr, tn, tx, dept_latitudes()).loc[:Y1]
    S.to_csv(os.path.join(D, 'spei3_obs_jjasmin.csv'), float_format='%.4f')
    S.columns.name = 'series'
    s = S.loc[Y0:Y1].stack().rename('SPEI3').reset_index()
    s[['dept', 'crop']] = s.series.str.rsplit('_', n=1, expand=True)
    ind = ind.merge(s[['dept', 'crop', 'year', 'SPEI3']], on=['dept', 'crop', 'year'], how='left')
    ind = ind.sort_values(['dept', 'crop', 'year']).reset_index(drop=True)
    VARS = ['P_JJAS', 'P_SO', 'onset_doy', 'maxDS', 'Tmax_JAS', 'SPEI3']
    slopes = []
    for v in VARS:
        ind[v + '_dt'] = np.nan
        for (dep, crop), g in ind.groupby(['dept', 'crop']):
            x, b = detrend_theilsen(g[v].values, g.year.values)
            ind.loc[g.index, v + '_dt'] = x
            slopes.append(dict(indicator=v, dept=dep, crop=crop, slope_per_decade=10 * b))
    ind.to_csv(os.path.join(D, f'indicateurs_obs_{Y0}-{Y1}.csv'), index=False, float_format='%.4f')
    T = pd.DataFrame(slopes).groupby('indicator', sort=False).slope_per_decade.describe(percentiles=[.1, .5, .9])
    T.to_csv(os.path.join(D, 'tendances_retirees.csv'))
    print(f'{ind.groupby(["dept", "crop"]).ngroups} séries x {ind.year.nunique()} saisons')
    print('Pentes Theil-Sen retirées (par décennie) :')
    print(T[['10%', '50%', '90%']].round(3).to_string())
