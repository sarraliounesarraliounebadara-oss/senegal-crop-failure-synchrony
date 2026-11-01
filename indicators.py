"""Seasonal agro-climatic indicators per department x crop series (shared by obs and CMIP6).
Onset: Sivakumar (1988) / ANACIM - first day from 1 May with >= 20 mm in 1-3 days (day 1 wet),
no dry spell > 7 days in the next 30 days; searched until 30 September."""
import numpy as np
import pandas as pd

WET = 1.0


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


def onset(p):
    n = len(p)
    for t in range(n):
        if p[t] < WET or p[t:t + 3].sum() < 20:
            continue
        r = _runs(p[t + 1:min(t + 31, n)] < WET)
        if r and max(r) > 7:
            continue
        return t
    return None


def load_daily(path):
    df = pd.read_csv(path, index_col=0)
    idx = pd.Index(df.index.astype(str))
    vals = df.values.astype('f8')
    if 'pr' in path and np.nanmax(vals) < 5:          # kg m-2 s-1 -> mm/day
        vals = vals * 86400
    if ('tas' in path or 'Tm' in path) and np.nanmean(vals) > 200:   # K -> degC
        vals = vals - 273.15
    return pd.DataFrame(vals, index=idx, columns=df.columns)


def rain_indicators(path, y0, y1):
    df = load_daily(path)
    yr = df.index.str[:4].astype(int)
    md = df.index.str[5:]
    rows = []
    for y in range(y0, y1 + 1):
        sel = (yr == y) & (md >= '05-01') & (md <= '10-31')
        b = df[sel]
        if len(b) < 180:
            continue
        bmd = np.array(md[sel])
        jjas = (bmd >= '06-01') & (bmd <= '09-30')
        so = (bmd >= '09-01')
        upto = int((bmd <= '09-30').sum())          # onset / dry spells searched up to 30 Sep
        for col in df.columns:
            pall = b[col].values
            if np.isnan(pall).any():
                continue
            p = pall[:upto]
            o = onset(p)
            if o is None:
                ds, od = 120, 274
            else:
                r = _runs(p[o:] < WET)
                ds, od = (max(r) if r else 0), pd.Timestamp(2001, 5, 1).day_of_year + o
            dep, crop = col.rsplit('_', 1)
            rows.append(dict(dept=dep, crop=crop, year=y, P_JJAS=pall[jjas].sum(), P_SO=pall[so].sum(),
                             onset_doy=od, maxDS=ds))
    return pd.DataFrame(rows)


def heat_indicators(path, y0, y1, thr=35.0):
    df = load_daily(path)
    yr = df.index.str[:4].astype(int)
    md = df.index.str[5:]
    sel = (md >= '07-01') & (md <= '09-30') & (yr >= y0) & (yr <= y1)
    b = df[sel]
    g = b.groupby(yr[sel])
    hot = (b > thr).groupby(yr[sel]).sum()
    mean = g.mean()
    out = []
    for col in df.columns:
        dep, crop = col.rsplit('_', 1)
        out.append(pd.DataFrame({'dept': dep, 'crop': crop, 'year': mean.index,
                                 'Tmax_JAS': mean[col].values, 'hot_days_JAS': hot[col].values}))
    return pd.concat(out, ignore_index=True)


def hazards(ind, base, q=0.2):
    """Hazard flags relative to each series' own baseline quantiles (base: (y0, y1))."""
    ind = ind.copy()
    b = ind[ind.year.between(*base)].groupby(['dept', 'crop'])
    thr = pd.DataFrame({'P_lo': b.P_JJAS.quantile(q), 'SO_lo': b.P_SO.quantile(q), 'on_hi': b.onset_doy.quantile(1 - q),
                        'ds_hi': b.maxDS.quantile(1 - q)})
    if 'Tmax_JAS' in ind:
        thr['T_hi'] = b.Tmax_JAS.quantile(1 - q)
    ind = ind.join(thr, on=['dept', 'crop'])
    ind['dry'] = (ind.P_JJAS <= ind.P_lo).astype(int)
    ind['late'] = (ind.onset_doy >= ind.on_hi).astype(int)
    ind['spell'] = (ind.maxDS >= ind.ds_hi).astype(int)
    ind['end'] = (ind.P_SO <= ind.SO_lo).astype(int)
    ind['rain_hazard'] = ((ind.dry + ind.late + ind.spell + ind.end) > 0).astype(int)
    if 'T_hi' in ind:
        ind['hot'] = (ind.Tmax_JAS >= ind.T_hi).astype(int)
        ind['hot_dry'] = (ind.hot & ind.dry).astype(int)
    return ind


def loreau(X):
    X = X[:, np.all(np.isfinite(X), 0)]
    return X.sum(1).var(ddof=1) / (X.std(0, ddof=1).sum() ** 2)
