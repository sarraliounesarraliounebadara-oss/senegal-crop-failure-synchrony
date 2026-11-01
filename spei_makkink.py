"""SPEI-3 computed from daily department x crop series (MIRCA2000-weighted), from raw variables.

1. Daily solar radiation Rs (MJ m-2 d-1)
   - observations: Hargreaves & Samani (1985), Rs = kRs * sqrt(Tmax - Tmin) * Ra, kRs = 0.16 (FAO-56 eq. 50),
     Ra = extraterrestrial radiation (FAO-56 eq. 21) at the department centroid latitude;
   - CMIP6: bias-adjusted surface radiation of Mbengue et al. (2026), converted to MJ m-2 d-1.
2. Daily Makkink (1957) PET with its original coefficients (Xu & Singh 2002):
   PET = 0.61 * D/(D+g) * Rs/lambda - 0.12 (mm d-1), floored at 0.
3. Monthly water balance WB = P - PET on a complete calendar (a month with a missing day is missing);
   SPEI-3 uses the 3-month running sum ending in the month.
4. Standardisation (Vicente-Serrano et al. 2010; Begueria et al. 2014): for each calendar month, the
   log-logistic distribution in its generalized-logistic form (Hosking & Wallis 1997) is fitted by unbiased
   probability-weighted moments (L-moments) on the reference period, as in the SPEI R package
   (parglo / pwm.ub). SPEI = Phi^-1(F), F bounded to [1e-4, 1 - 1e-4] (|SPEI| <= 3.72).
   All years of a series use the parameters fitted on its own reference period.
5. Seasonal index: minimum of monthly SPEI-3 over June-September.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, theilslopes

LAMBDA = 2.45          # latent heat of vaporisation, MJ kg-1
GAMMA_PSY = 0.0665     # psychrometric constant, kPa degC-1 (FAO-56, z ~ 0-100 m)
KRS = 0.16             # Hargreaves-Samani radiation coefficient, interior locations (FAO-56)
P_BOUND = 1e-4
JJAS = [6, 7, 8, 9]


# ------------------------------------------------------------------ radiation and PET
def ra_fao56(lat_deg, doy):
    """Extraterrestrial radiation, MJ m-2 d-1 (FAO-56 eqs. 21-25)."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def rs_hargreaves(tmax, tmin, lat_deg, doy, krs=KRS):
    return krs * np.sqrt(np.clip(tmax - tmin, 0, None)) * ra_fao56(lat_deg, doy)


def makkink(tmean, rs):
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    delta = 4098 * es / (tmean + 237.3) ** 2
    return np.clip(0.61 * delta / (delta + GAMMA_PSY) * rs / LAMBDA - 0.12, 0, None)


# ------------------------------------------------------------------ distribution
def lmoments_ub(x):
    """First three sample L-moments from unbiased PWMs b_r = E[x F^r] (Hosking 1990)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def glo_fit(x):
    """Generalized logistic (Hosking) parameters (xi, alpha, k) from L-moments. Defined for any |tau3| < 1."""
    l1, l2, l3 = lmoments_ub(x)
    if not l2 > 0:
        return np.nan, np.nan, np.nan
    k = -l3 / l2
    if abs(k) < 1e-6:
        return l1, l2, 0.0
    kpi = k * np.pi
    alpha = l2 * np.sin(kpi) / kpi
    xi = l1 - alpha * (1 / k - np.pi / np.sin(kpi))
    return xi, alpha, k


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


def spei_from_wb3(wb3, ref_years, min_ref=25):
    """wb3: DataFrame (monthly index on a complete calendar, columns = series).
    Returns SPEI-3 and a parameter table (series, month, xi, alpha, k, tau3, n_ref, share_out_of_support)."""
    out = pd.DataFrame(np.nan, index=wb3.index, columns=wb3.columns)
    rows = []
    ref_years = set(ref_years)
    in_ref = wb3.index.year.isin(ref_years)
    for m in range(1, 13):
        sel = wb3.index.month == m
        for c in wb3.columns:
            xr_ = wb3.loc[sel & in_ref, c].dropna().values
            if len(xr_) < min_ref:
                continue
            xi, al, k = glo_fit(xr_)
            if not np.isfinite([xi, al, k]).all():
                rows.append(dict(series=c, month=m, xi=xi, alpha=al, k=k, n_ref=len(xr_), status='no_spread'))
                continue
            v = wb3.loc[sel, c].values
            f = glo_cdf(v, xi, al, k)
            outside = np.nanmean((f <= P_BOUND) | (f >= 1 - P_BOUND))
            out.loc[sel, c] = norm.ppf(np.clip(f, P_BOUND, 1 - P_BOUND))
            rows.append(dict(series=c, month=m, xi=xi, alpha=al, k=k, n_ref=len(xr_),
                             share_bounded=outside, status='ok'))
    return out, pd.DataFrame(rows)


def seasonal_min_jjas(spei):
    s = spei[spei.index.month.isin(JJAS)]
    g = s.groupby(s.index.year)
    v = g.min()
    v[g.count() < len(JJAS)] = np.nan          # a season needs its four monthly values
    return v


# ------------------------------------------------------------------ aggregation
def daily_to_wb3(p, pet):
    """p, pet: daily DataFrames on the same index of 'YYYY-MM-DD' strings (any calendar).
    Monthly sums are kept only for months without missing values; the monthly series is placed on a
    complete calendar before the 3-month running sum, so gaps never merge non-consecutive months."""
    wb = pd.DataFrame(p.values - pet.values, index=p.index, columns=p.columns)
    key = pd.to_datetime(pd.Index(p.index).str[:7] + '-01')
    g = wb.groupby(key)
    s = g.sum()
    s[g.apply(lambda d: d.isna().any())] = np.nan
    s[g.size() < 28] = np.nan
    full = pd.date_range(s.index.min(), s.index.max(), freq='MS')
    s = s.reindex(full)
    return s.rolling(3, min_periods=3).sum()


def detrend_theilsen(s, years):
    """Remove a Theil-Sen linear trend, keeping the series mean (observed climate indicators)."""
    s = np.asarray(s, float)
    years = np.asarray(years, float)
    ok = np.isfinite(s)
    if ok.sum() < 10:
        return s * np.nan
    b, a, *_ = theilslopes(s[ok], years[ok])
    return s - (a + b * years) + np.nanmean(s)


if __name__ == '__main__':
    # ---- self-tests
    import lmoments3.distr as LD
    from scipy.stats import fisk, skewnorm
    rng = np.random.default_rng(0)
    worst = 0
    for kind, x in [('log-logistic (pos. skew)', fisk.rvs(c=6, loc=-150, scale=400, size=30, random_state=1)),
                    ('negative skew', skewnorm.rvs(-4, loc=100, scale=80, size=30, random_state=2)),
                    ('near-symmetric', rng.normal(0, 50, 30))]:
        xi, al, k = glo_fit(x)
        ref = LD.glo.lmom_fit(x)
        f1 = glo_cdf(x, xi, al, k)
        f2 = LD.glo.cdf(x, **ref)
        worst = max(worst, np.abs(f1 - f2).max())
        z = norm.ppf(np.clip(f1, P_BOUND, 1 - P_BOUND))
        print(f'{kind:26s} k={k:+.3f} (lmoments3 {ref["k"]:+.3f})  max|F-F_ref|={np.abs(f1 - f2).max():.1e}  '
              f'SPEI mean={z.mean():+.2f} sd={z.std(ddof=1):.2f}')
    big = fisk.rvs(c=6, loc=-150, scale=400, size=20000, random_state=3)
    z = norm.ppf(np.clip(glo_cdf(big, *glo_fit(big)), P_BOUND, 1 - P_BOUND))
    print(f'large sample: SPEI mean={z.mean():+.3f} sd={z.std():.3f}, max CDF gap with lmoments3 = {worst:.1e}')
    print('Ra at 14.5N on 15 Jul:', round(float(ra_fao56(14.5, 196)), 2), 'MJ m-2 d-1')
    print('Makkink at 28 degC, Rs=20:', round(float(makkink(28.0, 20.0)), 2), 'mm d-1')
    # aggregation: a gap must not join non-consecutive months
    d = pd.date_range('2000-04-01', '2001-11-30').strftime('%Y-%m-%d')
    keep = [x for x in d if int(x[5:7]) in range(4, 12)]
    P = pd.DataFrame({'a': 1.0}, index=keep); E = pd.DataFrame({'a': 0.0}, index=keep)
    w = daily_to_wb3(P, E)
    print('wb3 Apr-2001 (must be NaN, Feb-Mar missing):', w.loc['2001-04-01', 'a'], '| Jun-2001 (=91):', w.loc['2001-06-01', 'a'])
