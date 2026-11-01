#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Masque MIRCA2000 appliqué à la pluie et à la température, observations et modèles CMIP6.

Pour chaque département (limites de 1980, comme DAPSA) et chaque culture, moyenne journalière des
pixels pondérés par la surface pluviale récoltée de la culture (MIRCA2000, RFC : CROP2 maïs, CROP6 mil,
CROP7 sorgho, CROP16 arachide). Un pixel sans la culture a un poids nul ; aucune moyenne non pondérée.

Séries traitées : couples département x culture qui remplissent les deux conditions
  1. surface pluviale récoltée MIRCA2000 >= 1000 ha dans le département (--min-ha) ;
  2. série de rendement retenue dans rendements_detrendes_1986-2016.csv (sortie de detrend.py).
Les départements sans culture ou avec peu de culture disparaissent donc. Les séries de rendement
retenues qui n'atteignent pas le seuil MIRCA sont signalées dans le log.
Surface MIRCA d'un département : cellules de 5' comptées au prorata de leur part dans le département
(7 x 7 points par cellule).

Variables
  observations : pluie ARC2, Tmin et Tmax CHIRTS (fusionnées ANACIM), année entière
  CMIP6 (CDF-t, Mbengue et al. 2026), 9 modèles x SSP2-4.5 / SSP5-8.5 : pr, tasmin, tasmax, et srad
  (le rayonnement des modèles est nécessaire au calcul du SPEI-3 avec Makkink) ; avril-novembre.
Les modèles sont traités un par un (pr, tasmin, tasmax, srad), pour pouvoir calculer le SPEI d'un
modèle dès que ses quatre fichiers sont prêts.

Lancement :
    nohup python masque_mirca.py > masque_mirca.log 2>&1 &
    nohup python masque_mirca.py --jobs 4 > masque_mirca.log 2>&1 &     # 4 fichiers en parallèle
Suivi :
    tail -f masque_mirca.log
Le script reprend là où il s'est arrêté : un fichier déjà produit n'est pas recalculé.

Sorties (dans SORTIE) :
    <nom>_bycrop.csv.gz               une ligne par jour, une colonne par département_culture
    surfaces_mirca_departements.csv   surface MIRCA, série de rendement retenue, série traitée
    rapport_masque.csv                unités, calendrier, période, jours manquants et doublons par fichier
    poids_<grille>.nc                 poids utilisés, pour contrôle
Dépendances : numpy, pandas, xarray, netCDF4 (cftime), matplotlib.
"""
import os, sys, json, time, hashlib, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
try:
    import xarray as xr
    import cftime
    from matplotlib.path import Path
except ImportError as e:
    sys.exit(f"Module manquant ({e}). Installer : conda install -c conda-forge xarray netcdf4 matplotlib")

# ------------------------------------------------------------------ chemins
BASE = '/mnt/disque/departments'
ARC2 = f'{BASE}/climate_data/anacim_data/ARC2_merged.nc'
CHIRTS_TMIN = f'{BASE}/climate_data/anacim_data/CHIRTS_Tmin_merged.nc'
CHIRTS_TMAX = f'{BASE}/climate_data/anacim_data/CHIRTS_Tmax_merged.nc'
CMIP6 = f'{BASE}/climate_data/cmip6'          # <var>/<modele>_<ssp>_<var>Adjust_CDFt-L-1V-0L.nc
MIRCA_DIR = f'{BASE}/agro_data/mask_data/harvested_area_grids'
GEOJSON = f'{BASE}/climate_data/cmip6/departement_means/departements_1980.geojson'
RENDEMENTS = '/home/sarr/Documents/dossier/agro/rendements_detrendes_1986-2016.csv'
SORTIE = f'{BASE}/climate_data/masque_mirca'

# ------------------------------------------------------------------ paramètres
MIRCA_CROPS = {'groundnut': 16, 'maize': 2, 'millet': 6, 'sorghum': 7}
MODELES = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'FGOALS-g3',
           'GFDL-ESM4', 'MIROC6', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']
SSPS = ['ssp245', 'ssp585']
VARS_MODELES = ['pr', 'tasmin', 'tasmax', 'srad']
MOIS_MODELES = range(4, 12)          # avril-novembre suffit pour JJAS, SPEI-3 de juin à septembre et démarrage

p = argparse.ArgumentParser()
p.add_argument('--min-ha', type=float, default=1000.0, help='surface MIRCA minimale par département x culture (ha)')
p.add_argument('--toutes-series', action='store_true', help='ne pas restreindre aux séries de rendement retenues')
p.add_argument('--jobs', type=int, default=1, help='nombre de fichiers traités en parallèle')
p.add_argument('--only', default=None, help='ne traiter que les sorties dont le nom contient ce texte')
p.add_argument('--chunk', type=int, default=366)
args = p.parse_args()

JOBS = [(ARC2, 'obs_ARC2_pr'), (CHIRTS_TMIN, 'obs_CHIRTS_tasmin'), (CHIRTS_TMAX, 'obs_CHIRTS_tasmax')]
for m in MODELES:
    for s in SSPS:
        for v in VARS_MODELES:
            JOBS.append((f'{CMIP6}/{v}/{m}_{s}_{v}Adjust_CDFt-L-1V-0L.nc', f'{m}_{s}_{v}'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------ départements
def load_departments():
    gj = json.load(open(GEOJSON, encoding='utf-8'))
    deps = []
    for ft in gj['features']:
        g = ft['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        ext = [Path(np.asarray(pl[0])[:, :2]) for pl in polys]
        holes = [Path(np.asarray(r)[:, :2]) for pl in polys for r in pl[1:]]
        allpts = np.vstack([np.asarray(pl[0])[:, :2] for pl in polys])
        deps.append((ft['properties']['departement'], ext, holes, allpts.mean(0)))
    deps.sort(key=lambda d: d[0])
    return deps


def inside_dep(pts, ext, holes):
    inside = np.zeros(len(pts), bool)
    for pa in ext:
        inside |= pa.contains_points(pts)
    for h in holes:
        inside &= ~h.contains_points(pts)
    return inside


# ------------------------------------------------------------------ MIRCA
def read_mirca_window(path, lon0=-18.5, lon1=-10.5, lat0=11.5, lat1=17.5):
    hdr = {}
    with open(path) as fh:
        for _ in range(6):
            k, v = fh.readline().split()
            hdr[k.lower()] = float(v)
        nr, cs = int(hdr['nrows']), hdr['cellsize']
        xll = hdr.get('xllcorner', hdr.get('xllcenter', -180) - cs / 2)
        yll = hdr.get('yllcorner', hdr.get('yllcenter', -90) - cs / 2)
        ytop = yll + nr * cs
        r0, r1 = int((ytop - lat1) // cs), int((ytop - lat0) // cs) + 1
        c0, c1 = int((lon0 - xll) // cs), int((lon1 - xll) // cs) + 1
        rows = []
        for i, line in enumerate(fh):
            if i < r0:
                continue
            if i >= r1:
                break
            rows.append(np.array(line.split()[c0:c1], dtype='f8'))
    a = np.vstack(rows)
    a[a == hdr.get('nodata_value', -9999)] = 0
    a[a < 0] = 0
    return a, ytop - r0 * cs, xll + c0 * cs, cs      # surfaces (ha), lat du bord haut, lon du bord gauche, pas


def read_mirca():
    out = {}
    for crop, c in MIRCA_CROPS.items():
        f = os.path.join(MIRCA_DIR, f'ANNUAL_AREA_HARVESTED_RFC_CROP{c}_HA.ASC')
        if not os.path.exists(f):
            sys.exit(f'MIRCA absent : {f}')
        out[crop] = read_mirca_window(f)
    log('MIRCA lu :', list(out))
    return out


def select_series(deps, mirca):
    """Surface MIRCA par département x culture et liste des séries traitées."""
    sub = (np.arange(7) + 0.5) / 7
    dy, dx = np.meshgrid(sub, sub, indexing='ij')
    rows = []
    for crop, (a, top, left, cs) in mirca.items():
        I, J = np.nonzero(a > 0)
        plat = (top - (I[:, None] + dy.ravel()[None, :]) * cs).ravel()
        plon = (left + (J[:, None] + dx.ravel()[None, :]) * cs).ravel()
        pts = np.column_stack([plon, plat])
        share = a[I, J].repeat(dy.size) / dy.size
        for name, ext, holes, cen in deps:
            rows.append(dict(departement=name, culture=crop,
                             surface_mirca_ha=float(share[inside_dep(pts, ext, holes)].sum())))
    t = pd.DataFrame(rows)
    t['mirca_ok'] = t.surface_mirca_ha >= args.min_ha
    if args.toutes_series:
        t['serie_rendement_retenue'] = np.nan
        t['traitee'] = t.mirca_ok
    else:
        if not os.path.exists(RENDEMENTS):
            sys.exit(f'Rendements détrendés absents : {RENDEMENTS} (lancer detrend.py, ou --toutes-series)')
        y = pd.read_csv(RENDEMENTS)
        keep_y = set(zip(y.dept, y.crop))
        t['serie_rendement_retenue'] = [k in keep_y for k in zip(t.departement, t.culture)]
        t['traitee'] = t.mirca_ok & t.serie_rendement_retenue
        miss = t[t.serie_rendement_retenue & ~t.mirca_ok]
        for _, r in miss.iterrows():
            log(f'   ATTENTION série de rendement retenue sous le seuil MIRCA : {r.departement}_{r.culture} '
                f'({r.surface_mirca_ha:.0f} ha)')
    t.to_csv(os.path.join(SORTIE, 'surfaces_mirca_departements.csv'), index=False, float_format='%.0f')
    gone = sorted(set(t.departement) - set(t[t.traitee].departement))
    log(f'Seuil MIRCA {args.min_ha:.0f} ha : {int(t.mirca_ok.sum())} couples sur {len(t)} ; '
        f'séries traitées : {int(t.traitee.sum())} ; départements sans aucune série : {", ".join(gone) if gone else "aucun"}')
    return set(zip(t[t.traitee].departement, t[t.traitee].culture))


# ------------------------------------------------------------------ poids sur la grille climatique
def coord_names(ds, var):
    dims = ds[var].dims
    lat = next(d for d in dims if d.lower() in ('lat', 'latitude', 'y', 'rlat'))
    lon = next(d for d in dims if d.lower() in ('lon', 'longitude', 'x', 'rlon'))
    return lat, lon


def build_weights(lat, lon):
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    dens = {}
    for crop, (a, top, left, cs) in MIRCA.items():
        ri = np.clip(((top - LAT.ravel()) // cs).astype(int), 0, a.shape[0] - 1)
        ci = np.clip(((LON.ravel() - left) // cs).astype(int), 0, a.shape[1] - 1)
        dens[crop] = a[ri, ci]                     # surface de la cellule MIRCA qui contient le pixel
    W, names, notes = [], [], []
    for name, ext, holes, cen in DEPS:
        if not any((name, c) in KEEP for c in dens):
            continue
        inside = inside_dep(pts, ext, holes)
        if inside.sum() == 0:
            notes.append(f'{name} : aucun pixel dans le département sur cette grille, série non produite')
            continue
        for crop, dv in dens.items():
            if (name, crop) not in KEEP:
                continue
            w = inside * dv
            if w.sum() == 0:
                notes.append(f'{name}_{crop} : aucun pixel avec la culture sur cette grille, série non produite')
                continue
            W.append(w / w.sum())
            names.append(f'{name}_{crop}')
    return np.array(W), names, notes


_WCACHE = {}
def get_weights(ds, var):
    la, lo = coord_names(ds, var)
    lat, lon = ds[la].values, ds[lo].values
    key = hashlib.md5(np.concatenate([lat, lon]).astype('f4').tobytes()).hexdigest()[:8]
    if key not in _WCACHE:
        W, names, notes = build_weights(lat, lon)
        _WCACHE[key] = (W, names)
        tag = f'{len(lat)}x{len(lon)}_{key}'
        for n in notes:
            log('   ' + n)
        wf = os.path.join(SORTIE, f'poids_{tag}.nc')
        if not os.path.exists(wf):
            tmp = wf + f'.{os.getpid()}.tmp'
            xr.DataArray(W.reshape(len(names), len(lat), len(lon)).astype('f4'), dims=('serie', 'lat', 'lon'),
                         coords={'serie': names, 'lat': lat, 'lon': lon}, name='poids').to_netcdf(tmp)
            os.replace(tmp, wf)
    return _WCACHE[key] + (la, lo)


# ------------------------------------------------------------------ réduction d'un fichier
def pick_var(ds):
    c = [v for v in ds.data_vars if 'time' in ds[v].dims and ds[v].ndim == 3]
    if len(c) != 1:
        raise ValueError(f'variable ambiguë : {list(ds.data_vars)}')
    return c[0]


def reduce_file(path, name):
    out_csv = os.path.join(SORTIE, name + '_bycrop.csv.gz')
    if os.path.exists(out_csv):
        return None
    if not os.path.exists(path):
        log('ABSENT :', path)
        return dict(sortie=name, fichier=path, statut='absent')
    t0 = time.time()
    ds = xr.open_dataset(path, decode_times=False)
    var = pick_var(ds)
    W, names, la, lo = get_weights(ds, var)
    da = ds[var].transpose('time', la, lo)
    tv = ds['time']
    cal = tv.attrs.get('calendar', 'standard')
    dates = cftime.num2date(tv.values, tv.attrs['units'], calendar=cal)
    ds_str = np.array([d.strftime('%Y-%m-%d') for d in dates])
    months = np.array([d.month for d in dates])
    nt = da.shape[0]
    res = np.full((nt, len(names)), np.nan, 'f4')
    Wt = W.T
    for i0 in range(0, nt, args.chunk):
        x = da.isel(time=slice(i0, i0 + args.chunk)).values.astype('f8')
        x = x.reshape(x.shape[0], -1)
        ok = np.isfinite(x)
        num = np.where(ok, x, 0.0) @ Wt
        den = ok.astype('f8') @ Wt
        with np.errstate(invalid='ignore', divide='ignore'):
            r = num / den
        r[den < 0.5] = np.nan                        # moins de la moitié du poids disponible : manquant
        res[i0:i0 + x.shape[0]] = r
    df = pd.DataFrame(res, columns=names)
    df.insert(0, 'date', ds_str)
    if not name.startswith('obs_'):
        df = df[np.isin(months, list(MOIS_MODELES))]
    df.to_csv(out_csv + '.tmp', index=False, float_format='%.5g', compression='gzip')
    os.replace(out_csv + '.tmp', out_csv)
    dup = int(pd.Series(ds_str).duplicated().sum())
    try:
        try:
            full = xr.date_range(dates[0], dates[-1], freq='D', calendar=cal, use_cftime=True)
        except Exception:
            full = xr.cftime_range(dates[0], dates[-1], freq='D', calendar=cal)
        miss = sorted(set(d.strftime('%Y-%m-%d') for d in full) - set(ds_str))
    except Exception:
        miss = ['controle_impossible']
    rep = dict(sortie=name, fichier=os.path.basename(path), statut='ok', variable=var,
               unites=ds[var].attrs.get('units', ''), calendrier=cal, debut=ds_str[0], fin=ds_str[-1],
               n_jours=nt, n_doublons=dup, n_manquants=len(miss), jours_manquants=';'.join(miss[:50]),
               n_series=len(names), minutes=round((time.time() - t0) / 60, 1))
    ds.close()
    log(f'{name}: {nt} jours {ds_str[0]}..{ds_str[-1]} ({cal}), {len(names)} séries, manquants={len(miss)}, '
        f'doublons={dup}, {rep["minutes"]} min')
    return rep


def run(job):
    path, name = job
    try:
        return reduce_file(path, name)
    except Exception as e:
        log('ERREUR', name, repr(e))
        return dict(sortie=name, fichier=path, statut='erreur: ' + repr(e)[:200])


# ------------------------------------------------------------------ programme principal
os.makedirs(SORTIE, exist_ok=True)
DEPS = load_departments()
MIRCA = read_mirca()
KEEP = select_series(DEPS, MIRCA)

if __name__ == '__main__':
    todo = [j for j in JOBS if (not args.only or args.only in j[1])
            and not os.path.exists(os.path.join(SORTIE, j[1] + '_bycrop.csv.gz'))]
    log(f'{len(todo)} fichiers à traiter sur {len(JOBS)}, {args.jobs} en parallèle, sortie : {SORTIE}')
    rep_f = os.path.join(SORTIE, 'rapport_masque.csv')
    rep = pd.read_csv(rep_f) if os.path.exists(rep_f) else pd.DataFrame(columns=['sortie'])

    def record(r):
        global rep
        if r:
            rep = pd.concat([rep[rep['sortie'] != r['sortie']], pd.DataFrame([r])], ignore_index=True)
            rep.to_csv(rep_f, index=False)

    if args.jobs <= 1:
        for j in todo:
            record(run(j))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for f in as_completed([ex.submit(run, j) for j in todo]):
                record(f.result())
    bad = rep[rep.statut != 'ok'] if 'statut' in rep else rep.iloc[:0]
    log('TERMINÉ' + (f' — {len(bad)} fichier(s) en erreur ou absent(s), voir rapport_masque.csv' if len(bad) else ''))
