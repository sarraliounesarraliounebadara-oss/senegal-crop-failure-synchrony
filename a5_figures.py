"""Figures 1, 7, 8, 9 and 10 of the Agricultural Systems manuscript (68 series, MIRCA masks, SPEI-3)."""
import os, sys, numpy as np, pandas as pd, xarray as xr, geopandas as gpd, matplotlib, regionmask
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
FIG = sys.argv[2] if len(sys.argv) > 2 else 'fig'
CLIM = os.environ.get('ARC2_CLIM', os.path.join(OUT, 'arc2_jjas_clim_1986_2016.nc'))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
g = gpd.read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'departements_1980.geojson'))
y = pd.read_csv(f'{OUT}/yields_detrended.csv')
y = y[y.fail_p20.notna()]


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
ORD = ['groundnut', 'millet', 'sorghum', 'maize']
AB = {'groundnut': 'G', 'millet': 'Mi', 'sorghum': 'S', 'maize': 'Ma'}
ns = y.groupby('dept').crop.apply(lambda s: '·'.join(AB[c] for c in ORD if c in set(s)))
g['n'] = g.departement.map(y.groupby('dept').crop.nunique()).fillna(0).astype(int)
clim = xr.open_dataset(CLIM).precip
sen = regionmask.from_geopandas(g.dissolve(), overlap=False).mask(clim.Lon, clim.Lat)
c = clim.where(sen.notnull())
ramp = ['#e8f0fa', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#1c5cab', '#104281', '#0d366b']
lev = [100, 200, 300, 400, 500, 600, 700, 900, 1100, 1400]
cmap, norm = ListedColormap(ramp), BoundaryNorm(lev, len(ramp))
fig, ax = plt.subplots(figsize=(6.7, 5.3))
pm = ax.contourf(c.Lon, c.Lat, c.values, levels=lev, cmap=cmap, norm=norm)
ax.contour(c.Lon, c.Lat, c.values, levels=[300, 500, 700, 900], colors='white', linewidths=.6)
g[g.n == 0].plot(ax=ax, facecolor='#d9d8d4', edgecolor='white', lw=.6, hatch='////')
g.boundary.plot(ax=ax, color='white', lw=.6)
g[g.n > 0].boundary.plot(ax=ax, color='#0b0b0b', lw=.8)
shift = {'Bambey': (-0.10, 0.10), 'Diourbel': (0.02, -0.04), 'Thies': (-0.08, -0.02), 'Mbacke': (0.08, 0.04),
         'Mbour': (-0.14, 0.02), 'Fatick': (0.10, 0.02), 'Gossas': (0.02, 0.04), 'Kaolack': (0.04, -0.03)}
for _, r in g[g.n > 0].iterrows():
    p = r.geometry.representative_point()
    dx, dy = shift.get(r.departement, (0, 0))
    ax.text(p.x + dx, p.y + dy, f'{r.departement}\n{ns[r.departement]}', ha='center', va='center', fontsize=5.0,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=.75))
ax.set_xlim(-17.65, -11.2); ax.set_ylim(12.25, 16.75); ax.set_aspect('equal')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
cax = fig.add_axes([0.10, 0.075, 0.42, 0.022])
cb = fig.colorbar(pm, cax=cax, orientation='horizontal', ticks=lev)
cb.set_label('Mean June–September rainfall, ARC2 1986–2016 (mm)', fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
fig.legend(handles=[Patch(facecolor='white', edgecolor='#0b0b0b', label='Retained department (crops: G, Mi, S, Ma)'),
                    Patch(facecolor='#d9d8d4', edgecolor='white', hatch='////', label='No retained series')],
           loc='lower right', bbox_to_anchor=(0.99, 0.04), fontsize=6.5, frameon=False)
fig.subplots_adjust(bottom=0.2)
save(fig, 'Fig1_study_area')

# ---------------------------------------------------------------- Fig. 7
yr = pd.read_csv(f'{OUT}/s31_obs_yearly_drought_failure.csv', index_col=0)
RR = pd.read_csv(f'{OUT}/s31_risk_by_crop.csv').set_index('crop')
fig, axs = plt.subplots(1, 3, figsize=(11, 3.15))
ax = axs[0]
ax.plot(yr.index, 100 * yr.fail, color='#b2412f', lw=1.8, label='Crop failures')
ax.plot(yr.index, 100 * yr.drought, color='#2c77b0', lw=1.8, label='Departments in drought')
ax.set_xlabel('Year'); ax.set_ylabel('Share (%)'); ax.legend(fontsize=7, loc='upper left')
ax.set_title('(a) Historical drought and failures, 1986–2016')
ax = axs[1]
r = np.corrcoef(yr.drought, yr.fail)[0, 1]
b1, b0 = np.polyfit(100 * yr.drought, 100 * yr.fail, 1)
ax.scatter(100 * yr.drought, 100 * yr.fail, color='#3b4a63', s=22, alpha=.85)
xx = np.array([0, 100]); ax.plot(xx, b0 + b1 * xx, color='#b2412f', lw=1.4)
for t in [2002, 2007, 2011, 2014]:
    ax.annotate(str(t), (100 * yr.drought[t], 100 * yr.fail[t]), xytext=(3, 3), textcoords='offset points', fontsize=6.5)
ax.set_xlabel('Departments in drought (%)'); ax.set_ylabel('Crop series failing (%)')
ax.set_title(f'(b) Yearly association (r = {r:.2f})')
ax = axs[2]
x = np.arange(4)
ax.bar(x - .18, 100 * RR.loc[ORD, 'fail_given_drought'], .36, color='#2c77b0', label='Drought')
ax.bar(x + .18, 100 * RR.loc[ORD, 'fail_given_not_drought'], .36, color='#d4a056', label='No drought')
ax.set_xticks(x, [c.capitalize() for c in ORD], rotation=30); ax.set_ylabel('Failure probability (%)')
ax.legend(fontsize=7); ax.set_title('(c) Crop-specific conditional risk')
fig.tight_layout(); save(fig, 'Fig7_drought_failure')

# ---------------------------------------------------------------- Fig. 8
O = pd.read_csv(f'{OUT}/s32_synchrony_decomposition.csv').set_index('crop').loc[ORD]
d = pd.read_csv(f'{OUT}/s32_decomposition_panel.csv')
tot = {}
for k in ['rel_anom', 'clim', 'resid']:
    tot[k] = sum(d[d.crop == cr].pivot(index='year', columns='dept', values='shock_' + k).drop(index=2004, errors='ignore')
                 .dropna(axis=1).sum(axis=1) for cr in ORD) / 1000
T = pd.DataFrame(tot)
T.to_csv(f'{OUT}/s33_national_shocks_complete.csv')
CB, CO, CG = '#1f77b4', '#ff7f0e', '#2ca02c'
fig, axs = plt.subplots(1, 2, figsize=(10, 3.45))
ax = axs[0]
ax.bar(x - .26, O.phi_total, .26, color=CB, label='Observed')
ax.bar(x, O.phi_climate, .26, color=CO, label='Rainfall model')
ax.bar(x + .26, O.phi_residual, .26, color=CG, label='Residual')
ax.set_xticks(x, [c.capitalize() for c in ORD]); ax.set_ylabel('Synchrony φ'); ax.legend(fontsize=7)
ax.set_title('(a) National production synchrony')
ax = axs[1]
ax.plot(T.index, T.rel_anom, color='black', lw=1.8, label='Observed')
ax.plot(T.index, T.clim, color=CO, lw=1.3, label='Rainfall-explained')
ax.plot(T.index, T.resid, color=CG, lw=1.3, label='Residual')
ax.axhline(0, color='#999999', lw=.6)
ax.set_xlabel('Year'); ax.set_ylabel('Yield-driven shock (kt)'); ax.legend(fontsize=7, loc='lower left')
ax.set_title('(b) All-crop national shocks')
fig.tight_layout(); save(fig, 'Fig8_rainfall_decomposition')

# ---------------------------------------------------------------- Fig. 9
R = pd.read_csv(f'{OUT}/s31_cmip6_by_model.csv')
R = R[R.version == 'raw']
obs = pd.read_csv(f'{OUT}/s31_obs_summary.csv', index_col=0).iloc[:, 0]
order = [('ssp245', 'hist', 'Historical\n1985–2014'), ('ssp245', 'near', 'SSP245\n2036–2065'), ('ssp245', 'far', 'SSP245\n2071–2100'),
         ('ssp585', 'near', 'SSP585\n2036–2065'), ('ssp585', 'far', 'SSP585\n2071–2100')]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.25))
for ax, (k, lab, ttl, ref, yl) in zip(axs, [('exposure', '%', '(a) Departments in drought', 20, (0, 105)),
                                            ('ws40', 'years/decade', '(b) Widespread drought frequency', obs.obs_ws40, (0, 11)),
                                            ('phi', 'φ', '(c) Spatial synchrony of SPEI-3', obs.obs_phi, (0.3, 1.0))]):
    for i, (s, w, _) in enumerate(order):
        v = R[(R.ssp == s) & (R.window == w)][k].values
        ax.scatter(i + np.linspace(-.15, .15, len(v)), v, facecolor='white', edgecolor='#1d5b79', s=26, zorder=3)
        ax.hlines(np.median(v), i - .25, i + .25, color='#c0622d', lw=3, zorder=4)
    ax.axhline(ref, color='black', ls=':', lw=1)
    ax.set_xticks(range(5), [o[2] for o in order], fontsize=7); ax.set_ylabel(lab); ax.set_ylim(*yl); ax.set_title(ttl)
    ax.yaxis.grid(True, color='#e4e3df', lw=.6); ax.set_axisbelow(True)
axs[2].text(4.45, obs.obs_phi + .012, 'observed', ha='right', fontsize=6.5)
fig.tight_layout(); save(fig, 'Fig9_projected_drought')

# ---------------------------------------------------------------- Fig. 10
Mm = pd.read_csv(f'{OUT}/s31_dept_freq_far.csv')
fig, axs = plt.subplots(1, 2, figsize=(10, 5.15))
vmin, vmax = 20, 95
for ax, (ssp, ttl) in zip(axs, [('ssp245', '(a) SSP2-4.5, 2071–2100'), ('ssp585', '(b) SSP5-8.5, 2071–2100')]):
    s = Mm[Mm.ssp == ssp].set_index('dept')
    m = g.merge(s, left_on='departement', right_index=True, how='left')
    m[m.freq_median.isna()].plot(ax=ax, color='#d9d8d4', edgecolor='#555555', lw=.5)
    mm = m[m.freq_median.notna()]
    mm.plot(ax=ax, column='freq_median', cmap='viridis', vmin=vmin, vmax=vmax, edgecolor='#555555', lw=.5)
    weak = mm[mm.n_increase < 7]
    for _, r in weak.iterrows():
        p = r.geometry.representative_point()
        ax.scatter(p.x, p.y, s=28, facecolor='white', edgecolor='black', zorder=5)
    ax.set_title(ttl); ax.set_axis_off()
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax))
cb = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=.045, pad=.04, aspect=40)
cb.set_label('Years below the historical drought threshold (%)')
save(fig, 'Fig10_drought_frequency_maps')
print('figures written to', FIG)
