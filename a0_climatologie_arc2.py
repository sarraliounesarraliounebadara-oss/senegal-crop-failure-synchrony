"""Mean June-September ARC2 rainfall 1986-2016 (background of Figure 1).
Usage: python a0_climatologie_arc2.py /mnt/disque/departments/climate_data/anacim_data/ARC2_merged.nc <out>"""
import sys, xarray as xr
ds = xr.open_dataset(sys.argv[1]).sel(time=slice('1986-01-01', '2016-12-31'))
v = [k for k in ds.data_vars if ds[k].ndim == 3][0]
jjas = ds[v].where(ds[v].time.dt.month.isin([6, 7, 8, 9]), drop=True)
clim = jjas.groupby('time.year').sum('time', min_count=100).mean('year').rename('precip')
clim.to_netcdf(f'{sys.argv[2]}/arc2_jjas_clim_1986_2016.nc')
print('ok', float(clim.min()), float(clim.max()))
