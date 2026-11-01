
# senegal-crop-failure-synchrony

Analysis scripts and derived data for the manuscript *Diversification and correlated harvest shocks in Senegal* (Sarr et al., submitted to *Agricultural Systems*).

The study uses department-level yields of groundnut, millet, sorghum and maize in Senegal (1986–2016), rainfall and temperature merged with station data, and nine bias-adjusted CMIP6 models.

## Repository content

| Files | Role |
| --- | --- |
| `detrend_rendements.py`, `masque_mirca.py`, `detrend_indicateurs.py` | Pre-processing, run where the raw data are stored: yield cleaning and Theil–Sen detrending, daily department × crop climate series weighted by MIRCA2000 rainfed areas, optional seasonal indicators |
| `a0_climatologie_arc2.py` to `a5_figures.py` | ARC2 climatology, rainfall indicators, department SPEI-3, drought–failure link, CMIP6 projections and Schaake shuffle, rainfall decomposition, figures |
| `spei_makkink.py`, `indicators.py` | Shared functions: SPEI-3 (Makkink potential evapotranspiration, log-logistic distribution fitted by L-moments); onset (Sivakumar 1988), seasonal indicators and synchrony index |
| `run_analyses.sh` | Runs `a0` to `a5` in order |
| `departements_1980.geojson` | Approximate 1980 department boundaries, rebuilt from present-day units, used for the maps |
| `derived_data/` | Derived tables behind the figures and tables of the manuscript |
<!-- Add here the synchrony, production and crop-portfolio scripts. -->

## Data not included

The raw inputs cannot be redistributed here.

| Data | Source |
| --- | --- |
| Department yields, sown area and production | Direction de l'Analyse, de la Prévision et des Statistiques Agricoles (DAPSA), on request |
| Daily ARC2 rainfall and CHIRTS temperature merged with station data | Agence Nationale de l'Aviation Civile et de la Météorologie (ANACIM), on request |
| MIRCA2000 rainfed crop areas | Public; Portmann et al. (2010), https://doi.org/10.1029/2008GB003435 |
| Downscaled, bias-adjusted CMIP6 projections | Mbengue et al. (2026), https://doi.org/10.1038/s41597-026-07059-9 |


## Requirements

Python 3.10 or later, with numpy, pandas, scipy, xarray, netCDF4, geopandas, regionmask and matplotlib. lmoments3 is needed only for the self-test of `spei_makkink.py`.

```bash
pip install numpy pandas scipy xarray netCDF4 geopandas regionmask matplotlib
```

## How to run

1. On the machine that holds the raw data, edit the paths at the top of `detrend_rendements.py` and `masque_mirca.py`, then run them in this order.
2. Set the three input paths and run the analyses:

```bash
MASQUE=/path/to/masque_mirca RENDEMENTS=/path/to/rendements_detrendes_1986-2016.csv ARC2=/path/to/ARC2_merged.nc bash run_analyses.sh
```

Results are written to `out/` and figures to `fig/`. Permutation tests and resampling use fixed random seeds, so results are reproducible.

## Derived data

| File | Content |
| --- | --- |
| `yearly_failure_shares.csv` | Yearly share of department–crop series in failure |
| `national_production_shocks_bycrop.csv` | National production shocks by crop |
| `national_crop_fractional_shocks_completepanel.csv` | National fractional production shocks, complete panel |
| `yearly_drought_and_failure.csv` | Yearly drought and failure shares |
| `Table7_crop_portfolio_covariance.csv` | Covariance of production shocks between crops |
| `TableS5_robustness_18_settings.csv` | Robustness checks across 18 settings |

## License

Code released under the MIT License (see `LICENSE`).

## Citation

Sarr AB et al. Diversification and correlated harvest shocks in Senegal. *Agricultural Systems* (submitted).

Contact: Alioune Badara Sarr, Université Assane Seck de Ziguinchor, sarr.aliounesarr.aliounebadara@gmail.com
