# Senegal crop-failure synchrony

This repository contains selected Python scripts supporting the manuscript **“Spatial and crop diversification under synchronized harvest shocks: A multi-scale assessment of Senegalese rainfed agriculture”** (Sarr et al., submitted to *Agricultural Systems*).

The study examines spatial and cross-crop synchrony in rainfed harvest shocks in Senegal, drawing on department-level yields for groundnut, millet, sorghum and maize (1986–2016), observational climate data, crop-specific rainfed harvested areas and nine bias-adjusted CMIP6 models.

The repository provides **analysis code, not a preprint or a complete data archive**. The underlying input datasets and derived result tables are not included.

## Repository contents

| File | Purpose |
| --- | --- |
| `detrend_rendements.py` | Cleans and selects department–crop yield series, applies Theil–Sen detrending, and identifies low-yield seasons. |
| `masque_mirca.py` | Extracts crop-specific daily climate series at department level using MIRCA2000 rainfed harvested-area weights. |
| `detrend_indicateurs.py` | Prepares and detrends climate indicators. |
| `a0_climatologie_arc2.py` | Computes the 1986–2016 June–September ARC2 rainfall climatology. |
| `a1_indicateurs_observes.py` | Calculates observed seasonal rainfall, heat and drought-related indicators for the retained department–crop series. |
| `a2_spei3_departements.py` | Computes department-level SPEI-3 from observational and CMIP6 water-balance series. |
| `a3_secheresse_projections_schaake.py` | Analyzes historical drought–crop-failure relationships and projected drought concurrence, including a Schaake-shuffle comparison. |
| `a4_decomposition_pluie.py` | Estimates the rainfall-associated component of yield anomalies using leave-one-year-out analysis. |
| `a5_figures.py` | Produces selected figures from intermediate analysis outputs. |
| `spei_makkink.py` | Provides functions used for potential evapotranspiration, SPEI-3 and detrending. |
| `indicators.py` | Provides functions for seasonal climate indicators and synchrony diagnostics. |
| `departements_1980.geojson` | Geographic boundaries used in the spatial analysis and mapping. |

## Input data

**Input datasets are not distributed in this repository.** Users must obtain them from the relevant sources and respect any applicable access and redistribution conditions.

| Data | Source |
| --- | --- |
| Department-level crop yields, harvested areas and production | Direction de l'Analyse, de la Prévision et des Statistiques Agricoles (DAPSA), Senegal; access through the data provider. |
| ARC2 rainfall and CHIRTS temperature data merged with station observations | Agence Nationale de l'Aviation Civile et de la Météorologie (ANACIM), Senegal; access through the data provider. |
| MIRCA2000 crop-specific rainfed harvested-area grids | Portmann et al. (2010), https://doi.org/10.1029/2008GB003435 |
| Downscaled, bias-adjusted CMIP6 climate projections | Mbengue et al. (2026), https://doi.org/10.1038/s41597-026-07059-9 |

## Software requirements

The scripts use Python 3.10 or later and scientific Python packages including NumPy, pandas, SciPy, xarray, netCDF4, GeoPandas, regionmask and Matplotlib. Some workflows also require additional dependencies according to the inputs and environment.

```bash
pip install numpy pandas scipy xarray netCDF4 geopandas regionmask matplotlib
```

## Running the available scripts

The preprocessing scripts contain paths to the original research environment. **Adapt these paths to your local data directories before running them.** The repository is not yet a self-contained, one-command replication workflow.

1. Configure the input and output paths near the beginning of `detrend_rendements.py`, then run:

   ```bash
   python detrend_rendements.py
   ```

2. Configure the paths in `masque_mirca.py`, including the yield file produced in the previous step and the climate and MIRCA2000 inputs, then run:

   ```bash
   python masque_mirca.py
   ```

3. As an example of an individual analysis step, the ARC2 rainfall climatology can be calculated with:

   ```bash
   mkdir -p out
   python a0_climatologie_arc2.py /path/to/ARC2_merged.nc out
   ```

The remaining analysis scripts require intermediate files with the paths and filenames expected by each script. Consult their header comments and source code before execution; additional preparation may be necessary to connect all steps.

## Data and reproducibility statement

The repository currently shares selected analysis scripts and a geographic boundary file. It does **not** contain the original input datasets, the intermediate files required by every script, or the derived CSV tables behind the manuscript's results. Accordingly, the public repository alone does not permit a full end-to-end reproduction of the study at present.

No version of the submitted manuscript is posted here as a preprint.

## License

The code is released under the MIT License; see `LICENSE`. Rights and reuse conditions for third-party input datasets are governed by their respective providers.

## Citation

Sarr, A. B., et al. *Spatial and crop diversification under synchronized harvest shocks: A multi-scale assessment of Senegalese rainfed agriculture.* Manuscript submitted to *Agricultural Systems*.

**Contact:** Alioune Badara Sarr — sarr.aliounesarr.aliounebadara@gmail.com
