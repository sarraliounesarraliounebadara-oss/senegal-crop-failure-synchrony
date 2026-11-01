#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rendements DAPSA 1986-2016 (30 départements, limites de 1980, 4 cultures) :
nettoyage, sélection des séries, détrendage Theil-Sen et repérage des mauvaises récoltes.

Étapes
1. Rendement = rendement_kg_ha si statut 'renseigne' ; 0 si production nulle sur surface semée ;
   manquant sinon.
2. Exclusion de 2004 (criquet pèlerin, perte non climatique) :
   - dans tout département ayant déclaré une production nulle sur surface semée pour au moins une culture ;
   - pour le maïs dans tous les départements.
3. Sélection : au moins 25 saisons renseignées sur 31 et surface médiane >= 1000 ha.
4. Tendance Theil-Sen (Sen 1968) par série ; anomalie relative = rendement / tendance - 1.
5. Mauvaise récolte = saison dans le quintile inférieur des anomalies de la série (P20).
   Les séries plates (P20 > -10 %) sont exclues.

Sortie : rendements_detrendes_1986-2016.csv dans DOSSIER_SORTIE
Dépendances : numpy, pandas, scipy.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

# ------------------------------------------------------------------ chemins
DAPSA = '/home/sarr/Documents/dossier/agro/DAPSA_1980_2016_30_departements_4_cultures.csv'
DOSSIER_SORTIE = '/home/sarr/Documents/dossier/agro'
SORTIE = os.path.join(DOSSIER_SORTIE, 'rendements_detrendes_1986-2016.csv')

# ------------------------------------------------------------------ paramètres
DEBUT, FIN = 1986, 2016
MIN_SAISONS = 25          # saisons renseignées minimales (sur 31)
MIN_SURFACE = 1000        # surface médiane minimale (ha)
QUANTILE_ECHEC = 0.20     # quintile inférieur
MIN_BAISSE = -0.10        # série plate si son P20 est au-dessus de -10 %
CULTURES = {'ARACHIDE': 'groundnut', 'MAIS': 'maize', 'MIL': 'millet', 'SORGHO': 'sorghum'}

d = pd.read_csv(DAPSA, encoding='utf-8-sig')
d['crop'] = d.culture.map(CULTURES)
d = d[(d.annee >= DEBUT) & (d.annee <= FIN)]
d['yield'] = np.where(d.statut == 'renseigne', d.rendement_kg_ha,
                      np.where(d.statut == 'production_nulle_superficie_positive', 0.0, np.nan))
d['zero_prod_flag'] = d.statut == 'production_nulle_superficie_positive'
d = d.rename(columns={'departement': 'dept', 'annee': 'year', 'superficie_ha': 'area_ha'})

# ------------------------------------------------------------------ exclusion 2004
CRIQUET = sorted(d[(d.year == 2004) & d.zero_prod_flag].dept.unique())
d.loc[(d.year == 2004) & d.dept.isin(CRIQUET), 'yield'] = np.nan
d.loc[(d.year == 2004) & (d.crop == 'maize'), 'yield'] = np.nan
print('2004 retiré dans :', CRIQUET, '+ maïs dans tous les départements')

# ------------------------------------------------------------------ sélection et détrendage
rows = []
for (dep, crop), g in d.groupby(['dept', 'crop']):
    g = g.sort_values('year')
    v = g[g['yield'].notna()]
    n, med_area = len(v), g.loc[g['yield'].notna(), 'area_ha'].median()
    if not (n >= MIN_SAISONS and med_area >= MIN_SURFACE):
        continue
    pente, ordonnee, *_ = theilslopes(v['yield'], v['year'])
    trend = np.maximum(ordonnee + pente * g['year'], 1e-6)
    anom = g['yield'] / trend - 1
    p20 = anom.quantile(QUANTILE_ECHEC)
    if p20 > MIN_BAISSE:
        print(f'exclue (série plate, P20 = {p20:+.0%}) : {dep} {crop}')
        continue
    for (_, r), t, a in zip(g.iterrows(), trend, anom):
        rows.append(dict(dept=dep, crop=crop, year=r.year, area_ha=r.area_ha, yield_kg_ha=r['yield'],
                         trend_kg_ha=t, rel_anom=a, fail_p20=(a <= p20) if pd.notna(a) else np.nan,
                         fail_m20=(a <= -0.20) if pd.notna(a) else np.nan, zero_prod_flag=r.zero_prod_flag,
                         p20_threshold=p20, n_years=n, median_area_ha=med_area))
y = pd.DataFrame(rows)
os.makedirs(DOSSIER_SORTIE, exist_ok=True)
y.to_csv(SORTIE, index=False)

# ------------------------------------------------------------------ résumé
s = y.groupby(['crop', 'dept']).agg(saisons=('n_years', 'first'), surface_mediane=('median_area_ha', 'first'),
                                     seuil_P20=('p20_threshold', 'first'),
                                     echecs=('fail_p20', lambda x: int((x == True).sum())))
print(s.round(2).to_string())
print('\nséries retenues par culture :', y.groupby('crop').dept.nunique().to_dict(),
      '| total :', y.groupby(['crop', 'dept']).ngroups)
print('sortie :', SORTIE)
