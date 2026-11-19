# MEM4: Machine Learning Teammate Coordination Report

**Project:** Potential Fishing Zone (PFZ) Prediction – North Indian Ocean  
**Role:** MEM4 – Dataset Integration and Feature Preparation  
**Audience:** ML Modeling Member (MEM5 / Modeling Subsystem)  
**Date:** 2026-11-14  

---

## 1. Purpose of this Document

As the MEM4 member responsible for dataset integration and feature engineering, my mandate is to provide you with clean, multi-modal, feature-prepared environmental predictors without preempting your modeling decisions.

Per project instructions, **Salinity** and **Fish-Catch observations** were excluded from this phase. This report details the available feature set, outlines target variable formulation options, highlights critical questions for your modeling architecture, and recommends spatial/temporal train-test splitting strategies.

---

## 2. Integrated Feature Space Overview

The delivered dataset (`output/MEM4_final_integrated_dataset.csv` and `output/MEM4_final_integrated_dataset.nc`) contains **8,516,178 clean ocean grid records** spanning **729 consecutive days (2020–2021)** across the entire North Indian Ocean ($0.125^\circ - 24.875^\circ\text{N}$, $60.125^\circ - 99.875^\circ\text{E}$) at $0.25^\circ \times 0.25^\circ$ resolution.

### Predictor Feature Groups:
1. **Thermal & Hydrodynamic Core:**
   - `sst` (°C): Sea Surface Temperature (front detection, thermal preference)
   - `uo`, `vo` (m/s): Eastward & Northward horizontal current velocities
   - `current_speed` (m/s): Engineered magnitude $\sqrt{u_o^2 + v_o^2}$
2. **Biological Productivity:**
   - `chlor_a` (mg/m³): Chlorophyll-$a$ phytoplankton concentration (primary productivity)
3. **Atmospheric Forcing:**
   - `u10`, `v10`, `wind_speed` (m/s): 10-meter wind vectors and magnitude (Ekman transport, coastal upwelling index)
4. **Geomorphological & Proximity Predictors:**
   - `elevation` (m): Bathymetry / seafloor depth (shelf breaks at ~200m depth, banks, trenches)
   - `dist_to_coast_km` (km): Distance to closest shoreline
5. **Altimetric / Mesoscale Eddy Features (Arabian Sea sub-domain):**
   - `adt`, `sla` (m): Absolute dynamic topography and sea level anomaly (mesoscale eddies, upwelling domes)
   - `ugos`, `vgos` (m/s): Geostrophic current velocity components
   - `ssh_mask` (binary): Indicator (1 = inside Arabian Sea altimetry box, 0 = outside)
6. **Temporal & Cyclical Encodings:**
   - Linear: `year`, `month`, `day`, `day_of_year`
   - Cyclical: `month_sin`, `month_cos` (ensuring month 12 connects seamlessly to month 1)
   - Categorical: `season` (Winter_NE_Monsoon, Pre_Monsoon, SW_Monsoon, Post_Monsoon)

---

## 3. Status of Fish-Catch Data & Suggested Target Formulations

Because field fish-catch records were omitted/not provided for this run, the dataset currently acts as an **environmental feature space**. To turn this into a supervised training set, here are the standard options for your consideration:

### Option A: Binary PFZ Classification (`PFZ_label` $\in \{0, 1\}$)
- **Concept:** Predict whether a given spatio-temporal cell is a Potential Fishing Zone.
- **Formulation via Catch Data (if acquired later):**
  - $\text{Catch} \ge \text{Threshold}$ (e.g. 75th percentile of commercial landings) $\rightarrow 1$ (PFZ), else $0$ (Non-PFZ).
- **Formulation via Proxy / Oceanographic Indices (if catch data remains unavailable):**
  - Thermal front gradient ($|\nabla \text{SST}| \ge 0.5^\circ\text{C} / \text{grid}$) + Elevated Chlorophyll-a ($\text{chlor\_a} \ge 0.5\,\text{mg/m}^3$) + Bathymetric shelf break ($-1000\text{m} \le \text{elevation} \le -50\text{m}$).
- **Evaluation Metrics:** Precision, Recall, PR-AUC, F1-Score (recommended due to class imbalance).

### Option B: Continuous Abundance / CPUE Regression (`target` $\in \mathbb{R}^+$)
- **Concept:** Predict Catch Per Unit Effort ($\text{CPUE} = \text{catch weight} / \text{fishing effort hours}$) or fish biomass density.
- **Evaluation Metrics:** RMSE, MAE, Spearman Rank Correlation.

---

## 4. Recommendations for Train / Validation / Test Splitting

Because oceanographic and fishery data exhibit **strong spatial and temporal autocorrelation**, random row-wise splitting (`train_test_split(shuffle=True)`) will cause severe data leakage and artificially inflated metrics.

### Recommended Approaches for ML Member:

1. **Temporal Horizon Split (Recommended):**
   - **Training Set:** Year 2020 (January 1, 2020 – December 31, 2020: 365 daily timesteps).
   - **Validation / Test Set:** Year 2021 (January 1, 2021 – December 30, 2021: 364 daily timesteps).
   - *Advantage:* Simulates true operational forecasting where past observations predict future fishing seasons.

2. **Spatial Block Cross-Validation (Alternative):**
   - Partition the North Indian Ocean into spatial blocks (e.g., Arabian Sea vs. Bay of Bengal, or $2^\circ \times 2^\circ$ blocks).
   - Train on one basin/block set, test on geographically held-out regions to test spatial generalization.

3. **Seasonal Block Cross-Validation:**
   - Train across 3 seasons, evaluate out-of-sample performance on the intense Southwest Monsoon.

---

## 5. Key Questions Requiring ML Member Confirmation

Please review and confirm:
1. **Target Definition:** Will your model use an external fishery landing dataset, commercial logbook GPS points, or an oceanographic proxy (thermal/chlorophyll frontal intersection)?
2. **Missing Chlorophyll Handling:** Chlorophyll-a has 91.1% missingness during cloud-heavy monsoon periods. Does your architecture support tree-based algorithms (LightGBM / XGBoost / CatBoost) that handle NaNs natively, or do you require spatio-temporal interpolation / monthly climatological imputation?
3. **Spatial Scope:** Do you intend to model the full basin ($100 \times 160$) or focus exclusively on the Eastern Arabian Sea coincident sub-domain ($54 \times 30$) where SSH altimetry is 100% complete?
4. **Train-Test Protocol:** Do you approve the 2020 (train) vs. 2021 (test) temporal holdout scheme?
