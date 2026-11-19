# MEM4: Feature Documentation & Machine Learning Metadata

**Project:** Potential Fishing Zone (PFZ) Prediction – North Indian Ocean  
**Role:** MEM4 – Dataset Integration and Feature Preparation  
**Generated:** 2026-11-14 21:56:09  

---

## 1. Feature Dictionary

| Column Name | Type | Physical Range | Units | Meaning & Oceanographic Significance |
| :--- | :--- | :--- | :--- | :--- |
| `date` | string (YYYY-MM-DD) | 2020-01-01 to 2021-12-30 | - | Observation calendar date. |
| `latitude` | float32 | [0.125, 24.875] | deg N | Grid point latitude in WGS84 coordinates. |
| `longitude` | float32 | [60.125, 99.875] | deg E | Grid point longitude in WGS84 coordinates. |
| `sst` | float32 | [15.39, 33.63] | °C | Sea Surface Temperature. Crucial for identifying thermal fronts, upwelling, and species thermal preferences. |
| `uo` | float32 | [-1.97, 2.09] | m/s | Eastward velocity component of surface ocean currents. Positive = eastward, negative = westward. |
| `vo` | float32 | [-2.49, 1.53] | m/s | Northward velocity component of surface ocean currents. Positive = northward, negative = southward. |
| `current_speed` | float32 | [0.00, 2.54] | m/s | **Engineered Feature:** Overall surface flow speed $\sqrt{u_o^2 + v_o^2}$. High current gradients indicate frontal convergence where forage fish congregate. |
| `chlor_a` | float32 | [0.03, 85.08] | mg/m³ | Chlorophyll-a concentration. Proxy for phytoplankton biomass, indicating high biological productivity and primary feeding grounds. |
| `u10` | float32 | [-20.52, 20.76] | m/s | 10-meter eastward zonal wind speed. Drives Ekman transport and coastal upwelling. |
| `v10` | float32 | [-18.09, 18.15] | m/s | 10-meter northward meridional wind speed. Strong alongshore winds trigger coastal upwelling along the western Indian coast. |
| `wind_speed` | float32 | [0.00, 21.64] | m/s | Total wind magnitude $\sqrt{u_{10}^2 + v_{10}^2}$. Affects upper-layer mixing and sea surface turbulence. |
| `elevation` | float32 | [-5292.5, 2818.8] | m | Seafloor bathymetric depth (negative values represent ocean depth). Shelf breaks (depth ~200m) are prime fishing grounds. |
| `dist_to_coast_km` | float32 | [0.006, 1181.7] | km | Shortest distance to coastline. Separates shallow artisanal inshore fisheries from offshore industrial pelagic fisheries. |
| `land_mask` | int8 | {0, 1} | binary | 0 = Ocean cell, 1 = Land cell. |
| `adt` | float32 | [0.64, 1.25] | m | Absolute Dynamic Topography (Sea Surface Height). Distinguishes cyclonic eddies (low SSH, nutrient upwelling) from anticyclonic eddies (high SSH). |
| `sla` | float32 | [-0.10, 0.47] | m | Sea Level Anomaly relative to mean sea surface. |
| `ugos` | float32 | [-1.09, 0.89] | m/s | Eastward geostrophic velocity component from satellite altimetry. |
| `vgos` | float32 | [-0.62, 0.78] | m/s | Northward geostrophic velocity component from satellite altimetry. |
| `ssh_mask` | int8 | {0, 1} | binary | SSH domain indicator: 1 = within Eastern Arabian Sea altimetry box, 0 = outside coverage. |
| `year` | int16 | {2020, 2021} | year | **Engineered Feature:** Calendar year. |
| `month` | int8 | [1, 12] | 1-12 | **Engineered Feature:** Calendar month. |
| `day` | int8 | [1, 31] | 1-31 | **Engineered Feature:** Day of month. |
| `day_of_year` | int16 | [1, 366] | 1-366 | **Engineered Feature:** Day of year (Julian day index). |
| `month_sin` | float32 | [-1.0, 1.0] | dimensionless | **Engineered Feature:** Cyclical sine feature $\sin(2\pi (m-1)/12)$. Preserves continuity between December (12) and January (1). |
| `month_cos` | float32 | [-1.0, 1.0] | dimensionless | **Engineered Feature:** Cyclical cosine feature $\cos(2\pi (m-1)/12)$. Preserves cyclical seasonality for tree and neural network models. |
| `season` | string | 4 categories | text | **Engineered Feature:** North Indian Ocean monsoon season (Winter_NE_Monsoon, Pre_Monsoon, SW_Monsoon, Post_Monsoon). |

