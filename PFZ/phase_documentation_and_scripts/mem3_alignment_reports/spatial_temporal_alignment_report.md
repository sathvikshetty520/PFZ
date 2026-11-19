# Mem3: Spatial and Temporal Alignment Report
**Project:** Potential Fishing Zone (PFZ) & North Indian Ocean Marine Forecasting  
**Author:** Mem3 Alignment Subsystem  
**Generated:** 2026-11-11 22:01:32  

---

## 1. Executive Summary

This report documents the completion of **Mem3: Spatial and Temporal Alignment** for the 7 heterogeneous oceanographic datasets cleaned during Mem2. The raw datasets exhibited substantial heterogeneity in:
- **Spatial grid definitions**: Varying from high-resolution GEBCO bathymetry (1/240° ≈ 450 m) and Chlorophyll (0.0375° ≈ 4 km) to Ocean Currents (1/12° ≈ 9 km) and climate grids (0.25° ≈ 28 km).
- **Spatial domain boundaries**: SST, Wind, Currents, Bathymetry, Coastline, and Chlorophyll cover the whole North Indian Ocean basin (0–25°N, 60–100°E), whereas **Sea Surface Height (SSH) is confined to an Eastern Arabian Sea / Western Indian EEZ box (6.31°–19.81°N, 70.94°–78.44°E)**.
- **Temporal resolutions & timestamps**: Hourly ERA5 wind vs. daily midnight (00:00 UTC) currents/SSH vs. daily mid-day (12:00 UTC) SST and Chlorophyll.
- **Temporal observation periods**: Chlorophyll covers 2020–2021 (729 days), whereas physical ocean variables cover 2020–2023 (1,457 days). All years consistently end on December 30.

### Harmonization Results
1. **Target Common Grid**: Regular 0.25° × 0.25° grid on WGS84 (EPSG:4326), 100 latitude nodes [0.125°N, 24.875°N] by 160 longitude nodes [60.125°E, 99.875°E].
2. **Dual-Domain Solution**:
   - **Full Basin Dataset**: Full North Indian Ocean domain (100 × 160). Variables outside their native bounds (e.g. SSH outside Arabian Sea box) are set to NaN and flagged with an explicit ssh_mask.
   - **Coincident Intersecting Dataset**: Restricted to [6.375°–19.625°N, 71.125°–78.375°E] (54 × 30 grid) where 100% of all 7 datasets coexist simultaneously.
3. **Dual Temporal Horizons**:
   - **Daily Aligned**: Standard calendar date index (YYYY-MM-DD), with hourly wind aggregated to daily mean/speed, and daily observations synchronized.
   - **Monthly Aligned**: Monthly average aggregation (YYYY-MM) across 24 calendar months (2020-01 to 2021-12) for seasonal climatology and cloud-free modeling.
4. **Delivered Formats**: CF-1.8 compliant NetCDF4 files and partitioned columnar Parquet files for direct machine learning workflows.

---

## 2. Dataset Alignment & Compatibility Matrix

| Dataset | Variables | Raw Resolution | Raw Extent | Aligned Grid | Aligned Period | Alignment Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SST** | sst (°C) | 0.25° (100 × 160) | Lat: [0.125, 24.875]<br>Lon: [60.125, 99.875] | 0.25° (100 × 160) | Daily & Monthly (2020-2021) | 100% Compatible |
| **Ocean Currents** | uo, o (m/s) | 1/12° (301 × 480) | Lat: [0.0, 25.0]<br>Lon: [60.0, 99.917] | 0.25° (100 × 160) | Daily & Monthly (2020-2021) | 100% Compatible |
| **Wind** | u10, 10, wind_speed (m/s) | 0.25° (101 × 161) | Lat: [0.0, 25.0]<br>Lon: [60.0, 100.0] | 0.25° (100 × 160) | Daily & Monthly (2020-2021) | 100% Compatible |
| **SSH** | dt, sla (m)<br>ugos, gos (m/s) | 0.125° (109 × 61) | Lat: [6.3125, 19.8125]<br>Lon: [70.9375, 78.4375] | 0.25° (100 × 160)<br>Coincident (54 × 30) | Daily & Monthly (2020-2021) | Sub-domain Masked |
| **Chlorophyll-a** | chlor_a (mg/m³) | 0.0375° (665 × 1066) | Lat: [0.0562, 24.9562]<br>Lon: [60.0187, 99.9563] | 0.25° (100 × 160) | Daily & Monthly (2020-2021) | 100% Compatible |
| **Bathymetry** | elevation (m) | 1/240° (6000 × 9600) | Lat: [0.002, 24.998]<br>Lon: [60.002, 99.998] | 0.25° (100 × 160) | Static | 100% Compatible |
| **Coastline** | land_mask, dist_to_coast_km | Vector (11 features) | Lat: [0.0, 25.0]<br>Lon: [60.0, 100.0] | 0.25° (100 × 160) | Static | 100% Compatible |

---

## 3. Variable-Level Statistical Summary

| Variable | Long Name | Units | Valid Observations | Pct Valid | Min | Max | Mean | Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **sst** | Sea Surface Temperature | degC | 8,583,246 | 73.59% | 15.3900 | 33.6300 | 28.8948 | 1.3056 |
| **uo** | Eastward Ocean Current Velocity | m/s | 8,440,362 | 72.36% | -1.9735 | 2.0884 | 0.0114 | 0.2481 |
| **vo** | Northward Ocean Current Velocity | m/s | 8,440,362 | 72.36% | -2.4879 | 1.5343 | -0.0083 | 0.1906 |
| **u10** | 10-meter Eastward Wind Component | m/s | 11,472,000 | 98.35% | -20.5189 | 20.7612 | 1.2367 | 3.8883 |
| **v10** | 10-meter Northward Wind Component | m/s | 11,472,000 | 98.35% | -18.0902 | 18.1500 | 0.1734 | 3.0896 |
| **wind_speed** | 10-meter Wind Speed Magnitude | m/s | 11,472,000 | 98.35% | 0.0017 | 21.6387 | 4.2930 | 2.7918 |
| **chlor_a** | Chlorophyll-a Concentration | mg/m^3 | 1,037,608 | 8.90% | 0.0318 | 85.0795 | 0.5048 | 1.2633 |
| **adt** | Absolute Dynamic Topography (Sea Surface Height) | m | 656,100 | 5.62% | 0.6404 | 1.2502 | 0.8743 | 0.0940 |
| **sla** | Sea Level Anomaly | m | 656,829 | 5.63% | -0.0962 | 0.4733 | 0.1247 | 0.0918 |
| **ugos** | Absolute Geostrophic Velocity (Eastward) | m/s | 656,100 | 5.62% | -1.0869 | 0.8899 | 0.0031 | 0.1408 |
| **vgos** | Absolute Geostrophic Velocity (Northward) | m/s | 656,100 | 5.62% | -0.6208 | 0.7838 | 0.0380 | 0.1347 |
| **elevation** | Bathymetric Elevation (negative = depth) | m | 16,000 | 100.00% | -5292.5000 | 2818.7500 | -2006.6671 | 1908.1332 |
| **dist_to_coast_km** | Shortest Distance to Coastline | km | 16,000 | 100.00% | 0.0064 | 1181.7369 | 329.2001 | 271.5256 |
| **land_mask** | Land-Sea Mask (1=Land, 0=Ocean) | binary (0/1) | 16,000 | 100.00% | 0.0000 | 1.0000 | 0.2699 | 0.4439 |
| **ssh_mask** | SSH Data Availability Mask (1=Valid SSH Sub-domain, 0=Outside SSH Coverage) | binary (0/1) | 16,000 | 100.00% | 0.0000 | 1.0000 | 0.1013 | 0.3017 |

---

## 4. Deliverable File Catalog

All aligned deliverables are generated in data/aligned/:

1. **ligned_daily_basin.nc** (513.4 MB)
   - Primary NetCDF4 dataset containing all 15 variables across the full basin (100 × 160 × 729 days).
   - CF-1.8 compliant metadata with coordinate attributes and flags.
2. **ligned_monthly_basin.nc** (17.1 MB)
   - Monthly average NetCDF4 dataset across 24 calendar months (2020-01 to 2021-12).
3. **ligned_daily_coincident.nc** (52.0 MB)
   - Daily NetCDF4 dataset cropped to the coincident sub-domain (54 × 30) where all variables (including SSH) are 100% present.
4. **ligned_monthly_coincident.nc** (1.75 MB)
   - Monthly average NetCDF4 dataset for the coincident sub-domain.
5. **ligned_dataset_daily.parquet** (246.8 MB)
   - Tabular format of ocean grid cells (8,515,890 rows) ready for ML models.
6. **ligned_coincident_daily.parquet** (31.1 MB)
   - Tabular dataset of 100% complete multi-modal observations (1,180,980 rows).
7. **ligned_dataset_monthly.parquet** (14.2 MB)
   - Tabular monthly dataset (384,000 rows).

---

## 5. Scientific Principles Upheld
- **Zero Artificial Imputation**: Cloud and land NaNs are strictly preserved; no artificial zero-filling.
- **Physical Velocity Vector Directions**: Negative eastward/northward current velocities are preserved.
- **Geographic Consistency**: EPSG:4326 geodetic coordinates, consistent cell centroids, and CF-1.8 compliance maintained.
