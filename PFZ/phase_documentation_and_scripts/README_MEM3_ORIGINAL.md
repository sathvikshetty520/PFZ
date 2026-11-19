# README for Mem3 Aligned Oceanographic Dataset Package
# Potential Fishing Zone (PFZ) Project - North Indian Ocean

## Overview
This package contains the complete spatially and temporally aligned oceanographic datasets produced during Phase 3 (Mem3).
All 7 heterogeneous marine datasets (SST, Ocean Currents, Wind, Sea Surface Height, Chlorophyll-a, Bathymetry, Coastline) have been aligned to a common 0.25 deg x 0.25 deg latitude-longitude grid and matched by common date and month across 2020-2021.

## Package Contents

### 1. netcdf/ (CF-1.8 Compliant NetCDF4 Gridded Datasets)
- ligned_daily_basin.nc (513.4 MB): Full North Indian Ocean basin (0.125 deg - 24.875 deg N, 60.125 deg - 99.875 deg E, 100x160 grid) at daily resolution for 729 days (2020-01-01 to 2021-12-30). Contains all 15 variables including ssh_mask.
- ligned_monthly_basin.nc (17.1 MB): Full basin monthly mean fields across 24 calendar months (2020-01 to 2021-12).
- ligned_daily_coincident.nc (52.0 MB): Coincident sub-domain (6.375 deg - 19.625 deg N, 71.125 deg - 78.375 deg E, 54x30 grid) where 100% of all variables (including SSH) coexist.
- ligned_monthly_coincident.nc (1.75 MB): Coincident sub-domain monthly mean fields.

### 2. tabular_parquet/ (Columnar Data for Machine Learning)
- ligned_dataset_daily.parquet (246.8 MB): 8,515,890 daily ocean records with all physical, biological, and static variables.
- ligned_coincident_daily.parquet (31.1 MB): 1,180,980 daily records for the 100% complete multi-modal coincident sub-domain.
- ligned_dataset_monthly.parquet (14.2 MB): 384,000 monthly mean records.

### 3. reports/ (Audit & Documentation)
- spatial_temporal_alignment_report.md: Detailed engineering report.
- lignment_compatibility_matrix.csv: Compatibility comparison before & after alignment.
- ariable_statistical_summary.csv: Full statistical metrics and observation counts.
- lignment_summary.html: Interactive visual report with summary cards and tables.

### 4. scripts/ (Reproducibility & Verification)
- lign_spatial_temporal.py: The production alignment pipeline script.
- erify_and_generate_reports.py: Automated test suite and report generator.

## Variables Included
- sst: Sea Surface Temperature (deg C)
- uo, o: Eastward & Northward Ocean Current Velocities (m/s)
- u10, 10: 10m Eastward & Northward Wind Components (m/s)
- wind_speed: 10m Wind Speed Magnitude (m/s)
- chlor_a: Chlorophyll-a Concentration (mg/m^3)
- dt: Absolute Dynamic Topography / SSH (m)
- sla: Sea Level Anomaly (m)
- ugos, gos: Absolute Geostrophic Current Velocities (m/s)
- elevation: Bathymetric Elevation (negative = depth) (m)
- dist_to_coast_km: Shortest Distance to Coastline (km)
- land_mask: Binary Land-Sea Mask (1=Land, 0=Ocean)
- ssh_mask: Binary SSH Data Availability Mask (1=Valid, 0=Outside Coverage)

## Quick Start (Python)

### Reading NetCDF with Xarray:
`python
import xarray as xr
ds = xr.open_dataset('netcdf/aligned_daily_basin.nc')
print(ds)
`

### Reading Parquet with Pandas:
`python
import pandas as pd
df = pd.read_parquet('tabular_parquet/aligned_dataset_daily.parquet')
print(df.head())
`
