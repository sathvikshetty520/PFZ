import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import shapefile
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely import distance

print('=== STARTING SPATIAL & TEMPORAL ALIGNMENT PIPELINE ===')
start_pipeline_time = time.time()

# Ensure directories
os.makedirs('data/aligned/reports', exist_ok=True)

# 1. DEFINE TARGET GRIDS & TIMESTAMPS
print('\n[1/8] Setting up Common Target Coordinate Grid...')
target_lat = np.linspace(0.125, 24.875, 100) # 0.25 deg resolution
target_lon = np.linspace(60.125, 99.875, 160) # 0.25 deg resolution

# Dates: 2020 has 365 days (ends Dec 30), 2021 has 364 days (ends Dec 30)
dates_2020 = pd.date_range('2020-01-01', '2020-12-30', freq='D')
dates_2021 = pd.date_range('2021-01-01', '2021-12-30', freq='D')
dates_2022 = pd.date_range('2022-01-01', '2022-12-30', freq='D')
dates_2023 = pd.date_range('2023-01-01', '2023-12-30', freq='D')

dates_2yr = dates_2020.append(dates_2021)
dates_4yr = dates_2yr.append(dates_2022).append(dates_2023)

n_time_2yr = len(dates_2yr)
n_time_4yr = len(dates_4yr)
n_lat = len(target_lat)
n_lon = len(target_lon)

print(f'Target Grid: {n_lat} latitudes x {n_lon} longitudes (0.25 deg)')
print(f'Primary Multi-modal Period (2020-2021): {n_time_2yr} daily steps')
print(f'Extended Physical Period (2020-2023): {n_time_4yr} daily steps')

# 2. PROCESS STATIC FIELDS: BATHYMETRY & COASTLINE
print('\n[2/8] Processing Static Fields (Bathymetry & Coastline)...')
t0 = time.time()
ds_bathy = xr.open_dataset('data/cleaned/Bathymetry_cleaned.nc')
bathy_interp = ds_bathy['elevation'].interp(lat=target_lat, lon=target_lon, method='linear').values.astype(np.float32)
ds_bathy.close()

# Coastline
sf = shapefile.Reader('data/cleaned/Coastline_cleaned/Coastline_cleaned.shp')
geoms = [shape(s) for s in sf.shapes()]
land_poly = unary_union(geoms)
prep_poly = prep(land_poly)
coastline = land_poly.boundary

T_LAT, T_LON = np.meshgrid(target_lat, target_lon, indexing='ij')
pts = [Point(lon, lat) for lat, lon in zip(T_LAT.ravel(), T_LON.ravel())]

inside_mask = np.array([prep_poly.contains(pt) for pt in pts], dtype=np.int8).reshape(n_lat, n_lon)
dist_km = (distance(coastline, pts) * 111.32).astype(np.float32).reshape(n_lat, n_lon)

# SSH Valid Domain Mask: Lat [6.3125, 19.8125], Lon [70.9375, 78.4375]
ssh_mask = np.zeros((n_lat, n_lon), dtype=np.int8)
for i, la in enumerate(target_lat):
    for j, lo in enumerate(target_lon):
        if 6.3125 <= la <= 19.8125 and 70.9375 <= lo <= 78.4375:
            ssh_mask[i, j] = 1

print(f'Static fields computed in {time.time() - t0:.2f}s.')
print(f'Bathymetry range: {np.nanmin(bathy_interp):.1f}m to {np.nanmax(bathy_interp):.1f}m')
print(f'Land cells: {np.sum(inside_mask)}, Ocean cells: {n_lat*n_lon - np.sum(inside_mask)}')
print(f'SSH valid cells: {np.sum(ssh_mask)}')

# 3. PROCESS SST (2020-2023)
print('\n[3/8] Processing Sea Surface Temperature (SST)...')
t0 = time.time()
sst_arrays_4yr = []
for yr in [2020, 2021, 2022, 2023]:
    ds_sst = xr.open_dataset(f'data/cleaned/SST_{yr}_cleaned.nc')
    # squeeze depth
    vals = ds_sst['sst'].values
    if vals.ndim == 4 and vals.shape[1] == 1:
        vals = vals[:, 0, :, :]
    sst_arrays_4yr.append(vals.astype(np.float32))
    ds_sst.close()

sst_4yr = np.concatenate(sst_arrays_4yr, axis=0)
sst_2yr = sst_4yr[:n_time_2yr]
print(f'SST processed in {time.time() - t0:.2f}s: shape {sst_2yr.shape}, valid range [{np.nanmin(sst_2yr):.2f}, {np.nanmax(sst_2yr):.2f}] degC')

# 4. PROCESS OCEAN CURRENTS (2020-2023)
print('\n[4/8] Processing Ocean Currents (uo, vo)...')
t0 = time.time()
uo_arrays_4yr = []
vo_arrays_4yr = []
for yr in [2020, 2021, 2022, 2023]:
    ds_curr = xr.open_dataset(f'data/cleaned/Ocean_Current_{yr}_cleaned.nc')
    curr_regrid = ds_curr.interp(latitude=target_lat, longitude=target_lon, method='linear')
    uo_val = curr_regrid['uo'].values
    vo_val = curr_regrid['vo'].values
    if uo_val.ndim == 4 and uo_val.shape[1] == 1:
        uo_val = uo_val[:, 0, :, :]
        vo_val = vo_val[:, 0, :, :]
    uo_arrays_4yr.append(uo_val.astype(np.float32))
    vo_arrays_4yr.append(vo_val.astype(np.float32))
    ds_curr.close()

uo_4yr = np.concatenate(uo_arrays_4yr, axis=0)
vo_4yr = np.concatenate(vo_arrays_4yr, axis=0)
uo_2yr = uo_4yr[:n_time_2yr]
vo_2yr = vo_4yr[:n_time_2yr]
print(f'Ocean Currents processed in {time.time() - t0:.2f}s: uo range [{np.nanmin(uo_2yr):.2f}, {np.nanmax(uo_2yr):.2f}] m/s, vo range [{np.nanmin(vo_2yr):.2f}, {np.nanmax(vo_2yr):.2f}] m/s')

# 5. PROCESS WIND (2020-2023)
print('\n[5/8] Processing Wind (u10, v10, daily mean & wind speed)...')
t0 = time.time()
ds_wind = xr.open_dataset('data/cleaned/Wind_cleaned.nc')
if ds_wind.latitude.values[0] > ds_wind.latitude.values[-1]:
    ds_wind = ds_wind.reindex(latitude=ds_wind.latitude.values[::-1])

# Daily resample mean
wind_daily = ds_wind.resample(time='1D').mean()
wind_regrid = wind_daily.interp(latitude=target_lat, longitude=target_lon, method='linear')

# Match dates_4yr exactly
wind_time_idx = pd.to_datetime(wind_regrid.time.values).strftime('%Y-%m-%d')
target_time_idx = dates_4yr.strftime('%Y-%m-%d')
time_map = {d: i for i, d in enumerate(wind_time_idx)}
indices = [time_map[d] for d in target_time_idx if d in time_map]

u10_4yr = wind_regrid['u10'].values[indices].astype(np.float32)
v10_4yr = wind_regrid['v10'].values[indices].astype(np.float32)
wspd_4yr = np.sqrt(u10_4yr**2 + v10_4yr**2).astype(np.float32)

u10_2yr = u10_4yr[:n_time_2yr]
v10_2yr = v10_4yr[:n_time_2yr]
wspd_2yr = wspd_4yr[:n_time_2yr]
ds_wind.close()
print(f'Wind processed in {time.time() - t0:.2f}s: wspd range [{np.nanmin(wspd_2yr):.2f}, {np.nanmax(wspd_2yr):.2f}] m/s')

# 6. PROCESS SSH (2020-2023)
print('\n[6/8] Processing Sea Surface Height (SSH: adt, sla, ugos, vgos)...')
t0 = time.time()
ssh_vars = ['adt', 'sla', 'ugos', 'vgos']
ssh_arrays_4yr = {v: [] for v in ssh_vars}

for yr in [2020, 2021, 2022, 2023]:
    ds_ssh = xr.open_dataset(f'data/cleaned/SSH_{yr}_cleaned.nc')
    # interp to target grid - values outside SSH bounds become NaN
    ssh_regrid = ds_ssh.interp(latitude=target_lat, longitude=target_lon, method='linear')
    for v in ssh_vars:
        ssh_arrays_4yr[v].append(ssh_regrid[v].values.astype(np.float32))
    ds_ssh.close()

ssh_4yr_dict = {v: np.concatenate(ssh_arrays_4yr[v], axis=0) for v in ssh_vars}
ssh_2yr_dict = {v: ssh_4yr_dict[v][:n_time_2yr] for v in ssh_vars}
adt_min = np.nanmin(ssh_2yr_dict['adt'])
adt_max = np.nanmax(ssh_2yr_dict['adt'])
print(f'SSH processed in {time.time() - t0:.2f}s: adt valid range [{adt_min:.2f}, {adt_max:.2f}] m')

# 7. PROCESS CHLOROPHYLL-A (2020-2021)
print('\n[7/8] Processing Chlorophyll-a from CSVs (Streaming chunked regrid)...')
t0 = time.time()
chlor_daily_list = []

# Prepare regular grid interpolator mesh
T_LAT, T_LON = np.meshgrid(target_lat, target_lon, indexing='ij')
target_pts = np.column_stack([T_LAT.ravel(), T_LON.ravel()])

with zipfile.ZipFile('data.zip', 'r') as z:
    for yr_csv, expected_days in [('data/cleaned/Chlorophyll_2020_cleaned.csv', 365),
                                  ('data/cleaned/Chlorophyll_2021_cleaned.csv', 364)]:
        print(f'Streaming {yr_csv} ({expected_days} days)...')
        t_yr = time.time()
        with z.open(yr_csv) as f:
            # Read in chunks of 708890 rows (1 full day)
            chunksize = 708890
            reader = pd.read_csv(f, chunksize=chunksize, usecols=['latitude', 'longitude', 'chlor_a'])
            for day_idx, chunk in enumerate(reader):
                if day_idx == 0:
                    src_lats = chunk['latitude'].values[:665*1066:1066][::-1] # ascending
                    src_lons = chunk['longitude'].values[:1066]                # ascending
                    
                ch_2d = chunk['chlor_a'].values.reshape(665, 1066)[::-1, :]
                interp = RegularGridInterpolator((src_lats, src_lons), ch_2d, method='linear', bounds_error=False, fill_value=np.nan)
                regrid_day = interp(target_pts).reshape(n_lat, n_lon).astype(np.float32)
                chlor_daily_list.append(regrid_day)
                
                if (day_idx + 1) % 90 == 0 or (day_idx + 1) == expected_days:
                    print(f'  Processed day {day_idx + 1}/{expected_days} ({time.time() - t_yr:.1f}s)...')

chlor_2yr = np.stack(chlor_daily_list, axis=0)
print(f'Chlorophyll-a processed in {time.time() - t0:.2f}s: shape {chlor_2yr.shape}, valid range [{np.nanmin(chlor_2yr):.4f}, {np.nanmax(chlor_2yr):.2f}] mg/m3')

# 8. ASSEMBLE XARRAY DATASETS & EXPORT
print('\n[8/8] Assembling Aligned Datasets, Tabular Files & Reports...')
t0 = time.time()

# Broadcast static fields to time dimension or store as 2D coordinates/variables
# CF-compliant conventions
coords_2yr = {
    'time': ('time', dates_2yr),
    'latitude': ('latitude', target_lat, {'units': 'degrees_north', 'standard_name': 'latitude', 'axis': 'Y'}),
    'longitude': ('longitude', target_lon, {'units': 'degrees_east', 'standard_name': 'longitude', 'axis': 'X'})
}

data_vars_basin_2yr = {
    'sst': (['time', 'latitude', 'longitude'], sst_2yr, {
        'units': 'degree_C', 'long_name': 'Sea Surface Temperature', 'standard_name': 'sea_surface_temperature'
    }),
    'uo': (['time', 'latitude', 'longitude'], uo_2yr, {
        'units': 'm s-1', 'long_name': 'Eastward Ocean Current Velocity', 'standard_name': 'eastward_sea_water_velocity'
    }),
    'vo': (['time', 'latitude', 'longitude'], vo_2yr, {
        'units': 'm s-1', 'long_name': 'Northward Ocean Current Velocity', 'standard_name': 'northward_sea_water_velocity'
    }),
    'u10': (['time', 'latitude', 'longitude'], u10_2yr, {
        'units': 'm s-1', 'long_name': '10-meter Eastward Wind Component', 'standard_name': 'eastward_wind'
    }),
    'v10': (['time', 'latitude', 'longitude'], v10_2yr, {
        'units': 'm s-1', 'long_name': '10-meter Northward Wind Component', 'standard_name': 'northward_wind'
    }),
    'wind_speed': (['time', 'latitude', 'longitude'], wspd_2yr, {
        'units': 'm s-1', 'long_name': '10-meter Wind Speed Magnitude', 'standard_name': 'wind_speed'
    }),
    'chlor_a': (['time', 'latitude', 'longitude'], chlor_2yr, {
        'units': 'mg m-3', 'long_name': 'Chlorophyll-a Concentration', 'standard_name': 'mass_concentration_of_chlorophyll_a_in_sea_water'
    }),
    'adt': (['time', 'latitude', 'longitude'], ssh_2yr_dict['adt'], {
        'units': 'm', 'long_name': 'Absolute Dynamic Topography (Sea Surface Height)', 'standard_name': 'sea_surface_height_above_geoid'
    }),
    'sla': (['time', 'latitude', 'longitude'], ssh_2yr_dict['sla'], {
        'units': 'm', 'long_name': 'Sea Level Anomaly', 'standard_name': 'sea_surface_height_above_sea_level'
    }),
    'ugos': (['time', 'latitude', 'longitude'], ssh_2yr_dict['ugos'], {
        'units': 'm s-1', 'long_name': 'Absolute Geostrophic Velocity (Eastward)', 'standard_name': 'surface_geostrophic_eastward_sea_water_velocity'
    }),
    'vgos': (['time', 'latitude', 'longitude'], ssh_2yr_dict['vgos'], {
        'units': 'm s-1', 'long_name': 'Absolute Geostrophic Velocity (Northward)', 'standard_name': 'surface_geostrophic_northward_sea_water_velocity'
    }),
    'elevation': (['latitude', 'longitude'], bathy_interp, {
        'units': 'm', 'long_name': 'Bathymetric Elevation (negative = depth)', 'standard_name': 'height_above_reference_ellipsoid'
    }),
    'dist_to_coast_km': (['latitude', 'longitude'], dist_km, {
        'units': 'km', 'long_name': 'Shortest Distance to Coastline'
    }),
    'land_mask': (['latitude', 'longitude'], inside_mask, {
        'units': '1', 'long_name': 'Land-Sea Mask (1=Land, 0=Ocean)', 'flag_values': [0, 1], 'flag_meanings': 'ocean land'
    }),
    'ssh_mask': (['latitude', 'longitude'], ssh_mask, {
        'units': '1', 'long_name': 'SSH Data Availability Mask (1=Valid SSH Sub-domain, 0=Outside SSH Coverage)', 'flag_values': [0, 1], 'flag_meanings': 'outside_ssh valid_ssh_domain'
    })
}

attrs_basin = {
    'title': 'Spatially & Temporally Aligned North Indian Ocean PFZ Dataset (Daily)',
    'institution': 'PFZ Ocean Modeling Team - Phase 3 (Mem3)',
    'spatial_resolution': '0.25 degree x 0.25 degree',
    'temporal_resolution': 'Daily (2020-01-01 to 2021-12-30)',
    'crs': 'EPSG:4326 (WGS84)',
    'conventions': 'CF-1.8',
    'creation_date': pd.Timestamp.now().isoformat()
}

ds_daily_basin = xr.Dataset(data_vars=data_vars_basin_2yr, coords=coords_2yr, attrs=attrs_basin)
daily_basin_path = 'data/aligned/aligned_daily_basin.nc'
print(f'Saving {daily_basin_path}...')
ds_daily_basin.to_netcdf(daily_basin_path)

# Monthly Aggregated Basin Dataset
print('Generating Monthly Aggregated Basin Dataset...')
# Separate dynamic variables from static variables
dynamic_vars = ['sst', 'uo', 'vo', 'u10', 'v10', 'wind_speed', 'chlor_a', 'adt', 'sla', 'ugos', 'vgos']
static_vars = ['elevation', 'dist_to_coast_km', 'land_mask', 'ssh_mask']

ds_monthly_dynamic = ds_daily_basin[dynamic_vars].resample(time='1MS').mean()
for sv in static_vars:
    ds_monthly_dynamic[sv] = ds_daily_basin[sv]

ds_monthly_basin = ds_monthly_dynamic
ds_monthly_basin.attrs['title'] = 'Spatially & Temporally Aligned North Indian Ocean PFZ Dataset (Monthly Mean)'
ds_monthly_basin.attrs['temporal_resolution'] = 'Monthly Mean (24 months, 2020-01 to 2021-12)'

monthly_basin_path = 'data/aligned/aligned_monthly_basin.nc'
print(f'Saving {monthly_basin_path}...')
ds_monthly_basin.to_netcdf(monthly_basin_path)

# Coincident Sub-Domain Dataset (Lat: 6.3125 to 19.8125, Lon: 70.9375 to 78.4375)
print('Generating Coincident Sub-domain Datasets (100% spatial overlap)...')
sub_lat_mask = (target_lat >= 6.3125) & (target_lat <= 19.8125)
sub_lon_mask = (target_lon >= 70.9375) & (target_lon <= 78.4375)

ds_daily_coincident = ds_daily_basin.sel(latitude=target_lat[sub_lat_mask], longitude=target_lon[sub_lon_mask])
ds_daily_coincident.attrs['title'] = 'Coincident Multi-Modal PFZ Dataset (Daily - 100% Shared Domain)'
ds_daily_coincident.attrs['domain'] = 'Eastern Arabian Sea / Indian EEZ'

daily_coincident_path = 'data/aligned/aligned_daily_coincident.nc'
print(f'Saving {daily_coincident_path}...')
ds_daily_coincident.to_netcdf(daily_coincident_path)

ds_monthly_coincident = ds_monthly_basin.sel(latitude=target_lat[sub_lat_mask], longitude=target_lon[sub_lon_mask])
ds_monthly_coincident.attrs['title'] = 'Coincident Multi-Modal PFZ Dataset (Monthly Mean - 100% Shared Domain)'
monthly_coincident_path = 'data/aligned/aligned_monthly_coincident.nc'
print(f'Saving {monthly_coincident_path}...')
ds_monthly_coincident.to_netcdf(monthly_coincident_path)

# Tabular Parquet Export
print('Exporting Tabular Parquet Deliverables...')
df_monthly = ds_monthly_basin.to_dataframe().reset_index()
# format date
df_monthly['time'] = df_monthly['time'].dt.strftime('%Y-%m')
df_monthly.rename(columns={'time': 'month'}, inplace=True)
monthly_parquet_path = 'data/aligned/aligned_dataset_monthly.parquet'
df_monthly.to_parquet(monthly_parquet_path, index=False)
print(f'Saved {monthly_parquet_path} ({len(df_monthly):,} rows)')

# Coincident daily parquet (highly compact and 100% populated)
df_coincident_daily = ds_daily_coincident.to_dataframe().reset_index()
df_coincident_daily['time'] = df_coincident_daily['time'].dt.strftime('%Y-%m-%d')
df_coincident_daily.rename(columns={'time': 'date'}, inplace=True)
coincident_parquet_path = 'data/aligned/aligned_coincident_daily.parquet'
df_coincident_daily.to_parquet(coincident_parquet_path, index=False)
print(f'Saved {coincident_parquet_path} ({len(df_coincident_daily):,} rows)')

# Also sample daily basin parquet
# To keep file size optimized, save daily ocean points in parquet
ocean_mask_pts = ds_daily_basin['land_mask'].values == 0
print('Exporting Daily Basin Parquet (Ocean Cells)...')
df_daily_ocean = ds_daily_basin.where(ds_daily_basin['land_mask'] == 0).to_dataframe().dropna(subset=['sst', 'wind_speed'], how='all').reset_index()
df_daily_ocean['time'] = df_daily_ocean['time'].dt.strftime('%Y-%m-%d')
df_daily_ocean.rename(columns={'time': 'date'}, inplace=True)
daily_parquet_path = 'data/aligned/aligned_dataset_daily.parquet'
df_daily_ocean.to_parquet(daily_parquet_path, index=False)
print(f'Saved {daily_parquet_path} ({len(df_daily_ocean):,} rows)')

print(f'\nAll datasets exported in {time.time() - t0:.2f}s!')
print(f'Total Pipeline Duration: {time.time() - start_pipeline_time:.2f}s')
