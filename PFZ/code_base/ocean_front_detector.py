"""
Oceanographic Front Detector & INCOIS-Benchmarked PFZ Labeler
Project: AI-Based Dynamic Fishing Zone Prediction System (Sahyadri College / VTU)

Implements:
1. Spatial SST thermal gradient computation (Sobel / Central Difference).
2. Chlorophyll-a front & bloom thresholding.
3. Frontal persistence across time.
4. INCOIS-benchmarked Potential Fishing Zone (PFZ) indicator.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import sobel
from .config import PFZ_CRITERIA

def compute_spatial_gradients(grid_2d: np.ndarray) -> np.ndarray:
    """
    Computes 2D spatial gradient magnitude sqrt(dx^2 + dy^2).
    Uses reflection/nearest handling for NaN margins.
    """
    valid_mask = ~np.isnan(grid_2d)
    filled = np.nan_to_num(grid_2d, nan=np.nanmean(grid_2d) if np.any(valid_mask) else 0.0)
    
    # Sobel spatial gradient filters along latitude (axis 0) and longitude (axis 1)
    grad_y = sobel(filled, axis=0) / 4.0
    grad_x = sobel(filled, axis=1) / 4.0
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude[~valid_mask] = np.nan
    return magnitude

def compute_ocean_front_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates thermal fronts, chlorophyll bloom fronts, and frontal overlap
    for each daily time slice across the spatial grid.
    """
    print("Computing INCOIS-benchmarked oceanographic front features...")
    df = df.copy()
    
    # Group by date to process 2D spatial slices
    processed_slices = []
    unique_dates = df["date"].unique()

    for d in unique_dates:
        slice_df = df[df["date"] == d].copy()
        
        # Sort spatial grid
        slice_df = slice_df.sort_values(by=["latitude", "longitude"])
        lats = sorted(slice_df["latitude"].unique())
        lons = sorted(slice_df["longitude"].unique())
        n_lat, n_lon = len(lats), len(lons)
        
        if len(slice_df) != n_lat * n_lon:
            # Fallback if grid is not rectangular in this slice: point-wise gradient approx
            slice_df["sst_gradient"] = np.abs(slice_df["sst"] - slice_df["sst"].mean()).fillna(0)
            slice_df["chlor_a_front"] = (slice_df["chlor_a"] >= PFZ_CRITERIA["chlor_min"]).astype(float).fillna(0)
            slice_df["front_overlap"] = ((slice_df["sst_gradient"] > PFZ_CRITERIA["sst_grad_threshold"]) & 
                                         (slice_df["chlor_a_front"] > 0)).astype(float)
            processed_slices.append(slice_df)
            continue
            
        # Reshape to 2D grid
        sst_grid = slice_df["sst"].values.reshape((n_lat, n_lon))
        sst_grad = compute_spatial_gradients(sst_grid).flatten()
        slice_df["sst_gradient"] = sst_grad
        
        # Chlorophyll front & bloom presence
        chlor_vals = slice_df["chlor_a"].values
        chlor_front = np.where(
            (chlor_vals >= PFZ_CRITERIA["chlor_min"]) & (chlor_vals <= PFZ_CRITERIA["chlor_max"]),
            1.0,
            0.0
        )
        slice_df["chlor_a_front"] = chlor_front
        
        # INCOIS Frontal Overlap: Co-occurrence of thermal front and chlorophyll productivity
        thermal_front_active = slice_df["sst_gradient"] >= PFZ_CRITERIA["sst_grad_threshold"]
        slice_df["front_overlap"] = np.where(thermal_front_active & (slice_df["chlor_a_front"] > 0), 1.0, 0.0)
        
        processed_slices.append(slice_df)

    res_df = pd.concat(processed_slices, ignore_index=True)
    
    # Fill remaining NaNs in gradients safely
    res_df["sst_gradient"] = res_df["sst_gradient"].fillna(0.0)
    res_df["chlor_a_front"] = res_df["chlor_a_front"].fillna(0.0)
    res_df["front_overlap"] = res_df["front_overlap"].fillna(0.0)
    
    # Calculate temporal persistence: 3-day rolling front presence
    res_df = res_df.sort_values(by=["latitude", "longitude", "date"])
    res_df["sst_persistence"] = (
        res_df.groupby(["latitude", "longitude"])["front_overlap"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        .fillna(0.0)
    )
    
    # Generate INCOIS Ground-Truth PFZ Classification Target (1 = Potential Fishing Zone, 0 = Non-PFZ)
    # Based on INCOIS PFZ advisory rules:
    # 1. SST between 25°C and 30.5°C
    # 2. Significant thermal gradient (front)
    # 3. Elevated primary productivity or front persistence
    # 4. Continental shelf / slope region (-1000m <= elevation <= -25m)
    c = PFZ_CRITERIA
    incois_condition = (
        (res_df["sst"] >= c["sst_min"]) & (res_df["sst"] <= c["sst_max"]) &
        (res_df["elevation"] >= c["shelf_break_min_depth"]) & (res_df["elevation"] <= c["shelf_break_max_depth"]) &
        (
            (res_df["front_overlap"] == 1.0) | 
            (res_df["sst_gradient"] >= c["sst_grad_threshold"] * 1.2) |
            (res_df["sst_persistence"] >= 0.5)
        )
    )
    res_df["incois_pfz_target"] = incois_condition.astype(int)
    
    # Continuous PFZ suitability index (0.0 to 1.0)
    norm_grad = np.clip(res_df["sst_gradient"] / 0.8, 0, 1)
    norm_pers = np.clip(res_df["sst_persistence"], 0, 1)
    shelf_score = np.where((res_df["elevation"] >= -200) & (res_df["elevation"] <= -30), 1.0, 0.5)
    res_df["pfz_suitability_score"] = np.round(
        (0.4 * norm_grad + 0.3 * norm_pers + 0.2 * shelf_score + 0.1 * res_df["front_overlap"]), 
        3
    )

    print(f"✓ Front features computed. Positive PFZ cells: {res_df['incois_pfz_target'].sum():,} "
          f"({res_df['incois_pfz_target'].mean()*100:.2f}% of domain)")
    return res_df
