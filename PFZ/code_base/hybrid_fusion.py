"""
Hybrid Spatial-Temporal Fusion Layer
Project: AI-Based Dynamic Fishing Zone Prediction System (Sahyadri College / VTU)

Implements:
1. Multi-modal ensembling of Spatial Model (LightGBM/RF) and Temporal Front Model.
2. Calibration of final Potential Fishing Zone (PFZ) probability score per grid cell.
3. Advisory confidence classification (High / Moderate / Low / Non-PFZ).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class HybridPFZFusionLayer:
    def __init__(self, spatial_weight: float = 0.65, temporal_weight: float = 0.35):
        """
        Initializes the hybrid fusion layer.
        spatial_weight: Importance of spatial oceanographic feature patterns.
        temporal_weight: Importance of temporal front persistence & stability.
        """
        total = spatial_weight + temporal_weight
        self.w_spatial = spatial_weight / total
        self.w_temporal = temporal_weight / total

    def fuse(self, spatial_probs: np.ndarray, temporal_scores: np.ndarray, 
             bathymetry_depth: np.ndarray = None, dist_to_coast: np.ndarray = None) -> np.ndarray:
        """
        Fuses spatial and temporal predictions into a single probability-of-productive-zone score.
        Applies bathymetric and coastal accessibility gates.
        """
        # Linear and non-linear synergy fusion
        base_fused = (self.w_spatial * spatial_probs) + (self.w_temporal * temporal_scores)
        
        # Non-linear boost when both spatial and temporal signals agree strongly
        synergy_boost = np.where((spatial_probs >= 0.6) & (temporal_scores >= 0.5), 0.08, 0.0)
        fused = np.clip(base_fused + synergy_boost, 0.0, 1.0)

        # Bathymetric masking: Pelagic shoals and benthic habitats decline sharply in abyssal trenches (> 3000m depth)
        if bathymetry_depth is not None:
            deep_abyssal_mask = bathymetry_depth < -3000.0
            fused[deep_abyssal_mask] *= 0.40  # Down-weight open abyssal plains

        # Cap and round to 4 decimals
        return np.round(fused, 4)

    def assign_pfz_advisory_tiers(self, final_probabilities: np.ndarray) -> np.ndarray:
        """
        Assigns fisherman-friendly advisory tiers:
        - HIGH (>= 0.70): Prime PFZ, sharp fronts, high primary productivity.
        - MODERATE (0.50 - 0.69): Favourable foraging condition.
        - LOW (0.35 - 0.49): Marginal condition.
        - NON_PFZ (< 0.35): Unfavourable / dispersed fish distribution.
        """
        tiers = np.empty(len(final_probabilities), dtype=object)
        tiers[final_probabilities >= 0.70] = "HIGH_PFZ"
        tiers[(final_probabilities >= 0.50) & (final_probabilities < 0.70)] = "MODERATE_PFZ"
        tiers[(final_probabilities >= 0.35) & (final_probabilities < 0.50)] = "LOW_PFZ"
        tiers[final_probabilities < 0.35] = "NON_PFZ"
        return tiers

    def generate_advisory_dataframe(self, df_grid: pd.DataFrame, 
                                   spatial_probs: np.ndarray, 
                                   temporal_scores: np.ndarray) -> pd.DataFrame:
        """
        Constructs a complete prediction table with coordinates, date, scores, and tiers.
        """
        res = df_grid[["date", "latitude", "longitude", "sst", "chlor_a", "current_speed", "elevation", "dist_to_coast_km"]].copy()
        
        elev = df_grid["elevation"].values if "elevation" in df_grid.columns else None
        dist = df_grid["dist_to_coast_km"].values if "dist_to_coast_km" in df_grid.columns else None

        res["spatial_prob"] = np.round(spatial_probs, 4)
        res["temporal_persistence"] = np.round(temporal_scores, 4)
        res["pfz_probability"] = self.fuse(spatial_probs, temporal_scores, elev, dist)
        res["advisory_tier"] = self.assign_pfz_advisory_tiers(res["pfz_probability"].values)
        
        return res
