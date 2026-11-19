"""
Personalized Route Suggestion & Fuel Optimization Engine
Project: AI-Based Dynamic Fishing Zone Prediction System (Sahyadri College / VTU)

Implements:
1. Great-circle Haversine distance and nautical compass bearing (heading).
2. Boat-level personalization (Artisanal OBM, Mechanized Trawler, Deep-sea Vessel).
3. Fuel consumption estimation and diesel savings vs unguided random wandering.
4. Strict deduplication of overlapping/identical routes (only unique geographic hotspots).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .config import KEY_PORTS

# Vessel profile specifications common along the Karnataka/Kerala coastline
VESSEL_PROFILES = {
    "artisanal": {
        "name": "Artisanal Outboard Motor (OBM / Fiber Canoe)",
        "engine_hp": 15,
        "cruising_speed_knots": 7.0,
        "fuel_burn_liters_per_hour": 3.5,
        "max_range_km": 40.0,
        "fuel_type": "Kerosene / Petrol"
    },
    "small_mechanized": {
        "name": "Small Mechanized Inshore Trawler / Gillnetter",
        "engine_hp": 90,
        "cruising_speed_knots": 8.5,
        "fuel_burn_liters_per_hour": 14.0,
        "max_range_km": 110.0,
        "fuel_type": "Diesel"
    },
    "deep_sea_trawler": {
        "name": "Multi-Day Deep-Sea Mechanized Trawler",
        "engine_hp": 250,
        "cruising_speed_knots": 9.5,
        "fuel_burn_liters_per_hour": 32.0,
        "max_range_km": 250.0,
        "fuel_type": "Diesel"
    }
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, str]:
    """
    Computes navigation bearing (0° - 360°) and compass cardinal direction from start to destination.
    Uses standard spherical great-circle forward azimuth formula.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    initial_bearing = math.degrees(math.atan2(y, x))
    compass_bearing = (initial_bearing + 360.0) % 360.0

    cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(compass_bearing / 22.5) % 16
    return round(compass_bearing, 1), cardinals[idx]

class PFZRoutePlanner:
    def __init__(self, port_name: str = "Mangaluru Old Port (Bunder)"):
        if port_name not in KEY_PORTS:
            raise ValueError(f"Port '{port_name}' not found. Available: {list(KEY_PORTS.keys())}")
        self.port_name = port_name
        self.port_info = KEY_PORTS[port_name]
        self.port_lat = self.port_info["lat"]
        self.port_lon = self.port_info["lon"]

    def find_optimal_routes(self, pfz_df: pd.DataFrame, 
                            vessel_type: str = "small_mechanized", 
                            top_k: int = 5,
                            diesel_price_inr: float = 88.0,
                            min_probability: float = 0.50,
                            **kwargs) -> List[Dict[str, Any]]:
        """
        Ranks and suggests optimal fishing zone destinations tailored to the boat's
        home port, operational range, fuel capacity, and minimum confidence threshold.
        
        Strict Non-Overlapping Rule:
        - Deduplicates by geographic coordinates (lat, lon).
        - If two candidate routes have the same set of values (same distance ±3km, depth, or direction),
          only ONE among them is considered.
        - Spatially separates routes so each suggested route branches out to a distinct fishing ground.
        """
        vessel = VESSEL_PROFILES.get(vessel_type, VESSEL_PROFILES["small_mechanized"])
        max_range = vessel["max_range_km"]
        speed_kmh = vessel["cruising_speed_knots"] * 1.852
        burn_rate = vessel["fuel_burn_liters_per_hour"]

        # Dynamic sensitivity weighting based on user-chosen confidence threshold:
        # At higher thresholds (e.g. 85-99%), model confidence dominates heavily.
        # At lower thresholds (e.g. 50-70%), closer proximity and fuel savings are prioritized.
        conf_weight = max(0.40, min(0.85, min_probability))
        dist_weight = 1.0 - conf_weight

        # Step 1: Deduplicate incoming dataframe by (latitude, longitude), taking maximum probability
        if "latitude" in pfz_df.columns and "longitude" in pfz_df.columns:
            df_unique = pfz_df.sort_values(by="pfz_probability", ascending=False).drop_duplicates(
                subset=["latitude", "longitude"]
            )
        else:
            df_unique = pfz_df

        candidates = []
        for _, row in df_unique.iterrows():
            prob = float(row.get("pfz_probability", 0.0))
            if prob < min_probability:
                continue

            lat, lon = float(row["latitude"]), float(row["longitude"])
            dist_km = haversine_distance(self.port_lat, self.port_lon, lat, lon)
            
            # Within boat range and at least 5 km offshore
            if dist_km > max_range or dist_km < 5.0:
                continue

            bearing_deg, cardinal = calculate_bearing(self.port_lat, self.port_lon, lat, lon)
            dist_nm = dist_km / 1.852

            # One-way travel time & round-trip fuel
            travel_time_hours = dist_km / speed_kmh
            roundtrip_travel_hours = travel_time_hours * 2.0
            fuel_liters = round(roundtrip_travel_hours * burn_rate, 1)
            fuel_cost_inr = round(fuel_liters * diesel_price_inr, 0)

            # Estimated fuel consumed during unguided random wandering (2.4x baseline)
            unguided_fuel_liters = round(fuel_liters * 2.4, 1)
            estimated_fuel_saved_liters = round(unguided_fuel_liters - fuel_liters, 1)
            savings_inr = round(estimated_fuel_saved_liters * diesel_price_inr, 0)

            proximity_score = max(0.0, 1.0 - (dist_km / max_range))
            eff_score = (conf_weight * prob) + (dist_weight * proximity_score)

            depth_val = float(row.get("elevation", -65.0))
            candidates.append({
                "latitude": round(lat, 3),
                "longitude": round(lon, 3),
                "pfz_probability": round(prob, 4),
                "advisory_tier": row.get("advisory_tier", "HIGH_PFZ" if prob >= 0.70 else "MODERATE_PFZ"),
                "sst": round(float(row.get("sst", 28.5)), 1),
                "chlor_a": round(float(row.get("chlor_a", 1.2)), 2),
                "depth_m": round(depth_val, 0),
                "distance_km": round(dist_km, 1),
                "distance_nm": round(dist_nm, 1),
                "bearing_degrees": bearing_deg,
                "compass_heading": cardinal,
                "travel_time_one_way_hrs": round(travel_time_hours, 1),
                "roundtrip_fuel_liters": fuel_liters,
                "fuel_cost_inr": fuel_cost_inr,
                "fuel_saved_liters": estimated_fuel_saved_liters,
                "fuel_savings_inr": savings_inr,
                "efficiency_score": round(eff_score, 4)
            })

        # Step 2: Sort dynamically by efficiency score so the threshold slider directly impacts the ranking
        candidates.sort(key=lambda x: (x["efficiency_score"], x["pfz_probability"]), reverse=True)

        # Step 3: Enforce strict uniqueness (filter out overlapping/redundant routes)
        # If two routes have the same set of values (same distance ±3km & same heading, or same lat/lon),
        # ONLY ONE is kept!
        selected_routes = []
        for cand in candidates:
            is_redundant = False
            for sel in selected_routes:
                # Check spatial separation distance
                sep_km = haversine_distance(cand["latitude"], cand["longitude"], sel["latitude"], sel["longitude"])
                dist_diff = abs(cand["distance_km"] - sel["distance_km"])
                depth_diff = abs(cand["depth_m"] - sel["depth_m"])

                # If overlapping in space (< 15 km) OR identical distance and heading, it is redundant
                if sep_km < 14.0 or (dist_diff < 3.0 and cand["compass_heading"] == sel["compass_heading"]) or (dist_diff < 1.0 and depth_diff < 2.0):
                    is_redundant = True
                    break

            if not is_redundant:
                selected_routes.append(cand)

            if len(selected_routes) >= top_k:
                break

        return selected_routes[:top_k]
