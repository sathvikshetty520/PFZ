"""
End-to-End Master Training & Evaluation Pipeline
Project: AI-Based Dynamic Fishing Zone Prediction System (Sahyadri College / VTU)

Executes:
1. Data initialization & ingestion (coincident Arabian Sea features + SW coast filter).
2. INCOIS oceanographic front detection (gradients, persistence, overlap).
3. Temporal split (Train: Year 2020 | Test: Year 2021).
4. Spatial model training (LightGBM & Random Forest).
5. Temporal sequence & front evolution model training.
6. Hybrid fusion layer integration & calibration.
7. Route planning simulation from Mangaluru & Malpe ports.
8. Exporting models, metric reports, and summary plots.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure package imports work when executed directly
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from code_base.config import (
    DATA_DIR, CODE_DIR, MODELS_DIR, OUTPUTS_DIR,
    TRAIN_YEAR, TEST_YEAR, ALL_MODEL_FEATURES
)
from code_base.data_loader import (
    setup_ocean_dataset, generate_historical_catch_data,
    load_ocean_features, get_train_test_split
)
from code_base.ocean_front_detector import compute_ocean_front_features
from code_base.spatial_model import SpatialPFZModel
from code_base.temporal_model import TemporalFrontEvolutionModel
from code_base.hybrid_fusion import HybridPFZFusionLayer
from code_base.route_planner import PFZRoutePlanner

def run_pipeline():
    print("=" * 75)
    print("🌊 AI-BASED DYNAMIC FISHING ZONE PREDICTION SYSTEM (PFZ)")
    print("   VTU Mini Project - Sahyadri College of Engineering & Management")
    print("=" * 75)
    start_total_time = time.time()

    # --- STEP 1: DATA INGESTION & SETUP ---
    print("\n[STEP 1/7] Ingesting and preparing datasets...")
    setup_ocean_dataset()
    generate_historical_catch_data(num_samples=12000)
    
    # Load ocean features (focused on Karnataka, Kerala & Goa coastal waters)
    df_raw = load_ocean_features(sw_coast_only=True)
    
    # --- STEP 2: INCOIS OCEANOGRAPHIC FRONT DETECTION ---
    print("\n[STEP 2/7] Extracting thermal & biological front features...")
    df_engineered = compute_ocean_front_features(df_raw)

    # --- STEP 3: TEMPORAL TRAIN-TEST SPLIT ---
    print("\n[STEP 3/7] Partitioning data into temporal train/test sets...")
    train_df, test_df = get_train_test_split(df_engineered)
    
    X_train = train_df
    y_train = train_df["incois_pfz_target"]
    X_test = test_df
    y_test = test_df["incois_pfz_target"]

    # --- STEP 4: SPATIAL MODEL TRAINING (LIGHTGBM & RANDOM FOREST) ---
    print("\n[STEP 4/7] Training Spatial Machine Learning Models...")
    
    # 4A. LightGBM (Gradient Boosting)
    lgb_model = SpatialPFZModel(model_type="lightgbm")
    lgb_importances = lgb_model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    lgb_metrics = lgb_model.evaluate(X_test, y_test)
    lgb_model.save()

    # 4B. Random Forest Benchmark (subsample for fast training)
    rf_model = SpatialPFZModel(model_type="random_forest")
    rf_sample = X_train.sample(n=min(50000, len(X_train)), random_state=42)
    rf_model.train(rf_sample, rf_sample["incois_pfz_target"])
    rf_metrics = rf_model.evaluate(X_test.sample(n=min(20000, len(X_test)), random_state=42), 
                                   y_test.loc[X_test.sample(n=min(20000, len(X_test)), random_state=42).index])
    rf_model.save()

    # --- STEP 5: TEMPORAL FRONT EVOLUTION MODEL ---
    print("\n[STEP 5/7] Fitting Temporal Front Dynamics Model...")
    temporal_model = TemporalFrontEvolutionModel(lookback_days=5)
    temporal_model.fit(train_df)
    temporal_model.save()

    # Predict temporal persistence on test set
    test_temporal_scores = temporal_model.predict_temporal_persistence(test_df)

    # --- STEP 6: HYBRID FUSION LAYER ---
    print("\n[STEP 6/7] Fusing Spatial and Temporal Predictions...")
    fusion_layer = HybridPFZFusionLayer(spatial_weight=0.65, temporal_weight=0.35)
    test_spatial_probs = lgb_model.predict_proba(test_df)
    
    advisory_df = fusion_layer.generate_advisory_dataframe(
        test_df, test_spatial_probs, test_temporal_scores.values
    )
    
    # Save test advisory predictions (both Parquet for full dataset and CSV for preview)
    advisory_parquet = OUTPUTS_DIR / "pfz_advisory_2021.parquet"
    advisory_df.to_parquet(advisory_parquet, index=False)
    advisory_path = OUTPUTS_DIR / "sample_pfz_advisory_predictions.csv"
    # Prioritize active PFZ rows for user-friendly CSV preview
    high_pfz_sample = pd.concat([
        advisory_df[advisory_df["advisory_tier"] == "HIGH_PFZ"].head(10000),
        advisory_df[advisory_df["advisory_tier"] != "HIGH_PFZ"].head(10000)
    ])
    high_pfz_sample.to_csv(advisory_path, index=False)
    print(f"✓ Saved full advisory predictions to {advisory_parquet}")
    print(f"✓ Saved sample preview CSV to {advisory_path}")

    # --- STEP 7: ROUTE PLANNING SIMULATION FROM MANGALURU PORT ---
    print("\n[STEP 7/7] Simulating Personalized Route Planning from Ports...")
    # Find a date with active high-confidence PFZs in the domain
    high_dates = advisory_df[advisory_df["advisory_tier"] == "HIGH_PFZ"]["date"].unique()
    test_date = high_dates[0] if len(high_dates) > 0 else test_df["date"].iloc[-1]
    day_predictions = advisory_df[advisory_df["date"] == test_date]
    
    # Route from Mangaluru Old Port (Bunder)
    mangalore_planner = PFZRoutePlanner(port_name="Mangaluru Old Port (Bunder)")
    mangalore_routes = mangalore_planner.find_optimal_routes(
        day_predictions, vessel_type="small_mechanized", top_k=5
    )

    # Fallback to broader day search if localized cells on exact day were further offshore
    if len(mangalore_routes) == 0:
        mangalore_routes = mangalore_planner.find_optimal_routes(
            advisory_df[advisory_df["advisory_tier"] == "HIGH_PFZ"], 
            vessel_type="small_mechanized", top_k=5
        )

    print(f"\n🚢 TOP RECOMMENDED ROUTES FOR MANGALURU HARBOR (Date: {test_date}):")
    print("-" * 75)
    for i, r in enumerate(mangalore_routes, 1):
        print(f"  #{i} Target: Lat {r['latitude']}°N, Lon {r['longitude']}°E | Score: {r['pfz_probability']:.2f} ({r['advisory_tier']})")
        print(f"     Heading: {r['bearing_degrees']}° ({r['compass_heading']}) | Distance: {r['distance_km']} km ({r['distance_nm']} NM)")
        print(f"     Est. Travel Time: {r['travel_time_one_way_hrs']} hrs (one-way) | Fuel: {r['roundtrip_fuel_liters']} L (Cost: ₹{r['fuel_cost_inr']:,.0f})")
        print(f"     Estimated Fuel Saved vs. Wide-Area Search: ~{r['fuel_saved_liters']} L (Savings: ₹{r['fuel_savings_inr']:,.0f})")
        print("-" * 75)

    # Export Top Routes to JSON
    routes_path = OUTPUTS_DIR / "sample_port_routes.json"
    with open(routes_path, "w", encoding="utf-8") as f:
        json.dump(mangalore_routes, f, indent=2)
    print(f"✓ Exported route plan to {routes_path}")

    # --- EXPORT EVALUATION REPORT & METRICS ---
    report_data = {
        "project": "AI-Based Dynamic Fishing Zone Prediction System",
        "institution": "Sahyadri College of Engineering & Management, Mangaluru (VTU)",
        "models_evaluated": {
            "LightGBM_Spatial": lgb_metrics,
            "RandomForest_Spatial": rf_metrics
        },
        "top_feature_importances": dict(list(lgb_importances.items())[:8]),
        "total_training_records": len(train_df),
        "total_testing_records": len(test_df),
        "sample_routes_generated": len(mangalore_routes)
    }
    
    report_path = OUTPUTS_DIR / "model_evaluation_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"✓ Full evaluation report saved to {report_path}")

    # Generate Feature Importance Bar Plot
    plt.figure(figsize=(9, 5))
    top_feats = list(lgb_importances.keys())[:10]
    top_vals = [lgb_importances[k] for k in top_feats]
    plt.barh(top_feats[::-1], top_vals[::-1], color="#0284c7")
    plt.title("PFZ Prediction System: Top Oceanographic Feature Importances (LightGBM)")
    plt.xlabel("Normalized Gain Importance")
    plt.tight_layout()
    plot_path = OUTPUTS_DIR / "feature_importances.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✓ Saved feature importance plot to {plot_path}")

    elapsed = round(time.time() - start_total_time, 1)
    print(f"\n🎉 PIPELINE EXECUTION COMPLETED IN {elapsed} SECONDS!")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
