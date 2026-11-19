# AI-Based Dynamic Fishing Zone Prediction System (PFZ)

**Mini Project:** Bachelor of Engineering in Computer Science & Engineering  
**Institution:** Sahyadri College of Engineering & Management, Mangaluru (VTU)  
**Academic Year:** 2026 – 2027  
**Faculty Guide:** Prof. Chaitrashree M, Department of CSE  
**Students:** Gahan Shetty, Krish Meghapara, Sathvik V Shetty, Swayam S Rai  

---

## 1. Project Overview & Societal Relevance

In coastal India (Karnataka, Kerala, Goa, Tamil Nadu), small-scale and artisanal fishermen spend between 40% and 65% of their total operating expenditure on diesel fuel wandering the sea in unguided searches for productive fishing grounds. 

While the **Indian National Centre for Ocean Information Services (INCOIS)** provides satellite-based Potential Fishing Zone (PFZ) advisories based on thermal and chlorophyll fronts, traditional advisories:
1. Suffer from spatial coarse resolution and broad boundaries.
2. Lack boat-level personalization (cannot customize routes for boat range, fuel capacity, or engine horsepower).
3. Do not learn adaptively from historical fish landings or dynamic multi-day front persistence.

This project delivers a **hybrid machine learning decision-support system** that fuses multi-modal satellite oceanographic data with historical catch records and generates:
- **Dynamic PFZ Probability Heatmaps** at 0.25° × 0.25° spatial resolution.
- **Personalized GPS Route Suggestions** with compass headings, travel times, and fuel savings calculations from local fishing harbors (e.g., Mangaluru Old Port, Malpe Harbor, Karwar, Beypore, Kochi).
- A bilingual (English & Kannada) low-bandwidth web dashboard for coastal fishermen.

---

## 2. Directory Architecture

The repository is structured strictly into two core subdirectories:

```
PFZ/
├── data/                               # Dedicated Data Repository
│   ├── coastal_ports.json              # GPS coordinates & metadata for Karnataka & Kerala harbors
│   ├── historical_catch_landing_data.csv # CMFRI-calibrated species landing records (2020-2021)
│   ├── ocean_features_daily.parquet    # Spatio-temporally aligned oceanographic multi-modal dataset
│   ├── tabular_parquet/                # Aligned daily & monthly basin/coincident parquet files
│   ├── netcdf/                         # CF-1.8 gridded NetCDF files (basin & coincident daily/monthly)
│   └── mem4_integrated_dataset/        # Integrated multi-modal CSV & NetCDF datasets with QC reports
│
├── code_base/                          # Machine Learning & Software Pipeline
│   ├── __init__.py                     # Python package definition
│   ├── config.py                       # Hyperparameters, coordinate bounds, and INCOIS thresholds
│   ├── data_loader.py                  # Ingestion, CMFRI landing generator, & temporal train/test split
│   ├── ocean_front_detector.py         # INCOIS-benchmarked thermal & biological front detector
│   ├── spatial_model.py                # LightGBM Gradient Boosting & Random Forest spatial models
│   ├── temporal_model.py               # Multi-day front evolution & temporal persistence model
│   ├── hybrid_fusion.py                # Synergy fusion layer producing calibrated PFZ scores
│   ├── route_planner.py                # Navigation bearing, fuel estimation, & route optimization
│   ├── train_pipeline.py               # Master end-to-end training and evaluation script
│   ├── dashboard.py                    # Bilingual (English/Kannada) Streamlit interactive UI
│   └── saved_models/                   # Persisted trained joblib model checkpoints
│
├── outputs/                            # Auto-generated Evaluation Deliverables
│   ├── model_evaluation_metrics.json   # Out-of-sample ROC-AUC, PR-AUC, F1, Precision, Recall
│   ├── feature_importances.png         # Bar plot of top driving oceanographic variables
│   ├── sample_pfz_advisory_predictions.csv # Fused PFZ probabilities across spatial grid
│   ├── pfz_advisory_2021.parquet       # Full 2021 test year gridded predictions
│   └── sample_port_routes.json         # Optimal route recommendations from Mangaluru port
│
├── phase_documentation_and_scripts/    # Historical Phase Reports & Pipelines
│   ├── mem3_alignment_reports/         # Alignment compatibility matrices & engineering audit
│   ├── phase_data_scripts/             # Early spatial interpolation and alignment scripts
│   └── README_MEM3_ORIGINAL.md         # Original Phase 3 dataset specification
│
└── README.md                           # Documentation & execution guide
```

---

## 3. Machine Learning Architecture

```
                                  [ MULTI-MODAL DATA ]
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       Satellite & Numerical Reanalysis                    Historical Catch Records
    (SST, Current u/v, Wind, Chlor-a, ADT, SLA)         (Species, Landings, Fishing Effort, CPUE)
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                              [ INCOIS FRONT DETECTION ENGINE ]
                           - Spatial SST Gradient: |∇SST|
                           - Chlorophyll-a Bloom Boundary
                           - Frontal Overlap: |∇SST| ∩ Chlor_a
                           - 3-Day Thermal Front Persistence
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         [ 1. SPATIAL ML MODEL ]                             [ 2. TEMPORAL MODEL ]
     LightGBM Gradient Boosting / RF                     Multi-Day Front Evolution
     (Learns static & dynamic habitat)                   (Captures advection & front stability)
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                                 [ 3. HYBRID FUSION LAYER ]
                     P_final = w1 * P_spatial + w2 * S_temporal + Synergy
                                              │
                                              ▼
                             [ 4. NAVIGATION & ROUTE PLANNER ]
                               - Haversine Navigation Heading (Bearing)
                               - Vessel Engine Fuel Burn Optimization
                               - Estimated Diesel Savings vs. Random Search
```

---

## 4. Execution & Quick Start

### A. Run End-to-End Model Training & Evaluation
To train the spatial models, fit the temporal evolution layer, generate evaluation metrics, and simulate routes:

```bash
# From the PFZ root folder:
python code_base/train_pipeline.py
```

### B. Launch the Interactive Web Dashboard
To launch the bilingual web interface:

```bash
streamlit run code_base/dashboard.py
```

---

## 5. Alignment with UN Sustainable Development Goals (SDGs)

* **SDG 14 (Life Below Water):** Encourages targeted, precision fishing that avoids non-productive and ecologically sensitive marine areas, reducing habitat disruption and unnecessary marine emissions.
* **SDG 1 (No Poverty):** Saves between 40 to 80 Liters of diesel per trip (₹3,500 – ₹7,000 INR savings per voyage), directly bolstering the net take-home income of artisanal and small-scale fishing families.
* **SDG 9 (Industry, Innovation & Infrastructure):** Bridges space technology (ISRO/CMEMS satellites), artificial intelligence, and mobile decision-support for traditional coastal communities.
