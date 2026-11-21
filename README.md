# 🐟 Matsya-AI: AI-Based Dynamic Fishing Zone Prediction System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML-LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-informational.svg)](Matsya_AI_IEEE_Research_Paper.docx)

> **VTU Mini Project (Academic Year 2026–2027)**  
> **Department of Computer Science & Engineering**  
> **Sahyadri College of Engineering & Management, Mangaluru** (Autonomous Institution affiliated to Visvesvaraya Technological University, Belagavi)

---

## 👥 Project Team & Mentorship
- **Project Guide:** Prof. Chaitrashree M (Assistant Professor, Dept. of CSE)
- **Team Authors:**
  - Gahan Shetty (`4SF24CS060`)
  - Krish Meghapara (`4SF24CS088`)
  - Sathvik V Shetty (`4SF24CS186`)
  - Swayam S Rai (`4SF24CS228`)

---

## 🌊 Overview & Problem Statement
In the coastal waters of the Southeastern Arabian Sea (Karnataka, Kerala, Goa), artisanal and small-scale mechanized fishermen spend **40% to 65% of their trip expenditure on diesel fuel** searching blindly for productive pelagic shoals. 

While the **Indian National Centre for Ocean Information Services (INCOIS)** disseminates daily satellite Potential Fishing Zone (PFZ) advisories based on Sea Surface Temperature (SST) and chlorophyll-a fronts, traditional advisories:
1. Suffer from broad spatial resolutions and heuristic thresholding.
2. Lack harbor-to-sea navigational routing and engine-specific fuel burn calculations.
3. Lack closed-loop adaptation to dynamic multi-day front persistence and commercial catch landing outcomes.

**Matsya-AI** solves this through a multi-modal spatio-temporal AI fusion engine that integrates satellite oceanography (SST, chlorophyll-a, surface geostrophic currents, GEBCO bathymetry) with empirical CMFRI catch landing records. It provides **dynamic PFZ probability heatmaps**, **geodesic waypoint navigation**, and **trip fuel savings analytics** through an accessible bilingual (English/Kannada) web GIS dashboard.

---

## 🚀 Key Highlights & Experimental Results

- **Model Accuracy (Holdout 2021 Dataset — 331,968 instances):**
  - **Overall Accuracy:** `99.98%`
  - **ROC-AUC & PR-AUC:** `1.000`
  - **Precision (PFZ Class):** `99.97%` *(Only 4 false alarms out of 15,428 positive predictions)*
  - **Recall (PFZ Class):** `99.68%` *(Missed only 49 events out of 15,473 actual zones)*
  - **F1-Score:** `0.9983`
  - **Inference Latency:** `~0.38 seconds` (LightGBM core)
- **Physical Feature Attribution:**
  - Thermal front gradient magnitude $\nabla\text{SST}$ accounts for **49.18%** of model importance.
  - GEBCO continental shelf bathymetry (-30 m to -200 m) accounts for **42.95%**.
  - Combined physical drivers govern **>92.1%** of predictive capability.
- **Quantified Fleet Fuel Savings:**
  - **38% to 44% reduction** in search diesel consumption per voyage.
  - Route #1 from Mangaluru Old Port saves **~57.1 L of diesel (₹5,025/trip)** for small mechanized trawlers.
  - Estimated **1.9× to 2.4× boost** in Catch per Unit Effort (CPUE).

---

## 🏛️ System Architecture

```
                                [ MULTI-MODAL SATELLITE INPUTS ]
                   MODIS/Oceansat SST | Sentinel-3 Chl-a | CMEMS Currents | GEBCO Depth
                                                │
                                                ▼
                             [ OCEANOGRAPHIC FRONT DETECTION ENGINE ]
                         - 2D Sobel Thermal & Chlorophyll Gradients (|∇SST|, |∇Chl|)
                         - Dual Front Overlap Coincidence Index (O_front)
                         - Multi-Day Hydrodynamic Persistence Modeling (Π_t)
                         - Continental Shelf Break Filter (-30m to -200m)
                                                │
                                                ▼
                               [ SPATIO-TEMPORAL HYBRID ML FUSION ]
                         - LightGBM Spatial Classifier (w_s = 0.65)
                         - Temporal Front Evolution Model (w_t = 0.35)
                         - Calibrated Probabilistic Synergy Output P_PFZ
                         - Strict Terrestrial & Coastal Marine Masking
                                                │
                                                ▼
                             [ HARBOR-CENTRIC ROUTE OPTIMIZATION ]
                         - Great-Circle Navigation & Forward Azimuth Steering
                         - Multi-Criteria Fuel Efficiency Score (Confidence vs Range)
                         - Non-Overlapping Route Deduplication (Δd ≥ 14 km)
                                                │
                                                ▼
                             [ BILINGUAL STREAMLIT WEB GIS DASHBOARD ]
                         - Tab 1: Clean Google Satellite Route Navigation
                         - Tab 2: Biological & Upwelling Fish Density Heatmap (4-tier scale)
                         - English & Kannada (ಕನ್ನಡ) Crew Interface
```

---

## 📂 Repository Structure

```
.
├── README.md                          # Repository documentation & guide
├── requirements.txt                   # Project Python dependencies
├── .env.example                       # Environment template for API keys
├── .gitignore                         # Secure exclusion of credentials, raw data & binaries
│
└── PFZ/
    ├── code_base/                     # Core Machine Learning & Web Pipeline
    │   ├── config.py                  # Physical thresholds, coordinates, and hyperparameters
    │   ├── data_loader.py             # Satellite feature ingestion & CMFRI landing generator
    │   ├── ocean_front_detector.py    # 2D gradient extraction & persistence tracking
    │   ├── spatial_model.py           # LightGBM and Random Forest spatial classifiers
    │   ├── temporal_model.py          # Temporal front dynamics & advection persistence
    │   ├── hybrid_fusion.py           # Calibrated spatio-temporal fusion engine
    │   ├── route_planner.py           # Haversine distance, compass bearing & fuel optimization
    │   ├── train_pipeline.py          # Master model training & holdout validation script
    │   └── dashboard.py               # Streamlit GIS dashboard (Google Satellite & Heatmap)
    │
    ├── data/                          # Ports metadata, CMFRI landings & aligned datasets
    │   ├── coastal_ports.json         # Harbors coordinates (Mangaluru, Malpe, Karwar, etc.)
    │   └── historical_catch_landing_data.csv # Species catch records
    │
    └── outputs/                       # Model Evaluation & Visual Heatmaps
        ├── fig1_system_architecture.png       # Full block diagram
        ├── fig2_satellite_ocean_heatmaps.png   # 4-panel spatial ocean grids
        ├── fig3_front_gradients.png           # Thermal/chlorophyll front gradients
        ├── fig4_model_evaluation.png          # ROC, PR curves & Confusion Matrix
        ├── fig6_pfz_routes_map.png            # Regional PFZ heatmap & Mangaluru routes
        ├── fig7_dashboard_heatmap.png         # Dual-mode operational Web GIS interface
        ├── fig8_seasonal_dynamics.png         # 4-season upwelling evolution (2021)
        ├── fig9_fleet_fuel_economics.png      # Fleet fuel savings & CPUE benchmark
        ├── feature_importances.png            # Top driving feature attribution bar chart
        ├── model_evaluation_metrics.json      # Complete quantitative evaluation metrics
        └── pfz_advisory_2021.parquet          # 331,968-row 2021 holdout advisory dataset
```

---

## ⚡ Quick Start & Execution

### 1. Prerequisites
- Python 3.10 or 3.11
- Pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or manually install core packages:
pip install streamlit lightgbm scikit-learn pandas numpy matplotlib joblib python-docx
```

### 3. Configure Environment Variables
Copy the template and configure your Google Maps API key (with Maps JavaScript API enabled):
```bash
cp .env.example .env
```
Edit `.env`:
```env
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```
*(Note: `.env` is strictly ignored by `.gitignore` to protect sensitive credentials).*

### 4. Run the Master Training Pipeline
```bash
python PFZ/code_base/train_pipeline.py
```

### 5. Launch the Web GIS Dashboard
```bash
streamlit run PFZ/code_base/dashboard.py
```
Or simply double-click `PFZ/run_dashboard.bat` on Windows.

---

## 📜 License & Citation
Developed as an academic mini-project under the Department of Computer Science & Engineering, **Sahyadri College of Engineering & Management, Mangaluru**. For research or educational inquiries, please contact the project team or department guide.
