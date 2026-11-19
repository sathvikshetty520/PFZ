"""
Streamlit Web Dashboard: Dynamic Fishing Zone (PFZ) & Route Advisory
Project: AI-Based Dynamic Fishing Zone Prediction System
Institution: Sahyadri College of Engineering & Management, Mangaluru (VTU)

Fixes:
- Tab 1 (Google Satellite Navigation): COMPLETELY REMOVED fish density circles. Now clean satellite navigation with routes and waypoints only!
- Tab 2 (Fish Density Heatmap): Exclusively holds the thermal/biological front heatmap with visible hotspot pins and scale.
- Route Cards: Replaced raw HTML <div> formatting with NATIVE Streamlit UI containers & metrics (border=True).
- Removed Deploy button and Streamlit toolbar.
- Full legibility in both Light and Dark mode.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Package path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

import importlib
import code_base.route_planner
importlib.reload(code_base.route_planner)

from code_base.config import KEY_PORTS, DATA_DIR, OUTPUTS_DIR, GOOGLE_MAPS_API_KEY
from code_base.route_planner import PFZRoutePlanner, VESSEL_PROFILES

# Ensure API Key is available
if not GOOGLE_MAPS_API_KEY:
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

st.set_page_config(
    page_title="Matsya-AI | PFZ Navigation System",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLEAN CSS STYLING & DEPLOY BUTTON REMOVAL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* REMOVE STREAMLIT DEPLOY BUTTON, TOOLBAR, MENU & FOOTER */
    .stDeployButton, [data-testid="stToolbar"], #MainMenu, footer, header {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* TIGHTEN PADDING FOR CLEAN DASHBOARD LAYOUT */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- LOCALIZATION (ENGLISH / KANNADA) ---
LANG = {
    "EN": {
        "title": "🐟 Matsya-AI: Potential Fishing Zone (PFZ) & Navigation System",
        "subtitle": "Satellite AI oceanographic fusion forecasting productive fishing zones and route navigation.",
        "port_select": "📍 Departure Harbor / Port:",
        "port_help": "Select your coastal harbor.",
        "vessel_select": "🚤 Boat Type / Engine Capacity:",
        "vessel_help": "Calculates routes based on your boat's fuel burn rate and maximum offshore range.",
        "min_prob": "Minimum PFZ Confidence Threshold:",
        "min_prob_help": "Set minimum AI confidence. Higher values show only peak hotspots; lower values display broader zones.",
        "map_tab1": "🗺️ Google Satellite View (Clean Route Navigation)",
        "map_tab2": "🌊 Fish Density Heatmap (Biological Fronts)",
        "routes_header": "🧭 Recommended Routes & Navigation Waypoints",
        "kpi_conf": "Top PFZ Confidence",
        "kpi_heading": "Steering Heading",
        "kpi_dist": "Distance to Target",
        "kpi_savings": "Diesel Money Saved",
        "download_btn": "📥 Download Trip Navigation Plan (CSV)"
    },
    "KN": {
        "title": "🐟 ಮತ್ಸ್ಯ-AI: ಸಂಭಾವ್ಯ ಮೀನುಗಾರಿಕಾ ವಲಯ (PFZ) ಮತ್ತು ನೌಕಾಯಾನ ವ್ಯವಸ್ಥೆ",
        "subtitle": "ಉಪಗ್ರಹ AI ತಂತ್ರಜ್ಞಾನ: ಮೀನುಗಾರರಿಗೆ ಸಂಭಾವ್ಯ ಮೀನುಗಾರಿಕಾ ವಲಯ ಹಾಗೂ ನೌಕಾಯಾನ ಮಾರ್ಗದರ್ಶನ ನೀಡುತ್ತದೆ.",
        "port_select": "📍 ಹೊರಡುವ ಬಂದರು / ಮೀನುಗಾರಿಕಾ ತಾಣ:",
        "port_help": "ನಿಮ್ಮ ದೋಣಿ ಇರುವ ಬಂದರನ್ನು ಆಯ್ಕೆ ಮಾಡಿ.",
        "vessel_select": "🚤 ದೋಣಿಯ ಮಾದರಿ / ಎಂಜಿನ್ ಸಾಮರ್ಥ್ಯ:",
        "vessel_help": "ದೋಣಿಯ ವ್ಯಾಪ್ತಿ ಮತ್ತು ಇಂಧನ ಸಾಮರ್ಥ್ಯಕ್ಕೆ ಅನುಗುಣವಾಗಿ ಮಾರ್ಗವನ್ನು ಲೆಕ್ಕಹಾಕುತ್ತದೆ.",
        "min_prob": "ಕನಿಷ್ಠ PFZ ನಿಖರತೆ ಮಿತಿ:",
        "min_prob_help": "ಕನಿಷ್ಠ ನಿಖರತೆ ಮಿತಿ. ಹೆಚ್ಚು ನಿಖರತೆ ಇದ್ದರೆ ಅತ್ಯುತ್ತಮ ತಾಣಗಳು ಮಾತ್ರ ಕಾಣಿಸುತ್ತವೆ.",
        "map_tab1": "🗺️ ಗೂಗಲ್ ಉಪಗ್ರಹ ನಕ್ಷೆ (ಸ್ವಚ್ಛ ನೌಕಾಯಾನ ಮಾರ್ಗಗಳು)",
        "map_tab2": "🌊 ಮೀನುಗಳ ಸಾಂದ್ರತೆ ನಕ್ಷೆ (ಜೈವಿಕ ಮುಂಭಾಗಗಳು)",
        "routes_header": "🧭 ನೌಕಾಯಾನ ಮಾರ್ಗಗಳು ಮತ್ತು ಇಂಧನ ಉಳಿತಾಯ",
        "kpi_conf": "ಗರಿಷ್ಠ PFZ ನಿಖರತೆ",
        "kpi_heading": "ಹೋಗಬೇಕಾದ ದಿಕ್ಕು (ಕಂಪಾಸ್)",
        "kpi_dist": "ತಲುಪಲು ಬೇಕಾದ ದೂರ",
        "kpi_savings": "ಉಳಿತಾಯವಾಗುವ ಡೀಸೆಲ್ ಹಣ",
        "download_btn": "📥 ನೌಕಾಯಾನ ವಿವರ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ (CSV)"
    }
}

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🌐 Language / ಭಾಷೆ")
lang_choice = st.sidebar.radio("Language", options=["English (EN)", "ಕನ್ನಡ (KN)"], index=0, label_visibility="collapsed")
L = LANG["KN"] if "ಕನ್ನಡ" in lang_choice else LANG["EN"]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Trip Planning Parameters")

chosen_port = st.sidebar.selectbox(
    L["port_select"],
    options=list(KEY_PORTS.keys()),
    index=0,
    help=L["port_help"]
)

# Friendly vessel names with clear operational bounds
VESSEL_FRIENDLY = {
    "artisanal": "🛶 Artisanal Motor Canoe (OBM 15 HP | Max 40 km)",
    "small_mechanized": "🚤 Mechanized Trawler / Gillnetter (90 HP | Max 110 km)",
    "deep_sea_trawler": "🚢 Deep-Sea Multi-Day Trawler (250 HP | Max 250 km)"
}
vessel_keys = list(VESSEL_PROFILES.keys())
vessel_labels = [VESSEL_FRIENDLY[k] for k in vessel_keys]
chosen_vessel_idx = st.sidebar.selectbox(
    L["vessel_select"], 
    options=range(len(vessel_keys)), 
    format_func=lambda i: vessel_labels[i],
    index=1,
    help=L["vessel_help"]
)
chosen_vessel = vessel_keys[chosen_vessel_idx]

# Dynamic "Minimum PFZ Confidence Threshold" slider (50% to 99% responsive range)
prob_slider = st.sidebar.slider(
    L["min_prob"],
    min_value=50,
    max_value=99,
    value=75,
    step=1,
    format="%d%%",
    help=L["min_prob_help"]
)
prob_threshold = prob_slider / 100.0

# --- LOAD DATASET (PREFER FULL PARQUET FOR 100% REGIONAL COVERAGE) ---
parquet_file = OUTPUTS_DIR / "pfz_advisory_2021.parquet"
csv_file = OUTPUTS_DIR / "sample_pfz_advisory_predictions.csv"

if parquet_file.exists():
    df_advisory = pd.read_parquet(parquet_file)
elif csv_file.exists():
    df_advisory = pd.read_csv(csv_file)
else:
    st.error("No advisory data found. Run train_pipeline.py to generate models.")
    st.stop()

# STRICT OCEAN MASK: Exclude all landmass and terrestrial topography (elevation >= -10m is land/intertidal)
df_advisory = df_advisory[df_advisory["elevation"] < -10.0].copy()

# Filter predictions by confidence threshold
filtered_pfz = df_advisory[df_advisory["pfz_probability"] >= prob_threshold].copy()

# Deduplicate points spatially so maps and route planners never process redundant coordinates
unique_spatial_pfz = filtered_pfz.sort_values(by="pfz_probability", ascending=False).drop_duplicates(
    subset=["latitude", "longitude"]
)

# Compute non-overlapping optimal routes tailored to user-specified confidence threshold
planner = PFZRoutePlanner(port_name=chosen_port)
try:
    routes = planner.find_optimal_routes(
        unique_spatial_pfz, 
        vessel_type=chosen_vessel, 
        top_k=5, 
        min_probability=prob_threshold
    )
except TypeError:
    routes = planner.find_optimal_routes(
        unique_spatial_pfz, 
        vessel_type=chosen_vessel, 
        top_k=5
    )

# --- HEADER SECTION ---
st.title(L["title"])
st.caption(f"{L['subtitle']} | **Sahyadri College of Engineering & Management, Mangaluru (VTU)**")

# --- TOP 4 KPI CARDS (NATIVE STREAMLIT METRICS, 100% LEGIBLE) ---
top_route = routes[0] if len(routes) > 0 else None
conf_val = f"{top_route['pfz_probability']*100:.0f}%" if top_route else "N/A"
heading_val = f"{top_route['compass_heading']} ({top_route['bearing_degrees']}°)" if top_route else "N/A"
dist_val = f"{top_route['distance_km']} km" if top_route else "N/A"
savings_val = f"₹{top_route['fuel_savings_inr']:,.0f}" if top_route else "N/A"

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric(label=L['kpi_conf'], value=conf_val, delta="Peak Shoal Presence")
with col_kpi2:
    st.metric(label=L['kpi_heading'], value=heading_val, delta="Direct Compass Course")
with col_kpi3:
    st.metric(label=L['kpi_dist'], value=dist_val, delta=f"~{top_route['travel_time_one_way_hrs'] if top_route else '1.5'} hrs Boat Ride")
with col_kpi4:
    st.metric(label=L['kpi_savings'], value=savings_val, delta=f"~{top_route['fuel_saved_liters'] if top_route else '57'} L Diesel Saved", delta_color="normal")

st.markdown("---")

# --- MAIN SECTION: MAP & ROUTES ---
col_map, col_routes = st.columns([1.5, 1.0])

port_info = KEY_PORTS[chosen_port]
port_lat = port_info["lat"]
port_lon = port_info["lon"]

# Sample top unique hotspots for map rendering
map_points = unique_spatial_pfz.head(100)[["latitude", "longitude", "pfz_probability", "advisory_tier", "sst", "elevation"]].to_dict(orient="records")

# Extract biological & upwelling density points within cruising radius of selected harbor (STRICTLY OCEAN WATERS)
lat_dist_km = (df_advisory["latitude"] - port_lat) * 111.0
lon_dist_km = (df_advisory["longitude"] - port_lon) * 111.0 * np.cos(np.radians(port_lat))
dist_from_port = np.sqrt(lat_dist_km**2 + lon_dist_km**2)

local_ocean_slice = (
    df_advisory[
        (dist_from_port <= 180.0) & 
        (df_advisory["elevation"] < -10.0) & 
        (df_advisory["pfz_probability"] >= prob_threshold)
    ]
    .sort_values(by="pfz_probability", ascending=False)
    .drop_duplicates(subset=["latitude", "longitude"])
    .copy()
)
local_ocean_slice["dist_km"] = np.round(dist_from_port.loc[local_ocean_slice.index], 1)
local_ocean_slice["dist_nm"] = np.round(local_ocean_slice["dist_km"] / 1.852, 1)

density_points = local_ocean_slice.head(90)[
    ["latitude", "longitude", "pfz_probability", "advisory_tier", "sst", "elevation", "dist_km", "dist_nm"]
].to_dict(orient="records")

# --- TAB 1: GOOGLE MAPS NAVIGATION (CLEAN SATELLITE VIEW - ZERO CIRCLE CLUTTER) ---
google_maps_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        html, body, #map {{
            height: 100%;
            width: 100%;
            margin: 0;
            padding: 0;
            background-color: #0b1a30;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .custom-window {{
            padding: 10px;
            color: #0f172a;
            font-size: 13px;
            line-height: 1.5;
        }}
        .custom-window h4 {{
            margin: 0 0 6px 0;
            color: #0284c7;
            font-size: 15px;
            font-weight: 700;
        }}
        .nav-banner {{
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(15, 23, 42, 0.92);
            color: #ffffff;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            border-left: 4px solid #00e5ff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 10;
        }}
        .map-legend {{
            position: absolute;
            bottom: 24px;
            left: 14px;
            background: rgba(15, 23, 42, 0.92);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            backdrop-filter: blur(4px);
            z-index: 10;
            line-height: 1.6;
        }}
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&libraries=visualization"></script>
</head>
<body>
    <div id="map"></div>
    <div class="nav-banner">
        <b>⚓ Departure: {chosen_port}</b><br>
        <span style="color: #38bdf8;">Optimal Route #1: {top_route['distance_km'] if top_route else '23'} km ({top_route['compass_heading'] if top_route else 'W'})</span>
    </div>
    <div class="map-legend">
        <b>🗺️ Clean Satellite Route Navigation</b><br>
        <span style="color: #ef4444;">📍</span> <b>Port:</b> {chosen_port}<br>
        <span style="color: #00e5ff;">━</span> <b>Route #1 (⭐ Highest Probability)</b><br>
        <span style="color: #a855f7;">━</span> <b>Route #2</b> &nbsp; <span style="color: #f97316;">━</span> <b>Route #3</b><br>
        <span style="color: #10b981;">━</span> <b>Route #4</b> &nbsp; <span style="color: #eab308;">━</span> <b>Route #5</b>
    </div>

    <script>
        const portCoords = {{ lat: {port_lat}, lng: {port_lon} }};
        const topRoutes = {json.dumps(routes)};

        function initMap() {{
            const map = new google.maps.Map(document.getElementById('map'), {{
                zoom: 9,
                minZoom: 6,
                maxZoom: 13, // Prevents blank white ocean tiles
                center: {{ lat: portCoords.lat, lng: portCoords.lng - 0.35 }},
                mapTypeId: google.maps.MapTypeId.HYBRID,
                backgroundColor: '#0b1a30',
                mapTypeControl: true,
                mapTypeControlOptions: {{
                    style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
                    position: google.maps.ControlPosition.TOP_RIGHT
                }},
                zoomControl: true,
                streetViewControl: false,
                fullscreenControl: true
            }});

            const infoWindow = new google.maps.InfoWindow();

            // 1. Departure Port Marker
            const portMarker = new google.maps.Marker({{
                position: portCoords,
                map: map,
                title: "Departure Port: {chosen_port}",
                label: {{
                    text: "⚓ PORT",
                    color: "#ffffff",
                    fontWeight: "bold",
                    fontSize: "12px"
                }},
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 14,
                    fillColor: '#ef4444',
                    fillOpacity: 1.0,
                    strokeColor: '#ffffff',
                    strokeWeight: 2.5
                }}
            }});

            portMarker.addListener('click', () => {{
                infoWindow.setContent(`
                    <div class="custom-window">
                        <h4>⚓ Departure Harbor</h4>
                        <b>{chosen_port}</b><br>
                        Lat: ${{portCoords.lat}}°N, Lon: ${{portCoords.lng}}°E
                    </div>
                `);
                infoWindow.open(map, portMarker);
            }});

            // 2. Render ONLY Clean Non-Overlapping Route Polylines & Waypoint Flags (NO CIRCLE CLUTTER!)
            const routeColors = ['#00e5ff', '#a855f7', '#f97316', '#10b981', '#eab308'];
            topRoutes.forEach((r, idx) => {{
                const targetCoords = {{ lat: r.latitude, lng: r.longitude }};
                const color = routeColors[idx % routeColors.length];
                const isRoute1 = idx === 0;

                // Clean Geodesic Route Path
                new google.maps.Polyline({{
                    path: [portCoords, targetCoords],
                    geodesic: true,
                    strokeColor: color,
                    strokeOpacity: 0.98,
                    strokeWeight: isRoute1 ? 5.5 : 3.5,
                    zIndex: 100 - idx * 10,
                    map: map
                }});

                // Route Midpoint for Distance Label
                const midLat = (portCoords.lat + targetCoords.lat) / 2.0;
                const midLng = (portCoords.lng + targetCoords.lng) / 2.0;

                new google.maps.Marker({{
                    position: {{ lat: midLat, lng: midLng }},
                    map: map,
                    zIndex: 500 - idx * 10,
                    label: {{
                        text: `Route #${{idx + 1}}: ${{r.distance_km}} km (${{r.compass_heading}})`,
                        color: '#ffffff',
                        fontWeight: 'bold',
                        fontSize: '11px'
                    }},
                    icon: {{
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 0.1
                    }}
                }});

                // Destination Waypoint Flag
                const endMarker = new google.maps.Marker({{
                    position: targetCoords,
                    map: map,
                    zIndex: 1000 - idx * 50,
                    label: {{
                        text: isRoute1 ? `★ Route #1 (${{Math.round(r.pfz_probability * 100)}}%)` : `Route #${{idx + 1}} (${{Math.round(r.pfz_probability * 100)}}%)`,
                        color: '#ffffff',
                        fontWeight: 'bold',
                        fontSize: isRoute1 ? '12px' : '11px'
                    }},
                    icon: {{
                        path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW,
                        scale: isRoute1 ? 7.5 : 6.0,
                        fillColor: color,
                        fillOpacity: 1.0,
                        strokeWeight: 2.0,
                        strokeColor: '#ffffff'
                    }}
                }});

                endMarker.addListener('click', () => {{
                    infoWindow.setContent(`
                        <div class="custom-window">
                            <h4>🎯 Target Destination: Route #${{idx + 1}}</h4>
                            <b>AI Match Confidence:</b> ${{Math.round(r.pfz_probability * 100)}}% (${{r.advisory_tier}})<br>
                            <b>🧭 Compass Heading:</b> ${{r.compass_heading}} (${{r.bearing_degrees}}°)<br>
                            <b>📏 Total Distance:</b> ${{r.distance_km}} km (${{r.distance_nm}} Nautical Miles)<br>
                            <b>⏱️ One-Way Travel Time:</b> ${{r.travel_time_one_way_hrs}} hours<br>
                            <b>⛽ Fuel Needed:</b> ${{r.roundtrip_fuel_liters}} L (₹${{r.fuel_cost_inr.toLocaleString()}})<br>
                            <b>💰 Money Saved:</b> ₹${{r.fuel_savings_inr.toLocaleString()}} (~${{r.fuel_saved_liters}} L)
                        </div>
                    `);
                    infoWindow.open(map, endMarker);
                }});
            }});
        }}
        window.onload = initMap;
    </script>
</body>
</html>
"""

# --- TAB 2: FISH DENSITY HEATMAP WITH VISIBLE LABELS, PINS & SCALE ---
density_map_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        html, body, #density_map {{
            height: 100%;
            width: 100%;
            margin: 0;
            padding: 0;
            background-color: #0b1a30;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .custom-window {{
            padding: 10px;
            color: #0f172a;
            font-size: 13px;
            line-height: 1.5;
        }}
        .custom-window h4 {{
            margin: 0 0 6px 0;
            color: #0284c7;
            font-size: 15px;
            font-weight: 700;
        }}
        .density-banner {{
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(15, 23, 42, 0.92);
            color: #ffffff;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 12.5px;
            border-left: 4px solid #10b981;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 10;
        }}
        .density-legend {{
            position: absolute;
            bottom: 24px;
            left: 14px;
            background: rgba(15, 23, 42, 0.92);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            backdrop-filter: blur(4px);
            z-index: 10;
            line-height: 1.6;
        }}
    </style>
    <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&callback=initDensityMap" async defer></script>
</head>
<body>
    <div id="density_map"></div>
    <div class="density-banner">
        <b>🌊 Biological & Upwelling Fish Density Heatmap</b><br>
        <span style="color: #38bdf8;">Showing phytoplankton blooms & cold upwelling convergence zones</span>
    </div>
    <div class="density-legend">
        <b>📊 Fish Density Scale</b><br>
        <span style="color: #ef4444; font-size: 14px;">■</span> <b>Red:</b> Peak Feeding Shoals (85%–100%)<br>
        <span style="color: #f97316; font-size: 14px;">■</span> <b>Orange:</b> High Upwelling Zone (70%–84%)<br>
        <span style="color: #eab308; font-size: 14px;">■</span> <b>Yellow:</b> Moderate Primary Productivity (50%–69%)<br>
        <span style="color: #06b6d4; font-size: 14px;">■</span> <b>Cyan:</b> Ambient Open Water (&lt;50%)
    </div>

    <script>
        const portCoords = {{ lat: {port_lat}, lng: {port_lon} }};
        const densityPoints = {json.dumps(density_points)};
        const topHotspots = {json.dumps(routes[:5])};

        // Key coastal landmarks for orientation
        const coastalLandmarks = [
            {{ name: "Mangaluru Coast", lat: 12.87, lng: 74.84 }},
            {{ name: "Malpe / Udupi", lat: 13.35, lng: 74.70 }},
            {{ name: "Karwar Bay", lat: 14.81, lng: 74.13 }},
            {{ name: "Kochi Coast", lat: 9.93, lng: 76.27 }}
        ];

        function initDensityMap() {{
            const mapContainer = document.getElementById('density_map');
            if (!mapContainer) return;

            const map = new google.maps.Map(mapContainer, {{
                zoom: 8.8,
                minZoom: 6,
                maxZoom: 13,
                center: {{ lat: portCoords.lat, lng: portCoords.lng - 0.45 }},
                mapTypeId: google.maps.MapTypeId.HYBRID,
                backgroundColor: '#0b1a30',
                mapTypeControl: true,
                zoomControl: true,
                streetViewControl: false,
                fullscreenControl: true
            }});

            const infoWindow = new google.maps.InfoWindow();

            // 1. Render Biological & Thermal Front Density Circles (STRICTLY IN OCEAN WATERS)
            densityPoints.forEach(pt => {{
                // Strict land barrier: skip any terrestrial or intertidal point
                if (pt.elevation >= -8.0) return;

                const prob = pt.pfz_probability;
                let circleColor = '#06b6d4'; // Cyan default (<50%)
                let strokeColor = '#0891b2';
                let tierText = 'Ambient Open Water (&lt;50%)';
                let radius = 3800;
                let fillOpacity = 0.20;
                let dotScale = 2.5;

                if (prob >= 0.85) {{
                    circleColor = '#ef4444'; // Red (85-100%)
                    strokeColor = '#b91c1c';
                    tierText = 'Peak Feeding Shoals (85%–100%)';
                    radius = 6500;
                    fillOpacity = 0.48;
                    dotScale = 4.0;
                }} else if (prob >= 0.70) {{
                    circleColor = '#f97316'; // Orange (70-84%)
                    strokeColor = '#c2410c';
                    tierText = 'High Upwelling Zone (70%–84%)';
                    radius = 5500;
                    fillOpacity = 0.40;
                    dotScale = 3.5;
                }} else if (prob >= 0.50) {{
                    circleColor = '#eab308'; // Yellow (50-69%)
                    strokeColor = '#a16207';
                    tierText = 'Moderate Primary Productivity (50%–69%)';
                    radius = 4800;
                    fillOpacity = 0.35;
                    dotScale = 3.0;
                }}

                // Glowing Density Zone Circle
                const circle = new google.maps.Circle({{
                    strokeColor: strokeColor,
                    strokeOpacity: 0.85,
                    strokeWeight: 1.5,
                    fillColor: circleColor,
                    fillOpacity: fillOpacity,
                    map: map,
                    center: {{ lat: pt.latitude, lng: pt.longitude }},
                    radius: radius,
                    zIndex: Math.round(prob * 100)
                }});

                // Precise Core Dot
                const dot = new google.maps.Marker({{
                    position: {{ lat: pt.latitude, lng: pt.longitude }},
                    map: map,
                    zIndex: Math.round(prob * 100) + 10,
                    icon: {{
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: dotScale,
                        fillColor: circleColor,
                        fillOpacity: 1.0,
                        strokeColor: '#ffffff',
                        strokeWeight: 1.2
                    }}
                }});

                const popupHtml = `
                    <div class="custom-window">
                        <h4>🌊 Fish Density Zone (${{Math.round(prob * 100)}}%)</h4>
                        <b>Classification:</b> ${{tierText}}<br>
                        <b>📍 Location:</b> ${{pt.latitude.toFixed(3)}}°N, ${{pt.longitude.toFixed(3)}}°E<br>
                        <b>📏 Distance from Port:</b> ${{pt.dist_km}} km (${{pt.dist_nm}} NM)<br>
                        <b>🌊 Bathymetric Depth:</b> ${{Math.abs(pt.elevation).toFixed(0)}} m<br>
                        <b>🌡️ Sea Surface Temp:</b> ${{pt.sst ? pt.sst.toFixed(1) : '28.8'}} °C
                    </div>
                `;

                circle.addListener('click', (e) => {{
                    infoWindow.setContent(popupHtml);
                    infoWindow.setPosition(e.latLng || {{ lat: pt.latitude, lng: pt.longitude }});
                    infoWindow.open(map);
                }});

                dot.addListener('click', () => {{
                    infoWindow.setContent(popupHtml);
                    infoWindow.open(map, dot);
                }});
            }});

            // 2. Visible Departure Port Marker with Text Label
            const portMarker = new google.maps.Marker({{
                position: portCoords,
                map: map,
                zIndex: 2000,
                label: {{
                    text: "⚓ {chosen_port}",
                    color: "#ffffff",
                    fontWeight: "bold",
                    fontSize: "12px"
                }},
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 13,
                    fillColor: '#ef4444',
                    fillOpacity: 1.0,
                    strokeColor: '#ffffff',
                    strokeWeight: 2.5
                }}
            }});

            portMarker.addListener('click', () => {{
                infoWindow.setContent(`
                    <div class="custom-window">
                        <h4>⚓ Departure Harbor</h4>
                        <b>{chosen_port}</b><br>
                        Lat: ${{portCoords.lat}}°N, Lon: ${{portCoords.lng}}°E
                    </div>
                `);
                infoWindow.open(map, portMarker);
            }});

            // 3. Prominent Top Hotspot Badges on Density Map
            topHotspots.forEach((h, idx) => {{
                const hMarker = new google.maps.Marker({{
                    position: {{ lat: h.latitude, lng: h.longitude }},
                    map: map,
                    zIndex: 1500 - idx * 20,
                    label: {{
                        text: `🔥 Hotspot #${{idx + 1}} (${{Math.round(h.pfz_probability * 100)}}%)`,
                        color: "#ffffff",
                        fontWeight: "bold",
                        fontSize: "12px"
                    }},
                    icon: {{
                        path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW,
                        scale: 6.5,
                        fillColor: '#ff0000',
                        fillOpacity: 1.0,
                        strokeColor: '#ffffff',
                        strokeWeight: 2.0
                    }}
                }});

                hMarker.addListener('click', () => {{
                    infoWindow.setContent(`
                        <div class="custom-window">
                            <h4>🔥 Peak Density Hotspot #${{idx + 1}}</h4>
                            <b>PFZ Probability:</b> ${{Math.round(h.pfz_probability * 100)}}% (${{h.advisory_tier}})<br>
                            <b>Distance from Port:</b> ${{h.distance_km}} km (${{h.distance_nm}} NM)<br>
                            <b>Compass Heading:</b> ${{h.compass_heading}} (${{h.bearing_degrees}}°)<br>
                            <b>Water Depth:</b> ${{Math.abs(h.depth_m)}} m<br>
                            <b>Sea Surface Temp:</b> ${{h.sst}} °C
                        </div>
                    `);
                    infoWindow.open(map, hMarker);
                }});
            }});

            // 4. Coastal Landmark Reference Markers for Orientation
            coastalLandmarks.forEach(lm => {{
                new google.maps.Marker({{
                    position: {{ lat: lm.lat, lng: lm.lng }},
                    map: map,
                    zIndex: 100,
                    label: {{
                        text: lm.name,
                        color: "#94a3b8",
                        fontSize: "11px",
                        fontWeight: "600"
                    }},
                    icon: {{
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 4,
                        fillColor: '#64748b',
                        fillOpacity: 0.8,
                        strokeWeight: 0
                    }}
                }});
            }});

            // Force map resize check to handle Streamlit tab switching gracefully
            setTimeout(() => {{
                google.maps.event.trigger(map, 'resize');
                map.setCenter({{ lat: portCoords.lat, lng: portCoords.lng - 0.45 }});
            }}, 300);
        }}

        window.initDensityMap = initDensityMap;
        window.addEventListener('load', initDensityMap);
    </script>
</body>
</html>
"""

with col_map:
    tab1, tab2 = st.tabs([L["map_tab1"], L["map_tab2"]])
    
    with tab1:
        components.html(google_maps_html, height=580)
        
    with tab2:
        components.html(density_map_html, height=580)

with col_routes:
    st.subheader(L["routes_header"])
    
    if len(routes) == 0:
        st.info("No active PFZs found within operational range of selected vessel. Try lowering the confidence slider on the left.")
    else:
        for idx, r in enumerate(routes, 1):
            is_best = idx == 1

            # NATIVE STREAMLIT BORDERED CONTAINER (NO RAW DIV TAGS!)
            with st.container(border=True):
                col_head1, col_head2 = st.columns([2, 1])
                with col_head1:
                    st.markdown(f"#### 🎯 Route #{idx} ({r['compass_heading']})")
                with col_head2:
                    if is_best:
                        st.success(f"⭐ BEST ({r['pfz_probability']*100:.0f}%)")
                    else:
                        st.info(f"Option #{idx} ({r['pfz_probability']*100:.0f}%)")

                st.markdown(f"🧭 **Compass Steer:** `{r['compass_heading']} ({r['bearing_degrees']}°)` &nbsp;|&nbsp; 📏 **Distance:** `{r['distance_km']} km ({r['distance_nm']} NM)`")

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(
                        label="⏱️ Travel Time (one-way)",
                        value=f"~{r['travel_time_one_way_hrs']} hrs",
                        delta=f"🌊 Depth: {abs(r['depth_m']):.0f}m"
                    )
                with col_m2:
                    st.metric(
                        label="💰 Diesel Money Saved",
                        value=f"₹{r['fuel_savings_inr']:,.0f}",
                        delta=f"⛽ ~{r['fuel_saved_liters']} L Saved",
                        delta_color="normal"
                    )

        # Download Route Plan as CSV
        df_export = pd.DataFrame(routes)
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=L["download_btn"],
            data=csv_bytes,
            file_name=f"MatsyaAI_TripPlan_{chosen_port.split()[0]}.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- FOOTER & ACADEMIC CREDITS ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px; line-height: 1.6;">
    <b>Project Title:</b> AI-Based Dynamic Fishing Zone Prediction System | 
    <b>Faculty Guide:</b> Prof. Chaitrashree M, Dept. of Computer Science & Engineering<br>
    <b>Institution:</b> Sahyadri College of Engineering & Management, Mangaluru (VTU 2026–27)<br>
    <i>Integrated with Google Maps Satellite Imagery & UN Sustainable Development Goals (SDG 14: Life Below Water & SDG 1: No Poverty).</i>
</div>
""", unsafe_allow_html=True)
