import math
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from openpyxl import load_workbook
from streamlit_folium import st_folium

# Optional Fiona support for KML
try:
    import fiona
    HAS_FIONA = True
except Exception:
    HAS_FIONA = False

# -------------------------
# Constants & Helpers
# -------------------------
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT',
    'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
}

st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Unified Clustering + Batch Navigation")

# Sidebar tips
with st.sidebar:
    st.markdown("### Tips")
    st.markdown(
        "- CSV must include LATITUDE/LONGITUDE.\n"
        "- Clustering radius is in **meters**.\n"
        "- If KML ward reading fails, convert to GeoJSON.\n"
        "- Select start location by clicking on the map."
    )

# -------------------------
# Utility functions
# -------------------------
def normalize_subcategory(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(float(lat1)); phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None) -> str:
    base = "https://www.google.com/maps/dir/"
    origin = f"{origin_lat},{origin_lon}"
    destination = f"{dest_lat},{dest_lon}"
    waypoint_str = ""
    if waypoints:
        waypoint_str = "/".join([f"{lat},{lon}" for lat, lon in waypoints])
    full_path = "/".join(filter(None, [origin, waypoint_str, destination]))
    return f"{base}{full_path}"

def hyperlinkify_excel(excel_path: str, sheet_name: str = "Clustering Application Summary") -> None:
    try:
        wb = load_workbook(excel_path)
        ws = wb[sheet_name]
        for row in range(2, ws.max_row + 1):
            for col_idx in (11, 12):  # BEFORE, AFTER
                link = ws.cell(row, col_idx).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col_idx).hyperlink = link
                    ws.cell(row, col_idx).style = "Hyperlink"
        wb.save(excel_path)
    except Exception:
        pass

def load_wards_uploaded(file) -> gpd.GeoDataFrame | None:
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read()); tmp_path = tmp.name
        if suffix in ("geojson", "json"):
            wards = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                wards = gpd.read_file(tmp_path, driver="KML")
            else:
                st.error("Install Fiona or convert KML to GeoJSON.")
                return None
        else:
            st.error("Unsupported ward file type.")
            return None
        if wards.crs is None:
            wards.set_crs(epsg=4326, inplace=True)
        elif wards.crs.to_string() != "EPSG:4326":
            wards = wards.to_crs(epsg=4326)
        return wards[wards.geometry.notna() & ~wards.geometry.is_empty]
    except Exception:
        return None

# -------------------------
# Session State
# -------------------------
if "visited_ticket_ids" not in st.session_state:
    st.session_state.visited_ticket_ids = set()
if "skipped_ticket_ids" not in st.session_state:
    st.session_state.skipped_ticket_ids = set()
if "current_target_id" not in st.session_state:
    st.session_state.current_target_id = None
if "batch_target_ids" not in st.session_state:
    st.session_state.batch_target_ids = set()
if "map_center" not in st.session_state:
    st.session_state.map_center = [28.6139, 77.2090]  # Default: Delhi
if "origin_coords" not in st.session_state:
    st.session_state.origin_coords = None

# -------------------------
# UI — Inputs
# -------------------------
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV file", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundaries (optional)", type=["geojson", "json", "kml"])

subcategory_options = [
    "Pothole","Sand piled on roadsides + Mud/slit on roadside","Greening of Central Verges","Unsurfaced Parking Lots"
]
st.subheader("Step 2: Select Issue Subcategory")
subcategory_option = st.selectbox("Choose subcategory:", subcategory_options)

st.subheader("Step 3: Clustering Parameters")
radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15)
min_samples = st.number_input("Minimum per cluster", 1, 100, 2)

# -------------------------
# Core logic
# -------------------------
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        missing = sorted(list(REQUIRED_COLS - set(df.columns)))
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        st.session_state.map_center = [df['LATITUDE'].mean(), df['LONGITUDE'].mean()]
        df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
        df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()].copy()
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df = df.dropna(subset=['LATITUDE','LONGITUDE'])
        if df.empty:
            st.warning("No valid rows after filtering.")
            st.stop()

        coords_rad = np.radians(df[['LATITUDE','LONGITUDE']].to_numpy())
        eps_rad = float(radius_m) / EARTH_RADIUS_M
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
        df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
        df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

        gdf_all = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
        wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
        if wards_gdf is not None:
            try:
                gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
            except Exception:
                st.warning("Ward join failed.")

        clustered = gdf_all[gdf_all['IS_CLUSTERED']].copy()
        summary_sheet = clustered.copy()

        excel_filename = "Clustering_Application_Summary.xlsx"
        with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
            summary_sheet.to_excel(writer, index=False, sheet_name="Clustering Application Summary")
        hyperlinkify_excel(excel_filename)

        # -------------------------
        # Step 4A — Navigate
        # -------------------------
        st.subheader("Step 4A: Navigate to Nearest Tickets")
        m = folium.Map(location=st.session_state.map_center, zoom_start=12)
        if st.session_state.origin_coords:
            folium.Marker(
                [st.session_state.origin_coords['lat'], st.session_state.origin_coords['lng']],
                popup="Start", icon=folium.Icon(color="green")
            ).add_to(m)
        map_data = st_folium(m, width=725, height=400)
        if map_data and map_data.get("last_clicked"):
            st.session_state.origin_coords = map_data["last_clicked"]

        # -------------------------
        # Step 5 — Map
        # -------------------------
        st.subheader("Step 5: Map Display")
        display_map = folium.Map(location=st.session_state.map_center, zoom_start=13)
        ticket_markers = folium.FeatureGroup(name="Tickets")
        target_id = st.session_state.get("current_target_id")
        batch_ids = st.session_state.get("batch_target_ids", set())

        for _, row in gdf_all.iterrows():
            rid = str(row["ISSUE ID"])
            is_first = (rid == str(target_id)) if target_id else False
            in_batch = rid in batch_ids
            color = "green" if is_first else ("orange" if in_batch else "red")
            folium.CircleMarker(
                [float(row["LATITUDE"]), float(row["LONGITUDE"])],
                radius=7, color=color, fill=True, fill_color=color, fill_opacity=0.9,
                popup=(f"Cluster {row['CLUSTER NUMBER']}<br>"
                       f"Issue ID: {rid}<br>"
                       f"Ward: {row['WARD']}")
            ).add_to(ticket_markers)

        ticket_markers.add_to(display_map)
        folium.LayerControl().add_to(display_map)
        st_folium(display_map, use_container_width=True)

        # Save map + download
        html_filename = "Clustering_Application_Map.html"
        display_map.save(html_filename)

        st.subheader("Step 6: Downloads")
        with open(excel_filename, "rb") as f:
            st.download_button("⬇️ Download Excel", f, file_name=excel_filename)
        with open(html_filename, "rb") as f:
            st.download_button("⬇️ Download Map", f, file_name=html_filename)

        st.success("✅ Processing complete.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV to start.")
