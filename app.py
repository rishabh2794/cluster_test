import math
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from sklearn.cluster import DBSCAN

import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from openpyxl import load_workbook

# Auto geolocation (browser)
try:
    from streamlit_geolocation import st_geolocation
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False

# Optional: improve KML handling if available
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

st.set_page_config(layout="wide")
st.title("Unified Clustering + Batch Navigation — Hotspots + Route to Next N Tickets")

with st.sidebar:
    st.markdown("### Tips")
    st.markdown(
        "- CSV must include LATITUDE/LONGITUDE in decimal degrees.\n"
        "- Clustering radius is in **meters** (converted to **radians** for haversine).\n"
        "- If KML ward reading fails, convert to GeoJSON and try again.\n"
        "- Auto location requires allowing the browser's location permission."
    )

subcategory_options = [
    "Pothole",
    "Sand piled on roadsides + Mud/slit on roadside",
    "Garbage dumped on public land",
    "Unpaved Road",
    "Broken Footpath / Divider",
    "Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards",
    "Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.",
    "Overflowing Dustbins",
    "Barren land to be greened",
    "Greening of Central Verges",
    "Unsurfaced Parking Lots"
]

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
    # The Google Maps URL structure for waypoints is /origin/waypoint1/waypoint2/.../destination
    if waypoints:
        wp_str = "/".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        dest_str = f"{wp_str}/{dest_lat},{dest_lon}"
    else:
        dest_str = f"{dest_lat},{dest_lon}"
    
    # Travel mode mapping for URL
    travel_modes = {"driving": "data=!4m2!4m1!3e0", "walking": "data=!4m2!4m1!3e2", "two_wheeler": "data=!4m2!4m1!3e0"}
    travel = travel_modes.get(mode, "data=!4m2!4m1!3e0") # Default to driving

    return f"{base}{origin}/{dest_str}/{travel}"


def hyperlinkify_excel(excel_path: str, sheet_name: str = "Clustering Application Summary") -> None:
    try:
        wb = load_workbook(excel_path)
        ws = wb[sheet_name]
        for row in range(2, ws.max_row + 1):
            for col_idx in (11, 12):  # BEFORE (K), AFTER (L)
                link = ws.cell(row, col_idx).value
                if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col_idx).hyperlink = link
                    ws.cell(row, col_idx).style = "Hyperlink"
        wb.save(excel_path)
    except Exception as e:
        st.warning(f"Excel hyperlinking skipped: {e}")

def load_wards_uploaded(file) -> gpd.GeoDataFrame | None:
    """Read GeoJSON/JSON/KML wards into EPSG:4326. Return None on failure and show an error."""
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix in ("geojson", "json"):
            wards = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
                wards = gpd.read_file(tmp_path, driver='KML')
            else:
                st.error(
                    "Reading KML requires the `fiona` package. Please install it (`pip install fiona`) or convert the KML to GeoJSON."
                )
                return None
        else:
            st.error("Unsupported ward file type. Please upload GeoJSON/JSON/KML.")
            return None

        if wards is None or wards.empty:
            st.error("Ward file loaded but contains no features.")
            return None

        if wards.crs is None:
            st.info("Ward file had no CRS. Assuming EPSG:4326 (WGS84).")
            wards.set_crs(epsg=4326, inplace=True)
        elif wards.crs.to_string() != "EPSG:4326":
            wards = wards.to_crs(epsg=4326)

        wards = wards[wards.geometry.notna()].copy()
        wards = wards[~wards.geometry.is_empty]
        return wards
    except Exception as e:
        st.error(f"Error reading ward file: {e}")
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

# -------------------------
# UI — Inputs
# -------------------------
st.subheader("Step 1: Upload Required Files")
csv_file = st.file_uploader("Upload CSV file with issues", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundary file (GeoJSON/JSON/KML, optional)", type=["geojson", "json", "kml"])

st.subheader("Step 2: Select Issue Subcategory")
subcategory_option = st.selectbox("Choose issue subcategory to analyze:", subcategory_options)

st.subheader("Step 3: Set Clustering Parameters")
radius_m = st.number_input("Clustering Radius (meters)", min_value=1, max_value=1000, value=15)
min_samples = st.number_input("Minimum Issues per Cluster", min_value=1, max_value=100, value=2)
if radius_m < 10 or min_samples < 2:
    st.warning("⚠️ Low values may lead to too many tiny clusters. Proceed with caution.")

# -------------------------
# Core logic
# -------------------------
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        missing = sorted(list(REQUIRED_COLS - set(df.columns)))
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Filter & clean
        df = df.copy()
        df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
        desired = subcategory_option.strip().lower()
        df = df[df['SUBCATEGORY_NORM'] == desired].copy()
        if df.empty:
            st.info("No rows found for the selected subcategory.")
            st.stop()

        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        if df.empty:
            st.info("All rows had invalid/missing coordinates after cleaning.")
            st.stop()

        # Ensure string-friendly columns
        for col in ['CREATED AT', 'STATUS', 'ADDRESS', 'BEFORE PHOTO', 'AFTER PHOTO', 'ISSUE ID', 'ZONE', 'WARD', 'CITY']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # --- DBSCAN (Haversine) ---
        coords_deg = df[['LATITUDE', 'LONGITUDE']].to_numpy()
        if coords_deg.shape[0] < max(1, int(min_samples)):
            st.info(f"Need at least {min_samples} points to form a cluster. Only {coords_deg.shape[0]} rows available.")
            st.stop()

        coords_rad = np.radians(coords_deg)
        eps_rad = float(radius_m) / EARTH_RADIUS_M
        db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
        labels = db.fit_predict(coords_rad)
        df['CLUSTER NUMBER'] = labels  # -1 = noise
        df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

        # --- Optional Ward Join ---
        wards_gdf = None
        if ward_file is not None:
            wards_gdf = load_wards_uploaded(ward_file)

        # GeoDataFrame of ALL points (not only clusters)
        gdf_all = gpd.GeoDataFrame(
            df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'].astype(float), df['LATITUDE'].astype(float)), crs="EPSG:4326"
        )
        if wards_gdf is not None and not wards_gdf.empty:
            try:
                gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
            except Exception as e:
                st.warning(f"Spatial join failed; proceeding without ward attribution. Error: {e}")

        # --- Clustered subset for Excel summary ONLY ---
        clustered = gdf_all[gdf_all['IS_CLUSTERED']].copy()
        if clustered.empty:
            st.warning("DBSCAN found no clusters (all points are noise). You can still use navigation and map with all points.")

        # Cluster sizes for non-noise
        if not clustered.empty:
            sizes = clustered.groupby('CLUSTER NUMBER')['ISSUE ID'].count().rename("NUMBER OF ISSUES")
            summary_sheet = clustered.drop(columns=['SUBCATEGORY_NORM']).copy()
            summary_sheet = summary_sheet.merge(sizes, on='CLUSTER NUMBER', how='left')
            summary_sheet = summary_sheet[[
                'CLUSTER NUMBER', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
                'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
                'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
            ]].sort_values(['CLUSTER NUMBER', 'CREATED AT'])
        else:
            summary_sheet = pd.DataFrame(columns=[
                'CLUSTER NUMBER', 'NUMBER OF ISSUES', 'ISSUE ID', 'ZONE', 'WARD',
                'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE',
                'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS'
            ])

        # --- Excel export (clustered points only) ---
        excel_filename = "Clustering_Application_Summary.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            summary_sheet.to_excel(writer, index=False, sheet_name='Clustering Application Summary')
        hyperlinkify_excel(excel_filename)

        # -------------------------
        # Step 4A — Navigate to Nearest Tickets (Batch up to 10)
        # -------------------------
        st.subheader("Step 4A: Navigate to Nearest Tickets (Batch)")

        unique_statuses = sorted(gdf_all['STATUS'].dropna().astype(str).unique().tolist())
        default_statuses = [s for s in unique_statuses if s.lower() in ("open", "pending", "in progress")]
        include_statuses = st.multiselect("Eligible ticket statuses", options=unique_statuses, default=default_statuses or unique_statuses)

        wards_in_data = sorted(gdf_all['WARD'].dropna().astype(str).unique().tolist())
        ward_filter = st.multiselect("Limit to ward(s) (optional)", options=wards_in_data, default=[])

        travel_mode = st.selectbox("Travel mode", ["driving", "walking", "two_wheeler"], index=0)
        batch_size = st.slider("Batch size (next N tickets)", min_value=1, max_value=10, value=10)

        # Get current location (auto-fetch with manual fallback if library is missing)
        origin_lat = origin_lon = None
        st.markdown("### Your Location")
        if HAVE_GEO:
            st.caption("Click the button below to fetch your current location automatically (requires browser permission).")
            # The st_geolocation() function renders a button to trigger the browser's location API
            loc = st_geolocation()
            if loc and isinstance(loc, dict) and loc.get("latitude") and loc.get("longitude"):
                origin_lat = float(loc["latitude"])
                origin_lon = float(loc["longitude"])
                st.success(f"✅ Location automatically fetched: {origin_lat:.6f}, {origin_lon:.6f}")
        else:
            st.warning(
                "Auto-geolocation is not available. Please install the required package:\n\n"
                "`pip install streamlit-geolocation`"
            )
            # Re-introduce manual input as the ONLY fallback if the library is missing
            st.info("Alternatively, you can provide your location manually:")
            manual_lat = st.text_input("Manual latitude", value="")
            manual_lon = st.text_input("Manual longitude", value="")
            if manual_lat.strip() and manual_lon.strip():
                try:
                    origin_lat = float(manual_lat)
                    origin_lon = float(manual_lon)
                    st.success(f"✅ Manual location set: {origin_lat:.6f}, {origin_lon:.6f}")
                except Exception:
                    st.error("Invalid manual coordinates. Example: 26.8467 (lat), 80.9462 (lon)")


        # Build pool from ALL filtered points
        pool = gdf_all.copy()
        if include_statuses:
            pool = pool[pool['STATUS'].astype(str).isin(include_statuses)]
        if ward_filter:
            pool = pool[pool['WARD'].astype(str).isin([str(w) for w in ward_filter])]
        if st.session_state.visited_ticket_ids:
            pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
        if st.session_state.skipped_ticket_ids:
            pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

        st.write(f"Eligible tickets remaining: **{len(pool)}**")

        # Greedy nearest-neighbor batch
        sequence_rows = []
        if origin_lat is not None and origin_lon is not None and not pool.empty:
            cur_lat, cur_lon = origin_lat, origin_lon
            pool2 = pool.copy()
            for _ in range(min(batch_size, len(pool2))):
                pool2['__dist_m'] = pool2.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
                nxt = pool2.sort_values('__dist_m', ascending=True).iloc[0]
                sequence_rows.append(nxt)
                cur_lat, cur_lon = float(nxt['LATITUDE']), float(nxt['LONGITUDE'])
                pool2 = pool2[pool2['ISSUE ID'] != nxt['ISSUE ID']]

        if sequence_rows:
            # Build navigation URL with waypoints
            waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for r in sequence_rows]
            first_stop, last_stop = waypoints[0], waypoints[-1]
            
            # The destination for the URL is the last point. All points before it are waypoints.
            mid_waypoints = waypoints[:-1] if len(waypoints) > 1 else []
            nav_url = google_maps_url(origin_lat, origin_lon, last_stop[0], last_stop[1], mode=travel_mode, waypoints=mid_waypoints)


            # Approx total distance & list
            total_m = 0.0
            prev = (origin_lat, origin_lon)
            leg_dists = []
            for (lat, lon) in waypoints:
                d = haversine_m(prev[0], prev[1], lat, lon)
                leg_dists.append(int(d))
                total_m += d
                prev = (lat, lon)
            eta_min = total_m / (30 * 1000 / 60)  # Approx ETA based on 30 km/h average speed

            st.success(
                f"Batch route ready: **{len(sequence_rows)}** tickets | Total distance ≈ **{int(total_m)} m** | ETA ≈ **{eta_min:.1f} min**"
            )
            st.markdown(f"[🧭 Open continuous navigation in Google Maps]({nav_url})")

            # Show ordered list
            list_df = pd.DataFrame({
                "#": list(range(1, len(sequence_rows)+1)),
                "ISSUE ID": [str(r['ISSUE ID']) for r in sequence_rows],
                "WARD": [str(r.get('WARD', '')) for r in sequence_rows],
                "STATUS": [str(r.get('STATUS', '')) for r in sequence_rows],
                "Leg dist (m)": leg_dists
            })
            st.dataframe(list_df, use_container_width=True)

            # Highlight on map (first = green, rest = orange)
            st.session_state.current_target_id = str(sequence_rows[0]['ISSUE ID'])
            st.session_state.batch_target_ids = {str(r['ISSUE ID']) for r in sequence_rows}

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Mark first as Done (visited)"):
                    st.session_state.visited_ticket_ids.add(str(sequence_rows[0]['ISSUE ID']))
                    st.experimental_rerun()
            with c2:
                if st.button("⏭️ Skip first ticket"):
                    st.session_state.skipped_ticket_ids.add(str(sequence_rows[0]['ISSUE ID']))
                    st.experimental_rerun()
        else:
            st.info("Provide your location to compute a batch route.")

        # -------------------------
        # Step 5 — Map Display (ALL filtered points, batch highlighted)
        # -------------------------
        st.subheader("Step 5: Map Display Options")
        st.write(f"**Total clusters found (non-noise)**: {df[df['CLUSTER NUMBER']!=-1]['CLUSTER NUMBER'].nunique()}")
        center_on_first = st.checkbox("Center map on first stop (if any)", value=False)

        map_type = st.radio(
            "Select map type:",
            [
                "Show all markers (Type 1)",
                "Use Dynamic Clustering (Type 2)"
            ],
            index=0
        )

        # Map center
        if center_on_first and sequence_rows:
            map_center = [float(sequence_rows[0]['LATITUDE']), float(sequence_rows[0]['LONGITUDE'])]
            zoom_level = 16
        else:
            map_center = [float(df['LATITUDE'].mean()), float(df['LONGITUDE'].mean())]
            zoom_level = 13

        m = folium.Map(location=map_center, zoom_start=zoom_level)

        # Ward overlay (if provided)
        if wards_gdf is not None and not wards_gdf.empty:
            try:
                folium.GeoJson(wards_gdf, name="Wards").add_to(m)
            except Exception:
                pass

        target_id = st.session_state.get('current_target_id')
        batch_ids = st.session_state.get('batch_target_ids', set())

        if map_type == "Show all markers (Type 1)":
            for _, row in gdf_all.iterrows():
                rid = str(row['ISSUE ID'])
                is_first = (rid == str(target_id)) if target_id else False
                in_batch = rid in batch_ids
                if is_first:
                    color, size = 'green', 10
                elif in_batch:
                    color, size = 'orange', 9
                else:
                    color, size = 'red', 7
                folium.CircleMarker(
                    location=[float(row['LATITUDE']), float(row['LONGITUDE'])],
                    radius=size,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9 if (is_first or in_batch) else 0.85,
                    popup=(
                        f"Cluster {int(row['CLUSTER NUMBER']) if row['IS_CLUSTERED'] else 'N/A'}<br>"
                        f"Issue ID: {row['ISSUE ID']}<br>"
                        f"Ward: {row['WARD']}<br>"
                        f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
                    )
                ).add_to(m)
        else:
            mc = MarkerCluster(name="Tickets").add_to(m)
            for _, row in gdf_all.iterrows():
                rid = str(row['ISSUE ID'])
                is_first = (rid == str(target_id)) if target_id else False
                in_batch = rid in batch_ids
                icon_color = 'green' if is_first else ('orange' if in_batch else 'red')
                folium.Marker(
                    location=[float(row['LATITUDE']), float(row['LONGITUDE'])],
                    popup=(
                        f"Cluster {int(row['CLUSTER NUMBER']) if row['IS_CLUSTERED'] else 'N/A'}<br>"
                        f"Issue ID: {row['ISSUE ID']}<br>"
                        f"Ward: {row['WARD']}<br>"
                        f"Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}"
                    ),
                    icon=folium.Icon(color=icon_color, icon='info-sign')
                ).add_to(mc)

        folium.LayerControl().add_to(m)
        html_filename = "Clustering_Application_Map.html"
        m.save(html_filename)

        # -------------------------
        # Step 6 — Downloads
        # -------------------------
        st.subheader("Step 6: Download Outputs")
        with open(excel_filename, "rb") as f:
            st.download_button("Download Clustering Application Summary (Excel)", f, file_name=excel_filename)
        with open(html_filename, "rb") as f:
            st.download_button("Download Clustering Application Map (HTML)", f, file_name=html_filename)

        st.success("✅ Processing complete. Batch route + map + downloads are ready.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload the required CSV file to proceed.")