import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger

logger = setup_logger("wind_transport")

class WindTransportAnalyzer:
    """
    Models regional pollution transport pathways, advection trajectories,
    distances, bearings, and matches upwind fire hotspots with downwind cities.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.bbox = self.config['spatial']['india_bbox']
        self.outputs_dir = self.config['paths']['outputs_dir']
        os.makedirs(self.outputs_dir, exist_ok=True)
        logger.info("Initialized comparative Wind Transport Analyzer.")

    def calculate_wind_vectors(self, u_grid: np.ndarray, v_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates wind speed (m/s) and wind direction bearing (degrees).
        """
        wind_speed = np.sqrt(u_grid**2 + v_grid**2)
        # Meteorological angle (wind blowing from): (atan2(u,v)*180/pi + 360) % 360
        wind_dir = (np.arctan2(u_grid, v_grid) * 180 / np.pi + 360) % 360
        return wind_speed, wind_dir

    def simulate_advection_trajectory(self, start_coords: Tuple[float, float], 
                                      u_grid: np.ndarray, v_grid: np.ndarray, 
                                      lon_mesh: np.ndarray, lat_mesh: np.ndarray,
                                      duration_hours: int = 24, time_step_sec: int = 3600,
                                      direction: str = "forward") -> gpd.GeoDataFrame:
        """
        Simulates air parcel transport using time-step integration.
        start_coords: (lon, lat)
        direction: 'forward' (dispersion tracking) or 'backward' (source allocation)
        """
        raster_h, raster_w = u_grid.shape
        trajectory_points = [Point(start_coords[0], start_coords[1])]
        
        curr_lon, curr_lat = start_coords
        n_steps = int((duration_hours * 3600) / time_step_sec)
        multiplier = 1.0 if direction == "forward" else -1.0
        
        for step in range(n_steps):
            # Map lat/lon to grid pixels
            col_idx = int(((curr_lon - self.bbox[0]) / (self.bbox[2] - self.bbox[0])) * (raster_w - 1))
            row_idx = int(((curr_lat - self.bbox[1]) / (self.bbox[3] - self.bbox[1])) * (raster_h - 1))
            
            # Check grid boundaries
            if col_idx < 0 or col_idx >= raster_w or row_idx < 0 or row_idx >= raster_h:
                break
                
            u_val = u_grid[row_idx, col_idx]
            v_val = v_grid[row_idx, col_idx]
            
            if np.isnan(u_val) or np.isnan(v_val):
                break
                
            # Displacements (meters)
            dx = u_val * time_step_sec * multiplier
            dy = v_val * time_step_sec * multiplier
            
            # Degrees conversion: 1 deg lat ~ 111,000m, 1 deg lon ~ 111,000 * cos(lat)m
            d_lat = dy / 111000.0
            cos_lat = np.cos(np.radians(curr_lat))
            d_lon = dx / (111000.0 * max(cos_lat, 0.01))
            
            curr_lon += d_lon
            curr_lat += d_lat
            
            trajectory_points.append(Point(curr_lon, curr_lat))
            
        if len(trajectory_points) < 2:
            return gpd.GeoDataFrame(columns=['type', 'geometry'], geometry='geometry', crs="EPSG:4326")
            
        line = LineString(trajectory_points)
        gdf_trajectory = gpd.GeoDataFrame(
            {
                'type': [f"{direction}_trajectory"],
                'duration_hrs': [duration_hours],
                'step_count': [len(trajectory_points) - 1]
            },
            geometry=[line],
            crs="EPSG:4326"
        )
        return gdf_trajectory

    def calculate_transport_distance(self, trajectory_gdf: gpd.GeoDataFrame) -> float:
        """
        Projects the trajectory LineString to a metric projection (EPSG:3857)
        and computes the total advection distance in kilometers.
        """
        if trajectory_gdf.empty:
            return 0.0
        try:
            # Project to meters and get length
            metric_line = trajectory_gdf.to_crs(epsg=3857).geometry.iloc[0]
            distance_km = metric_line.length / 1000.0
            return float(distance_km)
        except Exception as e:
            logger.error(f"Error calculating transport distance: {e}")
            return 0.0

    def determine_transport_direction(self, trajectory_gdf: gpd.GeoDataFrame) -> Tuple[float, str]:
        """
        Calculates the start-to-end bearing direction in degrees and maps it to a compass heading.
        """
        if trajectory_gdf.empty:
            return 0.0, "Unknown"
            
        line = trajectory_gdf.geometry.iloc[0]
        start_pt = line.coords[0]
        end_pt = line.coords[-1]
        
        # Bearing = atan2(sin(d_lon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon))
        lon1, lat1 = np.radians(start_pt[0]), np.radians(start_pt[1])
        lon2, lat2 = np.radians(end_pt[0]), np.radians(end_pt[1])
        
        d_lon = lon2 - lon1
        y = np.sin(d_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        
        # Map bearing to cardinal direction
        cardinals = [
            (0, 22.5, "N"), (22.5, 67.5, "NE"), (67.5, 112.5, "E"), (112.5, 157.5, "SE"),
            (157.5, 202.5, "S"), (202.5, 247.5, "SW"), (247.5, 292.5, "W"), (292.5, 337.5, "NW"),
            (337.5, 360.0, "N")
        ]
        
        heading = "N"
        for low, high, tag in cardinals:
            if low <= bearing < high:
                heading = tag
                break
                
        return float(bearing), heading

    def check_fire_to_city_transport(self, fire_coords: Tuple[float, float], city_coords: Tuple[float, float], 
                                     u_grid: np.ndarray, v_grid: np.ndarray, 
                                     lon_mesh: np.ndarray, lat_mesh: np.ndarray,
                                     duration_hours: int = 24, time_step_sec: int = 3600,
                                     buffer_km: float = 30.0) -> Dict[str, Any]:
        """
        Determines if a forward trajectory from fire coordinates passes near a city.
        Returns travel distance, transit time, and intersection flag.
        """
        # 1. Simulate forward trajectory from fire source
        gdf_traj = self.simulate_advection_trajectory(
            start_coords=fire_coords,
            u_grid=u_grid,
            v_grid=v_grid,
            lon_mesh=lon_mesh,
            lat_mesh=lat_mesh,
            duration_hours=duration_hours,
            time_step_sec=time_step_sec,
            direction="forward"
        )
        
        if gdf_traj.empty:
            return {"intersect": False, "min_distance_km": np.nan, "transit_hours": np.nan}
            
        line = gdf_traj.geometry.iloc[0]
        city_point = Point(city_coords[0], city_coords[1])
        
        # 2. Project geometry to metric EPSG:3857 for distance calculations
        # Set geometries crs to WGS84 before re-projecting
        gdf_line = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326").to_crs(epsg=3857)
        gdf_city = gpd.GeoDataFrame(geometry=[city_point], crs="EPSG:4326").to_crs(epsg=3857)
        
        proj_line = gdf_line.geometry.iloc[0]
        proj_city = gdf_city.geometry.iloc[0]
        
        # 3. Find minimum distance from line to city center
        min_dist_m = proj_line.distance(proj_city)
        min_dist_km = min_dist_m / 1000.0
        
        # 4. Determine if it intersects city buffer
        intersects = min_dist_km <= buffer_km
        
        transit_hours = np.nan
        if intersects:
            # Find the step index closest to the city coordinates
            coords = list(line.coords)
            dists = []
            for (lon, lat) in coords:
                # Euclidean degrees distance to find the index
                dists.append(np.sqrt((lon - city_coords[0])**2 + (lat - city_coords[1])**2))
                
            closest_idx = np.argmin(dists)
            # Travel time: index * step size in hours
            transit_hours = closest_idx * (time_step_sec / 3600.0)
            
        return {
            "intersect": bool(intersects),
            "min_distance_km": float(min_dist_km),
            "transit_hours": float(transit_hours),
            "total_distance_km": self.calculate_transport_distance(gdf_traj)
        }

    def generate_static_trajectory_map(self, trajectory_gdf: gpd.GeoDataFrame, 
                                       city_coords_dict: Dict[str, Tuple[float, float]] = None) -> str:
        """
        Renders and saves a static map showing the wind transport trajectory
        and proximity markers to cities.
        """
        if trajectory_gdf.empty:
            logger.warning("Empty trajectory. Cannot plot static map.")
            return ""
            
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0f172a') # Dark slate background
        ax.set_facecolor('#1e293b')
        
        # Plot trajectory line
        line = trajectory_gdf.geometry.iloc[0]
        lons, lats = zip(*line.coords)
        ax.plot(lons, lats, color='#ef4444', linewidth=3, label='Pollution Trajectory', zorder=2)
        ax.scatter(lons[0], lats[0], color='#22c55e', s=120, marker='o', label='Hotspot Source', zorder=3)
        ax.scatter(lons[-1], lats[-1], color='#eab308', s=120, marker='X', label='24h Endpoint', zorder=3)
        
        # Plot major cities
        if city_coords_dict:
            for name, (lon, lat) in city_coords_dict.items():
                ax.scatter(lon, lat, color='#ffffff', s=80, marker='^', zorder=3)
                ax.annotate(name, (lon, lat), textcolor='#e2e8f0', xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
                
        # Formatting
        ax.set_title("Eulerian Air Parcel Advection Map (ERA5 850hPa)", color='#38bdf8', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Longitude (°E)", color='#94a3b8')
        ax.set_ylabel("Latitude (°N)", color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
        
        # Save image
        output_path = os.path.join(self.outputs_dir, "trajectory_map.png")
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved static trajectory map to: {output_path}")
        return output_path

    def get_sparse_wind_vectors(self, u_grid: np.ndarray, v_grid: np.ndarray, 
                                lon_mesh: np.ndarray, lat_mesh: np.ndarray, 
                                stride: int = 3) -> pd.DataFrame:
        """
        Downsamples wind grids to create a sparse set of wind vectors
        suitable for overlaying direction arrows on Folium maps.
        """
        sub_u = u_grid[::stride, ::stride]
        sub_v = v_grid[::stride, ::stride]
        sub_lon = lon_mesh[::stride, ::stride]
        sub_lat = lat_mesh[::stride, ::stride]
        
        flat_u = sub_u.flatten()
        flat_v = sub_v.flatten()
        flat_lon = sub_lon.flatten()
        flat_lat = sub_lat.flatten()
        
        valid = (~np.isnan(flat_u)) & (~np.isnan(flat_v))
        
        speed, direction = self.calculate_wind_vectors(flat_u[valid], flat_v[valid])
        
        df_vectors = pd.DataFrame({
            'longitude': flat_lon[valid],
            'latitude': flat_lat[valid],
            'u': flat_u[valid],
            'v': flat_v[valid],
            'speed': speed,
            'direction': direction
        })
        
        return df_vectors
