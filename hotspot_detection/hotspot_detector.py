import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN, KMeans
import yaml
import time
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger

logger = setup_logger("hotspot_detector")

class HCHOHotspotDetector:
    """
    Identifies HCHO hotspots from Sentinel-5P HCHO column density grids.
    Implements and compares Static Threshold, Z-score, DBSCAN, and K-Means methods,
    documenting the optimal scientific workflow.
    
    --------------------------------------------------------------------------
    SCIENTIFIC COMPARISON & RECOMMENDATION: WHICH METHOD IS BEST?
    
    1. Anomaly Selection: Static Threshold vs. Z-Score
       - Static Threshold: Uses a fixed value (e.g., 3.0e15 molec/cm2). While simple,
         it fails to account for seasonal background cycles and sensor drift.
         In winter, background levels drop, causing the static threshold to miss active plumes.
         In summer, natural biogenic VOC increases trigger false positives.
       - Z-Score (Recommended): Dynamically scales with background conditions by using
         Z = (x - mean)/std. This automatically adjusts to seasonality, making it 
         robust for detecting relative local hotspots year-round.
         
    2. Spatial Clustering: DBSCAN vs. K-Means
       - K-Means: Forces every point into one of K clusters. This is sub-optimal because:
         a) It requires pre-defining the number of clusters (K), which is unknown in nature.
         b) It cannot handle spatial noise, pulling standalone outlier pixels into clusters.
         c) It assumes spherical clusters, failing to capture elongated advection plumes.
       - DBSCAN (Recommended): Density-Based Spatial Clustering of Applications with Noise:
         a) Does not require pre-specifying K; it discovers the natural number of clusters.
         b) Automatically isolates isolated outlier pixels as noise (label -1).
         c) Can find clusters of arbitrary shapes, matching the physical geometry of dispersing plumes.
         
    CONCLUSION:
       The gold standard is a HYBRID Z-SCORE + DBSCAN workflow. Z-score handles dynamic,
       seasonally-adapted outlier selection, while DBSCAN maps them into physically
       realistic, noise-free emission hotspot polygons.
    --------------------------------------------------------------------------
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("Initialized comparative HCHO Hotspot Detector.")

    def detect_by_static_threshold(self, hcho_grid: np.ndarray, threshold_val: float = 3.0e15) -> np.ndarray:
        """
        Flags pixels exceeding a fixed absolute column density value.
        """
        logger.info(f"Filtering pixels by static threshold: {threshold_val:.2e} molec/cm2")
        outlier_mask = (hcho_grid > threshold_val) & (~np.isnan(hcho_grid))
        return outlier_mask

    def detect_by_zscore(self, hcho_grid: np.ndarray, threshold_z: float = 2.5) -> Tuple[np.ndarray, float, float]:
        """
        Flags pixels exceeding the background baseline by k standard deviations.
        """
        valid_pixels = hcho_grid[~np.isnan(hcho_grid)]
        if len(valid_pixels) == 0:
            return np.zeros_like(hcho_grid, dtype=bool), 0.0, 0.0
            
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)
        anomaly_threshold = mean_val + (threshold_z * std_val)
        
        outlier_mask = (hcho_grid > anomaly_threshold) & (~np.isnan(hcho_grid))
        return outlier_mask, mean_val, anomaly_threshold

    def cluster_hotspots_dbscan(self, outlier_mask: np.ndarray, lon_mesh: np.ndarray, lat_mesh: np.ndarray, eps_km: float = 50.0, min_samples: int = 3) -> gpd.GeoDataFrame:
        """
        Groups outlier pixels into density-based clusters using DBSCAN,
        filtering out spatial noise.
        """
        y_indices, x_indices = np.where(outlier_mask)
        if len(x_indices) == 0:
            return gpd.GeoDataFrame(columns=['cluster_id', 'geometry'], geometry='geometry', crs="EPSG:4326")
            
        lons = lon_mesh[y_indices, x_indices]
        lats = lat_mesh[y_indices, x_indices]
        
        df_coords = pd.DataFrame({'longitude': lons, 'latitude': lats})
        gdf_points = gpd.GeoDataFrame(df_coords, geometry=gpd.points_from_xy(df_coords['longitude'], df_coords['latitude']), crs="EPSG:4326")
        
        # Project to metric CRS (EPSG:3857) to use kilometer-based distances
        gdf_metric = gdf_points.to_crs(epsg=3857)
        coords_matrix = np.column_stack((gdf_metric.geometry.x, gdf_metric.geometry.y))
        
        # Fit DBSCAN
        db = DBSCAN(eps=eps_km * 1000, min_samples=min_samples)
        labels = db.fit_predict(coords_matrix)
        gdf_points['cluster_id'] = labels
        
        # Filter out noise points (-1)
        gdf_clusters = gdf_points[gdf_points['cluster_id'] != -1]
        
        if gdf_clusters.empty:
            return gpd.GeoDataFrame(columns=['cluster_id', 'geometry'], geometry='geometry', crs="EPSG:4326")
            
        # Dissolve points into convex hulls
        polygons_list = []
        cluster_ids = []
        point_counts = []
        
        for c_id in gdf_clusters['cluster_id'].unique():
            c_points = gdf_clusters[gdf_clusters['cluster_id'] == c_id]
            points_geom = MultiPoint(c_points.geometry.tolist())
            hull = points_geom.convex_hull
            
            if hull.geom_type in ['Point', 'LineString']:
                hull = hull.buffer(0.1) # Buffer points/lines to form polygons
                
            polygons_list.append(hull)
            cluster_ids.append(c_id)
            point_counts.append(len(c_points))
            
        gdf_polygons = gpd.GeoDataFrame(
            {
                'cluster_id': cluster_ids,
                'point_count': point_counts,
                'geometry': polygons_list,
                'method': 'DBSCAN'
            },
            geometry='geometry',
            crs="EPSG:4326"
        )
        return gdf_polygons

    def cluster_hotspots_kmeans(self, outlier_mask: np.ndarray, lon_mesh: np.ndarray, lat_mesh: np.ndarray, n_clusters: int = 5) -> gpd.GeoDataFrame:
        """
        Groups outlier pixels into K centroid-based clusters using K-Means.
        Note: This forces noise points into clusters and assumes spherical clusters.
        """
        y_indices, x_indices = np.where(outlier_mask)
        if len(x_indices) == 0:
            return gpd.GeoDataFrame(columns=['cluster_id', 'geometry'], geometry='geometry', crs="EPSG:4326")
            
        lons = lon_mesh[y_indices, x_indices]
        lats = lat_mesh[y_indices, x_indices]
        
        df_coords = pd.DataFrame({'longitude': lons, 'latitude': lats})
        gdf_points = gpd.GeoDataFrame(df_coords, geometry=gpd.points_from_xy(df_coords['longitude'], df_coords['latitude']), crs="EPSG:4326")
        
        # Fit K-Means using coordinates
        # Limit clusters to the number of points if points count < requested clusters
        actual_clusters = min(n_clusters, len(gdf_points))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(np.column_stack((gdf_points['longitude'], gdf_points['latitude'])))
        gdf_points['cluster_id'] = labels
        
        polygons_list = []
        cluster_ids = []
        point_counts = []
        
        for c_id in gdf_points['cluster_id'].unique():
            c_points = gdf_points[gdf_points['cluster_id'] == c_id]
            points_geom = MultiPoint(c_points.geometry.tolist())
            hull = points_geom.convex_hull
            
            if hull.geom_type in ['Point', 'LineString']:
                hull = hull.buffer(0.1)
                
            polygons_list.append(hull)
            cluster_ids.append(c_id)
            point_counts.append(len(c_points))
            
        gdf_polygons = gpd.GeoDataFrame(
            {
                'cluster_id': cluster_ids,
                'point_count': point_counts,
                'geometry': polygons_list,
                'method': 'K-Means'
            },
            geometry='geometry',
            crs="EPSG:4326"
        )
        return gdf_polygons

    def correlate_hotspots_with_fires(self, gdf_hotspots: gpd.GeoDataFrame, fires_df: pd.DataFrame, buffer_km: float = 30.0) -> gpd.GeoDataFrame:
        """
        Calculates active fire statistics within a buffer zone around each HCHO hotspot.
        """
        gdf_hotspots = gdf_hotspots.copy()
        if gdf_hotspots.empty or fires_df.empty:
            gdf_hotspots['fire_count'] = 0
            gdf_hotspots['cumulative_frp'] = 0.0
            return gdf_hotspots
            
        try:
            fires_gdf = gpd.GeoDataFrame(fires_df, geometry=gpd.points_from_xy(fires_df['longitude'], fires_df['latitude']), crs="EPSG:4326")
            hotspots_metric = gdf_hotspots.to_crs(epsg=3857)
            fires_metric = fires_gdf.to_crs(epsg=3857)
            
            hotspots_metric['buffered_geom'] = hotspots_metric.geometry.buffer(buffer_km * 1000)
            hotspots_metric = hotspots_metric.set_geometry('buffered_geom')
            joined = gpd.sjoin(fires_metric, hotspots_metric, how='inner', predicate='within')
            
            if joined.empty:
                gdf_hotspots['fire_count'] = 0
                gdf_hotspots['cumulative_frp'] = 0.0
                return gdf_hotspots
                
            fire_summary = joined.groupby('cluster_id').agg(
                fire_count=('frp', 'count'),
                cumulative_frp=('frp', 'sum')
            ).reset_index()
            
            gdf_hotspots = gdf_hotspots.merge(fire_summary, on='cluster_id', how='left')
            gdf_hotspots['fire_count'] = gdf_hotspots['fire_count'].fillna(0).astype(int)
            gdf_hotspots['cumulative_frp'] = gdf_hotspots['cumulative_frp'].fillna(0.0)
            return gdf_hotspots
        except Exception as e:
            logger.error(f"Error in spatial hotspot fire correlation: {e}")
            gdf_hotspots['fire_count'] = 0
            gdf_hotspots['cumulative_frp'] = 0.0
            return gdf_hotspots

    def compare_hotspot_methods(self, hcho_grid: np.ndarray, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> Dict[str, Any]:
        """
        Runs and compares all hotspot detection methods.
        Returns a dictionary of execution metrics and comparison results.
        """
        comparison_results = {}
        
        # 1. Anomaly Extraction comparison
        t0 = time.time()
        static_mask = self.detect_by_static_threshold(hcho_grid, threshold_val=3.0e15)
        static_time = time.time() - t0
        
        t0 = time.time()
        z_mask, z_mean, z_thresh = self.detect_by_zscore(hcho_grid, threshold_z=2.5)
        z_time = time.time() - t0
        
        comparison_results['anomaly_selection'] = {
            'static_threshold_count': int(np.sum(static_mask)),
            'static_exec_time_sec': static_time,
            'zscore_threshold_count': int(np.sum(z_mask)),
            'zscore_mean_val': float(z_mean),
            'zscore_threshold_val': float(z_thresh),
            'zscore_exec_time_sec': z_time
        }
        
        # Use the robust Z-score anomalies to compare spatial clustering methods
        t0 = time.time()
        gdf_dbscan = self.cluster_hotspots_dbscan(z_mask, lon_mesh, lat_mesh, eps_km=50.0, min_samples=3)
        dbscan_time = time.time() - t0
        
        t0 = time.time()
        gdf_kmeans = self.cluster_hotspots_kmeans(z_mask, lon_mesh, lat_mesh, n_clusters=5)
        kmeans_time = time.time() - t0
        
        # Calculate noise ratios
        y_indices, x_indices = np.where(z_mask)
        total_anomalies = len(x_indices)
        
        dbscan_points_clustered = gdf_dbscan['point_count'].sum() if not gdf_dbscan.empty else 0
        dbscan_noise_points = total_anomalies - dbscan_points_clustered
        dbscan_noise_ratio = dbscan_noise_points / total_anomalies if total_anomalies > 0 else 0
        
        # K-Means forces all points into clusters, meaning noise ratio is always 0%
        kmeans_noise_ratio = 0.0
        
        comparison_results['clustering'] = {
            'total_anomaly_points': total_anomalies,
            'dbscan_clusters_found': len(gdf_dbscan),
            'dbscan_points_clustered': int(dbscan_points_clustered),
            'dbscan_noise_points': int(dbscan_noise_points),
            'dbscan_noise_ratio': float(dbscan_noise_ratio),
            'dbscan_exec_time_sec': dbscan_time,
            'kmeans_clusters_requested': 5,
            'kmeans_clusters_found': len(gdf_kmeans),
            'kmeans_noise_ratio': kmeans_noise_ratio,
            'kmeans_exec_time_sec': kmeans_time
        }
        
        logger.info("Completed comparative analysis of hotspot detection methods.")
        return comparison_results
