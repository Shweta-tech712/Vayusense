import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import yaml
from typing import Dict, List, Tuple, Any, Union
from utils.logger import setup_logger

logger = setup_logger("biomass_burning")

class BiomassBurningAnalyzer:
    """
    Analyzes NASA FIRMS active fire datasets to map biomass burning source regions,
    frequencies, densities, agricultural seasons, and administrative district distributions.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.bbox = self.config['spatial']['india_bbox']
        logger.info("Initialized Biomass Burning Analyzer.")

    @staticmethod
    def classify_indian_agricultural_season(date: pd.Timestamp) -> str:
        """
        Classifies dates into Indian agricultural burning seasons:
        - Rabi Harvest (April - May): Wheat residue burning
        - Monsoon (June - September): High rainfall, low fire activity
        - Kharif Harvest (October - November): Heavy rice residue stubble burning
        - Winter (December - March): General clearing and heating fires
        """
        month = date.month
        if month in [4, 5]:
            return "Rabi Harvest (Wheat)"
        elif month in [6, 7, 8, 9]:
            return "Monsoon Season"
        elif month in [10, 11]:
            return "Kharif Harvest (Rice Stubble)"
        else:
            return "Winter Season"

    def calculate_fire_metrics(self, fires_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates baseline metrics: Total fire counts, mean FRP, and high confidence counts.
        """
        logger.info("Calculating baseline active fire metrics...")
        if fires_df.empty:
            return {"total_fires": 0, "mean_frp": 0.0, "high_confidence_fires": 0}
            
        high_conf_thresh = self.config['sensors']['firms']['confidence_threshold']
        
        # Verify confidence column naming mapping
        conf_col = 'confidence' if 'confidence' in fires_df.columns else fires_df.columns[0]
        
        total_fires = len(fires_df)
        mean_frp = float(fires_df['frp'].mean())
        
        # Filter high confidence points
        if pd.api.types.is_numeric_dtype(fires_df[conf_col]):
            high_conf = len(fires_df[fires_df[conf_col] >= high_conf_thresh])
        else:
            # Handle categorical flags if present in VIIRS ('nominal', 'high')
            high_conf = len(fires_df[fires_df[conf_col].astype(str).str.lower().isin(['high', 'nominal'])])
            
        metrics = {
            "total_fires": total_fires,
            "mean_frp_mw": mean_frp,
            "high_confidence_fires": high_conf,
            "cumulative_frp_mw": float(fires_df['frp'].sum())
        }
        return metrics

    def analyze_fire_frequency(self, fires_df: pd.DataFrame, spatial_res_degrees: float = 0.1) -> pd.DataFrame:
        """
        Identifies burning recurrence by grouping coordinate inputs into grid cells.
        Helps isolate persistent source regions (e.g. crop fields in Punjab).
        """
        logger.info(f"Analyzing fire recurrence frequencies at {spatial_res_degrees} degree resolution...")
        if fires_df.empty:
            return pd.DataFrame(columns=['grid_lon', 'grid_lat', 'fire_count', 'mean_frp', 'recurrence_days'])
            
        df = fires_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Round coordinates to make a spatial grid
        df['grid_lon'] = np.round(df['longitude'] / spatial_res_degrees) * spatial_res_degrees
        df['grid_lat'] = np.round(df['latitude'] / spatial_res_degrees) * spatial_res_degrees
        
        # Aggregate counts, mean FRP, and number of unique dates fires occurred
        frequency_df = df.groupby(['grid_lon', 'grid_lat']).agg(
            fire_count=('frp', 'count'),
            mean_frp=('frp', 'mean'),
            recurrence_days=('date', lambda x: x.dt.date.nunique())
        ).reset_index()
        
        frequency_df = frequency_df.sort_values(by='fire_count', ascending=False)
        return frequency_df

    def analyze_fire_density(self, fires_df: pd.DataFrame, spatial_res_km: float = 50.0) -> pd.DataFrame:
        """
        Calculates spatial fire density as number of fire points per 1000 sq km.
        """
        logger.info(f"Calculating spatial fire densities using {spatial_res_km}km grids...")
        if fires_df.empty:
            return pd.DataFrame(columns=['grid_lon', 'grid_lat', 'fire_density_per_1000km2'])
            
        # Convert to GeoDataFrame and project to metric Web Mercator (EPSG:3857)
        fires_gdf = gpd.GeoDataFrame(
            fires_df,
            geometry=gpd.points_from_xy(fires_df['longitude'], fires_df['latitude']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        
        # Extract projected bounds
        minx, miny, maxx, maxy = fires_gdf.total_bounds
        
        # Generate grid boundaries
        grid_size = spatial_res_km * 1000 # Convert to meters
        cols = list(np.arange(minx, maxx + grid_size, grid_size))
        rows = list(np.arange(miny, maxy + grid_size, grid_size))
        
        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))
                
        # Create grid dataframe
        grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:3857")
        grid['grid_id'] = range(len(grid))
        
        # Count fires in each grid cell
        joined = gpd.sjoin(fires_gdf, grid, how='inner', predicate='within')
        counts = joined.groupby('grid_id').size().reset_index(name='fire_count')
        
        grid = grid.merge(counts, on='grid_id', how='left').fillna(0)
        
        # Density formula: count / (cell_area_m2 / 1,000,000) * 1000 = count / cell_area_km2 * 1000
        cell_area_km2 = (grid_size / 1000) ** 2
        grid['fire_density_per_1000km2'] = (grid['fire_count'] / cell_area_km2) * 1000
        
        # Convert back to WGS84 for mapping output
        grid_wgs84 = grid.to_crs(epsg=4326)
        
        # Extract centroids coordinates for plotting
        grid_wgs84['longitude'] = grid_wgs84.geometry.centroid.x
        grid_wgs84['latitude'] = grid_wgs84.geometry.centroid.y
        
        return pd.DataFrame(grid_wgs84.drop(columns='geometry'))

    def analyze_temporal_trends(self, fires_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregates active fires by seasonal periods and months.
        """
        logger.info("Executing seasonal and monthly aggregations...")
        if fires_df.empty:
            empty_season = pd.DataFrame(columns=['season', 'fire_count', 'mean_frp'])
            empty_month = pd.DataFrame(columns=['month', 'fire_count', 'mean_frp'])
            return empty_season, empty_month
            
        df = fires_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['season'] = df['date'].apply(self.classify_indian_agricultural_season)
        
        # 1. Seasonal analysis
        seasonal_df = df.groupby('season').agg(
            fire_count=('frp', 'count'),
            mean_frp=('frp', 'mean'),
            cumulative_frp=('frp', 'sum')
        ).reset_index()
        
        # 2. Monthly analysis
        monthly_df = df.groupby('month').agg(
            fire_count=('frp', 'count'),
            mean_frp=('frp', 'mean'),
            cumulative_frp=('frp', 'sum')
        ).reset_index()
        
        # Sort months chronologically
        monthly_df = monthly_df.sort_values(by='month')
        
        return seasonal_df, monthly_df

    def analyze_district_wise(self, fires_df: pd.DataFrame, district_shp_path: str = None) -> pd.DataFrame:
        """
        Intersects active fire points with district polygons to calculate local loads.
        If no shapefile path is provided, falls back to a grid-based spatial partition of India.
        """
        logger.info("Running district-level spatial join analysis...")
        if fires_df.empty:
            return pd.DataFrame(columns=['district_name', 'fire_count', 'cumulative_frp'])
            
        fires_gdf = gpd.GeoDataFrame(
            fires_df,
            geometry=gpd.points_from_xy(fires_df['longitude'], fires_df['latitude']),
            crs="EPSG:4326"
        )
        
        # Check if administrative shapefile is present
        if district_shp_path and os.path.exists(district_shp_path):
            try:
                logger.info(f"Loading district shapefile from: {district_shp_path}")
                districts_gdf = gpd.read_file(district_shp_path)
                
                # Align coordinate projections
                if districts_gdf.crs != fires_gdf.crs:
                    districts_gdf = districts_gdf.to_crs(fires_gdf.crs)
                    
                # Identify standard district names column
                name_col = [col for col in districts_gdf.columns if col.lower() in ['district', 'dist_name', 'name_2', 'dtname']]
                name_col = name_col[0] if name_col else districts_gdf.columns[0]
                
                # Perform spatial join
                joined = gpd.sjoin(fires_gdf, districts_gdf, how='inner', predicate='within')
                
                district_stats = joined.groupby(name_col).agg(
                    fire_count=('frp', 'count'),
                    cumulative_frp=('frp', 'sum'),
                    mean_frp=('frp', 'mean')
                ).reset_index().rename(columns={name_col: 'district_name'})
                
                return district_stats.sort_values(by='fire_count', ascending=False)
                
            except Exception as e:
                logger.error(f"Error reading district shapefile. Falling back to grid partition: {e}")
                
        # FALLBACK: Construct a grid-based spatial partition representing India's districts
        logger.info("Constructing fallback grid-based spatial partitions...")
        
        # We divide the bounding box of India into 3° x 3° grid cells
        bbox = self.config['spatial']['india_bbox']
        grid_lons = np.arange(bbox[0], bbox[2] + 3.0, 3.0)
        grid_lats = np.arange(bbox[1], bbox[3] + 3.0, 3.0)
        
        polygons = []
        names = []
        grid_idx = 1
        
        for i in range(len(grid_lons)-1):
            for j in range(len(grid_lats)-1):
                poly = Polygon([
                    (grid_lons[i], grid_lats[j]),
                    (grid_lons[i+1], grid_lats[j]),
                    (grid_lons[i+1], grid_lats[j+1]),
                    (grid_lons[i], grid_lats[j+1])
                ])
                polygons.append(poly)
                names.append(f"Region-Zone-{grid_idx}")
                grid_idx += 1
                
        grid_gdf = gpd.GeoDataFrame({'district_name': names, 'geometry': polygons}, crs="EPSG:4326")
        
        # Spatial join with generated grid
        joined = gpd.sjoin(fires_gdf, grid_gdf, how='inner', predicate='within')
        district_stats = joined.groupby('district_name').agg(
            fire_count=('frp', 'count'),
            cumulative_frp=('frp', 'sum'),
            mean_frp=('frp', 'mean')
        ).reset_index()
        
        return district_stats.sort_values(by='fire_count', ascending=False)
