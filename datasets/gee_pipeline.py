import os
import ee
import yaml
import pandas as pd
from typing import Dict, List, Any
from utils.logger import setup_logger
from utils.gee_auth import initialize_gee

logger = setup_logger("gee_pipeline")

class GEEPipeline:
    """
    Automated Google Earth Engine Pipeline.
    Manages querying, daily aggregation, and exporting of Sentinel-5P,
    MODIS (or custom INSAT-3D assets) AOD, and meteorological metrics.
    """
    def __init__(self, config_path: str = "config/config.yaml", insat3d_asset_path: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize the Earth Engine connection
        initialize_gee()
        
        # 1. Define India Administrative Boundary
        # Using the USDOS LSIB Simple 2017 collection, filtered for India
        self.india_boundary = (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                               .filter(ee.Filter.eq("country_na", "India")))
        
        self.insat3d_asset_path = insat3d_asset_path
        logger.info("Initialized GEE Pipeline with India Boundary geometry.")

    def get_aod_collection(self) -> ee.ImageCollection:
        """
        Retrieves the AOD dataset. Falls back to MODIS MCD19A2 if a private
        INSAT-3D GEE asset path is not configured.
        """
        if self.insat3d_asset_path:
            logger.info(f"Loading custom INSAT-3D AOD from private asset: {self.insat3d_asset_path}")
            return ee.ImageCollection(self.insat3d_asset_path)
        else:
            logger.info("Custom INSAT-3D GEE asset path not provided. Defaulting to MODIS MCD19A2 AOD.")
            m_config = self.config['sensors']['modis']
            return (ee.ImageCollection(m_config['aod_collection'])
                    .select(m_config['aod_band']))

    def filter_and_mask_s5p(self, start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Queries Sentinel-5P HCHO, masks low quality (qa_value < 0.5) pixels,
        and clips the output to the India boundary geometry.
        """
        logger.info(f"Filtering Sentinel-5P HCHO from {start_date} to {end_date}...")
        s5p_config = self.config['sensors']['sentinel_5p']
        
        collection = (ee.ImageCollection(s5p_config['hcho_collection'])
                      .filterDate(start_date, end_date)
                      .filterBounds(self.india_boundary))
        
        def mask_s5p_image(img: ee.Image) -> ee.Image:
            # Mask pixels with qa_value < 0.5 (clouds, snow, ice)
            qa = img.select(s5p_config['hcho_qa_band'])
            mask = qa.gte(s5p_config['hcho_qa_threshold'])
            
            # Select HCHO column density and multiply by 1e4 for standard representation (optional)
            hcho = img.select(s5p_config['hcho_band'])
            return hcho.updateMask(mask).clip(self.india_boundary)
            
        return collection.map(mask_s5p_image)

    def filter_and_mask_aod(self, start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Retrieves AOD (MODIS/INSAT-3D), clips to India, and scales values.
        """
        logger.info(f"Filtering AOD from {start_date} to {end_date}...")
        aod_coll = self.get_aod_collection()
        
        filtered = (aod_coll
                    .filterDate(start_date, end_date)
                    .filterBounds(self.india_boundary))
        
        def preprocess_aod(img: ee.Image) -> ee.Image:
            # If using MODIS, scale by 0.001 to get actual optical depth
            if not self.insat3d_asset_path:
                scaled_aod = img.multiply(0.001).rename("AOD")
            else:
                scaled_aod = img.rename("AOD")
            return scaled_aod.clip(self.india_boundary)
            
        return filtered.map(preprocess_aod)

    def calculate_daily_composites(self, image_collection: ee.ImageCollection, start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Generates daily mean composite grids over the temporal window.
        Loops through days using GEE date list mapping to handle missing orbital tracks.
        """
        logger.info("Computing daily composites across the study period...")
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        
        n_days = end.difference(start, 'days')
        date_list = ee.List.sequence(0, n_days.subtract(1)).map(lambda n: start.advance(n, 'day'))
        
        def make_daily_composite(d: ee.Parameter) -> ee.Image:
            curr_date = ee.Date(d)
            daily_set = image_collection.filterDate(curr_date, curr_date.advance(1, 'day'))
            
            # Use mean reducer to average overlapping orbits or pixel retrievals
            composite = daily_set.mean()
            
            # Embed metadata properties
            return (composite
                    .set('system:time_start', curr_date.millis())
                    .set('date_formatted', curr_date.format('YYYY-MM-DD')))
                    
        daily_collection = ee.ImageCollection(date_list.map(make_daily_composite))
        # Remove empty images (days with zero observations)
        return daily_collection.filter(ee.Filter.listContains("system:band_names", image_collection.first().bandNames().get(0)))

    def export_raster_to_drive(self, image: ee.Image, date_str: str, folder: str = "ISRO_GEE_Exports", scale: int = 1000) -> ee.batch.Task:
        """
        Submits an asynchronous task to export a daily GeoTIFF composite
        to Google Drive, clipped to the India boundary shape.
        """
        task_name = f"India_Grid_{date_str}"
        logger.info(f"Submitting GeoTIFF export task for {date_str} (Scale: {scale}m)...")
        
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=task_name,
            folder=folder,
            fileNamePrefix=task_name,
            region=self.india_boundary.geometry(),
            scale=scale,
            crs=self.config['spatial']['projection'],
            maxPixels=1e13
        )
        task.start()
        logger.info(f"Task started. Check GEE console with Task ID: {task.id}")
        return task

    def export_stats_to_csv(self, image: ee.Image, stations_fc: ee.FeatureCollection, date_str: str, folder: str = "ISRO_GEE_Exports") -> ee.batch.Task:
        """
        Extracts zonal stats or point values at station locations and exports them
        as a CSV file to Google Drive.
        """
        task_name = f"Station_Metrics_{date_str}"
        logger.info(f"Submitting CSV statistics export task for {date_str}...")
        
        # Sample points matching coordinates
        sampled = image.reduceRegions(
            collection=stations_fc,
            reducer=ee.Reducer.mean(),
            scale=1000
        )
        
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=task_name,
            folder=folder,
            fileNamePrefix=task_name,
            fileFormat="CSV"
        )
        task.start()
        logger.info(f"CSV task started. Task ID: {task.id}")
        return task

    def run_daily_export_pipeline(self, start_date: str, end_date: str, stations_csv_path: str = None) -> List[ee.batch.Task]:
        """
        Executes the full pipeline: downloads data, filters boundaries, aggregates composites,
        and triggers Google Drive exports for both GeoTIFFs and station point tables.
        """
        tasks = []
        
        # Load datasets
        hcho_coll = self.filter_and_mask_s5p(start_date, end_date)
        aod_coll = self.filter_and_mask_aod(start_date, end_date)
        
        # Combine images by band matching
        def combine_bands(img: ee.Image) -> ee.Image:
            date = img.date()
            # Match the corresponding day's AOD image
            matching_aod = aod_coll.filterDate(date, date.advance(1, 'day')).first()
            
            # If AOD is present, combine bands; otherwise add a dummy constant band
            aod_band = ee.Algorithms.If(
                matching_aod,
                matching_aod.select("AOD"),
                ee.Image.constant(0).rename("AOD").updateMask(ee.Image.constant(0))
            )
            return img.addBands(ee.Image(aod_band))
            
        combined_collection = ee.ImageCollection(hcho_coll.map(combine_bands))
        daily_collection = self.calculate_daily_composites(combined_collection, start_date, end_date)
        
        # Convert daily images to list for iterating
        daily_list = daily_collection.toList(daily_collection.size())
        size = daily_list.size().getInfo()
        
        logger.info(f"Found {size} days with valid satellite composites. Submitting GEE export tasks...")
        
        # Prepare station points FeatureCollection if provided
        stations_fc = None
        if stations_csv_path and os.path.exists(stations_csv_path):
            df = pd.read_csv(stations_csv_path)
            features = []
            for idx, row in df.iterrows():
                geom = ee.Geometry.Point([float(row['longitude']), float(row['latitude'])])
                feat = ee.Feature(geom, {'station_name': str(row['station'])})
                features.append(feat)
            stations_fc = ee.FeatureCollection(features)
            
        for i in range(size):
            img = ee.Image(daily_list.get(i))
            date_str = img.get('date_formatted').getInfo()
            
            # 1. Export raster GeoTIFF
            raster_task = self.export_raster_to_drive(img, date_str)
            tasks.append(raster_task)
            
            # 2. Export tabular CSV (if station coords are provided)
            if stations_fc:
                csv_task = self.export_stats_to_csv(img, stations_fc, date_str)
                tasks.append(csv_task)
                
        return tasks
