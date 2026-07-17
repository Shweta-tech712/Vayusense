import ee
import os
import time
import math
import logging
from cachetools import TTLCache
import datetime
import pandas as pd

from backend.config.gee_config import initialize_gee, check_gee_connection

logger = logging.getLogger("live_weather_service")

# In-memory Smart Cache (30 min TTL)
_weather_cache = TTLCache(maxsize=1000, ttl=1800)

class LiveWeatherService:
    def __init__(self):
        # We need the historical dataset for fallback logic (e.g. BLH)
        self.era5_df = None
        self._load_historical_dataset()

    def _load_historical_dataset(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(base_dir, "..", "datasets", "processed", "era5_processed.csv")
            if os.path.exists(dataset_path):
                self.era5_df = pd.read_csv(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load historical ERA5 dataset for fallback: {e}")

    def get_historical_blh(self, lat: float, lon: float) -> tuple[float, str]:
        """Gets closest historical BLH from dataset if available"""
        if self.era5_df is not None and not self.era5_df.empty:
            # Find closest matching lat/lon
            df = self.era5_df
            df['distance'] = ((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)**0.5
            closest_row = df.loc[df['distance'].idxmin()]
            if 'boundary_layer_height' in closest_row and pd.notnull(closest_row['boundary_layer_height']):
                return float(closest_row['boundary_layer_height']), str(closest_row.get('date', 'historical_mean'))
        return 1037.0, "historical_mean"  # Global dataset mean fallback

    def get_historical_fallback(self, lat: float, lon: float) -> dict:
        """Complete historical fallback if GEE fails entirely"""
        if self.era5_df is None or self.era5_df.empty:
            raise ValueError("No historical data available for fallback.")
        
        df = self.era5_df
        df['distance'] = ((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)**0.5
        row = df.loc[df['distance'].idxmin()]
        
        return {
            "weather": {
                "temperature": float(row.get("temperature", row.get("temperature_mean", 300.0))),
                "humidity": float(row.get("humidity", 50.0)),
                "pressure": float(row.get("pressure", 1013.25)),
                "rainfall": float(row.get("rainfall", 0.0)),
                "wind_speed": float(row.get("wind_speed", 2.0)),
                "wind_direction": float(row.get("wind_direction", 180.0)),
                "boundary_layer_height": float(row.get("boundary_layer_height", 1037.0))
            },
            "weather_source": "Historical ERA5",
            "dataset": "local_csv",
            "requested_time": datetime.datetime.utcnow().isoformat() + "Z",
            "dataset_time": str(row.get("date", "Unknown")),
            "retrieval_time": "0ms",
            "is_live": False,
            "confidence": 0.8
        }

    def get_live_weather(self, lat: float, lon: float) -> dict:
        start_time = time.time()
        
        # Phase 5: Smart Cache
        cache_key = (round(lat, 2), round(lon, 2))
        if cache_key in _weather_cache:
            result = _weather_cache[cache_key]
            result["retrieval_time"] = f"{(time.time() - start_time) * 1000:.0f}ms (cached)"
            logger.info(f"Weather cache hit for {cache_key}")
            return result

        logger.info(f"Weather cache miss for {cache_key}. Querying Live GEE ERA5...")
        
        # Initialize GEE
        is_init, _ = initialize_gee()
        if not is_init:
            logger.warning("GEE initialization failed, falling back to historical ERA5.")
            return self.get_historical_fallback(lat, lon)

        try:
            point = ee.Geometry.Point([lon, lat])
            
            # Phase 1: Retrieve from ERA5/HOURLY (or ERA5_LAND)
            dataset = 'ECMWF/ERA5_LAND/HOURLY'
            
            # Filter to the last 90 days to prevent full-collection sort timeouts
            end_date = datetime.datetime.utcnow()
            start_date = end_date - datetime.timedelta(days=90)
            
            collection = ee.ImageCollection(dataset).filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Get latest available image
            image = collection.filterBounds(point).sort('system:time_start', False).first()
            dataset_time = datetime.datetime.utcfromtimestamp(image.get('system:time_start').getInfo() / 1000).isoformat() + "Z"
            
            # Reduce region to get values
            data = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=11132,
                maxPixels=1e9
            ).getInfo()

            # Phase 2: Weather Variables & Conversions
            # Temperature (K -> C)
            temp_k = data.get('temperature_2m')
            temp_c = temp_k - 273.15 if temp_k is not None else 25.0
            
            # Pressure (Pa -> hPa)
            pressure_pa = data.get('surface_pressure')
            pressure_hpa = pressure_pa / 100.0 if pressure_pa is not None else 1013.25
            
            # Rainfall (m -> mm)
            precip_m = data.get('total_precipitation')
            rainfall_mm = precip_m * 1000.0 if precip_m is not None else 0.0
            
            # Wind speed & direction
            u = data.get('u_component_of_wind_10m', 0.0)
            v = data.get('v_component_of_wind_10m', 0.0)
            if u is None: u = 0.0
            if v is None: v = 0.0
            wind_speed = math.sqrt(u**2 + v**2)
            wind_dir = (math.degrees(math.atan2(u, v)) + 360) % 360
            
            # Relative Humidity via Clausius-Clapeyron
            dew_k = data.get('dewpoint_temperature_2m')
            if dew_k is not None and temp_k is not None:
                dew_c = dew_k - 273.15
                e = 6.11 * 10.0 ** (7.5 * dew_c / (237.3 + dew_c))
                es = 6.11 * 10.0 ** (7.5 * temp_c / (237.3 + temp_c))
                humidity = min(100.0, max(0.0, (e / es) * 100.0))
            else:
                humidity = 50.0

            # Phase 3: Boundary Layer Height (BLH)
            blh_data = None
            try:
                # Try getting BLH from the base ERA5 dataset (often delayed/daily)
                blh_end = datetime.datetime.utcnow()
                blh_start = blh_end - datetime.timedelta(days=90)
                blh_img = ee.ImageCollection('ECMWF/ERA5/DAILY').filterDate(
                    blh_start.strftime('%Y-%m-%d'), blh_end.strftime('%Y-%m-%d')
                ).filterBounds(point).sort('system:time_start', False).first()
                blh_data = blh_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=27830
                ).getInfo().get('boundary_layer_height')
            except Exception as blh_e:
                logger.warning(f"Failed to fetch live BLH: {blh_e}")
                pass
                
            is_blh_fallback = False
            blh_source = "ECMWF/ERA5/DAILY"
            
            if blh_data is not None:
                blh = float(blh_data)
            else:
                is_blh_fallback = True
                blh, _ = self.get_historical_blh(lat, lon)
                blh_source = "Historical ERA5"

            retrieval_ms = (time.time() - start_time) * 1000

            result = {
                "weather": {
                    "temperature": round(temp_c, 2),
                    "humidity": round(humidity, 2),
                    "pressure": round(pressure_hpa, 2),
                    "rainfall": round(rainfall_mm, 2),
                    "wind_speed": round(wind_speed, 2),
                    "wind_direction": round(wind_dir, 2),
                    "boundary_layer_height": round(blh, 2)
                },
                "weather_source": "ERA5 Live",
                "dataset": dataset,
                "requested_time": datetime.datetime.utcnow().isoformat() + "Z",
                "dataset_time": dataset_time,
                "retrieval_time": f"{retrieval_ms:.0f}ms",
                "is_live": True,
                "confidence": 1.0,
                "blh_metadata": {
                    "source": blh_source,
                    "is_fallback": is_blh_fallback
                }
            }

            # Add to cache
            _weather_cache[cache_key] = result
            
            # Phase 10: Logging
            logger.info(f"Live Weather Retrieved for {lat},{lon} | Dataset: {dataset} | TS: {dataset_time} | Time: {retrieval_ms:.0f}ms")

            return result

        except Exception as e:
            logger.error(f"Live weather retrieval failed for {lat},{lon}: {e}")
            logger.info("Falling back to historical dataset.")
            return self.get_historical_fallback(lat, lon)

live_weather_service = LiveWeatherService()
