import ee
import datetime

class LiveSatelliteService:
    def __init__(self):
        self.POLLUTANTS = {
            "no2": {
                "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
                "band": "NO2_column_number_density",
                "scale": 1000
            },
            "so2": {
                "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
                "band": "SO2_column_number_density",
                "scale": 1000
            },
            "co": {
                "collection": "COPERNICUS/S5P/OFFL/L3_CO",
                "band": "CO_column_number_density",
                "scale": 1000
            },
            "o3": {
                "collection": "COPERNICUS/S5P/OFFL/L3_O3",
                "band": "O3_column_number_density",
                "scale": 1000
            },
            "hcho": {
                "collection": "COPERNICUS/S5P/OFFL/L3_HCHO",
                "band": "tropospheric_HCHO_column_number_density",
                "scale": 1000
            }
        }

    def get_live_satellite_data(self, latitude: float, longitude: float, radius_km: float = 10.0):
        """
        Query live Sentinel-5P data for a given location.
        Uses the last 7 days of data to ensure we don't get empty/null values due to clouds.
        """
        geom = ee.Geometry.Point([longitude, latitude]).buffer(radius_km * 1000)
        
        # Look back up to 7 days
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=7)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        results = {}
        acquisition_dates = []

        for name, info in self.POLLUTANTS.items():
            try:
                coll = ee.ImageCollection(info["collection"]) \
                         .filterBounds(geom) \
                         .filterDate(start_date_str, end_date_str) \
                         .select(info["band"])
                
                # We take the mean over the last 7 days to avoid nulls
                mean_img = coll.mean()
                
                mean_dict = mean_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=info["scale"],
                    maxPixels=1e9
                ).getInfo()
                
                val = mean_dict.get(info["band"])
                results[name] = float(val) if val is not None else None
                
                # For the date, we can just get the latest image's time if available
                # Count first to avoid calling .first() on empty collection
                count = int(coll.size().getInfo())
                if count > 0:
                    latest_img = ee.Image(coll.sort("system:time_start", False).first())
                    latest_time = latest_img.get("system:time_start").getInfo()
                    if latest_time is not None:
                        dates_obj = datetime.datetime.fromtimestamp(latest_time / 1000.0, tz=datetime.timezone.utc)
                        acquisition_dates.append(dates_obj)
            except Exception as e:
                print(f"Error querying {name}: {e}")
                results[name] = None
                
        latest_date = max(acquisition_dates).strftime("%Y-%m-%d %H:%M:%S UTC") if acquisition_dates else "Unknown"
        results["acquisition_date"] = latest_date
        results["source"] = "Google Earth Engine (Sentinel-5P)"
        
        return results

# Singleton instance
live_satellite_service = LiveSatelliteService()
