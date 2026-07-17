import pandas as pd
import glob
import os
import json

processed_dir = r"c:\Users\Vikram Kharat\Desktop\ISRO\backend\datasets\processed"
csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))

report = {}

for path in csv_files:
    filename = os.path.basename(path)
    try:
        df = pd.read_csv(path)
        has_lat = "latitude" in df.columns
        has_lon = "longitude" in df.columns
        has_date = "date" in df.columns
        
        lat_min = float(df["latitude"].min()) if has_lat else None
        lat_max = float(df["latitude"].max()) if has_lat else None
        lon_min = float(df["longitude"].min()) if has_lon else None
        lon_max = float(df["longitude"].max()) if has_lon else None
        date_min = str(df["date"].min()) if has_date else None
        date_max = str(df["date"].max()) if has_date else None
        
        report[filename] = {
            "exists": True,
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "empty_rows": int((df.isnull().sum(axis=1) == len(df.columns)).sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "lat_range": [lat_min, lat_max],
            "lon_range": [lon_min, lon_max],
            "date_range": [date_min, date_max],
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "num_samples": len(df)
        }
    except Exception as e:
        report[filename] = {
            "exists": True,
            "error": str(e)
        }

# Also verify final dataset
final_path = r"c:\Users\Vikram Kharat\Desktop\ISRO\backend\datasets\final\v1\aqi_training_dataset.csv"
if os.path.exists(final_path):
    try:
        df = pd.read_csv(final_path)
        report["aqi_training_dataset.csv"] = {
            "exists": True,
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "empty_rows": int((df.isnull().sum(axis=1) == len(df.columns)).sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "lat_range": [float(df["latitude"].min()), float(df["latitude"].max())],
            "lon_range": [float(df["longitude"].min()), float(df["longitude"].max())],
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "num_samples": len(df)
        }
    except Exception as e:
         report["aqi_training_dataset.csv"] = {"exists": True, "error": str(e)}
else:
    report["aqi_training_dataset.csv"] = {"exists": False}

print(json.dumps(report, indent=2))
