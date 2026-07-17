import os
import json
import pandas as pd
import numpy as np

processed_dir = r"c:\Users\Vikram Kharat\Desktop\ISRO\backend\datasets\processed"
reports_dir = r"c:\Users\Vikram Kharat\Desktop\ISRO\backend\reports"

os.makedirs(reports_dir, exist_ok=True)

files = {
    "CPCB": "cpcb_processed.csv",
    "ERA5": "era5_processed.csv",
    "NASA FIRMS": "firms_processed.csv",
    "INSAT-3D": "insat_aod_processed.csv",
    "Sentinel-5P": "sentinel5p_merged.csv",
    "Fusion Dataset": "../final/v1/aqi_training_dataset.csv"
}

validation_results = {}

for name, filename in files.items():
    path = os.path.abspath(os.path.join(processed_dir, filename))
    if not os.path.exists(path):
        validation_results[name] = {"exists": False}
        continue
        
    try:
        df = pd.read_csv(path)
        null_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        
        # Coordinate check
        coords_valid = True
        if "latitude" in df.columns and "longitude" in df.columns:
            invalid_coords = df[
                (df["latitude"] < 8.4) | (df["latitude"] > 37.6) |
                (df["longitude"] < 68.1) | (df["longitude"] > 97.4)
            ]
            coords_valid = invalid_coords.empty
            
        # Date continuity (check if dates can be parsed)
        dates_valid = True
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
            except Exception:
                dates_valid = False
                
        # Outlier counts (using 3 sigma rule for numerical columns)
        num_cols = df.select_dtypes(include=[np.number]).columns
        outliers = 0
        for col in num_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                outliers += int(((df[col] - col_mean).abs() > 3 * col_std).sum())
                
        # Statistics summary
        stats = {}
        for col in num_cols[:5]: # Include first 5 numeric columns
            stats[col] = {
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0,
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0.0,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0.0
            }
            
        validation_results[name] = {
            "exists": True,
            "row_count": len(df),
            "columns_count": len(df.columns),
            "column_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": null_count,
            "duplicate_rows": duplicate_count,
            "coordinate_validity": "Valid" if coords_valid else "Invalid coords outside India bounds",
            "date_continuity": "Valid" if dates_valid else "Invalid dates",
            "outlier_points_count": outliers,
            "sample_statistics": stats
        }
    except Exception as e:
        validation_results[name] = {
            "exists": True,
            "error": str(e)
        }

# Write validation report
report_path = os.path.join(reports_dir, "dataset_validation_report.json")
with open(report_path, "w") as f:
    json.dump(validation_results, f, indent=2)
print(f"Dataset validation completed. Report saved to {report_path}")
