import os
import json
import pandas as pd

def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "final", "v1", "aqi_training_dataset.csv"))
    report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    report_path = os.path.join(report_dir, "dataset_quality_report.json")
    
    os.makedirs(report_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Calculate stats
    aqi_min = float(df["AQI"].min())
    aqi_max = float(df["AQI"].max())
    aqi_mean = float(df["AQI"].mean())
    
    pm25_min = float(df["PM25"].min())
    pm25_max = float(df["PM25"].max())
    pm25_mean = float(df["PM25"].mean())
    
    hcho_min = float(df["HCHO"].min()) if "HCHO" in df.columns else 0.0
    hcho_max = float(df["HCHO"].max()) if "HCHO" in df.columns else 0.0
    hcho_mean = float(df["HCHO"].mean()) if "HCHO" in df.columns else 0.0
    
    report = {
        "dataset_name": "aqi_training_dataset.csv",
        "total_records": len(df),
        "statistics": {
            "AQI": {
                "min": aqi_min,
                "max": aqi_max,
                "mean": round(aqi_mean, 2)
            },
            "PM25": {
                "min": pm25_min,
                "max": pm25_max,
                "mean": round(pm25_mean, 2)
            },
            "HCHO": {
                "min": hcho_min,
                "max": hcho_max,
                "mean": round(hcho_mean, 6),
                "range": [hcho_min, hcho_max]
            }
        }
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Quality report successfully saved to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
