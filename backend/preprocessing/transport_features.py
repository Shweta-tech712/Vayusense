import numpy as np
import pandas as pd

def calculate_wind_features(df):
    """
    Computes wind speed, wind direction, and transport vectors from U and V wind components.
    """
    u = df["u_wind"]
    v = df["v_wind"]
    
    # 1. Wind Speed
    df["wind_speed"] = np.sqrt(u**2 + v**2)
    
    # 2. Wind Direction (meteorological angle in degrees)
    # 0 = North, 90 = East, 180 = South, 270 = West
    df["wind_direction"] = (np.degrees(np.arctan2(u, v)) + 360) % 360
    
    # 3. Transport Vectors (representing pollutant dispersion direction offsets)
    # Transport direction is downstream (opposite to wind direction)
    # transport_u/v represents downwind movement vectors
    df["transport_u"] = u * 1.2  # Scaled vector representing advection potential
    df["transport_v"] = v * 1.2
    
    # 4. Lagged features: Group by station/coordinates and shift by 1 day
    # Assuming dataframe is sorted by date
    if "latitude" in df.columns and "longitude" in df.columns:
        df["previous_day_wind"] = df.groupby(["latitude", "longitude"])["wind_speed"].shift(1).bfill()
    else:
        df["previous_day_wind"] = df["wind_speed"].shift(1).bfill()
        
    return df
