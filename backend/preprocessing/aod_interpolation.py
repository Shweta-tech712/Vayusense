import numpy as np
import pandas as pd

class AODInterpolator:
    """
    Inverse Distance Weighting (IDW) spatial interpolator.
    Computes spatially continuous values for any target coordinates in India
    based on nearby station measurements.
    """
    def __init__(self, power=2.0):
        self.power = power

    def fit_interpolate(self, source_lats, source_lons, source_vals, target_lats, target_lons):
        """
        source_lats: 1D array of latitude coordinates of source points
        source_lons: 1D array of longitude coordinates of source points
        source_vals: 1D array of values at source points
        target_lats: 1D array of target latitude coordinates
        target_lons: 1D array of target longitude coordinates
        """
        # Filter out invalid / NaN values
        valid_mask = ~np.isnan(source_vals)
        src_lat = np.array(source_lats)[valid_mask]
        src_lon = np.array(source_lons)[valid_mask]
        src_val = np.array(source_vals)[valid_mask]
        
        if len(src_val) == 0:
            return np.full_like(target_lats, np.nan, dtype=float)
            
        interpolated = []
        
        for t_lat, t_lon in zip(target_lats, target_lons):
            # Calculate Euclidean distances in degrees (sufficient for regional interpolation)
            dists = np.sqrt((src_lat - t_lat)**2 + (src_lon - t_lon)**2)
            
            # Avoid division by zero for points exactly matching a source node
            zero_mask = dists < 1e-5
            if np.any(zero_mask):
                interpolated.append(float(src_val[zero_mask][0]))
                continue
                
            weights = 1.0 / (dists ** self.power)
            total_weight = np.sum(weights)
            
            if total_weight == 0:
                interpolated.append(np.nan)
            else:
                val = np.sum(weights * src_val) / total_weight
                interpolated.append(float(val))
                
        return np.array(interpolated)
