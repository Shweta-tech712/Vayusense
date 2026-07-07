import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import yaml
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger

logger = setup_logger("scientific_analysis")

class ScientificStatisticalAnalyzer:
    """
    Executes geostatistical correlation analyses, temporal cross-correlations,
    and lag analyses, exporting publication-grade scientific graphics.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.outputs_dir = self.config['paths']['outputs_dir']
        os.makedirs(self.outputs_dir, exist_ok=True)
        logger.info("Initialized Scientific Statistical Analyzer.")

    def calculate_correlations(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, float]:
        """
        Calculates Pearson (linear) and Spearman (monotonic rank) correlation
        coefficients, dropping null values pair-wise.
        """
        # Drop NaNs pair-wise to prevent math errors
        clean_df = df[[col1, col2]].dropna()
        
        if len(clean_df) < 5:
            logger.warning(f"Insufficient data points ({len(clean_df)}) to calculate correlations between {col1} and {col2}.")
            return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
            
        x = clean_df[col1]
        y = clean_df[col2]
        
        # 1. Pearson Correlation
        p_r, p_val = stats.pearsonr(x, y)
        
        # 2. Spearman Rank Correlation
        s_r, s_val = stats.spearmanr(x, y)
        
        metrics = {
            "pearson_r": float(p_r),
            "pearson_p_value": float(p_val),
            "spearman_r": float(s_r),
            "spearman_p_value": float(s_val),
            "data_points": len(clean_df)
        }
        logger.info(f"Correlations [{col1} vs {col2}]: Pearson R={p_r:.3f} (p={p_val:.2e}), Spearman R={s_r:.3f}")
        return metrics

    def calculate_cross_correlation(self, df: pd.DataFrame, col_cause: str, col_effect: str, 
                                    group_col: str = "station", max_lag: int = 5) -> Dict[int, float]:
        """
        Calculates cross-correlation coefficients across multiple temporal lags
        to model delay times between causes (e.g. fire FRP) and effects (e.g. urban AQI).
        Lags are evaluated within groups (e.g. stations) to prevent spatial mixing.
        """
        logger.info(f"Computing temporal cross-correlation between {col_cause} and {col_effect} (Max Lag: {max_lag} days)...")
        
        lag_correlations = {}
        
        # Sort values chronologically
        df = df.sort_values(by=[group_col, 'date'])
        
        for lag in range(-max_lag, max_lag + 1):
            temp_df = df.copy()
            # Shift cause series by the lag factor
            # positive lag: cause shifted forward in time (modeling delay before effect occurs)
            # negative lag: cause shifted backward in time
            temp_df['shifted_cause'] = temp_df.groupby(group_col)[col_cause].shift(lag)
            
            clean_df = temp_df[['shifted_cause', col_effect]].dropna()
            
            if len(clean_df) >= 10:
                r, _ = stats.pearsonr(clean_df['shifted_cause'], clean_df[col_effect])
                lag_correlations[lag] = float(r)
            else:
                lag_correlations[lag] = 0.0
                
        return lag_correlations

    def generate_scatter_plot(self, df: pd.DataFrame, col1: str, col2: str, 
                              label1: str, label2: str, title: str, filename: str) -> str:
        """
        Generates and exports a publication-ready scatter plot with a linear trendline.
        """
        clean_df = df[[col1, col2]].dropna()
        if clean_df.empty:
            return ""
            
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f172a') # Slate background
        ax.set_facecolor('#1e293b')
        
        x = clean_df[col1]
        y = clean_df[col2]
        
        # Plot coordinates scatter
        ax.scatter(x, y, color='#38bdf8', alpha=0.6, edgecolors='#0284c7', s=45, label='Data Points')
        
        # Fit linear regression trendline
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept
        
        ax.plot(line_x, line_y, color='#f43f5e', linewidth=2.5, linestyle='--',
                label=f'Fit (R={r_value:.2f}, p={p_value:.2e})')
        
        # Formatting
        ax.set_title(title, color='#38bdf8', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel(label1, color='#94a3b8')
        ax.set_ylabel(label2, color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
        
        # Save plot
        output_path = os.path.join(self.outputs_dir, filename)
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scientific scatter plot to: {output_path}")
        return output_path

    def generate_lag_plot(self, lag_correlations: Dict[int, float], title: str, 
                          filename: str, xlabel: str = "Time Lag (Days)") -> str:
        """
        Generates and exports a time-lag cross-correlation line plot.
        """
        lags = sorted(list(lag_correlations.keys()))
        corrs = [lag_correlations[lg] for lg in lags]
        
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        # Draw line and markers
        ax.plot(lags, corrs, color='#38bdf8', linewidth=2.5, marker='o', markersize=8, markerfacecolor='#f43f5e', markeredgecolor='#ffffff', label='Cross-Correlation')
        ax.axhline(0, color='#94a3b8', linestyle=':', alpha=0.7)
        ax.axvline(0, color='#f43f5e', linestyle='--', alpha=0.5, label='Zero Lag')
        
        # Highlight optimal lag (max correlation magnitude)
        abs_corrs = np.abs(corrs)
        opt_idx = np.argmax(abs_corrs)
        opt_lag = lags[opt_idx]
        opt_corr = corrs[opt_idx]
        
        ax.annotate(f"Optimal Lag: {opt_lag}d\n(R={opt_corr:.2f})", 
                    xy=(opt_lag, opt_corr), 
                    xytext=(opt_lag + 0.5, opt_corr + 0.05 if opt_corr > 0 else opt_corr - 0.1),
                    color='#38bdf8', fontweight='bold',
                    arrowprops=dict(facecolor='#38bdf8', shrink=0.05, width=1.5, headwidth=6))
        
        # Formatting
        ax.set_title(title, color='#38bdf8', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel(xlabel, color='#94a3b8')
        ax.set_ylabel("Pearson Correlation Coefficient (R)", color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
        
        # Save plot
        output_path = os.path.join(self.outputs_dir, filename)
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cross-correlation lag plot to: {output_path}")
        return output_path

    def analyze_fire_vs_hcho(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes complete correlation chain between active fire FRP
        and Sentinel-5P HCHO column density.
        """
        logger.info("Executing Fire vs HCHO scientific correlation...")
        results = {}
        
        # 1. Baseline correlations
        results['correlations'] = self.calculate_correlations(df, 'total_frp', 'hcho')
        
        # 2. Lag cross-correlations
        results['lag_correlations'] = self.calculate_cross_correlation(df, 'total_frp', 'hcho', max_lag=5)
        
        # 3. Generate plots
        self.generate_scatter_plot(
            df=df, col1='total_frp', col2='hcho',
            label1="Cumulative Fire Radiative Power (MW)",
            label2="Tropospheric HCHO Column Density (molec/cm2)",
            title="Biomass Burning Intensity (FRP) vs Formaldehyde (HCHO) Columns",
            filename="fire_vs_hcho_scatter.png"
        )
        
        self.generate_lag_plot(
            lag_correlations=results['lag_correlations'],
            title="Temporal Lag Correlation: Biomass Burning (FRP) to HCHO Response",
            filename="fire_vs_hcho_lag.png"
        )
        
        return results

    def analyze_fire_vs_aqi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes complete correlation chain between active fire FRP
        and CPCB ground station AQI.
        """
        logger.info("Executing Fire vs AQI scientific correlation...")
        results = {}
        
        # 1. Baseline correlations
        results['correlations'] = self.calculate_correlations(df, 'total_frp', 'cpcb_aqi')
        
        # 2. Lag cross-correlations
        results['lag_correlations'] = self.calculate_cross_correlation(df, 'total_frp', 'cpcb_aqi', max_lag=5)
        
        # 3. Generate plots
        self.generate_scatter_plot(
            df=df, col1='total_frp', col2='cpcb_aqi',
            label1="Cumulative Fire Radiative Power (MW)",
            label2="CPCB Ground Station AQI",
            title="Biomass Burning Intensity (FRP) vs CPCB Surface AQI",
            filename="fire_vs_aqi_scatter.png"
        )
        
        self.generate_lag_plot(
            lag_correlations=results['lag_correlations'],
            title="Temporal Lag Correlation: Biomass Burning (FRP) to Surface AQI Impact",
            filename="fire_vs_aqi_lag.png"
        )
        
        return results
