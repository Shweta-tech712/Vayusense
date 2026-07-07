import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from typing import Dict, Tuple, List, Any

# Ensure output directories exist
os.makedirs("data/outputs", exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard CPCB AQI Categories
CPCB_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

class AQIEvaluator:
    """
    Research-grade model evaluation suite for continuous surface AQI regression models
    and CPCB severity index classification mapping.
    """
    
    def __init__(self, style_sheet: str = "seaborn-v0_8-darkgrid"):
        self.style_sheet = style_sheet
        try:
            plt.style.use(style_sheet)
        except Exception:
            plt.style.use("ggplot") # Fallback style
            
        # Customize matplotlib for professional publication quality
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16,
            'savefig.dpi': 300,
            'figure.autolayout': True
        })
        
    @staticmethod
    def map_to_cpcb_category(aqi: float) -> str:
        """
        Maps a continuous AQI value to CPCB categories.
        """
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    def compute_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates scientific continuous regression metrics.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        pearson_r, p_val = stats.pearsonr(y_true, y_pred)
        spearman_r, _ = stats.spearmanr(y_true, y_pred)
        
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "pearson_r": float(pearson_r),
            "pearson_p_value": float(p_val),
            "spearman_r": float(spearman_r)
        }
        logger.info("Continuous regression metrics computed successfully.")
        return metrics

    def compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Maps continuous inputs to categorical CPCB scales and computes classification statistics.
        """
        cat_true = np.array([self.map_to_cpcb_category(y) for y in y_true])
        cat_pred = np.array([self.map_to_cpcb_category(y) for y in y_pred])
        
        # Compute confusion matrix with fixed labels list to align dimensions
        cm = confusion_matrix(cat_true, cat_pred, labels=CPCB_CLASSES)
        
        # Multi-class classification report dictionary
        report = classification_report(
            cat_true, cat_pred, 
            labels=CPCB_CLASSES, 
            output_dict=True, 
            zero_division=0
        )
        
        logger.info("Categorical classification metrics computed successfully.")
        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "categorical_true": cat_true,
            "categorical_pred": cat_pred
        }

    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "data/outputs/aqi_prediction_scatter.png"):
        """
        Generates publication-quality true vs predicted scatter plot.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Plot data points
        ax.scatter(y_true, y_pred, alpha=0.5, color='#0284c7', edgecolors='w', linewidths=0.5, label='AQI Predictions')
        
        # 1:1 reference line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0], 0),
            max(ax.get_xlim()[1], ax.get_ylim()[1], 500)
        ]
        ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=1.5, label='1:1 Line')
        
        # Calculate OLS fit line
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        x_vals = np.linspace(lims[0], lims[1], 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color='#ef4444', linewidth=2.0, label=f'OLS Fit (Slope: {slope:.2f})')
        
        # Annotate statistical summaries
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        stats_text = f"N = {len(y_true)}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.1f}\nMAE = {mean_absolute_error(y_true, y_pred):.1f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
        
        ax.set_xlabel('Ground Truth Observed AQI (CPCB)', fontsize=12)
        ax.set_ylabel('Model Predicted Surface AQI', fontsize=12)
        ax.set_title('CNN-LSTM Prediction Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(loc='lower right', frameon=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Prediction scatter plot saved to {save_path}")

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = "data/outputs/aqi_confusion_matrix.png"):
        """
        Plots confusion matrix using native Matplotlib.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Calculate row normalization percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized) # Clean divisions by zero
        
        # Plot using imshow
        im = ax.imshow(cm_normalized, cmap='Blues', interpolation='nearest')
        
        # Draw custom colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Normalized Row Frequencies')
        
        # Labels and ticks
        ax.set_xticks(np.arange(len(CPCB_CLASSES)))
        ax.set_yticks(np.arange(len(CPCB_CLASSES)))
        ax.set_xticklabels(CPCB_CLASSES, rotation=45, ha='right')
        ax.set_yticklabels(CPCB_CLASSES)
        
        # Grid label text annotations
        for i in range(len(CPCB_CLASSES)):
            for j in range(len(CPCB_CLASSES)):
                pct = cm_normalized[i, j]
                count = cm[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{count}\n({pct:.1%})", ha="center", va="center", color=color, fontsize=10)
        
        ax.set_xlabel('Predicted CPCB Severity Class', fontsize=12, labelpad=10)
        ax.set_ylabel('Actual CPCB Severity Class', fontsize=12, labelpad=10)
        ax.set_title('CPCB AQI Classification Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {save_path}")

    def plot_residuals_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "data/outputs/aqi_residuals_distribution.png"):
        """
        Plots prediction errors/residuals histogram overlaid with fit normal curves.
        """
        residuals = y_pred - y_true
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        # Plot residuals histogram using matplotlib hist
        ax.hist(residuals, bins=35, density=True, alpha=0.6, color='#10b981', edgecolor='white', label='Error Residuals')
        
        # Overlay normal distribution reference curve
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_res, std_res)
        ax.plot(x, p, 'r--', linewidth=2.0, label=f'Normal Fit ($\mu={mean_res:.1f}$, $\sigma={std_res:.1f}$)')
        
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Error Boundary')
        ax.axvline(mean_res, color='#ef4444', linestyle='-.', alpha=0.8, label=f'Mean Bias: {mean_res:.2f}')
        
        ax.set_xlabel('Prediction Error ($\hat{y} - y$)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Model Prediction Residuals Analysis', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Residuals distribution plot saved to {save_path}")

    def plot_class_performance(self, report: Dict[str, Any], save_path: str = "data/outputs/aqi_class_performance.png"):
        """
        Plots comparison bar chart of precision, recall, and F1-score across CPCB categories.
        """
        categories = []
        precisions = []
        recalls = []
        f1s = []
        
        for cls in CPCB_CLASSES:
            if cls in report:
                categories.append(cls)
                precisions.append(report[cls]['precision'])
                recalls.append(report[cls]['recall'])
                f1s.append(report[cls]['f1-score'])
                
        fig, ax = plt.subplots(figsize=(8.5, 6))
        
        x = np.arange(len(categories))
        width = 0.25
        
        # Grouped bar plots
        rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#3b82f6', edgecolor='#cbd5e1')
        rects2 = ax.bar(x, recalls, width, label='Recall', color='#10b981', edgecolor='#cbd5e1')
        rects3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#f59e0b', edgecolor='#cbd5e1')
        
        ax.set_ylabel('Metric Score (0.0 to 1.0)', fontsize=12)
        ax.set_xlabel('CPCB Air Quality Severity Category', fontsize=12)
        ax.set_title('CPCB Categorical Classification Metric Profile', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left', frameon=True)
        
        # Text labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f"{height:.2f}",
                                (rect.get_x() + rect.get_width() / 2., height + 0.01),
                                ha='center', va='bottom', fontsize=8, color='#334155')
                                
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
                            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Class performance bar chart saved to {save_path}")

    def evaluate_model_pipeline(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full evaluation and generates all metrics and graphs.
        """
        logger.info("Executing comprehensive model evaluation pipeline...")
        
        # 1. Calculate metrics
        reg_metrics = self.compute_regression_metrics(y_true, y_pred)
        cls_metrics = self.compute_classification_metrics(y_true, y_pred)
        
        # 2. Plot graphs
        self.plot_prediction_scatter(y_true, y_pred)
        self.plot_confusion_matrix(cls_metrics['confusion_matrix'])
        self.plot_residuals_distribution(y_true, y_pred)
        self.plot_class_performance(cls_metrics['classification_report'])
        
        logger.info("Model evaluation pipeline execution complete.")
        return {
            "regression": reg_metrics,
            "classification": cls_metrics['classification_report'],
            "confusion_matrix": cls_metrics['confusion_matrix']
        }

if __name__ == "__main__":
    # Test harness executing evaluation module validation with synthetic model predictions
    print("[RUNNING] Evaluating self-test data...")
    
    # Generate synthetic validation sets
    np.random.seed(42)
    sample_size = 500
    
    # Generate actual observed CPCB ground truth values
    actual_aqi = np.random.uniform(20.0, 480.0, size=sample_size)
    
    # Generate predictions with minor errors and a slight positive bias representing model residuals
    prediction_errors = np.random.normal(loc=1.5, scale=22.0, size=sample_size)
    predicted_aqi = np.clip(actual_aqi + prediction_errors, 0.0, 500.0)
    
    evaluator = AQIEvaluator()
    summary = evaluator.evaluate_model_pipeline(actual_aqi, predicted_aqi)
    
    print("\n--- REGRESSION METRICS ---")
    for key, value in summary['regression'].items():
        print(f"  {key.upper()}: {value:.4f}")
        
    print("\n--- CLASSIFICATION METRICS (MACRO AVERAGES) ---")
    macro = summary['classification']['macro avg']
    for key, value in macro.items():
        print(f"  MACRO {key.upper()}: {value:.4f}")
        
    print("\n[OK] Evaluation module self-test passed. Publication-quality plots saved in data/outputs/.")
