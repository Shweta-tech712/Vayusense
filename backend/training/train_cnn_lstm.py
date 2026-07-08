import os
import sys
import json
import logging
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input

# Ensure project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "model_training.log"))
    ]
)
logger = logging.getLogger("model_training")

class CNNLSTMTrainer:
    def __init__(self, data_dir=None, model_dir=None, config_path=None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "final", "v1")
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "..", "models", "v1")
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config", "model_config.json")
        os.makedirs(self.model_dir, exist_ok=True)
        self.load_configs()
        self.control_randomness()
        self.detect_gpu()

    def load_configs(self):
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            logger.info("Loaded model configurations successfully.")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def control_randomness(self):
        # Set random seeds for reproducibility
        seed = 42
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed controlled successfully. Seed value: {seed}")

    def detect_gpu(self):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"GPU detected: {gpus}. Model will train on GPU hardware accelerated interface.")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Failed to set memory growth: {e}")
        else:
            logger.info("No physical GPU found. Pipeline training falls back to host CPU processes.")

    def load_fused_data(self):
        logger.info("Loading fused train/test sequences from data directory...")
        X_train = np.load(os.path.join(self.data_dir, "X_train.npy"))
        X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        y_train = np.load(os.path.join(self.data_dir, "y_train.npy"))
        y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))
        
        logger.info(f"Data matrices successfully loaded: X_train {X_train.shape}, X_test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def build_multitask_model(self, num_features):
        logger.info("Constructing Multi-Task CNN-LSTM Deep Learning Architecture...")
        seq_len = self.config.get("sequence_length", 7)
        
        inputs = Input(shape=(seq_len, num_features), name="input_sequence")
        
        # Shared CNN-LSTM Feature Extractor
        x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same", name="shared_conv")(inputs)
        x = layers.BatchNormalization(name="shared_batchnorm")(x)
        x = layers.MaxPooling1D(pool_size=2, name="shared_maxpool")(x)
        x = layers.Dropout(0.2, name="shared_dropout_1")(x)
        
        x = layers.LSTM(units=64, return_sequences=False, name="shared_lstm")(x)
        x = layers.Dropout(0.2, name="shared_dropout_2")(x)
        
        shared_dense = layers.Dense(32, activation="relu", name="shared_dense")(x)
        
        # Head 1: AQI prediction (Linear output, Huber loss)
        aqi_head = layers.Dense(1, activation="linear", name="aqi_output")(shared_dense)
        
        # Head 2: PM2.5 prediction (Linear output, MSE loss)
        pm25_head = layers.Dense(1, activation="linear", name="pm25_output")(shared_dense)
        
        # Head 3: HCHO Hotspot Probability (Sigmoid output, BCE loss)
        hcho_head = layers.Dense(1, activation="sigmoid", name="hcho_output")(shared_dense)
        
        model = models.Model(inputs=inputs, outputs=[aqi_head, pm25_head, hcho_head], name="ISRO_CNN_LSTM_MultiTask")
        
        # Compile Model with joint loss functions
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.get("learning_rate", 0.001))
        model.compile(
            optimizer=optimizer,
            loss={
                "aqi_output": "huber",
                "pm25_output": "mean_squared_error",
                "hcho_output": "binary_crossentropy"
            },
            loss_weights={
                "aqi_output": 1.0,
                "pm25_output": 1.0,
                "hcho_output": 1.0
            }
        )
        
        model.summary(print_fn=logger.info)
        return model

    def train_model(self, model, X_train, y_train, X_test, y_test):
        logger.info("Initializing multi-loss gradient descent training loops...")
        
        # Output formatting for multi-head fitting
        y_train_dict = {
            "aqi_output": y_train[:, 0],
            "pm25_output": y_train[:, 1],
            "hcho_output": y_train[:, 2]
        }
        y_test_dict = {
            "aqi_output": y_test[:, 0],
            "pm25_output": y_test[:, 1],
            "hcho_output": y_test[:, 2]
        }
        
        # Callbacks
        model_path = os.path.join(self.model_dir, "cnn_lstm_aqi_model.keras")
        checkpoint = callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.get("early_stopping_patience", 5),
            restore_best_weights=True,
            mode="min",
            verbose=1
        )
        
        lr_reduction = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_test, y_test_dict),
            epochs=self.config.get("epochs", 10),
            batch_size=self.config.get("batch_size", 16),
            callbacks=[checkpoint, early_stopping, lr_reduction],
            verbose=1
        )
        
        logger.info(f"Model successfully saved to: {model_path}")
        return history

    def evaluate_model(self, model, X_test, y_test):
        logger.info("Evaluating validation split metrics...")
        
        preds = model.predict(X_test)
        pred_aqi = preds[0].flatten()
        pred_pm25 = preds[1].flatten()
        pred_hcho = preds[2].flatten()
        
        actual_aqi = y_test[:, 0]
        actual_pm25 = y_test[:, 1]
        actual_hcho = y_test[:, 2]
        
        # Load target scaler to evaluate in physical units
        import pickle
        target_scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "scalers", "target_scaler.pkl"))
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)
            
        preds_stacked = np.stack([pred_aqi, pred_pm25], axis=1)
        actuals_stacked = np.stack([actual_aqi, actual_pm25], axis=1)
        
        preds_inverse = target_scaler.inverse_transform(preds_stacked)
        actuals_inverse = target_scaler.inverse_transform(actuals_stacked)
        
        pred_aqi_phys = preds_inverse[:, 0]
        pred_pm25_phys = preds_inverse[:, 1]
        
        actual_aqi_phys = actuals_inverse[:, 0]
        actual_pm25_phys = actuals_inverse[:, 1]
        
        # Compute AQI Regression Metrics
        aqi_mae = mean_absolute_error(actual_aqi_phys, pred_aqi_phys)
        aqi_rmse = np.sqrt(mean_squared_error(actual_aqi_phys, pred_aqi_phys))
        aqi_r2 = r2_score(actual_aqi_phys, pred_aqi_phys)
        
        # Compute PM2.5 Regression Metrics
        pm25_mae = mean_absolute_error(actual_pm25_phys, pred_pm25_phys)
        pm25_rmse = np.sqrt(mean_squared_error(actual_pm25_phys, pred_pm25_phys))
        pm25_r2 = r2_score(actual_pm25_phys, pred_pm25_phys)
        
        # Compute HCHO classification indicators (hotspot threshold at 0.5)
        actual_hcho_class = (actual_hcho >= 0.5).astype(int)
        pred_hcho_class = (pred_hcho >= 0.5).astype(int)
        
        hcho_acc = accuracy_score(actual_hcho_class, pred_hcho_class)
        # Use zero_division=0 to handle cases where there are no predicted positives in small datasets
        hcho_prec = precision_score(actual_hcho_class, pred_hcho_class, zero_division=0)
        hcho_rec = recall_score(actual_hcho_class, pred_hcho_class, zero_division=0)
        hcho_f1 = f1_score(actual_hcho_class, pred_hcho_class, zero_division=0)
        
        evaluation = {
            "aqi": {"mae": float(aqi_mae), "rmse": float(aqi_rmse), "r2": float(aqi_r2)},
            "pm25": {"mae": float(pm25_mae), "rmse": float(pm25_rmse), "r2": float(pm25_r2)},
            "hcho": {
                "accuracy": float(hcho_acc),
                "precision": float(hcho_prec),
                "recall": float(hcho_rec),
                "f1_score": float(hcho_f1)
            }
        }
        
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        eval_file = os.path.join(reports_dir, "model_evaluation.json")
        with open(eval_file, "w") as f:
            json.dump(evaluation, f, indent=2)
            
        logger.info(f"Detailed performance metrics saved to {eval_file}")
        
        # Export Sample Predictions CSV
        comparison_df = pd.DataFrame({
            "actual_AQI": actual_aqi_phys,
            "predicted_AQI": pred_aqi_phys,
            "actual_PM25": actual_pm25_phys,
            "predicted_PM25": pred_pm25_phys,
            "actual_HCHO": actual_hcho,
            "predicted_HCHO": pred_hcho
        })
        comp_file = os.path.join(reports_dir, "sample_predictions.csv")
        comparison_df.to_csv(comp_file, index=False)
        logger.info(f"Sample prediction outputs exported to {comp_file}")
        
        return evaluation, comparison_df

    def save_explainability_background_samples(self, X_train):
        # Save background samples for SHAP analysis
        explain_dir = os.path.join(os.path.dirname(__file__), "..", "explainability")
        os.makedirs(explain_dir, exist_ok=True)
        
        # Grab background samples (limit to 20 samples to keep it efficient)
        num_samples = min(20, len(X_train))
        bg_samples = X_train[:num_samples]
        
        bg_file = os.path.join(explain_dir, "background_samples.npy")
        np.save(bg_file, bg_samples)
        logger.info(f"Saved {num_samples} background sequences for SHAP explainability at {bg_file}")

    def plot_figures(self, history, comparison_df):
        figures_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. Plot Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Train Loss (Total)")
        plt.plot(history.history["val_loss"], label="Val Loss (Total)")
        plt.title("CNN-LSTM Multi-Task Training Loss Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        loss_plot = os.path.join(figures_dir, "training_loss.png")
        plt.savefig(loss_plot, bbox_inches="tight")
        plt.close()
        
        # 2. Plot Prediction vs Actual for AQI
        plt.figure(figsize=(8, 8))
        plt.scatter(comparison_df["actual_AQI"], comparison_df["predicted_AQI"], color="blue", alpha=0.6, label="AQI Predictions")
        # Identity line
        min_val = min(comparison_df["actual_AQI"].min(), comparison_df["predicted_AQI"].min())
        max_val = max(comparison_df["actual_AQI"].max(), comparison_df["predicted_AQI"].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal fit")
        plt.title("Actual vs Predicted Air Quality Index (AQI)")
        plt.xlabel("Actual CPCB AQI")
        plt.ylabel("CNN-LSTM Predicted AQI")
        plt.grid(True)
        plt.legend()
        scatter_plot = os.path.join(figures_dir, "prediction_vs_actual.png")
        plt.savefig(scatter_plot, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Validation loss plots and regression figures exported to {figures_dir}")

    def save_model_metadata(self, evaluation):
        metadata_file = os.path.join(self.model_dir, "model_metadata.json")
        
        # Load feature names metadata to store in config
        feat_meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "metadata", "feature_names.json")
        features_list = []
        if os.path.exists(feat_meta_path):
            with open(feat_meta_path, "r") as f:
                features_list = json.load(f).get("features", [])
                
        meta = {
            "model_version": "v1.0.0",
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sequence_length": self.config.get("sequence_length", 7),
            "input_features": features_list,
            "metrics": evaluation
        }
        with open(metadata_file, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved versioned model metadata JSON: {metadata_file}")

    def save_history(self, history):
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        hist_file = os.path.join(reports_dir, "training_history.json")
        
        # Convert NumPy values in history to standard lists
        history_serializable = {}
        for k, v in history.history.items():
            history_serializable[k] = [float(x) for x in v]
            
        with open(hist_file, "w") as f:
            json.dump(history_serializable, f, indent=2)
        logger.info(f"Saved training history progress details to {hist_file}")

    def run(self):
        X_train, X_test, y_train, y_test = self.load_fused_data()
        
        num_features = X_train.shape[2]
        model = self.build_multitask_model(num_features)
        
        history = self.train_model(model, X_train, y_train, X_test, y_test)
        
        # Save training history details
        self.save_history(history)
        
        evaluation, comparison_df = self.evaluate_model(model, X_test, y_test)
        
        self.save_explainability_background_samples(X_train)
        
        self.plot_figures(history, comparison_df)
        
        self.save_model_metadata(evaluation)

if __name__ == "__main__":
    trainer = CNNLSTMTrainer()
    trainer.run()
