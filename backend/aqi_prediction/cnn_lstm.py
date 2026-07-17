import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Tuple, List
import yaml
from utils.logger import setup_logger

logger = setup_logger("cnn_lstm")

class AQICNNLSTMModel:
    """
    Research-grade CNN-LSTM model designed to predict Surface AQI using
    spatio-temporal satellite and meteorological inputs.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.time_steps = self.config['model']['sequence_length']
        self.patch_size = self.config['model']['spatial_patch_size']
        logger.info(f"Initialized model parameters: sequence_length={self.time_steps}, patch_size={self.patch_size}")

    def build_model(self, num_channels: int) -> tf.keras.Model:
        """
        Constructs the CNN-LSTM neural network architecture using Keras.
        """
        input_shape = (self.time_steps, self.patch_size, self.patch_size, num_channels)
        logger.info(f"Building Keras CNN-LSTM Model with input shape: {input_shape}")
        
        # Define Input Layer
        inputs = layers.Input(shape=input_shape, name="spatial_temporal_input")
        
        # 1st TimeDistributed Conv2D Block (filters=32)
        x = layers.TimeDistributed(
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            name="td_conv_1"
        )(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization(), name="td_bn_1")(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)), name="td_pool_1")(x)
        
        # 2nd TimeDistributed Conv2D Block (filters=64)
        x = layers.TimeDistributed(
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            name="td_conv_2"
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name="td_bn_2")(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)), name="td_pool_2")(x)
        
        # Flatten Spatial Grids for temporal sequence processing
        # Output shape: (batch_size, time_steps, flat_features)
        x = layers.TimeDistributed(layers.Flatten(), name="td_flatten")(x)
        
        # Stacked LSTM Layers to learn temporal dependencies
        # First LSTM layer returns sequences for the second LSTM
        x = layers.LSTM(64, return_sequences=True, dropout=0.2, name="lstm_1")(x)
        # Second Bidirectional LSTM summarizes temporal representations
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2), name="bidirectional_lstm_2")(x)
        
        # Fully Connected Regressor Block
        x = layers.Dense(64, activation='relu', name="dense_fc_1")(x)
        x = layers.Dropout(0.3, name="dropout_fc_1")(x)
        
        x = layers.Dense(32, activation='relu', name="dense_fc_2")(x)
        
        # Output Layer: predicts continuous ground AQI
        outputs = layers.Dense(1, activation='linear', name="aqi_output")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ISRO_CNN_LSTM_AQI_Predictor")
        
        # Compile Model
        lr = self.config['model']['learning_rate']
        optimizer = optimizers.Adam(learning_rate=lr)
        
        # Using Mean Squared Error for training, track MAE and RMSE metrics
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        logger.info("Model construction and compilation completed successfully.")
        return model

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Defines standard model callbacks including early stopping and
        dynamic learning rate decay.
        """
        callbacks_list = [
            # Early Stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on loss plateaus to fine-tune weights
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            ),
            # Save checkpoints for the best-performing iteration
            callbacks.ModelCheckpoint(
                filepath=self.config['model']['checkpoint_path'],
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        return callbacks_list
