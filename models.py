"""
Neural Network Models for Energy Forecasting and Anomaly Detection
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                   MultiHeadAttention, Conv1D, Flatten, 
                                   MaxPooling1D, UpSampling1D, LSTM)
from config import *

class ModelBuilder:
    """Class to build different neural network architectures"""
    
    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs

        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return x + res

    @staticmethod
    def build_temporal_transformer(input_shape, forecast_horizon=1):
        """Build Temporal Transformer model for forecasting"""
        inputs = Input(shape=input_shape)
        x = inputs

        for _ in range(NUM_TRANSFORMER_BLOCKS):
            x = ModelBuilder.transformer_encoder(x, HEAD_SIZE, NUM_HEADS, FF_DIM, DROPOUT)

        x = Flatten()(x)

        for dim in MLP_UNITS:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(DROPOUT)(x)

        outputs = Dense(forecast_horizon)(x)

        model = Model(inputs, outputs, name="Temporal_Transformer")
        model.compile(optimizer='adam', loss='mse')
        
        return model

    @staticmethod
    def build_cnn_lstm(input_shape, forecast_horizon=1):
        """Build CNN-LSTM Hybrid model for forecasting"""
        model = tf.keras.models.Sequential([
            Conv1D(filters=CNN_FILTERS[0], kernel_size=CNN_KERNEL_SIZE, 
                   activation='relu', input_shape=input_shape),
            Conv1D(filters=CNN_FILTERS[1], kernel_size=CNN_KERNEL_SIZE, 
                   activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=CNN_FILTERS[2], kernel_size=CNN_KERNEL_SIZE, 
                   activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=CNN_FILTERS[3], kernel_size=CNN_KERNEL_SIZE, 
                   activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            LSTM(LSTM_UNITS, activation='relu', return_sequences=False),
            Dropout(DROPOUT),
            Dense(forecast_horizon)
        ], name="CNN_LSTM_Hybrid")
        
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def build_autoencoder_cnn(input_shape):
        """Build CNN-based Autoencoder for anomaly detection"""
        # Encoder
        input_layer = Input(shape=input_shape)
        x = Conv1D(filters=AE_FILTERS[0], kernel_size=AE_KERNEL_SIZE, 
                   activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(filters=AE_FILTERS[1], kernel_size=AE_KERNEL_SIZE, 
                   activation='relu', padding='same')(x)
        encoded = MaxPooling1D(pool_size=2, padding='same')(x)

        # Decoder
        x = Conv1D(filters=AE_FILTERS[1], kernel_size=AE_KERNEL_SIZE, 
                   activation='relu', padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = Conv1D(filters=AE_FILTERS[0], kernel_size=AE_KERNEL_SIZE, 
                   activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(filters=input_shape[-1], kernel_size=AE_KERNEL_SIZE, 
                        activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_layer, decoded, name="Autoencoder_CNN")
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder

class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        
    def train_model(self, model, X_train, y_train, model_name):
        """Train a model with early stopping"""
        try:
            print(f"\nTraining {model_name}...")
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=EARLY_STOPPING_PATIENCE, 
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                callbacks=[early_stopping],
                verbose=1
            )
            
            self.models[model_name] = model
            self.histories[model_name] = history
            
            return model, history
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            raise
    
    def get_model(self, model_name):
        """Get a trained model by name"""
        return self.models.get(model_name)
    
    def get_history(self, model_name):
        """Get training history by name"""
        return self.histories.get(model_name) 