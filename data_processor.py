"""
Data Processing Module for Energy Forecasting and Anomaly Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import *

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df_hourly = None
        self.scaled_df_hourly = None
        
    def load_data(self):
        """Load and preprocess the household power consumption dataset"""
        print("Loading dataset...")
        try:
            df = pd.read_csv(DATA_PATH, sep=';',
                           parse_dates={'datetime': ['Date', 'Time']},
                           low_memory=False, na_values=['?'])
            print(f"Dataset loaded successfully from: {DATA_PATH}")
        except FileNotFoundError:
            print(f"Error: '{DATA_PATH}' not found.")
            raise
            
        # Set datetime as index
        df = df.set_index('datetime')
        print(f"Original DataFrame shape: {df.shape}")
        
        # Convert columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Handle missing values
        print("Handling missing values with time-based interpolation...")
        df.interpolate(method='time', inplace=True)
        df.fillna(df.mean(), inplace=True)
        print(f"DataFrame shape after imputation: {df.shape}")
        print(f"Number of NaN values after imputation: {df.isnull().sum().sum()}")
        
        return df
    
    def resample_and_feature_engineer(self, df):
        """Resample data to hourly frequency and add time-based features"""
        print("Resampling data to hourly frequency...")
        self.df_hourly = df[TARGET_COLUMN].resample('h').mean().to_frame()
        self.df_hourly.columns = ['active_power']
        
        # Add time-based features
        print("Adding time-based features...")
        self.df_hourly['hour'] = self.df_hourly.index.hour
        self.df_hourly['day_of_week'] = self.df_hourly.index.dayofweek
        self.df_hourly['day_of_year'] = self.df_hourly.index.dayofyear
        self.df_hourly['month'] = self.df_hourly.index.month
        self.df_hourly['year'] = self.df_hourly.index.year
        
        print("Hourly resampled and featured DataFrame head:")
        print(self.df_hourly.head())
        
        return self.df_hourly
    
    def scale_features(self):
        """Scale numerical features using MinMaxScaler"""
        print("Scaling numerical features...")
        scaled_data = self.scaler.fit_transform(self.df_hourly)
        self.scaled_df_hourly = pd.DataFrame(scaled_data, columns=self.df_hourly.columns, 
                                           index=self.df_hourly.index)
        print("Scaled DataFrame head:")
        print(self.scaled_df_hourly.head())
        
        return self.scaled_df_hourly
    
    def create_sequences(self, data, sequence_length, forecast_horizon=1):
        """Create input-output sequences for supervised learning"""
        X, y = [], []
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length : i + sequence_length + forecast_horizon])
        return np.array(X), np.array(y)
    
    def prepare_forecasting_data(self):
        """Prepare data for forecasting models"""
        print(f"\nCreating sequences for forecasting (input_seq_length={SEQUENCE_LENGTH}, forecast_horizon={FORECAST_HORIZON})...")
        
        X_scaled_forecasting, y_scaled_forecasting = self.create_sequences(
            self.scaled_df_hourly[['active_power']].values, 
            SEQUENCE_LENGTH, 
            FORECAST_HORIZON
        )
        
        # Split data into training and testing sets (chronological split)
        train_size_forecasting = int(len(X_scaled_forecasting) * TRAIN_SPLIT)
        X_train_forecasting = X_scaled_forecasting[:train_size_forecasting]
        X_test_forecasting = X_scaled_forecasting[train_size_forecasting:]
        y_train_forecasting = y_scaled_forecasting[:train_size_forecasting]
        y_test_forecasting = y_scaled_forecasting[train_size_forecasting:]
        
        print(f"X_train_forecasting shape: {X_train_forecasting.shape}, y_train_forecasting shape: {y_train_forecasting.shape}")
        print(f"X_test_forecasting shape: {X_test_forecasting.shape}, y_test_forecasting shape: {y_test_forecasting.shape}")
        
        return (X_train_forecasting, X_test_forecasting, 
                y_train_forecasting, y_test_forecasting)
    
    def prepare_anomaly_detection_data(self):
        """Prepare data for anomaly detection models"""
        print(f"\nCreating sequences for anomaly detection (input_seq_length={SEQUENCE_LENGTH})...")
        
        X_anomaly_detection, _ = self.create_sequences(
            self.scaled_df_hourly.values, 
            SEQUENCE_LENGTH, 
            1
        )
        
        train_size_ad = int(len(X_anomaly_detection) * TRAIN_SPLIT)
        X_train_ad = X_anomaly_detection[:train_size_ad]
        X_test_ad = X_anomaly_detection[train_size_ad:]
        
        print(f"X_train_ad shape: {X_train_ad.shape}, X_test_ad shape: {X_test_ad.shape}")
        
        return X_train_ad, X_test_ad
    
    def inverse_transform_predictions(self, predictions):
        """Inverse transform predictions back to original scale"""
        try:
            predictions_flat = predictions.flatten()
            num_features = self.scaled_df_hourly.shape[1]
            dummy_array = np.zeros((len(predictions_flat), num_features))
            dummy_array[:, 0] = predictions_flat
            inverted_data = self.scaler.inverse_transform(dummy_array)[:, 0]
            return inverted_data.reshape(predictions.shape)
        except Exception as e:
            print(f"Warning: Error in inverse transform: {str(e)}")
            # Return original predictions if inverse transform fails
            return predictions
    
    def process_all_data(self):
        """Complete data processing pipeline"""
        try:
            df = self.load_data()
            self.resample_and_feature_engineer(df)
            self.scale_features()
            
            forecasting_data = self.prepare_forecasting_data()
            anomaly_data = self.prepare_anomaly_detection_data()
            
            return forecasting_data, anomaly_data
        except Exception as e:
            print(f"Error in data processing pipeline: {str(e)}")
            raise 