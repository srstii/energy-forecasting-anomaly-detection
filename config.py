"""
Configuration file for Energy Forecasting and Anomaly Detection Project
"""

# Data Configuration
DATA_PATH = 'household_power_consumption.txt'
TARGET_COLUMN = 'Global_active_power'

# Sequence Configuration
SEQUENCE_LENGTH = 24 * 7  # 7 days of hourly data (168 hours)
FORECAST_HORIZON = 24     # Predict next 24 hours

# Data Split Configuration
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

# Model Configuration
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# Transformer Model Parameters
HEAD_SIZE = 256
NUM_HEADS = 4
FF_DIM = 4
NUM_TRANSFORMER_BLOCKS = 2
MLP_UNITS = [128]
DROPOUT = 0.2

# CNN-LSTM Model Parameters
CNN_FILTERS = [64, 64, 32, 16]
CNN_KERNEL_SIZE = 3
LSTM_UNITS = 100

# Autoencoder Model Parameters
AE_FILTERS = [32, 16]
AE_KERNEL_SIZE = 3

# Anomaly Detection Configuration
ANOMALY_THRESHOLD_PERCENTILE = 95

# Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 100 