# Energy Forecasting and Anomaly Detection Project

## Overview
This project implements advanced machine learning models for energy consumption forecasting and anomaly detection using household power consumption data.

## Features

### 🔍 **Enhanced Visualization System**
- **Training History Plots**: Detailed learning curves for all models
- **Comprehensive Model Comparison**: Multi-panel charts with radar plots and performance rankings
- **Individual Model Analysis**: Detailed 6-panel analysis for each model including:
  - Time series predictions
  - Scatter plots for accuracy
  - Residual analysis
  - Error distribution
  - Performance metrics
  - Summary statistics

### 📊 **Model Performance Analysis**
- **RMSE, MAE, and R-squared** metrics for all forecasting models
- **Composite performance scoring** with automatic ranking
- **Radar charts** for multi-dimensional comparison
- **Bar charts** for metric comparison

### 🏆 **Intelligent Model Recommendations**
- **Automatic ranking** of models by performance
- **Detailed analysis** of best performing model
- **Model characteristics** explanation
- **Production deployment** recommendations
- **Ensemble suggestions** for improved robustness

### 🔍 **Anomaly Detection Visualization**
- **Reconstruction error analysis**
- **Anomaly threshold visualization**
- **Error distribution plots**
- **Anomaly statistics** (pie charts)
- **Example anomaly** time series plots

## Models Implemented

### Forecasting Models
1. **Temporal Transformer**: Advanced attention-based model for long-term dependencies
2. **CNN-LSTM Hybrid**: Convolutional and recurrent neural network combination

### Anomaly Detection Model
1. **Autoencoder CNN**: Unsupervised anomaly detection using reconstruction error

## Generated Visualizations

### 📈 Training Analysis
- Training vs validation loss curves
- Learning curve analysis
- Training summary statistics

### 📊 Model Comparison
- **4-panel comprehensive comparison**:
  - RMSE vs MAE bar charts
  - R-squared comparison
  - Radar chart for multi-metric analysis
  - Performance ranking with composite scores

### 📋 Individual Model Analysis
- **6-panel detailed analysis** per model:
  - Time series prediction vs actual
  - Prediction accuracy scatter plot
  - Residual analysis
  - Error distribution histogram
  - Performance metrics bar chart
  - Summary statistics text box

### 🔍 Anomaly Detection
- **4-panel anomaly analysis**:
  - Reconstruction error over time
  - Error distribution histogram
  - Anomaly detection results (pie chart)
  - Error statistics bar chart

## Usage

```bash
python main.py
```

## Output

The enhanced system will generate:

1. **Training visualizations** for all models
2. **Comprehensive model comparison** charts
3. **Individual detailed analysis** for each model
4. **Anomaly detection** visualizations
5. **Model recommendations** with rankings
6. **Final comprehensive report**

## Model Recommendations

The system automatically:
- Ranks models by composite performance score
- Identifies the best model for production deployment
- Suggests ensemble combinations
- Provides detailed analysis of model characteristics
- Offers optimization recommendations

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Configuration

All parameters can be adjusted in `config.py`:
- Model hyperparameters
- Training parameters
- Visualization settings
- Anomaly detection thresholds 