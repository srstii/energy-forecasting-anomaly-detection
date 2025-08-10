"""
Evaluation and Visualization Module for Energy Forecasting and Anomaly Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *

class ModelEvaluator:
    """Class to evaluate model performance and create visualizations"""
    
    def __init__(self):
        self.results = {}
        self.forecasting_models = {}
        self.anomaly_models = {}
        
    def evaluate_forecasting(self, y_true, y_pred, model_name):
        """Evaluate forecasting model performance"""
        # Debug information
        print(f"\n🔍 Debug: {model_name} data shapes:")
        print(f"   y_true shape: {y_true.shape}")
        print(f"   y_pred shape: {y_pred.shape}")
        
        # Flatten arrays to 1D for metric calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        print(f"   y_true_flat shape: {y_true_flat.shape}")
        print(f"   y_pred_flat shape: {y_pred_flat.shape}")
        
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        print(f"\n--- {model_name} Forecasting Performance ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        self.results[model_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        self.forecasting_models[model_name] = {'y_true': y_true, 'y_pred': y_pred}
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    def plot_training_history(self, history, model_name):
        """Plot training and validation loss"""
        try:
            plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]), dpi=DPI)
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label=f'{model_name} Training Loss', linewidth=2)
            plt.plot(history.history['val_loss'], label=f'{model_name} Validation Loss', linewidth=2)
            plt.title(f'{model_name} Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot learning curves
            plt.subplot(1, 2, 2)
            epochs = range(1, len(history.history['loss']) + 1)
            plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            plt.title(f'{model_name} Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print training summary
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            best_epoch = np.argmin(history.history['val_loss']) + 1
            best_val_loss = min(history.history['val_loss'])
            
            print(f"\n📊 {model_name} Training Summary:")
            print(f"   Final Training Loss: {final_train_loss:.6f}")
            print(f"   Final Validation Loss: {final_val_loss:.6f}")
            print(f"   Best Epoch: {best_epoch}")
            print(f"   Best Validation Loss: {best_val_loss:.6f}")
            
        except Exception as e:
            print(f"Error plotting training history for {model_name}: {str(e)}")
    
    def plot_forecasting_comparison(self, y_true, y_pred_transformer, y_pred_cnn_lstm, 
                                  transformer_name="Temporal Transformer", 
                                  cnn_lstm_name="CNN-LSTM Hybrid"):
        """Plot actual vs predicted values for forecasting models"""
        try:
            # Create comprehensive comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            
            # Flatten all data first
            y_true_flat = y_true.flatten()
            y_pred_transformer_flat = y_pred_transformer.flatten()
            y_pred_cnn_lstm_flat = y_pred_cnn_lstm.flatten()
            
            # Plot 1: Time series comparison
            sample_size = min(FORECAST_HORIZON * 10, len(y_true_flat))
            time_points = range(sample_size)
            
            axes[0, 0].plot(time_points, y_true_flat[:sample_size], 
                            label='Actual Power Consumption', color='blue', linewidth=2, alpha=0.8)
            axes[0, 0].plot(time_points, y_pred_transformer_flat[:sample_size], 
                            label=f'{transformer_name} Predictions', color='orange', linestyle='--', linewidth=2)
            axes[0, 0].plot(time_points, y_pred_cnn_lstm_flat[:sample_size], 
                            label=f'{cnn_lstm_name} Predictions', color='green', linestyle=':', linewidth=2)
            axes[0, 0].set_title(f'Energy Forecasting Comparison (First {sample_size} Hours)')
            axes[0, 0].set_xlabel('Time (Hours)')
            axes[0, 0].set_ylabel('Global Active Power (kW)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Scatter plots for accuracy
            axes[0, 1].scatter(y_true_flat, y_pred_transformer_flat, 
                               alpha=0.6, label=f'{transformer_name}', color='orange')
            axes[0, 1].scatter(y_true_flat, y_pred_cnn_lstm_flat, 
                               alpha=0.6, label=f'{cnn_lstm_name}', color='green')
            axes[0, 1].plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 
                            'r--', lw=2, label='Perfect Prediction')
            axes[0, 1].set_xlabel('Actual Values')
            axes[0, 1].set_ylabel('Predicted Values')
            axes[0, 1].set_title('Prediction Accuracy Scatter Plot')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Residual plots
            residuals_transformer = y_true_flat - y_pred_transformer_flat
            residuals_cnn_lstm = y_true_flat - y_pred_cnn_lstm_flat
            
            axes[1, 0].scatter(y_pred_transformer_flat, residuals_transformer, 
                               alpha=0.6, label=f'{transformer_name}', color='orange')
            axes[1, 0].scatter(y_pred_cnn_lstm_flat, residuals_cnn_lstm, 
                               alpha=0.6, label=f'{cnn_lstm_name}', color='green')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residual Plot')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Error distribution
            axes[1, 1].hist(residuals_transformer, bins=50, alpha=0.7, 
                            label=f'{transformer_name}', color='orange')
            axes[1, 1].hist(residuals_cnn_lstm, bins=50, alpha=0.7, 
                            label=f'{cnn_lstm_name}', color='green')
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Error Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in forecasting comparison plot: {str(e)}")
    
    def plot_model_performance_comparison(self):
        """Create comprehensive model performance comparison charts"""
        try:
            if len(self.results) == 0:
                print("No results available for comparison")
                return
                
            # Prepare data for visualization
            models = list(self.results.keys())
            metrics = ['RMSE', 'MAE', 'R2']
            
            # Create performance comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            
            # Plot 1: Bar chart of RMSE and MAE
            x = np.arange(len(models))
            width = 0.35
            
            rmse_values = [self.results[model]['RMSE'] for model in models]
            mae_values = [self.results[model]['MAE'] for model in models]
            
            axes[0, 0].bar(x - width/2, rmse_values, width, label='RMSE', color='skyblue', alpha=0.8)
            axes[0, 0].bar(x + width/2, mae_values, width, label='MAE', color='lightcoral', alpha=0.8)
            axes[0, 0].set_xlabel('Models')
            axes[0, 0].set_ylabel('Error Metrics')
            axes[0, 0].set_title('Model Performance Comparison (RMSE vs MAE)')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: R-squared comparison
            r2_values = [self.results[model]['R2'] for model in models]
            bars = axes[0, 1].bar(models, r2_values, color=['gold' if i == max(r2_values) else 'lightgray' for i in r2_values])
            axes[0, 1].set_xlabel('Models')
            axes[0, 1].set_ylabel('R-squared Score')
            axes[0, 1].set_title('Model Performance Comparison (R-squared)')
            axes[0, 1].set_xticklabels(models, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, r2_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 3: Radar chart for comprehensive comparison
            try:
                # Create polar subplot for radar chart
                ax_radar = plt.subplot(2, 2, 3, projection='polar')
                
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]  # Complete the circle
                
                for i, model in enumerate(models):
                    values = [self.results[model][metric] for metric in metrics]
                    # Normalize R2 (higher is better) and error metrics (lower is better)
                    normalized_values = []
                    for j, metric in enumerate(metrics):
                        if metric == 'R2':
                            normalized_values.append(values[j])  # R2 is already 0-1
                        else:
                            # Normalize error metrics (lower is better)
                            max_error = max([self.results[m][metric] for m in models])
                            min_error = min([self.results[m][metric] for m in models])
                            if max_error == min_error:
                                normalized_values.append(1.0)
                            else:
                                normalized_values.append(1 - (values[j] - min_error) / (max_error - min_error))
                    
                    normalized_values += normalized_values[:1]
                    ax_radar.plot(angles, normalized_values, 'o-', linewidth=2, label=model)
                    ax_radar.fill(angles, normalized_values, alpha=0.25)
                
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(metrics)
                ax_radar.set_title('Model Performance Radar Chart')
                ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax_radar.grid(True)
                
            except Exception as e:
                print(f"Warning: Could not create radar chart: {str(e)}")
                # Fallback to a simple bar chart
                axes[1, 0].text(0.5, 0.5, 'Radar Chart\nNot Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes,
                               fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
                axes[1, 0].set_title('Model Performance Radar Chart (Not Available)')
            
            # Plot 4: Performance ranking
            # Calculate composite score (lower RMSE/MAE, higher R2 is better)
            composite_scores = []
            for model in models:
                # Normalize and combine metrics
                rmse_norm = 1 - (self.results[model]['RMSE'] - min(rmse_values)) / (max(rmse_values) - min(rmse_values))
                mae_norm = 1 - (self.results[model]['MAE'] - min(mae_values)) / (max(mae_values) - min(mae_values))
                r2_norm = self.results[model]['R2']
                composite_score = (rmse_norm + mae_norm + r2_norm) / 3
                composite_scores.append(composite_score)
            
            # Sort by composite score
            sorted_indices = np.argsort(composite_scores)[::-1]  # Descending order
            sorted_models = [models[i] for i in sorted_indices]
            sorted_scores = [composite_scores[i] for i in sorted_indices]
            
            bars = axes[1, 1].bar(range(len(sorted_models)), sorted_scores, 
                                  color=['gold', 'silver', 'bronze'][:len(sorted_models)])
            axes[1, 1].set_xlabel('Models')
            axes[1, 1].set_ylabel('Composite Performance Score')
            axes[1, 1].set_title('Model Performance Ranking')
            axes[1, 1].set_xticks(range(len(sorted_models)))
            axes[1, 1].set_xticklabels(sorted_models, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, sorted_scores):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            return sorted_models, sorted_scores
            
        except Exception as e:
            print(f"Error in model performance comparison: {str(e)}")
            return [], []
    
    def create_forecasting_summary(self):
        """Create a summary table of forecasting results"""
        if len(self.results) > 0:
            forecasting_results = pd.DataFrame(self.results).T
            print("\nForecasting Performance Summary:")
            print(forecasting_results)
            return forecasting_results
        return None
    
    def evaluate_anomaly_detection(self, X_test, X_test_pred, X_train, X_train_pred, 
                                 data_processor, model_name="Autoencoder"):
        """Evaluate anomaly detection performance"""
        try:
            print(f"\n--- {model_name} Anomaly Detection Performance ---")
            
            # Calculate reconstruction errors
            mse_reconstruction = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))
            mse_train_reconstruction = np.mean(np.power(X_train - X_train_pred, 2), axis=(1, 2))
            
            # Determine anomaly threshold
            threshold = np.percentile(mse_train_reconstruction, ANOMALY_THRESHOLD_PERCENTILE)
            print(f"Anomaly Detection Threshold ({ANOMALY_THRESHOLD_PERCENTILE}th percentile of training MSE): {threshold:.4f}")
            
            # Detect anomalies
            anomalies = mse_reconstruction > threshold
            anomaly_indices = np.where(anomalies)[0]
            print(f"Number of anomalies detected in test set: {len(anomaly_indices)}")
            
            # Safely map anomaly indices to original data timestamps
            try:
                # Calculate the correct starting index for test sequences
                total_sequences = len(X_train) + len(X_test)
                start_index_of_sequences = len(data_processor.scaled_df_hourly) - total_sequences
                train_size_ad = len(X_train)
                
                # Ensure indices are within bounds
                max_valid_index = len(data_processor.scaled_df_hourly) - 1
                valid_anomaly_indices = []
                valid_datetimes = []
                
                for idx in anomaly_indices:
                    calculated_index = start_index_of_sequences + train_size_ad + idx
                    if 0 <= calculated_index <= max_valid_index:
                        valid_anomaly_indices.append(idx)
                        valid_datetimes.append(data_processor.scaled_df_hourly.index[calculated_index])
                
                if valid_datetimes:
                    print(f"First {min(10, len(valid_datetimes))} detected anomalous periods (start times of sequences):")
                    for idx_dt in valid_datetimes[:10]:
                        print(idx_dt.strftime("%Y-%m-%d %H:%M"))
                else:
                    print("No valid anomaly datetimes found within bounds.")
                    valid_datetimes = []
                    
            except Exception as e:
                print(f"Warning: Could not map anomaly indices to datetimes: {str(e)}")
                valid_datetimes = []
                valid_anomaly_indices = anomaly_indices
            
            anomaly_result = {
                'mse_reconstruction': mse_reconstruction,
                'threshold': threshold,
                'anomaly_indices': np.array(valid_anomaly_indices),
                'anomaly_datetimes': valid_datetimes,
                'num_anomalies': len(valid_anomaly_indices)
            }
            
            self.anomaly_models[model_name] = anomaly_result
            return anomaly_result
            
        except Exception as e:
            print(f"Error in anomaly detection evaluation: {str(e)}")
            # Return a default result to prevent crashes
            return {
                'mse_reconstruction': np.array([]),
                'threshold': 0.0,
                'anomaly_indices': np.array([]),
                'anomaly_datetimes': [],
                'num_anomalies': 0
            }
    
    def plot_anomaly_detection(self, mse_reconstruction, threshold, anomaly_indices, model_name="Autoencoder"):
        """Plot anomaly detection results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            
            # Plot 1: Reconstruction error over time
            axes[0, 0].plot(mse_reconstruction, label='Reconstruction Error (Test Data)', alpha=0.8, linewidth=1)
            axes[0, 0].axhline(y=threshold, color='r', linestyle='--', linewidth=2,
                               label=f'Anomaly Threshold ({threshold:.4f})')
            axes[0, 0].scatter(anomaly_indices, mse_reconstruction[anomaly_indices], 
                               color='red', s=50, zorder=5, label='Anomaly Detected')
            axes[0, 0].set_title(f'Anomaly Detection using {model_name}')
            axes[0, 0].set_xlabel('Sequence Index in Test Set')
            axes[0, 0].set_ylabel('Mean Squared Reconstruction Error')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Error distribution
            axes[0, 1].hist(mse_reconstruction, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                               label=f'Threshold ({threshold:.4f})')
            axes[0, 1].set_xlabel('Reconstruction Error')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Reconstruction Error Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Anomaly statistics
            normal_count = len(mse_reconstruction) - len(anomaly_indices)
            anomaly_count = len(anomaly_indices)
            
            labels = ['Normal', 'Anomaly']
            sizes = [normal_count, anomaly_count]
            colors = ['lightgreen', 'red']
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Anomaly Detection Results')
            
            # Plot 4: Error statistics
            error_stats = {
                'Mean Error': np.mean(mse_reconstruction),
                'Std Error': np.std(mse_reconstruction),
                'Min Error': np.min(mse_reconstruction),
                'Max Error': np.max(mse_reconstruction),
                'Threshold': threshold
            }
            
            stat_names = list(error_stats.keys())
            stat_values = list(error_stats.values())
            
            bars = axes[1, 1].bar(stat_names, stat_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'red'])
            axes[1, 1].set_title('Error Statistics')
            axes[1, 1].set_ylabel('Error Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, stat_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in anomaly detection plot: {str(e)}")
    
    def plot_anomaly_example(self, data_processor, anomaly_result, sample_index=0):
        """Plot an example of detected anomalous data"""
        try:
            if len(anomaly_result['anomaly_indices']) == 0:
                print("No anomalies detected to plot.")
                return
                
            if sample_index >= len(anomaly_result['anomaly_indices']):
                print(f"Sample index {sample_index} is out of range. Using index 0.")
                sample_index = 0
                
            anomaly_idx = anomaly_result['anomaly_indices'][sample_index]
            
            # Safely calculate the original data index
            try:
                start_index_of_sequences = len(data_processor.scaled_df_hourly) - len(anomaly_result['mse_reconstruction'])
                train_size_ad = int(len(anomaly_result['mse_reconstruction']) / (1 - TRAIN_SPLIT) * TRAIN_SPLIT)
                
                first_anomaly_sequence_start_idx = start_index_of_sequences + train_size_ad + anomaly_idx
                
                # Ensure the index is within bounds
                if first_anomaly_sequence_start_idx < 0 or first_anomaly_sequence_start_idx >= len(data_processor.df_hourly):
                    print(f"Calculated index {first_anomaly_sequence_start_idx} is out of bounds. Skipping anomaly example plot.")
                    return
                
                # Ensure we don't exceed the dataframe bounds
                end_idx = min(first_anomaly_sequence_start_idx + SEQUENCE_LENGTH, len(data_processor.df_hourly))
                original_data_segment = data_processor.df_hourly.iloc[
                    first_anomaly_sequence_start_idx : end_idx
                ]['active_power']

                plt.figure(figsize=(12, 5))
                plt.plot(original_data_segment.index, original_data_segment.values, 
                        label='Original Active Power during Anomaly', linewidth=2, color='red')
                plt.title(f'Original Data during Detected Anomaly (Starting {original_data_segment.index[0].strftime("%Y-%m-%d %H:%M")})')
                plt.xlabel('Time')
                plt.ylabel('Global Active Power (kW)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()
                
            except Exception as e:
                print(f"Error plotting anomaly example: {str(e)}")
                
        except Exception as e:
            print(f"Error in plot_anomaly_example: {str(e)}")
    
    def recommend_best_models(self):
        """Provide comprehensive model recommendations"""
        try:
            print("\n" + "="*80)
            print("MODEL RECOMMENDATION ANALYSIS")
            print("="*80)
            
            if len(self.results) == 0:
                print("No forecasting models available for recommendation")
                return None, []
            
            # Get sorted models by performance
            try:
                sorted_models, sorted_scores = self.plot_model_performance_comparison()
            except Exception as e:
                print(f"Warning: Could not generate performance comparison plots: {str(e)}")
                # Fallback: manually calculate rankings
                models = list(self.results.keys())
                composite_scores = []
                for model in models:
                    metrics = self.results[model]
                    # Simple composite score
                    composite_score = (1/metrics['RMSE'] + 1/metrics['MAE'] + metrics['R2']) / 3
                    composite_scores.append(composite_score)
                
                # Sort by composite score
                sorted_indices = np.argsort(composite_scores)[::-1]
                sorted_models = [models[i] for i in sorted_indices]
                sorted_scores = [composite_scores[i] for i in sorted_indices]
            
            print(f"\n🏆 MODEL RANKING (Best to Worst):")
            for i, (model, score) in enumerate(zip(sorted_models, sorted_scores)):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                print(f"{rank_emoji} {model}: {score:.3f}")
            
            # Detailed analysis
            if len(sorted_models) > 0:
                best_model = sorted_models[0]
                best_metrics = self.results[best_model]
                
                print(f"\n📊 DETAILED ANALYSIS OF BEST MODEL: {best_model}")
                print(f"   RMSE: {best_metrics['RMSE']:.4f}")
                print(f"   MAE: {best_metrics['MAE']:.4f}")
                print(f"   R-squared: {best_metrics['R2']:.4f}")
                
                # Model comparison
                print(f"\n📈 MODEL COMPARISON:")
                for model in sorted_models:
                    metrics = self.results[model]
                    print(f"   {model}:")
                    print(f"     - RMSE: {metrics['RMSE']:.4f}")
                    print(f"     - MAE: {metrics['MAE']:.4f}")
                    print(f"     - R-squared: {metrics['R2']:.4f}")
                
                # Recommendations
                print(f"\n💡 RECOMMENDATIONS:")
                print(f"1. 🎯 PRIMARY MODEL: Use {best_model} for production deployment")
                if len(sorted_models) > 1:
                    print(f"2. 🔄 ENSEMBLE: Consider combining {sorted_models[0]} and {sorted_models[1]} for improved robustness")
                print(f"3. 📊 MONITORING: Track RMSE and MAE metrics regularly")
                print(f"4. 🔧 OPTIMIZATION: Fine-tune hyperparameters for {best_model} to improve R-squared")
                
                # Model characteristics
                print(f"\n🔍 MODEL CHARACTERISTICS:")
                for model in sorted_models:
                    if "Transformer" in model:
                        print(f"   {model}: Excellent for capturing long-term dependencies and complex patterns")
                    elif "CNN-LSTM" in model:
                        print(f"   {model}: Good for spatial-temporal feature extraction")
                    elif "Autoencoder" in model:
                        print(f"   {model}: Specialized for anomaly detection")
                
                return best_model, sorted_models
            else:
                print("No models available for analysis")
                return None, []
                
        except Exception as e:
            print(f"Error in model recommendations: {str(e)}")
            return None, []
    
    def generate_final_report(self, forecasting_results, anomaly_result):
        """Generate a comprehensive final report"""
        try:
            print("\n" + "="*80)
            print("ENERGY FORECASTING AND ANOMALY DETECTION PROJECT - FINAL REPORT")
            print("="*80)
            
            # Get model recommendations
            try:
                best_model, all_models = self.recommend_best_models()
            except Exception as e:
                print(f"Warning: Could not generate model recommendations: {str(e)}")
                best_model, all_models = None, []
            
            # Forecasting Results
            print("\n--- FORECASTING PERFORMANCE SUMMARY ---")
            if forecasting_results is not None:
                print(forecasting_results)
                
                try:
                    best_model_name = forecasting_results.sort_values(by='RMSE').index[0]
                    best_rmse = forecasting_results.loc[best_model_name, 'RMSE']
                    best_mae = forecasting_results.loc[best_model_name, 'MAE']
                    best_r2 = forecasting_results.loc[best_model_name, 'R2']
                    
                    print(f"\n🏆 BEST FORECASTING MODEL: {best_model_name}")
                    print(f"   RMSE: {best_rmse:.4f}")
                    print(f"   MAE: {best_mae:.4f}")
                    print(f"   R-squared: {best_r2:.4f}")
                except Exception as e:
                    print(f"Warning: Could not determine best model: {str(e)}")
            
            # Anomaly Detection Results
            print("\n--- ANOMALY DETECTION SUMMARY ---")
            if anomaly_result and anomaly_result.get('num_anomalies', 0) > 0:
                print(f"🔍 Total Anomalies Detected: {anomaly_result['num_anomalies']}")
                print(f"📊 Anomaly Threshold: {anomaly_result['threshold']:.4f}")
                print(f"📈 Detection Method: Autoencoder-based reconstruction error")
            else:
                print("⚠️ No anomalies detected or anomaly detection failed")
            
            # Project Conclusion
            print("\n--- PROJECT CONCLUSION ---")
            print("✅ This project successfully demonstrates an AI-driven system for both energy forecasting and anomaly detection.")
            print("✅ Multiple deep learning architectures were implemented and compared.")
            print("✅ The system can identify unusual energy consumption patterns.")
            print("✅ Results provide insights for energy management and optimization.")
            
            print("\n--- FUTURE WORK CONSIDERATIONS ---")
            print("1. Hyperparameter optimization for improved performance")
            print("2. Multivariate forecasting (multiple energy features)")
            print("3. Real-time deployment and monitoring")
            print("4. Integration with smart grid systems")
            print("5. Advanced interpretability techniques")
            
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"Error generating final report: {str(e)}")
            print("Basic project completion message:")
            print("✅ Energy Forecasting and Anomaly Detection Project completed!")
            print("📊 Check the generated visualizations and analysis above.")

    def plot_individual_model_performance(self, model_name):
        """Create detailed performance analysis for a specific model"""
        if model_name not in self.forecasting_models:
            print(f"No data available for {model_name}")
            return
            
        model_data = self.forecasting_models[model_name]
        y_true = model_data['y_true']
        y_pred = model_data['y_pred']
        metrics = self.results[model_name]
        
        # Flatten data for plotting
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: Time series comparison
        sample_size = min(FORECAST_HORIZON * 20, len(y_true_flat))
        time_points = range(sample_size)
        
        axes[0, 0].plot(time_points, y_true_flat[:sample_size], 
                        label='Actual Values', color='blue', linewidth=2, alpha=0.8)
        axes[0, 0].plot(time_points, y_pred_flat[:sample_size], 
                        label='Predicted Values', color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title(f'{model_name} - Time Series Prediction')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Power Consumption (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[0, 1].scatter(y_true_flat, y_pred_flat, alpha=0.6, color='green')
        axes[0, 1].plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title(f'{model_name} - Prediction Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residual plot
        residuals = y_true_flat - y_pred_flat
        axes[0, 2].scatter(y_pred_flat, residuals, alpha=0.6, color='orange')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 2].set_xlabel('Predicted Values')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title(f'{model_name} - Residual Analysis')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{model_name} - Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Metrics comparison
        metric_names = ['RMSE', 'MAE', 'R²']
        metric_values = [metrics['RMSE'], metrics['MAE'], metrics['R2']]
        colors = ['red', 'orange', 'green']
        
        bars = axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[1, 1].set_title(f'{model_name} - Performance Metrics')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 6: Performance summary
        axes[1, 2].axis('off')
        summary_text = f"""
{model_name} Performance Summary

RMSE: {metrics['RMSE']:.4f}
MAE: {metrics['MAE']:.4f}
R-squared: {metrics['R2']:.4f}

Total Predictions: {len(y_true_flat)}
Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}
Standard Deviation of Errors: {np.std(residuals):.4f}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 {model_name} Performance Analysis:")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE: {metrics['MAE']:.4f}")
        print(f"   R-squared: {metrics['R2']:.4f}")
        print(f"   Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
        print(f"   Error Standard Deviation: {np.std(residuals):.4f}")

    def plot_all_models_individual_analysis(self):
        """Create individual performance analysis for all models"""
        print("\n📊 GENERATING INDIVIDUAL MODEL ANALYSIS")
        for model_name in self.forecasting_models.keys():
            print(f"\n--- Analyzing {model_name} ---")
            self.plot_individual_model_performance(model_name) 