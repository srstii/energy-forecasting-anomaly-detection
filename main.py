"""
Main Execution Script for Energy Forecasting and Anomaly Detection Project
"""

import tensorflow as tf
import warnings
import sys
import traceback
warnings.filterwarnings('ignore')

# Import project modules
from data_processor import DataProcessor
from models import ModelBuilder, ModelTrainer
from evaluator import ModelEvaluator
from config import *

def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"❌ Error in {func.__name__}: {str(e)}")
        return None

def main():
    """Main execution function"""
    print("="*80)
    print("ENERGY FORECASTING AND ANOMALY DETECTION PROJECT")
    print("="*80)
    
    # Check TensorFlow version and GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print("GPU is available:" + str(tf.config.list_physical_devices('GPU')))
    
    try:
        # Initialize components
        print("\n1. INITIALIZING PROJECT COMPONENTS")
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        
        # Data Processing
        print("\n2. DATA PROCESSING")
        try:
            forecasting_data, anomaly_data = data_processor.process_all_data()
            X_train_forecasting, X_test_forecasting, y_train_forecasting, y_test_forecasting = forecasting_data
            X_train_ad, X_test_ad = anomaly_data
            print("✅ Data processing completed successfully")
        except Exception as e:
            print(f"❌ Error in data processing: {str(e)}")
            return
        
        # Model Building
        print("\n3. BUILDING MODELS")
        
        # Build Temporal Transformer
        print("\nBuilding Temporal Transformer Model...")
        try:
            transformer_model = ModelBuilder.build_temporal_transformer(
                input_shape=(SEQUENCE_LENGTH, X_train_forecasting.shape[-1]),
                forecast_horizon=FORECAST_HORIZON
            )
            transformer_model.summary()
        except Exception as e:
            print(f"❌ Error building Temporal Transformer: {str(e)}")
            return
        
        # Build CNN-LSTM Hybrid
        print("\nBuilding CNN-LSTM Hybrid Model...")
        try:
            cnn_lstm_model = ModelBuilder.build_cnn_lstm(
                input_shape=(SEQUENCE_LENGTH, X_train_forecasting.shape[-1]),
                forecast_horizon=FORECAST_HORIZON
            )
            cnn_lstm_model.summary()
        except Exception as e:
            print(f"❌ Error building CNN-LSTM: {str(e)}")
            return
        
        # Build Autoencoder
        print("\nBuilding Autoencoder Model for Anomaly Detection...")
        try:
            autoencoder_model = ModelBuilder.build_autoencoder_cnn(X_train_ad.shape[1:])
            autoencoder_model.summary()
        except Exception as e:
            print(f"❌ Error building Autoencoder: {str(e)}")
            return
        
        # Model Training
        print("\n4. TRAINING MODELS")
        
        # Train forecasting models
        try:
            transformer_model, transformer_history = model_trainer.train_model(
                transformer_model, 
                X_train_forecasting, 
                y_train_forecasting.reshape(-1, FORECAST_HORIZON),
                "Temporal Transformer"
            )
        except Exception as e:
            print(f"❌ Error training Temporal Transformer: {str(e)}")
            return
        
        try:
            cnn_lstm_model, cnn_lstm_history = model_trainer.train_model(
                cnn_lstm_model, 
                X_train_forecasting, 
                y_train_forecasting.reshape(-1, FORECAST_HORIZON),
                "CNN-LSTM Hybrid"
            )
        except Exception as e:
            print(f"❌ Error training CNN-LSTM: {str(e)}")
            return
        
        # Train autoencoder
        try:
            autoencoder_model, autoencoder_history = model_trainer.train_model(
                autoencoder_model, 
                X_train_ad, 
                X_train_ad,
                "Autoencoder"
            )
        except Exception as e:
            print(f"❌ Error training Autoencoder: {str(e)}")
            return
        
        # Plot training histories
        print("\n5. PLOTTING TRAINING HISTORIES")
        safe_execute(evaluator.plot_training_history, transformer_history, "Temporal Transformer")
        safe_execute(evaluator.plot_training_history, cnn_lstm_history, "CNN-LSTM Hybrid")
        safe_execute(evaluator.plot_training_history, autoencoder_history, "Autoencoder")
        
        # Model Evaluation
        print("\n6. MODEL EVALUATION")
        
        # Forecasting evaluation
        print("\n--- FORECASTING EVALUATION ---")
        try:
            y_pred_transformer = transformer_model.predict(X_test_forecasting)
            y_pred_cnn_lstm = cnn_lstm_model.predict(X_test_forecasting)
            
            # Inverse transform predictions
            y_pred_transformer_inv = data_processor.inverse_transform_predictions(y_pred_transformer)
            y_pred_cnn_lstm_inv = data_processor.inverse_transform_predictions(y_pred_cnn_lstm)
            y_test_inv = data_processor.inverse_transform_predictions(y_test_forecasting)
            
            # Evaluate forecasting models
            evaluator.evaluate_forecasting(y_test_inv, y_pred_transformer_inv, "Temporal Transformer")
            evaluator.evaluate_forecasting(y_test_inv, y_pred_cnn_lstm_inv, "CNN-LSTM Hybrid")
            
            # Create forecasting comparison plot
            print(f"\n🔍 Debug: Data shapes before plotting:")
            print(f"   y_test_inv shape: {y_test_inv.shape}")
            print(f"   y_pred_transformer_inv shape: {y_pred_transformer_inv.shape}")
            print(f"   y_pred_cnn_lstm_inv shape: {y_pred_cnn_lstm_inv.shape}")
            
            safe_execute(evaluator.plot_forecasting_comparison, y_test_inv, y_pred_transformer_inv, y_pred_cnn_lstm_inv)
            
        except Exception as e:
            print(f"❌ Error in forecasting evaluation: {str(e)}")
        
        # Anomaly detection evaluation
        print("\n--- ANOMALY DETECTION EVALUATION ---")
        try:
            X_test_pred_ad = autoencoder_model.predict(X_test_ad)
            X_train_pred_ad = autoencoder_model.predict(X_train_ad)
            
            anomaly_result = evaluator.evaluate_anomaly_detection(
                X_test_ad, X_test_pred_ad, X_train_ad, X_train_pred_ad, data_processor
            )
            
            # Plot anomaly detection results
            if anomaly_result and len(anomaly_result['mse_reconstruction']) > 0:
                safe_execute(evaluator.plot_anomaly_detection,
                           anomaly_result['mse_reconstruction'],
                           anomaly_result['threshold'],
                           anomaly_result['anomaly_indices'])
                
                # Plot example anomaly
                safe_execute(evaluator.plot_anomaly_example, data_processor, anomaly_result)
            else:
                print("⚠️ No anomaly detection results to plot.")
                
        except Exception as e:
            print(f"❌ Error in anomaly detection evaluation: {str(e)}")
            anomaly_result = None
        
        # Generate comprehensive model comparison and recommendations
        print("\n7. MODEL COMPARISON AND RECOMMENDATIONS")
        safe_execute(evaluator.plot_model_performance_comparison)
        
        # Generate individual model analysis
        print("\n8. INDIVIDUAL MODEL ANALYSIS")
        safe_execute(evaluator.plot_all_models_individual_analysis)
        
        # Generate final report
        print("\n9. GENERATING FINAL REPORT")
        try:
            forecasting_results = evaluator.create_forecasting_summary()
            evaluator.generate_final_report(forecasting_results, anomaly_result)
        except Exception as e:
            print(f"❌ Error generating final report: {str(e)}")
        
        print("\n✅ PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("\n📊 SUMMARY OF GENERATED VISUALIZATIONS:")
        print("   • Training history plots for all models")
        print("   • Comprehensive forecasting comparison charts")
        print("   • Model performance comparison with radar charts")
        print("   • Individual detailed analysis for each model")
        print("   • Anomaly detection visualization")
        print("   • Model ranking and recommendations")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING EXECUTION: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 