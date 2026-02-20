"""
Comprehensive Model Training Pipeline
Implements multiple ML/DL algorithms: Linear Regression (baseline), SVR, Random Forest,
XGBoost, and optional LSTM. Includes Grid Search, logging, and chart generation.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Safe print for Windows console (avoids UnicodeEncodeError with emojis)
def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

# Optional: LSTM (Deep Learning)
LSTM_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    pass


class ModelTrainer:
    """Handles training and evaluation of multiple ML/DL models"""
    
    def __init__(self, data_path='data/solar_data.csv', log_dir='logs', chart_dir='outputs/charts'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.log_dir = log_dir
        self.chart_dir = chart_dir
        self._log_file = None
        self._lstm_lookback = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess dataset using data_loader logic"""
        _safe_print("📊 Loading dataset using data_loader...")
        
        try:
            # Add src to path to import data_loader
            import sys
            import os
            current_dir = os.getcwd()
            if 'src' not in sys.path:
                sys.path.append(os.path.join(current_dir, 'src'))
            
            from data_loader import load_and_process_data
            
            # Load data using existing loader
            df = load_and_process_data(data_dir='data')
            
            # Define specific features and target for Solar Prediction
            feature_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
            target_col = 'AC_POWER'
            
            # Verify columns exist
            missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
            if missing_cols:
                _safe_print(f"⚠️ Warning: Missing columns {missing_cols}")
                # Fallback to simple CSV load if loader fails to return expected structure
                # This handles cases where data_loader might return different structure
                df = pd.read_csv('data/solar_generation.csv')
                # Basic processing
                if 'DATE_TIME' in df.columns:
                     df = df.groupby('DATE_TIME').sum().reset_index() # simple aggregation
            
            # Final check
            if target_col not in df.columns:
                 target_col = df.columns[-1] # Fallback
            
            _safe_print(f"Target variable: {target_col}")
            _safe_print(f"Features: {feature_cols}")
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.feature_names = feature_cols
            
            _safe_print(f"✅ Data preprocessing complete")
            _safe_print(f"Training set: {X_train_scaled.shape}")
            _safe_print(f"Validation set: {X_val_scaled.shape}")
            _safe_print(f"Test set: {X_test_scaled.shape}")
            
            return (X_train_scaled, X_val_scaled, X_test_scaled, 
                    y_train, y_val, y_test)
            
        except Exception as e:
            _safe_print(f"❌ Error in data loading: {str(e)}")
            raise e
    
    def _log(self, msg):
        """Print and optionally write to log file."""
        _safe_print(msg)
        if self._log_file:
            self._log_file.write(msg + '\n')
            self._log_file.flush()
    
    def _start_logging(self):
        """Start writing to a log file in logs/."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._log_file = open(log_path, 'w', encoding='utf-8')
        self._log(f"Training log started at {datetime.now().isoformat()}")
        return log_path
    
    def _stop_logging(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None
    
    def train_baseline_model(self, X_train, X_val, y_train, y_val):
        """Train baseline Linear Regression model"""
        _safe_print("\n" + "="*60)
        _safe_print("🔵 Training Baseline Model: Linear Regression")
        _safe_print("="*60)
        
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        results = {
            'model': model,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time
        }
        
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = results
        
        for line in [f"✅ Training R²: {results['train_r2']:.4f}", f"✅ Validation R²: {results['val_r2']:.4f}", f"⏱️  Training time: {train_time:.2f}s"]:
            _safe_print(line); self._log(line)
        
        return results
    
    def train_random_forest(self, X_train, X_val, y_train, y_val):
        """Train Random Forest model with hyperparameter tuning"""
        _safe_print("\n" + "="*60)
        _safe_print("🌲 Training Random Forest Regressor")
        _safe_print("="*60)
        
        start_time = time.time()
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        _safe_print("🔍 Performing Grid Search...")
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='r2', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_time = time.time() - start_time
        
        _safe_print(f"✅ Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Metrics
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time,
            'feature_importance': dict(zip(self.feature_names, 
                                          best_model.feature_importances_))
        }
        
        self.models['Random Forest'] = best_model
        self.results['Random Forest'] = results
        
        self._log(f"✅ Best params: {grid_search.best_params_}")
        self._log(f"✅ Val R²: {results['val_r2']:.4f} | Val RMSE: {results['val_rmse']:.4f} | Time: {train_time:.2f}s")
        _safe_print(f"✅ Training R²: {results['train_r2']:.4f}")
        _safe_print(f"✅ Validation R²: {results['val_r2']:.4f}")
        _safe_print(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost model with hyperparameter tuning"""
        _safe_print("\n" + "="*60)
        _safe_print("🚀 Training XGBoost Regressor")
        _safe_print("="*60)
        
        start_time = time.time()
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        _safe_print("🔍 Performing Grid Search...")
        xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_time = time.time() - start_time
        
        _safe_print(f"✅ Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Metrics
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time,
            'feature_importance': dict(zip(self.feature_names,
                                          best_model.feature_importances_))
        }
        
        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = results
        
        self._log(f"✅ Best params: {grid_search.best_params_}")
        self._log(f"✅ Val R²: {results['val_r2']:.4f} | Val RMSE: {results['val_rmse']:.4f} | Time: {train_time:.2f}s")
        _safe_print(f"✅ Training R²: {results['train_r2']:.4f}")
        _safe_print(f"✅ Validation R²: {results['val_r2']:.4f}")
        _safe_print(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def train_svr(self, X_train, X_val, y_train, y_val):
        """Train Support Vector Regressor with Grid Search (SVM for regression)."""
        self._log("\n" + "="*60)
        self._log("🔷 Training Support Vector Regressor (SVR)")
        self._log("="*60)
        
        start_time = time.time()
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf', 'linear'],
            'epsilon': [0.01, 0.1]
        }
        
        self._log("🔍 Performing Grid Search...")
        svr = SVR()
        grid_search = GridSearchCV(
            svr, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_time = time.time() - start_time
        
        self._log(f"✅ Best parameters: {grid_search.best_params_}")
        
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time,
        }
        
        self.models['SVR'] = best_model
        self.results['SVR'] = results
        
        self._log(f"✅ Training R²: {results['train_r2']:.4f}")
        self._log(f"✅ Validation R²: {results['val_r2']:.4f}")
        self._log(f"✅ Validation RMSE: {results['val_rmse']:.4f}")
        self._log(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def _create_sequences(self, X, y, lookback=10):
        """Create sequences for LSTM: (samples, lookback, features) -> target."""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm(self, X_train, X_val, X_test, y_train, y_val, y_test, lookback=10, epochs=50):
        """Train LSTM (Deep Learning) on sequential windows. Optional if TensorFlow not installed."""
        if not LSTM_AVAILABLE:
            self._log("\n⚠️ LSTM skipped: TensorFlow not installed. Uncomment in requirements.txt to enable.")
            return None
        
        self._log("\n" + "="*60)
        self._log("🧠 Training LSTM (Deep Learning)")
        self._log("="*60)
        
        X_tr, y_tr = self._create_sequences(
            X_train, pd.Series(y_train.reset_index(drop=True)), lookback
        )
        X_v, y_v = self._create_sequences(
            X_val, pd.Series(y_val.reset_index(drop=True)), lookback
        )
        
        if len(X_tr) < 10 or len(X_v) < 5:
            self._log("⚠️ LSTM skipped: insufficient samples after sequence creation.")
            return None
        
        start_time = time.time()
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        history = model.fit(
            X_tr, y_tr, validation_data=(X_v, y_v),
            epochs=epochs, batch_size=32, callbacks=[early], verbose=0
        )
        
        train_time = time.time() - start_time
        
        y_train_flat = y_tr  # already 1D
        y_val_flat = y_v
        y_train_pred = model.predict(X_tr, verbose=0).flatten()
        y_val_pred = model.predict(X_v, verbose=0).flatten()
        
        results = {
            'model': model,
            'train_r2': r2_score(y_train_flat, y_train_pred),
            'val_r2': r2_score(y_val_flat, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_flat, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val_flat, y_val_pred)),
            'train_mae': mean_absolute_error(y_train_flat, y_train_pred),
            'val_mae': mean_absolute_error(y_val_flat, y_val_pred),
            'train_time': train_time,
            'history': history.history,
            'lookback': lookback,
        }
        
        self.models['LSTM'] = model
        self.results['LSTM'] = results
        
        self._log(f"✅ Training R²: {results['train_r2']:.4f}")
        self._log(f"✅ Validation R²: {results['val_r2']:.4f}")
        self._log(f"✅ Validation RMSE: {results['val_rmse']:.4f}")
        self._log(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def compare_models(self):
        """Compare all trained models and log comparison table."""
        self._log("\n" + "="*60)
        self._log("📊 MODEL COMPARISON")
        self._log("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train R²': [self.results[m]['train_r2'] for m in self.results],
            'Val R²': [self.results[m]['val_r2'] for m in self.results],
            'Val RMSE': [self.results[m]['val_rmse'] for m in self.results],
            'Val MAE': [self.results[m]['val_mae'] for m in self.results],
            'Train Time (s)': [round(self.results[m]['train_time'], 2) for m in self.results]
        })
        
        comparison_df = comparison_df.sort_values('Val R²', ascending=False)
        tbl = comparison_df.to_string(index=False)
        _safe_print(tbl)
        self._log(tbl)
        
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self._lstm_lookback = self.results.get('LSTM', {}).get('lookback')
        
        self._log(f"\n🏆 Best Model: {best_model_name}")
        self._log(f"   Validation R²: {self.results[best_model_name]['val_r2']:.4f}")
        _safe_print(f"\n🏆 Best Model: {best_model_name}")
        _safe_print(f"   Validation R²: {self.results[best_model_name]['val_r2']:.4f}")
        
        return comparison_df
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate best model on test set. Handles LSTM (sequence input)."""
        self._log("\n" + "="*60)
        self._log("🎯 FINAL TEST SET EVALUATION")
        self._log("="*60)
        
        if self.best_model_name == 'LSTM' and getattr(self, '_lstm_lookback', None) is not None:
            X_test_eval, y_test_eval = self._create_sequences(
                X_test, pd.Series(y_test.reset_index(drop=True)), self._lstm_lookback
            )
            y_pred = self.best_model.predict(X_test_eval, verbose=0).flatten()
        else:
            X_test_eval, y_test_eval = X_test, y_test
            y_pred = self.best_model.predict(X_test_eval)
        
        test_r2 = r2_score(y_test_eval, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
        test_mae = mean_absolute_error(y_test_eval, y_pred)
        safe_ratio = np.where(np.abs(y_test_eval) > 1e-6, (y_test_eval - y_pred) / y_test_eval, 0)
        test_mape = np.mean(np.abs(safe_ratio)) * 100
        
        for line in [
            f"Model: {self.best_model_name}",
            f"Test R²: {test_r2:.4f}",
            f"Test RMSE: {test_rmse:.4f}",
            f"Test MAE: {test_mae:.4f}",
            f"Test MAPE: {test_mape:.2f}%"
        ]:
            _safe_print(line)
            self._log(line)
        
        return {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'y_test': y_test_eval,
            'y_pred': y_pred
        }
    
    def save_models(self):
        """Save best model and preprocessor"""
        _safe_print("\n💾 Saving models...")
        
        # Save best model (joblib for sklearn/XGB; Keras .save for LSTM)
        if self.best_model_name == 'LSTM':
            self.best_model.save('models/best_model_lstm.keras')
        else:
            joblib.dump(self.best_model, 'models/best_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'trained_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'performance': {
                'val_r2': float(self.results[self.best_model_name]['val_r2']),
                'val_rmse': float(self.results[self.best_model_name]['val_rmse']),
                'val_mae': float(self.results[self.best_model_name]['val_mae'])
            }
        }
        
        if 'best_params' in self.results[self.best_model_name]:
            metadata['hyperparameters'] = self.results[self.best_model_name]['best_params']
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        _safe_print("✅ Models saved to 'models/' directory")
        _safe_print("   - best_model.pkl")
        _safe_print("   - scaler.pkl")
        _safe_print("   - model_metadata.json")
    
    def generate_charts(self, comparison_df, test_results):
        """Generate charts for Model Training & Validation (Section 6.5)."""
        os.makedirs(self.chart_dir, exist_ok=True)
        
        # 1. Model comparison bar chart (Val R², Val RMSE)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        models = comparison_df['Model'].tolist()
        x = np.arange(len(models))
        w = 0.35
        
        axes[0].bar(x - w/2, comparison_df['Val R²'], w, label='Val R²', color='steelblue')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=25, ha='right')
        axes[0].set_ylabel('R²')
        axes[0].set_title('Validation R² by Model')
        axes[0].legend()
        axes[0].set_ylim(0, 1.05)
        
        axes[1].bar(x - w/2, comparison_df['Val RMSE'], w, label='Val RMSE', color='coral')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=25, ha='right')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Validation RMSE by Model')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        self._log(f"📈 Saved {self.chart_dir}/model_comparison.png")
        
        # 2. Prediction vs Actual (best model) – use test set from last evaluation
        if test_results and 'y_test' in test_results and 'y_pred' in test_results:
            y_test = test_results['y_test']
            y_pred = test_results['y_pred']
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred, alpha=0.5, s=15)
            mn = min(y_test.min(), y_pred.min())
            mx = max(y_test.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')
            ax.set_xlabel('Actual AC Power')
            ax.set_ylabel('Predicted AC Power')
            ax.set_title(f'Prediction vs Actual ({self.best_model_name})')
            ax.legend()
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(self.chart_dir, 'prediction_vs_actual.png'), dpi=150, bbox_inches='tight')
            plt.close()
            self._log(f"📈 Saved {self.chart_dir}/prediction_vs_actual.png")
        
        # 3. Performance table saved as CSV for report
        comparison_path = os.path.join(self.chart_dir, 'model_comparison_table.csv')
        comparison_df.to_csv(comparison_path, index=False)
        self._log(f"📄 Saved {comparison_path}")
    
    def train_all(self):
        """Run complete training pipeline: Baseline, SVR, RF, XGBoost, optional LSTM; logs and charts."""
        log_path = self._start_logging()
        _safe_print(f"Logging to {log_path}")
        
        _safe_print("\n" + "="*80)
        _safe_print("🚀 STARTING COMPREHENSIVE MODEL TRAINING PIPELINE")
        _safe_print("="*80)
        self._log("STARTING COMPREHENSIVE MODEL TRAINING PIPELINE")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # 6.2 Baseline
        self.train_baseline_model(X_train, X_val, y_train, y_val)
        # 6.3 ML/DL: SVR, Random Forest, XGBoost, LSTM
        self.train_svr(X_train, X_val, y_train, y_val)
        self.train_random_forest(X_train, X_val, y_train, y_val)
        self.train_xgboost(X_train, X_val, y_train, y_val)
        self.train_lstm(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 6.6 Compare & 6.7 Select best
        comparison_df = self.compare_models()
        
        # Test set evaluation
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # 6.5 Charts and table
        self.generate_charts(comparison_df, test_results)
        
        # Save models
        self.save_models()
        
        self._stop_logging()
        _safe_print("\n" + "="*80)
        _safe_print("✅ TRAINING PIPELINE COMPLETE!")
        _safe_print("="*80)
        
        return comparison_df, test_results


if __name__ == "__main__":
    trainer = ModelTrainer()
    comparison_df, test_results = trainer.train_all()
