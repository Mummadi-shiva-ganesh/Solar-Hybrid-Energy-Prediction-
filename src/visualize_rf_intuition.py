"""
Random Forest Intuition Visualization
Compares the Linear Baseline against the Random Forest "Flexible" fit.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

def visualize_rf_intuition():
    plt.style.use('ggplot')
    
    # Robust root finding
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    _safe_print("Loading data...")
    try:
        import sys
        if project_root not in sys.path:
            sys.path.append(project_root)
        from src.data_loader import load_and_process_data
        df = load_and_process_data(data_dir='data')
        X = df[['IRRADIATION']].values
        y = df['AC_POWER'].values
    except Exception as e:
        _safe_print(f"Warning: Loading synthetic data instead: {e}")
        X = np.random.uniform(0, 1.2, 300).reshape(-1, 1)
        y = 30000 * X**1.2 + np.random.normal(0, 1500, X.shape) # Added a slight curve
        y = np.maximum(y, 0)

    # 1. Fit Linear Regression (The "Stiff" Model)
    lr = LinearRegression()
    lr.fit(X, y)
    
    # 2. Fit Random Forest (The "Flexible" Model)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Generate points for smooth plotting
    x_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_lr = lr.predict(x_range)
    y_rf = rf.predict(x_range)

    # Plotting
    plt.figure(figsize=(11, 7))
    plt.scatter(X, y, alpha=0.3, color='#95a5a6', s=20, label='Actual Data')
    
    plt.plot(x_range, y_lr, color='#e74c3c', linewidth=2, linestyle='--', label='Baseline (Linear Regression)')
    plt.plot(x_range, y_rf, color='#27ae60', linewidth=4, label='Random Forest (Flexible/Adaptive)')
    
    plt.title('Random Forest Intuition: Adaptive Prediction', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Solar Irradiation (W/m²)', fontsize=12)
    plt.ylabel('Power Output (kW)', fontsize=12)
    
    # Intuition Box
    info_text = ("Why Random Forest is Better:\n"
                 "1. Nonlinear: Notice how green line 'curves' with the data.\n"
                 "2. Robust: It ignores noise and captures real trends.\n"
                 "3. Ensemble: It's an average of 100 different decision trees.")
    
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save output
    output_path = 'outputs/charts/rf_intuition_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    _safe_print(f"Success! RF Intuition saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_rf_intuition()
