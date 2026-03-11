"""
XGBoost Intuition Visualization
Demonstrates the "Gradient Boosting" logic: how models are built sequentially 
to correct the errors (residuals) of previous steps.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

def visualize_xgboost_intuition():
    # Set professional style
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
        X = df[['IRRADIATION']].values[:500]
        y = df['AC_POWER'].values[:500]
    except Exception as e:
        _safe_print(f"Warning: Loading synthetic data instead: {e}")
        X = np.linspace(0, 1.2, 100).reshape(-1, 1)
        y = 30000 * np.sin(X * 1.5) + np.random.normal(0, 500, X.shape)

    # Simplified Gradient Boosting Simulation
    # Step 1: Initial Mean Prediction (Stage 0)
    y_pred0 = np.full_like(y, np.mean(y))
    residual1 = y - y_pred0
    
    # Step 2: Fit a small tree to the residuals (Stage 1)
    from sklearn.tree import DecisionTreeRegressor
    tree1 = DecisionTreeRegressor(max_depth=2)
    tree1.fit(X, residual1)
    y_pred1 = y_pred0 + 0.1 * tree1.predict(X) # Learning rate = 0.1
    residual2 = y - y_pred1
    
    # Step 3: Final XGBoost-like prediction (after many stages)
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(X, y)
        y_final = model.predict(X)
    except ImportError:
        # Fallback if xgboost is not installed
        rf = DecisionTreeRegressor(max_depth=5)
        rf.fit(X, y)
        y_final = rf.predict(X)

    # Plotting the "Correction" Logic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: The Residual Learning
    ax1.scatter(X, residual1, alpha=0.4, color='gray', s=15, label='Residuals (Errors) After Mean')
    x_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    ax1.plot(x_grid, 0.1 * tree1.predict(x_grid), color='red', linewidth=3, label='XGBoost Stage 1 Correction')
    ax1.set_title("How XGBoost Thinks: Correcting Errors", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Solar Irradiation")
    ax1.set_ylabel("Remaining Error")
    ax1.legend()

    # Right: Final Prediction vs Truth
    ax2.scatter(X, y, alpha=0.3, color='blue', s=15, label='Actual Data')
    ax2.plot(np.sort(X, axis=0), y_final[np.argsort(X, axis=0).flatten()], color='black', linewidth=3, label='Final XGBoost Ensemble')
    ax2.set_title("Final Outcome: Extremely Precise Fit", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Solar Irradiation")
    ax2.set_ylabel("Power Output")
    ax2.legend()

    plt.suptitle("XGBoost Intuition: The Art of Sequential Improvement", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save output
    output_path = 'outputs/charts/xgboost_intuition.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    _safe_print(f"Success! XGBoost Intuition saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_xgboost_intuition()
