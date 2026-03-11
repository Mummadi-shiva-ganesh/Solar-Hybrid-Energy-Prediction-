"""
Linear Regression Intuition Visualization
Plots the relationship between Irradiation and AC Power with the fitted regression line.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

def visualize_math_intuition():
    plt.style.use('ggplot')
    
    # Robustly find project root (one level up from src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    _safe_print("Extraction: Loading solar data...")
    try:
        import sys
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        from src.data_loader import load_and_process_data
        df = load_and_process_data(data_dir='data')
        
        # We focus on the strongest linear relationship: Irradiation vs AC Power
        X = df[['IRRADIATION']].values
        y = df['AC_POWER'].values
    except Exception as e:
        _safe_print(f"Warning: Could not load data: {e}. Generating synthetic demo.")
        X = np.random.uniform(0, 1.2, 500).reshape(-1, 1)
        y = 30000 * X + np.random.normal(0, 1000, X.shape)
        y = np.maximum(y, 0) # No negative power

    # Fit a simple 2D line for intuition
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    _safe_print(f"✅ Calculation: Slope (w1) = {slope:.2f}, Intercept (w0) = {intercept:.2f}")

    # Plotting
    plt.figure(figsize=(10, 7))
    
    # 1. Plot the actual observations
    plt.scatter(X, y, alpha=0.3, color='#3498db', s=20, label='Actual Data Points')
    
    # 2. Plot the Regression Line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_range)
    plt.plot(x_range, y_line, color='#e74c3c', linewidth=3, label='Fitted Line (Minimizes Error)')
    
    # Annotations for Math Intuition
    plt.title('Baseline Intuition: Linear Relationship (Power vs Sunlight)', fontsize=15, pad=20, fontweight='bold')
    plt.xlabel('Solar Irradiation (W/m²)', fontsize=12)
    plt.ylabel('AC Power Output (kW)', fontsize=12)
    
    # Equation Annotation
    eq_text = f"Equation: y = {slope:.1f}x + ({intercept:.1f})\n"
    eq_text += "Where:\n"
    eq_text += f"- Slope: {slope:.1f} W per unit sunlight\n"
    eq_text += f"- Intercept: {intercept:.1f} (Base output)"
    
    plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save output
    output_dir = 'outputs/charts'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'linear_math_intuition.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    _safe_print(f"🚀 Success! Intuition graph saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_math_intuition()
