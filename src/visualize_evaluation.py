"""
Model Evaluation Visualization
Generates a Residual Plot and Error Distribution to explain how 
we measure the accuracy of the solar predictor.
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

def visualize_evaluation_logic():
    plt.style.use('ggplot')
    
    # Robust root finding
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    _safe_print("Generating evaluation simulation...")
    
    # Simulate Evaluation (Actual vs Predicted from Test Set)
    np.random.seed(42)
    actual = np.random.uniform(5000, 25000, 500)
    # Simulate a high-performing model (XGBoost)
    # The "Error" (Residual) is what we evaluate
    residuals = np.random.normal(0, 400, 500) 
    predicted = actual + residuals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Residual Plot (The standard for evaluation)
    ax1.scatter(predicted, residuals, alpha=0.5, color='#2c3e50', edgecolors='white')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_title("Residual Plot (Error vs Prediction)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Predicted Output (kW)")
    ax1.set_ylabel("Error (Actual - Predicted)")
    
    # 2. Error Distribution (Quality Check)
    ax2.hist(residuals, bins=30, color='#27ae60', alpha=0.7, edgecolor='black')
    ax2.set_title("Distribution of Errors", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Error Magnitude (kW)")
    ax2.set_ylabel("Frequency")

    plt.suptitle("How We Evaluate: Analyzing 'The Mistake' (The Residual)", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save output
    output_path = 'outputs/charts/model_evaluation_intuition.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    _safe_print(f"Success! Evaluation intuition saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_evaluation_logic()
