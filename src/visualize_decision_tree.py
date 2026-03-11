"""
Decision Tree Visualization Script
Generates a graphical representation of the logical "checks" (splits) 
that a single tree in the Random Forest performs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

def visualize_tree():
    # Set professional style
    plt.style.use('default') # Classic style often better for tree diagrams
    
    # Robust root finding
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    _safe_print("📊 Processing: Loading data components...")
    try:
        import sys
        if project_root not in sys.path:
            sys.path.append(project_root)
        from src.data_loader import load_and_process_data
        
        df = load_and_process_data(data_dir='data')
        # Using the same features as model_trainer
        feature_names = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
        X = df[feature_names]
        y = df['AC_POWER']
    except Exception as e:
        _safe_print(f"Warning: Could not load data: {e}. Generating synthetic demo.")
        feature_names = ['Temp', 'Mod_Temp', 'Irradiation']
        X = pd.DataFrame(np.random.rand(100, 3), columns=feature_names)
        y = X['Irradiation'] * 30000 + np.random.normal(0, 1000, 100)

    # 1. Train a Shallow Tree
    # We use a small depth (max_depth=3) because a real tree (depth 15) is unreadable in a graph
    _safe_print("🧠 Logic: Training an explainable decision tree (depth=3)...")
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X, y)

    # 2. Plotting
    _safe_print("🎨 Rendering: Creating high-res tree diagram...")
    plt.figure(figsize=(20, 10), dpi=300)
    
    plot_tree(
        dt, 
        feature_names=feature_names, 
        filled=True, 
        rounded=True, 
        fontsize=10,
        precision=2
    )
    
    plt.title("Internal Logic: Decision Tree Splits (Simplified for Visualization)", fontsize=22, fontweight='bold', pad=20)
    
    # Add a footer caption
    plt.figtext(0.5, 0.02, 
                "Higher Irradiation values (Right paths) lead to higher predicted AC Power (Darker nodes).\n"
                "This single tree represents 1 of 100 experts in your Random Forest.", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Save output
    output_dir = 'outputs/charts'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'decision_tree_visualization.png')
    plt.savefig(save_path, bbox_inches='tight')
    _safe_print(f"🚀 Success! Tree diagram saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_tree()
