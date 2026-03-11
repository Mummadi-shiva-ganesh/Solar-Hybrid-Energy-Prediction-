# ☀️🔋 Solar-Battery Hybrid Energy Predictor

A high-performance Machine Learning solution for forecasting solar power generation and simulating battery storage dynamics in real-time.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [How the System Works](#-how-the-system-works)
3. [Deep Dive: Machine Learning Logic](#-deep-dive-machine-learning-logic)
4. [Technology Stack](#-technology-stack)
5. [Database Strategy (PostgreSQL)](#-database-strategy-postgresql)
6. [Installation & Setup](#-installation--setup)

---

## 📌 Project Overview
The **Solar-Battery Hybrid Energy Predictor** is designed to solve the intermittency problem of renewable energy. By analyzing environmental factors like sunlight intensity and temperature, the system provides accurate forecasts of energy production and helps users manage their battery storage effectively.

### Key Capabilities:
- **Energy Forecasting**: Predicts AC Power output (kW) using advanced ML models.
- **Battery Simulation**: Calculates State of Charge (SOC) and time-to-empty/full based on home load.
- **Real-Time Dashboard**: Visualizes the flow of energy between panels, batteries, and the grid.

---

## 🏗️ How the System Works
The system follows a **Client-Server / ETL-Inference** architecture:

1. **Extraction (ETL)**: Raw sensor data from two different sources (Solar Farm & Battery Experiments) are ingested.
2. **Transformation**: Data is merged, cleaned, and features like **State of Charge (SOC)** are derived using linear decay modeling.
3. **Inference (API)**: A Flask server loads the trained models. When a user moves a slider on the dashboard, the API:
   - Validates the input features.
   - Scales the features using a fitted `StandardScaler`.
   - Passes the vector to the **XGBoost/Random Forest** regressor.
   - Scales the output to a mini-project-friendly scale (0-800 kWh).
4. **User Interface**: The dynamic dashboard displays results instantly, showing whether the system is "Exporting" to the grid or "Draining" the battery.

---

## 🧠 Deep Dive: Machine Learning Logic

### 1. The Chosen Algorithm: XGBoost / Random Forest
While the project includes a baseline **Linear Regression (OLS)** model (providing ~94% accuracy), we primarily use **Ensemble Learning** methods for production:

- **Why ensemble?**: Unlike simple linear models, Random Forest and XGBoost use multiple decision trees to handle **non-linear relationships**. For example, solar panels actually lose efficiency when they get *too* hot—a pattern these models capture perfectly.
- **Handling Zero-States**: The models naturally learn that when `Irradiance = 0` (Night-time), the power must be `0`, avoiding the "noise" or negative predictions simple models might produce.

### 2. Feature Importance
The system predicts power based on three primary signals:
1. **Irradiation (>90% influence)**: The raw solar flux available.
2. **Module Temperature**: The actual heat of the panel surface (impacts efficiency).
3. **Ambient Temperature**: Surrounding site conditions.

### 3. The Prediction Math
The model calculates a "Plant Efficiency" based on input conditions:
$$ \text{Predicted Output} = f(\text{Irradiation, Temp, Module\_Temp}) $$
This raw output is then scaled for the user's specific system size (e.g., 5kW Home vs 30kW Farm).

---

## 🛠️ Technology Stack
- **Backend**: Python (Flask, Flask-CORS)
- **ML Engine**: Scikit-Learn, XGBoost, TensorFlow (Optional LSTM)
- **Data Persistence**: **PostgreSQL** (Selected for Time-Series performance)
- **Visualization**: Matplotlib, Seaborn (for training graphs)
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS

---

## 🗄️ Database Strategy (PostgreSQL)
We have selected **PostgreSQL** as the core database for this system because:
- **Time-Series Ready**: It efficiently handles the `DATE_TIME` sensor data.
- **Analyitcal Power**: It allows for complex SQL queries to track historical battery performance.
- **Hybrid Storage**: It stores solar data, weather records, battery experiments, and **prediction logs** for future model retraining.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.9+
- PostgreSQL (Recommended)
- pip

### 2. Quick Start
```bash
# 1. Install Libraries
pip install -r requirements.txt

# 2. Train the AI (Generates the best model)
python src/model_trainer.py

# 3. Start the API Server
python src/api.py
```

3. Open `index.html` in your browser to view the **Live Dashboard**.

---

*This project was developed for the III Year II Sem Industrial Oriented Mini Project at Sphoorthy Engineering College.*
