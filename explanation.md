# ☀️ Sopanel: End-to-End Solar Energy Prediction System

Sopanel is a comprehensive Machine Learning solution designed to predict solar energy production using environmental sensor data. This document provides a detailed walkthrough of the project, from raw data ingestion to real-time cloud deployment.

---

## 🚀 1. Project Overview
The primary goal of Sopanel is to solve the intermittency of solar energy by providing accurate, real-time power forecasts. 

**The Challenge**: Solar output fluctuates significantly based on cloud cover, time of day, and panel temperature.
**The Solution**: An AI-integrated dashboard that uses high-performance regressors (XGBoost/Random Forest) to predict energy yield (kWh) based on ambient conditions.

---

## 🛠️ 2. Technology Stack
Our system is built using a modern, scalable stack:
*   **Core Logic**: Python 3.9+
*   **Machine Learning**: Scikit-Learn, XGBoost
*   **Web Dashboard**: Streamlit (Premium Dark Theme)
*   **API Layer**: Flask (RESTful)
*   **Containerization**: Docker & Docker Compose
*   **DevOps**: GitHub Actions (CI/CD)
*   **Data Handling**: Pandas & NumPy

---

## 🏗️ 3. System Architecture
The project follows a modular, service-oriented architecture:

### A. Data Engineering (ETL)
We use a dataset representing a commercial-scale solar plant. The pipeline (`src/data_loader.py`) performs:
1.  **Ingestion**: Reading solar generation and weather sensor CSVs.
2.  **Merging**: Aligning power outputs with environmental conditions via timestamps.
3.  **Cleaning**: Handling missing values and filtering night-time noise.

### B. The Machine Learning Pipeline
The training script (`src/train_model.py`) runs a comparative study of multiple models:
*   **Linear Regression**: The baseline (simple math).
*   **Random Forest**: Captures complex, non-linear relationships.
*   **XGBoost**: Our production engine, highly optimized for tabular data.
*   **Evaluation**: Models are evaluated using **R² Score** (Accuracy) and **RMSE** (Error). The best model is automatically saved for production.

### C. Backend API (The Brain)
The Flask API (`src/api.py`) acts as the bridge. It:
*   Loads the trained model once at startup.
*   Exposes a `/predict` endpoint.
*   Validates incoming sensor data and returns a JSON prediction.

### D. Frontend Dashboard (The Face)
The **Streamlit Dashboard** (`src/app_streamlit.py`) provides a premium user experience:
*   **Real-time Inputs**: Sliders/fields for Ambient Temp, Module Temp, and Irradiation.
*   **Solar Visuals**: An animated 6-panel solar grid that "shines" based on predicted output.
*   **History Logs**: Tracks previous predictions for easy monitoring.

---

## 🔄 4. MLOps & Deployment Workflow
This project isn't just a script; it's a production-ready system.

### 🐳 Containerization
Everything runs inside **Docker**. The `Dockerfile` packages the API, ML model, and Dashboard together, ensuring it works on any machine (Windows, Mac, or Linux).

### ⚙️ CI/CD Pipeline
Every time we push code to GitHub:
1.  **Automation**: GitHub Actions starts a clean environment.
2.  **Testing**: Unit tests are run to ensure the API is correct.
3.  **Deployment**: A Docker image is automatically built and pushed to the **GitHub Container Registry (GHCR)**.

### 📈 Monitoring & Retraining
*   **Monitoring**: A dedicated script (`src/monitor.py`) checks for "Model Drift" (loss of accuracy over time).
*   **Retraining**: If accuracy drops, the `src/retrain_pipeline.py` script can be triggered to update the model with fresh data automatically.

---

## 🏃 5. How to Run the System
The entire system can be launched with a single command:

```bash
docker-compose up --build
```

*   **API Health**: [http://localhost:5000/api/health](http://localhost:5000/api/health)
*   **Web Dashboard**: [http://localhost:8501](http://localhost:8501)

---

## 🎯 6. Key Features Summary
*   **Highly Accurate**: Uses Ensemble Learning to reach >82% accuracy.
*   **Premium Design**: Custom CSS allows Streamlit to mirror a high-end "Sopanel" look.
*   **Production Ready**: Includes health checks, API keys, and automated pipelines.
*   **Scalable**: Containerized architecture ready for Cloud (AWS/Azure/GCP) deployment.

---
*Created for the Solar-Energy Prediction Project - 2026*
