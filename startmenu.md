# Solar Energy Prediction - Start Menu ☀️🔋

Welcome to the **Solar Energy Prediction** project! This guide provides detailed instructions on how to set up, run, and manage the entire infrastructure.

---

## 🚀 Quick Start (Recommended)

The easiest way to run everything is using **Docker**. This ensures all dependencies and environment variables are correctly configured.

### Using Docker Compose
1. **Ensure Docker Desktop is running.**
2. **Open your terminal** in the project root directory.
3. **Start the API:**
   ```bash
   docker-compose up -d --build api
   ```
4. **Access the Dashboard:** Open your browser to [http://localhost:5000](http://localhost:5000)

---

## 🛠️ Local Development Setup

If you prefer to run the project without Docker, follow these steps:

### 1. Prerequisites
- **Python 3.9+** installed.
- **Node.js** (Optional, only for advanced frontend development).

### 2. Installation
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the API Server
```bash
python src/api.py
```
The server will be available at `http://localhost:5000`.

---

## 📊 MLOps Integration

The project includes specialized pipelines for monitoring and retraining.

### Model Monitoring
This script checks for data drift, accuracy drops, and latency.
- **Docker:** `docker-compose run monitor`
- **Local:** `python src/monitor.py`

### Automated Retraining
Trigger a new training cycle if model performance degrades.
- **Docker:** `docker-compose --profile retrain up retrain`
- **Local:** `python src/retrain_pipeline.py`

---

## 📁 Project Structure Overview

- `src/`: Core logic (API, Training, Monitoring, Retraining).
- `web/`: Frontend files (HTML, JS, CSS).
- `models/`: Trained model and scaler artifacts.
- `data/`: Datasets used for training and testing.
- `docs/`: API specifications and documentation.
- `logs/`: System and monitoring logs.

---

## 🛡️ Security & Performance
- **API Key**: Protected via `X-API-KEY` header (`solar-yield-secret-2026`).
- **Input Validation**: Irradiation is strictly validated and rejected if outside the `[0, 1.0]` range.
- **Scaling**: Predictions are scaled to a realistic `0 - 500 kWh` range for the dashboard.

---

## ☁️ Deployment
For instructions on deploying to AWS, Azure, GCP, or Render, please refer to:
[**CLOUD_DEPLOYMENT.md**](file:///c:/Users/jyoth/OneDrive/Documents/Desktop/mini%20project/CLOUD_DEPLOYMENT.md)
