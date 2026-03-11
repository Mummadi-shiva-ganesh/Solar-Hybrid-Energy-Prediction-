# Quick Start Guide - Mini Project

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- PostgreSQL (Recommended for persistent data storage)
- Web browser (Chrome, Firefox, Edge, etc.)

---

## 📦 Installation Steps

### 1. Install Dependencies
```bash
# Navigate to project directory
cd "c:\Users\jyoth\OneDrive\Documents\Desktop\mini project"

# Install all required packages
pip install -r requirements.txt
```

**Expected Installation Time:** 2-5 minutes

---

## 🎯 Running the Project

### Step 1: Train the Models

```bash
# Run the model training pipeline
python src/model_trainer.py
```

**What this does:**
- Loads and preprocesses the dataset
- Trains Linear Regression (baseline)
- Trains Random Forest with hyperparameter tuning
- Trains XGBoost with hyperparameter tuning
- Compares all models
- Selects best model
- Saves model files to `models/` directory

**Expected Output:**
```
🚀 STARTING COMPREHENSIVE MODEL TRAINING PIPELINE
📊 Loading dataset...
✅ Data preprocessing complete
🔵 Training Baseline Model: Linear Regression
✅ Training R²: 0.XXXX
🌲 Training Random Forest Regressor
...
🏆 Best Model: XGBoost
💾 Saving models...
✅ TRAINING PIPELINE COMPLETE!
```

**Training Time:** 3-10 minutes (depending on dataset size)

---

### Step 2: Start the API Server

```bash
# Run the Flask API server
python src/api.py
```

**Expected Output:**
```
🚀 Starting Flask API Server
📡 API Endpoints:
   GET  /                    - Web interface
   GET  /api/health          - Health check
   ...
🌐 Server running at: http://localhost:5000
```

**Keep this terminal window open!**

---

### Step 3: Access the Web Interface

1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. You should see the **Solar Energy Prediction System** dashboard

---

## 🎮 Using the Application

### Making Predictions:

1. **Enter Input Parameters:**
   - Ambient Temperature (e.g., 25.5°C)
   - Module Temperature (e.g., 35.2°C)
   - Irradiation (e.g., 0.8 W/m²)

2. **Click "Generate Prediction"**

3. **View Results:**
   - Predicted Power Output (in kW)
   - Confidence Score (%)
   - Model Name Used

### Viewing Model Information:

The right panel shows:
- System Status (Active/Error)
- Model Type (e.g., XGBoost)
- Performance Metrics (R² Score, RMSE)
- Training Date
- Version

---

## 🧪 Testing API Endpoints

You can test the API using tools like Postman or curl:

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Get Model Info
```bash
curl http://localhost:5000/api/model/info
```

### Make Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "AMBIENT_TEMPERATURE": 25.5,
      "MODULE_TEMPERATURE": 35.2,
      "IRRADIATION": 0.8
    }
  }'
```

---

## 📊 Project Structure

```
mini project/
├── src/
│   ├── model_trainer.py      # Model training pipeline
│   ├── preprocessor.py        # Data preprocessing
│   ├── api.py                 # Flask API server
│   └── ...
├── web/
│   └── index.html             # Web interface
├── models/                     # Saved models (after training)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── data/                       # Dataset files (CSV ingest to PostgreSQL)
│   └── solar_data.csv
├── docs/                       # API and Database documentation
│   └── openapi.json
├── requirements.txt            # Dependencies
├── SECOND_REVIEW.md           # 2nd review documentation
├── ZEROTH_REVIEW.md           # 0th review documentation
└── CHANGELOG_07-02-2026.md    # Changes log
```

---

## ⚠️ Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Model not found"
**Solution:** Run training first
```bash
python src/model_trainer.py
```

### Issue: "Port 5000 already in use"
**Solution:** 
1. Stop other programs using port 5000
2. Or modify `src/api.py` line: `app.run(port=5001)`

### Issue: "API not responding"
**Solution:**
1. Check if `python src/api.py` is running
2. Check terminal for error messages
3. Verify `models/` directory has trained models

---

## 📝 Before 2nd Review

### Checklist:
- [ ] Install all dependencies
- [ ] Train all models successfully
- [ ] Test web interface works
- [ ] Take screenshots of:
  - Training logs
  - Model comparison table
  - Web interface
  - Prediction results
- [ ] Update SECOND_REVIEW.md with actual metrics
- [ ] Prepare demo for presentation

### Screenshots Needed:
1. Training process output
2. Model comparison table
3. Web dashboard
4. Prediction example
5. API response examples

---

## 🎓 For Presentation

### Demo Flow:
1. Show project structure
2. Explain architecture (use SECOND_REVIEW.md diagrams)
3. Run training pipeline (or show screenshots)
4. Start API server
5. Demonstrate web interface:
   - Enter parameters
   - Show prediction
   - Explain confidence scores
6. Show API endpoints in Postman
7. Discuss model selection rationale
8. Show performance metrics

---

## 📞 Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python src/model_trainer.py

# Start server
python src/api.py

# Access web UI
# Browser: http://localhost:5000

# Test API
curl http://localhost:5000/api/health
```

---

**Good Luck with Your 2nd Review! 🎉**

**Date:** February 20-21, 2026  
**Department:** CSE (AI & ML)  
**Institution:** Sphoorthy Engineering College
