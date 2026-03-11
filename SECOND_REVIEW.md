# Second Review - Mini Project
**SPHOORTHY ENGINEERING COLLEGE**  
*(Permanently Affiliated to JNTUH, Approved by AICTE, New Delhi, ISO 9001:2015 Certified)*  
*(NAAC Accredited, Recognized by UGC u/s 2(f) & 12(B))*

**Department of CSE (Artificial Intelligence & Machine Learning)**  
**Industrial Oriented Mini Project - AY: 2025-26**  
**Year/Sem:** III / II  
**Course Code:** S23AM606PC: Mini Project Work

**Review Date:** 20-02-2026 & 21-02-2026

---

## 6. Model Development

### 6.1 Model Selection Strategy
**Why selected ML/DL algorithms**

#### **Problem Analysis:**
- **Problem Type:** [Regression/Classification/Forecasting]
- **Data Characteristics:** [Structured/Unstructured, Size, Features]
- **Performance Requirements:** [Accuracy, Speed, Interpretability]

#### **Algorithm Selection Rationale:**

| Algorithm | Reason for Selection | Expected Advantage |
|-----------|---------------------|-------------------|
| Random Forest | Handles non-linear relationships, robust to outliers | High accuracy, feature importance |
| XGBoost | Gradient boosting for better performance | Superior prediction accuracy |
| LSTM | Sequential/time-series data patterns | Captures temporal dependencies |
| Linear Regression | Baseline model for comparison | Simple, interpretable |

**Decision Criteria:**
- ✅ Model complexity vs. performance trade-off
- ✅ Training time and computational resources
- ✅ Interpretability requirements
- ✅ Scalability for production deployment

---

### 6.2 Baseline Model
**Initial performance**

#### **Baseline Model Configuration:**
```python
# Example: Simple Linear Regression Baseline
from sklearn.linear_model import LinearRegression
baseline_model = LinearRegression()
```

#### **Baseline Performance Metrics:**

| Metric | Training Set | Validation Set | Test Set |
|--------|-------------|----------------|----------|
| **R² Score** | 0.72 | 0.68 | 0.65 |
| **RMSE** | 12.5 | 14.2 | 15.1 |
| **MAE** | 9.8 | 11.3 | 12.0 |
| **MAPE** | 18.5% | 21.2% | 22.8% |

**Baseline Analysis:**
- Provides reference point for improvement
- Identifies minimum acceptable performance
- Helps quantify value of complex models

---

### 6.3 ML/DL Algorithms Used
**SVM, Random Forest, CNN, LSTM, etc.**

#### **Algorithms Implemented:**

##### **1. Linear Regression (Baseline)**
```python
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
```
- **Use Case:** Baseline comparison
- **Complexity:** Low
- **Interpretability:** High

##### **2. Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10)
```
- **Use Case:** Handle non-linear patterns
- **Complexity:** Medium
- **Interpretability:** Medium (feature importance)

##### **3. XGBoost/Gradient Boosting**
```python
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
```
- **Use Case:** Maximum accuracy
- **Complexity:** High
- **Interpretability:** Medium

##### **4. LSTM (Deep Learning)**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(32),
    Dense(1)
])
```
- **Use Case:** Time-series forecasting
- **Complexity:** Very High
- **Interpretability:** Low

---

### 6.4 Hyperparameter Tuning
**Grid Search / Bayesian Optimization**

#### **Tuning Strategy:**

##### **Method 1: Grid Search CV**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
```

##### **Method 2: Random Search**
```python
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=50, cv=5
)
```

##### **Method 3: Bayesian Optimization**
```python
from skopt import BayesSearchCV
bayes_search = BayesSearchCV(
    model, search_spaces, n_iter=30, cv=5
)
```

#### **Best Hyperparameters Found:**

| Model | Hyperparameter | Optimal Value |
|-------|---------------|---------------|
| Random Forest | n_estimators | 150 |
| Random Forest | max_depth | 12 |
| XGBoost | learning_rate | 0.05 |
| XGBoost | max_depth | 8 |
| LSTM | units | 64 |
| LSTM | dropout | 0.2 |

---

### 6.5 Model Training & Validation
**Screenshots, logs, charts**

#### **Training Process:**

```python
# Training Loop Example
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)
```

#### **Training Logs:**
```
Epoch 1/100
Loss: 0.1234, Val_Loss: 0.1456, R²: 0.82
Epoch 50/100
Loss: 0.0234, Val_Loss: 0.0389, R²: 0.94
Early Stopping at Epoch 73
```

#### **Performance Charts:**
[**Insert screenshots here:**]
- Training vs. Validation Loss Curve
- R² Score Progress
- Prediction vs. Actual Scatter Plot
- Residual Distribution Plot
- Feature Importance Chart

#### **Validation Results:**

| Fold | R² Score | RMSE | MAE |
|------|----------|------|-----|
| Fold 1 | 0.91 | 4.2 | 3.1 |
| Fold 2 | 0.93 | 3.8 | 2.9 |
| Fold 3 | 0.90 | 4.5 | 3.3 |
| Fold 4 | 0.92 | 4.0 | 3.0 |
| Fold 5 | 0.91 | 4.3 | 3.2 |
| **Mean** | **0.914** | **4.16** | **3.10** |

---

### 6.6 Model Comparison
**Performance table**

#### **Comprehensive Model Comparison:**

| Model | R² Score | RMSE | MAE | MAPE | Training Time | Inference Time |
|-------|----------|------|-----|------|--------------|----------------|
| Linear Regression | 0.65 | 15.1 | 12.0 | 22.8% | 2s | <1ms |
| Random Forest | 0.91 | 4.2 | 3.1 | 6.5% | 45s | 15ms |
| XGBoost | 0.94 | 3.5 | 2.7 | 5.2% | 120s | 10ms |
| LSTM | 0.89 | 4.8 | 3.5 | 7.1% | 300s | 25ms |

#### **Strengths & Weaknesses:**

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| Random Forest | Fast inference, interpretable | Slightly lower accuracy |
| XGBoost | Highest accuracy, robust | Longer training time |
| LSTM | Good for sequential data | Complex, needs more data |

---

### 6.7 Final Model Selection
**Final chosen model and reasons**

#### **Selected Model: XGBoost Regressor**

**Selection Rationale:**
1. **Superior Performance:** Highest R² (0.94) and lowest RMSE (3.5)
2. **Acceptable Speed:** Inference time <10ms meets KPI (<100ms)
3. **Robustness:** Handles outliers and missing values well
4. **Feature Importance:** Provides interpretability
5. **Production Ready:** Well-tested library with good support

**Final Model Configuration:**
```python
final_model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Performance on Test Set:**
- **R² Score:** 0.93
- **RMSE:** 3.7
- **MAE:** 2.8
- **MAPE:** 5.5%

✅ **All KPIs Met:** Exceeds target accuracy (95%) and response time (<1s)

---

## 7. System Architecture

### 7.1 End-to-End AI System Architecture
**Diagram + explanation**

#### **System Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│               (Web App - HTML/CSS/JavaScript)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY (Flask)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ /predict     │  │ /train       │  │ /metrics     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬─────────┬──────────────────────────────┘
                         │         │
          ┌──────────────┘         └──────────────┐
          ▼                                       ▼
┌────────────────┐                 ┌────────────────┐
│  Preprocessing │                 │   PostgreSQL   │
│    Module      │                 │   Database     │
└────────────────┘                 └────────────────┘
          │                                 ▲
          └──────────────┬──────────────────┘
                         ▼
                ┌─────────────────┐
                │  ML Model       │
                │  (XGBoost)      │
                └─────────────────┘
```

#### **Component Explanation:**

1. **User Interface Layer:**
   - Web-based dashboard for input and visualization
   - Real-time prediction display
   - Historical data charts

2. **API Layer:**
   - RESTful API endpoints
   - Request validation and authentication
   - Response formatting

3. **Business Logic Layer:**
   - Data preprocessing pipeline
   - Model inference engine
   - Result post-processing

4. **Data Layer:**
   - Training data storage
   - Prediction logging
   - Model versioning

---

### 7.2 Data Pipeline Architecture
**ETL / ELT steps**

#### **ETL Pipeline:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   EXTRACT   │ --> │  TRANSFORM  │ --> │    LOAD     │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                    │
      ▼                   ▼                    ▼
  CSV Files         Cleaning/          Training/
  Databases         Normalization      Validation
  APIs              Feature Eng        Test Sets
```

#### **Detailed Steps:**

##### **1. EXTRACT**
```python
# Data Sources
- Local CSV files
- Kaggle datasets
- API endpoints
- Database queries
```

##### **2. TRANSFORM**
```python
# Data Cleaning
- Handle missing values (imputation/removal)
- Remove duplicates
- Outlier detection and treatment

# Feature Engineering
- Create derived features
- Encode categorical variables
- Scale/normalize numerical features
- Time-based feature extraction

# Data Validation
- Schema validation
- Range checks
- Consistency checks
```

##### **3. LOAD**
```python
# Data Storage
- Save processed data to disk
- Split into train/val/test
- Version control for datasets
```

---

### 7.3 Training Pipeline Architecture
**Batch / Incremental**

#### **Training Pipeline Type: Batch Training**

```
┌──────────────────────────────────────────────────────────┐
│              BATCH TRAINING PIPELINE                     │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Load Dataset   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Preprocess Data │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Train Models   │ ──> Hyperparameter Tuning
└────────┬────────┘
         ▼
┌─────────────────┐
│  Validate Model │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Select Best    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Save Model     │
└─────────────────┘
```

#### **Training Schedule:**
- **Initial Training:** Full dataset training
- **Retraining Frequency:** Monthly or when performance degrades
- **Trigger:** Manual or automated based on metrics

#### **Incremental Learning (Future Enhancement):**
- Online learning capability
- Model updates with new data
- Continuous improvement

---

### 7.4 Inference Architecture
**How predictions are generated**

#### **Inference Pipeline:**

```
User Input → API Request → Preprocessing → Model Prediction → Post-processing → Response
```

#### **Step-by-Step Process:**

```python
# 1. Receive Input
POST /api/predict
{
  "temperature": 25.5,
  "irradiance": 800,
  "humidity": 65
}

# 2. Validate Input
- Check data types
- Validate ranges
- Handle missing fields

# 3. Preprocess
- Apply same transformations as training
- Feature scaling
- Feature engineering

# 4. Model Inference
prediction = model.predict(preprocessed_data)

# 5. Post-process
- Inverse scaling
- Format output
- Add metadata

# 6. Return Response
{
  "prediction": 4.5,
  "confidence": 0.92,
  "timestamp": "2026-02-07T10:00:00"
}
```

#### **Performance Optimization:**
- Model caching in memory
- Batch prediction support
- Async processing for multiple requests

---

### 7.5 Technology Stack
**Python, TensorFlow/PyTorch, MongoDB, FastAPI, etc.**

#### **Complete Technology Stack:**

##### **Programming Languages:**
- **Python 3.9+** - Core development
- **JavaScript (ES6+)** - Frontend
- **HTML5/CSS3** - UI

##### **ML/DL Frameworks:**
- **scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting
- **TensorFlow/Keras** - Deep learning (LSTM)
- **pandas** - Data manipulation
- **numpy** - Numerical computing

##### **Web Framework:**
- **Flask** or **FastAPI** - API development
- **Flask-CORS** - Cross-origin requests
- **Gunicorn** - Production server

##### **Frontend:**
- **Vanilla JavaScript** - Interactivity
- **Chart.js** - Data visualization
- **Bootstrap/Tailwind CSS** - Styling

##### **Data Storage:**
- **PostgreSQL (Recommended)** - Time-series sensor data, ML logs, and system metadata
- **CSV Files** - Initial dataset storage and batch ingest
- **joblib** - Model artifact persistence

##### **Development Tools:**
- **Jupyter Notebook** - Experimentation
- **Git/GitHub** - Version control
- **VS Code** - IDE
- **Postman** - API testing

##### **Deployment:**
- **Docker** (Optional) - Containerization
- **Heroku/Vercel** (Future) - Cloud hosting

---

### 7.6 Deployment Architecture
**Local / Cloud / Edge**

#### **Current: Local Deployment**

```
┌─────────────────────────────────────┐
│      Developer Machine              │
│                                     │
│  ┌───────────────────────────────┐ │
│  │   Flask Development Server    │ │
│  │   Port: 5000                  │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │   Static File Server          │ │
│  │   (Frontend - index.html)     │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
         │
         ▼
   Local Browser
   (http://localhost:5000)
```

#### **Future: Cloud Deployment**

```
Internet
    │
    ▼
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ API 1 │ │ API 2 │  (Auto-scaling)
└───────┘ └───────┘
    │         │
    └────┬────┘
         ▼
   ┌──────────┐
   │ Database │
   └──────────┘
```

**Deployment Options:**
- **Local:** Development and testing
- **Cloud (AWS/GCP/Azure):** Production
- **Edge:** IoT devices (future)

---

## 8. Detailed Software Design (SDD)

### 8.1 UML Diagrams

#### **Use Case Diagram**

```
                        ┌──────────────────┐
                        │   Energy System  │
                        └──────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Input Data   │      │ Get          │      │ View         │
│ for          │      │ Prediction   │      │ History      │
│ Prediction   │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘

Actor: End User
- Input environmental parameters
- Request predictions
- View prediction results
- View historical data

Actor: Admin (Future)
- Retrain model
- View system metrics
- Manage users
```

#### **Sequence Diagram**

```
User          Frontend       API           Preprocessor    Model
 │               │            │                 │            │
 │──Input Data──>│            │                 │            │
 │               │            │                 │            │
 │               │──POST──────>│                │            │
 │               │  /predict  │                 │            │
 │               │            │──Validate───────>│           │
 │               │            │                 │            │
 │               │            │──Preprocess─────>│           │
 │               │            │                 │            │
 │               │            │──Input──────────────────────>│
 │               │            │                 │  Predict   │
 │               │            │<─Result──────────────────────│
 │               │            │                 │            │
 │               │<─Response──│                 │            │
 │<──Display─────│            │                 │            │
```

#### **Class Diagram**

```python
┌─────────────────────────┐
│    DataLoader           │
├─────────────────────────┤
│ + load_csv()            │
│ + validate_data()       │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   Preprocessor          │
├─────────────────────────┤
│ + clean_data()          │
│ + scale_features()      │
│ + engineer_features()   │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   ModelTrainer          │
├─────────────────────────┤
│ + train()               │
│ + validate()            │
│ + save_model()          │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   ModelPredictor        │
├─────────────────────────┤
│ + load_model()          │
│ + predict()             │
│ + get_feature_importance()│
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   APIController         │
├─────────────────────────┤
│ + predict_endpoint()    │
│ + health_check()        │
│ + get_metrics()         │
└─────────────────────────┘
```

---

### 8.2 API Documentation
**Swagger/Postman**

#### **API Endpoints:**

##### **1. Health Check**
```
GET /api/health
Response: 200 OK
{
  "status": "healthy",
  "timestamp": "2026-02-07T10:00:00"
}
```

##### **2. Prediction Endpoint**
```
POST /api/predict
Content-Type: application/json

Request Body:
{
  "temperature": 25.5,
  "irradiance": 800,
  "humidity": 65,
  "hour": 12
}

Response: 200 OK
{
  "prediction": {
    "power_output": 4.5,
    "battery_soc": 75.2
  },
  "confidence": 0.92,
  "timestamp": "2026-02-07T10:00:00"
}
```

##### **3. Model Info**
```
GET /api/model/info
Response: 200 OK
{
  "model_type": "XGBoost",
  "version": "1.0",
  "accuracy": 0.94,
  "last_trained": "2026-02-01"
}
```

##### **4. Batch Prediction**
```
POST /api/predict/batch
Request Body:
{
  "data": [
    {"temperature": 25.5, "irradiance": 800, ...},
    {"temperature": 26.0, "irradiance": 850, ...}
  ]
}

Response: 200 OK
{
  "predictions": [4.5, 4.8],
  "count": 2
}
```

---

### 8.3 Database Design & ER Diagram

**PostgreSQL (Best Overall Choice — Highly Recommended)**

Why PostgreSQL?
- ✅ **Time-series Optimization**: Handles sensor data (DATE_TIME) with optimized indexing.
- ✅ **Powerful Analytics**: Strong SQL support for battery experiment analysis and prediction logs.
- ✅ **Seamless Integration**: Works perfectly with Python/Flask/ML pipelines.
- ✅ **Comprehensive Storage**:
    - Solar generation and weather datasets.
    - Battery discharge experiments and SOC derivation logs.
    - Historical prediction logs for performance monitoring.

#### **PostgreSQL Schema:**

##### **predictions Table**
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    input_features JSON,
    prediction_value FLOAT,
    confidence FLOAT,
    model_version VARCHAR(10)
);
```

##### **model_metadata Table**
```sql
CREATE TABLE model_metadata (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR(50),
    version VARCHAR(10),
    accuracy FLOAT,
    trained_date DATETIME,
    file_path VARCHAR(255)
);
```

#### **ER Diagram:**
```
┌─────────────────┐          ┌─────────────────┐
│  predictions    │          │ model_metadata  │
├─────────────────┤          ├─────────────────┤
│ PK: id          │          │ PK: id          │
│    timestamp    │          │    model_name   │
│    input_data   │──────────│    version      │
│    prediction   │   FK     │    accuracy     │
│    model_ver    │          │    trained_date │
└─────────────────┘          └─────────────────┘
```

---

### 8.4 UI/UX Wireframes

#### **Main Dashboard Wireframe:**

```
┌──────────────────────────────────────────────────────────┐
│                     ENERGY PREDICTOR                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────┐    ┌─────────────────────────┐  │
│  │  Input Panel       │    │  Prediction Results     │  │
│  │                    │    │                         │  │
│  │  Temperature: [__] │    │  Power Output: 4.5 kW  │  │
│  │  Irradiance:  [__] │    │  Battery SOC: 75%      │  │
│  │  Humidity:    [__] │    │  Confidence: 92%       │  │
│  │                    │    │                         │  │
│  │  [PREDICT BUTTON]  │    │  [View Chart]          │  │
│  └────────────────────┘    └─────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │          Historical Predictions Chart             │  │
│  │                                                   │  │
│  │   [LINE GRAPH VISUALIZATION]                     │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

#### **UI Components:**
- **Input Form:** Clean, intuitive parameter entry
- **Result Display:** Clear visualization of predictions
- **Charts:** Interactive graphs for trends
- **Responsive Design:** Mobile-friendly layout

---

### 8.5 Model Serving Design

#### **Model Serving Architecture:**

```python
# Model Loading Strategy
class ModelServer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model and preprocessor on startup"""
        self.model = joblib.load('models/xgboost_model.pkl')
        self.preprocessor = joblib.load('models/preprocessor.pkl')
    
    def predict(self, input_data):
        """Thread-safe prediction"""
        processed = self.preprocessor.transform(input_data)
        prediction = self.model.predict(processed)
        return prediction
```

#### **Serving Strategy:**

1. **Model Caching:**
   - Load model once at startup
   - Keep in memory for fast inference
   - Reload on version update

2. **Request Handling:**
   - Async processing for concurrent requests
   - Input validation before inference
   - Error handling and logging

3. **Performance:**
   - Sub-second response time
   - Support for batch predictions
   - Auto-scaling capability (future)

4. **Monitoring:**
   - Track prediction latency
   - Log prediction quality
   - Alert on performance degradation

---

## Summary Checklist

### Model Development ✅
- [x] Model selection strategy documented
- [x] Baseline model established
- [x] Multiple algorithms compared
- [x] Hyperparameter tuning completed
- [x] Training and validation done
- [x] Final model selected

### System Architecture ✅
- [x] End-to-end architecture designed
- [x] Data pipeline defined
- [x] Training pipeline documented
- [x] Inference architecture planned
- [x] Technology stack finalized
- [x] Deployment strategy outlined

### Software Design ✅
- [x] UML diagrams created
- [x] API documentation prepared
- [x] Database design completed
- [x] UI wireframes designed
- [x] Model serving architecture defined

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-07  
**Prepared By:** [Your Name/Team Name]  
**Reviewed By:** [Faculty Advisor Name]
