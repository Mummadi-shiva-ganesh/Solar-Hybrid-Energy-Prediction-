# Solar-Battery Hybrid Energy Predictor

## 1. Project Title
**Solar-Battery Hybrid Energy Predictor**

A Machine Learning-powered web application that simulates and predicts solar power generation and battery usage for hybrid renewable energy systems.

## 2. Project Overview
### Purpose
This project provides a reliable tool for estimating energy availability in solar-battery hybrid systems. It predicts how much power a solar plant will generate based on weather conditions and calculates battery charging/discharging times based on home load.

### Real-World Problem Solved
Renewable energy is intermittent. Homeowners and plant managers often struggle to predict:
1.  How much energy their panels will produce on a given day.
2.  How long their battery backup will last during a grid outage.
This system solves that by accurately forecasting generation and storage status.

### Importance
-   **Energy Efficiency**: Helps users optimize their energy usage.
-   **Reliability**: Provides estimates for off-grid survival time.
-   **Scalability**: Can be adapted for small home systems or larger commercial plants.

## 3. System Architecture & Workflow
The system follows a typical **Client-Server Architecture**:

### 1. Data Processing Module
-   **Input**: Historical solar generation and battery usage data.
-   **Preprocessing**: Cleans, aligns, and engineers features from the raw data.
-   **Modeling**: A Random Forest model learns the relationship between weather and power output.
-   **Persistence**: **PostgreSQL** handles storage for time-series sensor data, ML logs, and battery experiment records.

### 2. Application Logic (Backend)
-   **API**: A Python Flask server (`src/api.py`) exposes the trained model.
-   **Inference**: Receives weather inputs from the user, computes the prediction, and returns the result.

### 3. User Interface (Frontend)
-   **Dashboard**: An interactive HTML5/Tailwind web page (`index.html`).
-   **User Flow**:
    1.  User adjusts sliders (Irradiance, Temperature, Load).
    2.  Frontend sends data to Backend API.
    3.  Backend returns predicted Solar Output.
    4.  Logic calculates Net Power = Solar Production - Home Load.
    5.  Dashboard updates visualizations (Green = Exporting, Red = Draining).

## 4. Dataset Description
### Source
-   **Solar Data**: Public Kaggle dataset (Indian Solar Power Plant, 30kW scale).
-   **Battery Data**: Experimental Li-Ion battery data (Kaggle).

### Type of Data
-   **Time-Series**: 15-minute interval records spanning 34 days.
-   **Numerical**: Sensor readings for voltage, current, and temperature.

### Important Features
-   `DATE_TIME`: Timestamp of the recording.
-   `AC_POWER` (Target): Total power output from the inverter (kW).
-   `AMBIENT_TEMPERATURE`: Site air temperature (°C).
-   `MODULE_TEMPERATURE`: Temperature of the solar panel surface (°C).
-   `IRRADIATION`: Solar flux intensity (kW/m²).
-   `SOC` (State of Charge): Percentage of battery capacity remaining (Derived).

### Split Strategy
-   **Training Set (80%)**: Used to train the model.
-   **Testing Set (20%)**: Used to evaluate performance on unseen data.

## 5. Data Preprocessing
Preprocessing ensures the data is clean and suitable for Machine Learning.

1.  **Time Alignment**: Merged Solar and Weather datasets using timestamps to ensure every power reading corresponds to specific weather conditions.
2.  **Handling Missing Values**: Used Inner Joins to remove incomplete records where sensor data was missing.
3.  **Feature Selection**: Dropped non-predictive columns like `SOURCE_KEY` and `PLANT_ID`.
4.  **Battery Simulation**: Derived a continuous State of Charge (SOC) variable using a Linear Decay model to simulate a full discharge cycle alongside the solar data.
5.  **Normalization**: While Random Forest is robust to scale, the application output is normalized to allow users to simulate different system sizes (e.g., 5kW vs 30kW) using a scaling factor.

**Why Proper Preprocessing is Necessary**: Real-world sensor data is often noisy. Without alignment and cleaning, the model would learn incorrect patterns (e.g., predicting power at night).

## 6. Algorithms Used
We implemented and compared two algorithms:

### 1. Random Forest Regressor (Selected Model)
-   **Why Chosen**: It handles non-linear relationships exceptionally well (e.g., the saturation of power output at high irradiance). It is also robust to outliers.
-   **How it works**: It builds multiple Decision Trees during training. Each tree gives a prediction, and the forest averages them to give the final output.
-   **Advantages**: High accuracy, prevents overfitting, handles "zero" values (night time) naturally.

### 2. Linear Regression (Baseline)
-   **Why Chosen**: To serve as a benchmark for performance.
-   **How it works**: Fits a straight line ensuring the minimum error between actual and predicted values.
-   **Limitations**: Failed to capture the non-linear "efficiency drop" at very high temperatures.

## 7. Model Training & Implementation
-   **Library**: `scikit-learn` (Python).
-   **Hyperparameters**:
    -   `n_estimators=100` (Number of trees in the forest).
    -   `random_state=42` (Ensures reproducibility).
-   **Training Process**:
    1.  Load data via `data_loader.py`.
    2.  Split data into Training (X_train) and Testing (X_test) sets.
    3.  Initialize the Random Forest Regressor.
    4.  Fit the model to `X_train` and `y_train`.
    5.  Save the trained model as a `.pkl` file for the API to use.

## 8. Evaluation Metrics
We evaluated the model using:

1.  **Mean Squared Error (MSE)**:
    -   Measures the average squared difference between estimated values and the actual value.
    -   **Why**: Penalizes larger errors more significantly, which is important for power forecasting.

2.  **R² Score (Coefficient of Determination)**:
    -   Represents the proportion of variance for the dependent variable that's explained by the independent variables.
    -   **Result**: High R² score (>0.90) indicates the model explains most of the variability in power output based on weather data.

## 9. Results & Output
### Model Performance
-   The Random Forest model demonstrated a strong positive correlation between `IRRADIATION` and `AC_POWER`.
-   It accurately predicts **0 kW** output during night times (Irradiance = 0).
-   It captures the efficiency loss due to high `MODULE_TEMPERATURE`.

### Application Output
-   **Inputs**: User sets Irradiance (e.g., 800 W/m²) and Temp (e.g., 25°C).
-   **Dashboard**: Displays "Estimated Solar Output: 4.2 kW".
-   **Visuals**: The "Net Power" card dynamically changes color based on whether the home is self-sufficient or drawing from the grid.

## 10. Technologies & Tools Used
-   **Programming Language**: Python 3.8+
-   **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`.
-   **Backend Framework**: Flask (REST API).
-   **Database**: PostgreSQL (Time-series optimization, ML logs).
-   **Frontend**: HTML5, JavaScript (Fetch API), Tailwind CSS (CDN).
-   **Tools**: VS Code, Git, GitHub.

## 11. Challenges Faced
1.  **Data Alignment**: Merging datasets from different sources (Solar vs Battery) with slightly different timestamps.
    -   *Solution*: Used pandas datetime indexing and nearest-neighbor interpolation or interval joining.
2.  **Night-Time Bias**: The dataset has many "0" values for power at night.
    -   *Solution*: Random Forest handled this naturally; no artificial balancing was needed as this is a real physical state.
3.  **Cross-Origin Errors (CORS)**: Establishing communication between the HTML frontend and Flask backend.
    -   *Solution*: Installed and configured `flask-cors` to allow local requests.

## 12. Future Enhancements
-   **Live Weather Integration**: Connect to an OpenWeatherMap API to fetch real-time local weather instead of manual sliders.
-   **Deep Learning**: Implement LSTM (Long Short-Term Memory) networks for time-series forecasting to predict power 24 hours in advance.
-   **Battery Health Monitoring**: Add more complex models to predict battery degradation over years.

## 13. Conclusion
The **Solar-Battery Hybrid Energy Predictor** successfully demonstrates how Machine Learning can optimize renewable energy systems. By accurately simulating power generation and battery usage, it empowers users to make informed decisions about their energy consumption. The project highlights the importance of data preprocessing and the effectiveness of ensemble methods like Random Forest in modeling physical systems.

## 14. How to Run the Project
### Prerequisites
-   Python 3.8+
-   Git

### Step-by-Step Instructions
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Mummadi-shiva-ganesh/proto.git
    cd proto
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model (Optional)**:
    If the model file is missing, regenerate it:
    ```bash
    python src/train_model.py
    ```

4.  **Start the Backend API**:
    ```bash
    python src/api.py
    ```
    *Keep this terminal running.*

5.  **Launch the Dashboard**:
    -   Find `index.html` in the project folder.
    -   Double-click to open it in your browser.
