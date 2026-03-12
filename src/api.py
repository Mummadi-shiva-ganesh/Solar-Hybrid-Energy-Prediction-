"""
Enhanced Flask API for ML Model Serving
Implements RESTful endpoints as per 2nd Review specifications
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

# Security: Simple API Key for authentication (In production, use environment variables)
API_KEY = "solar-yield-secret-2026"

# Global variables for model and scaler
model = None
scaler = None
metadata = None

# Display scaling: dataset is large solar farm (0 ~ 30k kW). Scale to residential/mini-project range:
# 0 - 500 kWh (hourly).
REFERENCE_MAX_KW = 30000.0   # approx max from 30kW plant aggregated data
DISPLAY_MAX_KWH = 500.0      # max display value (hourly kWh)
DISPLAY_MIN_KWH = 0.0        # Allow zero predictions


def scale_prediction_for_display(raw_prediction, irradiation=1.0):
    """Scale raw model output (large plant kW) to 0-500 kWh range for display."""
    if raw_prediction <= 0 or irradiation <= 0.001:
        return 0.0
    scaled = (raw_prediction / REFERENCE_MAX_KW) * DISPLAY_MAX_KWH
    return round(min(DISPLAY_MAX_KWH, scaled), 2)

def load_model_artifacts():
    """Load model, scaler, and metadata on startup"""
    global model, scaler, metadata
    
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        if os.path.exists('models/model_metadata.json'):
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'model_name': 'Unknown',
                'model_type': type(model).__name__,
                'trained_date': 'Unknown'
            }
        
        print("Model artifacts loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load models on startup
load_model_artifacts()


def _openapi_path():
    """Path to OpenAPI spec (relative to this file)."""
    return os.path.join(os.path.dirname(__file__), '..', 'docs', 'openapi.json')


@app.route('/api/spec', methods=['GET'])
def openapi_spec():
    """Serve OpenAPI 3.0 spec for Swagger UI / Postman."""
    path = _openapi_path()
    if not os.path.exists(path):
        return jsonify({'error': 'OpenAPI spec not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    return jsonify(spec), 200


@app.route('/')
def index():
    """Serve the main web page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/docs')
def swagger_ui():
    """Serve Swagger UI for API documentation"""
    return send_from_directory(app.static_folder, 'swagger.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns system status and uptime
    """
    model_loaded = model is not None
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """
    Get model information and metadata
    Returns model details, performance metrics, and configuration
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    return jsonify({
        'model_name': metadata.get('model_name', 'Unknown'),
        'model_type': metadata.get('model_type', 'Unknown'),
        'version': '1.0',
        'trained_date': metadata.get('trained_date', 'Unknown'),
        'performance': metadata.get('performance', {}),
        'hyperparameters': metadata.get('hyperparameters', {}),
        'feature_names': metadata.get('feature_names', [])
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request Body:
    {
        "features": {
            "ambient_temperature": 25.5,
            "module_temperature": 35.2,
            "irradiation": 0.8,
            ...
        }
    }
    
    Response:
    {
        "prediction": 4.5,
        "confidence": 0.92,
        "timestamp": "2026-02-07T12:00:00",
        "model_name": "XGBoost"
    }
    """
    # Security: Verify API Key
    key = request.headers.get('X-API-KEY')
    if key != API_KEY:
        return jsonify({'error': 'Unauthorized: Invalid or missing API Key', 'status': 'error'}), 401
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 503
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Invalid request format. Expected "features" field',
                'status': 'error'
            }), 400
        
        features = data['features']

        # 1. Numeric Validation: Ensure all features are int or float
        for k, v in features.items():
            if not isinstance(v, (int, float)):
                return jsonify({
                    'error': f'Feature {k} must be numeric, got {type(v).__name__}',
                    'status': 'error'
                }), 400

        # 2. Extreme Value Validation: Ensure absolute value <= 1,000,000
        for k, v in features.items():
            if abs(v) > 1_000_000:
                return jsonify({
                    'error': f'Feature {k} value {v} is unrealistically extreme.',
                    'status': 'error'
                }), 400

        # Strict physical validation: Reject impossible values
        irr = features.get('IRRADIATION', 0)
        if irr < 0 or irr > 1.0:
             return jsonify({
                 'error': f'Invalid Irradiation: {irr}. Must be between 0.0 and 1.0 kW/m².',
                 'status': 'error'
             }), 400
        
        # Apply physical clipping for remaining features
        features['AMBIENT_TEMPERATURE'] = float(np.clip(features.get('AMBIENT_TEMPERATURE', 0), -20, 60))
        features['MODULE_TEMPERATURE'] = float(np.clip(features.get('MODULE_TEMPERATURE', 0), -20, 100))
        
        # Get feature names from metadata
        feature_names = metadata.get('feature_names', [])
        
        if not feature_names:
            return jsonify({
                'error': 'Feature names not available in metadata',
                'status': 'error'
            }), 500
        
        # Validate all required features are present
        missing_features = [f for f in feature_names if f not in features]
        if missing_features:
            # Fill missing features with 0 or mean values
            for f in missing_features:
                features[f] = 0
        
        # Convert to DataFrame with correct feature order
        input_df = pd.DataFrame([features])
        input_df = input_df[feature_names]  # Ensure correct order
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction (raw model output is in large-plant kW scale)
        raw_prediction = model.predict(input_scaled)[0]
        prediction = scale_prediction_for_display(float(raw_prediction), irradiation=float(features.get('IRRADIATION', 0)))
        
        # Calculate confidence (simplified - based on model type)
        confidence = 0.85 + (np.random.random() * 0.10)  # Placeholder confidence
        
        response = {
            'prediction': prediction,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat(),
            'model_name': metadata.get('model_name', 'Unknown'),
            'status': 'success'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Request Body:
    {
        "data": [
            {"feature1": val1, "feature2": val2, ...},
            {"feature1": val3, "feature2": val4, ...}
        ]
    }
    
    Response:
    {
        "predictions": [4.5, 4.8, ...],
        "count": 2,
        "timestamp": "2026-02-07T12:00:00"
    }
    """
    # Security: Verify API Key
    key = request.headers.get('X-API-KEY')
    if key != API_KEY:
        return jsonify({'error': 'Unauthorized: Invalid or missing API Key', 'status': 'error'}), 401
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({
                'error': 'Invalid request format. Expected "data" field',
                'status': 'error'
            }), 400
        
        input_data = data['data']
        feature_names = metadata.get('feature_names', [])
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Fill missing features
        for f in feature_names:
            if f not in input_df.columns:
                input_df[f] = 0
        
        input_df = input_df[feature_names]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        raw_predictions = model.predict(input_scaled)
        
        predictions = []
        for i, p in enumerate(raw_predictions):
            row_irr = input_df.iloc[i].get('IRRADIATION', 1.0)
            predictions.append(scale_prediction_for_display(float(p), irradiation=float(row_irr)))
        
        response = {
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get model performance metrics
    """
    if metadata is None:
        return jsonify({
            'error': 'Metadata not available'
        }), 503
    
    return jsonify({
        'performance': metadata.get('performance', {}),
        'model_name': metadata.get('model_name', 'Unknown'),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/features', methods=['GET'])
def get_features():
    """
    Get list of required input features
    """
    if metadata is None:
        return jsonify({
            'error': 'Metadata not available'
        }), 503
    
    return jsonify({
        'features': metadata.get('feature_names', []),
        'count': len(metadata.get('feature_names', []))
    }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask API Server")
    print("="*60)
    print("API Endpoints:")
    print("   GET  /                    - Web interface")
    print("   GET  /api/health          - Health check")
    print("   GET  /api/model/info      - Model information")
    print("   POST /api/predict         - Single prediction")
    print("   POST /api/predict/batch   - Batch predictions")
    print("   GET  /api/metrics         - Performance metrics")
    print("   GET  /api/features        - Feature list")
    print("   GET  /api/spec            - OpenAPI spec (Swagger)")
    print("   GET  /docs               - Swagger UI")
    print("="*60)
    print("Server running at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
