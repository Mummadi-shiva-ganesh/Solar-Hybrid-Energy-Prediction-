"""
MLOps: Enhanced Automated Retraining Pipeline (Section 10.7)
Handles: data loading, training, evaluation, versioning, and safe model swap.
Trigger: cron / manual / performance-threshold event.

Usage:
    python src/retrain_pipeline.py              # full retrain
    python src/retrain_pipeline.py --dry-run    # evaluate only, no model swap
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            encoding="utf-8"
        ),
    ]
)
log = logging.getLogger("retrain")

# ---------------------------------------------------------------------------
# Thresholds – retrain only if improvement meets bar
# ---------------------------------------------------------------------------
MIN_R2_IMPROVEMENT = 0.001   # new model must beat old by at least 0.1pp
R2_RETRAIN_TRIGGER = 0.90    # force retrain if current R² drops below this


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Load solar generation + weather data; fall back to mock data if unavailable."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

    try:
        from data_loader import load_and_process_data
        df = load_and_process_data(data_dir=os.path.join(PROJECT_ROOT, "data"))
        log.info(f"Loaded real data: {len(df)} rows")
    except Exception as e:
        log.warning(f"Real data load failed ({e}), using mock data")
        from create_mock_data import create_mock_dataset
        df = create_mock_dataset()
        log.info(f"Mock data generated: {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df):
    feature_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    target_col   = "AC_POWER" if "AC_POWER" in df.columns else "DC_POWER"

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    log.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    log.info("Training Random Forest Regressor …")
    t0    = time.time()
    model = RandomForestRegressor(
        n_estimators=150, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    log.info(f"Training complete in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2   = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))

    safe_ratio = np.where(np.abs(y_test) > 1e-6,
                          np.abs((y_test - y_pred) / y_test), 0)
    mape = float(np.mean(safe_ratio) * 100)

    log.info(f"  R²   = {r2:.4f}")
    log.info(f"  RMSE = {rmse:.2f} kW")
    log.info(f"  MAE  = {mae:.2f} kW")
    log.info(f"  MAPE = {mape:.2f} %")
    return {"val_r2": r2, "val_rmse": rmse, "val_mae": mae, "val_mape": mape}


# ---------------------------------------------------------------------------
# 5. Version Management
# ---------------------------------------------------------------------------
def get_next_version():
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return "1.0"
    with open(meta_path) as f:
        old = json.load(f)
    try:
        return str(round(float(old.get("version", "1.0")) + 0.1, 1))
    except (ValueError, TypeError):
        return "1.1"


def get_current_r2():
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return 0.0
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("performance", {}).get("val_r2", 0.0)


# ---------------------------------------------------------------------------
# 6. Safe Model Swap (backup old → write new atomically)
# ---------------------------------------------------------------------------
def swap_model(model, scaler, metrics, feature_names, version):
    backup_dir = os.path.join(MODEL_DIR, f"backup_v{version}")
    os.makedirs(backup_dir, exist_ok=True)

    # Backup existing artifacts
    for fname in ("best_model.pkl", "scaler.pkl", "model_metadata.json"):
        src = os.path.join(MODEL_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, fname))
            log.info(f"  Backed up {fname} → {backup_dir}")

    # Write new artifacts
    joblib.dump(model,  os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    metadata = {
        "model_name":    "Random Forest",
        "model_type":    type(model).__name__,
        "version":       version,
        "trained_date":  datetime.now().isoformat(),
        "feature_names": feature_names,
        "performance":   metrics,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"  New model saved as version {version}")
    return metadata


# ---------------------------------------------------------------------------
# 7. Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(dry_run: bool = False):
    log.info("=" * 60)
    log.info("  AUTOMATED RETRAINING PIPELINE")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if dry_run:
        log.info("  [DRY RUN — no model swap will occur]")
    log.info("=" * 60)

    # Step 1 — Load
    df = load_data()

    # Step 2 — Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # Step 3 — Train
    model = train_model(X_train, y_train)

    # Step 4 — Evaluate
    log.info("--- New Model Evaluation -----------------------------")
    new_metrics = evaluate(model, X_test, y_test)

    # Step 5 — Compare with current model
    current_r2 = get_current_r2()
    new_r2     = new_metrics["val_r2"]
    log.info(f"  Current model R²: {current_r2:.4f}")
    log.info(f"  New     model R²: {new_r2:.4f}")

    should_swap = (
        new_r2 > current_r2 + MIN_R2_IMPROVEMENT
        or current_r2 < R2_RETRAIN_TRIGGER
    )

    if dry_run:
        log.info("  [DRY RUN] Would swap model: " + ("YES" if should_swap else "NO"))
        log.info("Pipeline finished (dry run).")
        return new_metrics

    # Step 6 — Swap if better
    if should_swap:
        version = get_next_version()
        swap_model(model, scaler, new_metrics, feature_names, version)
        log.info(f"  [SUCCESS] Model swapped to version {version}")
    else:
        log.info(
            f"  [SKIPPED] New model (R²={new_r2:.4f}) did not improve current "
            f"(R²={current_r2:.4f}) by >={MIN_R2_IMPROVEMENT}. Keeping current model."
        )

    log.info("=" * 60)
    log.info("  RETRAINING PIPELINE COMPLETE")
    log.info("=" * 60)
    return new_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar Yield – Retrain Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate only; do not swap model artifacts")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_pipeline(dry_run=args.dry_run)
