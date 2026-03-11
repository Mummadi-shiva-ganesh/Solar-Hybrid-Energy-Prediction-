"""
MLOps: Model Monitoring Module (Section 10.6)
Covers: Data Drift Detection, Accuracy Decay, Latency Tracking, Health Reporting

Usage:
    python src/monitor.py            # run full monitoring report
    python src/monitor.py --drift    # drift check only
    python src/monitor.py --latency  # latency benchmark only
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "monitor.log"), encoding="utf-8"),
    ]
)
log = logging.getLogger("monitor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_artifacts():
    model  = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
    with open(os.path.join(PROJECT_ROOT, "models", "model_metadata.json")) as f:
        meta = json.load(f)
    return model, scaler, meta


def _reference_stats(feature_names):
    """
    Reference (training-time) feature statistics.
    In production these would be stored during training; here we derive
    representative statistics from the Kaggle solar dataset ranges.
    """
    stats = {
        "AMBIENT_TEMPERATURE": {"mean": 28.0, "std": 8.0,  "min": 10.0, "max": 45.0},
        "MODULE_TEMPERATURE":  {"mean": 38.0, "std": 10.0, "min": 15.0, "max": 60.0},
        "IRRADIATION":         {"mean": 0.45, "std": 0.30, "min": 0.00, "max": 1.00},
    }
    return {k: stats[k] for k in feature_names if k in stats}


# ---------------------------------------------------------------------------
# 10.6.1  Data Drift Detection  (Population Stability Index — PSI)
# ---------------------------------------------------------------------------
def _psi(expected, actual, bins=10):
    """
    Population Stability Index between expected (reference) and actual distributions.
    PSI < 0.10 : No drift
    PSI 0.10–0.25 : Minor drift
    PSI > 0.25 : Major drift (alert)
    """
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max()) + 1e-9
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid division by zero / log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def check_drift(incoming_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Compare incoming feature distributions against reference stats.
    Returns dict: { feature: {psi, status, mean_shift, std_ratio} }
    """
    log.info("─── Data Drift Check ───────────────────────────────────")
    ref = _reference_stats(feature_names)
    results = {}

    for feat in feature_names:
        if feat not in incoming_df.columns:
            continue
        actual_vals = incoming_df[feat].dropna().values
        ref_stats   = ref.get(feat, {})

        # Simulate reference distribution from known stats
        ref_vals = np.random.normal(
            loc=ref_stats.get("mean", 0),
            scale=ref_stats.get("std", 1),
            size=len(actual_vals)
        )
        ref_vals = np.clip(ref_vals, ref_stats.get("min", -np.inf),
                           ref_stats.get("max", np.inf))

        psi_val    = _psi(ref_vals, actual_vals)
        mean_shift = abs(actual_vals.mean() - ref_stats.get("mean", actual_vals.mean()))
        std_ratio  = actual_vals.std() / (ref_stats.get("std", 1) + 1e-9)

        if psi_val < 0.10:
            status = "✅ STABLE"
        elif psi_val < 0.25:
            status = "⚠️  MINOR DRIFT"
        else:
            status = "🚨 MAJOR DRIFT"

        results[feat] = {
            "psi":        round(psi_val, 4),
            "mean_shift": round(mean_shift, 4),
            "std_ratio":  round(std_ratio, 4),
            "status":     status,
        }
        log.info(f"  {feat:30s} PSI={psi_val:.4f}  {status}")

    return results


# ---------------------------------------------------------------------------
# 10.6.2  Accuracy Monitoring
# ---------------------------------------------------------------------------
def check_accuracy(model, scaler, meta) -> dict:
    """
    Report stored validation metrics and compare against KPI thresholds.
    In production, this would evaluate against recent labeled production data.
    """
    log.info("─── Accuracy Monitoring ─────────────────────────────────")
    perf = meta.get("performance", {})

    val_r2   = perf.get("val_r2",   0.0)
    val_rmse = perf.get("val_rmse", 9999)
    val_mae  = perf.get("val_mae",  9999)

    r2_ok   = val_r2   >= 0.90
    rmse_ok = val_rmse <= 5000
    mae_ok  = val_mae  <= 3000

    log.info(f"  val_r2   = {val_r2:.4f}  {'✅' if r2_ok   else '🚨'} (threshold ≥ 0.90)")
    log.info(f"  val_rmse = {val_rmse:.2f}  {'✅' if rmse_ok else '🚨'} (threshold ≤ 5000 kW)")
    log.info(f"  val_mae  = {val_mae:.2f}  {'✅' if mae_ok  else '🚨'} (threshold ≤ 3000 kW)")

    overall = "✅ PASS" if (r2_ok and rmse_ok and mae_ok) else "🚨 FAIL — Consider retraining"
    log.info(f"  Overall Accuracy Status: {overall}")

    return {
        "val_r2":   val_r2,
        "val_rmse": val_rmse,
        "val_mae":  val_mae,
        "r2_ok":    r2_ok,
        "rmse_ok":  rmse_ok,
        "mae_ok":   mae_ok,
        "overall":  overall,
    }


# ---------------------------------------------------------------------------
# 10.6.3  Latency Benchmark
# ---------------------------------------------------------------------------
def check_latency(model, scaler, meta, n_runs: int = 100) -> dict:
    """
    Benchmark single-sample inference latency over n_runs iterations.
    KPI: p50 < 50ms, p95 < 100ms.
    """
    log.info("─── Latency Benchmark ───────────────────────────────────")
    feature_names = meta.get("feature_names", [])
    sample = pd.DataFrame(
        [[25.5, 35.2, 0.8][:len(feature_names)]],
        columns=feature_names
    )

    latencies = []
    for _ in range(n_runs):
        t0   = time.perf_counter()
        X_sc = scaler.transform(sample)
        _    = model.predict(X_sc)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    p50_ok = p50 < 50
    p95_ok = p95 < 100

    log.info(f"  Runs     : {n_runs}")
    log.info(f"  p50      : {p50:.2f} ms  {'✅' if p50_ok else '⚠️'} (threshold < 50ms)")
    log.info(f"  p95      : {p95:.2f} ms  {'✅' if p95_ok else '⚠️'} (threshold < 100ms)")
    log.info(f"  p99      : {p99:.2f} ms")

    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
            "p50_ok": p50_ok, "p95_ok": p95_ok}


# ---------------------------------------------------------------------------
# Full Report
# ---------------------------------------------------------------------------
def run_full_report():
    log.info("=" * 60)
    log.info("  SOLAR YIELD — MODEL MONITORING REPORT")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    model, scaler, meta = load_artifacts()

    log.info(f"  Model   : {meta.get('model_name')} v{meta.get('version', '?')}")
    log.info(f"  Trained : {meta.get('trained_date', 'unknown')}")
    log.info("")

    # Simulate recent incoming data (would be real prod data in production)
    np.random.seed(0)
    feature_names = meta.get("feature_names", [])
    incoming = pd.DataFrame({
        "AMBIENT_TEMPERATURE": np.random.normal(28, 8, 200).clip(10, 45),
        "MODULE_TEMPERATURE":  np.random.normal(38, 10, 200).clip(15, 60),
        "IRRADIATION":         np.random.uniform(0, 1, 200),
    })[feature_names]

    drift_results    = check_drift(incoming, feature_names)
    accuracy_results = check_accuracy(model, scaler, meta)
    latency_results  = check_latency(model, scaler, meta)

    # Summary
    log.info("")
    log.info("--- SUMMARY ---------------------------------------------")
    all_stable     = all("STABLE" in v["status"] for v in drift_results.values())
    accuracy_ok    = accuracy_results["r2_ok"]
    latency_ok     = latency_results["p95_ok"]
    overall_health = "HEALTHY" if (all_stable and accuracy_ok and latency_ok) else "ACTION REQUIRED - Data Drift Detected"
    log.info(f"  Data Drift    : {'No drift detected' if all_stable else 'Drift detected'}")
    log.info(f"  Model Accuracy: {accuracy_results['overall']}")
    log.info(f"  Latency p95   : {latency_results['p95_ms']:.2f} ms - {'OK' if latency_ok else 'Slow'}")
    log.info(f"  System Health : {overall_health}")
    log.info("=" * 60)

    # Persist report
    report = {
        "timestamp":  datetime.now().isoformat(),
        "model":      meta.get("model_name"),
        "drift":      drift_results,
        "accuracy":   accuracy_results,
        "latency":    latency_results,
        "health":     overall_health,
    }
    report_path = os.path.join(LOG_DIR, "monitoring_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Report saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Monitoring")
    parser.add_argument("--drift",   action="store_true", help="Run drift check only")
    parser.add_argument("--latency", action="store_true", help="Run latency benchmark only")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    model, scaler, meta = load_artifacts()
    feature_names = meta.get("feature_names", [])

    if args.drift:
        np.random.seed(0)
        incoming = pd.DataFrame({
            "AMBIENT_TEMPERATURE": np.random.normal(28, 8, 200).clip(10, 45),
            "MODULE_TEMPERATURE":  np.random.normal(38, 10, 200).clip(15, 60),
            "IRRADIATION":         np.random.uniform(0, 1, 200),
        })[feature_names]
        check_drift(incoming, feature_names)
    elif args.latency:
        check_latency(model, scaler, meta)
    else:
        run_full_report()
