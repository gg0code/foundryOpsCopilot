"""
Foundry AI Demo — Model Training

Trains four production-grade models on the synthetic FG260 heat dataset:

    1. Defect Classifier   (XGBoost, 9 classes + None)   target accuracy >82%
    2. Severity Classifier (XGBoost, 4 classes)          target accuracy >78%
    3. Yield Regressor     (LightGBM)                    target R²       >0.75
    4. Warranty Regressor  (LightGBM)                    target R²       >0.70

A SHAP explainer is fit on the defect classifier for root-cause attribution
(used by the demo page's root cause panel).

All models, the feature schema, and a metrics report are written to /models.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
import shap

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

NUMERIC_FEATURES = [
    "pattern_age_cycles",
    "ambient_temp_C",
    "ambient_humidity_pct",
    "sand_moisture_pct",
    "mold_hardness_B",
    "mold_temp_C",
    "binder_pct",
    "mold_permeability",
    "C_pct", "Si_pct", "Mn_pct", "S_pct", "P_pct", "Cr_pct",
    "carbon_equivalent",
    "pour_temp_C",
    "superheat_C",
    "pour_rate_kg_per_s",
    "pour_delay_min",
    "cooling_rate_C_per_min",
    "fade_time_min",
    "inoculant_dose_pct",
    "core_moisture_pct",
    "furnace_temp_drift_C",
    "bore_diameter_mm",
    "deck_height_mm",
    "wall_thickness_mm",
    "surface_roughness_Ra_um",
    "casting_weight_kg",
]

CATEGORICAL_FEATURES = [
    "shift", "furnace_id", "mold_line_id", "pattern_id", "season",
]

BOOL_FEATURES = ["mn_s_ratio_ok"]


def _sanitize(name: str) -> str:
    """LightGBM trips on '-' and other JSON-special chars in feature names."""
    return (
        name.replace("-", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
    )


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals; return a numeric design matrix + column names."""
    X = df[NUMERIC_FEATURES + BOOL_FEATURES].copy()
    X["mn_s_ratio_ok"] = X["mn_s_ratio_ok"].astype(int)

    cats = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES, dtype=int)
    X = pd.concat([X.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    X.columns = [_sanitize(c) for c in X.columns]
    feature_names = list(X.columns)
    return X, feature_names


# ------------------------------------------------------------------
# Trainers
# ------------------------------------------------------------------

def _class_balanced_weights(y):
    """Weight inversely proportional to class frequency (rare classes get more pull)."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n = len(y)
    n_classes = len(classes)
    # sklearn 'balanced' formula
    w = np.array([n / (n_classes * freq[v]) for v in y])
    return w


def _balanced_accuracy(y_true, y_pred, n_classes):
    """Macro-averaged per-class recall. Robust to imbalance."""
    per_class = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        per_class.append((y_pred[mask] == c).mean())
    return float(np.mean(per_class))


def train_defect_classifier(X_train, X_test, y_train, y_test, le: LabelEncoder):
    print("\n[1/4] Training Defect Classifier (XGBoost, 10-class)...")
    t0 = time.time()
    sample_w = _class_balanced_weights(y_train)
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_weight=2,
        objective="multi:softprob",
        num_class=len(le.classes_),
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_w)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal = _balanced_accuracy(y_test, y_pred, len(le.classes_))
    print(f"  Defect classifier accuracy: {acc*100:.2f}%  |  balanced: {bal*100:.2f}%   ({time.time()-t0:.1f}s)")
    print(f"  Target: >82%  →  {'PASS' if acc > 0.82 else 'CHECK'}")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))
    return model, acc, bal, cm.tolist()


def train_severity_classifier(X_train, X_test, y_train, y_test, le: LabelEncoder):
    print("\n[2/4] Training Severity Classifier (XGBoost, 4-class)...")
    t0 = time.time()
    sample_w = _class_balanced_weights(y_train)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_weight=2,
        objective="multi:softprob",
        num_class=len(le.classes_),
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_w)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal = _balanced_accuracy(y_test, y_pred, len(le.classes_))
    print(f"  Severity classifier accuracy: {acc*100:.2f}%  |  balanced: {bal*100:.2f}%   ({time.time()-t0:.1f}s)")
    print(f"  Target: >78%  →  {'PASS' if acc > 0.78 else 'CHECK'}")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))
    return model, acc, bal, cm.tolist()


def train_yield_regressor(X_train, X_test, y_train, y_test, w_train):
    print("\n[3/4] Training Yield Regressor (LightGBM)...")
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=600,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_samples=10,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Yield R²: {r2:.3f}   MAE: {mae:.2f}%   ({time.time()-t0:.1f}s)")
    print(f"  Target: R²>0.75  →  {'PASS' if r2 > 0.75 else 'CHECK'}")
    return model, r2, mae


def train_warranty_regressor(X_train, X_test, y_train, y_test, w_train):
    print("\n[4/4] Training Warranty Risk Regressor (LightGBM)...")
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=600,
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_samples=10,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Warranty R²: {r2:.3f}   MAE: {mae:.2f}   ({time.time()-t0:.1f}s)")
    print(f"  Target: R²>0.70  →  {'PASS' if r2 > 0.70 else 'CHECK'}")
    return model, r2, mae


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FOUNDRY AI DEMO — MODEL TRAINING")
    print("=" * 60)

    csv_path = DATA_DIR / "heats_2025.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run generate_data.py first. Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    # "None" gets auto-parsed to NaN — restore as string label
    df["defect_class"] = df["defect_class"].fillna("None").astype(str)
    df["severity"] = df["severity"].fillna("None").astype(str)
    print(f"Loaded {len(df):,} heats from {csv_path.name}")

    X, feature_names = build_features(df)

    # --- Defect targets ---
    defect_le = LabelEncoder()
    y_defect = defect_le.fit_transform(df["defect_class"])
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X, y_defect, test_size=0.2, random_state=SEED, stratify=y_defect)
    defect_model, defect_acc, defect_bal, defect_cm = train_defect_classifier(Xd_tr, Xd_te, yd_tr, yd_te, defect_le)

    # --- Severity targets ---
    sev_le = LabelEncoder()
    y_sev = sev_le.fit_transform(df["severity"])
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X, y_sev, test_size=0.2, random_state=SEED, stratify=y_sev)
    sev_model, sev_acc, sev_bal, sev_cm = train_severity_classifier(Xs_tr, Xs_te, ys_tr, ys_te, sev_le)

    # --- Yield regression ---
    # Stratify by severity to keep test set representative across all classes
    y_yield = df["yield_pct"].values
    Xy_tr, Xy_te, yy_tr, yy_te = train_test_split(
        X, y_yield, test_size=0.2, random_state=SEED, stratify=y_sev
    )
    yield_model, yield_r2, yield_mae = train_yield_regressor(Xy_tr, Xy_te, yy_tr, yy_te, None)

    # --- Warranty regression ---
    y_warr = df["warranty_risk_score"].values
    Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
        X, y_warr, test_size=0.2, random_state=SEED, stratify=y_sev
    )
    warr_model, warr_r2, warr_mae = train_warranty_regressor(Xw_tr, Xw_te, yw_tr, yw_te, None)

    # --- SHAP explainer on defect model ---
    print("\nFitting SHAP TreeExplainer on defect classifier...")
    t0 = time.time()
    explainer = shap.TreeExplainer(defect_model)
    # quick sanity sample
    _ = explainer.shap_values(X.iloc[:5])
    print(f"  SHAP explainer ready ({time.time()-t0:.1f}s)")

    # --- Persist artifacts ---
    print("\nSaving models...")
    joblib.dump(defect_model, MODELS_DIR / "defect_classifier.pkl")
    joblib.dump(sev_model,    MODELS_DIR / "severity_classifier.pkl")
    joblib.dump(yield_model,  MODELS_DIR / "yield_regressor.pkl")
    joblib.dump(warr_model,   MODELS_DIR / "warranty_regressor.pkl")
    joblib.dump(explainer,    MODELS_DIR / "shap_explainer.pkl")
    joblib.dump(defect_le,    MODELS_DIR / "defect_label_encoder.pkl")
    joblib.dump(sev_le,       MODELS_DIR / "severity_label_encoder.pkl")

    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    metrics = {
        "defect_classifier": {
            "accuracy": round(defect_acc, 4),
            "balanced_accuracy": round(defect_bal, 4),
            "target": 0.82,
            "classes": list(defect_le.classes_),
            "confusion_matrix": defect_cm,
        },
        "severity_classifier": {
            "accuracy": round(sev_acc, 4),
            "balanced_accuracy": round(sev_bal, 4),
            "target": 0.78,
            "classes": list(sev_le.classes_),
            "confusion_matrix": sev_cm,
        },
        "yield_regressor": {
            "r2": round(float(yield_r2), 4),
            "mae_pct": round(float(yield_mae), 4),
            "target_r2": 0.75,
        },
        "warranty_regressor": {
            "r2": round(float(warr_r2), 4),
            "mae": round(float(warr_mae), 4),
            "target_r2": 0.70,
        },
        "n_train_rows": int(len(Xd_tr)),
        "n_test_rows": int(len(Xd_te)),
        "n_features": len(feature_names),
        "seed": SEED,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED")
    print("=" * 60)
    print(f"  Defect classifier:    {defect_acc*100:.2f}% accuracy  (target 82%)")
    print(f"  Severity classifier:  {sev_acc*100:.2f}% accuracy  (target 78%)")
    print(f"  Yield regressor:      R² = {yield_r2:.3f}            (target 0.75)")
    print(f"  Warranty regressor:   R² = {warr_r2:.3f}            (target 0.70)")
    print(f"\nArtifacts written to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
