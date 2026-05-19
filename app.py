"""
FoundryOps Copilot — Flask backend.

Routes:
    GET  /                          Landing page (narrative + assumptions panel)
    GET  /demo                      Interactive prediction demo
    GET  /analytics                 Analytics dashboard
    GET  /capabilities              Descriptive page covering the four AI capabilities
    POST /api/predict               Predict defect + yield + warranty from heat parameters
    GET  /api/assumptions           Full metallurgical + cost assumptions JSON
    GET  /api/dictionary            Parsed data dictionary CSV
    GET  /api/model_info            Model accuracy + feature names + class lists
    GET  /api/analytics/*           Aggregated analytics endpoints (see route definitions)
    GET  /api/presets               Current / Now / AI-Optimal preset payloads
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from pathlib import Path
from threading import Timer

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

app = Flask(__name__, template_folder="templates", static_folder="static")
# Auto-reload templates so edits to *.html files are picked up without server restart.
# (debug=False otherwise caches templates indefinitely.)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
CORS(app)

# ------------------------------------------------------------------
# Load assets at startup
# ------------------------------------------------------------------
print("==> Loading assumptions & models...")
ASSUMPTIONS = json.loads((DATA_DIR / "metallurgical_assumptions.json").read_text(encoding="utf-8"))
COSTS = json.loads((DATA_DIR / "cost_assumptions.json").read_text(encoding="utf-8"))

DF = pd.read_csv(DATA_DIR / "heats_2025.csv")
DF["defect_class"] = DF["defect_class"].fillna("None").astype(str)
DF["severity"] = DF["severity"].fillna("None").astype(str)
DF["timestamp"] = pd.to_datetime(DF["timestamp"])
DF["month"] = DF["timestamp"].dt.month

defect_model = joblib.load(MODELS_DIR / "defect_classifier.pkl")
severity_model = joblib.load(MODELS_DIR / "severity_classifier.pkl")
yield_model = joblib.load(MODELS_DIR / "yield_regressor.pkl")
warranty_model = joblib.load(MODELS_DIR / "warranty_regressor.pkl")
shap_explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
defect_le = joblib.load(MODELS_DIR / "defect_label_encoder.pkl")
severity_le = joblib.load(MODELS_DIR / "severity_label_encoder.pkl")
FEATURE_NAMES = json.loads((MODELS_DIR / "feature_names.json").read_text(encoding="utf-8"))
METRICS = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))

try:
    VERIFICATION_REPORT = (DATA_DIR / "verification_report.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    VERIFICATION_REPORT = "(Run `python generate_data.py --verify` to populate this report.)"

print(f"==> Loaded {len(DF):,} historical heats and 4 trained models.")


# ------------------------------------------------------------------
# Feature vector assembly (mirrors train_model.py)
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


def _sanitize(name: str) -> str:
    return (name.replace("-", "_").replace(" ", "_").replace("(", "")
                .replace(")", "").replace("/", "_"))


def vectorize(payload: dict) -> pd.DataFrame:
    """Convert a slider payload to the trained feature schema (one-hot encoded)."""
    row = {k: 0 for k in FEATURE_NAMES}

    # Numeric features (use payload value or fall back to dataset median)
    for f in NUMERIC_FEATURES:
        row[f] = float(payload.get(f, float(DF[f].median())))

    # Boolean
    row["mn_s_ratio_ok"] = int(bool(payload.get("mn_s_ratio_ok", True)))

    # Derived: carbon_equivalent if not provided
    if "carbon_equivalent" not in payload or payload.get("carbon_equivalent") is None:
        row["carbon_equivalent"] = row["C_pct"] + (row["Si_pct"] + row["P_pct"]) / 3.0

    # Derived: superheat if not provided
    if "superheat_C" not in payload or payload.get("superheat_C") is None:
        row["superheat_C"] = row["pour_temp_C"] - ASSUMPTIONS["pour_temperature"]["liquidus_C"]

    # One-hot categoricals
    shift = payload.get("shift", "A")
    for s in ("A", "B", "C"):
        col = _sanitize(f"shift_{s}")
        if col in row:
            row[col] = 1 if s == shift else 0

    furnace = payload.get("furnace_id", "F1")
    for f in ("F1", "F2"):
        col = _sanitize(f"furnace_id_{f}")
        if col in row:
            row[col] = 1 if f == furnace else 0

    mold_line = payload.get("mold_line_id", "ML-1")
    for ml in ("ML-1", "ML-2"):
        col = _sanitize(f"mold_line_id_{ml}")
        if col in row:
            row[col] = 1 if ml == mold_line else 0

    pattern = payload.get("pattern_id", "PT-A")
    for p in ("PT-A", "PT-B", "PT-C"):
        col = _sanitize(f"pattern_id_{p}")
        if col in row:
            row[col] = 1 if p == pattern else 0

    season = payload.get("season", "Winter")
    for s in ("Winter", "Pre_monsoon", "Monsoon", "Post_monsoon"):
        col = _sanitize(f"season_{s}")
        if col in row:
            row[col] = 1 if s == season else 0

    return pd.DataFrame([row])[FEATURE_NAMES]


# ------------------------------------------------------------------
# Business cost calculation
# ------------------------------------------------------------------

def compute_business_impact(severity_probs: dict, customer_complaint_p: float,
                            yield_pct: float, warranty_score: float) -> dict:
    """Convert model outputs into an expected per-casting business cost (₹)."""
    c = COSTS

    p_minor = severity_probs.get("Minor_Rework", 0)
    p_major = severity_probs.get("Major_Rework", 0)
    p_scrap = severity_probs.get("Scrap", 0)
    p_none  = severity_probs.get("None", 0)

    rework_cost = (p_minor + p_major * 0.6) * c["rework_cost_per_casting"]
    scrap_cost = (p_scrap + p_major * 0.4) * c["scrap_cost_per_casting"]

    delay_minutes = (p_minor * 8 + p_major * 18 + p_scrap * 35)
    delay_cost = delay_minutes * c["delay_cost_per_minute"]

    complaint_cost = customer_complaint_p * c["customer_complaint_cost"]
    warranty_cost = (warranty_score / 10.0) * c["warranty_escalation_rate"] * c["warranty_claim_cost"]

    total = rework_cost + scrap_cost + delay_cost + complaint_cost + warranty_cost

    return {
        "scrap_cost":      round(scrap_cost, 0),
        "rework_cost":     round(rework_cost, 0),
        "delay_cost":      round(delay_cost, 0),
        "complaint_cost":  round(complaint_cost, 0),
        "warranty_cost":   round(warranty_cost, 0),
        "total_cost":      round(total, 0),
        "annualized":      round(total * c["annual_volume_target_castings"], 0),
    }


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/capabilities")
def capabilities():
    return render_template("capabilities.html")


@app.route("/assumptions")
def assumptions_page():
    """Structured HTML view of the metallurgical + cost assumptions."""
    return render_template("assumptions.html",
                           assumptions=ASSUMPTIONS, costs=COSTS,
                           verification=VERIFICATION_REPORT)


# ------------------------------------------------------------------
# API: prediction
# ------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    payload = request.get_json(force=True)
    X = vectorize(payload)

    # Defect classification
    defect_probs_arr = defect_model.predict_proba(X)[0]
    defect_classes = list(defect_le.classes_)
    defect_probs = {c: float(p) for c, p in zip(defect_classes, defect_probs_arr)}
    p_any_defect = 1.0 - defect_probs.get("None", 0.0)

    # Top non-None defect class (what the operator would investigate)
    non_none = [(c, p) for c, p in defect_probs.items() if c != "None"]
    non_none.sort(key=lambda x: -x[1])
    top_defect_class, top_defect_p = (non_none[0] if non_none else ("None", 0.0))

    # Severity prediction
    sev_probs_arr = severity_model.predict_proba(X)[0]
    sev_classes = list(severity_le.classes_)
    sev_probs = {c: float(p) for c, p in zip(sev_classes, sev_probs_arr)}
    top_severity = max(sev_probs.items(), key=lambda kv: kv[1])[0]

    # Disposition derived from severity
    if top_severity == "None":
        disposition = "OK"
    elif top_severity in ("Minor_Rework", "Major_Rework"):
        disposition = "Rework"
    else:
        disposition = "Scrap"

    # Yield + warranty regression
    yield_pct = float(yield_model.predict(X)[0])
    warranty_risk = float(warranty_model.predict(X)[0])

    # Customer complaint probability — proportional to escape risk * severity
    customer_complaint_p = min(1.0, p_any_defect * 0.05 + (warranty_risk / 10.0) * 0.05)

    # Business impact
    impact = compute_business_impact(sev_probs, customer_complaint_p, yield_pct, warranty_risk)

    # SHAP root cause attribution for the top non-None class
    try:
        shap_values = shap_explainer.shap_values(X)
        # XGBoost multi-class returns array of shape (n_classes, n_features) per row
        # depending on shap version. Normalise.
        if isinstance(shap_values, list):
            sv = np.array(shap_values)  # (n_classes, 1, n_features)
        else:
            sv = np.array(shap_values)
        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
            if sv.shape[0] == len(defect_classes):
                sv_row = sv[:, 0, :]      # (n_classes, n_features)
            else:
                sv_row = sv[0]            # (n_features, n_classes)
                sv_row = sv_row.T         # (n_classes, n_features)
        else:
            sv_row = sv

        cls_idx = defect_classes.index(top_defect_class) if top_defect_class in defect_classes else 0
        contribs = sv_row[cls_idx]
        contrib_pairs = sorted(zip(FEATURE_NAMES, contribs), key=lambda kv: -abs(float(kv[1])))[:6]
        root_causes = [
            {"feature": f, "shap_value": float(v), "direction": "increases" if float(v) > 0 else "decreases"}
            for f, v in contrib_pairs
        ]
    except Exception as ex:
        root_causes = [{"feature": "unavailable", "shap_value": 0.0, "direction": "n/a", "error": str(ex)}]

    latency_ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        "defect_probabilities": defect_probs,
        "top_defect_class": top_defect_class,
        "top_defect_probability": float(top_defect_p),
        "p_any_defect": float(p_any_defect),
        "risk_gauge_pct": float(min(100, p_any_defect * 100)),
        "severity_probabilities": sev_probs,
        "predicted_severity": top_severity,
        "disposition": disposition,
        "predicted_yield_pct": round(yield_pct, 2),
        "predicted_warranty_risk": round(warranty_risk, 2),
        "customer_complaint_p": round(float(customer_complaint_p), 4),
        "business_impact": impact,
        "root_causes": root_causes,
        "latency_ms": round(latency_ms, 1),
    })


# ------------------------------------------------------------------
# API: assumptions / model info
# ------------------------------------------------------------------

@app.route("/api/assumptions")
def get_assumptions():
    return jsonify({
        "metallurgical": ASSUMPTIONS,
        "costs": COSTS,
        "verification_report": VERIFICATION_REPORT,
    })


@app.route("/api/dictionary")
def get_dictionary():
    """Parsed data dictionary CSV — column / type / range / meaning."""
    rows = pd.read_csv(DATA_DIR / "data_dictionary.csv").to_dict(orient="records")
    return jsonify(rows)


@app.route("/api/model_info")
def get_model_info():
    return jsonify({
        "metrics": METRICS,
        "feature_names": FEATURE_NAMES,
        "defect_classes": list(defect_le.classes_),
        "severity_classes": list(severity_le.classes_),
        "n_training_rows": len(DF),
    })


# ------------------------------------------------------------------
# API: analytics
# ------------------------------------------------------------------

@app.route("/api/analytics/overview")
def analytics_overview():
    total = len(DF)
    n_scrap = int((DF["disposition"] == "Scrap").sum())
    n_rework = int((DF["disposition"] == "Rework").sum())
    n_complaint = int((DF["disposition"] == "Customer_Reject").sum())
    scrap_cost = n_scrap * COSTS["scrap_cost_per_casting"]
    rework_cost = n_rework * COSTS["rework_cost_per_casting"]
    complaint_cost = n_complaint * COSTS["customer_complaint_cost"]
    warranty_cost = n_complaint * COSTS["warranty_escalation_rate"] * COSTS["warranty_claim_cost"]
    total_cost = scrap_cost + rework_cost + complaint_cost + warranty_cost

    return jsonify({
        "total_heats": total,
        "scrap_rate_pct":   round(n_scrap / total * 100, 2),
        "rework_rate_pct":  round(n_rework / total * 100, 2),
        "complaint_rate_pct": round(n_complaint / total * 100, 3),
        "scrap_cost":     int(scrap_cost),
        "rework_cost":    int(rework_cost),
        "complaint_cost": int(complaint_cost),
        "warranty_cost":  int(warranty_cost),
        "total_cost":     int(total_cost),
        "annual_savings_target": COSTS["annual_savings_inr"],
        "annual_savings_label": COSTS["annual_savings_label"],
    })


@app.route("/api/analytics/pareto")
def analytics_pareto():
    """Cost-weighted Pareto of defect classes."""
    sc = COSTS["scrap_cost_per_casting"]
    rc = COSTS["rework_cost_per_casting"]
    cc = COSTS["customer_complaint_cost"]
    rows = []
    for cls, sub in DF[DF["defect_class"] != "None"].groupby("defect_class"):
        scrap = int((sub["disposition"] == "Scrap").sum())
        rework = int((sub["disposition"] == "Rework").sum())
        comp = int((sub["disposition"] == "Customer_Reject").sum())
        cost = scrap * sc + rework * rc + comp * cc
        rows.append({"defect_class": cls, "count": int(len(sub)), "cost_inr": int(cost),
                     "scrap": scrap, "rework": rework, "complaint": comp})
    rows.sort(key=lambda r: -r["cost_inr"])
    return jsonify(rows)


@app.route("/api/analytics/monthly")
def analytics_monthly():
    rows = []
    for m, sub in DF.groupby("month"):
        rows.append({
            "month": int(m),
            "heats": int(len(sub)),
            "scrap_rate_pct":  round((sub["disposition"] == "Scrap").mean() * 100, 2),
            "rework_rate_pct": round((sub["disposition"] == "Rework").mean() * 100, 2),
            "complaint_rate_pct": round((sub["disposition"] == "Customer_Reject").mean() * 100, 3),
            "avg_humidity_pct": round(float(sub["ambient_humidity_pct"].mean()), 1),
            "porosity_rate_pct": round((sub["defect_class"] == "Gas_Porosity").mean() * 100, 2),
        })
    rows.sort(key=lambda r: r["month"])
    return jsonify(rows)


def _cpk(series, usl, lsl):
    mu = series.mean()
    sigma = series.std()
    if sigma <= 0:
        return None
    return float(min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma)))


@app.route("/api/analytics/dimensional")
def analytics_dimensional():
    tols = ASSUMPTIONS["dimensional_tolerances"]
    out = {}
    for label, col in (
        ("Bore_Diameter",  ("bore_diameter_mm", tols["Bore_Diameter_mm"])),
        ("Deck_Height",    ("deck_height_mm",   tols["Deck_Height_mm"])),
        ("Wall_Thickness", ("wall_thickness_mm", tols["Wall_Thickness_mm"])),
    ):
        column, tol = col
        s = DF[column]
        bins = 30
        counts, edges = np.histogram(s, bins=bins)
        out[label] = {
            "values": s.tolist()[:500],   # cap for payload size
            "bins":   [round(float(x), 4) for x in edges],
            "counts": [int(x) for x in counts],
            "mean":   round(float(s.mean()), 4),
            "std":    round(float(s.std()), 4),
            "USL":    tol["USL"],
            "LSL":    tol["LSL"],
            "target": tol["target"],
            "Cpk":    round(_cpk(s, tol["USL"], tol["LSL"]), 3),
            "Cpk_target": 1.33,
        }
    return jsonify(out)


@app.route("/api/analytics/breakdowns")
def analytics_breakdowns():
    def split(col):
        rows = []
        for k, sub in DF.groupby(col):
            rows.append({
                "key": str(k),
                "heats": int(len(sub)),
                "scrap_rate_pct": round((sub["disposition"] == "Scrap").mean() * 100, 2),
                "rework_rate_pct": round((sub["disposition"] == "Rework").mean() * 100, 2),
                "complaint_rate_pct": round((sub["disposition"] == "Customer_Reject").mean() * 100, 3),
                "defect_rate_pct": round((sub["defect_class"] != "None").mean() * 100, 2),
            })
        return rows
    return jsonify({
        "shift":     split("shift"),
        "furnace":   split("furnace_id"),
        "mold_line": split("mold_line_id"),
        "season":    split("season"),
        "pattern":   split("pattern_id"),
    })


@app.route("/api/analytics/pattern_wear")
def analytics_pattern_wear():
    buckets = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200), (1200, 1500)]
    rows = []
    for lo, hi in buckets:
        sub = DF[(DF["pattern_age_cycles"] >= lo) & (DF["pattern_age_cycles"] < hi)]
        if len(sub) == 0:
            continue
        rows.append({
            "bucket": f"{lo}-{hi}",
            "midpoint": (lo + hi) / 2,
            "heats": int(len(sub)),
            "dim_nc_rate_pct": round((sub["defect_class"] == "Dimensional_NC").mean() * 100, 2),
            "any_defect_rate_pct": round((sub["defect_class"] != "None").mean() * 100, 2),
        })
    return jsonify(rows)


@app.route("/api/analytics/correlations")
def analytics_correlations():
    """Pearson correlations of top numeric features with 'any defect' indicator."""
    is_defect = (DF["defect_class"] != "None").astype(int)
    corrs = []
    for f in NUMERIC_FEATURES:
        c = float(DF[f].corr(is_defect))
        if np.isfinite(c):
            corrs.append({"feature": f, "correlation": round(c, 4)})
    corrs.sort(key=lambda r: -abs(r["correlation"]))
    return jsonify(corrs[:15])


@app.route("/api/analytics/sample")
def analytics_sample():
    n = int(request.args.get("n", 25))
    cols = [
        "heat_id", "timestamp", "shift", "furnace_id", "pattern_id",
        "season", "ambient_humidity_pct", "sand_moisture_pct",
        "pour_temp_C", "pattern_age_cycles", "defect_class", "severity", "disposition",
    ]
    sample = DF[cols].head(n).copy()
    sample["timestamp"] = sample["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    return jsonify(sample.to_dict(orient="records"))


# ------------------------------------------------------------------
# API: presets
# ------------------------------------------------------------------

@app.route("/api/presets")
def get_presets():
    """Two slider presets driving the Typical heat → AI Recommended demo moment."""
    presets = {
        "typical": {
            "label": "Typical heat",
            "description": "A typical day at the plant — aged pattern, monsoon humidity, drifted furnace.",
            "params": {
                "shift": "B", "furnace_id": "F1", "mold_line_id": "ML-1",
                "pattern_id": "PT-C", "pattern_age_cycles": 1100, "season": "Monsoon",
                "ambient_temp_C": 30, "ambient_humidity_pct": 88,
                "sand_moisture_pct": 5.2, "mold_hardness_B": 79, "mold_temp_C": 44,
                "binder_pct": 2.0, "mold_permeability": 92,
                "C_pct": 3.30, "Si_pct": 2.10, "Mn_pct": 0.62, "S_pct": 0.10,
                "P_pct": 0.085, "Cr_pct": 0.03,
                "pour_temp_C": 1395, "pour_rate_kg_per_s": 6.8, "pour_delay_min": 6.5,
                "cooling_rate_C_per_min": 14.0, "fade_time_min": 8,
                "inoculant_dose_pct": 0.18, "core_moisture_pct": 3.8,
                "furnace_temp_drift_C": 12.5,
                "bore_diameter_mm": 89.493, "deck_height_mm": 220.00,
                "wall_thickness_mm": 4.50, "surface_roughness_Ra_um": 5.5,
                "casting_weight_kg": 215,
                "mn_s_ratio_ok": True,
            },
        },
        "ai_recommended": {
            "label": "AI Recommended setpoint",
            "description": "Setpoints recommended by the AI for this batch.",
            "params": {
                "shift": "A", "furnace_id": "F2", "mold_line_id": "ML-1",
                "pattern_id": "PT-A", "pattern_age_cycles": 200, "season": "Winter",
                "ambient_temp_C": 22, "ambient_humidity_pct": 55,
                "sand_moisture_pct": 3.2, "mold_hardness_B": 86, "mold_temp_C": 32,
                "binder_pct": 1.95, "mold_permeability": 128,
                "C_pct": 3.45, "Si_pct": 2.15, "Mn_pct": 0.75, "S_pct": 0.08,
                "P_pct": 0.06, "Cr_pct": 0.02,
                "pour_temp_C": 1418, "pour_rate_kg_per_s": 7.8, "pour_delay_min": 1.2,
                "cooling_rate_C_per_min": 10.0, "fade_time_min": 3,
                "inoculant_dose_pct": 0.22, "core_moisture_pct": 2.8,
                "furnace_temp_drift_C": 5.0,
                "bore_diameter_mm": 89.500, "deck_height_mm": 220.00,
                "wall_thickness_mm": 4.50, "surface_roughness_Ra_um": 4.5,
                "casting_weight_kg": 215,
                "mn_s_ratio_ok": True,
            },
        },
    }
    return jsonify(presets)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def _open_browser():
    webbrowser.open_new("http://localhost:5000/")


if __name__ == "__main__":
    # Open browser ~1s after server starts (skip on reloader child)
    import os as _os
    if _os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1.0, _open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
