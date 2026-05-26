"""
FoundryOps Copilot — Flask backend.

Routes:
    GET  /                          Landing page (narrative + assumptions panel)
    GET  /demo                      Interactive prediction demo
    GET  /analytics                 Analytics dashboard
    GET  /capabilities              Descriptive page covering the four AI capabilities
    GET  /ask                       Conversational assistant page (AI That Converses)
    POST /api/predict               Predict defect + yield + warranty from heat parameters
    POST /api/ask                   Answer foundry questions with Claude (grounded in dataset summary)
    GET  /api/assumptions           Full metallurgical + cost assumptions JSON
    GET  /api/dictionary            Parsed data dictionary CSV
    GET  /api/model_info            Model accuracy + feature names + class lists
    GET  /api/analytics/*           Aggregated analytics endpoints (see route definitions)
    GET  /api/presets               Current / Now / AI-Optimal preset payloads
"""

from __future__ import annotations

import json
import os
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

# Load a project-root .env (git-ignored) if python-dotenv is installed, so keys
# like GROQ_API_KEY / ANTHROPIC_API_KEY can live in foundry/.env instead of the
# shell. Real environment variables still take precedence and override the file.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

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
# Conversational assistant (/ask) — "AI That Converses"
#
# Answers foundry questions with Claude, grounded in a deterministic summary
# of heats_2025.csv. The summary is built ONCE at startup and placed in the
# cached system prompt (prompt caching), so the volatile content per request
# is only the user's conversation. Degrades gracefully: if the anthropic SDK
# isn't installed or ANTHROPIC_API_KEY is unset, the rest of the demo is
# unaffected and /api/ask returns a friendly "not configured" message.
# ------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-opus-4-7"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_ASK_MESSAGES = 20        # cap conversation length sent to the API
MAX_ASK_CHARS = 2000         # cap per-message length

try:
    import anthropic
except ImportError:           # SDK optional — demo runs without it
    anthropic = None

try:
    import openai
except ImportError:           # SDK optional — only needed for the Groq path
    openai = None


def _init_assistant():
    """Pick a chat backend from whatever key is present.

    Preference order: Anthropic (Claude) if ANTHROPIC_API_KEY is set, else Groq
    (OpenAI-compatible API) if GROQ_API_KEY is set. Returns (provider, client),
    or (None, None) so the rest of the demo runs unaffected when neither is set.
    """
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return "anthropic", anthropic.Anthropic()
        except Exception as ex:   # bad key format, etc. — don't crash the app
            print(f"==> Claude client init failed ({ex}).")
    if openai is not None and os.environ.get("GROQ_API_KEY"):
        try:
            return "groq", openai.OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
        except Exception as ex:
            print(f"==> Groq client init failed ({ex}).")
    return None, None


def _lakh(x: float) -> str:
    return f"₹{x / 1e5:.1f} L"


def build_ask_context() -> str:
    """Assemble a deterministic, cache-friendly text summary of the heat dataset.

    Built once at startup and placed in the cached system prompt for /api/ask.
    Determinism matters: pandas groupby sorts keys, and there are no timestamps
    or random values, so the rendered bytes are stable across requests and the
    prompt cache prefix stays valid.
    """
    lines: list[str] = []
    n = len(DF)
    t_min = DF["timestamp"].min().strftime("%Y-%m-%d")
    t_max = DF["timestamp"].max().strftime("%Y-%m-%d")

    lines.append("=== DATASET SUMMARY (synthetic; part KE-CYL-V4-220, FG260 IS 210, OEM-TATA) ===")
    lines.append(f"Records: {n:,} heats. Period: {t_min} to {t_max}. ~14 heats/day across 3 shifts.")

    scrap = (DF["disposition"] == "Scrap").mean() * 100
    rework = (DF["disposition"] == "Rework").mean() * 100
    comp = (DF["disposition"] == "Customer_Reject").mean() * 100
    defect = (DF["defect_class"] != "None").mean() * 100
    lines.append(f"Overall: defect rate {defect:.1f}%, scrap {scrap:.1f}%, rework {rework:.1f}%, customer rejects {comp:.2f}%.")

    n_scrap = int((DF["disposition"] == "Scrap").sum())
    n_rework = int((DF["disposition"] == "Rework").sum())
    n_comp = int((DF["disposition"] == "Customer_Reject").sum())
    scrap_cost = n_scrap * COSTS["scrap_cost_per_casting"]
    rework_cost = n_rework * COSTS["rework_cost_per_casting"]
    comp_cost = n_comp * COSTS["customer_complaint_cost"]
    warr_cost = n_comp * COSTS["warranty_escalation_rate"] * COSTS["warranty_claim_cost"]
    total_cost = scrap_cost + rework_cost + comp_cost + warr_cost
    lines.append(
        f"Cost of poor quality across the dataset: scrap {_lakh(scrap_cost)} ({n_scrap} castings), "
        f"rework {_lakh(rework_cost)} ({n_rework}), complaints {_lakh(comp_cost)} ({n_comp}), "
        f"warranty reserve {_lakh(warr_cost)}; total {_lakh(total_cost)}."
    )
    lines.append(
        f"Annual savings opportunity for this part: {COSTS.get('annual_savings_label', '')} "
        f"(~₹{COSTS['annual_savings_inr'] / 1e5:.0f} L)."
    )

    sc, rc, cc = COSTS["scrap_cost_per_casting"], COSTS["rework_cost_per_casting"], COSTS["customer_complaint_cost"]
    pareto = []
    for cls, sub in DF[DF["defect_class"] != "None"].groupby("defect_class"):
        s = int((sub["disposition"] == "Scrap").sum())
        r = int((sub["disposition"] == "Rework").sum())
        c = int((sub["disposition"] == "Customer_Reject").sum())
        pareto.append((cls, len(sub), s * sc + r * rc + c * cc))
    pareto.sort(key=lambda x: -x[2])
    lines.append("Defects by total cost (highest first):")
    for cls, cnt, cost in pareto:
        lines.append(f"  - {cls.replace('_', ' ')}: {cnt} heats, {_lakh(cost)}")

    def breakdown(col: str, label: str) -> list[str]:
        out = [f"{label} (defect rate / scrap rate):"]
        for k, sub in DF.groupby(col):
            dr = (sub["defect_class"] != "None").mean() * 100
            sr = (sub["disposition"] == "Scrap").mean() * 100
            out.append(f"  - {k}: {dr:.1f}% / {sr:.1f}% (n={len(sub)})")
        return out

    lines += breakdown("shift", "By shift")
    lines += breakdown("furnace_id", "By furnace")
    lines += breakdown("season", "By season")
    lines += breakdown("pattern_id", "By pattern")

    lines.append("Monthly defect rate and average humidity:")
    for m, sub in DF.groupby("month"):
        dr = (sub["defect_class"] != "None").mean() * 100
        hum = sub["ambient_humidity_pct"].mean()
        por = (sub["defect_class"] == "Gas_Porosity").mean() * 100
        lines.append(f"  - Month {int(m):02d}: defect {dr:.1f}%, humidity {hum:.0f}%, gas-porosity {por:.1f}%")

    tols = ASSUMPTIONS.get("dimensional_tolerances", {})
    lines.append("Dimensional capability (Cpk; OEM-TATA requires >= 1.33):")
    for label, col, key in (("Bore diameter", "bore_diameter_mm", "Bore_Diameter_mm"),
                            ("Deck height", "deck_height_mm", "Deck_Height_mm"),
                            ("Wall thickness", "wall_thickness_mm", "Wall_Thickness_mm")):
        if key in tols and col in DF:
            mu, sd = DF[col].mean(), DF[col].std()
            if sd > 0:
                cpk = min((tols[key]["USL"] - mu) / (3 * sd), (mu - tols[key]["LSL"]) / (3 * sd))
                lines.append(f"  - {label}: Cpk {cpk:.2f} ({'PASS' if cpk >= 1.33 else 'FAIL'})")

    lines.append("Pattern wear — dimensional-NC rate by pattern age (retire ~1200 cycles; triples past 800):")
    for lo, hi in [(0, 400), (400, 800), (800, 1200), (1200, 1500)]:
        sub = DF[(DF["pattern_age_cycles"] >= lo) & (DF["pattern_age_cycles"] < hi)]
        if len(sub):
            dnc = (sub["defect_class"] == "Dimensional_NC").mean() * 100
            lines.append(f"  - {lo}-{hi} cycles: {dnc:.1f}% dim-NC (n={len(sub)})")

    is_def = (DF["defect_class"] != "None").astype(int)
    corrs = []
    for f in NUMERIC_FEATURES:
        if f in DF:
            cval = DF[f].corr(is_def)
            if np.isfinite(cval):
                corrs.append((f, float(cval)))
    corrs.sort(key=lambda x: -abs(x[1]))
    lines.append("Top correlations with 'any defect':")
    for f, cval in corrs[:10]:
        lines.append(f"  - {f}: {cval:+.2f}")

    lines.append(
        "Key cost assumptions: scrap ₹{:,}/casting, rework ₹{:,}/casting, complaint ₹{:,}/incident, "
        "warranty ₹{:,}/claim ({:.0%} of complaints escalate), delay ₹{:,}/min.".format(
            COSTS["scrap_cost_per_casting"], COSTS["rework_cost_per_casting"], COSTS["customer_complaint_cost"],
            COSTS["warranty_claim_cost"], COSTS["warranty_escalation_rate"], COSTS["delay_cost_per_minute"])
    )
    lines.append(
        "Metallurgy: FG260 grey iron; C 3.1-3.6%, Si 1.8-2.4%, Mn 0.5-0.9%, S 0.06-0.12%; Mn/S rule Mn>=1.7*S+0.3; "
        "carbon equivalent CE=C+(Si+P)/3 target 4.0-4.3; pour temp 1400-1430C (liquidus ~1180C + 220-250C superheat)."
    )
    lines.append(
        "Story facts: Shift A (OP-104, senior) lowest defects; Shift B (OP-217, junior, monsoon-sensitive) highest; "
        "Shift C (OP-308) mid. Furnace F1 runs ~8C hotter than F2 and drifts +12-15C by year-end (lining wear). "
        "Bore Cpk currently FAILS the OEM-TATA 1.33 requirement; Deck and Wall pass."
    )
    lines.append(
        "AI capabilities in this system: defect & severity classifiers (XGBoost), yield & warranty regressors (LightGBM), "
        "SHAP root-cause attribution, continuous anomaly detection (AI That Watches), Bayesian setpoint optimizer "
        "(AI That Acts), and per-prediction confidence bands with human-in-the-loop review (AI That Knows Its Limits)."
    )
    return "\n".join(lines)


ASK_SYSTEM_PREAMBLE = (
    "You are FoundryOps Copilot, Zero Zeta's conversational assistant for a foundry operations demo built for "
    "IndieFoundry. You answer questions about ONE demonstration dataset: the casting-heat history for a single engine "
    "cylinder block (KE-CYL-V4-220, Grey Cast Iron FG260 per IS 210, primary customer OEM-TATA). This is "
    "metallurgically rigorous SYNTHETIC data, not real customer data — be upfront about that if asked.\n\n"
    "Ground every answer in the DATASET SUMMARY below. Rules:\n"
    "- Cite concrete numbers from the summary (rates, ₹ costs, Cpk, correlations). Use ₹ and lakh/crore.\n"
    "- Be concise and direct: a short paragraph or a tight bullet list. This is a live demo — no long essays.\n"
    "- If something isn't in the summary, say what you can reasonably infer and note the limitation; never invent "
    "specific numbers.\n"
    "- For 'what setpoint / how do we fix X' questions, give practical, physics-aware guidance grounded in the "
    "breakdowns (monsoon humidity, pattern wear past 800 cycles, furnace F1 drift, Bore Cpk failing) and frame it as "
    "what the optimizer (AI That Acts) would target — directional advice is fine, but make clear exact setpoints "
    "come from the model on real data.\n"
    "- For 'how confident / when to trust the AI' questions, answer in terms of confidence bands and human-in-the-loop "
    "review (AI That Knows Its Limits): wide bands on rare or extreme conditions get flagged for a metallurgist.\n"
    "- Never promise committed percentage outcomes; describe capabilities and the data, not guarantees.\n"
    "- Plain text or simple markdown (bold, hyphen bullets) only. No tables, no code blocks."
)

ASK_DATA_SUMMARY = build_ask_context()
ASK_SYSTEM_PROMPT = ASK_SYSTEM_PREAMBLE + "\n\n" + ASK_DATA_SUMMARY
ASK_PROVIDER, ask_client = _init_assistant()
_provider_label = {
    "anthropic": f"Claude ({ANTHROPIC_MODEL})",
    "groq": f"Groq ({GROQ_MODEL})",
}.get(ASK_PROVIDER, "disabled (set ANTHROPIC_API_KEY or GROQ_API_KEY to enable)")
print(f"==> Conversational /ask: {_provider_label}.")


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


@app.route("/ask")
def ask_page():
    """Conversational assistant page (AI That Converses)."""
    return render_template("ask.html", assistant_enabled=bool(ask_client))


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
# API: conversational assistant
# ------------------------------------------------------------------

@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Answer a foundry question with Claude, grounded in the cached dataset summary.

    Accepts either {"messages": [{role, content}, ...]} for multi-turn chat, or a
    single {"question": "..."}. Returns {"answer", "latency_ms", "usage"}.
    """
    if ask_client is None:
        return jsonify({
            "error": "not_configured",
            "answer": ("The conversational assistant isn't configured on this server yet. "
                       "Set ANTHROPIC_API_KEY (Claude) or GROQ_API_KEY (Groq) in the environment "
                       "to enable it — the rest of the demo works without it."),
        }), 503

    payload = request.get_json(force=True, silent=True) or {}
    raw_messages = payload.get("messages")
    if not raw_messages:
        q = (payload.get("question") or "").strip()
        raw_messages = [{"role": "user", "content": q}] if q else []

    # Sanitize: keep only well-formed user/assistant text turns, cap count and length,
    # and ensure the history starts on a user turn (Messages API requirement).
    messages = []
    for m in raw_messages[-MAX_ASK_MESSAGES:]:
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:MAX_ASK_CHARS]})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if not messages:
        return jsonify({"error": "empty", "answer": "Please type a question."}), 400

    try:
        t0 = time.perf_counter()
        if ASK_PROVIDER == "anthropic":
            resp = ask_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1500,
                output_config={"effort": "medium"},
                # Stable summary in the cached system block; only the conversation varies.
                system=[{
                    "type": "text",
                    "text": ASK_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
            )
            answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            u = resp.usage
            usage = {
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
            }
        else:  # groq — OpenAI-compatible chat completions
            resp = ask_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=1024,
                temperature=0.3,
                messages=[{"role": "system", "content": ASK_SYSTEM_PROMPT}] + messages,
            )
            answer = (resp.choices[0].message.content or "").strip()
            u = resp.usage
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
            }
        return jsonify({
            "answer": answer or "(The assistant returned an empty response — try rephrasing.)",
            "provider": ASK_PROVIDER,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "usage": usage,
        })
    except Exception as ex:
        return jsonify({"error": "api_error", "answer": f"The assistant hit an error: {ex}"}), 502


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
