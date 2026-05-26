# Foundry AI Demo — Kirloskar POC

**Built by Zero Zeta · For Kirloskar (automotive & engine castings) · 2025 POC**

A three-page Flask web application demonstrating AI-driven casting defect
prediction, root-cause analysis, and quantified business impact for a single
high-volume part. Built on **metallurgically rigorous synthetic data**
generated to **IS 210 / FG260** grey iron standards, with **real ML models**
(XGBoost, LightGBM, SHAP) trained in `train_model.py`.

---

## 30-second pitch

> A working AI system that predicts defects in your engine block castings
> **before** the pour — and tells you which knob to turn to avoid the defect.
> Slide a parameter, watch the gauge swing, see ₹35K of expected loss melt
> to ₹2K. On synthetic-but-realistic data today; on your data in 8 weeks.

---

## What this demo proves

1. **You can predict defect class, severity, yield, and warranty risk** from
   pre-pour heat parameters at production-grade accuracy (>82% defect, >78%
   severity, R² > 0.7 for yield and warranty).
2. **Predictions are explainable** — SHAP attribution shows the operator
   exactly which parameters pushed the heat toward a defect.
3. **The business impact is quantified at the casting level** — scrap +
   rework + delay + complaint + warranty in ₹, rolled up to annual savings.
4. **The synthetic data foundation is auditable** by a metallurgist —
   every chemistry range, defect mechanism, cooling rule traces to a real
   foundry reference (IS 210, AFS Cast Iron Handbook, IIF, Heine/Loper/Rosenthal).

---

## Metallurgical rigor statement

The synthetic dataset (`data/heats_2025.csv`, 5,000 heats) follows:

- **IS 210** Grey Iron Castings Specification — all chemistry within FG260 ranges
- **AFS Cast Iron Handbook** — defect mechanisms (gas porosity, cold shut, shrinkage)
- **IIF Process Control Standards** — operator/shift/seasonal patterns
- **Heine, Loper, Rosenthal · *Principles of Metal Casting*** — pour temperature, cooling rate, pattern wear curves

Key encoded rules:

- Carbon Equivalent `CE = C + (Si+P)/3`, eutectic band 4.0–4.3
- Manganese/Sulphur balance: `Mn ≥ 1.7·S + 0.3`
- Pouring superheat: liquidus + 220–250°C → 1400–1430°C optimal
- Porosity scales non-linearly: `(moisture)^1.8 × (humidity)^1.5 × (delay)^0.8`
- Pattern wear: dim_NC rate triples at 800–1200 cycles
- Furnace lining drift: +12–15°C cumulative over a campaign
- OEM Cpk requirement: ≥1.33 per automotive PPAP

Run `python generate_data.py --verify` to print the full audit report. It
is also saved to `data/verification_report.txt` and displayed on the
landing page.

---

## Three pages, one story

| URL | What it does |
| --- | --- |
| `/`            | Landing — narrative, business case, full transparency on assumptions |
| `/demo`        | Interactive sliders + prediction + SHAP root cause + business impact |
| `/analytics`   | 5,000-heat retrospective: Pareto, monthly trend, Cpk, pattern wear, model performance |

A **DEMO · synthetic data** badge is present on every page header — this is
not hidden.

---

## How to run

```bash
./run.sh
```

This one-command launch will:

1. `pip install -r requirements.txt`
2. Run `python generate_data.py --verify` if `data/heats_2025.csv` is missing
3. Run `python train_model.py` if models are missing
4. Start Flask on http://localhost:5000 and open your browser

Total cold-start time: ~60 seconds.

### Manual steps

```bash
# Regenerate data with full audit report
python generate_data.py --verify

# Retrain all four models
python train_model.py

# Start the server only (data + models must exist)
python app.py
```

### Enabling the "Ask AI" page (`/ask`)

The conversational assistant is **optional** — every other page works without it.
It uses whichever key is present: **Claude** if `ANTHROPIC_API_KEY` is set, otherwise
**Groq** (OpenAI-compatible, fast and low-cost) if `GROQ_API_KEY` is set.

```bash
# Option A — Claude
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."

# Option B — Groq (reuse a gsk_... key)
export GROQ_API_KEY=gsk_...                # PowerShell: $env:GROQ_API_KEY = "gsk_..."
# optional model override (default: llama-3.3-70b-versatile)
export GROQ_MODEL=openai/gpt-oss-120b

python app.py
```

**Prefer a file?** Copy `.env.example` to `.env` in the project root and put the key
there (`GROQ_API_KEY=gsk_...`). `.env` is git-ignored, and `app.py` loads it on
startup via `python-dotenv`. Keep the key in `.env` or your shell — **never commit it**.

The startup log prints which backend is active. If neither key is set, `/ask` loads
and shows a "not configured" notice; the rest of the demo is unaffected. Answers are
grounded in a summary of `data/heats_2025.csv` (Claude path additionally caches that
summary in the system prompt).

---

## Data files inventory

| File | Contents |
| --- | --- |
| `data/heats_2025.csv`              | 5,000 heats, 45 columns — full feature set + labels |
| `data/data_dictionary.csv`         | Column-by-column metallurgical meaning |
| `data/metallurgical_assumptions.json` | All physics rules, chemistry ranges, defect drivers in machine-readable form |
| `data/cost_assumptions.json`       | Scrap/rework/delay/complaint/warranty costs in INR |
| `data/verification_report.txt`     | Output of `generate_data.py --verify` |
| `models/defect_classifier.pkl`     | XGBoost — 10-class (None + 9 defect types) |
| `models/severity_classifier.pkl`   | XGBoost — 4-class (None / Minor / Major / Scrap) |
| `models/yield_regressor.pkl`       | LightGBM — yield % regression |
| `models/warranty_regressor.pkl`    | LightGBM — warranty risk score 0–10 |
| `models/shap_explainer.pkl`        | TreeExplainer for the defect classifier |
| `models/feature_names.json`        | Feature schema for inference |
| `models/metrics.json`              | Accuracy / R² / confusion matrices |

---

## Three-minute demo script

### 1 · Landing page (90 sec)

> "Here are the eight foundry challenges you live with daily. Recognize them?"
>
> "Here are the five business gains AI delivers, with industry-typical ranges."
>
> "Here's the ML stack — XGBoost for defects, LightGBM for yield, SHAP for
> root cause. No magic; everything is auditable."
>
> **Open the assumptions panel.** "Every chemistry range follows IS 210 FG260.
> Every defect mechanism traces to a foundry textbook. The metallurgical
> verification report is right here — feel free to audit our rules. This is
> the same audit your team can run on `generate_data.py --verify`."

### 2 · Demo page — `Current` preset (60 sec)

Click **Current**. The gauge sweeps to red.

> "This is today's plant state. Pattern PT-C at 1,100 cycles. Monsoon
> humidity. Furnace F1 with end-of-campaign drift. Shift B (junior operator)."
>
> "Risk gauge: 86%. Most likely defect: dimensional non-conformance.
> Predicted disposition: rework or scrap."
>
> "Root cause panel shows pattern age + cooling rate + humidity as the
> dominant drivers."
>
> "Business impact stack: ~₹35,000 expected per casting. Annualized to
> ₹1.4 Cr across this single part."

### 3 · Demo page — `AI Optimal` preset (30 sec)

Click **AI Optimal**. The gauge sweeps to green.

> "Same plant, same crew. AI-recommended setpoints: fresh pattern,
> dry-batch sand, tightened pour temp, F2 instead of F1, Shift A."
>
> "Gauge drops to under 2%. Predicted disposition: OK."
>
> "Per-casting expected loss: ₹2,000. Annual savings: ~₹90 Lakhs recovered.
> Across a plant's portfolio of 50+ parts, 8–15× that."

### 4 · Analytics page (30 sec)

> "5,000 heats — real Pareto by cost, not just count. Real Cpk numbers:
> Bore 1.21 (failing OEM-TATA's 1.33), Deck 1.45 (passing), Wall 1.62 (passing).
> Pattern wear curve crossing 800 cycles. Monsoon humidity spike overlay.
> Operator/shift breakdown showing Shift B effect."
>
> "On your data, the patterns differ, but the insights land the same way."

---

## Honest note: synthetic data

This demo runs on metallurgically rigorous synthetic data — not on
Kirloskar production data. The data generator (`generate_data.py`) is
auditable, every rule cited to a foundry reference. **Replace it with
your data and the same code path applies** — re-run `train_model.py`
and the demo trains in 60 seconds on your numbers.

---

## 8-week POC roadmap

| Week | Activity |
| --- | --- |
| 1–2 | Data extraction from your ERP/MES/spectrometer logs · field mapping to our schema |
| 3–4 | Model training on your data · validation against existing QC dispositions |
| 5   | Validation: holdout test, drift checks, operator workshop on SHAP outputs |
| 6–7 | Pilot — operators see live predictions on incoming pours; A/B against current process |
| 8   | Full deployment, integration with QC/dispatch, dashboards, handover documentation |

End state: operators get pre-pour predictions in real time; QC sees
Pareto and Cpk dashboards updated daily; plant head sees ₹ saved
month-over-month.

---

## Tech stack

- **Backend:** Python 3.10+, Flask 3, Flask-CORS
- **ML:** scikit-learn, XGBoost, LightGBM, SHAP
- **Data:** pandas, numpy, joblib
- **Frontend:** Vanilla HTML/CSS/JS, Chart.js (via CDN at page load only)
- **Theme:** Custom CSS variables inspired by zerozeta.com — clean, minimal, light mode
- **No runtime internet dependencies.** CDN libraries load once at page load; all predictions run locally.

---

## Success criteria

The demo succeeds if, after seeing it:

1. A Kirloskar metallurgist asks **"can we audit your data generation rules?"** (curiosity, not suspicion)
2. A Kirloskar plant head asks **"how long to run this on our data?"** (intent, not skepticism)
3. The 6.8% → 3.5% scrap story is **repeated back** by someone in the room (narrative stickiness)
4. We are **invited to scope an 8-week paid POC** (commercial outcome)

---

## License & contact

Built by Zero Zeta as a customer proof-of-concept. Not for redistribution.
For questions or to scope your POC: contact your Zero Zeta account team.
