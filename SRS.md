# Software Requirements Specification
## FoundryOps Copilot — Casting Defect Prediction & Operational Co-pilot

| Field | Value |
| --- | --- |
| Product | FoundryOps Copilot |
| Built by | Zero Zeta |
| Target customer | IndieFoundry (automotive & engine castings) |
| Version | 0.4.0 |
| Document status | Draft — generated from working POC codebase |
| Document date | 2026-05-17 |

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for **FoundryOps Copilot**, a customer-facing proof-of-concept web application demonstrating AI-driven casting defect prediction, root-cause analysis, and quantified business impact for a single automotive engine-block part.

The system is delivered as a **live laptop demo** to be shown to plant heads, quality engineers, metallurgists, and CTOs at IndieFoundry with the goal of converting the demo into an 8-week paid POC engagement.

### 1.2 Scope

**In scope:**
- A three-page Flask web application (plus two supporting pages) covering narrative, interactive prediction, retrospective analytics, AI capabilities, and assumption transparency.
- A metallurgically rigorous synthetic dataset of 5,000 heats compliant with IS 210 FG260 grey-iron specification.
- Four trained ML models (XGBoost classifiers, LightGBM regressors) plus a SHAP explainer.
- HTTP / JSON API for prediction and analytics.
- Local execution; no runtime cloud or internet dependency on the prediction path.

**Out of scope (this release):**
- Real customer-data ingestion (planned for the 8-week POC).
- Multi-plant deployment, multi-part portfolio.
- Predictive maintenance, vision-based surface inspection, energy optimisation engine (Phase 2/3/4).
- Direct closed-loop process control or PLC integration.
- User authentication, multi-tenancy, audit trail.

### 1.3 Definitions, acronyms, abbreviations

| Term | Meaning |
| --- | --- |
| FG260 | Grey Cast Iron grade 260 MPa min tensile strength per IS 210 |
| IS 210 | Indian Standard for Grey Iron Castings specification |
| Cpk | Process Capability Index — `min[(USL−μ)/(3σ), (μ−LSL)/(3σ)]` |
| USL / LSL | Upper / Lower Specification Limit |
| CE | Carbon Equivalent — `C + (Si + P) / 3` |
| Mn/S rule | `Mn ≥ 1.7·S + 0.3` — sulphur neutralisation by manganese |
| OEM | Original Equipment Manufacturer (customer of the foundry) |
| PPAP | Production Part Approval Process (automotive QC standard) |
| SHAP | SHapley Additive exPlanations — game-theoretic feature attribution |
| Heat | One pour cycle producing one casting batch |
| KE-CYL-V4-220 | The anchor part: V4 engine cylinder block, 2.2 L displacement |
| POC | Proof of Concept (the 8-week engagement we aim to scope) |

### 1.4 References

Authoritative metallurgical references the synthetic data must respect:

1. **IS 210** — Indian Standard for Grey Iron Castings Specification.
2. **AFS Cast Iron Handbook** — American Foundry Society, §4 (Chemistry & Defects), §6 (Porosity).
3. **Heine, Loper, Rosenthal** — *Principles of Metal Casting* (Tata McGraw-Hill, 3rd ed.), §3 (Sand systems), §5 (Liquidus & superheat), §6 (Gas porosity mechanism).
4. **IIF** — Institute of Indian Foundrymen, Process Control Standards.

Project documents:

5. `CLAUDE.md` — non-negotiable principles and project conventions.
6. `demo.md` — original customer demo build brief.
7. `CHANGELOG.md` — version history.

### 1.5 Document overview

Sections 2–7 follow the IEEE 830 SRS structure adapted for this project: overall description, functional requirements, external interfaces, non-functional requirements, data, use cases. Section 8 lists verification criteria. Section 9 captures roadmap items. Appendices document the cost-model formulas and endpoint contracts.

---

## 2. Overall description

### 2.1 Product perspective

FoundryOps Copilot is a **standalone single-process Flask application**. It loads four pre-trained scikit-learn-compatible model files, a 5,000-row CSV dataset, and two JSON assumption files at startup. All inference and analytics are computed locally; the only runtime network calls are CDN fetches at page load (Google Fonts, Chart.js) — predictions never traverse the network beyond `localhost`.

The product is positioned as a **demonstration of methodology**, not a delivered production system. The same architecture is intended to be re-trained on real customer data during the POC.

### 2.2 Product functions (high level)

1. **Synthesise data:** generate 5,000 metallurgically valid heat records per IS 210 FG260, with all 12 encoded physics rules and a self-audit `--verify` report.
2. **Train models:** fit XGBoost defect classifier, XGBoost severity classifier, LightGBM yield regressor, LightGBM warranty risk regressor, and a SHAP TreeExplainer.
3. **Predict:** per-heat defect probability, severity probability, expected yield %, warranty risk score, and SHAP-driven root-cause attribution.
4. **Quantify business impact:** convert per-heat probabilities into ₹ scrap / rework / delay / complaint / warranty costs using calibrated unit costs.
5. **Explain the data:** structured presentation of every chemistry range, defect mechanism, cooling rule, and cost assumption — auditable by a foundry metallurgist.
6. **Visualise retrospective analytics:** Pareto by cost, monthly trend with humidity overlay, dimensional Cpk histograms, shift / furnace / season breakdowns, pattern wear curve, model performance card.
7. **Communicate AI capabilities:** descriptive narrative of four AI layers (Q&A, anomaly watch, optimisation, confidence intervals) with illustrative examples.

### 2.3 User characteristics

The system serves four user personas during the live demo session:

| Persona | Background | What they evaluate |
| --- | --- | --- |
| **Plant head / CTO** | 20+ yr ops experience, P&L responsibility | Business impact, scrap reduction story, deployment timeline |
| **Metallurgist** | Foundry science, IS 210 fluent | Whether chemistry rules and defect mechanisms are sound |
| **Quality engineer** | PPAP, Cpk, SPC | Whether dimensional capability and warranty story are credible |
| **Shift operator** (proxy view) | Hands-on at the pour | Whether the prediction UI is operable and the corrective actions are actionable |

The application **does not require** users to authenticate, train, or learn dashboards.

### 2.4 Operating environment

| Component | Requirement |
| --- | --- |
| OS | Windows 10/11, macOS 12+, or Linux (any modern distribution) |
| Python | 3.10–3.14 |
| Browser | Chrome 110+, Edge 110+, Firefox 115+, Safari 16+ (responsive down to 720 px width) |
| RAM | 1 GB free (models + dataset fit comfortably) |
| Disk | ~500 MB (deps + models) |
| Network | Internet on first run only (pip install + CDN font/chart-lib fetch); subsequent runs are fully local |

### 2.5 Design and implementation constraints

These constraints derive from `CLAUDE.md` and are **non-negotiable**:

1. **Metallurgical rigor.** Every chemistry range, defect mechanism, and physical rule traces to a cited foundry reference. No invented thresholds.
2. **Honesty about synthetic data.** A `DEMO · synthetic data` badge appears on every page header. The `/assumptions` page documents every rule.
3. **Anchored narrative.** Part = KE-CYL-V4-220, material = FG260 IS 210, customer = OEM-TATA, current state 6.8% scrap / 4.2% rework / 14 OEM complaints. Do not drift.
4. **Current → AI-Optimal demo coherence.** Two slider presets must produce dramatically different gauge readings, defect predictions, and business impact.
5. **Real ML.** Predictions originate from trained gradient-boosting models — never hand-coded rules at inference time.
6. **Inference latency ≤ 100 ms** per `/api/predict` call.
7. **No runtime internet dependencies.** CDN libraries are acceptable at page load; prediction must be local.
8. **No promised percentages on public-facing pages.** Operational-capability language only; numeric outcomes belong inside the calibrated POC deliverable.
9. **Locked tech stack.** Python only — no PyTorch, TensorFlow, cloud SDKs, or databases beyond CSV + pickle. Frontend is vanilla HTML/CSS/JS with Chart.js via CDN.

### 2.6 Assumptions and dependencies

- All metallurgical, operational, and cost assumptions are surfaced in `data/metallurgical_assumptions.json` and `data/cost_assumptions.json`. Both are human-readable and editable per plant.
- The 5,000-row dataset is regenerated deterministically with seed 42 — no run-to-run variability.
- The four model targets (`>82%`, `>78%`, `R² > 0.75`, `R² > 0.70`) are validated on a stratified 20% holdout per `train_model.py`.

---

## 3. Functional requirements

Requirement IDs use the prefix `FR-` followed by a module code: `DG` (data generation), `MT` (model training), `PR` (prediction), `LP` (landing page), `DM` (demo page), `AN` (analytics), `AC` (AI capabilities), `AS` (assumptions).

### 3.1 Data generation (FR-DG)

| ID | Requirement |
| --- | --- |
| **FR-DG-01** | The generator shall produce a deterministic 5,000-row CSV at `data/heats_2025.csv` covering the calendar year 2025, ~14 heats/day. |
| **FR-DG-02** | All chemistry values shall satisfy the IS 210 FG260 ranges defined in `metallurgical_assumptions.json` for ≥99% of rows. |
| **FR-DG-03** | The Mn/S rule (`Mn ≥ 1.7·S + 0.3`) shall be satisfied for ≥95% of rows; the remainder shall be flagged as quality-risk channel inputs. |
| **FR-DG-04** | Carbon Equivalent shall be computed as `CE = C + (Si + P)/3` and reported on every row. |
| **FR-DG-05** | Defect class shall be assigned by softmax sampling over per-row log-odds scores, with each score driven by the cited mechanism (Heine §6.4 porosity, §3.7 pattern wear, etc.). |
| **FR-DG-06** | The Bore Diameter Cpk shall land in [1.10, 1.25] (intentionally failing the OEM 1.33 requirement) so the analytics page tells the credible automotive-quality story. |
| **FR-DG-07** | Deck Height Cpk ≥ 1.40 and Wall Thickness Cpk ≥ 1.55 (both passing). |
| **FR-DG-08** | Gas-porosity rate in monsoon months shall exceed the winter baseline by ≥2×. |
| **FR-DG-09** | Dimensional-NC rate for pattern age ≥ 800 cycles shall exceed the < 500-cycle rate by ≥3×. |
| **FR-DG-10** | Furnace F1 cumulative temperature drift shall reach +12–15 °C at year end; F2 remains lower. |
| **FR-DG-11** | A `--verify` CLI flag shall print a metallurgical audit report (`data/verification_report.txt`) suitable for direct presentation to a foundry metallurgist. |
| **FR-DG-12** | Aggregate scrap rate ≈ 6.8% and rework rate ≈ 4.2% (current-state narrative). |

### 3.2 Model training (FR-MT)

| ID | Requirement |
| --- | --- |
| **FR-MT-01** | `train_model.py` shall load `data/heats_2025.csv`, encoding `None` defect / severity strings as preserved labels (not NaN). |
| **FR-MT-02** | The feature matrix shall combine numeric features (chemistry, environment, process, dimensional) and one-hot encoded categoricals (shift, furnace, mold line, pattern, season), with feature names sanitised for LightGBM. |
| **FR-MT-03** | A **defect classifier** (XGBoost, multi:softprob) shall be trained over 10 classes (None + 9 defects) with sample-weight balancing; target test accuracy > 82%. |
| **FR-MT-04** | A **severity classifier** (XGBoost, 4 classes: None / Minor_Rework / Major_Rework / Scrap) shall be trained with sample-weight balancing; target test accuracy > 78%. |
| **FR-MT-05** | A **yield regressor** (LightGBM) shall be trained on `yield_pct`; target test R² > 0.75. |
| **FR-MT-06** | A **warranty risk regressor** (LightGBM) shall be trained on `warranty_risk_score`; target test R² > 0.70. |
| **FR-MT-07** | A SHAP TreeExplainer shall be fit on the defect classifier and persisted. |
| **FR-MT-08** | All artifacts (`*.pkl`, `feature_names.json`, `metrics.json`, label encoders) shall be written to `/models`. |
| **FR-MT-09** | A confusion matrix shall be persisted for both classifiers; balanced accuracy shall be reported alongside raw accuracy. |
| **FR-MT-10** | Reproducibility: all training shall use `random_state = 42`. |

### 3.3 Prediction API (FR-PR)

| ID | Requirement |
| --- | --- |
| **FR-PR-01** | `POST /api/predict` shall accept a JSON payload of heat parameters and return defect probabilities, severity probabilities, predicted yield %, warranty risk, business impact, SHAP root-cause, and end-to-end latency. |
| **FR-PR-02** | Inference end-to-end shall complete in **≤ 100 ms** on a modern laptop (Intel/AMD/Apple Silicon, ≥ 8 GB RAM). |
| **FR-PR-03** | The response shall include a `risk_gauge_pct` field computed as `(1 − P(None)) × 100`, suitable for direct UI consumption. |
| **FR-PR-04** | Severity → disposition mapping: `None → OK`, `Minor_Rework | Major_Rework → Rework`, `Scrap → Scrap`. |
| **FR-PR-05** | Business impact shall be computed by `compute_business_impact()` per formulas in Appendix A. |
| **FR-PR-06** | Root cause shall return the top 6 SHAP features by `abs(shap_value)` for the most-likely non-None defect class, each annotated with `direction` (`increases` / `decreases`). |
| **FR-PR-07** | Missing payload fields shall fall back to dataset medians (numeric) or first one-hot category (categorical) — no crashes. |
| **FR-PR-08** | The endpoint shall reject payloads larger than 64 KB (Flask default). |

### 3.4 Landing page (FR-LP) — `GET /`

| ID | Requirement |
| --- | --- |
| **FR-LP-01** | A hero section shall display the brand mark + tagline, the rewritten subtitle (`Every pour carries risk. Your data can reduce it.`), and a balanced comparison panel showing scrap rate and OEM complaints in both current and AI-optimised states. |
| **FR-LP-02** | A demo-disclosure banner shall be present below the hero and never hidden. |
| **FR-LP-03** | A **Typical Foundry Challenges** grid of 8 cards shall be rendered; each card reveals a `How FoundryOps Copilot solves this` flap on hover/focus. |
| **FR-LP-04** | A **What the system delivers** grid of 5 capability cards (headline + subhead + body) shall replace the previous percentage-range display. No percentage ranges shall appear on this page. |
| **FR-LP-05** | A POC-calibrated business-impact strip shall sit below the cards. The expandable `See illustrative calculation` panel shall lead with a prominent `⚠ ILLUSTRATIVE — NOT A COMMITMENT` disclaimer. |
| **FR-LP-06** | An **AI/ML Stack** grid of 4 cards explaining XGBoost / LightGBM / SHAP / Causal Features shall be present. |
| **FR-LP-07** | An **Assumptions** section shall link prominently to `/assumptions` (full structured view). |
| **FR-LP-08** | A 3-step **How It Works** explainer (Ingest → Train → Deploy) and a 5-step **8-week POC roadmap** timeline shall be present. |

### 3.5 Demo page (FR-DM) — `GET /demo`

| ID | Requirement |
| --- | --- |
| **FR-DM-01** | The page shall expose **12 numeric sliders + 5 dropdowns + 1 inferred CE readout** representing operator-controllable heat parameters. |
| **FR-DM-02** | Slider groups (Operations, Environment, Mold, Process, Chemistry, Dimensional) shall be **collapsible** (`<details>` elements). Operations, Environment, and Mold open by default. |
| **FR-DM-03** | Three preset buttons shall be available: **Current**, **Now**, **AI Optimal**. Clicking a preset shall load its parameter set, mark the button active, and trigger an immediate prediction. |
| **FR-DM-04** | The Current preset shall produce a defect-risk gauge ≥ 60% (red zone). The AI Optimal preset shall produce a gauge ≤ 15% (green zone). |
| **FR-DM-05** | Slider input shall be debounced at 150 ms before triggering `/api/predict`. |
| **FR-DM-06** | The risk gauge shall be rendered as an SVG arc, transitioning colour (green / amber / red) per the gauge value. |
| **FR-DM-07** | A **Most likely defect** card shall display the top non-None defect class + model confidence + predicted severity + disposition pill. |
| **FR-DM-08** | A **Business impact** stack shall show six line items (scrap, rework, delay, complaint, warranty, total) and provide a `How is each cost calculated?` info-tip with the formulas. |
| **FR-DM-09** | A **Root cause** panel shall render the top 6 SHAP features for the predicted defect class as positive/negative bars. |
| **FR-DM-10** | A **Recommended corrective actions** card shall surface 3–4 prescriptive actions tied to the predicted defect class. |
| **FR-DM-11** | A **Warranty risk** meter (0–10) with explanatory help-tip shall be rendered. |
| **FR-DM-12** | A reference SVG schematic of KE-CYL-V4-220 (4 bores, deck height, wall thickness callouts) shall be rendered inside the Dimensional collapsible group. |
| **FR-DM-13** | An **Annual extrapolation strip** shall compare *This setpoint extrapolated × 14,000* vs *AI-optimal extrapolated × 14,000* with a clear disclaimer that the numbers are worst-case extrapolations, not actual annual loss. |
| **FR-DM-14** | A latency pill shall display the round-trip prediction time in milliseconds. |

### 3.6 Analytics page (FR-AN) — `GET /analytics`

| ID | Requirement |
| --- | --- |
| **FR-AN-01** | A KPI strip shall show total heats, scrap rate %, rework rate %, and annual savings opportunity. |
| **FR-AN-02** | A **Business impact waterfall** chart shall break the total YTD cost into scrap, rework, OEM complaints, warranty exposure, and the rolled-up total. |
| **FR-AN-03** | A **Defect Pareto** chart shall plot defect classes by ₹ cost (bars) with count overlay (line). |
| **FR-AN-04** | A **Monthly trend** chart shall plot porosity rate and scrap rate as bars and humidity % as an overlay line. |
| **FR-AN-05** | **Dimensional capability** histograms shall be rendered for Bore Diameter, Deck Height, and Wall Thickness, each annotated with USL/LSL/target and the computed Cpk. |
| **FR-AN-06** | **Breakdown panels** shall split defect rate by shift, furnace, season, and pattern. |
| **FR-AN-07** | A **Pattern wear curve** shall plot dim_NC rate and any-defect rate across age buckets (0–200 to 1200–1500 cycles). |
| **FR-AN-08** | A **Feature correlations** chart shall list the top 15 Pearson correlations of numeric features against a binary defect indicator. |
| **FR-AN-09** | A **Model performance** card shall report all four model accuracy / R² metrics, target thresholds, and training set size from `models/metrics.json`. |
| **FR-AN-10** | A **Sample data preview** shall display the first 25 rows of the dataset with selected columns. |

### 3.7 AI Capabilities page (FR-AC) — `GET /capabilities`

| ID | Requirement |
| --- | --- |
| **FR-AC-01** | A 2×2 quadrant hero shall display the four AI layers (Converses / Watches / Acts / Knows-its-limits). Each tile links smooth-scroll to its detailed section. |
| **FR-AC-02** | Each layer detail section shall contain three columns: *What it does* (narrative) / *How it works* (technical stack) / *Illustrative example* (visual mock). |
| **FR-AC-03** | A *Why four layers, not one* statement panel shall articulate the architectural principle. |
| **FR-AC-04** | A **Phase 2 / 3 / 4 roadmap** shall describe future capability extensions beyond the four core layers. |
| **FR-AC-05** | Example panels shall be **descriptive mocks** — they shall not depend on a live Anthropic API key, IsolationForest endpoint, or Bayesian optimiser. |

### 3.8 Assumptions page (FR-AS) — `GET /assumptions`

| ID | Requirement |
| --- | --- |
| **FR-AS-01** | The page shall render a structured HTML view of `metallurgical_assumptions.json` and `cost_assumptions.json` — never raw JSON. |
| **FR-AS-02** | Required sections: Part & material · References followed · Chemistry ranges · Pour temperature & cooling rate bands · Porosity mechanism · Pattern wear & furnace drift · Dimensional tolerances · Operators & shifts · Seasonal patterns · Defect drivers · Severity distribution · Cost assumptions · Dataset metadata · Verification report. |
| **FR-AS-03** | Raw JSON (`/api/assumptions`) shall remain available, linked from the page header for developers. |

---

## 4. External interface requirements

### 4.1 User interfaces

- **Responsive** layout, supported viewport ≥ 720 px width. Below that, multi-column grids collapse to single-column stacks.
- **Theme:** Zero Zeta-inspired light mode (CSS variables in `static/css/theme.css`). Primary `#0E4DA6`, accent teal `#00B5A0`, risk-red `#DC2626`, risk-amber `#F59E0B`, risk-green `#16A34A`.
- **Typography:** Inter (UI), JetBrains Mono (numeric / code), loaded from Google Fonts CDN.
- **Branding:** Zero Zeta logo image (`static/img/zerozeta-logo.png`) appears in every nav and footer; product wordmark `· FoundryOps Copilot` follows it.
- **Mandatory persistent UI elements:**
  - `DEMO · synthetic data` badge in every nav.
  - Navigation links: Home / Live Demo / Analytics / AI Capabilities.

### 4.2 HTTP / API interfaces

All endpoints are served by `app.py` on `http://0.0.0.0:5000`. CORS is enabled for development convenience.

**Page routes (return rendered HTML):**

| Route | Template |
| --- | --- |
| `GET /` | `landing.html` |
| `GET /demo` | `demo.html` |
| `GET /analytics` | `analytics.html` |
| `GET /capabilities` | `capabilities.html` |
| `GET /assumptions` | `assumptions.html` |

**API routes (return JSON):**

| Route | Description |
| --- | --- |
| `POST /api/predict` | Predict outcomes for a single heat (see Appendix B for full payload schema). |
| `GET /api/assumptions` | Raw JSON of metallurgical + cost assumptions + verification report text. |
| `GET /api/dictionary` | Parsed data dictionary (column / type / range / metallurgical meaning). |
| `GET /api/model_info` | Model metrics, feature names, defect/severity class lists. |
| `GET /api/presets` | Three slider presets (`current`, `now`, `ai_optimal`) and their parameter payloads. |
| `GET /api/analytics/overview` | Total heats, rates, ₹ costs, annual savings target. |
| `GET /api/analytics/pareto` | Defect classes ranked by ₹ cost. |
| `GET /api/analytics/monthly` | Per-month porosity rate, scrap rate, complaint rate, humidity. |
| `GET /api/analytics/dimensional` | Bore / Deck / Wall histograms + Cpk per dimension. |
| `GET /api/analytics/breakdowns` | Defect-rate splits by shift / furnace / mold line / season / pattern. |
| `GET /api/analytics/pattern_wear` | Dim_NC and any-defect rates by pattern-age bucket. |
| `GET /api/analytics/correlations` | Top 15 Pearson correlations with binary defect indicator. |
| `GET /api/analytics/sample?n=N` | First N rows of the dataset as records. |

### 4.3 Software interfaces

| Library | Min version | Purpose |
| --- | --- | --- |
| Flask | 3.0 | HTTP server, template rendering, JSON serialisation |
| Flask-CORS | 4.0 | CORS for dev workflows |
| pandas | 2.1 | Data loading, aggregation, sample serialisation |
| numpy | 1.26 | Numerical kernels, RNG, histograms |
| scikit-learn | 1.3 | Train/test split, label encoders, metrics |
| xgboost | 2.0 | Defect + severity classifiers |
| lightgbm | 4.1 | Yield + warranty regressors |
| shap | 0.44 | TreeExplainer for root-cause attribution |
| joblib | 1.3 | Model serialisation |
| Chart.js | 4.4 (CDN) | All analytics charts |

### 4.4 Hardware interfaces

None. Pure software; no PLC, sensor, or CNC integration is part of this release. (Phase 4 roadmap covers closed-loop process control.)

---

## 5. Non-functional requirements

| ID | Requirement | Verification |
| --- | --- | --- |
| **NFR-PERF-01** | Cold start ≤ 60 s (pip install + data gen + model train + Flask boot). | Measured during `./run.sh`. |
| **NFR-PERF-02** | Warm start ≤ 6 s when artifacts exist. | Measured. |
| **NFR-PERF-03** | `/api/predict` end-to-end latency ≤ 100 ms. | Latency pill in demo UI; server logs. |
| **NFR-PERF-04** | Landing / demo / analytics page first paint ≤ 2 s on a 100 Mbps connection. | Manual stopwatch + DevTools. |
| **NFR-RELY-01** | Predictions are deterministic for a given payload (no run-to-run drift). | Inputs hashed and compared. |
| **NFR-RELY-02** | Graceful degradation: if `data/heats_2025.csv` or any model file is missing, `app.py` shall raise a clear error pointing at `generate_data.py` / `train_model.py` rather than crashing silently. | Tested. |
| **NFR-SEC-01** | No customer data is stored, transmitted, or required at runtime. | Source-code review. |
| **NFR-SEC-02** | No secrets in source control. `.gitignore` excludes `.env` and credential files. | Git review. |
| **NFR-USE-01** | All five pages function on Chrome / Edge / Firefox / Safari current-major releases. | Cross-browser smoke test. |
| **NFR-USE-02** | The `DEMO · synthetic data` badge is visible on every page header at all viewports. | Visual inspection. |
| **NFR-USE-03** | All cost calculation formulas are discoverable in-product (info-tips, `/assumptions`). | UI test. |
| **NFR-MAINT-01** | All metallurgical + cost assumptions are externalised to JSON; no magic numbers in inference code. | Source review. |
| **NFR-MAINT-02** | Template auto-reload (`TEMPLATES_AUTO_RELOAD = True`) so HTML edits do not require a Flask restart. | App config. |
| **NFR-PORT-01** | Runs on Python 3.10, 3.11, 3.12, 3.13, 3.14. | Tested on 3.14. |
| **NFR-COMPL-01** | Synthetic data complies with IS 210 FG260 chemistry ranges. | `--verify` report. |
| **NFR-COMPL-02** | OEM Cpk threshold (≥ 1.33 per automotive PPAP) is documented and tracked. | `/assumptions` + analytics page. |

---

## 6. Data requirements

### 6.1 Synthetic dataset schema

`data/heats_2025.csv` — 5,000 rows, 45 columns. Full column dictionary is in `data/data_dictionary.csv` and exposed via `GET /api/dictionary`. Highlights:

| Domain | Columns | Notes |
| --- | --- | --- |
| Identity | `heat_id`, `timestamp` | unique per row |
| Operations | `shift`, `operator_id`, `furnace_id`, `mold_line_id`, `pattern_id`, `pattern_age_cycles` | categorical or integer |
| Environment | `season`, `ambient_temp_C`, `ambient_humidity_pct`, `sand_moisture_pct`, `core_moisture_pct`, `mold_temp_C` | seasonally correlated |
| Mold | `mold_hardness_B`, `binder_pct`, `mold_permeability` | continuous |
| Chemistry | `C_pct`, `Si_pct`, `Mn_pct`, `S_pct`, `P_pct`, `Cr_pct`, `carbon_equivalent`, `mn_s_ratio_ok` | IS 210 FG260 ranges, joint distribution constraints |
| Process | `pour_temp_C`, `superheat_C`, `pour_rate_kg_per_s`, `pour_delay_min`, `cooling_rate_C_per_min`, `fade_time_min`, `inoculant_dose_pct`, `furnace_temp_drift_C` | continuous |
| Dimensional | `bore_diameter_mm`, `deck_height_mm`, `wall_thickness_mm`, `surface_roughness_Ra_um` | OEM-critical |
| Outcomes | `defect_class`, `severity`, `disposition`, `customer_complaint`, `yield_pct`, `casting_weight_kg`, `warranty_risk_score`, `oem_customer` | model targets |

### 6.2 Model artifacts (`/models`)

| File | Type | Notes |
| --- | --- | --- |
| `defect_classifier.pkl` | XGBoost multi-class | 10 classes (None + 9 defects) |
| `severity_classifier.pkl` | XGBoost multi-class | 4 classes (None, Minor_Rework, Major_Rework, Scrap) |
| `yield_regressor.pkl` | LightGBM | target `yield_pct` |
| `warranty_regressor.pkl` | LightGBM | target `warranty_risk_score` |
| `shap_explainer.pkl` | shap.TreeExplainer | fit on `defect_classifier` |
| `defect_label_encoder.pkl` | sklearn LabelEncoder | |
| `severity_label_encoder.pkl` | sklearn LabelEncoder | |
| `feature_names.json` | list[str] | column order for inference |
| `metrics.json` | dict | accuracy / R² / confusion matrices / target thresholds |

### 6.3 Configuration files (`/data`)

| File | Contents |
| --- | --- |
| `metallurgical_assumptions.json` | Part / material / references / chemistry ranges / Cpk targets / defect drivers / severity distribution / operators / seasons / dataset metadata |
| `cost_assumptions.json` | All ₹ unit costs, current vs target state KPIs, annual volume |
| `data_dictionary.csv` | Column-by-column metallurgical meaning |
| `verification_report.txt` | Output of `generate_data.py --verify` |

---

## 7. Use cases

### UC-1 — Metallurgist audits data rules

**Actor:** Foundry metallurgist (IndieFoundry)
**Trigger:** Reaches the landing page during the demo session.
**Flow:**
1. Reads the **Typical Foundry Challenges** grid and recognises the pains.
2. Opens the **Assumptions** section → clicks `view full assumptions →`.
3. Walks through chemistry ranges, defect mechanisms, cooling rules, Cpk targets on `/assumptions`.
4. Reads the metallurgical verification report at the bottom.
5. Asks "can we audit `generate_data.py` ourselves?"
**Success criterion:** The metallurgist accepts the rules as principled and asks audit questions rather than rejection questions.

### UC-2 — Plant head reviews business impact

**Actor:** Plant head / CTO
**Trigger:** Wants the ₹ story.
**Flow:**
1. Reads the **What the system delivers** capability cards.
2. Expands `See illustrative calculation` and reads the side-by-side current vs target math under the ILLUSTRATIVE disclaimer.
3. Asks "what would this look like on our data?"
**Success criterion:** The plant head asks about timeline and POC scoping, not "where did the 6.8% come from."

### UC-3 — Quality engineer / operator exercises the demo

**Actor:** Quality engineer
**Trigger:** Clicks `Live Demo`.
**Flow:**
1. Page loads with the **Current** preset pre-selected; gauge fills red at ≥ 60%.
2. Operator reads the most-likely defect (often Gas_Porosity or Cold_Shut), business impact stack, root-cause SHAP bars.
3. Operator clicks **AI Optimal**; gauge sweeps to green ≤ 15%; impact drops to ~₹1.8K per casting.
4. Operator hovers `How is each cost calculated?` to see the formulas.
5. Operator inspects the reference part drawing inside the Dimensional collapsible group.
**Success criterion:** The flow takes ≤ 2 minutes and the swing is dramatic and credible.

### UC-4 — QC investigates retrospective analytics

**Actor:** Quality engineer
**Trigger:** Wants to verify the system understands their plant patterns.
**Flow:**
1. Clicks `Analytics`.
2. Confirms the Pareto-by-cost (not by count) and reads the top three defect classes.
3. Confirms Bore Cpk 1.21 (failing) vs Deck 1.45 (passing) vs Wall 1.62 (passing).
4. Inspects monthly trend with humidity overlay → recognises the monsoon spike.
5. Reads the model-performance card to confirm accuracy / R² claims.
**Success criterion:** QC repeats the Cpk and monsoon story back to the team unprompted.

### UC-5 — Customer requests POC scoping

**Actor:** Plant head + procurement
**Trigger:** End of demo.
**Flow:**
1. Reads the 8-week POC roadmap timeline on the landing page.
2. Asks for the proposal.
**Success criterion:** Zero Zeta is invited to scope the POC.

---

## 8. Verification & acceptance criteria

The release is accepted when **all** of the following pass:

1. `./run.sh` (Bash) or `python app.py` (PowerShell, after data + model artifacts exist) launches successfully and Chrome opens to `http://localhost:5000/` in under 60 s cold / 6 s warm.
2. `python generate_data.py --verify` exits 0 and the report shows:
   - All chemistry ranges within IS 210 FG260 for ≥99% of heats.
   - Mn/S rule satisfied for ≥95% of heats.
   - Bore Cpk in [1.10, 1.25], Deck Cpk ≥ 1.40, Wall Cpk ≥ 1.55.
   - Monsoon porosity multiplier ≥ 2×.
   - Pattern-wear multiplier ≥ 3× at age ≥ 800 cycles.
   - Furnace F1 year-end drift ≥ +12 °C.
3. `python train_model.py` reports defect accuracy > 82%, severity > 78%, yield R² approximately ≥ 0.68, warranty R² ≥ 0.70 (yield is allowed within ±0.1 of the 0.75 target given the small minority-class regime; see CHANGELOG 0.1.0 note).
4. All five page routes return HTTP 200; all 12 API routes return 200 with JSON.
5. The Current preset produces gauge ≥ 60% (red); the AI Optimal preset produces gauge ≤ 15% (green).
6. `/api/predict` round-trip latency ≤ 100 ms measured at the latency pill.
7. The `DEMO · synthetic data` badge is visible on every page header.
8. No percentage-range outcome statements appear on any public-facing page.
9. The ILLUSTRATIVE-NOT-A-COMMITMENT disclaimer is visible at the top of the expanded calculation panel.

---

## 9. Future enhancements (roadmap reference)

These are explicitly **out of scope for this release** but enumerated to bound the conversation. Tracked on `/capabilities` page.

| Phase | Item | Notes |
| --- | --- | --- |
| Phase 2 | 7-day defect forecast | Combines weather + production schedule + pattern wear |
| Phase 2 | Vision-based surface inspection | Camera on shakeout line, ~200 ms/casting |
| Phase 2 | Multi-part family expansion | Crankcase, head, housing — same architecture |
| Phase 3 | Predictive maintenance | LSTM on furnace + mold-line vibration / current |
| Phase 3 | Energy optimisation engine | Charge mix + holding time optimiser |
| Phase 3 | OEM PPAP automation | Automated PPAP documentation generation |
| Phase 4 | Multi-plant deployment | |
| Phase 4 | Supply chain AI | Charge-mix optimisation across vendors |
| Phase 4 | Closed-loop process control | Direct setpoint actuation, not just recommendation |

---

## Appendix A — Business cost formulas

Implemented in `app.py :: compute_business_impact()`. All cost constants live in `data/cost_assumptions.json`.

Given severity probabilities `p_minor`, `p_major`, `p_scrap`, `p_none`; customer-complaint probability `c`; warranty risk score `w` (0–10); and unit costs from configuration:

```
rework_cost     = (p_minor + 0.6 · p_major) × ₹1,800
scrap_cost      = (p_scrap + 0.4 · p_major) × ₹6,500
delay_minutes   = 8 · p_minor + 18 · p_major + 35 · p_scrap
delay_cost      = delay_minutes × ₹450 / min
complaint_cost  = c × ₹85,000
warranty_cost   = (w / 10) × 0.04 × ₹4,50,000
total_cost      = scrap_cost + rework_cost + delay_cost + complaint_cost + warranty_cost
annual_extrap   = total_cost × 14,000      # worst-case extrapolation, not actual annual loss
```

Where the **customer-complaint probability** itself is:

```
c = min(1, P(any defect) × 0.05 + (w / 10) × 0.05)
```

---

## Appendix B — `/api/predict` payload schema

**Request body** (all fields optional; missing numeric fields fall back to dataset medians, missing categoricals to defaults):

```jsonc
{
  // Operations
  "shift": "A" | "B" | "C",
  "furnace_id": "F1" | "F2",
  "mold_line_id": "ML-1" | "ML-2",
  "pattern_id": "PT-A" | "PT-B" | "PT-C",
  "pattern_age_cycles": 0..1500,
  "season": "Winter" | "Pre_monsoon" | "Monsoon" | "Post_monsoon",
  "furnace_temp_drift_C": 0..20,

  // Environment
  "ambient_temp_C": 15..40,
  "ambient_humidity_pct": 35..95,
  "sand_moisture_pct": 2.5..6.0,
  "core_moisture_pct": 2.0..5.0,

  // Mold
  "mold_hardness_B": 72..92,
  "mold_temp_C": 22..55,
  "binder_pct": 1.4..2.4,
  "mold_permeability": 80..160,

  // Process
  "pour_temp_C": 1340..1480,
  "pour_rate_kg_per_s": 4.5..10.5,
  "pour_delay_min": 0..15,
  "cooling_rate_C_per_min": 4..22,
  "fade_time_min": 0..30,
  "inoculant_dose_pct": 0.10..0.30,

  // Chemistry (FG260 IS 210)
  "C_pct": 3.10..3.60,
  "Si_pct": 1.80..2.40,
  "Mn_pct": 0.50..0.90,
  "S_pct": 0.06..0.12,
  "P_pct": 0.05..0.12,
  "Cr_pct": 0.00..0.20,

  // Dimensional (measured)
  "bore_diameter_mm": 89.40..89.60,
  "deck_height_mm": 219.85..220.15,
  "wall_thickness_mm": 4.20..4.80,
  "surface_roughness_Ra_um": 2.5..9.5,
  "casting_weight_kg": 205..225,

  // Derived (server-recomputes if missing or stale)
  "carbon_equivalent": null,
  "superheat_C": null,
  "mn_s_ratio_ok": true
}
```

**Response body:**

```jsonc
{
  "defect_probabilities": { "None": 0.137, "Gas_Porosity": 0.412, "Cold_Shut": 0.672, ... },
  "top_defect_class": "Cold_Shut",
  "top_defect_probability": 0.672,
  "p_any_defect": 0.863,
  "risk_gauge_pct": 86.3,
  "severity_probabilities": { "None": 0.13, "Minor_Rework": 0.10, "Major_Rework": 0.52, "Scrap": 0.25 },
  "predicted_severity": "Major_Rework",
  "disposition": "Rework",
  "predicted_yield_pct": 89.0,
  "predicted_warranty_risk": 3.72,
  "customer_complaint_p": 0.062,
  "business_impact": {
    "scrap_cost": 5460, "rework_cost": 1620, "delay_cost": 4140,
    "complaint_cost": 5252, "warranty_cost": 6696,
    "total_cost": 23168, "annualized": 324352000
  },
  "root_causes": [
    { "feature": "pour_temp_C", "shap_value": -1.84, "direction": "decreases" },
    ...
  ],
  "latency_ms": 71
}
```

---

## Appendix C — Project file inventory

```
foundary/
├── CLAUDE.md                         # non-negotiable principles
├── CHANGELOG.md                      # version history
├── SRS.md                            # this document
├── README.md                         # operator-facing run instructions + demo script
├── demo.md                           # original build brief
├── requirements.txt                  # Python deps (relaxed min versions)
├── run.sh                            # bash one-command launch (mac/linux)
├── app.py                            # Flask backend, all routes
├── generate_data.py                  # synthetic data generator w/ --verify
├── train_model.py                    # 4-model trainer
├── data/
│   ├── heats_2025.csv                # 5,000-row synthetic dataset
│   ├── data_dictionary.csv           # column metallurgical meaning
│   ├── metallurgical_assumptions.json
│   ├── cost_assumptions.json
│   └── verification_report.txt
├── models/
│   ├── defect_classifier.pkl
│   ├── severity_classifier.pkl
│   ├── yield_regressor.pkl
│   ├── warranty_regressor.pkl
│   ├── shap_explainer.pkl
│   ├── defect_label_encoder.pkl
│   ├── severity_label_encoder.pkl
│   ├── feature_names.json
│   └── metrics.json
├── templates/
│   ├── landing.html
│   ├── demo.html
│   ├── analytics.html
│   ├── capabilities.html
│   └── assumptions.html
└── static/
    ├── css/
    │   ├── theme.css                 # Zero Zeta-derived design system variables
    │   └── style.css                 # page-specific styles
    ├── js/
    │   ├── demo.js                   # demo page interactions
    │   └── analytics.js              # analytics charts
    └── img/
        └── zerozeta-logo.png
```

---

## Appendix D — Glossary of defect classes

| Class | Mechanism (per `metallurgical_assumptions.json`) | Typical severity |
| --- | --- | --- |
| **None** | No defect detected | OK |
| **Blow_Holes** | Water vapour and binder gas trapped during solidification | Minor rework |
| **Gas_Porosity** | H₂ dissociation from moisture, dissolved in melt, rejected at solidification | Major rework |
| **Shrinkage** | Insufficient feeding during liquid-to-solid contraction | Scrap |
| **Cold_Shut** | Insufficient superheat — metal streams solidify before fusion | Scrap |
| **Misrun** | Metal freezes before filling the cavity | Scrap |
| **Sand_Inclusion** | Mold wall erosion under metal flow stress | Major rework |
| **Dimensional_NC** | Pattern dimensional drift + thermal contraction variability | Minor rework |
| **Surface_Defects** | Thermal degradation of mold surface, vein defects | Minor rework |
| **Cracks** | Hot tear (during solidification) or cold crack (post-solidification residual stress) | Scrap |

---

*End of Software Requirements Specification — version 0.4.0 — 2026-05-17*
