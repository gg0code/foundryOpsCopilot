\# Build: Foundry AI Demo — Casting Quality \& Yield Optimizer (IndieFoundry POC)



\## Context



I'm building a customer-facing demo for \*\*IndieFoundry\*\*, a manufacturer of \*\*automotive and engine-related castings\*\*. We don't have customer data yet, so we'll generate \*\*metallurgically rigorous synthetic data\*\*, train real ML models on it, and serve predictions through a polished web UI.



\*\*Delivery: Live laptop demo.\*\* Single command launch in <30 seconds.



\## CRITICAL: Metallurgical Rigor Requirement



The synthetic data \*\*must follow real foundry physics and chemistry rules\*\* — not invented thresholds. A foundry metallurgist must be able to audit `generate\_data.py` and verify every causal rule against standard foundry references.



\*\*Material grade constraint:\*\* All chemistry, dimensional, and defect data must be consistent with \*\*Grey Cast Iron, Grade FG260 per IS 210\*\* (Indian standard for automotive engine block castings). This grade choice constrains every chemistry range.



\*\*Authoritative references the data generator must respect:\*\*

\- IS 210 — Grey Iron Castings specification

\- AFS (American Foundry Society) — Cast Iron Handbook

\- Heine, Loper, Rosenthal — \*Principles of Metal Casting\* (industry textbook)

\- IIF (Institute of Indian Foundrymen) — Process Control Standards



\## Physics \& Chemistry Rules to Encode



\### 1. Chemistry Constraints (Grey Iron FG260)



All values must respect IS 210 ranges for FG260 grey iron:



| Element | Range | Rationale |

|---|---|---|

| Carbon (C) | 3.10–3.60% | Below 3.0% = white iron risk; above 3.7% = excessive graphite, weak |

| Silicon (Si) | 1.80–2.40% | Graphitizer; balances C; too low = chill, too high = weak |

| Manganese (Mn) | 0.50–0.90% | Sulphide control; must satisfy Mn ≥ 1.7×S + 0.3% |

| Sulphur (S) | 0.06–0.12% | Higher than ductile iron; needs Mn balance |

| Phosphorus (P) | 0.05–0.12% | Improves fluidity but reduces toughness; over 0.15% = brittleness |

| Chromium (Cr) | ≤ 0.20% | Carbide former; high Cr → hard spots, cracks |



\### 2. Carbon Equivalent (CE) — Hard Rule



`CE = C + (Si + P) / 3`



\- \*\*CE = 4.0–4.3\*\*: Eutectic, ideal fluidity, lowest defect rate

\- \*\*CE < 4.0\*\* (hypoeutectic): Higher shrinkage risk, lower fluidity → more cold shut/misrun

\- \*\*CE > 4.3\*\* (hypereutectic): Kish graphite formation, surface defects, weak structure



Encode these CE-driven defect probabilities explicitly. Heats outside CE 4.0–4.3 should have measurably higher defect rates in the corresponding defect class.



\### 3. Mn/S Ratio Rule — Hard Rule



Must satisfy: \*\*Mn ≥ 1.7 × S + 0.3\*\* to neutralize sulphur as MnS.



Violations of this rule → higher hard spots, hot tear, and crack probability. Encode this as an explicit causal driver.



\### 4. Pouring Temperature Physics



Required superheat above liquidus:

\- Liquidus for FG260 ≈ 1180°C

\- Minimum pour temp = liquidus + 200°C ≈ 1380°C

\- Optimal pour temp = liquidus + 220–250°C ≈ 1400–1430°C

\- Excessive pour temp (>1460°C) → gas absorption, mold burn-on, porosity



Encode the relationship: too-low pour temp → cold shut/misrun probability rises \*\*non-linearly\*\* (use sigmoidal curve, not linear threshold). The probability of cold shut at 1340°C should be 4–5× higher than at 1380°C.



\### 5. Moisture–Hydrogen–Porosity Mechanism



Real metallurgy: H₂O in sand/atmosphere → dissociates at metal surface → atomic hydrogen absorbed → rejected during solidification → gas porosity.



Rule: defect probability for gas porosity follows

`P(porosity) ∝ (sand\_moisture)^1.5 × (humidity / 100)^1.2 × (pour\_delay)^0.8`



This non-linear compounding is what makes monsoon especially damaging — it's not additive.



\### 6. Cooling Rate Effects (Real Thermodynamics)



For grey iron solidification:

\- Cooling rate < 6°C/min: coarse graphite, weak section

\- Cooling rate 8–12°C/min: optimal pearlitic matrix, target microstructure

\- Cooling rate 12–16°C/min: fine graphite, harder

\- Cooling rate > 16°C/min: chill formation, white iron, cracks, hard spots



Higher cooling rates also amplify the effect of high Cr and P (carbide stabilizers).



\### 7. Pattern Wear Curve (Mechanical Engineering)



Pattern wear follows a known curve:

\- 0–500 cycles: minimal wear, dim\_NC rate baseline

\- 500–800 cycles: wear acceleration, dim\_NC rate +50%

\- 800–1200 cycles: dim\_NC rate triples (replacement threshold)

\- > 1200 cycles: dim\_NC rate 5× baseline, patterns should be retired



Encode as a continuous curve, not a step function.



\### 8. Furnace Drift (Real Equipment Behavior)



Induction furnace lining wears over a campaign:

\- New lining: nominal temperature accuracy ±5°C

\- After 200 heats: drift +3–5°C

\- After 500 heats: drift +8–10°C (refractory thinning)

\- After 700+ heats: drift +12–15°C, lining replacement due



Model F1's lining as 6 months older than F2's at year start.



\### 9. Dimensional Tolerances (Engine Block FG260)



Realistic CMM measurements for automotive engine block:

\- Bore Diameter: target 89.500 ± 0.050 mm (USL 89.55, LSL 89.45)

\- Deck Height: target 220.000 ± 0.080 mm

\- Wall Thickness: target 4.50 ± 0.15 mm

\- Surface Roughness Ra: target ≤ 6.3 µm



OEM Cpk requirement: ≥ 1.33 (industry standard for automotive PPAP).



Cpk formula: `Cpk = min\[(USL - μ)/(3σ), (μ - LSL)/(3σ)]`



The data must produce realistic Cpk values: Bore Cpk \~1.18 (failing), Deck Cpk \~1.45 (passing), Wall Cpk \~1.62 (passing). This is the centerpiece automotive insight on the Analytics page.



\### 10. Seasonal Patterns (Indian Climate Reality)



\- \*\*Winter (Dec–Feb):\*\* humidity 40–60%, ambient 16–24°C

\- \*\*Pre-monsoon (Mar–May):\*\* humidity 50–65%, ambient 28–38°C

\- \*\*Monsoon (Jun–Sep):\*\* humidity 75–95%, ambient 26–32°C

\- \*\*Post-monsoon (Oct–Nov):\*\* humidity 55–70%, ambient 22–30°C



Defect rates must show seasonal peak in monsoon for moisture-driven defects (blow holes, porosity) per the compounding formula in rule 5.



\### 11. Mass Balance for Chemistry



When generating chemistry, maintain realistic correlations:

\- C and CE are linked via formula (not independent)

\- Si and C have negative correlation in practice (Si replaces C in eutectic)

\- Mn and S are independent but Mn/S ratio is enforced

\- Random within ranges, but the JOINT distribution stays physically valid



\### 12. Defect Severity Distribution



For each defect type, severity distribution must reflect reality:

\- \*\*Blow Holes / Surface Defects:\*\* mostly rework (small, surface)

\- \*\*Gas Porosity / Sand Inclusion:\*\* mixed scrap/rework depending on location

\- \*\*Cold Shut / Misrun:\*\* mostly scrap (structural)

\- \*\*Shrinkage / Cracks:\*\* mostly scrap (structural integrity)

\- \*\*Dimensional NC:\*\* mostly rework (machining stock), 30% scrap if out of envelope



\## Verification Requirement



\*\*`generate\_data.py` must include a `--verify` flag\*\* that runs after generation and prints a metallurgical audit:



```

METALLURGICAL VERIFICATION REPORT

==================================

✓ All C values in IS 210 FG260 range (3.10–3.60%)

✓ All Si values in IS 210 FG260 range (1.80–2.40%)

✓ Mn/S ratio satisfied for 98.7% of heats (39 violations flagged as quality risk)

✓ Carbon Equivalent range: 3.92–4.41 (89% in eutectic band 4.0–4.3)

✓ Pour temp superheat ≥ 180°C for 96% of heats

✓ Bore Diameter Cpk: 1.18 (target ≥1.33 — FAILING, as designed)

✓ Deck Height Cpk: 1.45 (target ≥1.33 — passing)

✓ Wall Thickness Cpk: 1.62 (target ≥1.33 — passing)

✓ Monsoon porosity rate: 2.4× baseline (matches real-world seasonal effect)

✓ Pattern wear curve: dim\_NC rate 3.1× at >800 cycles (matches replacement threshold)

✓ Furnace F1 drift: +13.2°C at year end (matches refractory wear pattern)



ALL METALLURGICAL RULES VERIFIED

```



This printout is something you can SHOW the IndieFoundry metallurgist during the demo. It's the moment they trust the data.



\## Design Reference



Match the visual theme of \*\*https://zerozeta.com/\*\* — fetch and extract design system: colors, typography, spacing, button styles, card styles, overall aesthetic. The demo should feel like a Zero Zeta product.



\## Three-Page Structure



\### PAGE 1 — Landing Page (`/`)



The \*\*narrative setup\*\*. Customer reads BEFORE the demo.



\*\*Sections:\*\*



\*\*A. Hero\*\*

\- Headline: "AI for Foundry Excellence"

\- Sub: "Predict defects before pour. Cut scrap, rework, and warranty exposure. Built for automotive and engine castings."

\- CTAs: "See the Live Demo →" / "View Sample Analytics"



\*\*B. Demo Disclosure Banner\*\*

"This is a working demonstration running on metallurgically rigorous synthetic data. The ML models are real (XGBoost + LightGBM + SHAP). On your data, the same system delivers the same experience tuned to your floor."



\*\*C. Typical Foundry Challenges\*\* (8 cards)

1\. \*\*Unpredictable Scrap Rates\*\* — "5–10% of every batch lost, no way to know in advance"

2\. \*\*Multiple Defect Types\*\* — "Blow holes, gas porosity, shrinkage, cold shut, sand inclusion, misrun, dimensional non-conformance, surface defects, cracks — each with different causes"

3\. \*\*Tribal Knowledge Lock-in\*\* — "The best operators know what works. When they retire, the knowledge leaves"

4\. \*\*Seasonal Variation\*\* — "Monsoon humidity destroys quality consistency"

5\. \*\*OEM Tolerance Pressure\*\* — "Automotive customers demand Cpk ≥ 1.33. Miss it, lose the contract"

6\. \*\*Rework Cost Spiral\*\* — "Rework often more expensive than scrap — invisible in most dashboards"

7\. \*\*Customer Complaints \& Warranty\*\* — "One escaped defect can cost ₹4.5L in warranty"

8\. \*\*Energy Waste\*\* — "Furnaces over-melt 'just in case' — costs ₹50L/year per furnace"



\*\*D. Business Gains\*\* (5 outcome cards)

1\. \*\*Scrap Reduction: 30–50%\*\*

2\. \*\*Rework Reduction: 25–40%\*\*

3\. \*\*Energy Savings: 5–12%\*\*

4\. \*\*OEM Complaint Reduction: 40–70%\*\*

5\. \*\*Throughput Gain: 8–15%\*\*



Below: "For one automotive part (engine cylinder block), this translates to ₹90 Lakhs/year recovered. Across a plant's full portfolio, 8–15× that."



\*\*E. The AI/ML Stack\*\* (4 cards)

1\. \*\*XGBoost (Defect Prediction)\*\* — "Gradient-boosted decision tree classifier. Imagine 500 expert metallurgists each asking different yes/no questions about your heat parameters, then voting. Industry standard for tabular ML. Probability outputs in 50 ms."

2\. \*\*LightGBM (Yield \& Cost Regression)\*\* — "Same family as XGBoost, optimized for speed on continuous predictions."

3\. \*\*SHAP (Root Cause Explanation)\*\* — "SHapley Additive exPlanations — from game theory. For every prediction, tells you which parameters pushed the model toward 'defect' and by how much. No black box."

4\. \*\*Causal Feature Engineering\*\* — "We encode foundry physics: carbon equivalent, thermal gradients, pattern wear curves, monsoon adjustment factors. Model learns from data, features come from foundry science."



\*\*F. Mock Data Assumptions \& Definitions Panel\*\* (collapsible, default expanded)



Heading: "Full Transparency: What's in Our Mock Dataset"

Sub: "Every assumption documented. No hand-waving. Replace this with your data and the same logic applies."



\*\*F1. Dataset Overview\*\*

\- Part: Engine Cylinder Block KE-CYL-V4-220 (automotive, OEM-TATA)

\- Material grade: \*\*Grey Cast Iron FG260 per IS 210\*\*

\- Total records: 5,000 heats

\- Time period: 365 days (calendar year 2025)

\- Frequency: \~14 heats/day across 3 shifts

\- File: `data/heats\_2025.csv` (download link)



\*\*F2. Metallurgical Standards Followed\*\* (badge row)

Display badges/cards stating compliance with:

\- IS 210 (Grey Iron Castings Specification)

\- AFS Cast Iron Handbook

\- IIF Process Control Standards

\- Heine/Loper/Rosenthal \*Principles of Metal Casting\*



Caption: "Every chemistry range, defect mechanism, and cooling rule in our dataset traces to these references. A metallurgist can audit the code."



\*\*F3. Column Dictionary\*\*



Full table: \*\*Column | Type | Range / Values | Physical / Metallurgical Meaning\*\*



\[Full column dictionary as in previous prompt — every one of the 40+ columns listed with metallurgical meaning, not just statistical range.]



\*\*F4. Physics \& Chemistry Rules Encoded\*\*



Heading: "How defects are generated — the metallurgy we modeled"



9 cards (one per defect type) showing the causal drivers in metallurgical language:



\- \*\*Blow Holes\*\* ← core moisture >3%, gas evolution from binder decomposition, mold permeability insufficient | Mechanism: water vapor and binder gas trapped during solidification

\- \*\*Gas Porosity\*\* ← `P ∝ (moisture)^1.5 × (humidity)^1.2 × (delay)^0.8` | Mechanism: hydrogen dissociation from moisture, dissolved in melt, rejected at solidification

\- \*\*Shrinkage\*\* ← pour temp <1380°C, hypoeutectic CE (<4.0), thick sections without risers | Mechanism: insufficient feeding during liquid-to-solid contraction

\- \*\*Cold Shut\*\* ← pour temp <1370°C, pour delay >8 min, sigmoidal probability curve | Mechanism: insufficient superheat — metal streams solidify before fusion

\- \*\*Misrun\*\* ← pour temp <1360°C, pour rate <6 kg/s, thin sections + cold mold | Mechanism: metal freezes before filling cavity

\- \*\*Sand Inclusion\*\* ← mold hardness <78, binder <1.7%, moisture >5%, high pour rate erosion | Mechanism: mold wall erosion under metal flow stress

\- \*\*Dimensional NC\*\* ← pattern wear curve (>800 cycles = 3× rate), mold rigidity variance, thermal expansion mismatch | Mechanism: pattern dimensional drift + thermal contraction variability

\- \*\*Surface Defects\*\* ← mold temp >40°C, binder excess, mold-metal reactions | Mechanism: thermal degradation of mold surface, vein defects

\- \*\*Cracks\*\* ← cooling rate >14°C/min, Cr >0.2%, P >0.06%, mold restraint | Mechanism: hot tear (during solidification) or cold crack (post-solidification residual stress)



\*\*F5. Operational \& Business Assumptions\*\*



Bulleted list of every economic / operational assumption:

\- Overall defect rate baseline: 6.8% (matches executive narrative for current state)

\- Rework rate: 4.2%

\- Customer complaint rate: 0.21% of total heats (escaped defects)

\- Shift effects: A=4.2%, B=8.1% (junior operator + monsoon sensitivity), C=6.0%

\- Furnace F1 baseline 8°C hotter than F2, drifts +12–15°C by year-end (lining wear)

\- Monday morning Shift A: +1.5% defect rate (cold furnace start effect)

\- Pattern wear: dim\_NC rate triples at 800+ cycles

\- Scrap cost per casting: ₹6,500 (engine block — heavy material value)

\- Rework cost per casting: ₹1,800 (machining + inspection time)

\- Delay cost: ₹450/min of line stoppage

\- Customer complaint cost: ₹85,000 per incident (logistics + investigation + concession)

\- Warranty claim cost: ₹4,50,000 per claim (1 in 25 complaints escalates)

\- OEM Cpk requirement: ≥1.33 per automotive PPAP standard



\*\*F6. Metallurgical Verification Snapshot\*\*



Display the output of `generate\_data.py --verify` as a styled report block on the page. This is the audit trail in plain sight.



\*\*G. How It Works\*\* (3-step process)

\- Step 1: \*\*Ingest\*\* — "Your historical heat logs, chemistry, defect dispositions"

\- Step 2: \*\*Train\*\* — "Models learn the defect signatures unique to your plant"

\- Step 3: \*\*Deploy\*\* — "Operators see predictions in real-time before each pour"



\*\*H. POC Roadmap\*\*

8-week timeline visual: Data Extraction → Model Training → Validation → Pilot → Deployment



\*\*I. CTA Strip\*\*

"Ready to see it work?" — buttons to /demo and /analytics



\*\*J. Footer\*\*

Zero Zeta branding, "Built for IndieFoundry — Proof of Concept"



\### PAGE 2 — Interactive Demo (`/demo`)



\[Full demo page spec from previous prompt — preserved exactly:

\- Target part: Engine Cylinder Block KE-CYL-V4-220 (OEM-TATA)

\- 12 sliders + shift selector + pattern age + 3 presets (Current / Now / AI Optimal)

\- Circular risk gauge, defect prediction card, disposition card

\- Business impact stack (5 components: scrap + rework + delay + complaint + warranty)

\- SHAP-driven root cause panel + corrective actions

\- Warranty risk meter (0–10)

\- Annual impact strip at bottom

\- "DEMO · synthetic data" badge in header]



Navigation header across all pages: Logo | Home | Demo | Analytics



\### PAGE 3 — Analytics (`/analytics`)



\[Full analytics page spec from previous prompt — preserved exactly:

\- Business impact waterfall

\- Defect Pareto (by cost)

\- Monthly trend with humidity overlay

\- Shift / Furnace / Mold Line breakdowns

\- Heatmap month × shift

\- Pattern wear analysis

\- Dimensional Cpk histograms (Bore, Deck, Wall)

\- Customer complaint analysis

\- Feature correlation panel

\- Model performance card

\- Sample data preview]



\## Data Files \& Project Structure



All mock data files clearly named:



```

foundry-ai-demo/

├── README.md

├── requirements.txt

├── run.sh

├── generate\_data.py             # Data generator with --verify flag

├── train\_model.py

├── app.py

├── data/

│   ├── heats\_2025.csv                       # Main dataset, 5000 rows

│   ├── data\_dictionary.csv                  # Column definitions for reference

│   ├── metallurgical\_assumptions.json       # All physics rules in machine-readable form

│   ├── cost\_assumptions.json                # Business cost parameters

│   └── verification\_report.txt              # Output of --verify run

├── models/

│   ├── defect\_classifier.pkl

│   ├── severity\_classifier.pkl

│   ├── yield\_regressor.pkl

│   ├── warranty\_regressor.pkl

│   ├── shap\_explainer.pkl

│   ├── feature\_names.json

│   └── metrics.json

├── templates/

│   ├── landing.html

│   ├── demo.html

│   └── analytics.html

└── static/

&#x20;   ├── css/

&#x20;   │   ├── theme.css            # Zero Zeta-extracted theme

&#x20;   │   └── style.css            # Page-specific styles

&#x20;   ├── js/

&#x20;   │   ├── demo.js

&#x20;   │   └── analytics.js

&#x20;   └── img/

```



\## Backend API (unchanged from previous version)



Endpoints: `/`, `/demo`, `/analytics`, `POST /api/predict`, `GET /api/analytics/overview`, `/api/analytics/correlations`, `/api/analytics/pattern\_wear`, `/api/analytics/dimensional`, `/api/model\_info`, and \*\*new\*\* `GET /api/assumptions` → returns the full assumptions JSON for the landing page panel.



\## Models



Train all 4 models from previous spec:

\- Defect Classifier (XGBoost, 9 classes) — target accuracy >82%

\- Severity Classifier (XGBoost, 4 classes) — target accuracy >78%

\- Yield Regressor (LightGBM) — target R² >0.75

\- Warranty Risk Regressor (LightGBM) — target R² >0.70



Plus SHAP explainer for the defect classifier.



\## `run.sh`



```bash

\#!/bin/bash

set -e

pip install -r requirements.txt

\[ ! -f data/heats\_2025.csv ] \&\& python generate\_data.py --verify

\[ ! -f models/defect\_classifier.pkl ] \&\& python train\_model.py

python app.py

```



\## `README.md` Sections



1\. 30-second pitch

2\. What this demo proves

3\. \*\*Metallurgical rigor statement\*\* — list of standards followed

4\. Three pages explained

5\. How to run locally (one command)

6\. Data files inventory and what each contains

7\. Demo script — what to say at each page (3-minute customer flow)

8\. Honest note: synthetic data, physics-rigorous, drop-in replaceable with customer data

9\. 8-week POC roadmap to convert synthetic → real customer data

10\. Tech stack inventory



\## Demo Script (in README)



Three-minute customer flow:



1\. \*\*Landing page (90 sec):\*\*

&#x20;  - "Here are foundry challenges you live with daily."

&#x20;  - "Here's what AI can change — quantified outcomes."

&#x20;  - "Here's our ML stack — XGBoost, LightGBM, SHAP. Real ML, not magic."

&#x20;  - \*\*"Here's our mock data assumptions panel — every chemistry range follows IS 210 FG260, every defect mechanism is from foundry textbooks. You can audit the rules."\*\*

2\. \*\*Demo page — Current preset (60 sec):\*\*

&#x20;  - "Engine block heat with today's parameters."

&#x20;  - "Gauge: 60% defect risk. Most likely: dimensional non-conformance."

&#x20;  - "Root cause: pattern wear + cooling rate variance."

&#x20;  - "Business impact: ₹35K per casting (scrap + rework + delays + warranty reserve)."

3\. \*\*Demo page — AI Optimal preset (30 sec):\*\*

&#x20;  - "AI-recommended setpoints. Same furnace, same crew."

&#x20;  - "Gauge goes green. ₹35K → ₹2K per casting. Annual: ₹90L recovered."

4\. \*\*Analytics page (30 sec):\*\*

&#x20;  - "5,000 heats. Real Pareto. Real Cpk numbers — Bore 1.18 (failing OEM-TATA's 1.33), Deck 1.45 (passing)."

&#x20;  - "Pattern wear curve. Monsoon humidity spike. Shift B operator effect."

&#x20;  - "On your data, patterns differ, insights land the same way."



\## Critical Requirements



\- All three pages share Zero Zeta theme (fetch and apply)

\- \*\*Every chemistry, dimensional, and defect rule must trace to a metallurgical reference\*\*

\- `generate\_data.py --verify` produces an audit report — customer-visible

\- No runtime internet calls (CDN libs OK)

\- Inference <100ms per prediction

\- Synthetic data trains to >82% defect accuracy — if not, strengthen causal links

\- Coherent narrative: Current preset → red gauge \~60% + ₹35K cost; AI Optimal → green \~8% + ₹2K cost

\- Mobile-responsive

\- "DEMO · synthetic data" disclosure on every page header

\- Packages only: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, flask, flask-cors



\## Order of Execution



1\. \*\*First:\*\* Fetch https://zerozeta.com/, extract theme into `static/css/theme.css`

2\. Write `generate\_data.py` with full metallurgical rules + --verify flag

3\. Run `generate\_data.py --verify`, inspect verification report — every rule must pass

4\. `train\_model.py` — confirm all 4 model accuracy targets

5\. `app.py` with all endpoints (including `/api/assumptions`)

6\. `templates/landing.html` — most important for first impression, includes the assumptions panel

7\. `templates/analytics.html`

8\. `templates/demo.html`

9\. README + demo script + end-to-end test



\## When Done, Show Me



\- Zero Zeta theme extracted (colors, fonts listed)

\- Output of `generate\_data.py --verify` (the full metallurgical audit report)

\- Output of `train\_model.py` (4 model reports)

\- Three URLs working: `/`, `/demo`, `/analytics`

\- `./run.sh` works one-command

\- Demo script in README

\- Confirmation that landing page assumptions panel renders all 6 sub-sections (F1–F6)

