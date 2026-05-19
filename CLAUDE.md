\# CLAUDE.md — Foundry AI Demo Project



\## What This Project Is



A \*\*customer demo\*\* for Kirloskar (automotive/engine castings manufacturer in India). Three-page Flask web application demonstrating AI-driven casting defect prediction, root cause analysis, and business impact quantification. Built on metallurgically rigorous synthetic data because we don't have customer data yet.



\*\*Built by:\*\* Zero Zeta

\*\*Audience:\*\* Kirloskar plant heads, quality engineers, metallurgists, CTOs

\*\*Delivery mode:\*\* Live laptop demo (single command launch)

\*\*Goal:\*\* Convert one demo into an 8-week paid POC



\## Non-Negotiable Principles



These rules override any conflicting instruction or "improvement" idea:



1\. \*\*Metallurgical rigor is sacred.\*\* Every chemistry range, defect mechanism, and physical rule in the data generator must trace to a real foundry reference (IS 210, AFS Handbook, IIF standards, Heine/Loper/Rosenthal). Never invent thresholds that "sound right." If you need to add a rule, cite the metallurgical principle.



2\. \*\*Honesty about synthetic data is a feature, not a weakness.\*\* Every page header displays "DEMO · Trained on synthetic data." The landing page has an entire panel documenting mock data assumptions. Never hide, soften, or remove this disclosure.



3\. \*\*The story is anchored. Do not drift.\*\*

&#x20;  - Part: Engine Cylinder Block KE-CYL-V4-220

&#x20;  - Material: Grey Cast Iron FG260 per IS 210

&#x20;  - Customer: OEM-TATA (with OEM-MAHINDRA and OEM-ASHOK\_LEYLAND as secondary)

&#x20;  - Current state: 6.8% scrap + 4.2% rework + 14 OEM complaints

&#x20;  - Target state: 3.5% scrap + 2.0% rework + <3 complaints

&#x20;  - Annual savings: ₹90 Lakhs for this single part

&#x20;  - Total business impact: ₹1.42 Cr (current state)



4\. \*\*The Current → AI Optimal demo moment must always work.\*\*

&#x20;  - "Current" preset → red gauge \~60% defect risk, \~₹35K per-casting cost

&#x20;  - "AI Optimal" preset → green gauge \~8% defect risk, \~₹2K per-casting cost

&#x20;  - If any code change breaks this, the demo is broken. Test this flow after every change.



5\. \*\*Real ML, not rule engines.\*\* Predictions must come from trained XGBoost/LightGBM models, not from hand-coded rules. The rule engine was the v0 demo; we have moved past it.



6\. \*\*Inference latency must stay under 100ms.\*\* Sliders feel broken otherwise.



7\. \*\*No runtime internet dependencies.\*\* CDN libraries for fonts and chart libs are acceptable at page load; runtime predictions are local.



8\. \*\*No promised percentages on the landing page.\*\* Quantified outcomes (X% scrap reduction, Y% energy savings) live only inside the POC business case, calibrated to actual customer data. Public-facing pages describe \*operational capabilities\*, not committed numbers. This protects the company from defensive conversations during pre-POC discussions.



\## Project Structure



```

foundry-ai-demo/

├── CLAUDE.md                                # This file — read first

├── README.md                                # User-facing run instructions + demo script

├── requirements.txt

├── run.sh                                   # One-command launch

├── generate\_data.py                         # Synthetic data generator with --verify flag

├── train\_model.py                           # Trains 4 models + SHAP explainer

├── app.py                                   # Flask backend

├── data/

│   ├── heats\_2025.csv                       # Primary dataset, 5000 rows

│   ├── data\_dictionary.csv                  # Column definitions

│   ├── metallurgical\_assumptions.json       # Physics rules in machine-readable form

│   ├── cost\_assumptions.json                # Business cost parameters

│   └── verification\_report.txt              # Output of generate\_data.py --verify

├── models/

│   ├── defect\_classifier.pkl                # XGBoost, 9 classes

│   ├── severity\_classifier.pkl              # XGBoost, 4 classes

│   ├── yield\_regressor.pkl                  # LightGBM

│   ├── warranty\_regressor.pkl               # LightGBM

│   ├── shap\_explainer.pkl                   # For root cause panel

│   ├── feature\_names.json

│   └── metrics.json                         # Accuracy, confusion matrices, R²

├── templates/

│   ├── landing.html                         # Page 1 — narrative + assumptions panel

│   ├── demo.html                            # Page 2 — interactive prediction

│   └── analytics.html                       # Page 3 — data analytics

└── static/

&#x20;   ├── css/

&#x20;   │   ├── theme.css                        # Zero Zeta-extracted theme variables

&#x20;   │   └── style.css                        # Page-specific styles

&#x20;   ├── js/

&#x20;   │   ├── demo.js

&#x20;   │   └── analytics.js

&#x20;   └── img/

```



\## Domain Reference — Foundry \& Metallurgy Cheat Sheet



Use this when generating data, naming variables, or writing copy. Do not invent terms.



\*\*Material grade:\*\* Grey Cast Iron FG260 per IS 210

\- C: 3.10–3.60% | Si: 1.80–2.40% | Mn: 0.50–0.90% | S: 0.06–0.12% | P: 0.05–0.12% | Cr: ≤0.20%

\- Mn/S rule: Mn ≥ 1.7 × S + 0.3

\- Carbon Equivalent: CE = C + (Si + P)/3 — target 4.0–4.3 (eutectic)

\- Liquidus ≈ 1180°C, optimal superheat = liquidus + 220–250°C → pour temp 1400–1430°C



\*\*9 defect classes (in code use these exact strings):\*\*

`None, Blow\_Holes, Gas\_Porosity, Shrinkage, Cold\_Shut, Misrun, Sand\_Inclusion, Dimensional\_NC, Surface\_Defects, Cracks`



\*\*4 severity classes:\*\*

`None, Minor\_Rework, Major\_Rework, Scrap`



\*\*4 dispositions:\*\*

`OK, Rework, Scrap, Customer\_Reject`



\*\*Cpk formula:\*\* `Cpk = min\[(USL − μ)/(3σ), (μ − LSL)/(3σ)]`

\- OEM-TATA requires Cpk ≥ 1.33 (automotive PPAP standard)

\- Bore Cpk target \~1.18 (intentionally failing — drives the demo narrative)

\- Deck Cpk target \~1.45 (passing)

\- Wall Cpk target \~1.62 (passing)



\*\*Indian seasonal pattern:\*\*

\- Winter (Dec–Feb): humidity 40–60%, temp 16–24°C

\- Pre-monsoon (Mar–May): humidity 50–65%, temp 28–38°C

\- Monsoon (Jun–Sep): humidity 75–95%, temp 26–32°C — defect spike season

\- Post-monsoon (Oct–Nov): humidity 55–70%, temp 22–30°C



\*\*Operator/shift effects:\*\*

\- Shift A (OP-104, 12yr exp): 4.2% defect rate

\- Shift B (OP-217, 4yr exp): 8.1% — junior + monsoon sensitive

\- Shift C (OP-308, 8yr exp): 6.0%

\- Monday morning Shift A: +1.5% (cold furnace start)



\*\*Pattern wear curve:\*\* dim\_NC rate triples after 800 cycles, retires at \~1200.



\*\*Furnace drift:\*\* F1 baseline +8°C over F2, lining wear adds +12–15°C by year end.



\*\*Cost assumptions:\*\*

\- Scrap cost: ₹6,500 per casting | Rework cost: ₹1,800 per casting

\- Delay cost: ₹450/min line stoppage | Complaint cost: ₹85,000 per incident

\- Warranty claim cost: ₹4,50,000 per claim (1 in 25 complaints escalates)



\## Tech Stack — Locked



\*\*Python packages allowed:\*\* pandas, numpy, scikit-learn, xgboost, lightgbm, shap, flask, flask-cors, gunicorn



\*\*Not allowed:\*\* PyTorch, TensorFlow, Keras, sklearn-pipelines beyond basic use, any cloud SDK, any database — keep it CSV + pickle.



\*\*Frontend:\*\* Vanilla HTML + CSS + JS. Chart.js or D3 via CDN for analytics charts. No React, no Vue, no build step.



\*\*Design source:\*\* Theme extracted from https://zerozeta.com/ into `static/css/theme.css`. Do not invent new colors, fonts, or spacing — use the variables.



\## Model Targets (must hit these)



\- Defect Classifier accuracy: >82% on holdout

\- Severity Classifier accuracy: >78%

\- Yield Regressor R²: >0.75

\- Warranty Risk Regressor R²: >0.70



If any model misses target, strengthen causal links in `generate\_data.py` (do not weaken target).



\## Demo Narrative — The Three-Minute Script



This is the customer flow. Build everything to support it. Do not add features that break it.



\*\*1. Landing page (90 sec)\*\*

\- Customer reads foundry challenges, recognizes their pain

\- Sees the 5 business gain numbers

\- Sees the AI/ML stack explained transparently

\- \*\*Expands the assumptions panel — sees metallurgical standards followed, sees verification report\*\*

\- Conclusion: "this is real ML on principled data"



\*\*2. Demo page — Current preset (60 sec)\*\*

\- Click "Current" — gauge fills red at \~60%

\- Defect type: Dimensional\_NC (or Gas\_Porosity)

\- Root cause panel shows pattern wear + cooling rate + moisture

\- Business impact stack: \~₹35K per casting

\- Customer recognizes this as their current reality



\*\*3. Demo page — AI Optimal preset (30 sec)\*\*

\- Click "AI Optimal" — gauge sweeps to green at \~8%

\- Business impact: \~₹2K per casting

\- Annual impact strip: ₹90L recovered for this part



\*\*4. Analytics page (30 sec)\*\*

\- Pareto by cost (not just count)

\- Monthly trend showing monsoon spike with humidity overlay

\- Dimensional Cpk histograms — Bore failing at 1.18, Deck/Wall passing

\- Pattern wear curve crossing 800 cycles

\- Model performance card with real accuracy numbers



\## What to Do When the User Asks for Changes



\*\*Always:\*\*

\- Preserve metallurgical rigor — if a change would break a physics rule, push back

\- Preserve the Current → AI Optimal demo moment

\- Re-run `generate\_data.py --verify` after data generator changes

\- Re-train models if data changes

\- Keep the synthetic data disclosure visible



\*\*Ask before:\*\*

\- Adding new defect classes (changes the model schema)

\- Changing the target part or material grade (changes the whole narrative)

\- Removing the assumptions panel (this is the credibility anchor)

\- Changing the 3-page structure



\*\*Push back if asked to:\*\*

\- Hide or remove the "DEMO · synthetic data" disclosure

\- Use real customer data we don't have

\- Replace ML models with rules

\- Add features that require internet at runtime

\- Use libraries outside the locked stack



\## Style \& Code Conventions



\- \*\*Python:\*\* Type hints where useful, docstrings on every function, snake\_case

\- \*\*JS:\*\* Vanilla ES6+, no jQuery, debounce slider inputs at 150ms

\- \*\*CSS:\*\* Use theme.css variables, no hardcoded colors in component CSS

\- \*\*Filenames:\*\* snake\_case for Python, kebab-case for static assets

\- \*\*Comments:\*\* Cite the metallurgical principle when encoding physics. Example:

```python

&#x20; # Mn/S ratio rule: Mn ≥ 1.7\*S + 0.3 to neutralize sulphur as MnS

&#x20; # Reference: AFS Cast Iron Handbook, Section 4.2

&#x20; mn\_min = 1.7 \* sulphur + 0.3

```



\## Common Mistakes to Avoid



\- Generating chemistry with independent random values (must respect joint distributions)

\- Using linear thresholds where the metallurgy is non-linear (pour temp, moisture-porosity)

\- Forgetting to update `data\_dictionary.csv` when adding columns

\- Letting Bore Cpk land at 1.4+ (breaks the "OEM is unhappy" narrative — must be \~1.18)

\- Calling Carbon Equivalent "CE" in some places and "Carbon\_Equivalent" in others — pick one and stick

\- Forgetting that Monday morning Shift A has a defect penalty — this is a key story detail

\- Making the AI Optimal preset gauge land above 15% — must feel clearly green

\- Reintroducing percentage ranges on the landing page or any public-facing surface. Capability statements only. Percentages belong in the calibrated POC deliverable.



\## How to Run



```bash

./run.sh

```



This installs deps, generates data (if missing), trains models (if missing), and starts Flask on port 5000. Browser opens automatically to landing page.



To regenerate everything from scratch:

```bash

rm -rf data/heats\_2025.csv models/\*.pkl

./run.sh

```



To run metallurgical verification only:

```bash

python generate\_data.py --verify

```



\## URLs



\- `/` — Landing page (narrative + assumptions panel)

\- `/demo` — Interactive prediction demo

\- `/analytics` — Data analytics dashboard

\- `/api/predict` (POST) — JSON parameters in, prediction out

\- `/api/analytics/\*` (GET) — Aggregated stats for analytics page

\- `/api/assumptions` (GET) — Returns assumptions JSON for landing page



\## Success Definition



\*\*The demo succeeds if, after seeing it:\*\*

1\. A Kirloskar metallurgist asks "can we audit your data generation rules?" (curiosity, not suspicion)

2\. A Kirloskar plant head asks "how long to run this on our data?" (intent, not skepticism)

3\. The 6.8% → 3.5% scrap story is repeated back by someone in the room (narrative stickiness)

4\. We are invited to scope an 8-week paid POC (commercial outcome)



If a change you're about to make doesn't serve at least one of these outcomes, reconsider it.



\## When in Doubt



The user is preparing for a high-stakes customer meeting. \*\*Bias toward shipping a working, coherent demo over feature richness.\*\* A simpler demo that works flawlessly beats a richer demo that has any rough edge.



If a change is ambiguous, ask. If a metallurgical rule is unclear, look it up rather than guess. If two requirements conflict, surface the conflict instead of silently picking one.

