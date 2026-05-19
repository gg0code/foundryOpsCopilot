# Changelog — FoundryOps Copilot

All notable changes to this customer-facing demo are tracked here. Dates use ISO format.

## [0.4.0] — Operational Capability Framing — 2026-05-15

### Changed
- **Business Gains** section retitled to **"What the system delivers"**.
- Section eyebrow: `QUANTIFIED OUTCOMES` → `OPERATIONAL CAPABILITIES`.
- All five gain cards: replaced percentage ranges (30–50%, 25–40%, 5–12%, 40–70%, 8–15%) with
  capability **headline statements** — pre-pour intervention, earlier disposition decisions,
  furnace efficiency, warranty firewall, more good castings per shift.
- Footer strip below cards: reframed from committed savings (₹90 L / 8–15× portfolio) to
  **POC-calibrated business case** language.
- "See the math" panel renamed to **"See illustrative calculation"** with a prominent
  `⚠ ILLUSTRATIVE — NOT A COMMITMENT` disclaimer block at the top of the expanded panel.

### Rationale
- Aggressive percentage ranges (30–50%, 40–70%) created **defensive pressure** during customer
  conversations — clients pushed back asking us to defend specific numbers before any data review.
- Capability framing keeps the value clear without exposing the team to numerical cross-examination.
- Quantified outcomes now live exclusively inside the calibrated POC deliverable.

### Unaffected
- `/demo` Current → AI Optimal swing (those are live model outputs on synthetic data, not promises).
- `/analytics` retrospective stats (dataset-specific facts, clearly labelled as synthetic).
- `/capabilities` four-layer narrative.
- All hover-flap interactions on the five cards continue to work.

---

## [0.3.0] — Four-Layer AI Capabilities Page — 2026-05-15

### Added
- New `/capabilities` route — descriptive page covering four AI capability layers
  (Converses · Watches · Acts · Knows-its-limits).
- 2×2 quadrant hero with smooth-scroll links to each layer's detailed section.
- "What can AI do?" hover-pill in nav across all pages.
- Three-column layout per layer (Narrative / How it works / Illustrative example).
- "Why four layers, not one" philosophical statement panel.
- Phase 2/3/4 capability roadmap.

### Changed
- Rebranded **Foundry AI → FoundryOps Copilot** across nav logos, page titles, footers.
- Added `/capabilities` nav link to landing, demo, analytics pages.

---

## [0.2.0] — UX Pass — 2026-05-15

### Added
- Zero Zeta logo image (`static/img/zerozeta-logo.png`) replaces the gradient placeholder mark.
- Hover flaps on all challenge cards (8) and gain cards (5) explaining how the app delivers value.
- Reference SVG drawing of KE-CYL-V4-220 V4 engine block inside the Dimensional slider group.
- Warranty risk help tooltip — full explanation of what the score predicts and what drives it.
- Collapsible slider groups (Operations, Environment, Mold, Process, Chemistry, Dimensional).
- Savings calculation expandable panel with full Current vs Target math + assumptions cited.
- `app.py` Jinja `TEMPLATES_AUTO_RELOAD = True` so template edits show up without restart.

### Changed
- Hero subtitle rewritten — leads with "Every pour carries risk. Your data can reduce it."
- Hero comparison stats now symmetrical (scrap today vs with AI + complaints today vs with AI).

### Fixed
- `requirements.txt` loosened from `==` exact pins to `>=` minimums so installs work on Python 3.13/3.14
  without requiring source builds of pandas/numpy.

---

## [0.1.0] — Initial demo — 2026-05-15

### Added
- Three-page Flask demo: landing (narrative + assumptions panel), demo (interactive sliders),
  analytics (Pareto, monthly trend, Cpk histograms, pattern wear, model performance).
- Synthetic data generator (`generate_data.py --verify`) producing 5,000 FG260 heats with all
  twelve metallurgical rules encoded and cited (IS 210, AFS, IIF, Heine/Loper/Rosenthal).
- Four trained models: XGBoost defect classifier (10 classes), XGBoost severity classifier
  (4 classes), LightGBM yield regressor, LightGBM warranty risk regressor, plus SHAP explainer.
- `/api/predict` end-to-end with confidence, business impact, and SHAP root cause.
- Current / Now / AI-Optimal slider presets driving the gauge swing.
