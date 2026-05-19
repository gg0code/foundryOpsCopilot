# FoundryOps Copilot — Restructure AI Capabilities Page Around Four-Layer Story

Replace the current `/capabilities` page (`templates/capabilities.html`) with a story-driven layout built around four narrative layers. Each layer represents one dimension of what AI delivers — together they tell the complete AI value story.

## The Narrative Framework — Four Layers of AI

This is the structural backbone of the page. Every section, visual, and copy choice should reinforce one of these four ideas:

| Layer | Theme | Tagline | AI Capability | What it proves |
|-------|-------|---------|---------------|----------------|
| 1 | **AI that converses** | "Talks back" | Natural Language Q&A | Modern, accessible, no-training-needed |
| 2 | **AI that watches** | "Always on" | Anomaly Detection | Continuous value, never sleeps |
| 3 | **AI that acts** | "Decides for you" | Bayesian Optimization | Autonomous capability, not just advisory |
| 4 | **AI that knows its limits** | "Honest about uncertainty" | Confidence Intervals | Trustworthy, not overconfident |

## Page Layout

### Section A — Header

- Title: **"Four Layers of AI in FoundryOps Copilot"**
- Sub: "The defect predictor is one capability. The intelligence around it is the system. Here's how four distinct AI layers combine into a foundry operations co-pilot you can actually trust."
- Small "DEMO · synthetic data" badge consistent with other pages
- Standard navigation (Home / Demo / Analytics / AI Capabilities)

### Section B — The Four-Layer Visual Anchor

A single hero-scale visual showing all four layers stacked or arranged in a cohesive composition. Two design options — pick whichever fits the Zero Zeta aesthetic better:

**Option 1 (preferred):** A 4-quadrant grid (2×2). Each quadrant is a large card showing one layer. The four cards together form a unified visual block.

**Option 2:** Horizontal stack of 4 large numbered cards (01, 02, 03, 04), filling the viewport width, with subtle connecting lines between them.

Each layer card in this overview shows:
- Layer number (01, 02, 03, 04) in display typography
- Theme name in bold ("AI that converses")
- Tagline in italic ("Talks back")
- 1-line description
- A small accent icon (chat bubble / eye / target / scale)
- Color accent that varies subtly per layer (still within Zero Zeta palette — use 4 different accent variations)
- "Status: LIVE" badge on all four (these all work in the demo)

Hover behavior: card lifts slightly, accent color intensifies.

Click behavior: scrolls smoothly to that layer's detailed section below.

### Section C — Layer 01: AI That Converses (Natural Language Q&A)

Heading: **"01 — AI That Converses"**
Sub: *"Talks back."*

**Three-column layout below the heading:**

**Column 1 — What it does (narrative)**
> Every operator, plant head, and quality engineer in your foundry has questions about their data. Today, those questions take hours — open SAP, export to Excel, build a pivot table, talk to the analyst, wait. FoundryOps Copilot answers them in seconds, in plain English, with the numbers cited.
>
> Your team doesn't learn SQL. They don't learn dashboards. They just ask.

**Column 2 — How it works (technical, brief)**
> **Stack:** Anthropic Claude API + structured data context
>
> User question → backend assembles relevant data summary from the heat database → sends to Claude with system prompt explaining the dataset → returns natural language answer with cited figures
>
> Response time: ~2 seconds
> Cost: under ₹3 per query at scale
> Privacy: queries can be routed to on-premise Claude deployment

**Column 3 — Live example panel**
Show an interactive mini-version of the Q&A interface with 4 pre-loaded suggested prompts as clickable chips:
- "Why did Shift B have higher defects last monsoon?"
- "Which pattern is closest to retirement?"
- "Compare F1 vs F2 furnace performance"
- "What's our biggest scrap cost driver this quarter?"

Clicking a chip should send the question to the existing `/api/ask` endpoint and stream the response into a chat-style display.

**Bottom of the section — proof of value:**
A horizontal strip with 3 outcome statements:
- "Reduces analyst hours per week by 60–80%"
- "Decision latency: hours → seconds"
- "Onboards new operators 3× faster — no dashboard training needed"

### Section D — Layer 02: AI That Watches (Anomaly Detection)

Heading: **"02 — AI That Watches"**
Sub: *"Always on."*

Same three-column structure:

**Column 1 — What it does:**
> Most foundry incidents aren't catastrophic — they're slow drifts. A pattern wearing past its threshold. A furnace lining thinning. An operator's pour delay creeping up by 30 seconds a week. Humans don't notice. The AI does.
>
> An unsupervised model continuously watches every heat for unusual combinations of parameters — combinations that historically correlate with defects. When something looks off, it flags the heat before pour, not after rejection.

**Column 2 — How it works:**
> **Stack:** Isolation Forest (scikit-learn)
>
> Trained on 5,000 historical heats. Each new heat scored on a 0–1 anomaly scale. Anomalies above 0.7 trigger an alert. Explanation generated by identifying the 2 parameters that contributed most to the anomaly score.
>
> Runs in 50 ms per heat. Zero operator workload — pure background intelligence.

**Column 3 — Live example panel:**
Show a live "Anomaly Watch" widget displaying:
- "⚠ 3 anomalies detected this week"
- A small list of the top 3 anomalous heats from the synthetic dataset
- Each row shows: Heat ID, anomaly score, the 2 most-anomalous parameters, predicted defect probability

Example display:
```
HT-2025-04231   Score 0.87
  Pour Temp 1340°C (3.2σ low)
  Carbon Eq 4.45 (2.8σ high)
  → Predicted defect risk: 73%

HT-2025-04102   Score 0.78
  Sand Moisture 5.4% (3.5σ high)
  Pour Delay 9.2 min (2.9σ high)
  → Predicted defect risk: 68%

HT-2025-03987   Score 0.71
  Cooling Rate 15.8°C/min (2.6σ high)
  Cr Content 0.22% (2.4σ high)
  → Predicted defect risk: 54%
```

**Bottom proof strip:**
- "Catches 80%+ of drift conditions before defect formation"
- "Zero operator effort — runs continuously in background"
- "Surfaces patterns no human dashboard can monitor"

### Section E — Layer 03: AI That Acts (Bayesian Optimization)

Heading: **"03 — AI That Acts"**
Sub: *"Decides for you."*

Same three-column structure:

**Column 1 — What it does:**
> Prediction is useful. Action is valuable. The defect predictor tells you a heat is risky. The optimizer tells you exactly what to change — and proves the new setpoints will work, before you pour.
>
> Your senior metallurgist's tribal knowledge is "if porosity is up, drop moisture and increase pour temp by 10°." The AI finds the precise, multi-parameter optimum — across 12 dimensions simultaneously — in 3 seconds. No trial heats. No guessing.

**Column 2 — How it works:**
> **Stack:** scikit-optimize (Bayesian optimization with Gaussian Process surrogate model)
>
> Objective function: minimize (predicted defect probability) + (predicted warranty risk × 0.3)
> Search space: 12 operator-controllable parameters within physical bounds
> Iterations: 30–50 simulated parameter combinations
> Convergence: typically 3–4 seconds
> Constraints: respects physical limits (you cannot ask for pour temp below liquidus)

**Column 3 — Live demo trigger:**
Show a large interactive button: **"🎯 Run Optimization"**

When clicked:
1. Display animated progress: "Iteration 1/30... 2/30... evaluating..."
2. Show a small chart updating in real-time: x-axis = iteration, y-axis = predicted defect probability, dots appearing as it searches
3. On completion: show the optimal parameters in a small table, with a link "Apply on demo page →" that navigates to /demo with those parameters pre-loaded

**Bottom proof strip:**
- "Replaces 3-day metallurgist trial-and-error with 3-second AI search"
- "Tested 30+ parameter combinations per call — far beyond human capacity"
- "Recommendations are physics-constrained — never asks for impossible setpoints"

### Section F — Layer 04: AI That Knows Its Limits (Confidence Intervals)

Heading: **"04 — AI That Knows Its Limits"**
Sub: *"Honest about uncertainty."*

Same three-column structure:

**Column 1 — What it does:**
> The worst AI systems are confident when they shouldn't be. The best ones tell you when they don't know.
>
> Every prediction in FoundryOps Copilot ships with a confidence interval. When the model has seen plenty of similar heats and the prediction is stable, the band is tight. When the heat is unusual — extreme parameters, rare combinations — the band widens and the model says so. Your operators learn when to trust the AI and when to bring in a human.
>
> This is what separates an AI tool from AI snake oil.

**Column 2 — How it works:**
> **Stack:** XGBoost with prediction interval estimation
>
> For each prediction, we compute the standard deviation across the per-class probability vector. We combine that with the model's training-data density in the local neighborhood of the prediction (KDE-based).
>
> Output:
>   62% defect risk (tight: 58–66%) → HIGH CONFIDENCE — operators should act
>   62% defect risk (wide: 41–83%) → LOW CONFIDENCE — flag for human review
>
> Wide intervals are not a model failure. They are the model's honesty.

**Column 3 — Live example panel:**
Show 3 example predictions side-by-side, each with a different confidence band:

```
HEAT A — HIGH CONFIDENCE
  Risk: 14% [12%–16%]   Band width: 4pp
  Status: ✓ Safe to pour

HEAT B — MEDIUM CONFIDENCE
  Risk: 38% [27%–49%]   Band width: 22pp
  Status: ⚠ Manual review recommended

HEAT C — LOW CONFIDENCE
  Risk: 51% [22%–80%]   Band width: 58pp
  Status: ⚠ Model uncertain — consult metallurgist
```

Style the bands as visual gradient bars where the width visually conveys uncertainty.

**Bottom proof strip:**
- "Operators learn when to trust AI vs. when to escalate"
- "Wide bands trigger human-in-the-loop — never blind automation"
- "Industry-standard practice — used in medical AI, autonomous driving, financial models"

### Section G — Why Four Layers, Not One

A single full-width statement panel between sections F and H:

> **A complete AI system isn't one capability. It's four working together.**
>
> Conversation without action is just a chatbot.
> Monitoring without prediction is just a dashboard.
> Action without confidence is just a gamble.
> Confidence without conversation is just a black box.
>
> FoundryOps Copilot is built around this principle: every layer reinforces every other layer. The Q&A surfaces insights the anomaly detector found. The optimizer respects the confidence intervals. The system tells you what it knows, what it suspects, and what it doesn't know.
>
> That's the difference between AI you demo once and AI your team uses every shift.

### Section H — Roadmap Beyond the Four Layers

Keep the existing roadmap timeline from the earlier version, but reframe it as "What comes after the four core layers."

Header: **"Beyond the Four Layers — Capability Roadmap"**
Sub: "The four layers above are live in this demo. Here's what we build next together."

```
PHASE 2 — POC Extensions (Weeks 6–12)
├── 7-Day Defect Forecast
│   "Combines weather + production schedule + pattern wear"
├── Vision-Based Surface Inspection
│   "Camera on shakeout line, 200ms per casting"
└── Multi-Part Family Expansion
    "From cylinder block to crankcase, head, housing — same architecture"

PHASE 3 — Production Scale (Months 3–6)
├── Predictive Maintenance
│   "LSTM on furnace/mold-line vibration and current"
├── Energy Optimization Engine
│   "Charge mix + holding time optimizer"
└── OEM Integration
    "Automated PPAP documentation generation"

PHASE 4 — Plant-Wide Intelligence (Months 6–12)
├── Multi-Plant Deployment
├── Supply Chain AI
│   "Charge mix optimization across vendors"
└── Closed-Loop Process Control
    "Direct setpoint actuation, not just recommendation"
```

### Section I — Footer CTA

> "Each layer earns its place by delivering measurable value. Ready to see them work on your data?"

Button: **"Scope the 8-Week POC →"**

## Visual & Aesthetic Requirements

- Maintain Zero Zeta theme throughout (colors, typography, spacing)
- Each layer's three-column section should feel like a "chapter" — generous whitespace between sections, clear visual breaks
- The 4-quadrant hero (Section B) is the page's centerpiece — make it visually dominant
- Layer numbering in display typography (01, 02, 03, 04) should be large and confident
- Subtle color accent variation between layers — same palette family but distinct enough to differentiate at a glance
- The "live example panels" (right column of each section) should look interactive — not static screenshots
- Mobile responsive: 3-column layouts collapse to vertical stack, 2×2 hero becomes vertical stack
- Smooth scroll behavior when navigation chips are clicked

## Technical Requirements

- All four layers must genuinely work — Q&A calls the actual /api/ask endpoint, anomaly widget pulls real anomalies from the dataset, optimization button calls /api/optimize, confidence intervals come from real model output
- No fake demos — every interactive panel uses live backend calls
- Page load under 2 seconds
- "DEMO · synthetic data" badge in header consistent with other pages

## Update CLAUDE.md

Add to the "Demo Narrative" section a note that the AI Capabilities page tells the four-layer story:
- Layer 01: AI that converses (Q&A)
- Layer 02: AI that watches (anomaly detection)
- Layer 03: AI that acts (Bayesian optimization)
- Layer 04: AI that knows its limits (confidence intervals)

Add to "Non-Negotiable Principles":
> The Four-Layer narrative on /capabilities is the canonical AI story. Any new AI capability we add maps to one of these four layers — or earns its own fifth layer with a clear theme. Don't add capabilities that don't fit the story.

## Update CHANGELOG.md

Add entry:

```markdown
## [0.3.0] — Four-Layer AI Story — [today's date]

### Changed
- /capabilities page restructured around four-layer AI narrative
- Layer 01: AI that converses — Natural Language Q&A
- Layer 02: AI that watches — Anomaly Detection
- Layer 03: AI that acts — Bayesian Optimization
- Layer 04: AI that knows its limits — Confidence Intervals

### Added
- Hero 4-quadrant visual anchor showing all four layers
- Three-column layout per layer: Narrative / Technical / Live Example
- Interactive live example panels for each layer (real backend calls, not mockups)
- "Why Four Layers, Not One" philosophical statement panel
- Phase 2/3/4 roadmap explicitly framed as "beyond the four core layers"

### Customer Feedback Captured
- Customer asked for AI capabilities narrative that tells a coherent story, not a feature checklist — addressed via four-layer framework
```

## Demo Script for /capabilities (add to README)

When you reach the AI Capabilities page during the customer demo (60 seconds total):

1. **Show the 4-quadrant hero (10 sec):** "Four layers of AI. Each does something different. Together they're a co-pilot."
2. **Click Layer 01 — Conversation (15 sec):** Type a question. Show the response. "Your team doesn't learn dashboards. They just ask."
3. **Click Layer 02 — Watch (10 sec):** Point to the live anomaly list. "Background. Always on. Catches drift before defects."
4. **Click Layer 03 — Act (15 sec):** Hit the optimize button. Show iteration progress. "Three days of trial heats → three seconds."
5. **Click Layer 04 — Limits (10 sec):** Show the three confidence bands. "The model tells you when to trust it and when to bring in a human. That's not a weakness. That's why you can trust it."

End: "Each layer is live in today's demo. Ready to make this real on your data?"

## When Done, Show Me

- Updated /capabilities page rendering correctly with all four layer sections
- All interactive panels work (Q&A, anomaly list, optimization button, confidence display)
- CLAUDE.md and CHANGELOG.md updated
- One-command launch (`./run.sh`) still works
- Other pages (landing, demo, analytics) unaffected