// Foundry AI Demo — interactive prediction logic
// Reads slider/select inputs, debounces, calls /api/predict, paints results.

(() => {
    const SLIDER_IDS = [
        'pattern_age_cycles','ambient_humidity_pct','ambient_temp_C',
        'sand_moisture_pct','core_moisture_pct',
        'mold_hardness_B','mold_temp_C','binder_pct','mold_permeability',
        'pour_temp_C','pour_rate_kg_per_s','pour_delay_min','cooling_rate_C_per_min',
        'fade_time_min','inoculant_dose_pct','furnace_temp_drift_C',
        'C_pct','Si_pct','Mn_pct','S_pct','P_pct','Cr_pct',
        'bore_diameter_mm','deck_height_mm','wall_thickness_mm',
    ];
    const SELECT_IDS = ['shift','furnace_id','mold_line_id','pattern_id','season'];

    const inr = (n) => '₹' + Math.round(n).toLocaleString('en-IN');
    const fmt = (v, d=2) => (v == null || isNaN(v)) ? '—' : Number(v).toFixed(d);

    // AI-optimal preset expected per-casting loss (computed once at model training; ~₹1,800)
    // Used as the constant "best case" reference in the annual extrapolation strip.
    const AI_OPTIMAL_PER_CASTING = 1800;
    const ANNUAL_VOLUME = 14000;
    const AI_OPTIMAL_ANNUAL = AI_OPTIMAL_PER_CASTING * ANNUAL_VOLUME; // ₹2.52 Cr

    // ===================================================================
    // Slider <-> value rendering
    // ===================================================================
    function bindSliders() {
        for (const id of SLIDER_IDS) {
            const slider = document.getElementById(id);
            const label = document.getElementById('v-' + id);
            if (!slider) continue;
            const step = parseFloat(slider.step) || 1;
            const decimals = (slider.step && slider.step.includes('.'))
                ? slider.step.split('.')[1].length : 0;
            slider.addEventListener('input', () => {
                if (label) label.textContent = parseFloat(slider.value).toFixed(decimals);
                updateCE();
                debouncedPredict();
            });
        }
        for (const id of SELECT_IDS) {
            const sel = document.getElementById(id);
            if (!sel) continue;
            sel.addEventListener('change', debouncedPredict);
        }
    }

    function updateCE() {
        const c = parseFloat(document.getElementById('C_pct').value);
        const si = parseFloat(document.getElementById('Si_pct').value);
        const p = parseFloat(document.getElementById('P_pct').value);
        const ce = c + (si + p) / 3;
        const node = document.querySelector('#ce-readout span');
        if (node) node.textContent = ce.toFixed(2);
    }

    function readPayload() {
        const r = {};
        for (const id of SLIDER_IDS) {
            const el = document.getElementById(id);
            if (el) r[id] = parseFloat(el.value);
        }
        for (const id of SELECT_IDS) {
            const el = document.getElementById(id);
            if (el) r[id] = el.value;
        }
        // Derived
        r.carbon_equivalent = r.C_pct + (r.Si_pct + r.P_pct) / 3;
        r.superheat_C = r.pour_temp_C - 1180;
        // Mn/S rule
        r.mn_s_ratio_ok = r.Mn_pct >= 1.7 * r.S_pct + 0.3;
        // Surface roughness — derive lightly
        r.surface_roughness_Ra_um = 5.2;
        r.casting_weight_kg = 215;
        return r;
    }

    // ===================================================================
    // Network: predict with debounce
    // ===================================================================
    let predictAbort = null;
    let predictTimer = null;
    function debouncedPredict() {
        clearTimeout(predictTimer);
        predictTimer = setTimeout(doPredict, 150);
    }

    async function doPredict() {
        if (predictAbort) predictAbort.abort();
        predictAbort = new AbortController();
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(readPayload()),
                signal: predictAbort.signal,
            });
            if (!res.ok) throw new Error('predict ' + res.status);
            const data = await res.json();
            paintResults(data);
        } catch (e) {
            if (e.name !== 'AbortError') console.error('Predict failed', e);
        }
    }

    // ===================================================================
    // Paint results
    // ===================================================================
    function paintResults(d) {
        // ----- gauge -----
        const pct = Math.max(0, Math.min(100, d.risk_gauge_pct || (d.p_any_defect * 100)));
        const card = document.getElementById('gauge-card');
        const valueEl = document.getElementById('gauge-value');
        const labelEl = document.getElementById('gauge-label');
        const arcEl = document.getElementById('gauge-arc');

        // dasharray ~577.9 for r=92 full circle
        const CIRC = 2 * Math.PI * 92;
        arcEl.setAttribute('stroke-dasharray', CIRC.toFixed(1));
        arcEl.setAttribute('stroke-dashoffset', (CIRC * (1 - pct / 100)).toFixed(1));
        valueEl.textContent = Math.round(pct) + '%';

        // colour zones — match the visual narrative the spec demands
        let cls = 'risk-low', label = 'Low risk', stroke = '#16A34A';
        if (pct >= 60) { cls = 'risk-high'; label = 'High risk'; stroke = '#DC2626'; }
        else if (pct >= 25) { cls = 'risk-mid'; label = 'Elevated risk'; stroke = '#F59E0B'; }
        card.classList.remove('risk-low','risk-mid','risk-high');
        card.classList.add(cls);
        labelEl.textContent = label;
        arcEl.setAttribute('stroke', stroke);

        // ----- defect class card -----
        const className = (d.top_defect_class || '—').replace(/_/g,' ');
        document.getElementById('defect-class-name').textContent = className;
        document.getElementById('defect-class-prob').textContent =
            `model confidence: ${((d.top_defect_probability || 0) * 100).toFixed(1)}%`;

        const dispEl = document.getElementById('disposition-pill');
        dispEl.textContent = d.disposition || 'OK';
        dispEl.setAttribute('data-disp', d.disposition || 'OK');

        document.getElementById('yield-val').textContent =
            (d.predicted_yield_pct != null ? d.predicted_yield_pct.toFixed(1) : '—') + '%';
        document.getElementById('severity-val').textContent =
            (d.predicted_severity || '—').replace(/_/g,' ');

        // ----- business impact -----
        const b = d.business_impact || {};
        document.getElementById('cost-scrap').textContent     = inr(b.scrap_cost || 0);
        document.getElementById('cost-rework').textContent    = inr(b.rework_cost || 0);
        document.getElementById('cost-delay').textContent     = inr(b.delay_cost || 0);
        document.getElementById('cost-complaint').textContent = inr(b.complaint_cost || 0);
        document.getElementById('cost-warranty').textContent  = inr(b.warranty_cost || 0);
        document.getElementById('cost-total').textContent     = inr(b.total_cost || 0);

        // ----- warranty meter -----
        const w = d.predicted_warranty_risk || 0;
        document.getElementById('warranty-value').textContent = w.toFixed(1) + ' / 10';
        document.getElementById('warranty-needle').style.left = (w * 10).toFixed(1) + '%';

        // ----- root cause SHAP -----
        document.getElementById('root-cause-target').textContent = className;
        renderShap(d.root_causes || []);
        renderActions(d);

        // ----- annual strip (extrapolated, not actual annual loss) -----
        const perCasting = b.total_cost || 0;
        const annualThis = perCasting * ANNUAL_VOLUME;
        const inrCr = (n) => '₹' + (n / 1e7).toFixed(2) + ' Cr';
        document.getElementById('annual-this').textContent    = inrCr(annualThis);
        document.getElementById('annual-optimal').textContent = inrCr(AI_OPTIMAL_ANNUAL);
        const gap = annualThis - AI_OPTIMAL_ANNUAL;
        const gapEl = document.getElementById('annual-gap');
        const gapSub = document.getElementById('annual-gap-sub');
        if (gap > 0) {
            gapEl.textContent = inrCr(gap);
            gapEl.classList.remove('neg', 'zero');
            gapSub.textContent = 'improvement headroom vs AI-optimal';
        } else if (gap < -1e6) {
            // The slider is already better than the AI-optimal preset reference — rare but possible
            gapEl.textContent = inrCr(-gap);
            gapEl.classList.add('zero');
            gapSub.textContent = 'already below the AI-optimal reference';
        } else {
            gapEl.textContent = '— at target —';
            gapEl.classList.add('zero');
            gapSub.textContent = 'this setpoint already matches the AI-optimal extrapolation';
        }

        // ----- latency -----
        document.getElementById('latency-value').textContent =
            (d.latency_ms != null ? d.latency_ms.toFixed(0) : '—') + ' ms';
    }

    function renderShap(rows) {
        const cont = document.getElementById('shap-rows');
        if (!rows.length) { cont.innerHTML = '<div class="text-sm text-soft">No attributions available.</div>'; return; }
        const maxAbs = Math.max(...rows.map(r => Math.abs(r.shap_value || 0))) || 1;
        cont.innerHTML = rows.map(r => {
            const v = r.shap_value || 0;
            const w = Math.abs(v) / maxAbs * 100;
            const pos = v > 0;
            return `
                <div class="shap-row">
                    <div class="feat">${(r.feature || '').replace(/_/g,' ')}</div>
                    <div class="bar-wrap">
                        <div class="bar ${pos ? 'pos' : 'neg'}"
                             style="${pos ? 'left:50%;' : 'right:50%;'} width:${(w/2).toFixed(1)}%;"></div>
                    </div>
                    <div class="val">${v >= 0 ? '+' : ''}${v.toFixed(2)}</div>
                </div>`;
        }).join('');
    }

    function renderActions(d) {
        const top = d.top_defect_class;
        const yieldP = d.predicted_yield_pct || 99;
        const recs = [];

        // Use the defect class to suggest specific actions
        const map = {
            'Gas_Porosity':    ['Dry sand additional 2h pre-pour (target moisture <3.5%)',
                                'Add inoculant boost (FeSi 0.05% over base) before pour',
                                'Reduce pour delay to <3 min (cover ladle to limit H2 pickup)'],
            'Dimensional_NC':  ['Schedule pattern PT-C for refurbishment — wear at retirement threshold',
                                'Check mold hardness uniformity; tighten compaction setpoint to 84B±2',
                                'Re-zero CMM on next 5 castings and audit dimensional drift'],
            'Cold_Shut':       ['Raise pour temperature to 1410–1420°C (+15°C above current)',
                                'Cut pour delay <2 min from tap',
                                'Pre-heat ladle to ≥1380°C before tap'],
            'Misrun':          ['Raise pour temperature to 1415°C and increase pour rate to 7.5 kg/s',
                                'Heat the mold to ≥35°C if cold-batch start',
                                'Verify gating cross-section for thin sections'],
            'Shrinkage':       ['Raise pour temp to 1410–1420°C; verify riser geometry on thick sections',
                                'Bring CE back to 4.10–4.20 (currently hypoeutectic)',
                                'Reduce inoculation fade by inoculating in stream'],
            'Blow_Holes':      ['Reduce core moisture to <3.0% (extended baking)',
                                'Increase mold permeability — check vent placement',
                                'Decrease binder by 0.1% to lower gas evolution'],
            'Sand_Inclusion':  ['Increase mold hardness to ≥83B (verify squeeze pressure)',
                                'Raise bentonite content by 0.1%',
                                'Reduce sand moisture to <4.0%'],
            'Surface_Defects': ['Lower mold temperature to <40°C; allow cool-down between pours',
                                'Reduce binder to 1.9% (currently excess)',
                                'Check coating uniformity on cope/drag interfaces'],
            'Cracks':          ['Slow cooling rate to <12°C/min; insulate exposed sections',
                                'Restrict Cr addition; check returns for alloy carryover',
                                'Confirm Mn ≥ 1.7·S + 0.3 (sulphur neutralization)'],
        };
        if (top && map[top]) recs.push(...map[top]);

        // Generic high-yield action
        if (yieldP < 95) {
            recs.unshift(`Predicted yield is ${yieldP.toFixed(1)}% — investigate before pouring full batch`);
        }
        if (recs.length === 0) {
            recs.push('No deviations detected. Continue with planned heat parameters.');
        }
        document.getElementById('actions-list').innerHTML = recs.slice(0, 4).map(r =>
            `<div class="action-item"><div class="dot"></div><div>${r}</div></div>`
        ).join('');
    }

    // ===================================================================
    // Presets
    // ===================================================================
    let PRESETS = null;
    async function loadPresets() {
        const res = await fetch('/api/presets');
        PRESETS = await res.json();
    }

    function applyPreset(name) {
        if (!PRESETS || !PRESETS[name]) return;
        const p = PRESETS[name].params;
        for (const [k, v] of Object.entries(p)) {
            const el = document.getElementById(k);
            if (!el) continue;
            if (el.tagName === 'SELECT') {
                el.value = v;
            } else if (el.type === 'range') {
                el.value = v;
                const lbl = document.getElementById('v-' + k);
                if (lbl) {
                    const decs = (el.step && el.step.includes('.'))
                        ? el.step.split('.')[1].length : 0;
                    lbl.textContent = parseFloat(v).toFixed(decs);
                }
            }
        }
        updateCE();
        // mark active button
        document.querySelectorAll('.preset-buttons button').forEach(b => {
            b.classList.toggle('active', b.dataset.preset === name);
        });
        doPredict();
    }

    function bindPresets() {
        document.querySelectorAll('.preset-buttons button').forEach(btn => {
            btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
        });
        const reset = document.getElementById('reset-link');
        if (reset) reset.addEventListener('click', (e) => {
            e.preventDefault();
            applyPreset('typical');
        });
    }

    // ===================================================================
    // Boot
    // ===================================================================
    async function boot() {
        bindSliders();
        bindPresets();
        await loadPresets();
        applyPreset('typical');   // open on the "wow" preset
    }
    boot();
})();
