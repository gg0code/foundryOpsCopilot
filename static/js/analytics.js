// Foundry AI Demo — analytics dashboard
// Loads /api/analytics/* endpoints and renders Chart.js visualisations.

(() => {
    const inr = (n) => '₹' + Math.round(n).toLocaleString('en-IN');
    const inrShort = (n) => {
        if (n >= 1e7) return '₹' + (n / 1e7).toFixed(2) + ' Cr';
        if (n >= 1e5) return '₹' + (n / 1e5).toFixed(2) + ' L';
        return '₹' + Math.round(n).toLocaleString('en-IN');
    };

    const MONTH_NAMES = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

    const colors = {
        primary: '#0E4DA6',
        accent:  '#00B5A0',
        risk_high: '#DC2626',
        risk_mid:  '#F59E0B',
        risk_low:  '#16A34A',
        grid: '#E1E6EE',
        text: '#0F1A2E',
        soft: '#8290A5',
    };

    Chart.defaults.font.family = "'Inter', 'Segoe UI', sans-serif";
    Chart.defaults.color = colors.text;
    Chart.defaults.borderColor = colors.grid;

    // ========================================================
    // KPI strip
    // ========================================================
    async function loadOverview() {
        const r = await (await fetch('/api/analytics/overview')).json();
        document.getElementById('kpi-heats').textContent  = r.total_heats.toLocaleString('en-IN');
        document.getElementById('kpi-scrap').textContent  = r.scrap_rate_pct.toFixed(2) + '%';
        document.getElementById('kpi-rework').textContent = r.rework_rate_pct.toFixed(2) + '%';
        document.getElementById('kpi-savings').textContent = inrShort(r.annual_savings_target);
        document.getElementById('kpi-savings-lbl').textContent = r.annual_savings_label;

        // Waterfall
        const wf = ['scrap_cost','rework_cost','complaint_cost','warranty_cost'];
        const wfLabels = ['Scrap', 'Rework', 'OEM complaints', 'Warranty exposure'];
        new Chart(document.getElementById('chart-waterfall'), {
            type: 'bar',
            data: {
                labels: [...wfLabels, 'Total'],
                datasets: [{
                    label: 'Cost (INR)',
                    data: [...wf.map(k => r[k]), r.total_cost],
                    backgroundColor: [colors.risk_high, colors.risk_mid, '#EC4899', '#6B7280', colors.primary],
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false },
                           tooltip: { callbacks: { label: ctx => ' ' + inrShort(ctx.parsed.y) } } },
                scales: {
                    y: { grid: { color: colors.grid }, ticks: { callback: v => inrShort(v) } },
                    x: { grid: { display: false } },
                },
            },
        });
    }

    // ========================================================
    // Defect Pareto by cost
    // ========================================================
    async function loadPareto() {
        const data = await (await fetch('/api/analytics/pareto')).json();
        new Chart(document.getElementById('chart-pareto'), {
            type: 'bar',
            data: {
                labels: data.map(d => d.defect_class.replace(/_/g,' ')),
                datasets: [
                    { label: 'Cost', data: data.map(d => d.cost_inr),
                      backgroundColor: colors.primary, yAxisID: 'y' },
                    { label: 'Count', type: 'line', data: data.map(d => d.count),
                      borderColor: colors.accent, backgroundColor: colors.accent,
                      tension: 0.3, yAxisID: 'y1' },
                ],
            },
            options: {
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                plugins: { tooltip: { callbacks: { label: ctx =>
                    ctx.dataset.label === 'Cost' ? ' ' + inrShort(ctx.parsed.y)
                                                 : ' ' + ctx.parsed.y + ' heats' }}},
                scales: {
                    y: { grid: { color: colors.grid }, ticks: { callback: v => inrShort(v) } },
                    y1: { position: 'right', grid: { display: false }, ticks: { precision: 0 } },
                    x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 30, minRotation: 30 } },
                },
            },
        });
    }

    // ========================================================
    // Monthly trend
    // ========================================================
    async function loadMonthly() {
        const data = await (await fetch('/api/analytics/monthly')).json();
        new Chart(document.getElementById('chart-monthly'), {
            data: {
                labels: data.map(d => MONTH_NAMES[d.month]),
                datasets: [
                    { type: 'bar', label: 'Porosity rate (%)',
                      data: data.map(d => d.porosity_rate_pct),
                      backgroundColor: 'rgba(220,38,38,0.55)', borderColor: colors.risk_high,
                      borderWidth: 1, yAxisID: 'y' },
                    { type: 'bar', label: 'Scrap rate (%)',
                      data: data.map(d => d.scrap_rate_pct),
                      backgroundColor: 'rgba(14,77,166,0.5)', borderColor: colors.primary,
                      borderWidth: 1, yAxisID: 'y' },
                    { type: 'line', label: 'Avg humidity (%)',
                      data: data.map(d => d.avg_humidity_pct),
                      borderColor: colors.accent, backgroundColor: 'rgba(0,181,160,0.10)',
                      borderWidth: 2.5, tension: 0.35, fill: true, yAxisID: 'y1' },
                ],
            },
            options: {
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'bottom' } },
                scales: {
                    y:  { grid: { color: colors.grid }, title: { display: true, text: 'Rate (%)' } },
                    y1: { position: 'right', grid: { display: false }, min: 30, max: 100, title: { display: true, text: 'Humidity (%)' } },
                    x:  { grid: { display: false } },
                },
            },
        });
    }

    // ========================================================
    // Dimensional Cpk histograms
    // ========================================================
    async function loadDimensional() {
        const data = await (await fetch('/api/analytics/dimensional')).json();
        const cont = document.getElementById('cpk-charts');
        cont.innerHTML = '';
        for (const [label, info] of Object.entries(data)) {
            const card = document.createElement('div');
            const cpkClass = info.Cpk >= 1.33 ? 'pass' : 'fail';
            card.innerHTML = `
                <div style="margin-bottom: var(--sp-3); display:flex; justify-content:space-between; align-items:baseline;">
                    <strong>${label.replace(/_/g,' ')}</strong>
                    <span class="cpk-value ${cpkClass}">Cpk ${info.Cpk}</span>
                </div>
                <canvas id="cpk-${label}" height="160"></canvas>
                <div style="font-size: var(--fs-xs); color: var(--color-text-soft); margin-top: var(--sp-2);">
                    USL ${info.USL} · LSL ${info.LSL} · target ${info.target}
                </div>
            `;
            cont.appendChild(card);

            const centers = info.bins.slice(0, -1).map((b, i) => (b + info.bins[i+1]) / 2);
            new Chart(document.getElementById(`cpk-${label}`), {
                type: 'bar',
                data: {
                    labels: centers.map(c => c.toFixed(3)),
                    datasets: [{
                        label: 'count',
                        data: info.counts,
                        backgroundColor: cpkClass === 'pass' ? 'rgba(22,163,74,0.55)' : 'rgba(220,38,38,0.55)',
                        borderColor: cpkClass === 'pass' ? colors.risk_low : colors.risk_high,
                        borderWidth: 1, barPercentage: 1, categoryPercentage: 1,
                    }],
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        annotation: undefined,
                    },
                    scales: {
                        x: { ticks: { autoSkip: true, maxTicksLimit: 6 }, grid: { display: false } },
                        y: { grid: { color: colors.grid } },
                    },
                },
            });
        }
    }

    // ========================================================
    // Breakdowns (bars in HTML for compactness)
    // ========================================================
    async function loadBreakdowns() {
        const data = await (await fetch('/api/analytics/breakdowns')).json();

        function render(cont, rows) {
            const max = Math.max(...rows.map(r => r.defect_rate_pct), 1);
            cont.innerHTML = rows.map(r => `
                <div class="breakdown-bar-row">
                    <div class="lbl">${r.key}</div>
                    <div class="bar-bg"><div class="bar-fill" style="width:${(r.defect_rate_pct / max * 100).toFixed(1)}%"></div></div>
                    <div class="val">${r.defect_rate_pct.toFixed(2)}%</div>
                </div>`).join('');
        }
        render(document.getElementById('break-shift'),   data.shift);
        render(document.getElementById('break-furnace'), data.furnace);
        render(document.getElementById('break-season'),  data.season);
        render(document.getElementById('break-pattern'), data.pattern);
    }

    // ========================================================
    // Pattern wear curve
    // ========================================================
    async function loadWear() {
        const data = await (await fetch('/api/analytics/pattern_wear')).json();
        new Chart(document.getElementById('chart-wear'), {
            data: {
                labels: data.map(d => d.bucket),
                datasets: [
                    { type: 'line', label: 'Dim_NC rate (%)',
                      data: data.map(d => d.dim_nc_rate_pct),
                      borderColor: colors.risk_high, backgroundColor: 'rgba(220,38,38,0.15)',
                      tension: 0.3, fill: true, borderWidth: 3, pointRadius: 5 },
                    { type: 'line', label: 'Any defect rate (%)',
                      data: data.map(d => d.any_defect_rate_pct),
                      borderColor: colors.primary, backgroundColor: 'transparent',
                      tension: 0.3, borderWidth: 2, pointRadius: 3, borderDash: [4, 4] },
                ],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom' } },
                scales: {
                    y: { grid: { color: colors.grid }, title: { display: true, text: 'Defect rate (%)' } },
                    x: { grid: { display: false }, title: { display: true, text: 'Pattern age (cycles)' } },
                },
            },
        });
    }

    // ========================================================
    // Correlations
    // ========================================================
    async function loadCorrelations() {
        const data = await (await fetch('/api/analytics/correlations')).json();
        new Chart(document.getElementById('chart-correlations'), {
            type: 'bar',
            data: {
                labels: data.map(d => d.feature.replace(/_/g,' ')),
                datasets: [{
                    label: 'Pearson r',
                    data: data.map(d => d.correlation),
                    backgroundColor: data.map(d => d.correlation > 0 ? 'rgba(220,38,38,0.7)' : 'rgba(22,163,74,0.7)'),
                }],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: colors.grid }, min: -0.4, max: 0.4 },
                    y: { grid: { display: false } },
                },
            },
        });
    }

    // ========================================================
    // Model performance
    // ========================================================
    async function loadModelInfo() {
        const r = await (await fetch('/api/model_info')).json();
        const m = r.metrics;
        const cont = document.getElementById('model-metrics');
        const rows = [
            { name: 'Defect Classifier (XGBoost · 10 classes)', val: (m.defect_classifier.accuracy * 100).toFixed(1) + '%',
              target: 'target ≥ 82%' },
            { name: 'Severity Classifier (XGBoost · 4 classes)', val: (m.severity_classifier.accuracy * 100).toFixed(1) + '%',
              target: 'target ≥ 78%' },
            { name: 'Yield Regressor (LightGBM)', val: 'R² ' + m.yield_regressor.r2.toFixed(3),
              target: 'target ≥ 0.75' },
            { name: 'Warranty Risk Regressor (LightGBM)', val: 'R² ' + m.warranty_regressor.r2.toFixed(3),
              target: 'target ≥ 0.70' },
            { name: 'Training rows', val: m.n_train_rows.toLocaleString(), target: '20% holdout' },
            { name: 'Features used', val: m.n_features, target: 'one-hot + numeric' },
        ];
        cont.innerHTML = rows.map(r => `
            <div class="model-metric-row">
                <div class="metric-name">${r.name}</div>
                <div class="metric-val">${r.val}</div>
                <div class="metric-target">${r.target}</div>
            </div>`).join('');
    }

    // ========================================================
    // Sample data
    // ========================================================
    async function loadSample() {
        const rows = await (await fetch('/api/analytics/sample?n=25')).json();
        const tbl = document.getElementById('sample-table');
        const headers = Object.keys(rows[0] || {});
        tbl.innerHTML = `
            <thead><tr>${headers.map(h => `<th>${h.replace(/_/g, ' ')}</th>`).join('')}</tr></thead>
            <tbody>${rows.map(r => `<tr>${headers.map(h => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
    }

    // ========================================================
    // Boot
    // ========================================================
    Promise.all([
        loadOverview(), loadPareto(), loadMonthly(), loadDimensional(),
        loadBreakdowns(), loadWear(), loadCorrelations(), loadModelInfo(), loadSample(),
    ]).catch(e => console.error('Analytics load error', e));
})();
