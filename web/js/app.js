/* ───────────────────────────────────────────────────────────
   Secure V2V Communication — Dashboard JS
   ─────────────────────────────────────────────────────────── */

// ── Chart.js global defaults ────────────────────────────────
Chart.defaults.color           = '#64748b';
Chart.defaults.borderColor     = '#1f2d44';
Chart.defaults.font.family     = "'DM Sans', sans-serif";
Chart.defaults.plugins.legend.labels.usePointStyle = true;

const COLORS = {
  accent : '#00d4ff',
  purple : '#7c3aed',
  green  : '#10b981',
  orange : '#f59e0b',
  red    : '#ef4444',
  muted  : '#334155',
};

// ── Live clock ───────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('live-time');
  if (el) el.textContent = new Date().toUTCString().replace(' GMT', ' UTC');
}
setInterval(updateClock, 1000);
updateClock();

// ── Tab navigation ───────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Load data ────────────────────────────────────────────────
async function loadData() {
  // Try fetching from the server; fall back to inline demo data
  try {
    const res = await fetch('data/results.json');
    if (!res.ok) throw new Error('fetch failed');
    return await res.json();
  } catch {
    return DEMO_DATA;
  }
}

async function loadCSV() {
  try {
    const res = await fetch('data/simulation_sample.csv');
    if (!res.ok) throw new Error();
    return await res.text();
  } catch { return null; }
}

// ── Helpers ───────────────────────────────────────────────────
const fmt = (v, d = 2) => Number(v).toFixed(d);

function barConfig(labels, datasets, opts = {}) {
  return {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top' }, tooltip: { mode: 'index' } },
      scales: {
        y: { grid: { color: '#1f2d44' }, ticks: { font: { size: 11 } }, ...opts.y },
        x: { grid: { color: '#1f2d44' }, ticks: { font: { size: 11 } } },
      },
      ...opts,
    },
  };
}

// ── Render ────────────────────────────────────────────────────
async function init() {
  const data = await loadData();
  const { meta, class_distribution: dist, scenario_stats: scenarios,
          model_results: models } = data;

  renderMeta(meta);
  renderKPIs(models, dist);
  renderDistChart(dist);
  renderScenarioChart(scenarios);
  renderModelTable(models);
  renderModelCharts(models);
  renderMetricsTab(models, scenarios);
  renderConfusionMatrices(models);

  const csv = await loadCSV();
  if (csv) renderSimTable(csv);
}

// ── Meta info ─────────────────────────────────────────────────
function renderMeta(meta) {
  document.getElementById('meta-project').textContent    = meta.project;
  document.getElementById('meta-supervisor').textContent = meta.supervisor;
  document.getElementById('meta-generated').textContent  = meta.generated_at.replace('T',' ').replace('Z',' UTC');
  const ul = document.getElementById('meta-members');
  meta.members.forEach(m => {
    const li = document.createElement('li'); li.textContent = m; ul.appendChild(li);
  });
}

// ── KPIs ──────────────────────────────────────────────────────
function renderKPIs(models, dist) {
  const best     = models.reduce((a, b) => a.accuracy > b.accuracy ? a : b);
  const bestF1   = models.reduce((a, b) => a.f1 > b.f1 ? a : b);
  const fastest  = models.reduce((a, b) => a.latency_ms < b.latency_ms ? a : b);
  const totalSamples = dist.reduce((s, d) => s + d.count, 0);

  const kpis = [
    { label:'Best Accuracy',    value: fmt(best.accuracy)+'%',   sub: best.model },
    { label:'Best F1 Score',    value: fmt(bestF1.f1)+'%',       sub: bestF1.model },
    { label:'Fastest Inference',value: fastest.latency_ms+' ms', sub: fastest.model },
    { label:'Total Samples',    value: totalSamples.toLocaleString(), sub: 'training + test' },
   //{ label:'Models Trained',   value: models.length,            sub: 'IF · XGB · LSTM' },
    { label:'Driving Classes',  value: dist.length,              sub: 'Normal / Accident / Suspicious' },
  ];

  const row = document.getElementById('kpi-row');
  kpis.forEach(k => {
    row.innerHTML += `
      <div class="kpi-card">
        <div class="kpi-label">${k.label}</div>
        <div class="kpi-value">${k.value}</div>
        <div class="kpi-sub">${k.sub}</div>
      </div>`;
  });
}

// ── Class Distribution ─────────────────────────────────────────
function renderDistChart(dist) {
  new Chart(document.getElementById('chartDist'), {
    type: 'doughnut',
    data: {
      labels: dist.map(d => d.label),
      datasets: [{
        data: dist.map(d => d.count),
        backgroundColor: [COLORS.accent, COLORS.red, COLORS.orange],
        borderWidth: 2, borderColor: '#111827',
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'right' },
        tooltip: { callbacks: {
          label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString()} records`
        }},
      },
    },
  });
}

// ── Scenario Chart ─────────────────────────────────────────────
function renderScenarioChart(scenarios) {
  const labels  = scenarios.map(s => s.scenario);
  const metrics = [
    { key:'avg_speed',   label:'Avg Speed (km/h)', color: COLORS.accent },
    { key:'avg_accel',   label:'Avg Accel (m/s²)',  color: COLORS.orange },
    { key:'avg_proximity', label:'Avg Proximity (m)', color: COLORS.green },
  ];

  new Chart(document.getElementById('chartScenario'), barConfig(
    labels,
    metrics.map(m => ({
      label: m.label,
      data: scenarios.map(s => Math.abs(s[m.key])),
      backgroundColor: m.color + '99',
      borderColor: m.color,
      borderWidth: 1.5,
      borderRadius: 4,
    }))
  ));
}

// ── Model Table ────────────────────────────────────────────────
const MODEL_TYPES = {
  'Isolation Forest': 'Unsupervised / Anomaly',
  'XGBoost':          'Supervised / Classification',
 // 'LSTM':             'Deep Learning / Sequential',
};

function renderModelTable(models) {
  const tbody = document.getElementById('model-tbody');
  models.forEach(m => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td style="color:var(--text);font-weight:600">${m.model}</td>
      <td>${MODEL_TYPES[m.model] || '—'}</td>
      <td style="color:var(--accent3)">${fmt(m.accuracy)}</td>
      <td>${fmt(m.precision)}</td>
      <td>${fmt(m.recall)}</td>
      <td style="color:var(--accent)">${fmt(m.f1)}</td>
      <td style="color:var(--warn)">${m.latency_ms}</td>`;
    tbody.appendChild(row);
  });
}

// ── Model Charts ───────────────────────────────────────────────
function renderModelCharts(models) {
  const labels = models.map(m => m.model);

  // All metrics grouped bar
  new Chart(document.getElementById('chartModelMetrics'), barConfig(
    labels,
    [
      { label:'Accuracy',  data: models.map(m => m.accuracy),  backgroundColor: COLORS.accent+'99',  borderColor: COLORS.accent,  borderWidth:1.5, borderRadius:4 },
      { label:'Precision', data: models.map(m => m.precision), backgroundColor: COLORS.purple+'99', borderColor: COLORS.purple, borderWidth:1.5, borderRadius:4 },
      { label:'Recall',    data: models.map(m => m.recall),    backgroundColor: COLORS.green+'99',  borderColor: COLORS.green,  borderWidth:1.5, borderRadius:4 },
      { label:'F1 Score',  data: models.map(m => m.f1),        backgroundColor: COLORS.orange+'99', borderColor: COLORS.orange, borderWidth:1.5, borderRadius:4 },
    ],
    { y: { min: 80, max: 105 } }
  ));

  // Latency bar
  new Chart(document.getElementById('chartLatency'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Latency (ms/sample)',
        data: models.map(m => m.latency_ms),
        backgroundColor: [COLORS.accent+'99', COLORS.green+'99', COLORS.orange+'99'],
        borderColor:     [COLORS.accent,       COLORS.green,       COLORS.orange],
        borderWidth: 1.5, borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: '#1f2d44' }, title: { display:true, text:'ms per sample' } },
        x: { grid: { color: '#1f2d44' } },
      },
    },
  });
}

// ── Metrics Tab ────────────────────────────────────────────────
function renderMetricsTab(models, scenarios) {
  const labels = models.map(m => m.model);

  // Accuracy
  new Chart(document.getElementById('chartAccuracy'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Accuracy (%)',
        data: models.map(m => m.accuracy),
        backgroundColor: [COLORS.accent+'bb', COLORS.purple+'bb', COLORS.green+'bb'],
        borderColor:     [COLORS.accent,       COLORS.purple,       COLORS.green],
        borderWidth: 1.5, borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display:false } },
      scales: {
        y: { min:80, max:102, grid: { color:'#1f2d44' },
             ticks: { callback: v => v+'%' } },
        x: { grid: { color:'#1f2d44' } },
      },
    },
  });

  // F1
  new Chart(document.getElementById('chartF1'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'F1 Score (%)',
        data: models.map(m => m.f1),
        backgroundColor: [COLORS.orange+'bb', COLORS.red+'bb', COLORS.accent+'bb'],
        borderColor:     [COLORS.orange,       COLORS.red,       COLORS.accent],
        borderWidth: 1.5, borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display:false } },
      scales: {
        y: { min:80, max:102, grid: { color:'#1f2d44' },
             ticks: { callback: v => v+'%' } },
        x: { grid: { color:'#1f2d44' } },
      },
    },
  });

  // Precision vs Recall
  new Chart(document.getElementById('chartPR'), {
    type: 'radar',
    data: {
      labels: models.map(m => m.model),
      datasets: [
        {
          label: 'Precision',
          data: models.map(m => m.precision),
          borderColor: COLORS.accent, backgroundColor: COLORS.accent+'22',
          pointBackgroundColor: COLORS.accent, borderWidth: 2,
        },
        {
          label: 'Recall',
          data: models.map(m => m.recall),
          borderColor: COLORS.orange, backgroundColor: COLORS.orange+'22',
          pointBackgroundColor: COLORS.orange, borderWidth: 2,
        },
        {
          label: 'F1 Score',
          data: models.map(m => m.f1),
          borderColor: COLORS.green, backgroundColor: COLORS.green+'22',
          pointBackgroundColor: COLORS.green, borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 80, max: 102,
          grid: { color: '#1f2d44' },
          angleLines: { color: '#1f2d44' },
          ticks: { color: '#64748b', font: { size: 10 }, backdropColor: 'transparent' },
          pointLabels: { color: '#e2e8f0', font: { size: 12 } },
        },
      },
    },
  });

  // Scenario full stats
  const features = ['avg_speed','avg_accel','avg_braking','avg_proximity'];
  const featLabels = ['Avg Speed (km/h)','Avg Accel (m/s²)','Avg Braking','Avg Proximity (m)'];
  const scenColors = [COLORS.accent, COLORS.red, COLORS.orange];

  new Chart(document.getElementById('chartScenarioFull'), barConfig(
    featLabels,
    scenarios.map((s, i) => ({
      label: s.scenario,
      data: features.map(f => Math.abs(s[f])),
      backgroundColor: scenColors[i]+'99',
      borderColor: scenColors[i],
      borderWidth: 1.5, borderRadius: 4,
    }))
  ));

  // Scenario table
  const tbody = document.getElementById('scenario-tbody');
  scenarios.forEach(s => {
    tbody.innerHTML += `
      <tr>
        <td style="color:var(--text);font-weight:600">${s.scenario}</td>
        <td>${fmt(s.avg_speed)}</td>
        <td>${fmt(s.avg_accel)}</td>
        <td>${fmt(s.avg_braking, 3)}</td>
        <td>${fmt(s.avg_proximity)}</td>
      </tr>`;
  });
}

// ── Confusion Matrices ─────────────────────────────────────────
function renderConfusionMatrices(models) {
  const grid = document.getElementById('cm-grid');

  models.forEach(m => {
    const cm     = m.confusion_matrix;
    const labels = m.cm_labels;
    const maxVal = Math.max(...cm.flat());

    let rows = '';

    // Header row
    rows += '<tr><th style="color:var(--text-muted);font-size:.65rem">Pred ↓ / True →</th>';
    labels.forEach(l => { rows += `<th>${l}</th>`; });
    rows += '</tr>';

    // Data rows
    cm.forEach((row, i) => {
      rows += `<tr><th style="text-align:left;padding:.5rem .7rem;font-size:.7rem;color:var(--text-muted)">${labels[i]}</th>`;
      row.forEach((val, j) => {
        const ratio = val / maxVal;
        let cls = ratio > 0.5 ? 'cm-cell-high' : ratio > 0.1 ? 'cm-cell-mid' : 'cm-cell-low';
        // For correct predictions (diagonal) force high; else low
        if (i !== j && val > 0) cls = 'cm-cell-low';
        rows += `<td class="${cls}">${val.toLocaleString()}</td>`;
      });
      rows += '</tr>';
    });

    const div = document.createElement('div');
    div.className = 'cm-wrapper';
    div.innerHTML = `
      <div class="cm-model-name">${m.model}</div>
      <table class="cm-table">${rows}</table>
      <div class="cm-legend">
        <span><div class="dot dot-high"></div> Correct predictions</span>
        <span><div class="dot dot-low"></div> Misclassifications</span>
      </div>`;
    grid.appendChild(div);
  });
}

// ── Simulation CSV Table ───────────────────────────────────────
function renderSimTable(csv) {
  const lines   = csv.trim().split('\n');
  const headers = lines[0].split(',');
  const rows    = lines.slice(1, 201);    // first 200 rows

  const labelMap = { '0': 'Normal', '1': 'Accident', '2': 'Suspicious' };
  const labelColors = { '0': '#10b981', '1': '#ef4444', '2': '#f59e0b' };

  const thead = document.getElementById('sim-thead');
  thead.innerHTML = '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';

  const tbody = document.getElementById('sim-tbody');
  rows.forEach(line => {
    const cols = line.split(',');
    const labelIdx = headers.indexOf('label');
    let cells = cols.map((c, i) => {
      if (i === labelIdx) {
        const lbl = labelMap[c.trim()] || c;
        const clr = labelColors[c.trim()] || '#fff';
        return `<td style="color:${clr};font-weight:700">${lbl}</td>`;
      }
      return `<td>${parseFloat(c) % 1 === 0 ? c : Number(c).toFixed(3)}</td>`;
    }).join('');
    tbody.innerHTML += `<tr>${cells}</tr>`;
  });
}

// ── Demo data (fallback when no server) ───────────────────────
const DEMO_DATA = {
  meta: {
    project: "Secure Vehicle-to-Vehicle (V2V) Communication using AI",
    group_id: "BSIT-F25-005",
    supervisor: "Zunnurain Hussain",
    members: [
      "Abdul Hanan Sabir (03-135222-001)",
      "Dania Rasool (03-135222-012)",
      "Abdul Wahab (03-135232-004)"
    ],
    generated_at: "2026-05-04T11:30:09Z"
  },
  class_distribution: [
    { label:"Normal",     count:9000 },
    { label:"Accident",   count:3000 },
    { label:"Suspicious", count:3000 },
  ],
  scenario_stats: [
    { scenario:"Normal",     avg_speed:59.99, avg_accel:0.01,  avg_braking:0.108, avg_proximity:30.03 },
    { scenario:"Accident",   avg_speed:20.89, avg_accel:-7.96, avg_braking:0.899, avg_proximity:5.09  },
    { scenario:"Suspicious", avg_speed:110.19,avg_accel:4.03,  avg_braking:0.054, avg_proximity:7.97  },
  ],
  model_results: [
    {
      model:"Isolation Forest", accuracy:93.7, precision:92.23, recall:92.0, f1:92.12, latency_ms:0.107,
      confusion_matrix:[[1707,93],[96,1104]], cm_labels:["Normal","Anomaly"]
    },
    {
      model:"XGBoost", accuracy:99.97, precision:99.97, recall:99.97, f1:99.97, latency_ms:0.154,
      confusion_matrix:[[1799,0,1],[0,600,0],[0,0,600]], cm_labels:["Normal","Accident","Suspicious"]
    },
    {
      //model:"LSTM", accuracy:100.0, precision:100.0, recall:100.0, f1:100.0, latency_ms:20.423,
      confusion_matrix:[[183,0,0],[0,64,0],[0,0,53]], cm_labels:["Normal","Accident","Suspicious"]
    },
  ],
};

// ── Bootstrap ─────────────────────────────────────────────────
init();
