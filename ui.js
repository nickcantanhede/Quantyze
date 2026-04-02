/**
 * Quantyze Web UI — frontend logic
 *
 * Talks exclusively to the Flask backend served from main.py.
 * No business logic here — all decisions stay in the Python backend.
 */

'use strict';

/* ─── constants ─────────────────────────────────────────────────────────── */
const POLL_MS   = 600;   // polling interval while a job runs
const LOG_LINES = 400;   // max lines kept in log display

/* ─── state ─────────────────────────────────────────────────────────────── */
let simPollId    = null;
let trainPollId  = null;
let depthChart   = null;
let fillsChart   = null;
let appConfig    = null;

/* ─── DOM helpers ────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const show   = id => $(id) && $(id).classList.remove('hidden');
const hide   = id => $(id) && $(id).classList.add('hidden');
const setText = (id, value) => {
  const el = $(id);
  if (el) el.textContent = String(value);
};

/* ─── fetch helpers ──────────────────────────────────────────────────────── */
async function apiGet(url) {
  const r = await fetch(url);
  return r.json();
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { ok: r.ok, status: r.status, data: await r.json() };
}

/* ─── number formatting ──────────────────────────────────────────────────── */
function fmt2(v)  { return (v == null) ? '—' : Number(v).toFixed(2); }
function fmtN(v)  { return (v == null) ? '—' : String(v); }
function fmtPct(v){ return (v == null) ? '' : Number(v).toFixed(1) + '%'; }
function fmtKB(bytes) {
  if (bytes == null) return '—';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function sumCounts(values) {
  if (!Array.isArray(values)) return null;
  return values.reduce((total, value) => total + Number(value || 0), 0);
}

function overlayModeText(mode) {
  const labels = { baseline: 'Baseline', latest: 'Latest', none: 'None' };
  return labels[mode] || String(mode || '—');
}

function latestModelPath() {
  return appConfig?.training?.output_files?.latest_model || 'latest_model.pt';
}

function latestMetricsPath() {
  return appConfig?.training?.output_files?.latest_metrics || 'latest_training_metrics.json';
}

function applyDatasetPresetButtons(presets) {
  const presetMap = Object.fromEntries((presets || []).map(preset => [preset.id, preset]));
  document.querySelectorAll('.preset-btn[data-preset]').forEach(btn => {
    const preset = presetMap[btn.dataset.preset];
    if (!preset) return;
    btn.textContent = preset.button_label || preset.label || btn.textContent;
    btn.dataset.path = preset.path || '';
  });
}

function applyTrainingPresetLabels(presets) {
  const presetMap = Object.fromEntries((presets || []).map(preset => [preset.id, preset]));
  for (const presetId of ['sample', 'huge', 'lobster']) {
    const preset = presetMap[presetId];
    if (!preset) continue;
    setText(`train-src-${presetId}-label`, preset.label || presetId);
    setText(`train-src-${presetId}-meta`, `— ${preset.meta || preset.path || ''}`);
  }
}

function applyTrainingSummaryConfig(training) {
  if (!training) return;
  setText('training-feature-dim', training.feature_dim ?? 16);
  setText('training-label-horizon', training.label_horizon_events ?? 50);
  setText('training-train-split', training.train_split_percent ?? 80);
  setText('training-val-split', training.val_split_percent ?? 20);
  setText('training-split-seed', training.split_seed ?? 111);
  setText('training-epochs', training.epochs ?? 50);
  setText('training-optimizer', training.optimizer || 'Adam');
  setText('training-learning-rate', training.learning_rate ?? '3e-4');
  setText('training-latest-model-path', latestModelPath());
  setText('training-latest-metrics-path', latestMetricsPath());
}

function applyAppConfig(config) {
  if (!config) return;
  applyDatasetPresetButtons(config.simulation?.csv_presets);
  applyTrainingPresetLabels(config.training?.presets);
  applyTrainingSummaryConfig(config.training);
}

async function loadAppConfig() {
  try {
    appConfig = await apiGet('/api/config');
    applyAppConfig(appConfig);
  } catch (_) {
    appConfig = null;
  }
}

/* ─── tab switching ──────────────────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', String(b === btn));
      });
      document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.toggle('active', p.id === 'tab-' + tab);
      });
      if (tab === 'artifacts') loadArtifacts();
      if (tab === 'charts')    loadCharts();
    });
  });
}

/* ─── source radio toggle (simulate tab) ─────────────────────────────────── */
function initSourceToggle() {
  document.querySelectorAll('[name="sim-source"]').forEach(r => {
    r.addEventListener('change', () => {
      const isCsv = document.querySelector('[name="sim-source"]:checked')?.value === 'csv';
      isCsv ? show('csv-opts') : hide('csv-opts');
      isCsv ? hide('synthetic-opts') : show('synthetic-opts');
    });
  });
  // preset path buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const inp = $('data-path-inp');
      if (inp) inp.value = btn.dataset.path;
    });
  });
}

/* ─── custom path toggle (train tab) ─────────────────────────────────────── */
function initTrainToggle() {
  document.querySelectorAll('[name="train-src"]').forEach(r => {
    r.addEventListener('change', () => {
      const isCustom = document.querySelector('[name="train-src"]:checked')?.value === 'custom';
      isCustom ? show('train-path-inp') : hide('train-path-inp');
    });
  });
}

/* ─── status badge ───────────────────────────────────────────────────────── */
function setStatus(state) {
  const badge = $('status-badge');
  if (!badge) return;
  badge.className = 'status-badge ' + state;
  const labels = { idle: '● Idle', running: '● Running', done: '● Done', error: '● Error' };
  badge.textContent = labels[state] || '● ' + state;
}

/* ─── load overlay note on page load ─────────────────────────────────────── */
async function loadOverlayNote() {
  try {
    const st = await apiGet('/api/model-status');
    const note = $('overlay-note');
    if (!note) return;
    if (st.note) {
      note.textContent = st.note;
    } else {
      note.textContent = st.checkpoint_exists
        ? `Checkpoint: ${st.dataset_label || st.mode}`
        : 'No checkpoint found for selected mode.';
    }
  } catch (_) { /* non-fatal */ }
}

/* ══════════════════════════════════════════════════════════════════════════
   SIMULATE
══════════════════════════════════════════════════════════════════════════ */

async function runSimulation() {
  const source   = document.querySelector('[name="sim-source"]:checked')?.value || 'synthetic';
  const scenario = $('scenario-sel')?.value || 'balanced';
  const dataPath = $('data-path-inp')?.value?.trim() || '';
  const speed    = parseFloat($('speed-inp')?.value || '0') || 0;
  const overlay  = document.querySelector('[name="sim-overlay"]:checked')?.value || 'baseline';

  // Build request body
  const body = { source, speed, model: overlay };
  if (source === 'synthetic') {
    body.scenario = scenario;
  } else {
    if (!dataPath) { alert('Enter a CSV path or choose a preset.'); return; }
    body.data = dataPath;
  }

  // Show output panel, clear previous
  show('sim-output-panel');
  show('sim-prog-wrap');
  hide('sim-results-panel');
  $('sim-log').textContent = '';
  $('sim-pct').textContent = '';
  $('sim-prog-fill').style.width = '0%';
  setStatus('running');

  $('sim-run-btn').disabled = true;
  $('sim-run-btn').textContent = '⏳ Running…';

  const res = await apiPost('/api/simulate', body);
  if (!res.ok) {
    renderSimError(res.data?.error || 'Failed to start simulation.');
    $('sim-run-btn').disabled = false;
    $('sim-run-btn').textContent = '▶ Run Simulation';
    setStatus('error');
    return;
  }

  // Start polling
  simPollId = setInterval(pollSimStatus, POLL_MS);
}

async function pollSimStatus() {
  let st;
  try { st = await apiGet('/api/simulate/status'); }
  catch (_) { return; }

  updateSimLog(st.log || []);

  if (st.state === 'running') {
    const pct = Math.round(st.progress || 0);
    $('sim-prog-fill').style.width = pct + '%';
    $('sim-pct').textContent = fmtPct(st.progress);
    return;
  }

  // Terminal state
  clearInterval(simPollId);
  simPollId = null;

  $('sim-prog-fill').style.width = '100%';
  $('sim-pct').textContent = '';
  $('sim-run-btn').disabled = false;
  $('sim-run-btn').textContent = '▶ Run Simulation';

  if (st.state === 'done') {
    setStatus('done');
    hide('sim-prog-wrap');
    renderSimResults(st.results);
    // Enable charts tab
    enableCharts();
  } else if (st.state === 'error') {
    setStatus('error');
    renderSimError(st.error || 'Unknown error');
  }
}

function updateSimLog(lines) {
  const box = $('sim-log');
  if (!box) return;
  const trimmed = lines.slice(-LOG_LINES);
  box.textContent = trimmed.join('\n');
  box.scrollTop = box.scrollHeight;
}

function renderSimError(msg) {
  const panel = $('sim-results-panel');
  if (!panel) return;
  panel.innerHTML = `<div class="alert alert-err">Error: ${escHtml(msg)}</div>`;
  show('sim-results-panel');
}

function renderSimResults(r) {
  if (!r) return;
  const panel = $('sim-results-panel');
  if (!panel) return;

  const agentSection = r.agent_pnl != null ? `
    <div class="section-label">Agent Overlay</div>
    <div class="results-grid" style="grid-template-columns: repeat(2, 1fr)">
      ${metricCard('Agent P&L', fmt2(r.agent_pnl), pnlClass(r.agent_pnl))}
      ${metricCard('Overlay Mode', overlayModeText(r.overlay_mode))}
    </div>` : '';

  panel.innerHTML = `
    <div class="card">
      <div class="card-hd">Simulation Results — ${escHtml(r.source_label || '')}</div>

      <div class="section-label">Simulation Metrics</div>
      <div class="results-grid">
        ${metricCard('Total Filled',    fmt2(r.total_filled))}
        ${metricCard('Fill Count',      fmtN(r.fill_count))}
        ${metricCard('Cancel Count',    fmtN(r.cancel_count))}
        ${metricCard('Avg Slippage',    fmt2(r.average_slippage))}
        ${metricCard('Spread',          r.spread != null ? fmt2(r.spread) : 'N/A')}
        ${metricCard('Mid Price',       r.mid_price != null ? fmt2(r.mid_price) : 'N/A')}
      </div>
      ${agentSection}
    </div>`;
  show('sim-results-panel');
}

function metricCard(label, value, cls = '') {
  return `<div class="metric-card">
    <div class="metric-label">${escHtml(label)}</div>
    <div class="metric-value ${cls}">${escHtml(String(value))}</div>
  </div>`;
}

function pnlClass(v) {
  if (v == null) return '';
  return v > 0 ? 'positive' : v < 0 ? 'negative' : '';
}

/* ══════════════════════════════════════════════════════════════════════════
   TRAIN
══════════════════════════════════════════════════════════════════════════ */

async function runTraining() {
  const source = document.querySelector('[name="train-src"]:checked')?.value || 'sample';
  const customPath = $('train-path-inp')?.value?.trim() || '';

  if (source === 'custom' && !customPath) {
    alert('Enter a CSV path for custom training.');
    return;
  }

  const body = { source };
  if (source === 'custom') body.data_path = customPath;

  show('train-output-panel');
  show('train-prog-wrap');
  hide('train-results-panel');
  $('train-log').textContent = '';
  $('train-pct').textContent = 'Training…';
  setStatus('running');

  $('train-run-btn').disabled = true;
  $('train-run-btn').textContent = '⏳ Training…';

  const res = await apiPost('/api/train', body);
  if (!res.ok) {
    renderTrainError(res.data?.error || 'Failed to start training.');
    $('train-run-btn').disabled = false;
    $('train-run-btn').textContent = '▶ Start Training';
    setStatus('error');
    return;
  }

  trainPollId = setInterval(pollTrainStatus, POLL_MS);
}

async function pollTrainStatus() {
  let st;
  try { st = await apiGet('/api/train/status'); }
  catch (_) { return; }

  updateTrainLog(st.log || []);

  if (st.state === 'running') return;

  clearInterval(trainPollId);
  trainPollId = null;
  $('train-pct').textContent = '';
  hide('train-prog-wrap');
  $('train-run-btn').disabled = false;
  $('train-run-btn').textContent = '▶ Start Training';

  if (st.state === 'done') {
    setStatus('done');
    renderTrainResults(st.results);
  } else if (st.state === 'error') {
    setStatus('error');
    renderTrainError(st.error || 'Unknown error');
  }
}

function updateTrainLog(lines) {
  const box = $('train-log');
  if (!box) return;
  box.textContent = lines.slice(-LOG_LINES).join('\n');
  box.scrollTop = box.scrollHeight;
}

function renderTrainError(msg) {
  const panel = $('train-results-panel');
  if (!panel) return;
  panel.innerHTML = `<div class="alert alert-err">Error: ${escHtml(msg)}</div>`;
  show('train-results-panel');
}

function renderTrainResults(r) {
  if (!r) return;
  const panel = $('train-results-panel');
  if (!panel) return;

  const acc    = r.val_accuracy != null ? (r.val_accuracy * 100).toFixed(2) + '%' : '—';
  const base   = r.majority_baseline_accuracy != null ? (r.majority_baseline_accuracy * 100).toFixed(2) + '%' : '—';
  const names  = r.class_names || ['buy', 'sell', 'hold'];
  const recall = r.per_class_recall || [];
  const cm     = r.confusion_matrix || [];

  // Recall table
  const recallRows = names.map((n, i) =>
    `<tr><td>${escHtml(n)}</td><td>${recall[i] != null ? (recall[i] * 100).toFixed(1) + '%' : '—'}</td></tr>`
  ).join('');

  // Confusion matrix
  const cmHeader = '<tr><th>Actual ↓ / Pred →</th>' + names.map(n => `<th>${escHtml(n)}</th>`).join('') + '</tr>';
  const cmRows = cm.map((row, i) =>
    '<tr><th>' + escHtml(names[i] || String(i)) + '</th>' +
    row.map((v, j) => `<td class="${i === j ? 'diag' : ''}">${v}</td>`).join('') +
    '</tr>'
  ).join('');

  panel.innerHTML = `
    <div class="card">
      <div class="card-hd">Training Complete</div>

      <div class="section-label">Classifier Accuracy</div>
      <div class="results-grid" style="grid-template-columns: repeat(2, 1fr); margin-bottom:var(--s3)">
        ${metricCard('Validation Accuracy', acc, acc !== '—' && parseFloat(acc) > parseFloat(base) ? 'positive' : '')}
        ${metricCard('Majority Baseline', base)}
      </div>

      <div class="section-label">Per-Class Recall</div>
      <table class="cm-table" style="margin-bottom:var(--s3); width:auto">
        <thead><tr><th>Class</th><th>Recall</th></tr></thead>
        <tbody>${recallRows}</tbody>
      </table>

      <div class="section-label">Confusion Matrix</div>
      <div class="table-scroll">
        <table class="cm-table">
          <thead>${cmHeader}</thead>
          <tbody>${cmRows}</tbody>
        </table>
      </div>

      <div class="alert alert-ok" style="margin-top:var(--s3)">
        Checkpoint saved → <code>${escHtml(latestModelPath())}</code>
      </div>
    </div>`;
  show('train-results-panel');
}

/* ══════════════════════════════════════════════════════════════════════════
   ARTIFACTS
══════════════════════════════════════════════════════════════════════════ */

async function loadArtifacts() {
  const body = $('artifacts-body');
  if (!body) return;
  body.innerHTML = '<div class="loading-msg">Loading…</div>';

  try {
    const [art, baseline, latest, logSum] = await Promise.all([
      apiGet('/api/artifacts'),
      fetch('/api/metrics/baseline').then(r => r.ok ? r.json() : null),
      fetch('/api/metrics/latest').then(r => r.ok ? r.json() : null),
      apiGet('/api/log-summary'),
    ]);

    body.innerHTML = renderArtifacts(art, baseline, latest, logSum);
  } catch (e) {
    body.innerHTML = `<div class="alert alert-err">Failed to load artifacts: ${escHtml(String(e))}</div>`;
  }
}

function renderArtifacts(art, baseline, latest, logSum) {
  const am = art.active_model || {};
  const files = art.files || {};

  // Active model section
  const modeColor = { baseline: 'pill-mode', latest: 'pill-ok', none: 'pill-missing' };
  const overlayHtml = `
    <div class="card">
      <div class="card-hd">Active Model Overlay</div>
      <div class="overlay-info-row">
        <span class="key">Current mode</span>
        <span class="val"><span class="pill ${modeColor[am.mode] || 'pill-mode'}">${escHtml(overlayModeText(am.mode))}</span></span>
      </div>
      <div class="overlay-info-row">
        <span class="key">Dataset label</span>
        <span class="val">${escHtml(am.dataset_label || '—')}</span>
      </div>
      <div class="overlay-info-row">
        <span class="key">Checkpoint exists</span>
        <span class="val">${am.checkpoint_exists
          ? '<span class="pill pill-ok">yes</span>'
          : '<span class="pill pill-missing">no</span>'}</span>
      </div>
      ${am.note ? `<div class="alert alert-info" style="margin-top:var(--s2);font-size:0.8rem">${escHtml(am.note)}</div>` : ''}
    </div>`;

  // File status section
  const fileRows = Object.entries(files).map(([key, info]) => {
    const name = key.replace(/_/g, ' ');
    const pill = info.exists
      ? `<span class="pill pill-ok">✓</span>`
      : `<span class="pill pill-missing">missing</span>`;
    const size = info.exists ? fmtKB(info.size) : '';
    return `<div class="artifact-row">
      <span class="artifact-name">${escHtml(info.path || key)}</span>
      <span style="display:flex;align-items:center;gap:8px">
        <span class="artifact-size">${escHtml(size)}</span>${pill}
      </span>
    </div>`;
  }).join('');

  const filesHtml = `
    <div class="card">
      <div class="card-hd">File Status</div>
      ${fileRows}
    </div>`;

  // Log summary
  const logHtml = `
    <div class="card">
      <div class="card-hd">Execution Log (log.json)</div>
      ${logSum && logSum.count > 0 ? `
        <div class="overlay-info-row">
          <span class="key">Total records</span>
          <span class="val">${logSum.count}</span>
        </div>
        ${logSum.first ? `
        <div class="section-label">First Record</div>
        <pre class="log-box" style="max-height:80px">${escHtml(JSON.stringify(logSum.first, null, 2))}</pre>
        <div class="section-label">Last Record</div>
        <pre class="log-box" style="max-height:80px">${escHtml(JSON.stringify(logSum.last, null, 2))}</pre>` : ''}
      ` : '<div class="field-hint">No log records yet. Run a simulation first.</div>'}
    </div>`;

  // Metrics cards
  function metricsCard(title, m) {
    if (!m) return `<div class="card"><div class="card-hd">${title}</div><div class="field-hint">Not available.</div></div>`;
    const acc  = m.val_accuracy != null ? (m.val_accuracy * 100).toFixed(2) + '%' : '—';
    const base = m.majority_baseline_accuracy != null ? (m.majority_baseline_accuracy * 100).toFixed(2) + '%' : '—';
    const datasetLabel = m.dataset_label || m.dataset_path || '—';
    const valExamples = m.val_examples != null ? m.val_examples : sumCounts(m.val_true_counts);
    const fingerprint = m.training_config_fingerprint || '—';
    const datasetHash = m.dataset_sha256 ? String(m.dataset_sha256).slice(0, 12) : '—';
    return `<div class="card">
      <div class="card-hd">${title}</div>
      <div class="overlay-info-row"><span class="key">Val accuracy</span><span class="val">${acc}</span></div>
      <div class="overlay-info-row"><span class="key">Majority baseline</span><span class="val">${base}</span></div>
      <div class="overlay-info-row"><span class="key">Source dataset</span><span class="val">${escHtml(datasetLabel)}</span></div>
      <div class="overlay-info-row"><span class="key">Validation examples</span><span class="val">${escHtml(valExamples ?? '—')}</span></div>
      <div class="overlay-info-row"><span class="key">Dataset hash</span><span class="val"><code>${escHtml(datasetHash)}</code></span></div>
      <div class="overlay-info-row"><span class="key">Config fingerprint</span><span class="val"><code>${escHtml(fingerprint)}</code></span></div>
    </div>`;
  }

  let metricsNote = '';
  if (baseline && latest) {
    const sameDataset = (baseline.dataset_sha256 || baseline.dataset_path) === (latest.dataset_sha256 || latest.dataset_path);
    const sameConfig = baseline.training_config_fingerprint && latest.training_config_fingerprint
      && baseline.training_config_fingerprint === latest.training_config_fingerprint;
    if (sameDataset && !sameConfig) {
      metricsNote = `
        <div class="alert alert-info">
          Baseline and latest metrics point to the same source dataset label, but they were generated with different training configurations or preprocessing metadata.
        </div>`;
    }
  }

  return `
    <div class="artifacts-grid">
      ${overlayHtml}
      ${filesHtml}
      ${metricsNote}
      ${metricsCard('Baseline Training Metrics', baseline)}
      ${metricsCard('Latest Training Metrics', latest)}
      ${logHtml}
    </div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   CHARTS  (Phase 2)
══════════════════════════════════════════════════════════════════════════ */

function enableCharts() {
  // Switch Charts tab to show content; load data
  show('charts-body');
  hide('charts-empty');
}

async function loadCharts() {
  // Only load if simulation has run
  const simStatus = await apiGet('/api/simulate/status').catch(() => null);
  if (!simStatus || simStatus.state !== 'done') return;

  show('charts-body');
  hide('charts-empty');

  await Promise.all([loadDepthChart(), loadFillsChart()]);
}

async function loadDepthChart() {
  let data;
  try { data = await apiGet('/api/book/depth?levels=15'); }
  catch (_) { return; }

  const bids = (data.bids || []).reverse();  // price ascending → best bid at top visually
  const asks = data.asks || [];

  // Combined labels: all price levels
  const allPrices = [...bids.map(b => b.price), ...asks.map(a => a.price)].map(p => p.toFixed(2));
  const bidVols   = [...bids.map(b => b.volume), ...Array(asks.length).fill(0)];
  const askVols   = [...Array(bids.length).fill(0), ...asks.map(a => a.volume)];

  if (depthChart) { depthChart.destroy(); depthChart = null; }

  const ctx = $('depth-chart').getContext('2d');
  depthChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: allPrices,
      datasets: [
        {
          label: 'Bid Volume',
          data: bidVols,
          backgroundColor: 'rgba(63,185,80,0.55)',
          borderColor: 'rgba(63,185,80,0.8)',
          borderWidth: 1,
        },
        {
          label: 'Ask Volume',
          data: askVols,
          backgroundColor: 'rgba(248,81,73,0.45)',
          borderColor: 'rgba(248,81,73,0.7)',
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { labels: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` Vol: ${ctx.parsed.x.toFixed(2)}` } },
      },
      scales: {
        x: { stacked: true, ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#30363d' } },
        y: { stacked: true, ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#30363d' } },
      },
    },
  });
}

async function loadFillsChart() {
  let data;
  try { data = await apiGet('/api/trades?limit=200'); }
  catch (_) { return; }

  const trades = data.trades || [];
  if (trades.length === 0) return;

  const labels = trades.map((_, i) => i + 1);
  const prices = trades.map(t => t.exec_price);
  const colors = trades.map(t => t.side === 'buy' ? 'rgba(63,185,80,0.8)' : 'rgba(248,81,73,0.8)');

  if (fillsChart) { fillsChart.destroy(); fillsChart = null; }

  const ctx = $('fills-chart').getContext('2d');
  fillsChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Execution Price',
        data: trades.map((t, i) => ({ x: i + 1, y: t.exec_price })),
        backgroundColor: colors,
        pointRadius: 4,
        pointHoverRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const t = trades[ctx.dataIndex];
              return [
                ` Price: ${t.exec_price.toFixed(4)}`,
                ` Qty: ${t.filled_qty}`,
                ` Side: ${t.side}`,
              ];
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Fill #', color: '#8b949e' }, ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#30363d' } },
        y: { title: { display: true, text: 'Price', color: '#8b949e' }, ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: '#30363d' } },
      },
    },
  });

  // Also populate fills table
  renderFillsTable(trades.slice(-50));
  const countEl = $('fills-count');
  if (countEl) countEl.textContent = `(${trades.length} total, showing last 50)`;
}

function renderFillsTable(trades) {
  const tbody = $('fills-tbody');
  if (!tbody) return;
  tbody.innerHTML = trades.map(t => `
    <tr>
      <td>${escHtml(String(t.timestamp || '').replace('T', ' ').slice(0, 22))}</td>
      <td class="${t.side === 'buy' ? 'side-buy' : 'side-sell'}">${escHtml(t.side || '')}</td>
      <td>${fmt2(t.exec_price)}</td>
      <td>${fmt2(t.filled_qty)}</td>
      <td>${fmt2(t.remaining_qty)}</td>
      <td>${escHtml(String(t.maker_order_id || '').slice(0, 20))}</td>
    </tr>`).join('');
}

/* ─── escape HTML ────────────────────────────────────────────────────────── */
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ─── boot ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  initTabs();
  initSourceToggle();
  initTrainToggle();
  await loadAppConfig();
  loadOverlayNote();

  // Chart.js global defaults — dark theme
  if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#8b949e';
    Chart.defaults.borderColor = '#30363d';
    Chart.defaults.backgroundColor = '#161b22';
    Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  }
});
