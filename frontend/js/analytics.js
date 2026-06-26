/**
 * MEDIRAG-XAI — Analytics Dashboard JavaScript
 * Fetches /api/analytics and renders 5 Chart.js charts
 */

Chart.defaults.color             = '#94A3B8';
Chart.defaults.borderColor       = 'rgba(148,163,184,0.1)';
Chart.defaults.font.family       = "'Inter', system-ui, sans-serif";

let chartInstances = {};

const CHART_COLORS = {
  primary:  '#0EA5E9',
  accent:   '#6366F1',
  success:  '#10B981',
  warning:  '#F59E0B',
  danger:   '#F43F5E',
  info:     '#06B6D4',
  purple:   '#A855F7',
  pink:     '#EC4899',
  orange:   '#F97316',
  teal:     '#14B8A6',
  lime:     '#84CC16',
};

const PALETTE = Object.values(CHART_COLORS);

// ── Tooltip defaults ─────────────────────────────────────────────────────────
const TOOLTIP_OPTS = {
  backgroundColor: 'rgba(15,23,42,0.95)',
  titleColor:      '#F1F5F9',
  bodyColor:       '#94A3B8',
  borderColor:     'rgba(148,163,184,0.2)',
  borderWidth:     1,
  padding:         12,
  cornerRadius:    8,
};

// ── Fetch analytics data ─────────────────────────────────────────────────────
async function fetchAnalytics() {
  try {
    const data = await api.get('/api/analytics');
    return data;
  } catch (e) {
    console.warn('Analytics API error:', e);
    showToast('Using demo analytics data', 'info');
    return getDemoData();
  }
}

// ── Main render function ─────────────────────────────────────────────────────
async function renderDashboard() {
  const data = await fetchAnalytics();

  // Update KPIs
  animateCounter('kpiPredictions', data.total_predictions  || 0);
  animateCounter('kpiChats',       data.total_chats        || 0);
  animateCounter('kpiReports',     data.total_reports      || 0);
  animateCounter('kpiDrugChecks',  data.total_drug_checks  || 0);

  // Update timestamp
  const ts = document.getElementById('lastUpdated');
  if (ts) ts.textContent = new Date().toLocaleTimeString();

  // Render charts
  renderDiseaseChart(data.disease_distribution || {});
  renderConfidenceChart(data.prediction_confidence || [], data.confidence_labels || []);
  renderSymptomChart(data.common_symptoms || {});
  renderDrugAlertChart(data.drug_alerts || {});
  renderChatChart(data.chat_usage || []);

  showToast('Dashboard refreshed', 'success');
}

// ── Disease Distribution (Doughnut) ───────────────────────────────────────────
function renderDiseaseChart(distribution) {
  destroyChart('diseaseChart');
  const ctx = document.getElementById('diseaseChart');
  if (!ctx) return;

  const labels = Object.keys(distribution);
  const values = Object.values(distribution);

  chartInstances.disease = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data:            values,
        backgroundColor: PALETTE.slice(0, labels.length).map(c => c + 'CC'),
        borderColor:     PALETTE.slice(0, labels.length),
        borderWidth:     2,
        hoverOffset:     8,
      }],
    },
    options: {
      responsive: true,
      cutout:     '62%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color:    '#94A3B8',
            padding:  12,
            font:     { size: 11 },
            usePointStyle: true,
            pointStyleWidth: 8,
          },
        },
        tooltip: { ...TOOLTIP_OPTS },
      },
    },
  });
}

// ── Confidence Trend (Line) ──────────────────────────────────────────────────
function renderConfidenceChart(values, labels) {
  destroyChart('confidenceChart');
  const ctx = document.getElementById('confidenceChart');
  if (!ctx) return;

  chartInstances.confidence = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label:           'Avg Confidence (%)',
        data:            values,
        borderColor:     CHART_COLORS.primary,
        backgroundColor: 'rgba(14,165,233,0.08)',
        borderWidth:     2.5,
        pointRadius:     4,
        pointHoverRadius: 7,
        pointBackgroundColor: CHART_COLORS.primary,
        fill:            true,
        tension:         0.4,
      }],
    },
    options: {
      responsive:          true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#94A3B8', font: { size: 12 } } },
        tooltip: { ...TOOLTIP_OPTS, callbacks: { label: (c) => ` Confidence: ${c.raw}%` } },
      },
      scales: {
        x: { grid: { color: 'rgba(148,163,184,0.07)' }, ticks: { color: '#64748B' } },
        y: {
          grid:   { color: 'rgba(148,163,184,0.07)' },
          ticks:  { color: '#64748B', callback: (v) => `${v}%` },
          min:    50,
          max:    100,
        },
      },
    },
  });
}

// ── Common Symptoms (Horizontal Bar) ────────────────────────────────────────
function renderSymptomChart(symptoms) {
  destroyChart('symptomChart');
  const ctx = document.getElementById('symptomChart');
  if (!ctx) return;

  const labels = Object.keys(symptoms);
  const values = Object.values(symptoms);

  chartInstances.symptom = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label:           'Frequency',
        data:            values,
        backgroundColor: PALETTE.slice(0, labels.length).map(c => c + 'AA'),
        borderColor:     PALETTE.slice(0, labels.length),
        borderWidth:     1.5,
        borderRadius:    5,
        borderSkipped:   false,
      }],
    },
    options: {
      indexAxis:   'y',
      responsive:  true,
      plugins: {
        legend: { display: false },
        tooltip: { ...TOOLTIP_OPTS, callbacks: { label: (c) => ` Count: ${c.raw}` } },
      },
      scales: {
        x: { grid: { color: 'rgba(148,163,184,0.07)' }, ticks: { color: '#64748B' } },
        y: { grid: { display: false }, ticks: { color: '#94A3B8', font: { size: 11 } } },
      },
    },
  });
}

// ── Drug Alerts (Pie) ────────────────────────────────────────────────────────
function renderDrugAlertChart(alerts) {
  destroyChart('drugAlertChart');
  const ctx = document.getElementById('drugAlertChart');
  if (!ctx) return;

  const labels = Object.keys(alerts);
  const values = Object.values(alerts);

  const alertColors = [
    '#F43F5E', '#F59E0B', '#6366F1', '#0EA5E9', '#10B981',
  ];

  chartInstances.drugAlert = new Chart(ctx, {
    type: 'pie',
    data: {
      labels,
      datasets: [{
        data:            values,
        backgroundColor: alertColors.slice(0, labels.length).map(c => c + 'CC'),
        borderColor:     alertColors.slice(0, labels.length),
        borderWidth:     2,
        hoverOffset:     6,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color:   '#94A3B8',
            padding: 10,
            font:    { size: 11 },
            usePointStyle: true,
          },
        },
        tooltip: { ...TOOLTIP_OPTS },
      },
    },
  });
}

// ── Chat Usage (Bar) ─────────────────────────────────────────────────────────
function renderChatChart(usage) {
  destroyChart('chatChart');
  const ctx = document.getElementById('chatChart');
  if (!ctx) return;

  const labels = usage.map((_, i) => `Day ${i + 1}`);

  chartInstances.chat = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label:           'Chat Sessions',
        data:            usage,
        backgroundColor: 'rgba(16,185,129,0.5)',
        borderColor:     CHART_COLORS.success,
        borderWidth:     1.5,
        borderRadius:    5,
        borderSkipped:   false,
      }],
    },
    options: {
      responsive:  true,
      interaction: { mode: 'index' },
      plugins: {
        legend: { labels: { color: '#94A3B8', font: { size: 12 } } },
        tooltip: { ...TOOLTIP_OPTS, callbacks: { label: (c) => ` Sessions: ${c.raw}` } },
      },
      scales: {
        x: { grid: { color: 'rgba(148,163,184,0.07)' }, ticks: { color: '#64748B' } },
        y: { grid: { color: 'rgba(148,163,184,0.07)' }, ticks: { color: '#64748B' } },
      },
    },
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function destroyChart(id) {
  const key = id.replace('Chart', '').toLowerCase();
  if (chartInstances[key]) {
    chartInstances[key].destroy();
    delete chartInstances[key];
  }
}

function animateCounter(id, target) {
  const el  = document.getElementById(id);
  if (!el) return;
  const start    = 0;
  const duration = 1000;
  const step     = (target - start) / (duration / 16);
  let current    = start;
  const timer    = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = Math.floor(current).toLocaleString();
  }, 16);
}

function refreshData() {
  renderDashboard();
}

// ── Demo Fallback ────────────────────────────────────────────────────────────
function getDemoData() {
  return {
    disease_distribution: {
      "Diabetes": 145, "Hypertension": 132, "Bronchial Asthma": 98,
      "Tuberculosis": 67, "Heart attack": 89, "Migraine": 75,
      "Dengue": 88, "Typhoid": 58, "Pneumonia": 65, "Other": 141,
    },
    prediction_confidence: [72, 78, 85, 81, 88, 91, 87, 93, 89, 95, 92, 88, 94, 90, 96],
    confidence_labels: ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
    common_symptoms: {
      "Fever": 312, "Fatigue": 287, "Cough": 245, "Headache": 198,
      "Chest Pain": 156, "Breathlessness": 143, "Nausea": 132, "Joint Pain": 118,
    },
    drug_alerts: {
      "Contraindication": 45, "Pregnancy Warning": 23,
      "Allergy Alert": 31, "Drug Interaction": 67, "Safe": 234,
    },
    chat_usage: [12, 18, 25, 32, 28, 41, 35, 48, 52, 44, 38, 55, 61, 47, 58],
    total_predictions: 1100,
    total_chats:       594,
    total_reports:     234,
    total_drug_checks: 400,
  };
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderDashboard();
});
