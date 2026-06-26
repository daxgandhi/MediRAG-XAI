/**
 * MEDIRAG-XAI — Shared Application Utilities
 * API base URL, fetch wrapper, toast notifications, helpers
 */

const API_BASE = 'http://localhost:8000';

// ── API Client ──────────────────────────────────────────────────────────────
const api = {
  async post(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
    return response.json();
  },

  async postForm(endpoint, formData) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
    return response.json();
  },

  async get(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },
};

// ── Toast Notifications ────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = {
    success: 'fas fa-check-circle',
    error:   'fas fa-times-circle',
    warning: 'fas fa-exclamation-triangle',
    info:    'fas fa-info-circle',
  };
  const colors = {
    success: 'var(--success)',
    error:   'var(--danger)',
    warning: 'var(--warning)',
    info:    'var(--primary)',
  };

  const toast = document.createElement('div');
  toast.className = `toast-custom toast-${type}`;
  toast.innerHTML = `
    <i class="${icons[type] || icons.info}" style="color:${colors[type]};font-size:1.1rem;flex-shrink:0;margin-top:2px"></i>
    <div>
      <div style="font-size:0.85rem;font-weight:600;color:var(--text-primary)">${message}</div>
    </div>
    <button onclick="this.parentElement.remove()" style="background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:1rem;margin-left:auto">×</button>
  `;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

// ── Loading State Helpers ──────────────────────────────────────────────────
function setLoading(btnId, loading, text = 'Processing...') {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  if (loading) {
    btn._orig = btn.innerHTML;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>${text}`;
    btn.disabled = true;
  } else {
    btn.innerHTML = btn._orig || btn.innerHTML;
    btn.disabled = false;
  }
}

// ── Confidence Color ───────────────────────────────────────────────────────
function confidenceColor(pct) {
  if (pct >= 70) return 'var(--danger)';
  if (pct >= 40) return 'var(--warning)';
  return 'var(--success)';
}

function confidenceClass(pct) {
  if (pct >= 70) return 'badge-danger';
  if (pct >= 40) return 'badge-warning';
  return 'badge-success';
}

// ── Severity Color ─────────────────────────────────────────────────────────
function severityClass(severity) {
  const map = {
    critical: 'severity-critical',
    high:     'severity-high',
    moderate: 'severity-moderate',
    low:      'severity-low',
  };
  return map[(severity || '').toLowerCase()] || 'severity-moderate';
}

// ── Markdown-ish renderer ──────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/^#{1,3}\s+(.+)$/gm, '<div style="font-weight:700;color:var(--primary-light);margin:0.5rem 0">$1</div>')
    .replace(/\n/g, '<br>');
}

// ── Check API Status ───────────────────────────────────────────────────────
async function checkAPIStatus() {
  const el = document.getElementById('apiStatus');
  if (!el) return;
  try {
    await api.get('/health');
    el.textContent = '✅ API Connected';
    el.parentElement.className = 'badge-pill badge-success';
  } catch {
    el.textContent = '❌ API Offline';
    el.parentElement.className = 'badge-pill badge-danger';
    showToast('Backend is offline. Start with: python main.py', 'warning', 8000);
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkAPIStatus();
});
