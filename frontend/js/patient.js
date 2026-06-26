/**
 * MEDIRAG-XAI — Patient Portal JavaScript
 * Chatbot with RAG answers, typing animation, sources panel, session stats
 */

let chatHistory     = [];
let sessionId       = `session_${Date.now()}`;
let totalQuestions  = 0;
let totalSources    = 0;
let totalChunks     = 0;

const SUGGESTED_QUESTIONS = [
  "What is diabetes?",
  "What are the symptoms of hypertension?",
  "How can I manage high blood pressure?",
  "What are the side effects of metformin?",
  "What is tuberculosis and how is it treated?",
  "How does dengue fever spread?",
  "What is asthma?",
  "How is malaria diagnosed?",
  "What is the treatment for COVID-19?",
  "What foods should diabetics avoid?",
  "What is the DASH diet for hypertension?",
  "How do I recognize a heart attack?",
  "What is insulin and when is it needed?",
  "What is HbA1c?",
  "How can I prevent dengue fever?",
  "What are normal blood pressure levels?",
];

// ── Render Suggested Questions ───────────────────────────────────────────────
function renderSuggestions() {
  const container = document.getElementById('suggestedQuestions');
  if (!container) return;
  container.innerHTML = SUGGESTED_QUESTIONS.map(q => `
    <button class="suggested-q-btn" onclick="selectSuggestion('${q.replace(/'/g, "\\'")}')">
      <i class="fas fa-arrow-right me-2" style="color:var(--primary);font-size:0.7rem"></i>${q}
    </button>
  `).join('');
}

function selectSuggestion(question) {
  const input = document.getElementById('chatInput');
  if (input) {
    input.value = question;
    sendMessage();
  }
}

// ── Send Message ─────────────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = input?.value?.trim();
  if (!message) return;

  input.value = '';
  input.disabled = true;
  document.getElementById('sendBtn').disabled = true;

  // Add user bubble
  appendUserBubble(message);
  chatHistory.push({ role: 'user', content: message });

  // Show typing indicator
  const typingId = showTypingIndicator();

  try {
    const data = await api.post('/api/chat', {
      message,
      session_id: sessionId,
    });

    removeTypingIndicator(typingId);
    appendBotBubble(data.answer, data.sources, data.chunks);

    // Update sources panel
    updateSourcesPanel(data.sources, data.chunks);

    // Update stats
    totalQuestions++;
    totalSources += (data.sources || []).length;
    totalChunks  += (data.chunks  || []).length;
    updateStats();

    chatHistory.push({ role: 'bot', content: data.answer });

  } catch (e) {
    removeTypingIndicator(typingId);
    appendBotBubble(
      `⚠️ I couldn't connect to the backend. Error: ${e.message}\n\nPlease ensure the FastAPI server is running: **python main.py**`,
      [], []
    );
    showToast('Backend connection error', 'error');
  } finally {
    if (input) input.disabled = false;
    document.getElementById('sendBtn').disabled = false;
    input?.focus();
  }
}

// ── Chat Bubble Renderers ────────────────────────────────────────────────────
function appendUserBubble(message) {
  const chatWindow = document.getElementById('chatWindow');
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const wrapper = document.createElement('div');
  wrapper.className = 'd-flex flex-row-reverse gap-2 align-items-start';
  wrapper.innerHTML = `
    <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--primary));display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <i class="fas fa-user" style="color:#fff;font-size:0.85rem"></i>
    </div>
    <div>
      <div class="chat-bubble user">${escapeHtml(message)}</div>
      <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;text-align:right;margin-right:4px">You • ${now}</div>
    </div>
  `;
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

function appendBotBubble(answer, sources, chunks) {
  const chatWindow = document.getElementById('chatWindow');
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const sourceTags = (sources || []).map(s =>
    `<span class="source-tag"><i class="fas fa-file-alt"></i>${s}</span>`
  ).join('');

  const wrapper = document.createElement('div');
  wrapper.className = 'd-flex gap-2 align-items-start';
  wrapper.innerHTML = `
    <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--primary),var(--accent));display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <i class="fas fa-robot" style="color:#fff;font-size:0.85rem"></i>
    </div>
    <div style="flex:1;min-width:0">
      <div class="chat-bubble bot">
        <div style="line-height:1.7">${renderMarkdown(answer)}</div>
        ${sourceTags ? `<div class="mt-2 d-flex flex-wrap gap-1">${sourceTags}</div>` : ''}
      </div>
      <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;margin-left:4px">
        MediBot • ${now} ${sources?.length ? `• ${sources.length} source(s)` : ''}
      </div>
    </div>
  `;
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

function showTypingIndicator() {
  const chatWindow = document.getElementById('chatWindow');
  const id = `typing_${Date.now()}`;

  const wrapper = document.createElement('div');
  wrapper.id = id;
  wrapper.className = 'd-flex gap-2 align-items-start';
  wrapper.innerHTML = `
    <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--primary),var(--accent));display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <i class="fas fa-robot" style="color:#fff;font-size:0.85rem"></i>
    </div>
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatWindow.appendChild(wrapper);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  document.getElementById(id)?.remove();
}

// ── Sources Panel ─────────────────────────────────────────────────────────────
function updateSourcesPanel(sources, chunks) {
  const sourcePanel = document.getElementById('sourcesPanel');
  const chunksPanel = document.getElementById('chunksPanel');

  if (sourcePanel) {
    if (!sources?.length) {
      sourcePanel.innerHTML = `<div style="font-size:0.8rem;color:var(--text-muted);text-align:center;padding:0.5rem">No sources retrieved</div>`;
    } else {
      sourcePanel.innerHTML = sources.map(s => `
        <div class="d-flex align-items-center gap-2 mb-2 p-2" style="background:rgba(14,165,233,0.08);border-radius:8px;border:1px solid rgba(14,165,233,0.15)">
          <i class="fas fa-file-medical" style="color:var(--primary);font-size:0.9rem"></i>
          <span style="font-size:0.8rem;color:var(--text-secondary)">${s}</span>
        </div>
      `).join('');
    }
  }

  if (chunksPanel) {
    if (!chunks?.length) {
      chunksPanel.innerHTML = `<div style="font-size:0.8rem;color:var(--text-muted);text-align:center;padding:0.5rem">No evidence chunks retrieved</div>`;
    } else {
      chunksPanel.innerHTML = chunks.slice(0, 3).map(c => `
        <div class="source-citation mb-2">
          <strong style="font-size:0.72rem"><i class="fas fa-file me-1"></i>${c.source}</strong><br>
          <span style="font-size:0.75rem;color:var(--text-secondary)">"${c.text?.substring(0, 150)}..."</span>
        </div>
      `).join('');
    }
  }
}

// ── Stats Update ─────────────────────────────────────────────────────────────
function updateStats() {
  const q = document.getElementById('statQuestions');
  const s = document.getElementById('statSources');
  const c = document.getElementById('statChunks');
  if (q) q.textContent = totalQuestions;
  if (s) s.textContent = totalSources;
  if (c) c.textContent = totalChunks;
}

// ── Chat Controls ────────────────────────────────────────────────────────────
function clearChat() {
  const chatWindow = document.getElementById('chatWindow');
  if (!chatWindow) return;

  // Keep only welcome message
  const bubbles = chatWindow.querySelectorAll('.d-flex');
  bubbles.forEach((b, i) => { if (i > 0) b.remove(); });

  chatHistory    = [];
  totalQuestions = 0;
  totalSources   = 0;
  totalChunks    = 0;
  sessionId      = `session_${Date.now()}`;
  updateStats();

  updateSourcesPanel([], []);
  showToast('Chat cleared', 'info');
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  const w = document.getElementById('chatWindow');
  if (w) w.scrollTop = w.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderSuggestions();

  // Allow Enter key in chat input
  document.getElementById('chatInput')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
});
