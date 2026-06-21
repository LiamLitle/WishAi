// ═══════════════════════════════════════════════════════
//  WishAI Chat — app.js
//  IndexedDB + SSE streaming + gestion conversations
// ═══════════════════════════════════════════════════════

// ── STATE ──────────────────────────────────────────────
let db           = null;
let isGenerating = false;
let modelLoaded  = false;

let currentConv = null;   // conversation active

// ── INDEXEDDB ──────────────────────────────────────────
async function initDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('WishAIChat', 1);

    req.onupgradeneeded = e => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains('convs')) {
        const s = d.createObjectStore('convs', { keyPath: 'id' });
        s.createIndex('updatedAt', 'updatedAt', { unique: false });
      }
    };

    req.onsuccess = e => { db = e.target.result; resolve(); };
    req.onerror   = e => reject(e.target.error);
  });
}

function dbPut(conv) {
  return new Promise((res, rej) => {
    const tx = db.transaction('convs', 'readwrite');
    tx.objectStore('convs').put(conv);
    tx.oncomplete = res;
    tx.onerror    = rej;
  });
}

function dbGetAll() {
  return new Promise((res, rej) => {
    const tx  = db.transaction('convs', 'readonly');
    const req = tx.objectStore('convs').index('updatedAt').getAll();
    req.onsuccess = e => res(e.target.result.reverse());
    req.onerror   = rej;
  });
}

function dbDelete(id) {
  return new Promise((res, rej) => {
    const tx = db.transaction('convs', 'readwrite');
    tx.objectStore('convs').delete(id);
    tx.oncomplete = res;
    tx.onerror    = rej;
  });
}

async function saveConv() {
  if (!currentConv || currentConv.messages.length === 0) return;
  // Titre = premier message utilisateur
  if (currentConv.title === 'Nouvelle conversation' && currentConv.messages.length > 0) {
    const first = currentConv.messages.find(m => m.role === 'user');
    if (first) currentConv.title = first.text.slice(0, 45);
  }
  currentConv.updatedAt = new Date().toISOString();
  await dbPut(currentConv);
  renderConvList();
}

// ── CONVERSATIONS ──────────────────────────────────────
function newConv() {
  currentConv = {
    id:        Date.now(),
    title:     'Nouvelle conversation',
    messages:  [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  clearMessages();
  closeConvPanel();
  renderConvList();
}

async function switchConv(id) {
  const all  = await dbGetAll();
  const conv = all.find(c => c.id === id);
  if (!conv) return;
  currentConv = conv;
  closeConvPanel();
  replayMessages(conv.messages);
  renderConvList();
}

async function deleteConv(id, e) {
  e.stopPropagation();
  await dbDelete(id);
  if (currentConv && currentConv.id === id) newConv();
  renderConvList();
}

async function renderConvList() {
  const list = document.getElementById('conv-list');
  const all  = await dbGetAll();

  if (!all.length) {
    list.innerHTML = '<div class="conv-empty">Aucune conversation sauvegardee.</div>';
    return;
  }

  list.innerHTML = all.map(c => {
    const active = currentConv && currentConv.id === c.id ? ' active' : '';
    const date   = fmtDate(c.updatedAt);
    return `<div class="conv-item${active}" onclick="switchConv(${c.id})">
      <div class="conv-item-info">
        <div class="conv-title">${esc(c.title)}</div>
        <div class="conv-date">${date}</div>
      </div>
      <button class="conv-del" onclick="deleteConv(${c.id}, event)" title="Supprimer">&#x2715;</button>
    </div>`;
  }).join('');
}

function replayMessages(messages) {
  clearMessages(false);
  messages.forEach(m => addBubble(m.role, m.text, false));
  scrollBottom();
}

// ── MODELE ─────────────────────────────────────────────
async function refreshModels() {
  try {
    const models = await apiFetch('/api/models');
    const sel = document.getElementById('model-select');
    sel.innerHTML = models.length
      ? models.map(m => `<option value="${m.name}">${m.name} (${m.size_mb} Mo)</option>`).join('')
      : '<option value="">Aucun modele trouve</option>';
  } catch(e) {
    document.getElementById('model-select').innerHTML = '<option value="">Serveur inaccessible</option>';
  }
}

async function loadModel() {
  const name = document.getElementById('model-select').value;
  if (!name) return;
  document.getElementById('load-btn').disabled = true;
  setStatus('load', 'Chargement...');
  try {
    await apiFetch('/api/load', { method: 'POST', body: JSON.stringify({ name }) });
  } catch(e) {
    setStatus('err', 'Erreur : ' + e.message);
    document.getElementById('load-btn').disabled = false;
  }
}

async function checkStatus() {
  try {
    const s = await apiFetch('/api/status');
    const dot = document.getElementById('chip-dot');
    const lbl = document.getElementById('chip-label');
    const btn = document.getElementById('load-btn');

    if (s.loading) {
      dot.className = 'chip-dot load';
      lbl.textContent = 'Chargement...';
      setStatus('load', 'Chargement en cours...');
      btn.disabled = true;
      modelLoaded = false;
    } else if (s.error) {
      dot.className = 'chip-dot err';
      lbl.textContent = 'Erreur';
      setStatus('err', s.error);
      btn.disabled = false;
      modelLoaded = false;
    } else if (s.loaded) {
      const p = s.params >= 1e6
        ? (s.params / 1e6).toFixed(1) + 'M'
        : (s.params / 1e3).toFixed(0) + 'k';
      dot.className = 'chip-dot ok';
      lbl.textContent = s.name;
      setStatus('ok', p + ' params — ' + s.device.toUpperCase());
      btn.disabled = false;
      modelLoaded = true;
    } else {
      dot.className = 'chip-dot';
      lbl.textContent = 'Aucun modele';
      setStatus('', '');
      btn.disabled = false;
      modelLoaded = false;
    }
  } catch(e) {}
  updateSendBtn();
}

// ── ENVOI ──────────────────────────────────────────────
async function send() {
  if (isGenerating || !modelLoaded) return;

  const inputEl = document.getElementById('input');
  const prompt  = inputEl.value.trim();
  if (!prompt) return;

  inputEl.value = '';
  autoResize(inputEl);

  // Supprimer le welcome
  const w = document.getElementById('welcome');
  if (w) w.remove();

  const temperature = parseFloat(document.getElementById('temp-s').value);
  const max_tokens  = parseInt(document.getElementById('len-s').value);

  // Sauvegarder dans la conv
  currentConv.messages.push({ role: 'user', text: prompt, ts: Date.now() });
  addBubble('user', prompt);

  isGenerating = true;
  updateSendBtn();

  const aiBubble = addTypingBubble();
  let aiText = '';

  try {
    const resp = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prompt, temperature, max_tokens }),
    });

    if (!resp.ok) throw new Error('Erreur ' + resp.status);

    const reader = resp.body.getReader();
    const tdec   = new TextDecoder();
    let   buf    = '';
    let   started = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += tdec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let d;
        try { d = JSON.parse(line.slice(6)); } catch { continue; }
        if (d.error) throw new Error(d.error);
        if (d.token) {
          if (!started) { aiBubble.innerHTML = ''; started = true; }
          aiText += d.token;
          aiBubble.textContent = aiText;
          scrollBottom();
        }
        if (d.done) break;
      }
    }

    if (!started) aiBubble.textContent = '(vide)';

  } catch(e) {
    aiText = '[Erreur : ' + e.message + ']';
    aiBubble.innerHTML = '<span style="color:var(--red)">' + esc(aiText) + '</span>';
  }

  // Sauvegarder la reponse IA
  currentConv.messages.push({ role: 'ai', text: aiText, ts: Date.now() });
  await saveConv();

  isGenerating = false;
  updateSendBtn();
}

// ── BULLES ─────────────────────────────────────────────
function addBubble(role, text, doScroll = true) {
  const msgs = document.getElementById('messages');
  const row  = document.createElement('div');
  row.className = 'row ' + role;
  row.innerHTML =
    '<div class="av">' + (role === 'user' ? '&#128100;' : '&#129504;') + '</div>' +
    '<div class="bubble">' + esc(text) + '</div>';
  msgs.appendChild(row);
  if (doScroll) scrollBottom();
  return row.querySelector('.bubble');
}

function addTypingBubble() {
  const msgs = document.getElementById('messages');
  const row  = document.createElement('div');
  row.className = 'row ai';
  row.innerHTML =
    '<div class="av">&#129504;</div>' +
    '<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  msgs.appendChild(row);
  scrollBottom();
  return row.querySelector('.bubble');
}

function clearMessages(addWelcome = true) {
  const msgs = document.getElementById('messages');
  msgs.innerHTML = addWelcome
    ? '<div class="welcome" id="welcome"><div class="welcome-icon">&#129504;</div><p class="welcome-title">WishAI Chat</p><p class="welcome-hint">Charge un modele, puis tape un debut de texte.</p></div>'
    : '';
}

// ── PANELS UI ──────────────────────────────────────────
function toggleConvPanel() {
  document.getElementById('conv-panel').classList.toggle('open');
  document.getElementById('conv-overlay').classList.toggle('open');
}
function closeConvPanel() {
  document.getElementById('conv-panel').classList.remove('open');
  document.getElementById('conv-overlay').classList.remove('open');
}
function toggleModelPanel() {
  document.getElementById('model-panel').classList.toggle('open');
  document.getElementById('model-overlay').classList.toggle('open');
}
function closeModelPanel() {
  document.getElementById('model-panel').classList.remove('open');
  document.getElementById('model-overlay').classList.remove('open');
}

// ── UTILS ──────────────────────────────────────────────
function scrollBottom() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function updateSendBtn() {
  const btn = document.getElementById('send-btn');
  btn.disabled = isGenerating || !modelLoaded;
  btn.textContent = isGenerating ? '…' : '↑';
}

function setStatus(cls, text) {
  const el = document.getElementById('model-status');
  el.className = 'model-status' + (cls ? ' ' + cls : '');
  el.textContent = text;
}

function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

function esc(t) {
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmtDate(iso) {
  const d = new Date(iso);
  const now = new Date();
  const diff = (now - d) / 1000;
  if (diff < 60)    return 'maintenant';
  if (diff < 3600)  return Math.floor(diff/60) + ' min';
  if (diff < 86400) return Math.floor(diff/3600) + 'h';
  return d.toLocaleDateString('fr-FR', { day:'numeric', month:'short' });
}

async function apiFetch(url, opts = {}) {
  const r = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  return r.json();
}

// ── INIT ───────────────────────────────────────────────
async function init() {
  await initDB();
  newConv();
  await renderConvList();
  await refreshModels();
  checkStatus();
  setInterval(checkStatus, 2000);
}

init();
