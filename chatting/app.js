// ═══════════════════════════════════════════════════════
//  WishAI Chat — app.js
//  IndexedDB + SSE streaming + gestion conversations
// ═══════════════════════════════════════════════════════

// ── STATE ──────────────────────────────────────────────
let db               = null;
let isGenerating     = false;
let modelLoaded      = false;
let _cachedModelName = null;   // nom du modèle actuellement en cache serveur

let currentConv = null;   // conversation active

// ── FOND GLOW (bords violets + glow central) ───────────
(function initGlow() {
  const cv  = document.getElementById('bg-canvas');
  const ctx = cv.getContext('2d');

  function drawEdgeGlow(w, h, t) {
    const pulse = 1 + 0.08 * Math.sin(t / 2400);

    // Blob haut-centre
    const gT = ctx.createRadialGradient(w * 0.5, -h * 0.05, 0, w * 0.5, -h * 0.05, h * 0.52);
    gT.addColorStop(0,    `rgba(147,51,234,${0.70 * pulse})`);
    gT.addColorStop(0.35, `rgba(126,34,206,${0.40 * pulse})`);
    gT.addColorStop(0.65, `rgba(109,40,217,${0.14 * pulse})`);
    gT.addColorStop(1,    'rgba(0,0,0,0)');
    ctx.save(); ctx.scale(1, 0.45);
    ctx.fillStyle = gT; ctx.fillRect(0, 0, w, h * 2);
    ctx.restore();

    // Blob bas-centre
    const gB = ctx.createRadialGradient(w * 0.5, h * 1.05, 0, w * 0.5, h * 1.05, h * 0.52);
    gB.addColorStop(0,    `rgba(147,51,234,${0.65 * pulse})`);
    gB.addColorStop(0.35, `rgba(126,34,206,${0.38 * pulse})`);
    gB.addColorStop(0.65, `rgba(109,40,217,${0.13 * pulse})`);
    gB.addColorStop(1,    'rgba(0,0,0,0)');
    ctx.save(); ctx.scale(1, 0.45);
    ctx.fillStyle = gB; ctx.fillRect(0, h * 2 * (1 - 1.05 / 0.45) / 2, w, h * 2);
    ctx.restore();

    // Blob gauche
    const gL = ctx.createRadialGradient(-w * 0.04, h * 0.5, 0, -w * 0.04, h * 0.5, w * 0.48);
    gL.addColorStop(0,    `rgba(139,92,246,${0.65 * pulse})`);
    gL.addColorStop(0.35, `rgba(109,40,217,${0.35 * pulse})`);
    gL.addColorStop(0.65, `rgba(91,33,182,${0.12 * pulse})`);
    gL.addColorStop(1,    'rgba(0,0,0,0)');
    ctx.save(); ctx.scale(0.42, 1);
    ctx.fillStyle = gL; ctx.fillRect(0, 0, w * 2, h);
    ctx.restore();

    // Blob droite
    const gR = ctx.createRadialGradient(w * 1.04, h * 0.5, 0, w * 1.04, h * 0.5, w * 0.48);
    gR.addColorStop(0,    `rgba(139,92,246,${0.65 * pulse})`);
    gR.addColorStop(0.35, `rgba(109,40,217,${0.35 * pulse})`);
    gR.addColorStop(0.65, `rgba(91,33,182,${0.12 * pulse})`);
    gR.addColorStop(1,    'rgba(0,0,0,0)');
    ctx.save(); ctx.scale(0.42, 1);
    ctx.translate(w / 0.42, 0);
    ctx.fillStyle = gR; ctx.fillRect(-w * 2, 0, w * 2, h);
    ctx.restore();
  }

  function frame() {
    const w = window.innerWidth, h = window.innerHeight;
    cv.width = w; cv.height = h;
    const t = Date.now();

    ctx.clearRect(0, 0, w, h);
    drawEdgeGlow(w, h, t);

    requestAnimationFrame(frame);
  }

  frame();
})();

// ── TRANSITION WELCOME ↔ CHAT ──────────────────────────
function activateChat(instant = false) {
  const app  = document.getElementById('app');
  if (app.classList.contains('has-chat')) return;

  const wrap = document.getElementById('welcome-wrap');
  const iz   = document.getElementById('input-zone');
  const bg   = document.getElementById('bg-canvas');

  const doSwitch = () => {
    if (wrap) wrap.style.display = 'none';
    if (iz)   app.appendChild(iz);
    app.classList.add('has-chat');
  };

  if (instant) {
    doSwitch();
    if (bg) bg.classList.add('fade-out');
  } else {
    if (wrap) wrap.classList.add('hide');
    if (bg)   bg.classList.add('fade-out');
    setTimeout(doSwitch, 350);
  }
}

function deactivateChat() {
  const app  = document.getElementById('app');
  const wrap = document.getElementById('welcome-wrap');
  const iz   = document.getElementById('input-zone');
  const bg   = document.getElementById('bg-canvas');

  if (!app.classList.contains('has-chat')) return;

  app.classList.remove('has-chat');
  if (wrap) {
    wrap.style.display = '';
    wrap.classList.remove('hide');
    if (iz) wrap.appendChild(iz);
  }
  if (bg) bg.classList.remove('fade-out');
}

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
  if (messages.length > 0) activateChat(true);
  messages.forEach(m => addBubble(m.role, m.text, false));
  scrollBottom();
}

// ── MODELE ─────────────────────────────────────────────
async function refreshModels() {
  try {
    let models = await apiFetch('/api/models');
    models.sort((a, b) => b.size_mb - a.size_mb);
    const sel = document.getElementById('model-select-inline');
    if (!sel) return;
    sel.innerHTML = models.length
      ? models.map(m => `<option value="${m.name}">${m.name} (${m.size_mb} Mo)</option>`).join('')
      : '<option value="">Aucun modele trouve</option>';
    // Si un modèle est déjà en cache, sélectionner le bon
    if (_cachedModelName) {
      for (const opt of sel.options) {
        if (opt.value === _cachedModelName) { sel.value = _cachedModelName; break; }
      }
    }
    _syncLoadBtn();
  } catch(e) {
    const sel = document.getElementById('model-select-inline');
    if (sel) sel.innerHTML = '<option value="">Serveur inaccessible</option>';
  }
}

// Auto-chargement quand on change la sélection
function onModelSelectChange() {
  const name = document.getElementById('model-select-inline')?.value;
  if (!name) return;
  if (name === _cachedModelName) {
    // Déjà en cache — pas besoin de recharger
    _syncLoadBtn();
    return;
  }
  loadModel();
}

function _syncLoadBtn() {
  const sel = document.getElementById('model-select-inline');
  const btn = document.getElementById('load-inline-btn');
  if (!btn || !sel) return;
  const selected = sel.value;
  if (selected && selected === _cachedModelName) {
    btn.textContent = '✓ En cache';
    btn.disabled    = true;
    btn.classList.add('cached');
  } else {
    btn.textContent = 'Charger';
    btn.disabled    = false;
    btn.classList.remove('cached');
  }
}

async function loadModel() {
  const name = document.getElementById('model-select-inline')?.value;
  if (!name) return;
  const btn = document.getElementById('load-inline-btn');
  if (btn) { btn.disabled = true; btn.textContent = '…'; btn.classList.remove('cached'); }
  setStatus('load', 'Chargement...');
  try {
    await apiFetch('/api/load', { method: 'POST', body: JSON.stringify({ name }) });
  } catch(e) {
    setStatus('err', 'Erreur : ' + e.message);
    _syncLoadBtn();
  }
}

async function checkStatus() {
  try {
    const s   = await apiFetch('/api/status');
    const btn = document.getElementById('load-inline-btn');

    if (s.loading) {
      setStatus('load', 'Chargement...');
      if (btn) { btn.disabled = true; btn.textContent = '…'; }
      modelLoaded = false;
    } else if (s.error) {
      setStatus('err', s.error);
      _cachedModelName = null;
      _syncLoadBtn();
      modelLoaded = false;
    } else if (s.loaded) {
      const p = s.params >= 1e6
        ? (s.params / 1e6).toFixed(1) + 'M'
        : (s.params / 1e3).toFixed(0) + 'k';
      setStatus('ok', '⚡ ' + s.name + ' — ' + p);
      // Mettre à jour le cache et synchroniser le select
      if (_cachedModelName !== s.name) {
        _cachedModelName = s.name;
        const sel = document.getElementById('model-select-inline');
        if (sel) {
          for (const opt of sel.options) {
            if (opt.value === s.name) { sel.value = s.name; break; }
          }
        }
      }
      _syncLoadBtn();
      modelLoaded = true;
    } else {
      setStatus('', '');
      _cachedModelName = null;
      _syncLoadBtn();
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
  activateChat();
  currentConv.messages.push({ role: 'user', text: prompt, ts: Date.now() });
  addBubble('user', prompt);
  await _doGenerate(prompt);
}

async function regenerate() {
  if (isGenerating) return;
  const msgs = document.getElementById('messages');
  // Retire la dernière bulle IA
  const aiRows = msgs.querySelectorAll('.row.ai');
  if (!aiRows.length) return;
  aiRows[aiRows.length - 1].remove();
  if (currentConv.messages.at(-1)?.role === 'ai') currentConv.messages.pop();
  // Dernier prompt utilisateur
  const lastUser = [...currentConv.messages].reverse().find(m => m.role === 'user');
  if (!lastUser) return;
  await _doGenerate(lastUser.text);
}

async function _doGenerate(prompt) {
  const temperature = parseFloat(document.getElementById('temp-s').value);
  const max_tokens  = parseInt(document.getElementById('len-s').value);

  isGenerating = true;
  updateSendBtn();

  const aiBubble = addTypingBubble();
  let aiText    = '';
  let tokenCount = 0;
  const t0 = Date.now();

  try {
    const resp = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prompt, temperature, max_tokens }),
    });
    if (!resp.ok) throw new Error('Erreur ' + resp.status);

    const reader  = resp.body.getReader();
    const tdec    = new TextDecoder();
    let   buf     = '';
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
          tokenCount++;
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

  // Meta : temps + tokens + actions
  const elapsed = Date.now() - t0;
  const metaDiv = aiBubble.closest('.bubble-wrap')?.querySelector('.bubble-meta');
  if (metaDiv) {
    metaDiv.innerHTML =
      '<button class="meta-btn" onclick="copyBubble(this)" title="Copier">⎘ Copier</button>' +
      '<span class="meta-sep">·</span>' +
      '<span class="meta-info">' + elapsed + ' ms</span>' +
      '<span class="meta-sep">·</span>' +
      '<span class="meta-info">' + tokenCount + ' tokens</span>' +
      '<button class="meta-btn meta-regen" onclick="regenerate()" title="Régénérer">↺ Régénérer</button>';
  }

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
  const av = role === 'user' ? '&#128100;' : '&#129504;';
  row.innerHTML =
    '<div class="av">' + av + '</div>' +
    '<div class="bubble-wrap">' +
      '<div class="bubble">' + esc(text) + '</div>' +
      '<div class="bubble-meta">' +
        '<button class="meta-btn" onclick="copyBubble(this)" title="Copier">⎘ Copier</button>' +
      '</div>' +
    '</div>';
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
    '<div class="bubble-wrap">' +
      '<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>' +
      '<div class="bubble-meta"></div>' +
    '</div>';
  msgs.appendChild(row);
  scrollBottom();
  return row.querySelector('.bubble');
}

function copyBubble(btn) {
  const bubble = btn.closest('.bubble-wrap')?.querySelector('.bubble');
  if (!bubble) return;
  navigator.clipboard.writeText(bubble.textContent.trim()).then(() => {
    const prev = btn.textContent;
    btn.textContent = '✓ Copié !';
    btn.style.color = 'var(--green)';
    setTimeout(() => { btn.textContent = prev; btn.style.color = ''; }, 1800);
  });
}

function clearMessages(showWelcome = true) {
  document.getElementById('messages').innerHTML = '';
  if (showWelcome) {
    setWelcomeGreeting();
    deactivateChat();
  }
}

// ── PANELS UI ──────────────────────────────────────────
function toggleConvPanel() {
  const open = document.getElementById('conv-panel').classList.toggle('open');
  document.getElementById('conv-overlay').classList.toggle('open');
  const btn = document.querySelector('.sidebar-toggle');
  if (btn) btn.style.opacity = open ? '0' : '';
}
function closeConvPanel() {
  document.getElementById('conv-panel').classList.remove('open');
  document.getElementById('conv-overlay').classList.remove('open');
  const btn = document.querySelector('.sidebar-toggle');
  if (btn) btn.style.opacity = '';
}
function toggleModelPanel() {}
function closeModelPanel() {}

function toggleMoreOpts() {
  const panel = document.getElementById('more-opts-panel');
  const btn   = document.getElementById('more-opts-btn');
  if (!panel) return;
  const open = panel.classList.toggle('open');
  btn.textContent = open ? '✕ Moins d\'options' : '⚙ Plus d\'options';
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
function setWelcomeGreeting() {
  const el = document.getElementById('welcome-title');
  if (!el) return;
  const h = new Date().getHours();
  const greet = h < 5 ? 'Bonne nuit' : h < 12 ? 'Bonjour' : h < 18 ? 'Bonne après-midi' : 'Bonsoir';
  el.textContent = greet;
}

async function init() {
  await initDB();
  setWelcomeGreeting();
  newConv();
  await renderConvList();
  await refreshModels();
  checkStatus();
  setInterval(checkStatus, 2000);
}

init();
