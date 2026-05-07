/**
 * ASL Recognizer — Frontend Application
 * Webcam capture → FastAPI /api/predict → render result + skeleton
 * Fixes: N/M disambiguation + lighting warning
 */

const API_BASE    = '';
const PREDICT_URL = `${API_BASE}/api/predict`;
const HEALTH_URL  = `${API_BASE}/api/health`;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ── State ──────────────────────────────────────────────────────────
let stream         = null;
let isRunning      = false;
let rafId          = null;
let captureCanvas  = null;
let captureCtx     = null;
let currentWord    = '';
let wordHistory    = [];
let lastLetter     = '';
let letterHoldMs   = 0;
const HOLD_MS      = 1200;
let lastCommitTime = 0;
let lightingWarned = false;
let lightingTick   = 0;

// ── DOM refs ────────────────────────────────────────────────────────
const video          = document.getElementById('video');
const canvasOverlay  = document.getElementById('canvas-overlay');
const overlayCtx     = canvasOverlay ? canvasOverlay.getContext('2d') : null;
const skeletonCanvas = document.getElementById('skeleton-canvas');
const skeletonCtx    = skeletonCanvas ? skeletonCanvas.getContext('2d') : null;
const predLetter     = document.getElementById('pred-letter');
const predConf       = document.getElementById('pred-conf');
const predTime       = document.getElementById('pred-time');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');
const startBtn       = document.getElementById('btn-start');
const stopBtn        = document.getElementById('btn-stop');
const wordDisplay    = document.getElementById('word-display');
const wordHistoryEl  = document.getElementById('word-history');
const confBars       = document.getElementById('conf-bars');
const placeholder    = document.getElementById('camera-placeholder');
const predOverlay    = document.getElementById('prediction-overlay');

// ── Health check ────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(HEALTH_URL);
    const d = await r.json();
    if (!d.model_loaded) {
      setStatus('error', 'Model not loaded — run python model/train.py');
      return false;
    }
    setStatus('ready', 'Model ready · click Start');
    return true;
  } catch {
    setStatus('error', 'API offline — run uvicorn api.main:app');
    return false;
  }
}

// ── Camera ──────────────────────────────────────────────────────────
async function startCamera() {
  const healthy = await checkHealth();
  if (!healthy) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await video.play();

    captureCanvas        = document.createElement('canvas');
    captureCanvas.width  = 640;
    captureCanvas.height = 480;
    captureCtx = captureCanvas.getContext('2d');

    if (placeholder) placeholder.classList.add('hidden');
    if (predOverlay) predOverlay.classList.remove('hidden');
    startBtn.disabled = true;
    stopBtn.disabled  = false;
    isRunning = true;
    setStatus('live', 'Live · detecting');
    requestAnimationFrame(captureLoop);
  } catch (e) {
    setStatus('error', 'Camera denied — allow access in browser');
    console.error(e);
  }
}

function stopCamera() {
  isRunning = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  if (placeholder) placeholder.classList.remove('hidden');
  if (predOverlay) predOverlay.classList.add('hidden');
  hideLightingWarning();
  setStatus('ready', 'Stopped');
  clearOverlay();
  clearSkeleton();
}

// ── Capture loop ─────────────────────────────────────────────────────
let lastSendTime = 0;
const SEND_INTERVAL_MS = 120;

async function captureLoop(timestamp) {
  if (!isRunning) return;
  rafId = requestAnimationFrame(captureLoop);
  if (timestamp - lastSendTime < SEND_INTERVAL_MS) return;
  lastSendTime = timestamp;
  if (!video.videoWidth) return;

  captureCtx.save();
  captureCtx.translate(640, 0);
  captureCtx.scale(-1, 1);
  captureCtx.drawImage(video, 0, 0, 640, 480);
  captureCtx.restore();

  // Lighting check every ~3 seconds
  lightingTick++;
  if (lightingTick % 25 === 0) checkLighting();

  captureCanvas.toBlob(async (blob) => {
    if (!blob) return;
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');
    try {
      const res  = await fetch(PREDICT_URL, { method: 'POST', body: form });
      if (!res.ok) return;
      const data = await res.json();
      handlePrediction(data);
    } catch { /* network hiccup */ }
  }, 'image/jpeg', 0.7);
}

// ── N/M Disambiguation ───────────────────────────────────────────────
// Uses landmark geometry to distinguish N (2 fingers over thumb)
// from M (3 fingers over thumb) when model confidence is below 92%
function disambiguateNM(letter, confidence, landmarks) {
  if (!landmarks || landmarks.length < 21) return letter;
  if (letter !== 'N' && letter !== 'M')    return letter;
  if (confidence > 0.92) return letter; // trust the model at high confidence

  // Check if fingertips are curled: tip.y > MCP.y (lower on screen = curled)
  const indexCurled  = landmarks[8].y  > landmarks[5].y;
  const middleCurled = landmarks[12].y > landmarks[9].y;
  const ringCurled   = landmarks[16].y > landmarks[13].y;
  const curled = [indexCurled, middleCurled, ringCurled].filter(Boolean).length;

  if (curled >= 3) return 'M'; // 3 fingers over thumb = M
  if (curled <= 2) return 'N'; // 2 fingers over thumb = N
  return letter;
}

// ── Lighting check ───────────────────────────────────────────────────
function checkLighting() {
  if (!captureCtx || !captureCanvas) return;
  const imageData = captureCtx.getImageData(0, 0, captureCanvas.width, captureCanvas.height);
  const data = imageData.data;
  let brightness = 0;
  let samples    = 0;
  for (let i = 0; i < data.length; i += 4 * 40) {
    brightness += 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
    samples++;
  }
  const avg = brightness / samples;
  if (avg < 60) {
    showLightingWarning('⚠ Low lighting — move to a brighter area for best accuracy');
    if (!lightingWarned) {
      setStatus('live', 'Live · low lighting ⚠');
      lightingWarned = true;
    }
  } else {
    hideLightingWarning();
    if (lightingWarned) {
      setStatus('live', 'Live · detecting');
      lightingWarned = false;
    }
  }
}

function showLightingWarning(msg) {
  let el = document.getElementById('lighting-warning');
  if (!el) {
    el = document.createElement('div');
    el.id = 'lighting-warning';
    el.style.cssText = `
      position:absolute; bottom:5rem; left:1rem; right:1rem;
      background:rgba(239,159,39,0.15);
      border:1px solid rgba(239,159,39,0.4);
      border-radius:8px; padding:8px 12px;
      font-size:12px; color:#EF9F27;
      backdrop-filter:blur(6px); z-index:10;
    `;
    const wrap = video.parentElement;
    if (wrap) wrap.appendChild(el);
  }
  el.textContent = msg;
  el.style.display = 'block';
}

function hideLightingWarning() {
  const el = document.getElementById('lighting-warning');
  if (el) el.style.display = 'none';
}

// ── Handle prediction ─────────────────────────────────────────────────
function handlePrediction(data) {
  if (!data.hand_detected) {
    predLetter.textContent = '—';
    predConf.textContent   = '—';
    predTime.textContent   = `${data.time_ms} ms`;
    clearOverlay();
    clearSkeleton();
    lastLetter = '';
    return;
  }

  // N/M fix: apply geometric disambiguation when confidence is borderline
  const rawLetter = data.letter || '?';
  const letter    = disambiguateNM(rawLetter, data.confidence || 0, data.landmarks);
  const conf      = Math.round((data.confidence || 0) * 100);

  predLetter.textContent = letter;
  predConf.textContent   = `${conf}%`;
  predTime.textContent   = `${data.time_ms} ms`;

  if (data.landmarks && data.landmarks.length === 21) {
    drawSkeleton(data.landmarks);
  }

  renderConfBars(data.top5 || []);

  if (letter !== '?' && conf >= 60) {
    const now = Date.now();
    if (letter === lastLetter) {
      letterHoldMs += SEND_INTERVAL_MS;
      if (letterHoldMs >= HOLD_MS && now - lastCommitTime > HOLD_MS + 200) {
        commitLetter(letter);
        lastCommitTime = now;
        letterHoldMs   = 0;
      }
    } else {
      lastLetter   = letter;
      letterHoldMs = 0;
    }
  }
}

function commitLetter(letter) {
  currentWord += letter;
  renderWord();
}

function renderWord() {
  if (!wordDisplay) return;
  if (currentWord) {
    wordDisplay.textContent = currentWord;
    wordDisplay.classList.remove('empty');
  } else {
    wordDisplay.textContent = 'Hold a sign to build a word…';
    wordDisplay.classList.add('empty');
  }
}

// ── Skeleton drawing ─────────────────────────────────────────────────
function drawSkeleton(landmarks) {
  if (!skeletonCtx || !skeletonCanvas) return;
  const W = skeletonCanvas.width;
  const H = skeletonCanvas.height;
  skeletonCtx.clearRect(0, 0, W, H);
  const pts = landmarks.map(lm => ({ x: lm.x * W, y: lm.y * H }));

  skeletonCtx.strokeStyle = 'rgba(0,229,195,0.4)';
  skeletonCtx.lineWidth   = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    skeletonCtx.beginPath();
    skeletonCtx.moveTo(pts[a].x, pts[a].y);
    skeletonCtx.lineTo(pts[b].x, pts[b].y);
    skeletonCtx.stroke();
  }
  pts.forEach((pt, i) => {
    const isTip = [4,8,12,16,20].includes(i);
    skeletonCtx.beginPath();
    skeletonCtx.arc(pt.x, pt.y, isTip ? 5 : 3.5, 0, Math.PI * 2);
    skeletonCtx.fillStyle = isTip ? '#00e5c3' : 'rgba(0,229,195,0.6)';
    skeletonCtx.fill();
  });
}

function clearSkeleton() {
  if (!skeletonCtx || !skeletonCanvas) return;
  skeletonCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
}

function clearOverlay() {
  if (overlayCtx && canvasOverlay)
    overlayCtx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
}

// ── Confidence bars ──────────────────────────────────────────────────
function renderConfBars(top5) {
  if (!confBars) return;
  confBars.innerHTML = '';
  top5.forEach((item, i) => {
    const pct = Math.round((item.prob || 0) * 100);
    const div = document.createElement('div');
    div.className = 'conf-bar-item';
    div.innerHTML = `
      <div class="conf-bar-header">
        <span class="conf-bar-letter">${item.letter}</span>
        <span class="conf-bar-pct">${pct}%</span>
      </div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill ${i > 0 ? 'secondary' : ''}" style="width:${pct}%"></div>
      </div>`;
    confBars.appendChild(div);
  });
}

// ── Word builder controls ────────────────────────────────────────────
function wordBackspace() { currentWord = currentWord.slice(0, -1); renderWord(); }
function wordSpace()     { if (currentWord && currentWord[currentWord.length-1] !== ' ') { currentWord += ' '; renderWord(); } }
function wordCommit()    { const w = currentWord.trim(); if (!w) return; wordHistory.unshift(w); if (wordHistory.length > 8) wordHistory.pop(); currentWord = ''; renderWord(); renderHistory(); }
function wordClear()     { currentWord = ''; renderWord(); }

function renderHistory() {
  if (!wordHistoryEl) return;
  wordHistoryEl.innerHTML = wordHistory.map(w => `<span class="word-chip">${w}</span>`).join('');
}

// ── Status ───────────────────────────────────────────────────────────
function setStatus(state, text) {
  if (statusDot) {
    statusDot.className = 'status-dot';
    if (state === 'live')  statusDot.classList.add('live');
    if (state === 'error') statusDot.classList.add('error');
  }
  if (statusText) statusText.textContent = text;
}

// ── Init ─────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  renderWord();
  checkHealth();
  if (skeletonCanvas) { skeletonCanvas.width = 220; skeletonCanvas.height = 220; }
  if (canvasOverlay && video) {
    video.addEventListener('loadedmetadata', () => {
      canvasOverlay.width  = video.videoWidth  || 640;
      canvasOverlay.height = video.videoHeight || 480;
    });
  }
  if (startBtn) startBtn.addEventListener('click', startCamera);
  if (stopBtn)  stopBtn.addEventListener('click',  stopCamera);
  const btnBackspace = document.getElementById('btn-backspace');
  const btnSpace     = document.getElementById('btn-space');
  const btnSave      = document.getElementById('btn-save');
  const btnClear     = document.getElementById('btn-clear');
  if (btnBackspace) btnBackspace.addEventListener('click', wordBackspace);
  if (btnSpace)     btnSpace.addEventListener('click',     wordSpace);
  if (btnSave)      btnSave.addEventListener('click',      wordCommit);
  if (btnClear)     btnClear.addEventListener('click',     wordClear);
});