// Signet — entry point: UI wiring, word explorer, model card.
import { Pipeline } from './pipeline.js';
import { Recognizer } from './infer.js';

const $ = id => document.getElementById(id);
const video = $('cam');
const canvas = $('overlay');
const stage = document.querySelector('.stage');
const stageIdle = $('stageIdle');
const stageHud = $('stageHud');

const pipeline = new Pipeline(video, canvas);
const recognizer = new Recognizer();

let engineReady = false;
let lastInferAt = 0;
let handsGone = 0;

/* ---------------- boot ---------------- */
async function boot() {
  setNote('Loading labels...');
  try {
    const labels = await (await fetch('models/labels.json')).json();
    buildWordGrid(labels);
  } catch (e) {
    setNote('Could not load the word list. Serve the page over http(s), not file://.');
    return;
  }
  try {
    const meta = await (await fetch('models/meta.json')).json();
    fillModelCard(meta);
  } catch (e) { /* model card keeps placeholders */ }
  setNote('Engine idle. Enable the camera or replay a sample to load the models.');
}

function setNote(t) { $('engineNote').textContent = t; }

async function ensureEngine() {
  if (engineReady) return;
  setNote('Loading landmarkers and the recognition model (about 20 MB, once)...');
  await pipeline.loadModels('.');
  await recognizer.load('models/signet-100.onnx', 'models/labels.json');
  engineReady = true;
  setNote('Engine ready. MediaPipe landmarkers + transformer running locally in WebAssembly. The video stream never leaves this device.');
}

/* ---------------- live loop ---------------- */
pipeline.onFrame = async (buffer, lm, fps) => {
  const hasHands = !!(lm && (lm.rh || lm.lh));
  $('hudFps').textContent = fps ? `${fps} fps` : '-- fps';
  $('hudState').textContent = hasHands ? 'hands in frame' : 'no hands';
  $('hudState').classList.toggle('live', hasHands);
  // Without hands the buffer is just noise; after ~0.5 s of no hands, stop reading
  // and drop the smoothing state so an old sign cannot echo onto the tape.
  handsGone = hasHands ? 0 : handsGone + 1;
  if (handsGone === 15) {
    recognizer.reset();
    pipeline.clearBuffer();
    renderIdle();
  }
  if (handsGone >= 15) return;
  const now = performance.now();
  if (now - lastInferAt < 160) return;   // ~6 inference ticks/s
  lastInferAt = now;
  if (!engineReady) return;
  const out = await recognizer.infer(buffer);
  if (out) renderPrediction(out);
};

function renderIdle() {
  $('nowWord').textContent = '\u2014';
  $('confText').textContent = 'confidence --';
  $('confFill').style.width = '0%';
  $('topK').innerHTML = '';
}

function renderPrediction({ top, emitted }) {
  const best = top[0];
  if (best.prob >= 0.3) {
    $('nowWord').textContent = best.word.toLowerCase();
    $('confText').textContent = `confidence ${best.prob.toFixed(2)}`;
  } else {
    $('nowWord').textContent = '—';
    $('confText').textContent = 'confidence --';
  }
  $('confFill').style.width = `${Math.round(best.prob * 100)}%`;
  const ul = $('topK');
  ul.innerHTML = '';
  for (const t of top.slice(1)) {
    const li = document.createElement('li');
    const b = document.createElement('b');
    b.textContent = t.word.toLowerCase();
    const s = document.createElement('span');
    s.textContent = t.prob.toFixed(2);
    li.append(b, s);
    ul.appendChild(li);
  }
  if (emitted) addToTape(emitted.word);
}

function addToTape(word) {
  const tape = $('tape');
  const empty = tape.querySelector('.tape-empty');
  if (empty) empty.remove();
  const chip = document.createElement('span');
  chip.className = 'tape-word';
  chip.textContent = word.toLowerCase();
  tape.appendChild(chip);
  tape.scrollTop = tape.scrollHeight;
}

/* ---------------- controls ---------------- */
$('btnStart').addEventListener('click', async () => {
  $('btnStart').disabled = true;
  $('btnStart').textContent = 'Starting...';
  try {
    await ensureEngine();
    await pipeline.startCamera();
    stageIdle.style.display = 'none';
    stageHud.hidden = false;
    $('btnPause').disabled = false;
    $('btnClear').disabled = false;
    $('btnStop').disabled = false;
  } catch (e) {
    console.error(e);
    $('btnStart').disabled = false;
    $('btnStart').textContent = 'Enable camera';
    setNote('Could not start: ' + (e && e.message ? e.message : e) + '. Check camera permission and that you are on https or localhost.');
  }
});

$('btnSample').addEventListener('click', async () => {
  $('btnSample').disabled = true;
  try {
    await ensureEngine();
    const samples = await (await fetch('models/samples.json')).json();
    const s = samples[Math.floor(Math.random() * samples.length)];
    pipeline.stopCamera();   // a live feed and a replay must never mix in one buffer
    pipeline.stopSample();
    stageIdle.style.display = 'none';
    stageHud.hidden = false;
    pipeline.clearBuffer();
    recognizer.reset();
    handsGone = 0;
    setNote(`Replaying a recorded sign — the answer is "${s.label}". Landmark data only; no video.`);
    await pipeline.playSample(s);
  } catch (e) {
    console.error(e);
    setNote('Sample replay unavailable.');
  } finally {
    $('btnSample').disabled = false;
  }
});

$('btnPause').addEventListener('click', () => {
  pipeline.paused = !pipeline.paused;
  $('btnPause').textContent = pipeline.paused ? 'Resume' : 'Pause';
});

$('btnClear').addEventListener('click', () => {
  $('tape').innerHTML = '<span class="tape-empty">Words you sign will land here.</span>';
  pipeline.clearBuffer();
  recognizer.reset();
});

$('btnStop').addEventListener('click', () => {
  pipeline.stopCamera();
  pipeline.stopSample();
  stageIdle.style.display = 'flex';
  $('btnStart').disabled = false;
  $('btnStart').textContent = 'Enable camera';
  $('btnPause').disabled = true;
  $('btnPause').textContent = 'Pause';
  pipeline.paused = false;
  $('btnClear').disabled = true;
  $('btnStop').disabled = true;
  stageHud.hidden = true;
});

$('ctlThresh').addEventListener('input', e => {
  const v = e.target.value / 100;
  recognizer.threshold = v;
  $('ctlThreshVal').textContent = v.toFixed(2);
});

$('ctlMirror').addEventListener('change', e => {
  stage.classList.toggle('mirrored', e.target.checked);
});
stage.classList.add('mirrored');

$('ctlSkeleton').addEventListener('change', e => {
  pipeline.showSkeleton = e.target.checked;
  if (!e.target.checked) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
});

/* ---------------- word explorer ---------------- */
function buildWordGrid(labels) {
  const grid = $('wordsGrid');
  grid.innerHTML = '';
  labels.forEach((w, i) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `https://www.signasl.org/sign/${encodeURIComponent(w.toLowerCase())}`;
    a.target = '_blank';
    a.rel = 'noopener';
    a.append(document.createTextNode(w.toLowerCase() + ' '));
    const num = document.createElement('span');
    num.textContent = String(i + 1).padStart(2, '0');
    a.appendChild(num);
    li.appendChild(a);
    grid.appendChild(li);
  });
  $('wordFilter').addEventListener('input', e => {
    const q = e.target.value.trim().toLowerCase();
    let n = 0;
    grid.querySelectorAll('li').forEach((li, i) => {
      const show = !q || labels[i].toLowerCase().includes(q);
      li.style.display = show ? '' : 'none';
      if (show) n++;
    });
    $('wordCount').textContent = `${n} word${n === 1 ? '' : 's'}`;
  });
}

/* ---------------- model card ---------------- */
function fillModelCard(m) {
  if (m.params) $('mcParams').textContent = m.params.toLocaleString('en-US');
  const pct = x => (x * 100).toFixed(1) + '%';
  if (m.test_top1 != null) $('mcTest1').textContent = pct(m.test_top1);
  if (m.test_top5 != null) $('mcTest5').textContent = pct(m.test_top5);
  if (m.val_top1 != null) $('mcVal1').textContent = `${pct(m.val_top1)} (top-5 ${pct(m.val_top5)})`;
}

boot();
