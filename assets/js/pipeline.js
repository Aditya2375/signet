// Signet — pipeline: camera, MediaPipe landmarkers, ring buffer, skeleton overlay.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '../../vendor/vision/vision_bundle.mjs';

const DIMS = 150;
const BUFFER_MAX = 96;          // ~3 seconds at 30fps
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
];
const POSE_UPPER = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22];
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
  [9,10],[0,9],[0,10],[15,17],[16,18],[15,19],[16,20],[15,21],[16,22],
];

export class Pipeline {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.hands = null;
    this.pose = null;
    this.buffer = [];
    this.running = false;
    this.paused = false;
    this.showSkeleton = true;
    this.onFrame = null;        // callback(buffer, lastLandmarks, fps)
    this._lastTs = 0;
    this._fpsAcc = 0; this._fpsN = 0; this.fps = 0;
    this._lastLm = null;
    this._sampleMode = null;
  }

  async loadModels(base) {
    const fileset = await FilesetResolver.forVisionTasks(base + '/vendor/vision');
    this.hands = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: base + '/models/hand_landmarker.task' },
      runningMode: 'VIDEO',
      numHands: 2,
    });
    this.pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: base + '/models/pose_landmarker_lite.task' },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
  }

  async startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = stream;
    await this.video.play();
    this.running = true;
    this._loop();
  }

  stopCamera() {
    this.running = false;
    const s = this.video.srcObject;
    if (s) s.getTracks().forEach(t => t.stop());
    this.video.srcObject = null;
  }

  // Replay a recorded landmark sample through the same buffer path.
  async playSample(sample) {
    if (this._sampleMode) return;   // one replay at a time
    this._sampleMode = sample;
    this.running = true;
    let i = 0;
    const step = () => {
      if (!this.running || !this._sampleMode) return;
      const frame = Float32Array.from(sample.frames[i]);
      this.buffer.push(frame);
      if (this.buffer.length > BUFFER_MAX) this.buffer.shift();
      this._lastLm = this._featuresToLandmarks(frame);
      if (this.onFrame) this.onFrame(this.buffer, this._lastLm, 30);
      this._drawOverlay(this._lastLm);
      i++;
      if (i >= sample.frames.length) { this._sampleMode = null; this.running = false; return; }
      setTimeout(step, 33);
    };
    step();
  }

  stopSample() { this._sampleMode = null; this.running = false; }

  _featuresToLandmarks(f) {
    const rh = [], lh = [], pose = [];
    for (let k = 0; k < 21; k++) rh.push({ x: f[2 * k], y: f[2 * k + 1] });
    for (let k = 0; k < 21; k++) lh.push({ x: f[42 + 2 * k], y: f[42 + 2 * k + 1] });
    for (let k = 0; k < 33; k++) pose.push({ x: f[84 + 2 * k], y: f[84 + 2 * k + 1] });
    const clean = arr => arr[0] && arr[0].x > -1.5 ? arr : null;
    return { rh: clean(rh), lh: clean(lh), pose: clean(pose) };
  }

  _loop() {
    if (!this.running) return;
    const v = this.video;
    if (v.readyState >= 2 && !this.paused) {
      const t0 = performance.now();
      const ts = Math.max(this._lastTs + 1, t0);
      let handRes = null, poseRes = null;
      try {
        handRes = this.hands.detectForVideo(v, ts);
        poseRes = this.pose.detectForVideo(v, ts);
      } catch (e) { /* skip frame */ }
      this._lastTs = ts;
      const feat = new Float32Array(DIMS).fill(-2);
      const lm = { rh: null, lh: null, pose: null };
      if (handRes && handRes.landmarks) {
        for (let i = 0; i < handRes.landmarks.length; i++) {
          const label = handRes.handedness[i][0].categoryName; // 'Left' | 'Right'
          const pts = handRes.landmarks[i];
          const off = label === 'Right' ? 0 : 42;
          for (let k = 0; k < 21; k++) {
            feat[off + 2 * k] = pts[k].x;
            feat[off + 2 * k + 1] = pts[k].y;
          }
          if (label === 'Right') lm.rh = pts; else lm.lh = pts;
        }
      }
      if (poseRes && poseRes.landmarks && poseRes.landmarks[0]) {
        const pts = poseRes.landmarks[0];
        for (let k = 0; k < 33; k++) {
          feat[84 + 2 * k] = pts[k].x;
          feat[84 + 2 * k + 1] = pts[k].y;
        }
        lm.pose = pts;
      }
      this.buffer.push(feat);
      if (this.buffer.length > BUFFER_MAX) this.buffer.shift();
      this._lastLm = lm;
      this._drawOverlay(lm);
      const dt = performance.now() - t0;
      this._fpsAcc += dt; this._fpsN++;
      if (this._fpsN >= 20) {
        this.fps = Math.round(1000 / (this._fpsAcc / this._fpsN + 1e-6));
        this._fpsAcc = 0; this._fpsN = 0;
      }
      if (this.onFrame) this.onFrame(this.buffer, lm, this.fps);
    }
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      this.video.requestVideoFrameCallback(() => this._loop());
    } else {
      requestAnimationFrame(() => this._loop());
    }
  }

  clearBuffer() { this.buffer = []; }

  _drawHand(ctx, pts, color, w, h) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(pts[a].x * w, pts[a].y * h);
      ctx.lineTo(pts[b].x * w, pts[b].y * h);
    }
    ctx.stroke();
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  _drawOverlay(lm) {
    const v = this.video, c = this.canvas;
    if (this._sampleMode) {
      c.width = c.clientWidth; c.height = c.clientHeight;
    } else if (v.videoWidth && (c.width !== v.videoWidth || c.height !== v.videoHeight)) {
      c.width = v.videoWidth; c.height = v.videoHeight;
    }
    const { ctx } = this;
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    if (!this.showSkeleton || !lm) return;
    if (lm.pose) {
      const col = 'rgba(237,232,223,0.28)';
      ctx.strokeStyle = col; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (const [a, b] of POSE_CONNECTIONS) {
        ctx.moveTo(lm.pose[a].x * w, lm.pose[a].y * h);
        ctx.lineTo(lm.pose[b].x * w, lm.pose[b].y * h);
      }
      ctx.stroke();
      ctx.fillStyle = col;
      for (const i of POSE_UPPER) {
        ctx.beginPath();
        ctx.arc(lm.pose[i].x * w, lm.pose[i].y * h, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (lm.rh) this._drawHand(ctx, lm.rh, '#d4a24c', w, h);
    if (lm.lh) this._drawHand(ctx, lm.lh, '#ede8df', w, h);
  }
}
