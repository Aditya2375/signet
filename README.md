# Signet

Real-time, word-level American Sign Language recognition that runs entirely in the browser.

Open the page, allow the camera, sign one of a hundred ASL words, and watch it land on the
transcript tape. There is no server: the landmark detectors and the neural network download once
and run in WebAssembly on your device, and the video stream never leaves the tab.

**Live:** https://aditya2375.github.io/signet/

## What it does

- Recognizes 100 isolated ASL signs (the WLASL-100 vocabulary: book, drink, computer, go, who...)
  from a live webcam feed, at interactive frame rates, on CPU.
- Shows its work: a live skeleton overlay of what the detectors see, the current best guess with a
  confidence bar, and the four runners-up, so a wrong answer is never a silent one.
- Works without a camera too: eight recorded signs ship as landmark-only samples that replay
  through the exact same pipeline.
- Every decision threshold is yours: a slider trades missed signs against wrong ones in real time.

## How it works

```
webcam ──► MediaPipe hand + pose landmarkers ──► 75 points × (x, y) per frame
       ──► ring buffer (last ~3 s) resampled to 48 frames
       ──► transformer encoder (3 layers, width 192, ~950k params, ONNX, WASM)
       ──► softmax over 100 signs ──► temporal smoothing + threshold state machine ──► tape
```

- **Landmarks.** MediaPipe's hand landmarker (2 hands × 21 points) and pose landmarker
  (33 upper-body points) turn each video frame into 150 coordinates. Missing detections are
  encoded explicitly (-2) so the model can learn to distrust incomplete frames.
- **Model.** A small transformer encoder with a learned positional encoding and attention pooling
  over time. Trained on the official WLASL-100 split (1,442 / 338 / 258 clips) with temporal
  resampling, horizontal flip, spatial jitter and Gaussian noise as augmentation.
- **Inference.** ONNX Runtime Web in a single WASM thread. One forward pass every ~160 ms over the
  sliding buffer; predictions are EMA-smoothed and a word is only emitted after three consecutive
  ticks over the threshold, with a per-word cooldown.

## Honest numbers

Measured on the official WLASL-100 test split (258 clips, unseen signers included), the exact
model file this site serves:

| metric | value |
|---|---|
| test top-1 | 71.7% |
| test top-5 | 88.4% |
| validation top-1 | 73.7% |
| validation top-5 | 91.1% |
| parameters | 949,540 |

The same numbers ship in `models/meta.json`, which the site reads to fill its model card, so the
page can never drift from the measured values.

The numbers come from `training/train.py`, which prints and writes them after the final epoch.
Live accuracy on your webcam will differ: lighting, framing, camera quality and signing speed all
matter, and dataset clips are kinder than a bedroom camera. Top-5 is always visible for a reason.

## What it is not

- **Not sentence-level.** Isolated signs only; no grammar, no fingerspelling, no facial markers.
- **ASL only.** Not Indian Sign Language, not BSL, not any other sign language.
- **Not an accessibility tool.** It is a learning project. For real communication, use a human
  interpreter.

## Privacy

Static site, no backend, no analytics, no cookies. The camera feed is processed frame by frame in
the browser and immediately discarded. Nothing is recorded, stored or transmitted. Load the page,
pull the network cable, it keeps working.

## Data and licenses

- Code: MIT (see LICENSE).
- Training data: [WLASL](https://github.com/dxli94/WLASL) (WACV 2020, Best Paper Honourable
  Mention), used under the Computational Use of Data Agreement (C-UDA-1.0): academic and
  computational use only, **no commercial use**. Landmark sequences pre-extracted with MediaPipe
  as published by [Cristina Luna et al.](https://github.com/cristinalunaj/InterpretableTransformer_SignLanguage).
- Runtime models: MediaPipe hand/pose landmarkers (Google, Apache-2.0 task bundles), ONNX Runtime
  Web (MIT), MediaPipe Tasks Vision (Apache-2.0).
- Dictionary links: [SignASL](https://www.signasl.org/).

## Running locally

Any static file server works; the model fetches need `http(s)`, not `file://`:

```bash
cd site   # this folder
python3 -m http.server 8080
# open http://localhost:8080
```

## Training your own

`training/` has the full pipeline: parse the landmark CSVs (`prep.py`), train (`train.py`),
export ONNX and evaluate. CPU-only is fine; a full run takes well under an hour on one core.
You need the WLASL landmark inputs (see the data section above) — raw WLASL videos are not
required for this training path.

## Stack

Plain HTML/CSS/JS with ES modules, no build step, no framework. MediaPipe Tasks Vision,
ONNX Runtime Web. Everything vendored; the only external request is Google Fonts.
