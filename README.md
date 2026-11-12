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
  (33 body points) turn each video frame into 150 coordinates. Missing detections are
  encoded explicitly (-2) so the model can learn to distrust incomplete frames.
- **Model.** A small transformer encoder with a learned positional encoding and attention pooling
  over time. Trained on the official WLASL-100 split (1,442 / 338 / 258 clips) with temporal
  resampling, horizontal flip, spatial jitter and Gaussian noise as augmentation.
- **Inference.** ONNX Runtime Web in a single WASM thread. One forward pass every ~160 ms over the
  sliding buffer; predictions are EMA-smoothed and a word is only emitted after three consecutive
  ticks over the threshold, with a per-word cooldown.

## Honest numbers

Measured on the official WLASL-100 test split (258 held-out clips) with the exact
model file this site serves:

| metric | value |
|---|---|
| test top-1 | 71.7% (95% CI 65.9–76.9%, n=258) |
| test top-5 | 88.4% |
| validation top-1 | 73.7% |
| validation top-5 | 91.1% |
| parameters | 949,540 |

The same numbers ship in `models/meta.json`, which the site reads to fill its model card, so the
page can never drift from the measured values.

**What these numbers measure.** The evaluation feeds the model landmark sequences that were
pre-extracted from the dataset videos with MediaPipe (see Data and licenses). It does not run the
browser's own landmark detectors over raw video, so it is a dataset-landmark number, not a
webcam-pipeline number; a webcam-pipeline evaluation requires re-extracting landmarks from the
raw test videos with the exact browser models and is tracked as follow-up work. Live accuracy on
your webcam will differ regardless: lighting, framing, camera quality and signing speed all
matter, and dataset clips are kinder than a bedroom camera. Top-5 is always visible for a reason.

**Method notes.** Single training seed (379). The reported checkpoint is the best-validation
checkpoint over 220 epochs, so validation numbers are optimistically biased; test numbers come
from the untouched test split, evaluated once for the final metrics and again for the parity and
robustness analyses below. No model selection was done on the test set.

**ONNX parity.** The served ONNX model was checked against the PyTorch checkpoint on all 258
test clips: max absolute logit difference 0.00001, top-1 agreement 258/258. The site runs
exactly the model these numbers describe.

**Robustness, measured.** Same test set, inputs perturbed the way a webcam perturbs them:

| perturbation | top-1 | top-5 |
|---|---|---|
| none | 71.7% | 88.4% |
| mirrored camera (left/right swapped) | 73.3% | 88.4% |
| left hand undetected for the whole clip | 1.6% | 6.6% |
| pose track lost for the whole clip | 0.8% | 4.7% |
| both hands undetected | 1.2% | 6.2% |

The mirrored number says the model does not care which hand the detectors label Left or Right
(flip augmentation did its job). The other three rows are the honest weak spot: the training data
almost always contains full pose and both hands, so the model collapses when a track goes missing
for a long stretch. Brief detection dropouts are bridged by interpolation, but a hand held out of
frame for a whole sign is out of distribution. Training with landmark dropout to close this gap
is in `training/train2.py`.

## What it is not

- **Isolated-word recognition, not continuous signing.** One sign at a time, with a beat between
  signs; no grammar, no fingerspelling, no facial markers, no sentence-level anything.
- **ASL only.** Not Indian Sign Language, not BSL, not any other sign language.
- **Not an accessibility tool.** It is a learning project. For real communication, use a human
  interpreter.
- **Not broadly validated.** 100 words, one dataset, one seed. Accuracy across skin tones,
  lighting conditions, camera angles and signing styles is untested; expect variance.

## Privacy

Static site, no backend, no analytics, no cookies, self-hosted fonts. The camera feed is
processed frame by frame in the browser and immediately discarded. Nothing about you is recorded,
stored or transmitted. The host (GitHub Pages) sees ordinary page requests, like any web server.
Once the page and the models have loaded, you can pull the network cable and it keeps working.

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
python3 -m http.server 8080   # from the repository root
# open http://localhost:8080
```

## Training your own

`training/` has the full pipeline: parse the landmark CSVs (`prep.py`), train (`train.py`),
export ONNX and evaluate. CPU-only is fine; a full 220-epoch run takes about half an hour on a modern multi-core CPU.
You need the WLASL landmark inputs (see the data section above) — raw WLASL videos are not
required for this training path.

## Stack

Plain HTML/CSS/JS with ES modules, no build step, no framework. MediaPipe Tasks Vision,
ONNX Runtime Web. Everything vendored, fonts included; the page makes no third-party requests.
