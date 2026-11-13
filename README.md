# Signet

Word-level American Sign Language recognition that runs in the browser.

**Live accuracy on a webcam is unmeasured.** The numbers below come from landmark sequences that
were pulled from the WLASL test videos ahead of time, not from running this page's own hand and
pose detectors on a camera feed. Read them as dataset numbers.

Open the page, allow the camera, sign one of 100 ASL words, and it shows up on a running
transcript. There is no server. The landmark models and the network download once and run in
WebAssembly on your device, and the video never leaves the tab.

## What it does

- Tries to recognize 100 isolated ASL signs (the WLASL-100 vocabulary: book, drink, computer, go,
  who...) from a webcam at interactive speed on CPU. How well that works on a real camera has not
  been measured, see above.
- Draws the skeleton the detectors see, the top guess with a confidence bar, and the next four
  guesses, so a wrong answer is visible as one.
- Works without a camera. Eight recorded signs ship as landmark-only samples and replay through the
  same pipeline. This is a demo, not a test: it got 5 of the 8 right, which says nothing about
  accuracy.
- Has a threshold slider that trades missed signs against wrong ones.

## How it works

```
webcam ──► MediaPipe hand + pose landmarkers ──► 75 points × (x, y) per frame
       ──► ring buffer (last ~3 s) resampled to 48 frames
       ──► transformer encoder (3 layers, width 192, ~950k params, ONNX, WASM)
       ──► softmax over 100 signs ──► smoothing + threshold state machine ──► tape
```

- **Landmarks.** MediaPipe's hand landmarker (2 hands, 21 points each) and pose landmarker (33
  body points) turn each frame into 150 numbers. A missing detection is marked explicitly (-2) so
  the model can learn not to trust an incomplete frame.
- **Model.** A small transformer encoder with learned positional encoding and attention pooling
  over time. Trained on the official WLASL-100 split (1,442 train / 338 validation / 258 test
  clips) with temporal resampling, horizontal flip, spatial jitter and Gaussian noise.
- **Inference.** ONNX Runtime Web, one WASM thread. One forward pass about every 160 ms over the
  sliding buffer. Predictions are smoothed, and a word is only emitted after three ticks in a row
  over the threshold, with a cooldown per word.

## Numbers

All of these are on the 258 official WLASL-100 test clips, using the exact model file the site
serves, with landmarks extracted from the dataset videos in advance.

| metric | value |
|---|---|
| test top-1 | 65.9% (95% CI 59.9-71.4%, n=258) |
| test top-5 | 86.0% |
| validation top-1 | 70.7% |
| validation top-5 | 88.8% |
| parameters | 949,540 |

The same values are in `models/meta.json`, which the site reads to fill its model card.

**What they don't cover.** Nothing here runs the browser's own detectors over raw video. A
webcam evaluation would mean re-extracting landmarks from the raw test videos with the exact
browser models, and I haven't done that. Real cameras will differ: lighting, framing and signing
speed all matter, and dataset clips are easier than a bedroom webcam. That is why top-5 is always
on screen.

**Method notes.** One training seed (379). The checkpoint is the best-validation one over 220
epochs, so validation numbers lean optimistic. The test set was used for the final metrics and
again for the parity and perturbation checks below.

**Why this model and not the first one.** I trained two. The first (v1) scored 71.7% top-1 on the
test set, but when a landmark track was missing for a whole clip it fell apart: 1.6% with the
left hand undetected, 0.8% with the pose track lost. Webcams drop tracks all the time. The second
(v2, the one served) is the same architecture retrained with landmark dropout
(`training/train2.py`). I picked v2 after looking at those perturbation results, and those
results are test-set numbers, so the choice was not made on validation alone. v2 has a lower
point estimate on the clean test set (65.9% against 71.7%). With one seed and overlapping
confidence intervals, that gap is within noise. v1 is not shipped.

**ONNX parity.** The ONNX model matches the PyTorch checkpoint on all 258 test clips: max logit
difference 8e-6, top-1 agreement 258/258.

**Perturbations.** Same test set, with inputs damaged the way a webcam damages them. These are
test-set numbers, not webcam numbers.

| perturbation | v1 top-1 | v2 top-1 | v2 top-5 |
|---|---|---|---|
| none | 71.7% | 65.9% | 86.0% |
| mirrored (left/right swapped) | 73.3% | 65.9% | 86.0% |
| left hand undetected for the whole clip | 1.6% | 19.4% | 46.9% |
| pose track lost for the whole clip | 0.8% | 51.6% | 79.1% |
| both hands undetected | 1.2% | 6.6% | 18.2% |

The mirrored run for v2 comes from `training/eval_mirror.py` on the served model (output in `training/mirror.json`). Mirrored top-1 is 65.9% and top-5 is 86.0%, the same as unmirrored. Underneath that, the predicted label changes on 50 of 258 clips, 15 clips are right only unmirrored and 15 are right only mirrored, so the net is zero. On average it behaves as hand-agnostic, but not clip by clip.

`eval_mirror.py` and `eval_robust.py` need the extracted landmark sequences (`test.npz`, built by `prep.py` from the published WLASL landmark CSVs). Those are not in this repo, so `mirror.json` and `robust.json` can't be regenerated from the repo alone. The numbers in them are as produced on my machine.

The weak spot is missing hands. With one hand lost, v2 gets 19.4% right. With both gone it gets
6.6%. v2 handles a lost pose track much better than v1 but it still struggles when hands
disappear, which is why the live loop pauses when it sees none.

## What it is not

- **Isolated words, not continuous signing.** One sign at a time with a beat between signs. No
  grammar, no fingerspelling, no facial markers.
- **ASL only.** Not Indian Sign Language, not BSL, not any other sign language.
- **Not an accessibility tool.** It is a learning project. For real communication use a human
  interpreter.
- **Narrow testing.** 100 words, one dataset, one seed. Accuracy across skin tones, lighting,
  camera angles and signing styles has not been tested.

## Privacy

Static site, no backend, no analytics, no cookies, fonts hosted in the repo. The camera feed is
processed frame by frame in the browser and discarded. Nothing about you is recorded or sent
anywhere. The host sees ordinary page requests like any web server. Once the page and models
have loaded it keeps working offline.

## Data and licenses

- Code: MIT (see LICENSE).
- Training data: [WLASL](https://github.com/dxli94/WLASL) (Li et al., WACV 2020), released under
  the Computational Use of Data Agreement (C-UDA-1.0). Landmark sequences were pre-extracted with
  MediaPipe and published by
  [Cristina Luna et al.](https://github.com/cristinalunaj/InterpretableTransformer_SignLanguage).
- The trained model is a Result of that data under the agreement (s.2.2 and s.5.5). It is
  attributed to WLASL as its training data. See [models/NOTICE.md](models/NOTICE.md). This is
  a plain reading of the agreement, not legal advice.
- `models/samples.json` holds landmark coordinates derived from WLASL test videos. That is Data
  under the agreement, so it keeps the attribution and is for computational use only.
- Runtime: MediaPipe hand and pose landmarker bundles (Google, Apache-2.0), MediaPipe Tasks Vision
  (Apache-2.0), ONNX Runtime Web (MIT).
- Dictionary links go to [SignASL](https://www.signasl.org/).

## Running locally

Any static file server works. The model fetches need http(s), not `file://`:

```bash
python3 -m http.server 8080   # from the repository root
# open http://localhost:8080
```

## Training

`training/` has the pipeline: `prep.py` parses the landmark CSVs, `train.py` and `train2.py`
train, `finalize.py` exports ONNX, `eval_robust.py` and `eval_mirror.py` evaluate. CPU is enough.
A full 220-epoch run takes about half an hour on a recent multi-core machine. It needs the
published WLASL landmark files, not the raw videos.

## Stack

Plain HTML, CSS and JS with ES modules. No build step, no framework. MediaPipe Tasks Vision and
ONNX Runtime Web, both vendored along with the fonts, so the page makes no third-party requests.
