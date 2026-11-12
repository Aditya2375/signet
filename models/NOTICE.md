# Model and sample data notices

## Trained model (`signet-100.onnx`, `meta.json`, `labels.json`)

Trained on WLASL (WACV 2020), used under the Computational Use of Data Agreement
(C-UDA-1.0): academic and computational use only, no commercial use.
WLASL: https://github.com/dxli94/WLASL

Landmark sequences pre-extracted with MediaPipe as published by Cristina Luna et al.:
https://github.com/cristinalunaj/InterpretableTransformer_SignLanguage

## Replay samples (`samples.json`)

Each sample is a MediaPipe landmark sequence derived from one WLASL test video
(fields: `label`, `video_id`, `split`). Same source and terms as above:
WLASL data under C-UDA-1.0, non-commercial use only. Landmark coordinates only;
no video frames are included or redistributed.

## Runtime bundles

- MediaPipe hand/pose landmarker task bundles (Google): https://ai.google.dev/edge/mediapipe/solutions/terms
- MediaPipe Tasks Vision (Apache-2.0), ONNX Runtime Web (MIT): see `vendor/`.
