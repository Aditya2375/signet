# Model and sample data notices

## Trained model (`signet-100.onnx`, `meta.json`, `labels.json`)

Trained on WLASL (Li et al., WACV 2020), which is released under the Computational Use of Data
Agreement (C-UDA-1.0): https://github.com/dxli94/WLASL

Under that agreement an AI model trained on the Data is a Result (s.5.5), and the agreement puts
no restriction on Results (s.2.2). The model is released with the repository's MIT license and
credited to WLASL as its training data. This is a factual reading of the text, not legal advice.

Landmark sequences were pre-extracted with MediaPipe and published by Cristina Luna et al.:
https://github.com/cristinalunaj/InterpretableTransformer_SignLanguage

## Replay samples (`samples.json`)

Each sample is a MediaPipe landmark sequence derived from one WLASL test video (fields: `label`,
`video_id`, `split`). That makes it derived Data under C-UDA-1.0 (s.3). Keep the WLASL
attribution and use it for computational purposes only. Landmark coordinates only: no video
frames are included or redistributed. As the agreement says (s.4.1), the Data is provided as is,
with no warranty.

## Runtime bundles

- MediaPipe hand and pose landmarker task bundles (Google): https://ai.google.dev/edge/mediapipe/solutions/terms
- MediaPipe Tasks Vision (Apache-2.0), ONNX Runtime Web (MIT): see `vendor/`.
