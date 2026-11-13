"""Evaluate the served ONNX model on the WLASL-100 test landmarks, original and mirrored.

Mirrored = x -> 1 - x with left/right hand blocks and left/right pose points swapped, which is
what a flipped camera does. Reuses the helpers from eval_robust.py. Needs test.npz (from prep.py)
and the served model at models/signet-100.onnx. Writes training/mirror.json with top-1/top-5 for both, plus how many clips change
predicted label and how many are right in only one version.
"""
import json, math, sys
import numpy as np
import onnxruntime as ort

T_FIXED = 48
POSE_FLIP = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
             (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]
npz = sys.argv[1] if len(sys.argv) > 1 else '/tmp/slr/test.npz'
onnx = sys.argv[2] if len(sys.argv) > 2 else 'models/signet-100.onnx'
z = np.load(npz, allow_pickle=True)
seqs, labels = list(z['seqs']), z['labels']

def resample(seq, T=T_FIXED):
    n = seq.shape[0]
    idx = np.linspace(0, n - 1, T)
    i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0)[:, None, None]
    return seq[i0] * (1 - w) + seq[i1] * w

def flip(seq):
    out = seq.copy()
    present = out[:, :, 0] > -1.5
    out[:, :, 0] = np.where(present, 1.0 - out[:, :, 0], out[:, :, 0])
    rh = out[:, 0:21, :].copy(); out[:, 0:21, :] = out[:, 21:42, :]; out[:, 21:42, :] = rh
    for a, b in POSE_FLIP:
        pa = out[:, 42 + a, :].copy(); out[:, 42 + a, :] = out[:, 42 + b, :]; out[:, 42 + b, :] = pa
    return out

sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider'])
def predict(transform=None):
    preds = []
    for seq in seqs:
        s = resample(seq)
        if transform: s = transform(s)
        f = (s - 0.5).reshape(1, T_FIXED, 150).astype(np.float32)
        preds.append(np.argsort(-sess.run(['logits'], {'landmarks': f})[0][0])[:5])
    return np.array(preds)

def wilson(p, n, zs=1.96):
    d = 1 + zs**2 / n; c = p + zs**2 / (2 * n)
    h = zs * math.sqrt(p * (1 - p) / n + zs**2 / (4 * n * n))
    return (c - h) / d, (c + h) / d

n = len(labels)
po, pm = predict(), predict(transform=flip)
ok_o, ok_m = po[:, 0] == labels, pm[:, 0] == labels
res = {
    'n': n,
    'original_top1': float(ok_o.mean()), 'original_top5': float(np.mean([y in r for y, r in zip(labels, po)])),
    'mirrored_top1': float(ok_m.mean()), 'mirrored_top5': float(np.mean([y in r for y, r in zip(labels, pm)])),
    'mirrored_wilson95': list(map(float, wilson(float(ok_m.mean()), n))),
    'predicted_label_changed': int((po[:, 0] != pm[:, 0]).sum()),
    'right_only_unmirrored': int((ok_o & ~ok_m).sum()),
    'right_only_mirrored': int((~ok_o & ok_m).sum()),
    'right_in_both': int((ok_o & ok_m).sum()),
    'wrong_in_both': int((~ok_o & ~ok_m).sum()),
}
print(json.dumps(res, indent=1))
out = sys.argv[3] if len(sys.argv) > 3 else 'training/mirror.json'
json.dump(res, open(out, 'w'), indent=1)
