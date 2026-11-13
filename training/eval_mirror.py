"""Evaluate the served ONNX model on the WLASL-100 test landmarks, original and mirrored.

Mirrored = x -> 1 - x with left/right hand blocks and left/right pose points swapped, which is
what a flipped camera does. Reuses the helpers from eval_robust.py. Needs test.npz (from prep.py)
and the served model at models/signet-100.onnx. Prints plain top-1 / top-5 for both.
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
def metrics(transform=None):
    t1 = t5 = 0
    for seq, y in zip(seqs, labels):
        s = resample(seq)
        if transform: s = transform(s)
        f = (s - 0.5).reshape(1, T_FIXED, 150).astype(np.float32)
        order = np.argsort(-sess.run(['logits'], {'landmarks': f})[0][0])
        t1 += order[0] == y; t5 += y in order[:5]
    n = len(labels)
    return t1 / n, t5 / n

def wilson(p, n, zs=1.96):
    d = 1 + zs**2 / n; c = p + zs**2 / (2 * n)
    h = zs * math.sqrt(p * (1 - p) / n + zs**2 / (4 * n * n))
    return (c - h) / d, (c + h) / d

n = len(labels)
b1, b5 = metrics(); m1, m5 = metrics(transform=flip)
print(f'n={n}')
print(f'ORIGINAL top1={b1:.4f} top5={b5:.4f} wilson95={wilson(b1, n)}')
print(f'MIRRORED top1={m1:.4f} top5={m5:.4f} wilson95={wilson(m1, n)}')
json.dump({'n': n, 'original': [b1, b5], 'mirrored': [m1, m5], 'mirrored_wilson95': wilson(m1, n)},
          open('mirror.json', 'w'), indent=1)
