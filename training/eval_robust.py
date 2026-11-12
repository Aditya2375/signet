import json, math
import numpy as np
import onnxruntime as ort

T_FIXED = 48
POSE_FLIP = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
             (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]

z = np.load('test.npz', allow_pickle=True)
seqs, labels = list(z['seqs']), z['labels']
# parallel arrays of ids if prep stored them
ids = z['ids'] if 'ids' in z else None
print('ids stored:', ids is not None)

def resample(seq, T=T_FIXED):
    n = seq.shape[0]
    idx = np.linspace(0, n - 1, T)
    i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0)[:, None, None]
    return seq[i0] * (1 - w) + seq[i1] * w

def finalize(seq):
    return (seq - 0.5).reshape(1, T_FIXED, -1).astype(np.float32)

def flip(seq):
    out = seq.copy()
    present = out[:, :, 0] > -1.5
    out[:, :, 0] = np.where(present, 1.0 - out[:, :, 0], out[:, :, 0])
    rh = out[:, 0:21, :].copy(); out[:, 0:21, :] = out[:, 21:42, :]; out[:, 21:42, :] = rh
    for a, b in POSE_FLIP:
        pa = out[:, 42 + a, :].copy(); out[:, 42 + a, :] = out[:, 42 + b, :]; out[:, 42 + b, :] = pa
    return out

sess = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

def run(feat):
    return sess.run(['logits'], {'landmarks': feat})[0][0]

def metrics(transform=None, drop=None):
    top1 = top5 = 0
    for seq, y in zip(seqs, labels):
        s = resample(seq)
        if transform: s = transform(s)
        f = finalize(s).reshape(T_FIXED, 150)
        if drop == 'pose': f[:, 84:] = -2.5
        if drop == 'hands': f[:, :84] = -2.5
        if drop == 'onehand': f[:, 42:84] = -2.5   # left hand gone
        logits = run(f.reshape(1, T_FIXED, 150))
        order = np.argsort(-logits)
        top1 += order[0] == y
        top5 += y in order[:5]
    n = len(labels)
    return top1 / n, top5 / n

# parity vs pytorch on full test set
import torch, importlib.util
spec = importlib.util.spec_from_file_location('trainmod', 'train.py')
# train.py executes training on import; instead rebuild model class by exec of class def only
src = open('train.py').read()
b0 = src.index('class Block')
b1 = src.index('@torch.no_grad()')
ns = {'D_MODEL':192,'N_HEADS':6,'N_LAYERS':3,'D_FF':384,'N_CLASSES':100,'T_FIXED':48}
exec('import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math\n' + src[b0:b1], ns)
Model = ns['Model']
ck = torch.load('best.pt', map_location='cpu')
model = Model()
model.load_state_dict(ck['model'] if 'model' in ck else ck)
model.eval()
maxdiff = 0.0; agree = 0
with torch.no_grad():
    for seq, y in zip(seqs, labels):
        f = finalize(resample(seq)).reshape(1, T_FIXED, 150)
        lt = model(torch.from_numpy(f))[0].numpy()
        lo = run(f)
        maxdiff = max(maxdiff, float(np.max(np.abs(lt - lo))))
        agree += int(np.argmax(lt) == np.argmax(lo))
print(f'PARITY full-test: max|logit diff|={maxdiff:.5f} top1 agreement={agree}/{len(labels)}')

t1, t5 = metrics(); print(f'BASE        top1={t1:.4f} top5={t5:.4f}')
f1, f5 = metrics(transform=flip); print(f'FLIPPED     top1={f1:.4f} top5={f5:.4f}')
p1, p5 = metrics(drop='pose'); print(f'NO-POSE     top1={p1:.4f} top5={p5:.4f}')
h1, h5 = metrics(drop='onehand'); print(f'ONE-HAND    top1={h1:.4f} top5={h5:.4f}')
n1, n5 = metrics(drop='hands'); print(f'NO-HANDS    top1={n1:.4f} top5={n5:.4f}')

# Wilson CI for base top1
n = len(labels); p = t1; zsc = 1.96
den = 1 + zsc**2 / n
ctr = p + zsc**2 / (2 * n)
half = zsc * math.sqrt(p * (1 - p) / n + zsc**2 / (4 * n * n))
print(f'WILSON95 top1: {(ctr - half) / den:.4f} .. {(ctr + half) / den:.4f}')
json.dump({'base': [t1, t5], 'flipped': [f1, f5], 'no_pose': [p1, p5], 'one_hand': [h1, h5],
           'no_hands': [n1, n5], 'parity_maxdiff': maxdiff, 'parity_agree': f'{agree}/{len(labels)}',
           'wilson95': [(ctr - half) / den, (ctr + half) / den]},
          open('robust.json', 'w'), indent=1)
