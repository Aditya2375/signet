import json, sys
import numpy as np
import onnxruntime as ort

RUN = sys.argv[1] if len(sys.argv) > 1 else ""
onnx_path = f"/tmp/slr/model{RUN}.onnx"
metrics = json.load(open(f"/tmp/slr/metrics{RUN}.json"))

# parity + sanity: run the bundled sample replays through the exported ONNX model
names = json.load(open("/tmp/slr/labels.json"))
samples = json.load(open("/tmp/signet/models/samples.json"))

T_FIXED, DIMS = 48, 150

def resample(frames):
    n = len(frames)
    out = np.zeros((T_FIXED, DIMS), dtype=np.float32)
    for t in range(T_FIXED):
        pos = (n - 1) * t / (T_FIXED - 1)
        i0 = int(np.floor(pos)); i1 = min(i0 + 1, n - 1); w = pos - i0
        out[t] = frames[i0] * (1 - w) + frames[i1] * w
    return out - 0.5

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
correct = 0
for s in samples:
    frames = [np.array(f, dtype=np.float32) for f in s["frames"]]
    x = resample(frames)[None]
    logits = sess.run(["logits"], {"landmarks": x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    top = p.argsort()[::-1][:3]
    hit = names[top[0]] == s["label"]
    correct += hit
    print(f"sample '{s['label']}': top3={[ (names[i], round(float(p[i]),3)) for i in top ]} {'OK' if hit else 'MISS'}")

print(f"sample replay: {correct}/{len(samples)} top-1")
meta = {
    "params": metrics["params"],
    "val_top1": round(metrics["val_top1"], 4),
    "val_top5": round(metrics["val_top5"], 4),
    "test_top1": round(metrics["test_top1"], 4),
    "test_top5": round(metrics["test_top5"], 4),
}
json.dump(meta, open("/tmp/signet/models/meta.json", "w"), indent=1)
print("meta:", meta)
