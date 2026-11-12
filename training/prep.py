import csv, json, sys
import numpy as np

BASE = "./WLASL100/MediaPipe_data/x-y"
OUT = "/tmp/slr"
N_LANDM = 75  # RH21 LH21 P33, x/y interleaved

def load(path):
    seqs, labels = [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        assert header[-1] == "labels", header[-1]
        ncols = len(header) - 1
        assert ncols == N_LANDM * 2, ncols
        for row in r:
            lab = row[-1].strip().strip('[]').split(',')[0]
            labels.append(int(lab))
            T = row[0].strip().strip('[]').count(',') + 1
            arr = np.empty((T, N_LANDM, 2), dtype=np.float32)
            for li in range(N_LANDM):
                xs = row[2 * li].strip().strip('[]')
                ys = row[2 * li + 1].strip().strip('[]')
                arr[:, li, 0] = np.fromstring(xs, sep=',', dtype=np.float32)
                arr[:, li, 1] = np.fromstring(ys, sep=',', dtype=np.float32)
            seqs.append(arr)
    return seqs, np.array(labels, dtype=np.int64)

meta = {}
for split in ["train", "val", "test"]:
    s, l = load(f"{BASE}/WLASL100_{split}.csv")
    lens = np.array([x.shape[0] for x in s])
    meta[split] = dict(n=len(s), min=int(lens.min()), med=int(np.median(lens)),
                       max=int(lens.max()), classes=len(set(l.tolist())))
    np.savez(f"{OUT}/{split}.npz",
             seqs=np.array(s, dtype=object), labels=l, allow_pickle=True)
    print(split, meta[split])

json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
# class balance
tr = np.load(f"{OUT}/train.npz", allow_pickle=True)
import collections
c = collections.Counter(tr["labels"].tolist())
print("train per-class min/max:", min(c.values()), max(c.values()))
