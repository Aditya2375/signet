import json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(379); np.random.seed(379)
torch.set_num_threads(1)

T_FIXED = 48
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 3
D_FF = 384
N_CLASSES = 100
EPOCHS = int(os.environ.get("EPOCHS", 220))
BATCH = 32
LR = 3e-4
WD = 1e-2
WARMUP = 8

# pose flip pairs (MediaPipe 33-point pose)
POSE_FLIP = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
             (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]

def load(split):
    z = np.load(f"/tmp/slr/{split}.npz", allow_pickle=True)
    return list(z["seqs"]), z["labels"]

train_s, train_l = load("train")
val_s, val_l = load("val")
test_s, test_l = load("test")

def resample(seq, T=T_FIXED, jitter=False):
    n = seq.shape[0]
    if jitter:
        speed = np.random.uniform(0.8, 1.25)
        start = np.random.uniform(0.0, 0.15)
        stop = min(1.0, start + (1.0 / speed) * np.random.uniform(0.85, 1.0))
        idx = np.linspace(start * (n - 1), stop * (n - 1), T)
    else:
        idx = np.linspace(0, n - 1, T)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0)[:, None, None]
    return seq[i0] * (1 - w) + seq[i1] * w

def flip(seq):
    # raw coords in [0,1] or -2; mirror x where present
    out = seq.copy()
    present = out[:, :, 0] > -1.5
    out[:, :, 0] = np.where(present, 1.0 - out[:, :, 0], out[:, :, 0])
    # swap hands: RH = 0:21, LH = 21:42
    rh = out[:, 0:21, :].copy(); out[:, 0:21, :] = out[:, 21:42, :]; out[:, 21:42, :] = rh
    for a, b in POSE_FLIP:
        pa = out[:, 42 + a, :].copy(); out[:, 42 + a, :] = out[:, 42 + b, :]; out[:, 42 + b, :] = pa
    return out

def augment(seq):
    seq = resample(seq, jitter=True)
    if np.random.rand() < 0.5:
        seq = flip(seq)
    present = seq[:, :, 0:1] > -1.5
    scale = np.random.uniform(0.9, 1.1)
    tx, ty = np.random.uniform(-0.05, 0.05, 2)
    seq = np.where(present, (seq - 0.5) * scale + 0.5 + np.array([tx, ty]), seq)
    seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32) * present
    return seq

def finalize(seq):
    return (seq - 0.5).reshape(T_FIXED, -1).astype(np.float32)  # (48,150)

class DS(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, train):
        self.s, self.l, self.train = seqs, labels, train
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        seq = self.s[i]
        if self.train:
            x = finalize(augment(seq))
        else:
            x = finalize(resample(seq))
        return torch.from_numpy(x), int(self.l[i])

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.drop = nn.Dropout(0.1)
    def attn(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, N_HEADS, C // N_HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(C // N_HEADS), dim=-1)
        return self.proj((a @ v).transpose(1, 2).reshape(B, T, C))
    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(150, D_MODEL)
        self.ln_in = nn.LayerNorm(D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, T_FIXED, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.pool_q = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_CLASSES)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.pool_q, std=0.02)
    def forward(self, x):
        x = self.ln_in(self.inp(x)) + self.pos
        for b in self.blocks:
            x = b(x)
        s = torch.softmax((x @ self.pool_q.transpose(1, 2)).squeeze(-1), dim=-1)  # (B,T)
        x = (x * s.unsqueeze(-1)).sum(1)
        return self.head(self.ln_f(x))

@torch.no_grad()
def evaluate(model, ds):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=64)
    t1 = t5 = n = 0
    for x, y in dl:
        out = model(x)
        t1 += (out.argmax(1) == y).sum().item()
        t5 += (out.topk(5, 1).indices == y.unsqueeze(1)).any(1).sum().item()
        n += len(y)
    return t1 / n, t5 / n

def main():
    model = Model()
    nparams = sum(p.numel() for p in model.parameters())
    print("params:", nparams, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tr = DS(train_s, train_l, True)
    va = DS(val_s, val_l, False)
    te = DS(test_s, test_l, False)
    dl = torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    steps_per_ep = len(dl)
    total = EPOCHS * steps_per_ep
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = (0.0, 0.0)
    step = 0
    start_ep = 0
    CKPT = "/tmp/slr/ckpt.pt"
    if os.path.exists(CKPT):
        c = torch.load(CKPT)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        start_ep = c["epoch"] + 1; step = c["step"]; best = tuple(c["best"])
        print(f"resumed at epoch {start_ep}", flush=True)
    for ep in range(start_ep, EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in dl:
            lr = LR * min((step + 1) / (WARMUP * steps_per_ep), 0.5 * (1 + math.cos(math.pi * step / total)))
            for g in opt.param_groups: g["lr"] = lr
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            v1, v5 = evaluate(model, va)
            if v1 > best[0]:
                best = (v1, v5)
                torch.save(model.state_dict(), "/tmp/slr/best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "step": step, "best": best}, CKPT)
            print(f"ep {ep+1}/{EPOCHS} loss {loss.item():.3f} val {v1:.4f}/{v5:.4f} best {best[0]:.4f} {time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(torch.load("/tmp/slr/best.pt"))
    v1, v5 = evaluate(model, va)
    t1, t5 = evaluate(model, te)
    print(f"FINAL val {v1:.4f}/{v5:.4f} test {t1:.4f}/{t5:.4f}", flush=True)
    json.dump({"val_top1": v1, "val_top5": v5, "test_top1": t1, "test_top5": t5,
               "params": nparams}, open("/tmp/slr/metrics.json", "w"))
    model.eval()
    dummy = torch.zeros(1, T_FIXED, 150)
    torch.onnx.export(model, dummy, "/tmp/slr/model.onnx", opset_version=17,
                      input_names=["landmarks"], output_names=["logits"])
    print("exported onnx", flush=True)

if __name__ == "__main__":
    main()
