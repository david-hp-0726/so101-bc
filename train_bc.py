# train_bc_min.py
import os, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import loader  # expects load_all_episodes(manifest_path)

# ---------- config ----------
MANIFEST = "./data/so101_teleop/manifest.jsonl"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WD = 1e-4
OUT_DIR = "./checkpoints/so101_bc_min"
SEED = 0
# ----------------------------

# tiny helpers
def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class ImgActDS(Dataset):
    def __init__(self, manifest_path, img_size=224, standardize_targets=True):
        eps = loader.load_all_episodes(manifest_path)
        self.samples = []
        for ep in eps:
            root = Path(ep.get("meta", {}).get("root", "."))
            imgs = ep.get("image_path")
            acts = ep["action"]
            if imgs is None: 
                continue
            for p, a in zip(imgs, acts):
                if p is None: 
                    continue
                p = Path(p)
                if not p.is_file(): 
                    p = root / str(p)
                if p.is_file():
                    self.samples.append((str(p), a.astype(np.float32)))
        if not self.samples:
            raise RuntimeError("No samples found. Check image paths.")

        self.img_size = img_size
        self.std_targets = standardize_targets
        Y = np.stack([y for _, y in self.samples], 0)
        self.y_mean = Y.mean(0).astype(np.float32)
        self.y_std = Y.std(0).astype(np.float32); self.y_std[self.y_std < 1e-6] = 1.0

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = self.img_size / min(w, h)
            im = im.resize((int(round(w*s)), int(round(h*s))), Image.BILINEAR)
            w2, h2 = im.size
            left, top = (w2 - self.img_size)//2, (h2 - self.img_size)//2
            im = im.crop((left, top, left + self.img_size, top + self.img_size))
            x = torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.0
            mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            x = (x - mean) / std
        y = y if not self.std_targets else (y - self.y_mean) / self.y_std
        return x, torch.from_numpy(y).float()

def build_model(out_dim):
    try:
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        return m
    except Exception:
        return nn.Sequential(
            nn.Conv2d(3,16,5,2,2), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,out_dim),
        )

def split(ds_len, val_ratio=0.1, seed=0):
    idx = list(range(ds_len)); rng = random.Random(seed); rng.shuffle(idx)
    n_val = max(1, int(round(ds_len*val_ratio)))
    return idx[n_val:], idx[:n_val]

def main():
    set_seed(SEED)
    ds = ImgActDS(MANIFEST, IMG_SIZE, True)
    act_dim = len(ds.samples[0][1])
    tr_idx, va_idx = split(len(ds), 0.1, SEED)
    dtr = torch.utils.data.Subset(ds, tr_idx)
    dva = torch.utils.data.Subset(ds, va_idx)

    dl_tr = DataLoader(dtr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dl_va = DataLoader(dva, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(act_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    best = float("inf"); Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[data] n={len(ds)} | train={len(dtr)} val={len(dva)} | act_dim={act_dim}")
    print(f"[norm] mean={ds.y_mean.round(4).tolist()} std={ds.y_std.round(4).tolist()}")

    for e in range(1, EPOCHS+1):
        # train
        model.train(); tr_sum=0
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            tr_sum += loss.item()*x.size(0)
        tr_mse = tr_sum/len(dtr)
        print(f"[Epoch {e}]: training done")

        # val
        model.eval(); va_sum=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device), y.to(device)
                va_sum += loss_fn(model(x), y).item()*x.size(0)
        va_mse = va_sum/len(dva)
        print(f"[{e:03d}/{EPOCHS}] train_mse={tr_mse:.6f} val_mse={va_mse:.6f}")

        if va_mse < best:
            best = va_mse
            torch.save({
                "epoch": e,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "target_mean": ds.y_mean,
                "target_std": ds.y_std,
            }, os.path.join(OUT_DIR, "best.pt"))
            print("  ↳ saved best.pt")

    # quick sample preds (denormed)
    with torch.no_grad():
        xb, yb = next(iter(dl_va))
        pred = model(xb.to(device)).cpu().numpy()
        yb = yb.numpy()
        den = lambda z: z*ds.y_std + ds.y_mean
        for i in range(min(5, len(xb))):
            print("true:", np.round(den(yb[i]),3).tolist(),
                  "| pred:", np.round(den(pred[i]),3).tolist())

if __name__ == "__main__":
    main()
