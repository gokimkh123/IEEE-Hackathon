import argparse, os, json, pickle, random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def _to_gray(arr):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
        if np.issubdtype(gray.dtype, np.floating):
            gray = np.clip(gray, 0, 255)
        return gray.astype(np.uint8)
    raise ValueError(f"Unexpected image shape: {arr.shape}")

def select_channels(images_dict, channels='abc'):
    if isinstance(images_dict, np.ndarray):
        arr = _to_gray(images_dict)[..., None]
        return arr

    m = {k[-1].lower(): _to_gray(images_dict[k]) for k in images_dict}
    wanted = [ch for ch in channels if ch in m]
    if not wanted:
        raise KeyError(f"Requested channels '{channels}' not found. available={list(images_dict.keys())}")
    imgs = [m[ch] for ch in wanted]
    arr = np.stack(imgs, axis=-1)
    return arr

def try_clahe_on_a(img_hw_c, channels, use_clahe=False):
    if not use_clahe or 'a' not in channels:
        return img_hw_c
    try:
        import cv2
        idx_a = channels.index('a')
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_hw_c[..., idx_a] = clahe.apply(img_hw_c[..., idx_a].astype(np.uint8))
    except Exception:
        pass
    return img_hw_c

def flatten_labeled_items(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    items = []
    for part in sorted(d.keys()):
        for it in d[part]:
            items.append({
                'images': it['images'],
                'streak': it['streak_label'],
                'spot': it['spot_label'],
                'layer_id': it['layer_id'],
                'part': part
            })
    return items

def compute_channel_stats(items, channels, use_clahe):
    sums = np.zeros(len(channels), dtype=np.float64)
    sqs = np.zeros(len(channels), dtype=np.float64)
    npx_total = 0
    for it in items:
        arr = select_channels(it['images'], channels)
        arr = try_clahe_on_a(arr, channels, use_clahe)
        H, W, C = arr.shape
        flat = arr.reshape(-1, C).astype(np.float64)
        sums += flat.sum(0)
        sqs += (flat ** 2).sum(0)
        npx_total += H * W
    mean = (sums / npx_total) / 255.0
    var = (sqs / npx_total) - (sums / npx_total) ** 2
    std = np.sqrt(np.clip(var, 1e-12, None)) / 255.0
    return mean.tolist(), std.tolist()

class LabeledDataset(Dataset):
    def __init__(self, items, channels='abc', mean=None, std=None, use_clahe=False, augment=False):
        self.items = items
        self.channels = channels
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.use_clahe = use_clahe
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _maybe_aug(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=2)
        if random.random() < 0.2:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=1)
        return img.copy(), mask.copy()

    def __getitem__(self, idx):
        it = self.items[idx]
        img = select_channels(it['images'], self.channels)
        img = try_clahe_on_a(img, self.channels, self.use_clahe)
        img = img.astype(np.float32) / 255.0

        streak = it['streak'].astype(np.float32)
        spot = it['spot'].astype(np.float32)
        mask = np.stack([streak, spot], axis=0)

        if self.augment:
            img, mask = self._maybe_aug(img, mask)

        img = (img - self.mean) / (self.std + 1e-6)
        img = np.transpose(img, (2, 0, 1))

        img = np.ascontiguousarray(img, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)

        return torch.from_numpy(img), torch.from_numpy(mask)

class CNNEncoder(nn.Module):
    def __init__(self, in_ch, feat=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
        )
        self.feat = feat

    def forward(self, x):
        return self.net(x)

def build_grid_adj(h, w, device):
    idx = []
    for y in range(h):
        for x in range(w):
            u = y * w + x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        v = ny * w + nx
                        idx.append([u, v])
    idx = torch.tensor(idx, dtype=torch.long).t()
    vals = torch.ones(idx.size(1), dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, vals, (h * w, h * w), device=device)

    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    inv_deg = 1.0 / deg
    Dinv = torch.sparse_coo_tensor(
        torch.stack([torch.arange(h * w, device=device),
                     torch.arange(h * w, device=device)]),
        inv_deg, (h * w, h * w)
    )
    A_norm = torch.sparse.mm(Dinv, A)
    return A_norm.coalesce()

class SimpleGridGNN(nn.Module):
    def __init__(self, dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "w_self": nn.Linear(dim, dim, bias=False),
                "w_nei": nn.Linear(dim, dim, bias=False),
                "bn": nn.BatchNorm1d(dim)
            }))

    def forward(self, H, A_norm):
        N, V, D = H.shape
        X = H
        for l in self.layers:
            X_nei = torch.stack([torch.sparse.mm(A_norm, X[n]) for n in range(N)], dim=0)
            Y = l["w_self"](X) + l["w_nei"](X_nei)
            Y = F.relu(l["bn"](Y.reshape(-1, D))).reshape(N, V, D)
            X = Y
        return X

class SegHead(nn.Module):
    def __init__(self, feat=128, out_ch=2):
        super().__init__()
        self.proj = nn.Conv2d(feat, out_ch, 1)

    def forward(self, feat_map):
        return self.proj(feat_map)

class CNN_GNN_Seg(nn.Module):
    def __init__(self, in_ch=3, feat=128, gnn_layers=2, out_ch=2):
        super().__init__()
        self.enc = CNNEncoder(in_ch, feat=feat)
        self.gnn = SimpleGridGNN(dim=feat, layers=gnn_layers)
        self.head = SegHead(feat=feat, out_ch=out_ch)

    def forward(self, x, A_norm):
        f = self.enc(x)
        N, C, H, W = f.shape
        V = H * W
        Hn = f.permute(0, 2, 3, 1).reshape(N, V, C)
        Hn = self.gnn(Hn, A_norm)
        f2 = Hn.reshape(N, H, W, C).permute(0, 3, 1, 2)
        logits_low = self.head(f2)
        logits = F.interpolate(logits_low, size=(139, 250), mode='bilinear', align_corners=False)
        return logits

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(0, 2, 3))
        den = (probs + targets).sum(dim=(0, 2, 3)) + self.eps
        return (1 - (num + self.eps) / den).mean()

def bce_dice(logits, y, w=0.5):
    return F.binary_cross_entropy_with_logits(logits, y) + w * DiceLoss()(logits, y)

@torch.no_grad()
def batch_iou(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * y).sum(dim=(0, 2, 3))
    union = (p + y - p * y).sum(dim=(0, 2, 3)) + 1e-6
    return (inter + 1e-6) / union

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("[Device]", device)

    items = flatten_labeled_items(args.train_pkl)
    tr_idx, va_idx = train_test_split(range(len(items)), test_size=args.val_ratio, random_state=args.seed, shuffle=True)
    tr_items = [items[i] for i in tr_idx]
    va_items = [items[i] for i in va_idx]

    mean, std = compute_channel_stats(tr_items, args.channels, args.use_clahe)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir) / "norm_stats.json", "w") as f:
        json.dump({"channels": args.channels, "mean": mean, "std": std, "use_clahe": args.use_clahe}, f)

    ds_tr = LabeledDataset(tr_items, args.channels, mean, std, args.use_clahe, augment=True)
    ds_va = LabeledDataset(va_items, args.channels, mean, std, args.use_clahe, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = CNN_GNN_Seg(in_ch=len(args.channels), feat=args.feat, gnn_layers=args.gnn_layers).to(device)
    Hp, Wp = 35, 63
    A_norm = build_grid_adj(Hp, Wp, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_miou = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(dl_tr, desc=f"Epoch {ep}/{args.epochs} [train]"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x, A_norm)
            loss = bce_dice(logits, y, w=args.dice_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(ds_tr)
        sch.step()

        model.eval()
        va_loss = 0.0
        inter = torch.zeros(2, device=device)
        union = torch.zeros(2, device=device)
        with torch.no_grad():
            for x, y in tqdm(dl_va, desc=f"Epoch {ep}/{args.epochs} [valid]"):
                x = x.to(device)
                y = y.to(device)
                logits = model(x, A_norm)
                va_loss += bce_dice(logits, y, w=args.dice_w).item() * x.size(0)
                p = (torch.sigmoid(logits) > args.thr).float()
                inter += (p * y).sum(dim=(0, 2, 3))
                union += (p + y - p * y).sum(dim=(0, 2, 3)) + 1e-6
        va_loss /= len(ds_va)
        iou = (inter + 1e-6) / (union + 1e-6)
        miou = iou.mean().item()

        print(f"[E{ep}] tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} "
              f"IoU_streak={iou[0]:.4f} IoU_spot={iou[1]:.4f} mIoU={miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            os.makedirs(Path(args.save_dir) / "checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "best_miou": best_miou},
                       Path(args.save_dir) / "checkpoints" / "best.pt")
            print(f"  -> saved best (mIoU={best_miou:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--train_pkl", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="outputs")
    ap.add_argument("--channels", type=str, default="abc")
    ap.add_argument("--use_clahe", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dice_w", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--feat", type=int, default=768)
    ap.add_argument("--gnn_layers", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")

    args = ap.parse_args()

    if args.train_pkl is None:
        args.train_pkl = str(Path(args.data_dir) / "labeled_training_set.pkl")

    train(args)