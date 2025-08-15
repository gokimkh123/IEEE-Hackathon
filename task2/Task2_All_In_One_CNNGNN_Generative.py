#!/usr/bin/env python3
"""
Task 2 (Synthetic Image Generation) — All‑in‑one implementation using CNN+GNN cVAE
---------------------------------------------------------------------------------

What this single script provides:
  • A conditional VAE generator whose encoder/decoder are CNNs, with an 8‑neighbor
    grid GNN operating on the downsampled spatial feature map (35×63) — matching the
    Task 1 stride pattern.
  • Training pipeline (loads labeled + unlabeled pkl datasets and ignores labels).
  • Generation pipeline that produces the three required submission files:
      NIST_Task2_a.pkl, NIST_Task2_b.pkl, NIST_Task2_c.pkl
    each with shape (100, 139, 250, 3) and dtype uint8.
  • Optional: quick local FID/LPIPS estimators (for rough tuning only). These rely
    on torchvision’s pretrained Inception‑v3 / VGG16. They are not the organizer’s
    official scorers but help during development.

Usage examples (Windows ‘py’ alias assumed):
  # Train (uses datasets/labeled_training_set.pkl and unlabeled_training_set.pkl)
  py Task2_All_In_One_CNNGNN_Generative.py --mode train \
      --data_dir datasets --save_dir outputs_task2 --epochs 200 --batch_size 32

  # Generate 100 images per lighting and write Task 2 pkl files
  py Task2_All_In_One_CNNGNN_Generative.py --mode generate \
      --save_dir outputs_task2 --out_dir outputs_task2 --num 100

  # (Optional) Estimate local FID/LPIPS (development only)
  py Task2_All_In_One_CNNGNN_Generative.py --mode eval \
      --data_dir datasets --save_dir outputs_task2 --num_pairs 200

Dependencies:
  torch>=2.0, torchvision>=0.15, numpy, tqdm, scikit-learn (only if you extend splitting)
  (Same family as Task 1 requirements.)

Notes:
  • The dataset reader expects the NIST CASE ROI pkls with keys like 'Axxxxa', 'Axxxxb', 'Axxxxc'.
  • We train and reconstruct grayscale in [0,1]. For submission we replicate to 3 channels (uint8).
  • Lighting condition (a/b/c) is injected as one‑hot planes into both encoder and decoder.
  • The adjacency is an 8‑neighbor grid for the 35×63 latent feature map.
"""

import argparse
import os
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision
    from torchvision import transforms
except Exception:
    torchvision = None

# ==============================================================================
#                               Utilities & Data
# ==============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert (H,W) or (H,W,3) to (H,W) uint8 grayscale.
    Accepts either raw grayscale or 3‑channel arrays.
    """
    if arr.ndim == 2:
        g = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        g = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        if np.issubdtype(g.dtype, np.floating):
            g = np.clip(g, 0, 255)
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    return g.astype(np.uint8)


def load_task2_samples(data_dir: str) -> List[Dict[str, object]]:
    """Load every ROI frame from labeled + unlabeled sets.
    Returns list of dicts with keys: {'img': HxW uint8, 'cond': 'a'|'b'|'c'}
    """
    data_dir = Path(data_dir)
    paths = [data_dir / "labeled_training_set.pkl", data_dir / "unlabeled_training_set.pkl"]
    samples = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        for part in sorted(d.keys()):
            for it in d[part]:
                for k, v in it["images"].items():
                    cond = k[-1].lower()  # 'a' | 'b' | 'c'
                    if cond in ("a", "b", "c"):
                        img = _to_gray(v)
                        samples.append({"img": img, "cond": cond})
    if not samples:
        raise FileNotFoundError("No samples found. Ensure *training_set.pkl files exist under data_dir.")
    return samples


class ROIDataset(Dataset):
    """Task 2 Dataset: uses grayscale ROI and lighting condition as input.
    Returns (x_in, target) where:
      - x_in: (1+3, H, W) = [gray01, onehot(a/b/c) planes]
      - target: (1, H, W) = gray01
    """

    def __init__(self, items: List[Dict[str, object]], augment: bool = False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _maybe_aug(self, img01: np.ndarray) -> np.ndarray:
        # simple flips only; avoid heavy color/brightness changes
        if random.random() < 0.5:
            img01 = np.flip(img01, axis=1)
        if random.random() < 0.2:
            img01 = np.flip(img01, axis=0)
        return img01.copy()

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img01 = it["img"].astype(np.float32) / 255.0  # (H,W) in [0,1]
        if self.augment:
            img01 = self._maybe_aug(img01)
        img01 = np.ascontiguousarray(img01)

        # condition planes (3,H,W)
        cidx = {"a": 0, "b": 1, "c": 2}[it["cond"]]
        H, W = img01.shape
        cond_planes = np.zeros((3, H, W), dtype=np.float32)
        cond_planes[cidx, ...] = 1.0

        x_in = np.concatenate([img01[None, ...], cond_planes], axis=0)  # (1+3,H,W)
        target = img01[None, ...]                                        # (1,H,W)

        x_in = np.ascontiguousarray(x_in, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        return torch.from_numpy(x_in), torch.from_numpy(target)


# ==============================================================================
#                            8‑Neighbor Grid GNN
# ==============================================================================

def build_grid_adj(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Build sparse adjacency for an 8‑neighbor grid (h×w). Returns D^{-1}A."""
    idx = []
    for y in range(h):
        for x in range(w):
            u = y * w + x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        v = ny * w + nx
                        idx.append([u, v])
    idx = torch.tensor(idx, dtype=torch.long, device=device).t()  # (2, E)
    vals = torch.ones(idx.size(1), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(idx, vals, (h * w, h * w), device=device)
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    inv_deg = 1.0 / deg
    Dinv_idx = torch.arange(h * w, device=device)
    Dinv = torch.sparse_coo_tensor(
        torch.stack([Dinv_idx, Dinv_idx]), inv_deg, (h * w, h * w)
    )
    A_norm = torch.sparse.mm(Dinv, A).coalesce()
    return A_norm


class SimpleGridGNN(nn.Module):
    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "w_self": nn.Linear(dim, dim, bias=False),
                        "w_nei": nn.Linear(dim, dim, bias=False),
                        "bn": nn.BatchNorm1d(dim),
                    }
                )
            )

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # H: (N,V,D) where V = h*w; A_norm: (V,V) sparse
        N, V, D = H.shape
        X = H
        for l in self.layers:
            X_nei = torch.stack([torch.sparse.mm(A_norm, X[n]) for n in range(N)], dim=0)
            Y = l["w_self"](X) + l["w_nei"](X_nei)
            Y = F.relu(l["bn"](Y.reshape(-1, D))).reshape(N, V, D)
            X = Y
        return X


# ==============================================================================
#                               cVAE with CNN+GNN
# ==============================================================================

class CNNEncoder(nn.Module):
    """(C_in,139,250) -> (feat,35,63) via stride‑4 CNN"""

    def __init__(self, in_ch: int, feat: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),  # 139x250 -> 70x125
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),  # 70x125 -> 35x63
            nn.Conv2d(64, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
        )
        self.feat = feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNDecoder(nn.Module):
    """(z_dim+cond,35,63) -> (1,139,250) in [0,1]"""

    def __init__(self, z_dim: int = 64, cond_ch: int = 3, feat: int = 128):
        super().__init__()
        in_ch = z_dim + cond_ch
        self.decode = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
        )
        self.out_head = nn.Conv2d(feat, 1, 1)

    def forward(self, z_map: torch.Tensor, cond_planes: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_map, cond_planes], dim=1)  # (N,z_dim+3,35,63)
        f = self.decode(x)
        low = self.out_head(f)  # (N,1,35,63)
        up = F.interpolate(low, size=(139, 250), mode="bilinear", align_corners=False)
        return torch.sigmoid(up)


class CVAE_GNN(nn.Module):
    def __init__(self, in_ch: int = 4, feat: int = 128, gnn_layers: int = 2, z_dim: int = 64):
        super().__init__()
        self.enc = CNNEncoder(in_ch, feat=feat)
        self.gnn_e = SimpleGridGNN(dim=feat, layers=gnn_layers)
        self.mu_head = nn.Conv2d(feat, z_dim, 1)
        self.lv_head = nn.Conv2d(feat, z_dim, 1)
        self.gnn_d = SimpleGridGNN(dim=z_dim, layers=gnn_layers)
        self.dec = CNNDecoder(z_dim=z_dim, cond_ch=3, feat=feat)

    def encode(self, x: torch.Tensor, A_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.enc(x)  # (N,feat,35,63)
        N, C, H, W = f.shape
        V = H * W
        Hn = f.permute(0, 2, 3, 1).reshape(N, V, C)
        Hn = self.gnn_e(Hn, A_norm).reshape(N, H, W, C).permute(0, 3, 1, 2)
        mu = self.mu_head(Hn)
        logvar = self.lv_head(Hn)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond_planes_low: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        N, C, H, W = z.shape
        V = H * W
        Hn = z.permute(0, 2, 3, 1).reshape(N, V, C)
        Hn = self.gnn_d(Hn, A_norm).reshape(N, H, W, C).permute(0, 3, 1, 2)
        return self.dec(Hn, cond_planes_low)

    def forward(self, x_in: torch.Tensor, A_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_in = [gray01 (1ch), onehot planes (3ch)]
        cond_high = x_in[:, 1:, ...]  # (N,3,139,250)
        cond_low = F.interpolate(cond_high, size=(35, 63), mode="nearest")
        mu, logvar = self.encode(x_in, A_norm)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond_low, A_norm)
        return recon, mu, logvar


# ==============================================================================
#                                   Training
# ==============================================================================

def kld_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL per element then mean over batch/spatial/channel
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print("[Device]", device)

    # Load data
    all_items = load_task2_samples(args.data_dir)
    random.shuffle(all_items)
    n_total = len(all_items)
    n_val = max(int(n_total * args.val_ratio), 1)
    va_items = all_items[:n_val]
    tr_items = all_items[n_val:]

    ds_tr = ROIDataset(tr_items, augment=True)
    ds_va = ROIDataset(va_items, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = CVAE_GNN(in_ch=4, feat=args.feat, gnn_layers=args.gnn_layers, z_dim=args.z_dim).to(device)
    A_norm = build_grid_adj(35, 63, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = float("inf")
    log = []

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_rec, tr_kld = 0.0, 0.0
        for x_in, target in tqdm(dl_tr, desc=f"Epoch {ep}/{args.epochs} [train]"):
            x_in = x_in.to(device)
            target = target.to(device)
            recon, mu, logvar = model(x_in, A_norm)
            rec = F.mse_loss(recon, target)
            # KL warmup
            beta = args.beta * min(1.0, ep / float(max(args.kl_warmup, 1)))
            kld = kld_loss(mu, logvar)
            loss = rec + beta * kld
            opt.zero_grad(); loss.backward(); opt.step()
            tr_rec += rec.item() * x_in.size(0)
            tr_kld += kld.item() * x_in.size(0)
        sch.step()
        tr_rec /= len(ds_tr)
        tr_kld /= len(ds_tr)

        model.eval()
        va_rec, va_kld = 0.0, 0.0
        with torch.no_grad():
            for x_in, target in tqdm(dl_va, desc=f"Epoch {ep}/{args.epochs} [valid]"):
                x_in = x_in.to(device)
                target = target.to(device)
                recon, mu, logvar = model(x_in, A_norm)
                va_rec += F.mse_loss(recon, target).item() * x_in.size(0)
                va_kld += kld_loss(mu, logvar).item() * x_in.size(0)
        va_rec /= len(ds_va)
        va_kld /= len(ds_va)

        score = va_rec + args.beta * va_kld
        log.append({"epoch": ep, "rec_tr": tr_rec, "kld_tr": tr_kld, "rec_va": va_rec, "kld_va": va_kld, "score": score})
        print(f"[E{ep}] rec_tr={tr_rec:.6f} kld_tr={tr_kld:.6f}  rec_va={va_rec:.6f} kld_va={va_kld:.6f}  score={score:.6f}")

        if score < best_score:
            best_score = score
            ckpt_dir = Path(args.save_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_score": best_score,
                "log": log,
            }, ckpt_dir / "task2_best.pt")
            print(f"  -> Saved best checkpoint (score={best_score:.6f})")


# ==============================================================================
#                                   Generate
# ==============================================================================

@torch.no_grad()
def generate(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    save_dir = Path(args.save_dir)
    ckpt = torch.load(save_dir / "checkpoints" / "task2_best.pt", map_location=device)
    margs = ckpt["args"]

    model = CVAE_GNN(in_ch=4, feat=margs["feat"], gnn_layers=margs["gnn_layers"], z_dim=margs["z_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    A = build_grid_adj(35, 63, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def cond_low_planes(letter: str, N: int) -> torch.Tensor:
        idx = {"a": 0, "b": 1, "c": 2}[letter]
        planes = torch.zeros(N, 3, 35, 63, device=device)
        planes[:, idx, :, :] = 1.0
        return planes

    def to_rgb3_uint8(arr01: np.ndarray) -> np.ndarray:
        x = (np.clip(arr01, 0, 1) * 255.0).astype(np.uint8)
        return np.stack([x, x, x], axis=-1)

    for letter in ["a", "b", "c"]:
        N = args.num
        # sample z ~ N(0, I)
        z = torch.randn(N, margs["z_dim"], 35, 63, device=device)
        cond = cond_low_planes(letter, N)
        imgs01 = model.decode(z, cond, A).squeeze(1).cpu().numpy()  # (N,139,250)
        rgb = np.stack([to_rgb3_uint8(im) for im in imgs01], axis=0)  # (N,139,250,3)
        assert rgb.shape == (N, 139, 250, 3)
        fn = out_dir / f"NIST_Task2_{letter}.pkl"
        with open(fn, "wb") as f:
            pickle.dump(rgb, f)
        print(f"[Saved] {fn} shape={rgb.shape} dtype={rgb.dtype}")


# ==============================================================================
#                                   Eval (Optional)
# ==============================================================================

@torch.no_grad()
def _get_inception_features(images_uint8: np.ndarray, device: torch.device) -> np.ndarray:
    """images_uint8: (N,H,W,3) in [0,255] uint8. Returns (N,2048) features.
    Uses torchvision Inception v3 with fc replaced by Identity.
    """
    assert torchvision is not None, "torchvision is required for eval mode"
    N = images_uint8.shape[0]
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    # Preprocess to 299x299 float tensor in [0,1], then normalize as ImageNet
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1]
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    bs = 16
    for i in range(0, N, bs):
        batch = images_uint8[i : i + bs]
        tensors = torch.stack([preprocess(img) for img in batch], dim=0).to(device)
        f = model(tensors)  # (B,2048)
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    # Fréchet distance between two Gaussians
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2)).real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@torch.no_grad()
def _get_vgg_features(images_uint8: np.ndarray, device: torch.device) -> List[torch.Tensor]:
    assert torchvision is not None, "torchvision is required for eval mode"
    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features.to(device).eval()
    layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3 (approx indices)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _forward_feats(imgs: np.ndarray) -> List[torch.Tensor]:
        x = torch.stack([preprocess(im) for im in imgs], dim=0).to(device)
        feats = []
        h = x
        for i, m in enumerate(vgg):
            h = m(h)
            if i in layers:
                feats.append(h)
        return feats

    return _forward_feats


def _lpips_distance(fn_feats, imgA: np.ndarray, imgB: np.ndarray, device: torch.device) -> float:
    # imgA/imgB: (H,W,3) uint8
    feats_A = fn_feats(np.stack([imgA], axis=0))
    feats_B = fn_feats(np.stack([imgB], axis=0))
    d = 0.0
    for FA, FB in zip(feats_A, feats_B):
        # normalize channels
        FA = FA / (FA.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        FB = FB / (FB.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        d += torch.mean((FA - FB) ** 2).item()
    return float(d)


def evaluate_local(args: argparse.Namespace):
    """Quick‑n‑dirty local FID/LPIPS estimator for sanity checks.
    This is not the official metric, but useful for model iteration.
    """
    if torchvision is None:
        raise RuntimeError("torchvision is required for eval mode")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # Collect a reference set of real images for each lighting (up to max_n)
    items = load_task2_samples(args.data_dir)
    max_n = args.real_max
    by_cond = {"a": [], "b": [], "c": []}
    for it in items:
        if len(by_cond[it["cond"]]) < max_n:
            g = it["img"]
            rgb = np.stack([g, g, g], axis=-1)
            by_cond[it["cond"]].append(rgb)
    real_sets = {k: np.stack(v, axis=0) for k, v in by_cond.items() if v}

    # Load generated files from out_dir (assumes you ran --mode generate)
    out_dir = Path(args.save_dir)
    gen_sets = {}
    for letter in ["a", "b", "c"]:
        p = out_dir / f"NIST_Task2_{letter}.pkl"
        if p.exists():
            gen_sets[letter] = pickle.load(open(p, "rb"))

    # FID per condition
    for letter in ["a", "b", "c"]:
        if letter not in real_sets or letter not in gen_sets:
            print(f"[Eval] Skip {letter}: missing real or generated set")
            continue
        real = real_sets[letter]
        gen = gen_sets[letter]
        # features
        fr = _get_inception_features(real, device)
        fg = _get_inception_features(gen, device)
        mu_r, sig_r = fr.mean(0), np.cov(fr, rowvar=False)
        mu_g, sig_g = fg.mean(0), np.cov(fg, rowvar=False)
        fid = _compute_fid(mu_r, sig_r, mu_g, sig_g)
        print(f"[FID] {letter}: {fid:.3f}")

    # Intra‑LPIPS per condition (random pairs from generated set)
    fn_feats = _get_vgg_features(None, device)  # returns closure
    rng = np.random.default_rng(123)
    for letter in ["a", "b", "c"]:
        if letter not in gen_sets:
            continue
        gen = gen_sets[letter]
        n = len(gen)
        pairs = min(args.num_pairs, n * (n - 1) // 2)
        if pairs <= 0:
            continue
        idx = rng.choice(n, size=(pairs, 2), replace=False)
        dists = []
        for i, j in idx:
            d = _lpips_distance(fn_feats, gen[i], gen[j], device)
            dists.append(d)
        print(f"[LPIPS] {letter}: mean={np.mean(dists):.4f} (pairs={pairs})")


# ==============================================================================
#                                     Main
# ==============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Task 2: CNN+GNN cVAE all‑in‑one")
    ap.add_argument("--mode", type=str, choices=["train", "generate", "eval"], required=True)

    # Common
    ap.add_argument("--save_dir", type=str, default="outputs_task2")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu")

    # Train
    ap.add_argument("--data_dir", type=str, default="datasets")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--gnn_layers", type=int, default=2)
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    ap.add_argument("--kl_warmup", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # Generate
    ap.add_argument("--out_dir", type=str, default="outputs_task2")
    ap.add_argument("--num", type=int, default=100, help="images per lighting condition")

    # Eval (optional)
    ap.add_argument("--real_max", type=int, default=300, help="max real imgs per lighting for FID")
    ap.add_argument("--num_pairs", type=int, default=200, help="#pairs for LPIPS per lighting")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
    elif args.mode == "eval":
        evaluate_local(args)
