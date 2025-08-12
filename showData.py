import pickle, os, math, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _to_gray(arr):
    if arr.ndim == 2: return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        g = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        return np.clip(g, 0, 255).astype(np.uint8)
    raise ValueError(f"Unexpected image shape: {arr.shape}")

def _select_channels(images_dict, channels="abc"):
    if isinstance(images_dict, np.ndarray):
        return _to_gray(images_dict)[..., None], ["a"]
    m = {k[-1].lower(): _to_gray(v) for k, v in images_dict.items()}
    used = [ch for ch in channels if ch in m]
    if not used:
        raise KeyError(f"Requested channels '{channels}' not found. available={list(images_dict.keys())}")
    return np.stack([m[ch] for ch in used], axis=-1), used

def _overlay_mask(gray, mask, alpha=0.5):
    gray = gray.astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    color = np.stack([gray, gray, gray], axis=-1) / 255.0
    color[..., 1] = np.clip(color[..., 1] + alpha * mask, 0, 1)
    return color

def summarize_dataset(data_dir):
    data_dir = Path(data_dir)
    paths = {
        "labeled": data_dir / "labeled_training_set.pkl",
        "unlabeled": data_dir / "unlabeled_training_set.pkl",
        "test": data_dir / "test_set.pkl",
    }
    out = {}
    for name, p in paths.items():
        if not p.exists(): out[name] = {"exists": False}; continue
        d = pickle.load(open(p, "rb"))
        parts = sorted(d.keys())
        n_layers = sum(len(d[k]) for k in parts)
        ch_keys, has_spot, has_streak = set(), 0, 0
        for part in parts:
            for it in d[part]:
                ch_keys.update([k[-1].lower() for k in it["images"].keys()])
                has_spot += int(isinstance(it.get("spot_label"), np.ndarray))
                has_streak += int(isinstance(it.get("streak_label"), np.ndarray))
        out[name] = {
            "exists": True, "parts": parts, "num_layers": n_layers,
            "channels_present": sorted(list(ch_keys)),
            "spot_labeled": has_spot, "streak_labeled": has_streak,
        }
    return out

def dataset_stats(labeled_pkl, channels="abc", use_clahe=False, sample_max=None):
    d = pickle.load(open(labeled_pkl, "rb"))
    items = [it for part in sorted(d.keys()) for it in d[part]]
    if sample_max: items = items[:sample_max]
    sums = sqs = None; npx = 0
    spot_px = streak_px = 0; H = W = None

    for it in items:
        arr, used = _select_channels(it["images"], channels)
        if use_clahe and "a" in used:
            try:
                import cv2
                idx_a = used.index("a")
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                arr[..., idx_a] = clahe.apply(arr[..., idx_a].astype(np.uint8))
            except Exception: pass
        H, W, C = arr.shape
        flat = arr.reshape(-1, C).astype(np.float64) / 255.0
        if sums is None:
            sums = np.zeros(C, np.float64); sqs = np.zeros(C, np.float64)
        sums += flat.sum(0); sqs += (flat**2).sum(0); npx += H*W
        if isinstance(it.get("spot_label"), np.ndarray):   spot_px += (it["spot_label"] > 0).sum()
        if isinstance(it.get("streak_label"), np.ndarray): streak_px += (it["streak_label"] > 0).sum()

    mean = (sums / npx); var = (sqs / npx) - (sums / npx) ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return {
        "channels_used": used, "image_shape": (H, W),
        "mean_per_channel": mean.tolist(), "std_per_channel": std.tolist(),
        "spot_positive_pixels_%": 100.0 * spot_px / (len(items)*H*W),
        "streak_positive_pixels_%": 100.0 * streak_px / (len(items)*H*W),
        "num_samples": len(items),
    }

def show_sample(pkl_path, part="part01", idx=0, channels="abc", show_labels=True):
    d = pickle.load(open(pkl_path, "rb"))
    item = d[part][idx]
    arr, used = _select_channels(item["images"], channels)
    H, W, C = arr.shape
    extra = 2 if (show_labels and isinstance(item.get("streak_label"), np.ndarray) and isinstance(item.get("spot_label"), np.ndarray)) else 0
    ncols = C + extra
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4*ncols, 4))
    for i, ch in enumerate(used):
        plt.subplot(1, ncols, i+1)
        plt.title(f"img({ch}) [{H}x{W}]"); plt.imshow(arr[..., i], cmap="gray"); plt.axis("off")
    if extra == 2:
        plt.subplot(1, ncols, C+1); plt.title("streak label")
        plt.imshow(_overlay_mask(arr[..., 0], item["streak_label"])); plt.axis("off")
        plt.subplot(1, ncols, C+2); plt.title("spot label")
        plt.imshow(_overlay_mask(arr[..., 0], item["spot_label"])); plt.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    data_dir = r"C:/Users/geonhan/Desktop/IEEE Hackathon/IEEE_CASE_Hackathon_Data/datasets"
    print("Summary:", summarize_dataset(data_dir))
    stats = dataset_stats(os.path.join(data_dir, "labeled_training_set.pkl"), channels="abc", use_clahe=False, sample_max=200)
    print("Stats:", json.dumps(stats, indent=2))
    show_sample(os.path.join(data_dir, "labeled_training_set.pkl"), part="part01", idx=0, channels="abc", show_labels=True)
