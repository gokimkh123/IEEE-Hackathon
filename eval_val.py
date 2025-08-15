import argparse, json, pickle
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
from train import CNN_GNN_Seg, build_grid_adj, select_channels, try_clahe_on_a, bce_dice

@torch.no_grad()
def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() and a.device!="cpu" else "cpu")
    st = json.load(open(Path(a.save_dir)/"norm_stats.json"))
    ch, mean, std = st["channels"], np.array(st["mean"],np.float32), np.array(st["std"],np.float32)
    use_clahe = st.get("use_clahe", False)

    L = pickle.load(open(Path(a.data_dir)/"labeled_training_set.pkl","rb"))
    items = [it for part in sorted(L) for it in L[part]]
    if a.max_samples: items = items[:a.max_samples]

    model = CNN_GNN_Seg(in_ch=len(ch), feat=a.feat, gnn_layers=a.gnn_layers).to(device)
    ckpt = torch.load(Path(a.save_dir)/"checkpoints"/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    A = build_grid_adj(35,63,device)

    inter = torch.zeros(2, device=device); union = torch.zeros(2, device=device)
    loss_sum = 0.0
    for it in tqdm(items, desc="Eval"):
        img = select_channels(it["images"], ch)
        img = try_clahe_on_a(img, ch, use_clahe).astype(np.float32)/255.0
        img = (img - mean)/(std+1e-6)
        x = torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0).to(device)

        y = np.stack([it["streak_label"], it["spot_label"]], axis=0).astype(np.float32)
        y = torch.from_numpy(y).unsqueeze(0).to(device)

        logits = model(x, A)
        loss_sum += bce_dice(logits, y, w=0.5).item()
        p = (torch.sigmoid(logits) > a.thr).float()
        inter += (p*y).sum(dim=(0,2,3))
        union += (p + y - p*y).sum(dim=(0,2,3)) + 1e-6

    iou = (inter+1e-6)/(union+1e-6)
    print(f"IoU_streak={iou[0].item():.4f}  IoU_spot={iou[1].item():.4f}  mIoU={iou.mean().item():.4f}  loss={loss_sum/len(items):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--feat", type=int, default=768)
    ap.add_argument("--gnn_layers", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args(); main(a)
