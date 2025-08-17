import argparse, json, pickle, numpy as np, torch
from pathlib import Path
import matplotlib.pyplot as plt
from train import CNN_GNN_Seg, build_grid_adj, select_channels, try_clahe_on_a

@torch.no_grad()
def infer_one(model, A, x): return torch.sigmoid(model(x, A))[0].cpu().numpy()

def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() and a.device!="cpu" else "cpu")
    st = json.load(open(Path(a.save_dir)/"norm_stats.json"))
    ch, mean, std = st["channels"], np.array(st["mean"],np.float32), np.array(st["std"],np.float32)
    use_clahe = st.get("use_clahe", False)

    L = pickle.load(open(Path(a.data_dir)/"labeled_training_set.pkl","rb"))
    item = L[a.part][a.idx]

    model = CNN_GNN_Seg(in_ch=len(ch), feat=a.feat, gnn_layers=a.gnn_layers).to(device)
    ckpt = torch.load(Path(a.save_dir)/"checkpoints"/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    A = build_grid_adj(35,63,device)

    img = select_channels(item["images"], ch)
    img = try_clahe_on_a(img, ch, use_clahe).astype(np.float32)/255.0
    img_n = (img - mean)/(std+1e-6)
    x = torch.from_numpy(np.transpose(img_n,(2,0,1))).unsqueeze(0).to(device)

    prob = infer_one(model, A, x)
    pred0 = (prob[0] > a.thr_streak).astype(np.uint8)
    pred1 = (prob[1] > a.thr_spot).astype(np.uint8)
    pred   = np.stack([pred0, pred1], axis=0)

    def show(ax, arr, title):
        ax.imshow(arr, cmap="gray"); ax.set_title(title); ax.axis("off")

    fig,axs = plt.subplots(2,3, figsize=(12,8))
    show(axs[0,0], img[...,0], f"img({ch[0]})")
    show(axs[0,1], item["streak_label"], "GT streak"); show(axs[0,2], pred[0], "Pred streak")
    show(axs[1,1], item["spot_label"],   "GT spot");   show(axs[1,2], pred[1], "Pred spot")
    axs[1,0].axis("off")
    plt.tight_layout(); plt.show()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--part", type=str, default="part01")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--thr_streak", type=float, default=0.3)
    ap.add_argument("--thr_spot",   type=float, default=0.7)
    ap.add_argument("--feat", type=int, default=256)
    ap.add_argument("--gnn_layers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args(); main(a)