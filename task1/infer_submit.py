# %%

import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


from train import (
    CNN_GNN_Seg,
    build_grid_adj,
    select_channels,
    try_clahe_on_a
)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Device] {device}")

    test_pkl = Path(args.data_dir) / "test_set.pkl"
    with open(test_pkl, "rb") as f:
        test_dict = pickle.load(f)

    with open(Path(args.save_dir) / "norm_stats.json", "r") as f:
        st = json.load(f)
    channels = st["channels"]
    mean = np.array(st["mean"], dtype=np.float32)
    std  = np.array(st["std"],  dtype=np.float32)
    use_clahe = st.get("use_clahe", False)


    model = CNN_GNN_Seg(in_ch=len(channels), feat=args.feat, gnn_layers=args.gnn_layers).to(device)
    ckpt_path = Path(args.save_dir) / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Loaded] {ckpt_path}")

    A_norm = build_grid_adj(35, 63, device)


    total_cnt = sum(len(test_dict[p]) for p in test_dict)
    pbar = tqdm(total=total_cnt, desc="Inference")
    for part in sorted(test_dict.keys()):
        for item in test_dict[part]:

            img = select_channels(item["images"], channels)
            img = try_clahe_on_a(img, channels, use_clahe)
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / (std + 1e-6)

            x = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0).to(device)


            with torch.no_grad():
                logits = model(x, A_norm)
                prob = torch.sigmoid(logits)[0].cpu().numpy()


            pred = (prob > args.thr).astype(np.uint8)


            item["streak_label"] = pred[0]
            item["spot_label"]   = pred[1]

            pbar.update(1)
    pbar.close()


    out_path = Path(args.out) if args.out else Path("datasets/NIST_Task1.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(test_dict, f)
    print(f"[Saved] {out_path}  (attach masks into test dict as 'streak_label' & 'spot_label')")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="test_set.pkl이 있는 폴더")
    ap.add_argument("--save_dir", type=str, default="outputs", help="best.pt & norm_stats.json 위치")
    ap.add_argument("--out", type=str, default="NIST_Task1.pkl", help="제출 파일명")
    ap.add_argument("--thr", type=float, default=0.5, help="마스크 이진화 임계값")
    ap.add_argument("--feat", type=int, default=768, help="CNN/GNN 피처 채널 수 (train과 동일)")
    ap.add_argument("--gnn_layers", type=int, default=8, help="GNN 레이어 수 (train과 동일)")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu")
    args = ap.parse_args()
    main(args)
