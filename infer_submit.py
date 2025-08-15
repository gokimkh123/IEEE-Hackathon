# %%
"""
infer_submit.py
- 학습된 체크포인트(best.pt)를 불러와 test_set.pkl에 대해 추론
- 각 항목에 'streak_label'(채널0), 'spot_label'(채널1)을 uint8(0/1)로 붙임
- 결과 전체 딕셔너리를 NIST_Task1.pkl로 저장 (대회 제출 규격)
"""

import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---- 학습 때 썼던 구성요소 재사용: 같은 폴더의 train.py에서 import ----
from train import (
    CNN_GNN_Seg,          # CNN+GNN 모델
    build_grid_adj,       # 8-이웃 격자 그래프 (35x63) 희소 인접행렬
    select_channels,      # a/b/c 채널 선택
    try_clahe_on_a        # a 채널 CLAHE (옵션)
)

def main(args):
    # --------------------------------------------------
    # 환경/모델 준비
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Device] {device}")

    # 테스트 세트 로드
    test_pkl = Path(args.data_dir) / "test_set.pkl"
    with open(test_pkl, "rb") as f:
        test_dict = pickle.load(f)

    # 정규화 통계/채널/CLAHE 설정 로드 (train.py가 저장한 파일)
    with open(Path(args.save_dir) / "norm_stats.json", "r") as f:
        st = json.load(f)
    channels = st["channels"]                 # 예: "abc", "a", "ab"
    mean = np.array(st["mean"], dtype=np.float32)
    std  = np.array(st["std"],  dtype=np.float32)
    use_clahe = st.get("use_clahe", False)

    # 모델 구성 & 체크포인트 로드
    model = CNN_GNN_Seg(in_ch=len(channels), feat=args.feat, gnn_layers=args.gnn_layers).to(device)
    ckpt_path = Path(args.save_dir) / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Loaded] {ckpt_path}")

    # 인코더가 stride=4이므로 특징맵 크기 35x63 → 고정 격자 그래프(8-이웃) 1회 생성
    A_norm = build_grid_adj(35, 63, device)

    # --------------------------------------------------
    # 추론 루프: part01~04 순회하며 레이어별 예측 생성
    # --------------------------------------------------
    total_cnt = sum(len(test_dict[p]) for p in test_dict)
    pbar = tqdm(total=total_cnt, desc="Inference")
    for part in sorted(test_dict.keys()):
        for item in test_dict[part]:
            # 1) 입력 구성: 채널 선택 → (옵션) CLAHE → /255 → 정규화
            img = select_channels(item["images"], channels)
            img = try_clahe_on_a(img, channels, use_clahe)
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / (std + 1e-6)
            # (H,W,C) -> (1,C,H,W) 텐서
            x = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0).to(device)

            # 2) 추론
            with torch.no_grad():
                logits = model(x, A_norm)                # (1,2,139,250)
                prob = torch.sigmoid(logits)[0].cpu().numpy()  # (2,139,250)

            # 3) 이진화(임계값) 및 dtype 변환(제출 규격)
            pred = (prob > args.thr).astype(np.uint8)    # 0/1, uint8

            # 4) 테스트 딕셔너리에 결과 부착 (채널 순서: 0=streak, 1=spot)
            item["streak_label"] = pred[0]
            item["spot_label"]   = pred[1]

            pbar.update(1)
    pbar.close()

    # --------------------------------------------------
    # 결과 저장: NIST_Task1.pkl (필수 파일명)
    # --------------------------------------------------
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
    ap.add_argument("--gnn_layers", type=int, default=10, help="GNN 레이어 수 (train과 동일)")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu")
    args = ap.parse_args()
    main(args)
