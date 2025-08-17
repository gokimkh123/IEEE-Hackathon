# How it works (Current version without RCA)
1. Input Preprocessing
- Reads ROI images (lighting channels a/b/c) from .pkl.
- Stacks them in the requested --channels order to create an (H,W,C) array.
- Optionally applies CLAHE to channel 'a' (--use_clahe).
- Converts uint8 to float32 and scales by /255.0.
- Normalizes using per-channel (mean, std) calculated during training.

# 2. CNN Encoder
- The CNNEncoder downsamples with stride=4 to generate a feature map: (N, C_in, 139, 250) -> (N, feat, 35, 63).
- Uses MaxPool with ceil_mode=True to exactly match the graph size.

# 3. GNN Processing
- Creates an 8-neighbor grid graph with (35x63) patches as nodes (build_grid_adj).
- SimpleGridGNN updates node features using Mean Aggregation.
- There are no "causal weights" like in RCA here — just simple, equal-weight 8-neighbor message passing.

# 4. Mask Prediction & Upsampling
- GNN output is fed to a SegHead to generate a 2-channel logit map (streak/spot).
- Bilinear upsampling restores the original size (139x250).

# 5. Loss & Evaluation
- During training: BCEWithLogits + λ·Dice loss.
- During validation: Calculates IoU(streak), IoU(spot), and mIoU.
- Saves the model with the best mIoU as best.pt.

# 6. Inference & Submission
- infer_submit.py reads test_set.pkl and uses the channel/normalization info saved during training.
- Attaches two prediction masks to each item, saving them in the streak_label and spot_label fields.
- Saves the entire dictionary as NIST_Task1.pkl — exactly matching the competition's required structure.

# Execution Command
```docker
./run_pipeline.sh