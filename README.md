# 동작 원리 (RCA 없는 현재 버전)
1. 입력 전처리
.pkl에서 ROI 이미지(조명 a/b/c 채널)를 읽고,
요청한 --channels 순서로 쌓아서 (H,W,C) 배열로 만듭니다.

선택적으로 a 채널에 CLAHE 적용(--use_clahe).

uint8 → float32 변환 후 /255.0 스케일링.

학습 시 계산된 채널별 (mean,std)로 정규화.

# 2. CNN 인코더
CNNEncoder가 stride=4로 다운샘플하여
(N, C_in, 139, 250) → (N, feat, 35, 63) 특징맵 생성train.

ceil_mode=True로 MaxPool하여 그래프 크기와 정확히 일치.

# 3. GNN 처리
(35×63) 패치 단위를 노드로 하는 8-이웃 격자 그래프 생성 (build_grid_adj).

SimpleGridGNN이 Mean Aggregation으로 노드 피처를 업데이트train.

여기서 RCA에서 쓰는 "원인 가중치" 같은 것은 전혀 없음 — 그냥 동일 가중치 8이웃 메시지 패싱.

# 4. 마스크 예측 & 업샘플링
GNN 출력 → SegHead로 2채널 로짓맵(스트릭/스팟) 생성.

Bilinear 업샘플로 원래 크기 (139×250) 복원train.

# 5. 손실 & 평가
학습 시 BCEWithLogits + λ·Dice 손실.

검증 시 IoU(streak), IoU(spot), mIoU 계산.

최고 mIoU 모델을 best.pt로 저장.

# 6. 추론 & 제출
infer_submit.py가 test_set.pkl을 읽고, 학습 때 저장한 채널/정규화 정보 사용.

각 항목에 예측 마스크 2장을 붙여 streak_label, spot_label 필드에 저장.

전체 딕셔너리를 NIST_Task1.pkl로 저장 — 대회 요구 구조 그대로infer_submit.

# 실행 명령어
```docker
./run_pipeline.sh
```

# thr 최적값
**0.50**