핵심 아이디어 (RCA 없음, CNN+GNN만)
CNN 인코더(share): 139×250 → stride=4로 H′×W′ 피처맵 (예: 35×63, 채널 F=128).

GNN(Message Passing 2~3층): H′×W′ 그리드의 8-이웃 그래프를 고정으로 만들고, h' = ReLU(W_self h + W_nei mean(h_neighbors)) 형태로 단순 전달(의존 라이브러리 없이 torch.sparse로 구현).

디코더: (H′×W′,F) → (2,H′,W′) 로짓 → bilinear 업샘플로 (2,139,250), 시그모이드+0.5 임계값.

손실/지표: BCEWithLogits + Dice, 검증은 IoU_streak, IoU_spot, 평균 IoU.

제출: 테스트 딕셔너리 각 항목에 streak_label, spot_label(uint8) 붙여 NIST_Task1.pkl 저장