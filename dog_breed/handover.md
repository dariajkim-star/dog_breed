# Handover — 전처리 완료, 모델 작업 인수인계

작성: 8/14 팀원 A → 팀원 B. **split freeze 완료 — 지금부터 측정하는 성능 수치는 유효함.**

---

## 1. 기능별로 보는 데이터 — 뭘 돌릴 때 뭘 쓰나

### 기능 1 — 사진에서 강아지 찾기 (Object Detection)

**데이터: `data/processed/detection/`** (5.9GB) — ⚠️ MVP에서는 미사용, 전달 대상 아님

```text
detection/
├── images/train  18,397장   ← YOLO fine-tune 학습용 (Open Images)
├── images/val     1,540장   ← 학습 중 검증 (Open Images)
├── images/test      177장   ← 최종 외부 평가 (COCO)
├── labels/...               ← 개 위치 좌표 (YOLO txt, class 0=dog)
└── data.yaml                ← yolo train data=... 바로 가능
```

MVP는 사전학습 YOLO11n을 그대로 쓰므로 이 데이터가 필요 없음.
YOLO를 우리 데이터로 추가 학습하는 실험은 A 담당 (데이터도 A 컴퓨터에 있음).

### 기능 2 — 무슨 종인지 판별 (Breed 유사도) — ★ B의 메인

**데이터: `data/processed/breed_body/`** (2.1GB)

```text
breed_body/
├── train  31,135장   ← 25종 prototype 구축 (견종별 평균 embedding)
├── val     6,694장   ← threshold·temperature 튜닝은 전부 여기서
└── test    6,677장   ← 최종 성적표 (Top-1/3, sanity check) — 마지막에 1회만
```

- 이미 강아지만 잘라낸 518px 정사각 crop — 자르는 작업 없이 바로 embedding 추출
- 라벨 = 폴더명 (`ImageFolder`로 바로 로드, 클래스 알파벳순 0~24 = taxonomy.csv의 breed_id와 일치)

**보조: `data/processed/breed_head/`** (1.5GB) — 같은 구조, 머리만 crop.
"몸 전체 vs 머리" ablation 실험용 (Tsinghua head bbox 기반, 40,935장).

### 기능 3 — 개가 아닌 입력 거절 (OOD)

**데이터: `data/processed/ood/`** (0.2GB)

```text
ood/
├── val   고양이 1,200장   ← 거절 기준선(threshold) 결정
└── test  고양이 1,200장   ← 거절 성능(AUROC) 최종 평가
```

학습에 쓰지 않음 — 고양이 입력 시 25종 유사도가 전부 낮은지 확인하는 평가 전용.

