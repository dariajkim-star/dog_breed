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

### 모델 돌릴 때 등장하지 않는 것 (전달 불필요)

| 폴더 | 정체 |
|---|---|
| `data/raw/` | 원본 (수십 GB) — processed의 재료 |
| `data/labeled/` | 중간 정리본 (라벨 표준화됨, split 전) — crop 규칙 변경/재export 시에만 필요 |

---

## 2. 데이터 전달

- **B에게 전달: breed_body + breed_head + ood = 3.8GB** (zip, GitHub Release 첨부 예정 — 파일당 2GB 제한이라 body는 train / val+test 분할)
- 코드·문서·manifest는 git `main` 브랜치에 전부 있음 → `git pull`
- 모델 가중치는 전달 불필요: YOLO11n(ultralytics 자동 다운로드), DINOv2(torch.hub 자동)
- 데이터 재현이 필요하면: raw 다운로드 후 `build_manifest.py → freeze_taxonomy.py → build_split.py → export_processed.py` 순서 실행

## 3. 반드시 지킬 것

1. **crop은 `utils/crop.py::standard_crop()`만 사용** — processed가 이미 이 함수 산출물.
   inference에서 YOLO bbox를 받으면 같은 함수로 crop (expand=0.15)
2. **prototype은 train split에서만** (클래스당 ≤50장, outlier 하위 5~10% 제거, CL2N)
3. **모든 튜닝은 val에서, test는 최종 1회** — test를 여러 번 보면 사람 손 누수
4. DINOv2 입력은 14의 배수(224/336/518), ImageNet 정규화, RGB
5. MixUp/CutMix 금지, hue augmentation 금지, prototype엔 augmentation 자체 금지
6. Stanford test 수치는 참고치 (ImageNet 누수) — 공식 수치는 우리 test split

## 4. Taxonomy 확정 내역 (이의 없음 합의됨)

- 25종, breed_id 0~24 알파벳순 (`data/manifests/taxonomy.csv`)
- teddy→poodle 병합 (poodle 10,811장) / chinese_rural_dog DROP / cardigan+pembroke→corgi
- oxford `english_cocker_spaniel`은 미국 cocker와 별개 견종 — 25종 밖

## 5. 평가 계획 (README §7 다이어그램 참조)

| 트랙 | 데이터 | 지표 |
|---|---|---|
| 견종 판별 | breed_body/test | **purebred sanity check(메인 KPI)**, Top-1/3, Macro F1 |
| 거절 | ood + breed test | OOD AUROC, rejection rate |
| 검출 | detection/test (COCO) | recall, mAP@50 |
| 믹스견 | 수집 사진 10~20장 | **정성 평가만** — "믹스 비율 정확도" 발표 금지 |

## 6. 알려진 한계 (모델 탓하기 전에 확인)

- Tsinghua 라벨 노이즈 잔존 가능 (chihuahua 등 믹스 의심 샘플) — 이상하게 틀리면 라벨부터 의심
- embedding dedup 미실행 (pHash ≤10 그룹핑으로 대체) — 같은 개 다른 세션 사진의 잔존 가능성 낮게나마 존재
- Oxford 473장은 pseudo-bbox 신뢰도 문제로 제외 (`oxford_pseudo_bbox.parquet`)

## 7. A가 계속 지원하는 것

crop margin 재검증 / bbox QA / YOLO fine-tune 트랙 / QA 리포트 유지보수.
전처리 근거·규칙 전체는 `docs/PREPROCESSING.md` (정본) 참조.
