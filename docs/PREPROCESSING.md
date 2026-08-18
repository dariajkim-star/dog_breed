# 🐕 전처리 계획서 (정본)

> **이 문서가 전처리의 단일 기준입니다.** 다른 메모(crop규칙.txt, Dedup.txt 등)는 이 문서로 통합됨.
> 규칙을 바꾸면 반드시 이 문서를 먼저 고치고 코드를 고친다.
> 담당: 팀원 A · 최종 갱신: 8/14

---

## 요약 — 전처리를 한 문단으로

**폴더는 3층으로 나눔 — raw(원본 그대로) → labeled(라벨 정리본) → processed(모델이 실제로 보는 최종본).** 기준은 하나: 라벨 정리는 빨리 하되, train/test 나누기는 중복 검사가 끝난 뒤에만 함. 안 그러면 같은 사진이 학습용과 시험용에 동시에 들어가는 사고가 나기 때문.

1. **깨진 데이터는 지우지 않고 표시만 함** — "왜 뺐는지"를 기록해둬야 나중에 근거가 됨
2. **라벨 표기를 하나로 통일함** — 데이터셋마다 견종 이름이 제각각이라(toy_poodle, 테디컷 푸들, 오타까지) Stanford 기준으로 맞추고, 견종이 아닌 라벨(중국 잡종견)은 뺌
3. **중복부터 잡음** — 세 데이터셋 모두 인터넷에서 긁은 사진이라 겹칠 거라 의심했고, 실제로 검사해보니 Stanford 사진의 90%가 Tsinghua에 파일명만 바뀐 채 그대로 있었음. 완전히 같은 사진은 한 장만 남기고, 비슷한 사진들은 묶어서 같은 편(train 또는 test)으로만 가게 함
4. **그 다음에야 70/15/15로 나눔** — 견종 비율 유지하면서, 묶인 사진들은 통째로 이동
5. **crop은 함수 하나로 고정함** — 귀·꼬리가 잘리지 않게 박스를 12~15% 넓히고, 개가 늘어나 보이지 않게 정사각형으로 맞춤. 이 규칙을 함수 하나에만 두는 이유는, 기준 사진(prototype)과 사용자 사진이 다른 방식으로 잘리면 같은 개도 다르게 인식돼서 — 에러도 안 나면서 정확도만 조용히 깎이는 최악의 버그이기 때문

한 줄로 줄이면: **"모델 성능을 속이는 요인(중복, 라벨 뒤죽박죽, 잘림 불일치)을 학습 시작 전에 다 없애는 것"**이 전처리의 기준이었음.

---

## 0. 우리가 쓰는 모델 — 전처리는 여기에 맞춘다

| 역할 | 모델 | 전처리가 맞춰야 할 요구사항 |
|---|---|---|
| Stage 1 검출 | **YOLO11n** (COCO 사전학습, Day-1은 fine-tune 없음) | 라벨 = YOLO txt (`class cx cy w h`, 0~1 normalized). 입력 resize/letterbox는 **Ultralytics 내부 처리 — 우리가 미리 하지 않는다** |
| Stage 2 인코더 | **DINOv2 ViT-S/14** (frozen) | 입력은 **14의 배수** (224 기본 / 고해상도 실험은 336·518 — ⚠️ 448 아님). ImageNet 정규화(mean 0.485/0.456/0.406, std 0.229/0.224/0.225), RGB. 출력 **384-d** embedding |
| 실험 트랙 | ResNet50 / ArcFace | 224 입력, 동일 crop 규칙 재사용 (저장 데이터 변경 없음) |

**원칙: 디스크에는 모델 중립으로 저장** (이미지 원본 + YOLO txt + 폴더명 품종 라벨).
DINOv2용 resize·정규화는 저장하지 않고 `encode()` 코드 안에서 처리 → encoder 교체 시 데이터 재작업 0.

---

## 1. 산출물 폴더 구조 (표준)

### 왜 3층인가 — 출처별 → 기능별

```text
raw/        원본 그대로 (불변)
labeled/    정규화 계층 — "출처별" 유지. 라벨은 정리됐지만 split 배정 전.
            출처를 유지하는 이유: cross-dataset dedup 검증, 소스 층화, 문제 역추적
processed/  최종 학습 계층 — "기능별" (detection / breed). split freeze 후에만 생성.
            모델은 이 폴더만 본다.
```

- **최종 학습 폴더는 기능별이 맞다** (기능1: 개 검출 / 기능2: 견종 판별). 단, split이 train/val/test 폴더 구조에 박히기 때문에 **split freeze 전에 기능별로 굳히면 누수 통제 불가** → 그래서 labeled(출처별)를 중간에 둔다
- 견종 폴더는 불편이 아니라 라벨 그 자체 (PyTorch ImageFolder 표준: 폴더명=클래스). YOLO도 images/ 하위 폴더 공식 허용 (images↔labels 경로 치환 매칭)

```text
data/
├── raw/                          # 원본 그대로 (불변)
├── labeled/                      # ★ 라벨링 정리본 — 이미지/라벨 분리
│   ├── classes.txt               #   0=dog_body, 1=dog_head
│   ├── tsinghua/  images/<견종>/*.jpg + labels/<견종>/*.txt  (body+head)
│   ├── stanford/  images/<견종>/*.jpg + labels/<견종>/*.txt  (body)
│   ├── oxford/    images/<견종>/*.jpg + labels/<견종>/*.txt  (pseudo body + head)
│   ├── open_images/ train·validation/ images/ + labels/     (IsGroupOf·IsDepiction 제거)
│   └── coco/      validation/ images/ + labels/
├── manifests/                    # master_manifest, taxonomy, phash, dedup, split
├── processed/                    # split 반영 최종 학습셋 (breed_body/head, detection, ood)
└── reports/                      # inventory, dedup, bbox/crop gallery, QA
```

- 이미지 파일은 **hardlink** (복사 아님 — 디스크 추가 사용 0)
- 품종 라벨 = 폴더명 / 위치 라벨 = labels/*.txt / 전체 메타 = master_manifest.parquet
- bbox 내부 표준: **xyxy absolute pixel** (manifest) → YOLO export 시에만 normalized cxcywh 변환
- 이상 데이터는 삭제하지 않고 `usable=False` + `exclusion_reason`

---

## 2. Crop 규칙 (제1원칙: prototype = inference 완전 동일)

### 규칙

```text
dog bbox (xyxy abs)
   ↓ ① 각 변 12~15% expand      ← detector 오차 흡수 + 귀·꼬리·주둥이 보존
   ↓ ② 긴 변 기준 square         ← 개를 늘리는 게 아니라 "창"을 넓힘 (체형 왜곡 방지)
   ↓ ③ 이미지 경계 clamp         ← 좌표 음수/초과 버그 방지
   ↓ ④ 부족분 padding 후 resize  ← clamp로 잘린 만큼 회색 padding, 내용물 중앙 배치
Breed Encoder 입력 (224×224, DINOv2면 14의 배수)

⚠️ 순서 주의: square가 clamp보다 먼저다 (utils/crop.py 실제 구현 기준).
data/processed 전체가 이 순서로 잘렸으므로 순서 변경 = 데이터 재생성 필수.
```

### 이유 요약
- ① 귀 모양(스패니얼 vs 스피츠), 꼬리(시바 말림 vs 리트리버 깃털), 주둥이(퍼그 vs 콜리)가 판별 단서 — 타이트하면 잘리고, 30%+면 배경·옆 개가 embedding 오염. 12~15% = 관행(10~20%)의 절충
- ③ 무지성 resize는 닥스훈트를 보통 체형으로, 그레이하운드를 통통하게 왜곡. 400×200 → (창 확장) 400×400 → 224×224: 가로세로 같은 배율
- 엣지 케이스: 가장자리 개는 clamp 후 직사각형이 남음 → **(b) padding 채움으로 확정** (왜곡 0)

### 구현 규약
- **`standard_crop()` 함수는 utils/crop.py 하나에만 존재** — prototype 구축·inference·평가 전부 이 함수 호출. 규칙을 "지키는" 게 아니라 "어길 수 없게" 만든다
- train/inference 전처리 불일치 = 에러 없이 정확도만 깎이는 최악의 버그 (증명사진 DB와 CCTV 규격이 다르면 같은 개도 다른 좌표에 찍힘)
- 학습 시(fine-tune 트랙)에만 expand를 5~25% 랜덤화 (detector jitter 강건성)
- prototype 계산 이미지에는 augmentation 미적용

---

## 3. Dedup (시간 배분 1위 — 20~25%)

### 왜: 3개 소스가 전부 인터넷 수집 → 같은 사진이 소스 간 중복 → train/test로 갈라지면 시험지 유출

**실측 (8/14 pHash 완료): Stanford의 90.6%(18,645/20,580장)가 Tsinghua와 중복.** 예방이 아니라 실존 재해였음.

### 방식 지형도에서 선택한 3종 깔때기

| 단계 | 방식 | 잡는 것 | 확정 방법 |
|---|---|---|---|
| ① | **MD5** | 파일이 같은가 (복사본) | 자동 (오탐 0) — manifest에 계산 완료 |
| ② | **pHash 64-bit** | 구도가 같은가 (리사이즈·재압축판) | Hamming **≤5 자동 / 6~8 확정후보 / 9~10 육안** |
| ③ | **DINOv2 embedding cosine** | 내용이 같은가 (연속 컷·대담한 crop판) | **0.92~0.96 육안 튜닝 — 자동 확정 금지** |

탈락: aHash/dHash(pHash 하위호환) · MSE/SIFT 전수(9.8만 장 규모 불가) · 히스토그램("잔디밭 갈색 개" 전부 유사 — 품종 데이터 최악) · SSCD(성능↑이나 일정 비용↑ — 이후 개선 1순위)

### 함정: 순종은 원래 서로 닮음
- 다른 사모예드 두 마리가 cosine 0.93 나올 수 있음 → "진짜 중복(0.95)"과 분포가 겹침
- 대응: threshold 높게 시작(0.92~0.96) / 품종별 분포 확인(흰 개 품종 더 엄격) / **거대 그룹 경보**(한 그룹 수십 장 = threshold 뚫림 신호) / ③단계는 육안 필수
- **비대칭 원칙: 놓침(누수) >> 과다 묶음(낭비)** — 애매하면 묶는다

### 처리 2단계 (8/14 확정)

**1) 확정 중복(exact duplicate)은 keep-one 제거** — 사용자 결정
- 기준: MD5 동일 OR pHash Hamming ≤ 5 ("파일명만 다른 같은 사진")
- 정책: 그룹당 1장만 유지. **유지 우선순위 = 전체 데이터가 적은 소스** (oxford 7,390 > stanford 20,580 > tsinghua 70,432)
- raw 원본은 삭제하지 않음 — manifest에 `usable=False, exclusion_reason='exact_duplicate', dup_keep_id` 기록, `data/labeled/`에서만 탈락본(이미지+라벨) 제거
- **실행 결과 (8/14)**: 중복 그룹 19,053개, 제외 19,442장 (tsinghua 19,158 / stanford 208 / oxford 76) → usable 76,585장

**2) 애매 구간은 그룹핑 → group split** — Hamming 6~10 및 embedding 단계(0.92~0.96) 후보는 제거하지 않고 `dedup_group=DG_xxxx` 부여, split에서 그룹 통째로 같은 쪽 배치

### 현재 산출물
- `data/manifests/phash.parquet` (98,385장, 실패 0)
- `data/manifests/phash_pairs.parquet` (≤10: 25,205쌍 — ≤5: 19,949 / 6~8: 1,002 / 9~10: 4,254 / cross-dataset 20,839)
- `data/reports/exact_dedup_report.md` (keep-one 실행 리포트)
- master_manifest.parquet 갱신 (exact_duplicate 반영)

---

## 4. Taxonomy (라벨 정규화)

- Stanford 120종 마스터, 매핑 테이블 = `data/manifests/taxonomy_draft.csv` → freeze 시 `taxonomy.csv`
- 병합 원칙: "두 소스 annotator가 같은 라벨을 붙였을까" 아니면 병합 / poodle 사이즈 변종 병합(crop 후 크기 소실)
- 🔴 freeze 대기 3건: `teddy`(7,449 — 푸들 미용스타일, 병합 권장) / `chinese_rural_dog`(3,336 — DROP) / corgi(cardigan+pembroke 병합)
- 표기 통일: 소문자+언더스코어, 오타 수정(`Brabancon_griffo`), `Fila Braziliero` 폴더명 공백 주의
- MVP 25종: `data/manifests/mvp25_proposal.csv` (최소 클래스 GSD 363장)

## 5. Split

- **Stratified Group Split 70/15/15** — group = dedup 클러스터, stratify = 품종(+소스)
- test freeze 후 불변. freeze 전 성능 수치는 전부 폐기
- ⚠️ pHash 발견 반영: Tsinghua도 90%가 Stanford(ImageNet)와 동일 소스 → pretrain 데이터와의 이미지 중복 가능성 때문에 "외부 test"는 **Tsinghua 고유분(비중복)에서 구성**
- ⚠️ 주의: "DINOv2는 ImageNet을 안 봐서 누수 무관"이라고 주장하지 말 것 — DINOv2 사전학습 데이터 LVD-142M에는 ImageNet-22k 전체(~14.2M장)와 ImageNet 기반 retrieval 이미지가 포함됨 (https://arxiv.org/abs/2304.07193). 이미지 중복 우려는 ResNet과 동일하게 존재. 차이는 self-supervised라 **품종 label을 본 적이 없다**는 것뿐. DINOv2 선정 근거는 leakage-free가 아니라 "label 없이 학습된 범용 visual representation이 embedding/prototype similarity 방식에 적합"으로 잡는다

## 6. Augmentation (P2 — 후순위, on-the-fly만)

- 사용: HFlip / RandomResizedCrop(약) / brightness·contrast ±0.2~0.3 / label smoothing 0.1 / Random Erasing p=0.1
- 금지: hue shift(모색이 단서), vertical flip, rotation >15°, **MixUp/CutMix**(fine-grained 성능 저하 근거, embedding 학습엔 절대 금지)
- 저장하지 않는다 — train-time on-the-fly. prototype에는 미적용

## 7. Class Imbalance

- metric learning: P×K balanced sampler (P=32, K=4)
- 분류 fine-tune: effective number CB loss(β=0.999) 또는 1/√n — sampler와 이중 보정 금지

---

## 8. 작업 순서 및 현재 상태 (8/14)

| # | Task | 상태 | 산출물 |
|---|---|---|---|
| ① | 데이터 확보 (6소스) | ✅ | `data/raw/` |
| ② | Inventory | ✅ | `data/reports/dataset_inventory.md` |
| ③ | Taxonomy freeze | 🟡 결정 3건 대기 | taxonomy_draft.csv |
| ④⑤ | Manifest + Integrity + MD5 | ✅ | master_manifest.parquet (98,402행) |
| ⑥ | Dedup — pHash 완료, 그룹핑·embedding 남음 | 🟡 | phash.parquet, phash_pairs.parquet |
| — | **라벨링 정리 (오늘 최우선)** | 🟡 진행 중 | `data/labeled/` — Tsinghua/Stanford/OIv7/COCO 변환 중, Oxford pseudo-bbox 생성 중 |
| ⑦ | Split freeze | ⬜ | dedup 후 |
| ⑧⑨ | BBox QA + crop gallery (0/10/15/25% 육안) | ⬜ | 8/16 |
| ⑩ | processed/ export | ⬜ | 8/17 |
| ⑫⑬ | QA report + Handoff | ⬜ | 8/17 |

### 라벨 현황 요약
- 품종 라벨: 폴더명/파일명 기반 전 소스 보유 ✅ (노이즈 3건만 freeze 대기)
- bbox: Tsinghua(body+head) ✅ / Stanford(body) ✅ / Oxford(body 없음 → YOLO pseudo 생성 중, conf<0.5는 needs_review) / OIv7·COCO ✅(필터 적용)
- 세상에 없는 라벨: 믹스견 혼합 비율 → 그래서 prototype 방식 (지도학습 불필요 설계)
- OOD: 보류 (Oxford 고양이로 충분, 늑대·여우는 25종 탈락 견종 재활용 가능)
