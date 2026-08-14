# 🐕 전처리 계획서 (정본)

> **이 문서가 전처리의 단일 기준입니다.** 다른 메모(crop규칙.txt, Dedup.txt 등)는 이 문서로 통합됨.
> 규칙을 바꾸면 반드시 이 문서를 먼저 고치고 코드를 고친다.
> 담당: 팀원 A · 최종 갱신: 8/14

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
   ↓ ② 이미지 경계 clamp         ← 좌표 음수/초과 버그 방지
   ↓ ③ 긴 변 기준 square         ← 개를 늘리는 게 아니라 "창"을 넓힘 (체형 왜곡 방지)
   ↓ ④ 부족분 padding 후 resize  ← 엣지 케이스(개가 사진 가장자리)도 왜곡 0
Breed Encoder 입력 (224×224, DINOv2면 14의 배수)
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

### 처리: 삭제가 아니라 그룹핑
- `dedup_group=DG_xxxx` 부여 → split은 그룹 단위 (그룹 통째로 train 또는 test) → 누수 원천 차단

### 현재 산출물
- `data/manifests/phash.parquet` (98,385장, 실패 0)
- `data/manifests/phash_pairs.parquet` (≤10: 25,205쌍 — ≤5: 19,949 / 6~8: 1,002 / 9~10: 4,254 / cross-dataset 20,839)

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
- ⚠️ pHash 발견 반영: Tsinghua도 90%가 Stanford(ImageNet)와 동일 소스 → ImageNet 계열 pretrain 기준 "외부 test"는 **Tsinghua 고유분(비중복)에서 구성**. DINOv2(비-ImageNet)를 main으로 쓰는 이유가 강화됨

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
