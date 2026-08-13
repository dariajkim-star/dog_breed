#FindDogBreed
강아지 사진에서 강아지를 먼저 검출한 뒤, 순종 견종의 시각적 특징을 담은 Breed Encoder와 Breed Prototype을 이용해
**어떤 견종의 외형적 특징과 얼마나 유사한지 추정하는 프로젝트**입니다.

> **주의:** 최종 출력값은 이미지에서 관찰된 외형적 특징을 기준으로 계산한 **Phenotype Similarity Score**입니다.

---

## 0. Pain Point & 문제정의

**Pain Point** — 사람도, 보호소 전문가도 믹스견의 품종 구성을 외형으로 잘 맞히지 못한다
(DNA 대비 최우세 품종 정답률 56.7%, 두 품종 모두 정답 10.4% — Gunter et al. 2018).
그런데 기존 AI 품종 분류기는 한 품종만 단정(single-label)해서 오히려 더 나쁜 답을 준다.

**문제정의** — Single-label 분류 문제를 **다중 품종 유사도 추정 문제로 재정의**하고,
출력이 DNA가 아니라 Phenotype Similarity임을 시스템 설계 자체(출력 명칭, 면책 문구, Unknown 처리)에 박아 넣는다.

---

## 1. Stage 1 — Dog Detection

일반 사진에서 강아지가 어디에 있는지 찾는 Object Detection 단계입니다.

```text
[Day-1 경로 — 학습 없음]
YOLO11n (COCO 사전학습)
      ↓
class filter = dog
      ↓
Dog Detector  ✅ 설치 직후 동작

[개선 실험 트랙 — 이후]
Open Images V7 (21,586장)
      ↓
YOLO fine-tune
      ↓
robustness 개선 (가림·다중 개체·복잡 배경)
```

### 이 단계의 역할

YOLO는 견종을 구분하지 않습니다. 배우는 것은 오직:

> **"사진 속 어디에 강아지가 있는가?"**

### Crop 규칙 (train = inference 완전 동일 — 제1원칙)

```text
dog bbox
   ↓ 각 변 12~15% expand (이미지 경계 clamp)
   ↓ 긴 변 기준 square
   ↓ resize
Breed Encoder 입력
```

⚠️ 이 규칙은 **Prototype 구축 시와 Inference 시에 반드시 동일하게** 적용합니다.
crop 방식이 어긋나는 것이 실전 정확도를 가장 많이 깎는 사고 지점입니다.

---

## 2. Stage 2 — Breed Encoder

강아지 crop을 **Feature Vector(Embedding)** 로 변환하는 단계입니다.

```text
[Day-1 경로 — 학습 없음]
DINOv2 ViT-S/14 (frozen)
      ↓
crop → 384-d embedding → L2 정규화
✅ frozen feature + prototype만으로 fine-grained 분류가 강력하다는 근거 있음 (SimpleShot 계열)

[개선 실험 트랙 — 이후]
Tsinghua Dogs + Stanford Dogs (25종)
      ↓
ArcFace fine-tune (margin 0.3~0.5, P×K balanced sampler)
      ↓
breed 전용 embedding space
```

Encoder가 포착하는 견종별 외형 특징:
얼굴 형태 · 귀 모양 · 주둥이 비율 · 털 색상 · 털 texture · 체형 · 다리 비율 · 꼬리 형태 · 전체 silhouette

```text
Dog Crop
    ↓
Breed Encoder
    ↓
[0.213, -0.481, 0.092, ..., 0.331]   ← L2 normalized
```

---

## 3. Stage 3 — Breed Prototype Generation

각 순종 견종의 대표 Feature Vector를 생성합니다.

```text
Golden #1 → Embedding
Golden #2 → Embedding
...
Golden #N → Embedding
      ↓
정규화 → 평균 → 재정규화 (+ 전체 평균 centering, CL2N)
      ↓
Golden Retriever Prototype
```

- 클래스당 이미지 **20~50장이면 수렴** — 데이터가 많아도 50장 캡 가능
- centroid와 cosine 하위 5~10%는 outlier로 제거 후 재계산
- Prototype 계산 이미지에는 **augmentation 미적용** (crop 규칙만 동일 적용)

같은 방식으로 25개 견종 Prototype을 만듭니다.

---

## 4. Inference — Mixed-Breed Image

```text
사용자 믹스견 사진
      ↓
YOLO Dog Detector
      ↓
강아지 Crop (12~15% expand 규칙)
      ↓
Breed Encoder
      ↓
Feature Vector (L2 norm)
      ↓
25개 순종 Prototype과 Cosine Similarity
      ↓
┌─────────────────────────────────────┐
│  max similarity < threshold ?       │
│                                     │
│  YES → ⚠️ Unknown 출력:             │
│  "현재 지원하는 품종만으로            │
│   설명하기 어렵습니다"                │
│                                     │
│  NO  → Top-K Selection              │
│        ↓                            │
│  Calibration (temperature softmax)  │
│        ↓                            │
│  Golden Retriever 46%               │
│  Poodle            31%              │
│  Cocker Spaniel    14%              │
│  Other              9%              │
└─────────────────────────────────────┘
```

**Unknown/OOD 분기는 필수입니다.** 이 분기가 없으면 고양이 사진을 넣어도
"Golden 46%"가 출력됩니다. threshold는 validation의 순종 vs 고양이(Oxford)
max-similarity 분포에서 1개 값으로 결정합니다.

---

## 5. Prototype Similarity → Score

```text
Golden Retriever Prototype   → similarity 0.91
Poodle Prototype             → similarity 0.82
Cocker Spaniel Prototype     → similarity 0.61
Husky Prototype              → similarity 0.18
...
```

이 값은 그대로 퍼센트로 사용하지 않습니다.

```text
Cosine Similarity → Calibration → Normalization → Phenotype Similarity Score
```

---

## 6. Important Interpretation

결과를 이렇게 해석하면 **안 됩니다**: `Golden Retriever DNA 46%`

정확한 의미:

> **현재 지원하는 순종 견종들의 시각적 Prototype과 비교했을 때 나타나는 상대적인 외형 유사도**

최종 출력의 공식 명칭: **Phenotype-based Breed Similarity** (= Visual Breed Similarity Score)

---

## 7. 평가지표

| 우선순위 | 지표 | 방법 | 비고 |
|---|---|---|---|
| **메인 KPI** | Purebred sanity check | 순종 test 입력 시 해당 품종이 Top-1인 비율 | GT가 확실한 유일한 composition 검증 |
| 표준 | Top-1 / Top-3 Accuracy | 25종 test set | 공식 수치는 Tsinghua 외부 test 기준 (Stanford는 ImageNet 누수로 참고치) |
| 필수 | OOD 거절 | Oxford 고양이 vs 순종의 max-sim AUROC / rejection rate | Unknown 분기 검증 |
| 시간 되면 | Detection recall | COCO 177장 + OIv7 val 1,586장 | pretrained YOLO 성능 확인 |

> ❌ **금지**: "믹스견 품종 비율 정확도 XX%" — mixed GT가 없으므로 성립 불가.
> 믹스견은 정성 평가(Top-K 합리성) + 인간 베이스라인(MuttMix 데이터) 논의로 다룬다.

---

## 8. Full Pipeline

```text
[DAY-1 SETUP — 학습 없음]

YOLO11n (COCO pretrained, dog filter)     DINOv2 (frozen)
                                                ↓
                              Tsinghua + Stanford 25종 crop
                                                ↓
                                    클래스당 ≤50장 embedding
                                                ↓
                                       25 Breed Prototypes

[이후 실험 트랙]
OIv7 → YOLO fine-tune  /  ArcFace encoder 학습  /  head crop ablation

────────────────────────────────────

[INFERENCE]

User Dog Image
      ↓
YOLO Dog Detector
      ↓
Dog Crop (12~15% expand)
      ↓
Breed Encoder → Feature Vector
      ↓
25 Prototypes와 Cosine Similarity
      ↓
max-sim < threshold → "Unknown" 출력
      ↓ (통과 시)
Top-K + Calibration
      ↓
Golden Retriever 46% / Poodle 31% / Cocker 14% / Other 9%
```

---

## Current Scope

```text
Dog Detection (pretrained)
      ↓
Dog Crop (표준 규칙)
      ↓
Breed Encoder (frozen)
      ↓
Breed Prototype (25종)
      ↓
Top-K Phenotype Similarity + Unknown 거절
```


