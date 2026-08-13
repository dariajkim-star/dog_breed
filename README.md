# FindDogBreed

## Model Training & Inference Pipeline

강아지 사진에서 강아지를 먼저 검출한 뒤, 순종 견종의 시각적 특징을 학습한 Breed Encoder와 Breed Prototype을 이용해  
**어떤 견종의 외형적 특징과 얼마나 유사한지 추정하는 프로젝트**입니다.

> **주의:** 최종 출력값은 실제 DNA 혈통 비율이 아니라, 이미지에서 관찰된 외형적 특징을 기준으로 계산한 **Phenotype Similarity Score**입니다.

---

## 1. Stage 1 — Dog Detection

먼저 일반 사진에서 강아지가 어디에 있는지 찾는 Object Detection 모델을 학습합니다.

```text
Open Images V7
      ↓
YOLO
      ↓
Dog Detector
```

### 학습 목표

YOLO는 견종을 구분하지 않습니다.

이 단계에서 모델이 배우는 것은 오직:

> **"사진 속 어디에 강아지가 있는가?"**

입니다.

학습이 완료되면 입력 이미지에서 강아지의 Bounding Box를 찾고,  
해당 영역을 Crop하여 다음 Breed Recognition 단계로 전달합니다.

---

## 2. Stage 2 — Breed Encoder Training

강아지 영역이 확보되면, 순종 견종 데이터를 이용해 약 25개 견종의 시각적 특징을 학습합니다.

```text
Tsinghua Dogs
      +
Stanford Dogs
      ↓
Breed Encoder
      ↓
Breed Feature Vector
```

### 학습 목표

Breed Encoder는 다음과 같은 견종별 외형 특징을 구분하도록 학습합니다.

- 얼굴 형태
- 귀 모양
- 주둥이 비율
- 털 색상
- 털 texture
- 체형
- 다리 비율
- 꼬리 형태
- 전체 silhouette

최종적으로 각 강아지 이미지를 하나의 **Feature Vector(Embedding)** 로 변환합니다.

예시:

```text
Dog Image
    ↓
Breed Encoder
    ↓
[0.213, -0.481, 0.092, ..., 0.331]
```

이 벡터는 해당 강아지의 외형적 특징을 숫자로 표현한 값입니다.

---

## 3. Stage 3 — Breed Prototype Generation

Breed Encoder 학습이 끝난 뒤, 각 순종 견종의 대표 Feature Vector를 생성합니다.

예를 들어 Golden Retriever 이미지가 수백 장 있다면:

```text
Golden #1 → Embedding
Golden #2 → Embedding
Golden #3 → Embedding
...
Golden #N → Embedding
```

각 Embedding을 정규화한 뒤 평균을 계산하여  
**Golden Retriever Prototype**을 생성합니다.

같은 방식으로 약 25개 견종의 Prototype을 만듭니다.

```text
Golden Retriever Images
        ↓
Golden Prototype

Poodle Images
        ↓
Poodle Prototype

Husky Images
        ↓
Husky Prototype

...

25 Breeds
        ↓
25 Breed Prototypes
```

Prototype은 각 순종 견종의 **대표적인 외형 특징 좌표** 역할을 합니다.

---

## 4. Inference — Mixed-Breed Image

실제 사용자가 믹스견 사진을 입력하면 다음 순서로 처리합니다.

```text
사용자 믹스견 사진
      ↓
YOLO
      ↓
강아지만 Crop
      ↓
Breed Encoder
      ↓
이 강아지의 Feature Vector
      ↓
25개 순종 Prototype과 거리 비교
      ↓
가까운 견종들을 Top-K로 출력
      ↓
Similarity Calibration / Normalize
      ↓
Golden Retriever 46%
Poodle            31%
Cocker Spaniel    14%
Other               9%
```

---

## 5. Prototype Similarity

사용자 강아지의 Feature Vector와 각 Breed Prototype 간의 유사도를 계산합니다.

예:

```text
Golden Retriever Prototype   → similarity 0.91
Poodle Prototype             → similarity 0.82
Cocker Spaniel Prototype     → similarity 0.61
Labrador Prototype           → similarity 0.56
Husky Prototype              → similarity 0.18
...
```

이 값은 그대로 퍼센트로 사용하지 않습니다.

```text
Cosine Similarity
      ↓
Calibration
      ↓
Normalization
      ↓
Phenotype Similarity Score
```

최종적으로 가장 유사한 견종들을 Top-K 형태로 보여줍니다.

예:

```text
Golden Retriever   46%
Poodle             31%
Cocker Spaniel     14%
Other               9%
```

---

## 6. Important Interpretation

위 결과를 다음과 같이 해석하면 안 됩니다.

```text
Golden Retriever DNA 46%
Poodle DNA            31%
```

이는 실제 유전적 혈통 비율을 의미하지 않습니다.

정확한 의미는:

> **현재 지원하는 순종 견종들의 시각적 Prototype과 비교했을 때 나타나는 상대적인 외형 유사도**

입니다.

따라서 프로젝트의 최종 출력은 다음과 같이 정의합니다.

```text
Phenotype-based Breed Similarity
```

또는

```text
Visual Breed Similarity Score
```

---

## 7. Full Pipeline

```text
[TRAINING]

Open Images V7
      ↓
YOLO Training
      ↓
Dog Detector


Tsinghua Dogs + Stanford Dogs
      ↓
Breed Encoder Training
      ↓
Breed Embedding Space
      ↓
Breed별 Embedding 평균
      ↓
25 Breed Prototypes


────────────────────────────────────


[INFERENCE]

User Dog Image
      ↓
YOLO Dog Detector
      ↓
Dog Bounding Box
      ↓
Dog Crop
      ↓
Breed Encoder
      ↓
Feature Vector
      ↓
25 Breed Prototypes와 Cosine Similarity
      ↓
Top-K Selection
      ↓
Calibration / Normalization
      ↓

Golden Retriever   46%
Poodle             31%
Cocker Spaniel     14%
Other               9%
```

---

## Current Scope

현재 MVP는 다음 범위까지를 목표로 합니다.

```text
Dog Detection
      ↓
Dog Crop
      ↓
Breed Encoder
      ↓
Breed Embedding
      ↓
Breed Prototype
      ↓
Top-K Phenotype Similarity
```

별도의 DNA ancestry prediction 또는 supervised mixed-breed composition regression은 현재 MVP 범위에 포함하지 않습니다.
