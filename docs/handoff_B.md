# Handoff → 팀원 B (Breed Representation Lead)

작성: 8/13 밤, 팀원 A. **Provisional handoff**입니다 — split freeze 전이므로 아래 "주의" 필독.

---

## 1. 지금 바로 쓸 수 있는 것

| 산출물 | 경로 | 내용 |
|---|---|---|
| **Master Manifest** | `data/manifests/master_manifest.parquet` | 98,402행. 컬럼: image_id, source, image_path, original_breed, width/height, **body_bbox, head_bbox**(xyxy absolute pixel), n_dogs, usable, exclusion_reason, md5 |
| Taxonomy 초안 | `data/manifests/taxonomy_draft.csv` | source별 original→canonical 매핑 + merge/review 케이스 |
| **25종 선정안** | `data/manifests/mvp25_proposal.csv` | 수량·소스수·계열·선정근거. **아직 draft — 아래 결정 3건 회신 필요** |
| 견종별 수량 | `data/reports/breed_counts_canonical.csv`, `breed_counts_by_source.csv` | canonical 132종 집계 |
| Inventory | `data/reports/dataset_inventory.md` | 소스별 요약 + 제외 사유 |
| 구현 문서 | `README.md`, `docs/implementation_bori.md` | 파이프라인 정의 (보리 버전이 읽기 쉬움) |

### Raw 데이터 (전부 로컬 확보 완료, git 미포함)
- Tsinghua 70,432장(130종, body+head bbox) / Stanford 20,580장(120종, body bbox) / Oxford 7,390장(head ROI, 고양이는 OOD 전용)
- OIv7 21,586장 + COCO 177장 (detection용 — A 관할)
- MuttMix: **사진 없음 판정** — ancestry CSV + 인간 추측 데이터만 (`data/raw/muttmix/`)

---

## 2. B가 지금 시작할 수 있는 작업

1. **`encode()` 구현** — DINOv2 ViT-S/14 frozen 로드, crop→embedding(L2 norm). 인터페이스:
   ```python
   detect(image) -> list[DogCrop]            # A 소유
   encode(crop)  -> np.ndarray               # B 소유
   compose(vec)  -> list[tuple[str, float]]  # B 소유
   ```
2. **prototype 파이프라인 코드** — 클래스 평균→재정규화(CL2N), 클래스당 ≤50장 캡, outlier(cosine 하위 5~10%) 제거. 견종 목록은 mvp25_proposal 기준으로 **provisional** 구성
3. **GT bbox crop으로 스모크 테스트** — manifest의 body_bbox로 crop 생성 가능. crop 함수는 A가 `standard_crop()` 공용 유틸로 제공 예정(내일 오전). 그 전까지는 아래 규칙으로 임시 구현해도 됨:
   - bbox 각 변 **15% expand** → 이미지 경계 clamp → 긴 변 기준 square(부족분 padding) → resize
   - ⚠️ 단, A의 공용 함수가 나오면 반드시 교체 (prototype·inference 동일 함수 원칙)

---

## 3. ⚠️ 주의 — 아직 하면 안 되는 것

| 금지 | 이유 |
|---|---|
| **성능 수치 확정/기록** | dedup(8/15)·split freeze(8/16) 전 — 지금 숫자는 누수 가능성 있어 전부 폐기 대상. "v1 이전 수치 전부 폐기" 규칙 |
| Stanford test로 평가 | ImageNet subset이라 pretrained 백본 누수. 공식 수치는 Tsinghua 외부 test 기준 (split 확정 후) |
| 자체 crop 규칙 고정 | `standard_crop()` 공용 함수 확정 전. 임시 구현은 OK, 교체 전제 |
| MixUp/CutMix | fine-grained·metric learning에 성능 저하 — 기획 확정사항 |

---

## 4. 🔴 B 회신 필요 — taxonomy freeze blocker (내일 오전까지)

1. **`teddy` (Tsinghua 7,449장)** — 견종이 아니라 푸들 "테디컷" 미용 스타일. **poodle 병합** 권장 (병합 시 poodle 1만+ 장). 샘플 육안 확인 후 결정
2. **`chinese_rural_dog` (3,336장)** — 중국 잡종견, 순종 아님. **DROP** 권장 (정성 테스트 소재로 보관)
3. **corgi**: cardigan(3,063) + pembroke(386) — 꼬리 유무 차이인데 crop에서 꼬리 잘림 빈번. **병합** 권장

회신 오면 A가 `taxonomy.csv` FREEZE → `canonical_breed`/`breed_id` 컬럼 추가된 manifest v3 전달.

---

## 5. 일정 (A 트랙)

| 날짜 | A 작업 | B에 미치는 영향 |
|---|---|---|
| 8/14 | taxonomy FREEZE + `standard_crop()` 배포 + manifest v3 | breed_id 확정, crop 함수 교체 |
| 8/15 | Dedup (pHash는 오늘 밤 계산 중 → embedding → 육안) | — |
| 8/16 | **Split FREEZE** (오전, 즉시 통보) + bbox/crop gallery | 이때부터 성능 수치 유효 |
| 8/17 | breed_body/head/ood/detection export + QA report | **최종 dataset v1 수령** |
