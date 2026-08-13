# MixedBreed Vision — 1-Day 완성판 (전처리 제외)

> 원칙: **학습(training)을 크리티컬 패스에서 제거한다.**
> 사전학습 모델 2개(YOLO11 + DINOv2)를 그대로 조립하면 학습 0회로 전체 파이프라인이 성립하고,
> 남는 시간에 fine-tune을 "실험"으로 얹는다. 완성이 먼저, 개선은 그다음.

---

## 1. 무엇을 바꾸나 (원안 → 1-Day안)

| 항목 | 원안 | 1-Day안 | 근거 |
|---|---|---|---|
| Detector | YOLO11n/s를 OIv7로 fine-tune | **YOLO11n COCO 사전학습 그대로 사용** (COCO 80클래스에 dog 포함, class filter=16) | 학습 수 시간 절약. 사전학습 YOLO의 dog 검출은 이미 강력 |
| Breed encoder | ResNet50 fine-tune → ArcFace 학습 | **DINOv2 ViT-S/14 frozen** — 학습 없이 embedding 추출만 | SimpleShot 계열 근거: frozen feature + prototype만으로 fine-grained 강력 |
| 분류 방식 | ArcFace + P×K sampler 학습 | **Prototype nearest-centroid** (클래스 평균 → CL2N) | 학습 불필요, 코드 ~30줄 |
| Baseline 비교 | ResNet50 fine-tune baseline | **생략** (시간 남으면 linear probe 1개만) | 실험은 COULD로 강등 |
| Detection 학습 데이터 | OIv7 21,586장 | 사용 안 함 (**평가에만** COCO 177장 + OIv7 val 1,586장) | fine-tune 안 하므로 |
| 믹스 출력 | prototype similarity + temperature softmax | 동일 (변경 없음) | 원래 학습 불필요 |
| OOD | 학습 기반 threshold | **max cosine similarity 임계값** (val에서 1개 숫자만 튜닝) | 단순화 |
| Grad-CAM | ResNet 기반 CAM | **attention rollout(ViT) 또는 유사도 기반 heatmap**, 안 되면 최근접 이미지 갤러리로 대체 | ViT라 CAM 대신. 갤러리가 더 싸고 설득력 있음 |
| 데모 | 웹 UI | **Gradio 단일 파일** (이미지 업로드 → bbox + Top-3 bar + 최근접 3장) | 1시간 내 구축 가능 |

**아키텍처 자체는 불변**: detect → crop(12~15% expand) → encode → prototype similarity → Top-3 + OOD.
바뀌는 건 "각 블록을 학습으로 만드느냐, 사전학습으로 조립하느냐"뿐.

---

## 2. 하루 타임라인 (약 10h, 2인 병렬)

| 시간 | 팀원 A | 팀원 B |
|---|---|---|
| 09:00–10:00 | 환경 세팅(ultralytics, timm/torch.hub), `detect()` 구현: YOLO11n dog filter + 12~15% expand crop | DINOv2 로드, `encode()` 구현: crop→512d(→L2 norm) |
| 10:00–12:00 | detect 검증: COCO 177장 + OIv7 val로 recall 확인, 실패 galleries | **prototype 구축**: 25종 × train 이미지 embedding → 클래스 평균 → CL2N. `compose()` 구현 |
| 12:00–13:00 | 통합: detect→encode→compose 파이프라인 end-to-end 1회 관통 (사진 5장으로 스모크 테스트) | |
| 13:00–15:00 | **평가 러너 작성+실행**: purebred sanity check(test set에서 해당 품종 1위 비율), Top-1/Top-3 | OOD 임계값 튜닝: val 순종 vs Oxford 고양이 max-sim 분포 → threshold 1개 결정, AUROC 계산 |
| 15:00–17:00 | Gradio 데모: 업로드→bbox overlay→Top-3 bar chart→"외형 유사도" 면책 문구 | 최근접 순종 이미지 3장 검색(같은 embedding 재사용) + 데모에 연결 |
| 17:00–18:00 | 실패 사례 수집(가림/검은개/옆모습) | UMAP 1장 (25종 색깔 + 믹스견 1마리 위치) — 발표용 킬러 이미지 |
| 18:00–19:00 | 결과 표·스크린샷 정리, 발표 자료 반영 | 동일 |

**병렬화 핵심**: A와 B는 `detect()/encode()/compose()` 인터페이스 계약만 지키면 오전 내내 서로 대기 없음.
전처리 산출물(taxonomy, split, crop 규칙)은 이미 A 파트에서 공급된다고 가정.

---

## 3. MUST / SHOULD / COULD

### MUST (이게 없으면 미완성)
- [ ] YOLO11n 사전학습 detect() + expand crop
- [ ] DINOv2 frozen encode() + 25종 prototype
- [ ] Top-3 Visual Contribution Score 출력 (temperature softmax)
- [ ] OOD 거절 (max-sim threshold)
- [ ] 평가 2종: purebred sanity check + Top-1/Top-3
- [ ] Gradio 데모 (bbox + Top-3 bar + 면책 문구)

### SHOULD (오후에 시간 나면)
- [ ] 최근접 순종 이미지 3장 갤러리
- [ ] UMAP 시각화
- [ ] OOD AUROC 수치화 (고양이)
- [ ] COCO/OIv7 val detection recall 리포트

### COULD (하루 범위 밖 — 이후 실험 트랙)
- linear probe / ResNet50 fine-tune 비교
- ArcFace 학습, head crop ablation
- YOLO OIv7 fine-tune (mAP 개선)
- 어려운 OOD (늑대·여우), ECE calibration
- synthetic composition 실험

---

## 4. 리스크와 컷라인

| 리스크 | 컷 (더 싼 대안) |
|---|---|
| DINOv2 다운로드/GPU 문제 | timm의 `convnext_tiny` ImageNet feature로 교체 (성능 하락 감수) |
| prototype 계산이 오래 걸림 | 클래스당 이미지 50장으로 캡 (수렴 구간이라 손실 미미) |
| Gradio 막힘 | CLI 출력 + matplotlib 저장 이미지로 데모 대체 |
| OOD threshold가 애매 | "낮은 신뢰도" 경고 문구로만 처리하고 수치 발표 생략 |
| 시간 초과 | SHOULD 전부 버리고 MUST 6개만 — 그래도 발표 스토리 성립 |

---

## 5. 발표 스토리 (1-Day 버전)

1. 문제: 육안 판별 56.7% — 외형→혈통은 원리적으로 어려움 → 우리는 phenotype similarity로 정직하게 정의
2. 파이프라인: detection → crop → frozen foundation model embedding → prototype similarity
3. **"왜 학습을 안 했는가"가 아니라 "왜 학습이 필요 없는가"**: foundation model 시대의 실용적 설계 — SimpleShot 근거 인용
4. 평가: purebred sanity check + OOD 거절 + 인간 베이스라인(MuttMix 데이터) 대비 논의
5. 한계와 다음 단계: fine-tune 실험 트랙 (COULD 목록 그대로)
