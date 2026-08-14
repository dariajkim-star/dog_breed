# 세션 인계 노트 (8/14 저녁 기준)

새 창에서 이 문서 + `docs/PREPROCESSING.md`(전처리 정본) + `docs/handover.md`(기능별 데이터 가이드)를 읽으면 전체 맥락 복원됨.

## 프로젝트 한 줄
FindDogBreed — YOLO11n(검출) + DINOv2 frozen(embedding) + 25종 prototype 유사도로 믹스견의 **외형 기반 견종 조합(Phenotype Similarity)** 추정. DNA 아님(철칙). 마감 8/17.

## 사용자 = 팀원 A (전처리/데이터 담당). B = 모델 담당 (브랜치 hch, 건드리지 말 것)

## 현재 상태 (전부 완료된 것)
- **전처리 실질 종료** — raw 6소스 확보 → labeled(출처별 정규화) → processed(기능별) 완성
- taxonomy FREEZE: 25종 46,497장 (teddy→poodle, chinese_rural_dog DROP, corgi 병합 — 합의됨)
- exact dedup keep-one: 19,442장 제거 (Stanford의 90.6%가 Tsinghua에 중복이었음 — 발표 킬러 소재)
- split FREEZE: group-aware 70/15/15 = 32,549/6,978/6,970
- processed 검증 통과: breed_body 44,506 / breed_head 40,935 / ood 2,400 / detection 20,114쌍
- git: **main이 정본 브랜치** (master는 main과 동기화된 상태, hch=B 것). 전부 push됨
- MuttMix: 사진 없음 → 조건부 No-Go, 정성 평가로 전환 확정

## 진행 중 / 대기
- [ ] **사용자가 직접**: breed_body+breed_head+ood(3.8GB) zip → GitHub Release 업로드 → B에게 전달
  (zip을 git 커밋하면 push 깨짐 — 100MB 제한. Release 첨부는 파일당 2GB, body는 train/val+test 분할 필요)
- [ ] B에게 "main pull + docs/handover.md 읽기" 안내
- [ ] P2 잔여: bbox overlay 갤러리, EDA-2 시각화, crop margin ablation(0/10/15/25%), embedding dedup(개선 트랙)
- [ ] 8/16~17: 믹스견 정성평가 사진 10~20장 수집, 발표자료

## 핵심 규칙 (재확인용)
- crop은 `utils/crop.py::standard_crop()` 단일 함수만 (expand 0.15, square+padding, 518px)
- 튜닝은 val에서만, test는 최종 1회 / prototype은 train만 / MixUp·hue 금지
- Stanford test는 참고치(ImageNet 누수) / "믹스 비율 정확도 XX%" 발표 금지
- 라벨링 코드는 공식문서 스펙 확인 후 작성 (메모리에 저장됨)
- 진행률 보고는 산출물+경로 기준 (메모리에 저장됨)

## 주요 경로
- 정본 문서: `docs/PREPROCESSING.md` / 데이터 가이드: `docs/handover.md` / 기획: `README.md`
- 데이터 원장: `data/manifests/master_manifest.parquet` (canonical_breed, breed_id, dedup_group, split 컬럼)
- 스크립트 재현 순서: build_manifest → freeze_taxonomy → build_split → export_processed
- 리포트: `data/reports/` (preprocessing_summary, split_report, crop_gallery, exact_dedup_report)

## 미해결 사항 / 주의
- taxonomy_draft.csv는 draft 흔적 (정본은 taxonomy.csv) — 혼동 주의
- Oxford 473장 needs_review 제외 상태 (`oxford_pseudo_bbox.parquet`)
- OOD "어려움" 세트(늑대·여우)는 보류 중 — 25종 탈락 견종(dhole, dingo, African_hunting_dog) 재활용 가능
- Slack/Figma MCP 미인증 — 외부 전송 요청 시 claude.ai 커넥터 설정 필요
