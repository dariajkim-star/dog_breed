# Handoff → 팀원 B (FINAL — 전처리 완료)

작성: 8/14, 팀원 A. **split freeze 완료 — 지금부터 측정하는 성능 수치는 유효합니다.**

---

## 1. 모델 학습에 쓰는 폴더 (이것만 보면 됨)

```text
data/processed/
├── breed_body/ train|val|test/<breed>/*.jpg   ← encode()/prototype/평가 전부 여기
│     25종, 518×518, standard_crop(15%) 적용됨. 폴더명 = 클래스 라벨 (ImageFolder 호환)
│     train 31,135 / val 6,694 / test 6,677
├── breed_head/ (동일 구조, Tsinghua head crop 40,935장)   ← head ablation 실험용
├── ood/ val|test/cat/ (2,400장)                           ← OOD threshold 튜닝·평가
└── detection/ images|labels/train|val|test + data.yaml     ← YOLO fine-tune 트랙용 (A 관할)
```

- breed_id 0~24 = 견종 알파벳순 (`data/manifests/taxonomy.csv` 참조)
- 이미지 메타·bbox·dedup·split의 진실 원장 = `data/manifests/master_manifest.parquet`

## 2. 반드시 지킬 것

1. **crop은 `utils/crop.py::standard_crop()`만 사용** — processed의 crop이 이미 이 함수 산출물.
   inference에서 YOLO bbox를 받으면 **같은 함수**로 crop할 것 (expand=0.15, out_size는 encoder에 맞게)
2. **prototype은 train split에서만** 구축 (클래스당 ≤50장, outlier 하위 5~10% 제거, CL2N)
3. **test는 최종 1회** — val로 모든 튜닝(OOD threshold, temperature) 끝내고 test는 마지막에
4. DINOv2 입력은 14의 배수 (224/336/518), ImageNet 정규화, RGB
5. MixUp/CutMix 금지, hue augmentation 금지 (prototype엔 augmentation 자체를 금지)

## 3. Taxonomy 확정 내역 (이의 없음 확인됨)
- teddy→poodle 병합 (poodle 총 10,811장) / chinese_rural_dog DROP / cardigan+pembroke→corgi
- oxford `english_cocker_spaniel`은 미국 cocker와 별개 견종 — 25종 밖 (병합 안 됨)

## 4. 평가 계획 (README §7 그대로)
- 메인 KPI: purebred sanity check (test에서 해당 품종 Top-1 비율)
- Top-1/Top-3, OOD AUROC(ood/ 고양이), (여유 시) detection recall
- ❌ "믹스 비율 정확도" 발표 금지 — 믹스견은 정성 평가

## 5. 알려진 한계 (모델 탓하기 전에 확인)
- Tsinghua 라벨 노이즈 잔존 가능 — 이상하게 틀리는 샘플은 라벨부터 의심
- embedding dedup 미실행 — 같은 개 다른 세션 사진이 train/test에 갈릴 가능성 낮게나마 존재
- Oxford 473장은 pseudo-bbox 신뢰도 문제로 제외됨 (`oxford_pseudo_bbox.parquet`)

## 6. A가 계속 지원하는 것
- crop margin 갤러리 추가 검증, bbox QA, QA 리포트 유지보수
- YOLO fine-tune 트랙 (detection/ + data.yaml 준비 완료 — `yolo train data=data/processed/detection/data.yaml` 바로 가능)
