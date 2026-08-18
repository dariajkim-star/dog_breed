# 검출 모델 베이스라인 (1-stage vs 2-stage)

- 평가셋: `data/processed/detection/test` = COCO val2017 개 사진 **177장 / GT 218박스**
- 평가기: `scripts/eval_detection.py` (COCO 방식 101점 보간 AP, IoU 0.50:0.05:0.95)
- 모든 모델 COCO 사전학습 그대로, fine-tune 없음. 한 장씩 추론(batch 1)
- 입력 해상도는 각 모델의 표준 설정 — YOLO 640 / Faster R-CNN min 800·max 1333

| 계열 | 모델 | mAP@0.5 | mAP@0.5:0.95 | precision | recall | 검출 0건 | 속도 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1-stage | YOLO11n | 0.7581 | 0.6066 | 0.9070 | 0.7156 | 20.34% | 27.5 img/s |
| 1-stage | **YOLO11s** | 0.8160 | 0.7095 | 0.9198 | 0.7890 | 14.12% | 23.0 img/s |
| 1-stage | YOLO11m | 0.8587 | **0.7503** | 0.9091 | 0.8257 | 12.43% | 13.8 img/s |
| 2-stage | **Faster R-CNN R50-FPN** | **0.8894** | 0.6980 | 0.7908 | **0.8670** | **6.78%** | 2.0 img/s |

## 읽는 법

**1. Faster R-CNN은 "더 많이 찾고, 덜 정확하게 그린다"**
mAP@0.5는 가장 높은데(0.8894) mAP@0.5:0.95는 YOLO11m보다 낮다(0.6980 vs 0.7503).
IoU 0.5 기준으로는 개를 제일 잘 찾지만, 박스를 딱 맞게 그리는 능력은 떨어진다는 뜻이다.
검출 0건이 6.78%로 절반 수준인 것도 같은 이야기다.

**2. 이 프로젝트에서는 mAP@0.5:0.95가 중요하지 않다**
`standard_crop()`이 bbox를 15% 확장하고 정사각형으로 맞춘 뒤 518px로 리사이즈한다.
박스가 조금 헐렁하거나 빡빡해도 최종 crop은 거의 같다.
실제로 중요한 지표는 **검출 0건 비율** 하나다 — 못 찾으면 뒷단계가 아예 돌지 않는다.

**3. precision 비교는 주의**
conf 0.25는 모델마다 의미가 다르다(신뢰도 스케일이 서로 다름).
Faster R-CNN의 낮은 precision(0.7908)은 성능이 아니라 임계값 보정 문제일 수 있다.
임계값과 무관한 mAP가 공정한 비교 지표다.

**4. 속도 차이는 실측 7~11배**
YOLO11s 23.0 img/s vs Faster R-CNN 2.0 img/s.
다만 이 서비스는 사진 한 장씩 처리하므로 0.5초/장은 체감 가능한 수준은 아니다.

## 결론

검출 0건만 놓고 보면 Faster R-CNN이 가장 좋다(6.78%). 다만
- 그 차이(11s 대비 7.3%p)는 `--fallback-full-image`가 이미 흡수하고 있고
  (fallback 경로 Top-1 66.7% vs 정상 경로 78.2% -> 전체 기여 약 +0.85%p)
- 속도는 11배 느리다

따라서 **YOLO11s를 기본값으로 유지**하고, Faster R-CNN은 비교 대상으로 기록한다.
"1-stage가 더 정확해서"가 아니라 **"이 파이프라인이 요구하는 정확도를
1-stage가 이미 충족하고, 남는 정확도는 crop 단계에서 버려지기 때문"** 이 선택 근거다.

## 재현

```
python scripts/eval_detection.py --split test --model yolo11n.pt
python scripts/eval_detection.py --split test --model yolo11s.pt
python scripts/eval_detection.py --split test --model yolo11m.pt
python scripts/eval_detection.py --split test --model fasterrcnn
```
