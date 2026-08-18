# -*- coding: utf-8 -*-
"""Stage 1 — Dog Detection.

README 1의 역할만 담당한다.
YOLO11n(COCO pretrained)에서 dog class만 필터링해 bbox/confidence를 반환한다.
"""

from __future__ import annotations

import numpy as np


def _get_confidence(detection: tuple) -> float:
    """정렬 기준: (bbox, confidence) 튜플에서 confidence를 꺼낸다."""
    return detection[1]


class DogDetector:
    """YOLO 기반 dog detector.

    무거운 ultralytics import는 객체 생성 시점까지 미룬다.
    predict 사용법 공식 문서: https://docs.ultralytics.com/modes/predict/
    """

    DOG_CLASS_ID: int = 16  # COCO 클래스 번호 16 = 'dog'

    # 기본 가중치를 11n -> 11s로 올린 이유 (test = COCO val2017 dog 177장 실측):
    #   검출 0건  11n 20.34% -> 11s 14.12%   (답을 아예 못 내는 사진이 30% 감소)
    #   속도      27.5 -> 23.0 img/s          (사진 한 장씩 처리라 체감 없음)
    #   recall    0.7156 -> 0.7890
    # precision은 둘 다 0.91로 같다 — 모델을 키워 얻는 건 전부 recall이고,
    # 이 파이프라인은 못 찾으면 뒷단계가 아예 못 도므로 recall이 중요하다.
    def __init__(self, weights: str = "yolo11s.pt", conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf

    def detect(
        self, img: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """RGB 이미지에서 dog bbox를 검출한다.

        Returns:
            [((x1, y1, x2, y2), confidence), ...]
            confidence 내림차순, dog class만 포함.
        """
        # classes=[16]으로 개만 검출 — YOLO는 견종을 모르고 "개 위치"만 찾는다 (README 1)
        results = self.model.predict(
            img,
            conf=self.conf,
            classes=[self.DOG_CLASS_ID],
            verbose=False,
        )

        detections: list[tuple[tuple[float, float, float, float], float]] = []
        for result in results:
            if result.boxes is None:
                continue

            # GPU tensor일 수 있으므로 .cpu()로 옮긴 뒤 numpy로 변환
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, confidence in zip(xyxy, confs):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                detections.append(((x1, y1, x2, y2), float(confidence)))

        # 가장 확실한 검출이 맨 앞에 오도록 confidence 내림차순 정렬
        detections.sort(key=_get_confidence, reverse=True)
        return detections
