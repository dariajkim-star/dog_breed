# -*- coding: utf-8 -*-
"""Stage 1 — Dog Detection.

README 1의 역할만 담당한다.
YOLO11n(COCO pretrained)에서 dog class만 필터링해 bbox/confidence를 반환한다.
"""

from __future__ import annotations

import numpy as np


class DogDetector:
    """YOLO 기반 dog detector.

    무거운 ultralytics import는 객체 생성 시점까지 미룬다.
    """

    DOG_CLASS_ID: int = 16  # COCO 'dog'

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25) -> None:
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

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, confidence in zip(xyxy, confs):
                x1, y1, x2, y2 = (float(v) for v in box)
                detections.append(((x1, y1, x2, y2), float(confidence)))

        detections.sort(key=lambda item: item[1], reverse=True)
        return detections
