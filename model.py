# -*- coding: utf-8 -*-
"""model.py — 모델 계층 (README 1~5 단계 구현)

- DogDetector : YOLO11n(COCO pretrained)으로 dog bbox 검출 (README 1)
- BreedEncoder: DINOv2 ViT-S/14 frozen encoder → 384-d L2 정규화 embedding (README 2)
- build_prototypes: 클래스별 prototype 생성 (캡 50장, CL2N, outlier 제거 — README 3)
- similarity_scores / predict: cosine 유사도 + Unknown 분기 + temperature softmax (README 4~5)

학습 코드 없음 — Day-1 경로는 사전학습 가중치를 그대로 사용한다(frozen).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO


class DogDetector:
    """Stage 1 — Dog Detection (README 1).

    YOLO11n(COCO 사전학습)을 그대로 사용하고 dog 클래스만 필터링한다.
    Ultralytics predict 모드 공식 문서: https://docs.ultralytics.com/modes/predict/
    COCO 클래스 인덱스 16 = dog (Ultralytics 기준).
    """

    DOG_CLASS_ID: int = 16  # COCO 'dog' (Ultralytics 클래스 매핑 기준)

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25) -> None:
        self.model = YOLO(weights)
        self.conf = conf

    def detect(
        self, img: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """RGB 이미지에서 dog bbox 검출.

        Args:
            img: (H, W, 3) RGB uint8 numpy 배열.

        Returns:
            [((x1, y1, x2, y2), conf), ...] — confidence 내림차순 정렬, dog만 포함.
        """
        # classes=[16]으로 dog만 검출 (https://docs.ultralytics.com/modes/predict/)
        results = self.model.predict(
            img, conf=self.conf, classes=[self.DOG_CLASS_ID], verbose=False
        )
        out: list[tuple[tuple[float, float, float, float], float]] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, c in zip(xyxy, confs):
                x1, y1, x2, y2 = (float(v) for v in box)
                out.append(((x1, y1, x2, y2), float(c)))
        # confidence 내림차순 정렬
        out.sort(key=lambda t: t[1], reverse=True)
        return out


class BreedEncoder:
    """Stage 2 — Breed Encoder (README 2).

    DINOv2 ViT-S/14 (frozen, self-supervised)로 crop을 384-d embedding으로 변환.
    공식 저장소: https://github.com/facebookresearch/dinov2
    학습 0회 — eval 모드 + no_grad로만 사용한다.
    """

    # ImageNet 정규화 상수 (DINOv2 공식 transform과 동일)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # https://github.com/facebookresearch/dinov2 — torch.hub 로딩 (ViT-S/14, 384-d)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()  # frozen — 학습 없음
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    def _preprocess(self, crop: np.ndarray, size: int) -> torch.Tensor:
        """crop(RGB uint8) → 정규화된 (3, size, size) 텐서.

        ViT-S/14는 patch 크기 14이므로 입력 변은 14의 배수여야 한다 (518 = 14 * 37).
        """
        if size % 14 != 0:
            raise ValueError(f"입력 크기는 14의 배수여야 합니다: {size}")
        t = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().unsqueeze(0)
        # bilinear resize → [0,1] 스케일 → ImageNet mean/std 정규화
        t = torch.nn.functional.interpolate(
            t, size=(size, size), mode="bilinear", align_corners=False
        )
        t = t / 255.0
        mean = torch.from_numpy(self._MEAN).view(1, 3, 1, 1)
        std = torch.from_numpy(self._STD).view(1, 3, 1, 1)
        return (t - mean) / std

    @torch.no_grad()
    def encode(self, crop: np.ndarray, size: int = 518) -> np.ndarray:
        """단일 crop → (384,) float32 L2 정규화 embedding."""
        x = self._preprocess(crop, size).to(self.device)
        emb = self.model(x).squeeze(0).cpu().numpy().astype(np.float32)
        # L2 정규화 (README 2 — "crop → 384-d embedding → L2 정규화")
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-12)

    @torch.no_grad()
    def encode_batch(
        self, crops: list[np.ndarray], batch_size: int = 32, size: int = 518
    ) -> np.ndarray:
        """여러 crop을 batch로 인코딩 → (N, 384) float32, 각 행 L2 정규화."""
        embs: list[np.ndarray] = []
        for i in range(0, len(crops), batch_size):
            batch = torch.cat(
                [self._preprocess(c, size) for c in crops[i : i + batch_size]], dim=0
            ).to(self.device)
            out = self.model(batch).cpu().numpy().astype(np.float32)
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
            embs.append(out)
        return (
            np.concatenate(embs, axis=0) if embs else np.zeros((0, 384), dtype=np.float32)
        )


def _l2norm(v: np.ndarray) -> np.ndarray:
    """L2 정규화 (0-벡터 보호)."""
    return v / max(float(np.linalg.norm(v)), 1e-12)


def build_prototypes(
    embs_by_class: dict[str, np.ndarray],
    cap: int = 50,
    outlier_frac: float = 0.05,
) -> dict:
    """Stage 3 — Breed Prototype Generation (README 3).

    절차 (클래스별):
      1) 이미지 캡 50장 (README: "데이터가 많아도 50장 캡 가능")
      2) L2 정규화 → 평균 → 재정규화
      3) 전체 평균 centering (CL2N: 평균 빼고 재정규화)
      4) centroid와 cosine 하위 outlier_frac(기본 5%)를 outlier로 제거 후 재계산

    Args:
        embs_by_class: {breed: (N, 384) embedding 배열}
        cap: 클래스당 사용 이미지 상한
        outlier_frac: centroid 대비 cosine 하위 제거 비율

    Returns:
        {"prototypes": {breed: (384,) float32}, "global_mean": (384,) float32}
    """
    # 캡 적용 + 각 행 L2 정규화
    normed: dict[str, np.ndarray] = {}
    for breed, embs in embs_by_class.items():
        e = np.asarray(embs, dtype=np.float32)[:cap]
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        normed[breed] = e

    # CL2N용 전체 평균 (모든 클래스 embedding의 평균)
    global_mean = np.concatenate(list(normed.values()), axis=0).mean(axis=0).astype(np.float32)

    def _proto(e: np.ndarray) -> np.ndarray:
        """정규화 → 평균 → 재정규화 → CL2N(전체 평균 빼고 재정규화)."""
        centroid = _l2norm(e.mean(axis=0))
        return _l2norm(centroid - global_mean).astype(np.float32)

    prototypes: dict[str, np.ndarray] = {}
    for breed, e in normed.items():
        # 1차 centroid 계산
        centroid = _l2norm(e.mean(axis=0))
        # centroid와의 cosine 하위 outlier_frac 제거 (README 3 — outlier 제거 후 재계산)
        n_drop = int(len(e) * outlier_frac)
        if n_drop > 0 and len(e) - n_drop >= 1:
            cos = e @ centroid
            keep = np.argsort(cos)[n_drop:]  # cosine 오름차순에서 하위 n_drop 제외
            e = e[keep]
        prototypes[breed] = _proto(e)

    return {"prototypes": prototypes, "global_mean": global_mean}


def similarity_scores(emb: np.ndarray, proto: dict) -> dict[str, float]:
    """query embedding vs 각 prototype cosine 유사도 (README 4~5).

    prototype과 동일한 CL2N centering(전체 평균 빼고 재정규화)을 query에도 적용한다
    — train = inference 처리 동일 제1원칙.
    """
    q = _l2norm(np.asarray(emb, dtype=np.float32) - proto["global_mean"])
    # prototype은 이미 L2 정규화되어 있으므로 내적 = cosine similarity
    return {breed: float(q @ p) for breed, p in proto["prototypes"].items()}


def predict(
    emb: np.ndarray,
    proto: dict,
    threshold: float,
    top_k: int = 3,
    temperature: float = 0.1,
) -> dict:
    """Inference — Unknown 분기 + Top-K + temperature softmax calibration (README 4~5).

    max similarity < threshold → Unknown (OOD 거절, 필수 분기).
    통과 시 전체 유사도에 temperature softmax를 적용해 상대적 퍼센트로 변환한다.
    출력은 DNA 비율이 아니라 Phenotype Similarity Score이다 (README 6).

    Returns:
        {"unknown": bool, "max_sim": float, "topk": [(breed, score_pct), ...]}
    """
    sims = similarity_scores(emb, proto)
    breeds = list(sims.keys())
    vals = np.array([sims[b] for b in breeds], dtype=np.float32)
    max_sim = float(vals.max()) if len(vals) else float("-inf")

    # Unknown/OOD 분기 — 없으면 고양이 사진에도 "Golden 46%"가 나온다 (README 4)
    if max_sim < threshold:
        return {"unknown": True, "max_sim": max_sim, "topk": []}

    # Calibration: temperature softmax (README 5 — Similarity → Calibration → Normalization)
    logits = vals / temperature
    logits -= logits.max()  # 수치 안정화
    probs = np.exp(logits)
    probs /= probs.sum()

    order = np.argsort(-probs)[:top_k]
    topk = [(breeds[i], float(probs[i] * 100.0)) for i in order]
    return {"unknown": False, "max_sim": max_sim, "topk": topk}
