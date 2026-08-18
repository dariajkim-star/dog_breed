# -*- coding: utf-8 -*-
"""Stage 2 — Breed Encoder.

DINOv2 ViT-S/14 frozen encoder로 dog crop을 384-d L2 normalized embedding으로 변환한다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class BreedEncoder:
    """DINOv2 ViT-S/14 frozen feature encoder.

    공식 저장소: https://github.com/facebookresearch/dinov2
    - 입력 한 변은 patch 크기 14의 배수여야 한다 (518 = 14 × 37)
    - 전처리는 DINOv2 공식 transform과 동일한 ImageNet mean/std 정규화 사용
    """

    # ImageNet 통계값 — DINOv2가 학습 때 쓴 정규화 기준이라 추론 때도 똑같이 맞춰야 함
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    EMBEDDING_DIM = 384
    PATCH_SIZE = 14

    def __init__(self, device: Optional[str] = None) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device)

    def _preprocess(self, crop: np.ndarray, size: int):
        """crop(RGB uint8) → normalized BCHW tensor."""
        if size % self.PATCH_SIZE != 0:
            raise ValueError(
                f"입력 크기는 {self.PATCH_SIZE}의 배수여야 합니다: {size}"
            )

        torch = self._torch
        tensor = (
            torch.from_numpy(np.ascontiguousarray(crop))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
        )
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor / 255.0
        mean = torch.from_numpy(self._MEAN).view(1, 3, 1, 1)
        std = torch.from_numpy(self._STD).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def encode(self, crop: np.ndarray, size: int = 518) -> np.ndarray:
        """단일 crop → (384,) float32 L2 normalized embedding."""
        torch = self._torch
        with torch.no_grad():
            x = self._preprocess(crop, size).to(self.device)
            emb = self.model(x).squeeze(0).cpu().numpy().astype(np.float32)

        # L2 정규화 — max(..., 1e-12)는 0으로 나누기 방지 안전장치
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-12)

    def encode_batch(
        self,
        crops: list[np.ndarray],
        batch_size: int = 32,
        size: int = 518,
    ) -> np.ndarray:
        """여러 crop → (N, 384) float32, row-wise L2 normalized."""
        torch = self._torch
        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                # batch_size장씩 잘라서 전처리 → 하나의 배치 tensor로 합친다
                tensor_list = []
                for crop in crops[start : start + batch_size]:
                    tensor_list.append(self._preprocess(crop, size))
                batch = torch.cat(tensor_list, dim=0).to(self.device)

                out = self.model(batch).cpu().numpy().astype(np.float32)
                # 행별 L2 정규화 (1e-12는 0 나누기 방지)
                lengths = np.linalg.norm(out, axis=1, keepdims=True)
                out = out / np.maximum(lengths, 1e-12)
                embeddings.append(out)

        if not embeddings:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
        return np.concatenate(embeddings, axis=0)
