"""Deprecated compatibility facade.

새 코드에서는 stage별 모듈을 직접 import한다:
- detection.DogDetector
- encoder.BreedEncoder
- prototype.build_prototypes
- scoring.similarity_scores / scoring.predict

기존 코드의 `from model import ...`가 즉시 깨지지 않도록 re-export만 유지한다.
"""

from detection import DogDetector
from encoder import BreedEncoder
from prototype import build_prototypes
from scoring import predict, similarity_scores

__all__ = [
    "DogDetector",
    "BreedEncoder",
    "build_prototypes",
    "similarity_scores",
    "predict",
]
