"""VLM boundary-testing objectives."""

from mmb.core import CriterionCollection, MatrixDistance

from .targeted_balance import TargetedBalance
from .text_embedding_distance import TextEmbeddingDistance

__all__ = [
    "CriterionCollection",
    "MatrixDistance",
    "TargetedBalance",
    "TextEmbeddingDistance",
]
