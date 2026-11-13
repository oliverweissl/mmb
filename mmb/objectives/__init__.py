"""VLM boundary-testing objectives."""

from mmb.core import CriterionCollection, MatrixDistance

from .targeted_balance import SoftmaxBalance, TargetedBalance
from .text_embedding_distance import TextEmbeddingDistance

BOUNDARY_OBJECTIVES = {"logprob_gap": TargetedBalance, "softmax_balance": SoftmaxBalance}


def build_criteria(
    modality: str = "joint",
    boundary_objective: str = "logprob_gap",
    proximity_only: bool = False,
) -> list[type]:
    """Ordered criterion classes for a run; the boundary-proximity axis is always last.

    :param modality: ``joint`` | ``image_only`` | ``text_only`` | ``grounding``.
    :param boundary_objective: key of :data:`BOUNDARY_OBJECTIVES` (E1/E2 ablation).
    :param proximity_only: single-objective run (use with ``optimizer.algorithm: ga``).
    """
    crit: list[type] = []
    if not proximity_only:
        if modality != "text_only":
            crit.append(MatrixDistance)
        if modality != "image_only":
            crit.append(TextEmbeddingDistance)
    crit.append(BOUNDARY_OBJECTIVES[boundary_objective])
    return crit


__all__ = [
    "BOUNDARY_OBJECTIVES",
    "CriterionCollection",
    "MatrixDistance",
    "SoftmaxBalance",
    "TargetedBalance",
    "TextEmbeddingDistance",
    "build_criteria",
]
