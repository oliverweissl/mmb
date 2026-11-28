"""Text-mutation operators.

Run time: :class:`QuestionPoolOperator` (the continuous text block, genes (t, u)) is the only text operator.
The synonym MLM base and the semantic substitution operator (semantic.py) stay for scripts/build_question_pool.py,
which builds the validated question pool offline.
"""

from .base import OperatorContext, TextOperator, severity_to_k_max
from .question_pool import QuestionPoolOperator
from .synonym import NEGATION_PREFIX_RE, SynonymContext, SynonymOperator

__all__ = [
    "NEGATION_PREFIX_RE",
    "OperatorContext",
    "QuestionPoolOperator",
    "SynonymContext",
    "SynonymOperator",
    "TextOperator",
    "severity_to_k_max",
]
