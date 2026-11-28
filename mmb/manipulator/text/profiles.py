"""Profile library + operator-spec resolution + builder.

The composite manipulator (:mod:`.composite`) consumes resolved
:class:`OperatorSpec` lists. This module loads them from the YAML
library, applies overrides, and instantiates the concrete operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorSpec:
    """One row of a profile: operator name, severity, and extra kwargs."""

    name: str
    severity: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextProfile:
    """A named bundle of :class:`OperatorSpec` entries."""

    name: str
    operators: tuple[OperatorSpec, ...]


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


def load_profile_library(path: Path | str) -> dict[str, TextProfile]:
    """Load a profile-library YAML into ``{name: TextProfile}``.

    File format::

        profiles:
          casual_typing:
            operators:
              - {name: saliency, severity: 0.25}
              - {name: synonym, severity: 0.10}
          ...
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Profile library not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    library: dict[str, TextProfile] = {}
    for prof_name, prof_data in (raw.get("profiles") or {}).items():
        ops_data = prof_data.get("operators", [])
        ops = tuple(_spec_from_dict(item) for item in ops_data)
        library[prof_name] = TextProfile(name=prof_name, operators=ops)
    return library


def _spec_from_dict(item: dict[str, Any]) -> OperatorSpec:
    if "name" not in item or "severity" not in item:
        raise ValueError(
            f"operator entry missing 'name' or 'severity': {item!r}"
        )
    extras = {k: v for k, v in item.items() if k not in ("name", "severity")}
    return OperatorSpec(
        name=str(item["name"]),
        severity=float(item["severity"]),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------


def resolve_profile(
    library: dict[str, TextProfile],
    profile_name: str | None,
    operators: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[OperatorSpec, ...]:
    """Resolve a config block into the final list of operator specs.

    Priority: explicit *operators* > *profile_name + overrides* > *profile_name*.

    :raises KeyError: if *profile_name* is not in *library*.
    :raises ValueError: if neither *profile_name* nor *operators* is provided.
    """
    if operators:
        return tuple(_spec_from_dict(item) for item in operators)
    if profile_name is None:
        raise ValueError(
            "TextCompositeConfig requires either a profile name or an "
            "explicit operators list"
        )
    if profile_name not in library:
        raise KeyError(
            f"Profile {profile_name!r} not in library; available: "
            f"{sorted(library)}"
        )
    base = library[profile_name].operators
    if not overrides:
        return base
    return tuple(
        OperatorSpec(
            name=op.name,
            severity=float(overrides.get(op.name, {}).get("severity", op.severity)),
            extras={
                **op.extras,
                **{
                    k: v
                    for k, v in overrides.get(op.name, {}).items()
                    if k != "severity"
                },
            },
        )
        for op in base
    )


# ---------------------------------------------------------------------------
# Builder (lazy operator imports)
# ---------------------------------------------------------------------------


def build_operators_from_specs(
    specs: tuple[OperatorSpec, ...],
    content_pos: frozenset[str] | None = None,
    device: str | None = None,
    redis_url: str | None = None,
) -> list[Any]:
    """Instantiate the run-time text operators for a spec list (question_pool only; severity 0 skips a spec).

    ``content_pos``, ``device`` and ``redis_url`` are accepted for interface compatibility and unused: the
    question pool is precomputed (scripts/build_question_pool.py), so the run-time operator loads no model.

    :raises ValueError: on unknown or removed operator names.
    """
    from .operators.question_pool import POOL, QuestionPoolOperator

    out: list[Any] = []
    for spec in specs:
        if spec.severity == 0.0:
            continue
        if spec.name == "question_pool":
            out.append(QuestionPoolOperator(
                severity=spec.severity, pool=spec.extras.get("pool", POOL), w_txt=float(spec.extras.get("w_txt", 1.0)),
                d_max=None if spec.extras.get("d_max") is None else float(spec.extras["d_max"]),
                n_bands=None if spec.extras.get("n_bands") is None else int(spec.extras["n_bands"])))
        else:
            raise ValueError(
                f"unknown text operator {spec.name!r}; the run-time text manipulator supports question_pool only "
                f"(word substitutions and paraphrases are built offline into configs/text/question_pool.json)"
            )
    return out
