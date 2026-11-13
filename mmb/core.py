"""Minimal base abstractions (vendored subset of SMOO, which is no longer a dependency).

``Criterion``/``CriterionCollection`` (objectives), ``SUT``, ``Manipulator``,
``Optimizer``/``OptimizerCandidate`` and the ``SMOO`` workflow base.  Optional
heavy generators (StyleGAN-XL, diffusion) are still imported lazily from the
external ``smoo`` package by their backends.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


# --------------------------------------------------------------- objectives
class Criterion(ABC):
    """A named objective. Subclasses set ``_name`` and implement ``evaluate``."""

    _name: str

    def __init__(self, inverse: bool = False, allow_batched: bool = False) -> None:
        self._inverse = inverse
        self._allow_batched = allow_batched
        self._name = self._name + "Inv" if inverse else self._name

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Union[float, list[float]]: ...

    @property
    def name(self) -> str:
        return self._name

    def precondition(self, **kwargs: Any) -> None:
        """Optional hook run once per seed before evaluation."""


class CriterionCollection:
    def __init__(self, *criteria: Criterion) -> None:
        self._criteria = list(criteria)
        self._results: dict[str, Any] = {}

    def evaluate_all(self, **kwargs: Any) -> None:
        for c in self._criteria:
            self._results[c.name] = c.evaluate(**kwargs)

    @property
    def results(self) -> dict[str, Any]:
        return self._results

    @property
    def names(self) -> list[str]:
        return [c.name for c in self._criteria]

    @property
    def num_objectives(self) -> int:
        return len(self._criteria)


class MatrixDistance(Criterion):
    """Channel-wise normalised matrix norm between two image batches in [0, 1]."""

    _name = "MatrixDistance"
    _norms = ("fro", "nuc", "inf", "-inf", "1", "-1", "2", "-2")

    def __init__(self, inverse: bool = False, norm: str = "fro", return_tensor: bool = False) -> None:
        assert norm in self._norms, f"unsupported norm {norm!r}"
        super().__init__(inverse, allow_batched=True)
        self.norm = norm
        self._name += f"_{norm}"
        self._return_tensor = return_tensor

    def evaluate(
        self, *, images: list[Tensor], batch_dim: Optional[int] = None, **_: Any
    ) -> Union[list[float], float, Tensor]:
        if batch_dim is None:
            images = [i.unsqueeze(0) for i in images]
        elif batch_dim != 0:
            images = [i.transpose(0, batch_dim) for i in images]
        i1, i2, *_rest = images
        ub = torch.linalg.matrix_norm(torch.ones_like(i1), self.norm, dim=(-2, -1))
        scaled = torch.linalg.matrix_norm(i1 - i2, self.norm, dim=(-2, -1)) / ub
        res = torch.abs(float(self._inverse) - scaled.mean(dim=1))
        if self._return_tensor:
            return res
        out = res.float().tolist()
        return out[0] if batch_dim is None else out


# ----------------------------------------------------------------- SUT / IO
class SUT(ABC):
    """System under test."""

    _device: torch.device
    _dtype: torch.dtype
    _batch_size: int

    @abstractmethod
    def process_input(self, inpt: Any) -> Any: ...

    @abstractmethod
    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]: ...


class Manipulator(ABC):
    """Maps a genotype to model inputs."""

    @abstractmethod
    def manipulate(self, candidates: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def get_images(self, z: Tensor) -> Tensor: ...


# ---------------------------------------------------------------- optimizer
@dataclass
class OptimizerCandidate:
    solution: Optional[NDArray]
    fitness: Union[tuple[float, ...], float, list[float]]
    data: Optional[Any] = None
    parents: Optional[list["OptimizerCandidate"]] = None

    def __post_init__(self) -> None:
        if isinstance(self.fitness, float):
            self.fitness = (self.fitness,)
        elif isinstance(self.fitness, list):
            self.fitness = tuple(self.fitness)


class Optimizer(ABC):
    """Optimizer base keeping a running Pareto archive of ``best_candidates``."""

    _best_candidates: list[OptimizerCandidate]
    _previous_best: list[OptimizerCandidate]
    _x_current: NDArray
    _fitness: tuple[NDArray, ...]
    _bounds: tuple[int, int]
    _n_var: int

    def __init__(self, num_objectives: int) -> None:
        self._optimizer_type = type(self)
        self._num_objectives = num_objectives

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def get_x_current(self) -> NDArray: ...

    def assign_fitness(self, fitness: Iterable[NDArray], *data: Optional[Iterable[Any]]) -> None:
        fitness = tuple(f.numpy() if isinstance(f, Tensor) else f for f in fitness)
        assert len(fitness) == self._num_objectives, (
            f"{len(fitness)} fitness values found, {self._num_objectives} expected."
        )
        self._fitness = fitness
        new = np.ascontiguousarray(fitness).T
        old = np.ascontiguousarray([c.fitness for c in self._best_candidates])
        metrics = np.vstack((new, old))
        extra: list[Any] = [None] * new.shape[0] if not data else list(zip(*data))
        data_all = extra + [c.data for c in self._best_candidates]
        sols = np.vstack((self._x_current, np.array([c.solution for c in self._best_candidates])))
        le = metrics[:, None, :] <= metrics[None, :, :]
        lt = metrics[:, None, :] < metrics[None, :, :]
        dominated = (le.all(2) & lt.any(2)).any(0)
        keep = np.flatnonzero(~dominated)
        self._previous_best = self._best_candidates
        self._best_candidates = [
            OptimizerCandidate(solution=sols[i], fitness=metrics[i], data=data_all[i]) for i in keep
        ]

    def reset(self) -> None:
        self._x_current = np.random.uniform(*self._bounds, size=self._x_current.shape)
        self._best_candidates = [OptimizerCandidate(self._x_current[0], [np.inf] * self._num_objectives)]
        self._previous_best = self._best_candidates.copy()

    @property
    def best_candidates(self) -> list[OptimizerCandidate]:
        return self._best_candidates

    @property
    def previous_best(self) -> list[OptimizerCandidate]:
        return self._previous_best

    @property
    def optimizer_type(self) -> type:
        return self._optimizer_type

    @property
    def n_var(self) -> int:
        return self._n_var

# ----------------------------------------------------------------- workflow
EarlyTerm = Optional[Callable[[Any], tuple[bool, Optional[Any]]]]


class SMOO(ABC):
    """Workflow base: wires SUT, manipulator, optimizer and objectives."""

    def __init__(
        self,
        *,
        sut: SUT,
        manipulator: Manipulator,
        optimizer: Optimizer,
        objectives: CriterionCollection,
        restrict_classes: Optional[list[int]] = None,
        use_wandb: bool = False,
        early_termination: EarlyTerm = None,
    ) -> None:
        self._sut = sut
        self._manipulator = manipulator
        self._optimizer = optimizer
        self._objectives = objectives
        self._restrict_classes = restrict_classes
        self._use_wandb = use_wandb
        self._early_termination = early_termination or (lambda _: (False, None))

    @abstractmethod
    def test(self) -> None: ...

    @staticmethod
    def _cleanup() -> None:
        gc.collect()
        torch.cuda.empty_cache()
