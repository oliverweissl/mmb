"""Search algorithms selectable via ``optimizer.algorithm`` (E1 baselines).

``agemoea2`` is the thesis default; ``nsga2`` and ``ga`` (single objective)
are standard pymoo baselines; ``random`` redraws every generation from the
initial sampler with no selection pressure -- the equal-budget control that
isolates what directed search contributes.
"""

from __future__ import annotations

from typing import Type

import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.operators.sampling.rnd import IntegerRandomSampling


class _KeepAll(Survival):
    """No selection: the survivors are the evaluated offspring."""

    def _do(self, problem, pop, n_survive=None, **kwargs):
        return pop[:n_survive]


class RandomSearch(GeneticAlgorithm):
    """Selection-free baseline: every generation is a fresh sample.

    The sampler passed as ``sampling`` is reused each generation when it is a
    pymoo ``Sampling``; a pre-built matrix (used for generation 0 only) falls
    back to uniform integer sampling afterwards.
    """

    def __init__(self, pop_size: int = 50, sampling=None, **kwargs) -> None:
        sampling = sampling if sampling is not None else IntegerRandomSampling()
        super().__init__(pop_size=pop_size, sampling=sampling, survival=_KeepAll(),
                         eliminate_duplicates=False, **{k: v for k, v in kwargs.items()
                                                        if k not in ("crossover", "mutation")})
        self._sampler = sampling if isinstance(sampling, Sampling) else IntegerRandomSampling()

    def _infill(self):
        x = self._sampler(self.problem, self.n_offsprings).get("X")
        return Population.new(X=np.asarray(x))


ALGORITHMS: dict[str, Type[GeneticAlgorithm]] = {
    "agemoea2": AGEMOEA2,
    "nsga2": NSGA2,
    "ga": GA,
    "random": RandomSearch,
}


def resolve_algorithm(name: str) -> Type[GeneticAlgorithm]:
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"unknown optimizer.algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
