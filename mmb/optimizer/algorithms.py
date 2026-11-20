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


class HillClimb(GeneticAlgorithm):
    """(1+lambda) hill climber on the scalar proximity |g| (use with
    optimizer.proximity_only=true): every generation draws lambda = pop_size
    mutants of the incumbent (each mutant re-samples k ~ Geom(0.5) >= 1 genes
    uniformly within bounds) and keeps the best of incumbent + mutants.
    Classic SBST local-search baseline; generation 0 uses the same sampler
    as the other arms, so the comparison isolates the search operator."""

    def __init__(self, pop_size: int = 50, sampling=None, seed=None, **kwargs) -> None:
        sampling = sampling if sampling is not None else IntegerRandomSampling()
        super().__init__(pop_size=pop_size, sampling=sampling, survival=_KeepAll(),
                         eliminate_duplicates=False, **{k: v for k, v in kwargs.items()
                                                        if k not in ("crossover", "mutation")})
        self._best_x = None
        self._best_f = np.inf
        self._rng = np.random.default_rng(seed)

    def _update_incumbent(self, pop) -> None:
        F = pop.get("F")
        X = pop.get("X")
        if F is None or len(F) == 0 or F[0] is None:
            return
        f = np.asarray(F, dtype=float).reshape(len(F), -1)[:, 0]
        i = int(np.argmin(f))
        if f[i] < self._best_f:
            self._best_f = float(f[i])
            self._best_x = np.asarray(X[i]).copy()

    def _infill(self):
        self._update_incumbent(self.pop)
        xl = np.asarray(self.problem.xl, dtype=int)
        xu = np.asarray(self.problem.xu, dtype=int)
        n = self.n_offsprings
        base = self._best_x if self._best_x is not None else self.pop.get("X")[0]
        out = np.repeat(np.asarray(base)[None, :], n, axis=0).astype(int)
        d = out.shape[1]
        for r in range(n):
            k = int(min(d, self._rng.geometric(0.5)))
            idx = self._rng.choice(d, size=k, replace=False)
            out[r, idx] = self._rng.integers(xl[idx], xu[idx] + 1)
        return Population.new(X=out)


ALGORITHMS: dict[str, Type[GeneticAlgorithm]] = {
    "agemoea2": AGEMOEA2,
    "nsga2": NSGA2,
    "ga": GA,
    "random": RandomSearch,
    "hillclimb": HillClimb,
}


def resolve_algorithm(name: str) -> Type[GeneticAlgorithm]:
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"unknown optimizer.algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
