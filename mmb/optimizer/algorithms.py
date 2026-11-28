"""Search algorithms selectable via ``optimizer.algorithm`` (E1 baselines).

Real-coded genotype (2026-11-28): both custom baselines work on float genes.
``random`` redraws from the shared initial sampler; ``hillclimb`` draws its
mutants with the shared mutation operator (block-wise PM), so it differs
from the GA arms only in selection, not in the variation operators.

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
from pymoo.operators.sampling.rnd import FloatRandomSampling


class _KeepAll(Survival):
    """No selection: the survivors are the evaluated offspring."""

    def _do(self, problem, pop, n_survive=None, **kwargs):
        return pop[:n_survive]


class RandomSearch(GeneticAlgorithm):
    """Selection-free baseline: every generation is a fresh sample.

    The sampler passed as ``sampling`` is reused each generation when it is a
    pymoo ``Sampling``; a pre-built matrix (used for generation 0 only) falls
    back to uniform float sampling afterwards.
    """

    def __init__(self, pop_size: int = 50, sampling=None, **kwargs) -> None:
        sampling = sampling if sampling is not None else FloatRandomSampling()
        super().__init__(pop_size=pop_size, sampling=sampling, survival=_KeepAll(),
                         eliminate_duplicates=False, **{k: v for k, v in kwargs.items()
                                                        if k not in ("crossover", "mutation")})
        self._sampler = sampling if isinstance(sampling, Sampling) else FloatRandomSampling()

    def _infill(self):
        x = self._sampler(self.problem, self.n_offsprings).get("X")
        return Population.new(X=np.asarray(x))


class HillClimb(GeneticAlgorithm):
    """(1+lambda) hill climber on the scalar proximity |g| (use with
    optimizer.proximity_only=true): every generation draws lambda = pop_size
    mutants of the incumbent with the shared mutation operator (the one the
    GA arms use) and keeps the best of incumbent + mutants. Classic SBST
    local-search baseline; generation 0 uses the same sampler as the other
    arms, so the comparison isolates the selection scheme."""

    def __init__(self, pop_size: int = 50, sampling=None, mutation=None, seed=None, **kwargs) -> None:
        sampling = sampling if sampling is not None else FloatRandomSampling()
        super().__init__(pop_size=pop_size, sampling=sampling, survival=_KeepAll(),
                         eliminate_duplicates=False, **{k: v for k, v in kwargs.items()
                                                        if k not in ("crossover",)})
        if mutation is None:
            from pymoo.operators.mutation.pm import PM
            mutation = PM(eta=20)
        self._mut = mutation
        self._best_x = None
        self._best_f = np.inf

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
        n = self.n_offsprings
        base = self._best_x if self._best_x is not None else self.pop.get("X")[0]
        pop = Population.new(X=np.repeat(np.asarray(base, dtype=float)[None, :], n, axis=0))
        # every mutant must differ from the incumbent: apply with probability 1
        X = self._mut._do(self.problem, pop.get("X"), random_state=self.random_state)
        return Population.new(X=np.asarray(X, dtype=float))


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
