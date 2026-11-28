"""DeepJanus-style search (Riccio and Tonella, ESEC/FSE 2020) on the real-coded
genotype. DeepJanus evolves inputs towards the frontier of behaviours with two
objectives: closeness to the frontier and sparseness w.r.t. an archive of
frontier inputs already found (novelty), optimised with NSGA-II. Here:

* closeness  = the proximity objective |g| supplied by the tester (proximity_only);
* sparseness = range-normalised Euclidean distance of a genotype to the nearest
  archive member, ``||(x - a) / (xu - xl)|| / sqrt(n_var)`` in [0, 1] (1.0
  while the archive is empty), maximised;
* archive    = genotypes with |g| <= archive_tau that are at least
  archive_min_dist away from every member (DeepJanus' archive threshold).

Sampler, SBX and block-wise PM are the shared operators of every method
(:class:`~mmb.optimizer.real_pymoo_optimizer.RealPymooOptimizer`). The
tester-facing interface is unchanged: one fitness array in, best candidates
on the proximity objective out.
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem

from mmb.core import OptimizerCandidate

from .real_pymoo_optimizer import RealPymooOptimizer


class DeepJanusOptimizer(RealPymooOptimizer):
    def __init__(self, *args, archive_tau: float = 0.5, archive_min_dist: float = 0.02,
                 **kwargs) -> None:
        kwargs["algorithm"] = NSGA2
        self._archive: list[np.ndarray] = []
        self._archive_tau = archive_tau
        self._archive_min_dist = archive_min_dist
        self._aug_fitness = None
        super().__init__(*args, **kwargs)

    # NSGA-II runs on (closeness, -sparseness): two internal objectives.
    def _init_algorithm(self) -> None:
        self._setup_pymoo(2)
        self._best_candidates = [OptimizerCandidate(solution=self._x_current[0].copy(),
                                                    fitness=[np.inf] * self._num_objectives)]
        self._previous_best = self._best_candidates.copy()
        self._archive = []

    def _sparseness(self, x: np.ndarray) -> np.ndarray:
        if not self._archive:
            return np.ones(len(x))
        sp = self._space
        rng = np.where(sp.xu > sp.xl, sp.xu - sp.xl, 1.0)
        A = np.stack(self._archive)
        d = np.sqrt((((x[:, None, :] - A[None, :, :]) / rng) ** 2).mean(axis=2))
        return d.min(axis=1)

    def assign_fitness(self, fitness, *data) -> None:
        super().assign_fitness(fitness, *data)
        prox = np.asarray(list(fitness)[0], dtype=float)
        x = self._x_current
        sparse = self._sparseness(x)
        self._aug_fitness = (prox, -sparse)
        for i in np.argsort(prox):
            if prox[i] > self._archive_tau:
                break
            if not self._archive or self._sparseness(x[i:i + 1])[0] >= self._archive_min_dist:
                self._archive.append(x[i].copy())

    def update(self) -> None:
        static = StaticProblem(self._problem, F=np.column_stack(self._aug_fitness))
        Evaluator().eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)
        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._space.sanitize(self._pop_current.get("X"))

    @property
    def archive_size(self) -> int:
        return len(self._archive)
