"""DeepJanus-style search (Riccio and Tonella, ESEC/FSE 2020), adapted to the
discrete VLM genotype. DeepJanus evolves inputs towards the frontier of behaviours
with two objectives: closeness to the frontier and sparseness w.r.t. an archive of
frontier inputs already found (novelty), optimised with NSGA-II. Here:

* closeness  = the proximity objective |g| supplied by the tester (proximity_only);
* sparseness = normalised Hamming distance of a genotype to the nearest archive
  member (1.0 while the archive is empty), maximised;
* archive    = genotypes with |g| <= archive_tau that are at least archive_min_dist
  away from every member (DeepJanus' archive threshold).

The tester-facing interface is unchanged: one fitness array in, best candidates
on the proximity objective out.
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem
from pymoo.core.termination import NoTermination

from .discrete_pymoo_optimizer import DiscretePymooOptimizer


class DeepJanusOptimizer(DiscretePymooOptimizer):
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
        xl = np.zeros(self._n_var, dtype=np.int64)
        xu = self._gene_bounds - 1
        self._problem = Problem(n_var=self._n_var, n_obj=2, xl=xl, xu=xu, vtype=int)
        self._pymoo_algo = self._algorithm_cls(**self._build_params())
        if self._seed is not None:
            self._pymoo_algo.setup(self._problem, termination=NoTermination(), seed=self._seed)
        else:
            self._pymoo_algo.setup(self._problem, termination=NoTermination())
        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._sanitize(self._pop_current.get("X"))
        from mmb.core import OptimizerCandidate
        self._best_candidates = [OptimizerCandidate(solution=self._x_current[0].copy(),
                                                    fitness=[np.inf] * self._num_objectives)]
        self._previous_best = self._best_candidates.copy()
        self._archive = []

    def _sparseness(self, x: np.ndarray) -> np.ndarray:
        if not self._archive:
            return np.ones(len(x))
        A = np.stack(self._archive)
        d = (x[:, None, :] != A[None, :, :]).mean(axis=2)
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
        self._x_current = self._sanitize(self._pop_current.get("X"))

    @property
    def archive_size(self) -> int:
        return len(self._archive)
