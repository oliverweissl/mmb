"""Real-coded pymoo optimizer for the continuous genotype (all GA-type arms).

Replaces :class:`~mmb.optimizer.discrete_pymoo_optimizer.DiscretePymooOptimizer`
for the genotype of docs/METHODOLOGY_REVIEW_2026-11-28.md: float genes in a
box, two modality blocks. Every method uses the same three shared pieces:

* initial sampler: :class:`~mmb.optimizer.sparse_sampling.MagnitudeStratifiedSampling`;
* crossover: SBX (eta 15), :func:`~mmb.optimizer.variation.build_crossover`;
* mutation: block-wise PM (eta 20, coin flip over blocks),
  :class:`~mmb.optimizer.variation.BlockPolynomialMutation`.

Algorithms: ``agemoea2``, ``nsga2``, ``ga`` (single objective; the
MIMICRY-style arm is this optimizer with the softmax-balance objective),
``random`` and ``hillclimb`` (:mod:`mmb.optimizer.algorithms`).

Per seed the tester calls :meth:`set_space` (or the legacy name
:meth:`update_gene_bounds`) with a :class:`~mmb.optimizer.genotype.GenotypeSpace`
or with the manipulator itself.
"""
from __future__ import annotations

import logging
from typing import Any, Type

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from mmb.core import Optimizer, OptimizerCandidate

from .algorithms import resolve_algorithm
from .genotype import GenotypeSpace
from .sparse_sampling import MagnitudeStratifiedSampling
from .variation import (DEFAULT_CROSSOVER_ETA, DEFAULT_CROSSOVER_PROB, DEFAULT_MUTATION_ETA,
                        build_crossover, build_mutation)

logger = logging.getLogger(__name__)


def as_space(obj) -> GenotypeSpace:
    """A GenotypeSpace, or one read from a manipulator-like object."""
    return obj if isinstance(obj, GenotypeSpace) else GenotypeSpace.from_manipulator(obj)


class RealPymooOptimizer(Optimizer):
    """pymoo GA-type search on the real-coded, two-block genotype.

    :param space: genotype space; may be set later per seed (:meth:`set_space`).
    :param num_objectives: number of objectives (minimised).
    :param pop_size: individuals per generation.
    :param algorithm: pymoo class or name (see :mod:`mmb.optimizer.algorithms`).
    :param algo_params: extra constructor kwargs; ``sampling``/``crossover``/
        ``mutation`` override the shared operators (do not, for balanced runs).
    :param mutation_prob: per-gene probability inside the chosen block
        (``None``: 1/len(block)).
    :param mutation_eta: PM distribution index (default 20).
    :param crossover_prob: SBX per-mating probability (default 0.9).
    :param crossover_eta: SBX distribution index (default 15).
    :param seed: seed of the pymoo RNG and of the shared sampler.
    :param gene_bounds: ignored (legacy placeholder of the integer runner).
    """

    def __init__(self, space: GenotypeSpace | None = None, num_objectives: int = 1,
                 pop_size: int = 50, algorithm: Type[GeneticAlgorithm] | str = AGEMOEA2,
                 algo_params: dict[str, Any] | None = None, mutation_prob: float | None = None,
                 mutation_eta: float = DEFAULT_MUTATION_ETA,
                 crossover_prob: float = DEFAULT_CROSSOVER_PROB,
                 crossover_eta: float = DEFAULT_CROSSOVER_ETA, seed: int | None = None,
                 gene_bounds: Any = None) -> None:
        super().__init__(num_objectives)
        self._pop_size = pop_size
        self._algorithm_cls = resolve_algorithm(algorithm) if isinstance(algorithm, str) else algorithm
        if self._algorithm_cls.__name__ == "GA" and num_objectives != 1:
            raise ValueError("algorithm 'ga' is single-objective; configure exactly one objective")
        self._algo_params = dict(algo_params) if algo_params else {}
        self._seed = seed
        self._mutation_op = build_mutation(space, prob=mutation_prob, eta=mutation_eta)
        self._crossover_op = build_crossover(prob=crossover_prob, eta=crossover_eta)
        self._space: GenotypeSpace | None = None
        self._n_var = 0
        if space is not None:
            self.set_space(space)

    # ------------------------------------------------------------ setup
    def _sampler(self):
        s = self._algo_params.get("sampling")
        if s is None:
            s = MagnitudeStratifiedSampling(seed=self._seed)
            self._algo_params["sampling"] = s
        if hasattr(s, "bind"):
            s.bind(self._space)
        return s

    def _build_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"pop_size": self._pop_size, "sampling": self._sampler(),
                                  "crossover": self._crossover_op, "mutation": self._mutation_op}
        params.update({k: v for k, v in self._algo_params.items() if k != "sampling"})
        return params

    def _problem_for(self, n_obj: int) -> Problem:
        sp = self._space
        return Problem(n_var=sp.n_var, n_obj=n_obj, xl=sp.xl.copy(), xu=sp.xu.copy(), vtype=float)

    def _setup_pymoo(self, n_obj: int) -> None:
        self._problem = self._problem_for(n_obj)
        self._pymoo_algo = self._algorithm_cls(**self._build_params())
        self._pymoo_algo.setup(self._problem, termination=NoTermination(), seed=self._seed)
        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._space.sanitize(self._pop_current.get("X"))

    def _init_algorithm(self) -> None:
        self._setup_pymoo(self._num_objectives)
        self._best_candidates = [OptimizerCandidate(solution=self._x_current[0].copy(),
                                                    fitness=[np.inf] * self._num_objectives)]
        self._previous_best = self._best_candidates.copy()

    def set_space(self, space) -> None:
        """Bind the genotype space of the current seed and redraw generation 0."""
        self._space = as_space(space)
        self._n_var = self._space.n_var
        self._mutation_op.bind(self._space)
        self._init_algorithm()

    def update_gene_bounds(self, space) -> None:
        """Legacy name of :meth:`set_space` (the tester calls it per seed)."""
        self.set_space(space)

    def set_sampling(self, sampling: Any) -> None:
        """Replace the initial sampler (takes effect at the next set_space).

        Only real-coded samplers that can be bound to the genotype space
        (``bind(space)``, e.g. MagnitudeStratifiedSampling) are accepted: the
        legacy integer samplers (sparse, sparse_multitier, ...) would draw
        integer ranks into the real genotype.
        """
        if not hasattr(sampling, "bind"):
            raise ValueError(
                f"{type(sampling).__name__} is an integer-genotype sampler; the real-coded "
                "optimizer needs optimizer.sampling.mode = 'stratified_magnitude'")
        self._algo_params = {**self._algo_params, "sampling": sampling}

    # ------------------------------------------------------------ Optimizer API
    def update(self) -> None:
        static = StaticProblem(self._problem, F=np.column_stack(self._fitness))
        Evaluator().eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)
        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._space.sanitize(self._pop_current.get("X"))

    def get_x_current(self) -> NDArray[np.float64]:
        return self._x_current

    def reset(self) -> None:
        self._init_algorithm()

    # ------------------------------------------------------------ properties
    @property
    def space(self) -> GenotypeSpace | None:
        return self._space

    @property
    def pop_size(self) -> int:
        return self._pop_size

    @property
    def mutation(self):
        return self._mutation_op

    @property
    def crossover(self):
        return self._crossover_op

    @property
    def gene_bounds(self) -> NDArray[np.float64]:
        """(n_var, 2) array of [xl, xu] (legacy accessor)."""
        return np.column_stack([self._space.xl, self._space.xu])
