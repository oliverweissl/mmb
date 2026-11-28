"""Shared real-coded variation operators (every method uses these).

* Crossover: simulated binary crossover (SBX, Deb and Agrawal 1995), pymoo's
  :class:`~pymoo.operators.crossover.sbx.SBX` with eta = 15, prob = 0.9,
  per-variable prob 0.5 (pymoo's NSGA-II / AGE-MOEA-II defaults).
* Mutation: polynomial mutation (PM, Deb and Goyal 1996), eta = 20, applied
  block-wise: each offspring first picks ONE modality block with a fair coin
  (user decision 2026-11-28), then mutates each gene of that block with
  probability 1 / len(block) (pymoo's 1 / n_var rule, applied inside the
  block). With only one non-empty block (image-only or text-only runs) the
  coin is skipped.

See docs/SEARCH_REAL_CODED.md.
"""
from __future__ import annotations

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import mut_pm

from .genotype import GenotypeSpace

DEFAULT_CROSSOVER_PROB = 0.9
DEFAULT_CROSSOVER_ETA = 15.0
DEFAULT_MUTATION_ETA = 20.0
DEFAULT_MUTATION_PROB = 0.9  # per-offspring application probability (pymoo PM default)


def build_crossover(prob: float = DEFAULT_CROSSOVER_PROB, eta: float = DEFAULT_CROSSOVER_ETA) -> SBX:
    """SBX on float genes, no rounding (pymoo defaults: prob_var = 0.5, prob_bin = 0.5)."""
    return SBX(prob=prob, eta=eta)


class BlockPolynomialMutation(Mutation):
    """Polynomial mutation restricted to one coin-flipped modality block.

    :param space: genotype space (bounds and block slices).
    :param eta: PM distribution index (larger -> smaller steps).
    :param prob: per-offspring application probability.
    :param prob_var: per-gene probability inside the chosen block; ``None``
        = 1 / len(block).
    :param at_least_once: force at least one mutated gene in the chosen
        block, so that an applied mutation always changes the offspring.
    :param image_prob: probability of choosing the image block (0.5: coin).
    """

    def __init__(self, space: GenotypeSpace | None = None, eta: float = DEFAULT_MUTATION_ETA,
                 prob: float = DEFAULT_MUTATION_PROB, prob_var: float | None = None,
                 at_least_once: bool = True, image_prob: float = 0.5) -> None:
        super().__init__(prob=prob)
        self.space = space
        self.eta_value = float(eta)
        self.block_prob_var = prob_var
        self.at_least_once = at_least_once
        self.image_prob = image_prob

    def bind(self, space: GenotypeSpace) -> None:
        self.space = space

    def block_choice(self, n: int, random_state) -> np.ndarray:
        """Index into ``space.blocks`` per offspring (the 50/50 coin)."""
        if len(self.space.blocks) < 2:
            return np.zeros(n, dtype=int)
        return (random_state.random(n) >= self.image_prob).astype(int)  # 0 = image, 1 = text

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        sp = self.space
        if sp is None:
            raise RuntimeError("BlockPolynomialMutation needs a GenotypeSpace (call bind)")
        X = np.asarray(X, dtype=float)
        Xp = X.copy()
        choice = self.block_choice(len(X), random_state)
        for bi, (_, s) in enumerate(sp.blocks):
            rows = np.flatnonzero(choice == bi)
            if rows.size == 0:
                continue
            L = s.stop - s.start
            p = self.block_prob_var if self.block_prob_var is not None else 1.0 / L
            sub = mut_pm(X[rows, s], sp.xl[s], sp.xu[s], np.full(rows.size, self.eta_value),
                         np.full(rows.size, p), at_least_once=self.at_least_once,
                         random_state=random_state)
            Xp[rows, s] = sub
        return sp.sanitize(Xp)


def build_mutation(space: GenotypeSpace | None = None, prob: float | None = None,
                   eta: float = DEFAULT_MUTATION_ETA) -> BlockPolynomialMutation:
    """Shared block-wise PM. ``prob`` (config ``mutation.prob``) overrides
    the per-gene probability inside the chosen block (default 1/len(block))."""
    return BlockPolynomialMutation(space, eta=eta, prob_var=prob)
