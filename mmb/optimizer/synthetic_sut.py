"""Synthetic SUTs with a known boundary in genotype space (tests, figures).

A synthetic SUT maps a population of genotypes to the signed margin
m = sign(g_anchor) * g used by MMBA (m > 0: nominal answer, m < 0: crossed;
m(0) = base > 0). It has an image part that is smooth (linear) in the image
block and a text part that is piecewise constant in t (bands of width
1/n_bands, as the manipulator's text output changes stepwise):

    m(g) = base - k_img * mean(c * d) - k_txt * floor(t * n_bands) / n_bands

with d a fixed +-1 pattern. Along the global scale line scale(g, lam) the
crossing lam is known in closed form for image-only and text-only genotypes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genotype import GenotypeSpace


@dataclass
class SyntheticSUT:
    space: GenotypeSpace
    base: float = 1.0
    k_img: float = 3.0
    k_txt: float = 2.0
    n_bands: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        d = self.space.image_slice.stop - self.space.image_slice.start
        self.direction = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=d)
        self.n_calls = 0

    def image_score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        c = X[:, self.space.image_slice]
        return c @ self.direction / max(len(self.direction), 1) if c.shape[1] else np.zeros(len(X))

    def text_score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        s = self.space.text_slice
        if s.stop <= s.start:
            return np.zeros(len(X))
        t = X[:, s.start]
        return np.floor(t * self.n_bands) / self.n_bands

    def margin(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        self.n_calls += len(X)
        return self.base - self.k_img * self.image_score(X) - self.k_txt * self.text_score(X)

    def image_crossing_lam(self, g: np.ndarray) -> float:
        """Smallest crossing lam of scale(g, lam) for a genotype with t = 0."""
        a = float(self.k_img * self.image_score(g)[0])
        return self.base / a if a > 0 else np.inf

    @property
    def text_threshold(self) -> float:
        """Smallest t that crosses on its own (image part zero)."""
        need = self.base / self.k_txt   # floor(t*K)/K > need
        return (np.floor(need * self.n_bands) + 1) / self.n_bands


def run_search(opt, sut: SyntheticSUT, n_gen: int) -> int:
    """Tester loop: evaluate each generation, feed margin and |m| fitness."""
    n_evals = 0
    for _ in range(n_gen):
        x = opt.get_x_current()
        m = sut.margin(x)
        n_evals += len(x)
        if hasattr(opt, "observe_margin"):
            opt.observe_margin(m)
        opt.assign_fitness([np.abs(m)] * opt._num_objectives)
        opt.update()
    return n_evals
