"""Shared initial sampler of every method (real-coded genotype, 2026-11-28).

:class:`MagnitudeStratifiedSampling` draws generation 0 sparse in MAGNITUDE: target magnitudes stratified from small
to large, modes image-only / text-only / both cycled over the strata (docs/SEARCH_REAL_CODED.md). The integer
samplers of the former rank-ladder genotype (sparse, sparse_multitier, score-guided, FPS) were removed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling


class MagnitudeStratifiedSampling(Sampling):
    """Shared initial sampler for the real-coded genotype, sparse in magnitude.

    Individual k of n gets a target magnitude from a stratified grid that is
    dense at small magnitudes, ``r_k = r_min + (r_max - r_min) * ((k + U_k) / n) ** power``
    (one draw per stratum, so small and large edits are both present from
    generation 0). Its active blocks cycle over image-only, text-only and
    both (one or two modes when a block is empty), so every mode spans the
    whole magnitude range. Within an active block:

    * image: c = r * sqrt(d) * v / ||v||, v ~ N(0, I), clipped to the box
      (RMS(c) = r before clipping; clipping only lowers it);
    * text: the magnitude gene t = r, the choice genes u ~ U[xl, xu).

    Inactive blocks are zero in their magnitude genes (u is still drawn; it
    has no effect while t = 0). Rows are shuffled. The sampler has its own
    seeded RNG, so generation 0 is identical for every method at one seed.

    :param space: genotype space; the optimizer binds it per seed if None.
    :param r_min, r_max: magnitude range of the strata.
    :param power: stratum spacing exponent (>1: denser at small magnitudes).
    :param seed: RNG seed.
    """

    def __init__(self, space=None, r_min: float = 0.02, r_max: float = 1.0,
                 power: float = 2.0, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= r_min < r_max <= 1.0:
            raise ValueError(f"need 0 <= r_min < r_max <= 1, got {r_min}, {r_max}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        self.space = space
        self.r_min, self.r_max, self.power = r_min, r_max, power
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def bind(self, space) -> None:
        """Set the genotype space (per seed) and restart the RNG stream."""
        self.space = space
        self._rng = np.random.default_rng(self._seed)

    def magnitudes(self, n: int) -> NDArray[np.float64]:
        k = np.arange(n)
        u = self._rng.random(n)
        return self.r_min + (self.r_max - self.r_min) * ((k + u) / n) ** self.power

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray[np.float64]:
        sp = self.space
        if sp is None:
            raise ValueError("MagnitudeStratifiedSampling needs a GenotypeSpace (bind it)")
        if int(problem.n_var) != sp.n_var:
            raise ValueError(f"problem.n_var {problem.n_var} != genotype n_var {sp.n_var}")
        rng = self._rng
        blocks = [name for name, _ in sp.blocks]
        modes = [(b,) for b in blocks] + ([tuple(blocks)] if len(blocks) == 2 else [])
        r = self.magnitudes(n_samples)
        off = int(rng.integers(len(modes))) if modes else 0
        X = np.zeros((n_samples, sp.n_var))
        for i in range(n_samples):
            for b in (modes[(i + off) % len(modes)] if modes else ()):
                s = sp.block(b)
                mag = sp.magnitude_mask[s]
                idx = np.arange(s.start, s.stop)
                if b == "image":
                    v = rng.standard_normal(mag.sum())
                    v = r[i] * np.sqrt(v.size) * v / max(np.linalg.norm(v), 1e-12)
                    X[i, idx[mag]] = v
                else:
                    X[i, idx[mag]] = r[i]
        # choice genes (u): uniform everywhere
        choice = np.zeros(sp.n_var, bool)
        for _, s in sp.blocks:
            choice[s] = True
        choice &= ~sp.magnitude_mask
        if choice.any():
            X[:, choice] = sp.xl[choice] + rng.random((n_samples, choice.sum())) * (sp.xu - sp.xl)[choice]
        X = sp.sanitize(X)
        return X[rng.permutation(n_samples)]


def build_real_sampler(space=None, seed: int | None = None, **kwargs) -> MagnitudeStratifiedSampling:
    """The shared initial sampler for every method (config: sampling.mode
    'stratified_magnitude'; kwargs r_min, r_max, power)."""
    return MagnitudeStratifiedSampling(space, seed=seed, **kwargs)


def build_sampler_from_config(sampling_cfg, seed: int | None = None, **_: object) -> MagnitudeStratifiedSampling:
    """Sampler for ``optimizer.sampling`` (mode ``stratified_magnitude``); the genotype space is bound per seed.
    Extra keyword arguments (text_dim, codebook, ...) of the former multi-mode signature are ignored."""
    if sampling_cfg.mode != "stratified_magnitude":
        raise ValueError(f"Unknown sampling mode {sampling_cfg.mode!r}; expected 'stratified_magnitude'")
    return MagnitudeStratifiedSampling(r_min=sampling_cfg.r_min, r_max=sampling_cfg.r_max,
                                       power=sampling_cfg.power, seed=seed)
