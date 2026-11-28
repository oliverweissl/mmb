"""MMBA search on the real-coded genotype: cross first, then shrink the magnitude.

Phase 1 (cross): generation 0 comes from the shared magnitude-stratified sampler.
Each following generation keeps the individuals with the lowest signed margin
m = sign(g_anchor) * g (m < -tau: the predicted answer differs from the nominal
one), recombines and mutates them with the shared operators (SBX + block-wise
PM) and adds a few fresh samples for exploration. Guidance is the margin only;
magnitude is not optimised in this phase. Ablation ``guided=False``: fresh
samples only.

Phase 2 (minimise magnitude), from the first crossing C (the smallest-magnitude
crossing of that generation):

(a) global line: bracket the smallest crossing lam of scale(C, lam) on [0, 1]
    (lam = 0 is the unedited seed, non-crossing by the anchor definition);
(b) block lines: scale ONE block of the incumbent by mu in [0, 1] (the other
    block fixed), bracket the smallest crossing mu (mu = 0 is evaluated first:
    a block that is not needed is dropped), then alternate to the other block;
    stop when a full round shrinks no block by more than eps;
(c) each line stops when its ON/OFF pair differs by <= eps in its scale
    parameter; the last line's pair is the boundary pair.

A line is bracketed by batched bisection (k-section): the tester evaluates one
generation of pop_size individuals at a time, so a generation evaluates
pop_size equally spaced points inside the current bracket [lo, hi] and the
new bracket is (largest non-crossing point below the smallest crossing point,
smallest crossing point). The bracket shrinks by a factor pop_size + 1 per
generation (k = 1 is plain bisection). On a non-monotone line it keeps the
smallest crossing seen.

When the lines have converged and budget is left, Phase 1 restarts, guided by
the margin as before but restricted to weighted magnitude < |C| (C2 decision
2026-11-28): every offspring and fresh sample at or above |C| is rescaled
(scale(g, lam)) to a magnitude drawn in [0.5, 1) * |C|. A crossing below |C|
becomes the incumbent and Phase 2 runs again from (a).

Magnitudes are the weighted ones of :class:`~mmb.optimizer.genotype.GenotypeSpace`
(w_img * RMS(c), w_txt * t, L2 over blocks); they select the first crossing,
rank restarts and are the incumbent's fitness.

Interface: the tester calls observe_margin(m) for each generation before
assign_fitness(); best_candidates holds the incumbent (fitness = its
magnitude), boundary_pair the ON/OFF pair (the tester calls annotate_pair
with whether the rendered OFF/ON texts differ), trace one record per generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pymoo.core.population import Population

from mmb.core import OptimizerCandidate

from .genotype import IMAGE, TEXT
from .real_pymoo_optimizer import RealPymooOptimizer

GLOBAL = "global"


@dataclass
class _Line:
    kind: str                 # "global" | "image" | "text"
    base: np.ndarray          # genotype at parameter 1
    lo: float = 0.0
    hi: float = 1.0
    lo_known: bool = True     # lo has been evaluated (or is the seed) and does not cross
    lo_m: Optional[float] = None
    hi_m: Optional[float] = None
    removed: bool = False     # block line: mu = 0 still crosses (block dropped)
    mus: np.ndarray = field(default_factory=lambda: np.zeros(0))


class MMBAOptimizer(RealPymooOptimizer):
    def __init__(self, *args, tau: float = 0.0, elite_frac: float = 0.34,
                 explore_frac: float = 0.2, guided: bool = True, eps: float = 1e-3,
                 max_block_lines: int = 8, restart_scale: tuple[float, float] = (0.5, 1.0),
                 **kwargs) -> None:
        self._guided = guided  # False (ablation): Phase 1 draws fresh samples only
        kwargs["algorithm"] = "random"  # only used to draw generation 0 from the sampler
        self._tau = tau
        self._elite_frac = elite_frac
        self._explore_frac = explore_frac
        self._eps = eps
        self._max_block_lines = max_block_lines
        self._restart_scale = restart_scale
        self._margin = None
        super().__init__(*args, **kwargs)

    def _init_algorithm(self) -> None:
        super()._init_algorithm()
        self._rng = np.random.default_rng(self._seed)
        self._phase = 1
        self._gen = 0
        self._best = None            # (genotype, margin)
        self._line: _Line | None = None
        self._pair = None
        self._block_lines = 0
        self._no_gain = 0
        self._converged = False
        self._last_block = None
        self._trace: list[dict] = []

    # ---------------------------------------------------------------- hooks
    def observe_margin(self, m) -> None:
        self._margin = np.asarray(m, dtype=float)

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def trace(self) -> list[dict]:
        return self._trace

    @property
    def boundary_pair(self) -> dict | None:
        """ON/OFF pair of the last converged line: ``on`` crosses, ``off``
        does not, ``gap`` = their distance in the line's scale parameter,
        ``line`` = global | image | text, ``kind`` = text_step (rendered
        texts differ) | continuous (same text) | None (not annotated yet)."""
        return self._pair

    def annotate_pair(self, texts_differ: bool) -> None:
        """Record whether the rendered OFF/ON texts differ (set by the tester)."""
        if self._pair is not None:
            self._pair["text_differs"] = bool(texts_differ)
            self._pair["kind"] = "text_step" if texts_differ else "continuous"

    def assign_fitness(self, fitness, *data) -> None:
        """Store fitness; once a crossing exists, best_candidates stays the
        incumbent (the base class would replace it by the |m| Pareto set)."""
        super().assign_fitness(fitness, *data)
        if self._best is not None:
            self._set_best(*self._best)

    # ---------------------------------------------------------------- helpers
    def _crossing(self, m: np.ndarray) -> np.ndarray:
        return m < -self._tau

    def _fresh(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, self._n_var))
        s = self._algo_params["sampling"]
        x = s.do(self._problem, n, random_state=self._rng).get("X")
        return self._space.sanitize(np.asarray(x, dtype=float))

    def _vary(self, parents: np.ndarray, n: int) -> np.ndarray:
        """n offspring of the parent pool by the shared SBX + block-wise PM."""
        k = len(parents)
        n_mat = (n + 1) // 2
        mat = self._rng.integers(0, k, size=(n_mat, 2))
        pop = Population.new(X=np.asarray(parents, dtype=float))
        off = self._crossover_op.do(self._problem, pop, parents=mat, random_state=self._rng)
        off = self._mutation_op.do(self._problem, off, random_state=self._rng)
        return self._space.sanitize(off.get("X")[:n])

    def _point(self, line: _Line, mu: float) -> np.ndarray:
        if line.kind == GLOBAL:
            return self._space.scale(line.base, mu)
        return self._space.scale_block(line.base, mu, line.kind)

    def _line_points(self, n: int) -> np.ndarray:
        L = self._line
        if L.lo_known:
            mus = np.linspace(L.lo, L.hi, n + 2)[1:-1]
        else:
            mus = np.concatenate([[L.lo], np.linspace(L.lo, L.hi, n + 1)[1:-1]])
        L.mus = mus
        return np.stack([self._point(L, mu) for mu in mus])

    def _set_best(self, g: np.ndarray, m: float) -> None:
        self._best = (np.asarray(g, dtype=float).copy(), float(m))
        mag = float(self._space.magnitude(self._best[0]))
        self._best_candidates = [OptimizerCandidate(solution=self._best[0].copy(),
                                                    fitness=[mag] * self._num_objectives)]

    def _start_line(self, kind: str) -> None:
        g, m = self._best
        self._line = _Line(kind=kind, base=g.copy(), lo=0.0, hi=1.0,
                           lo_known=(kind == GLOBAL), hi_m=m)

    def _tell_line(self, m: np.ndarray) -> None:
        L = self._line
        mus = L.mus
        cr = self._crossing(m)
        if not L.lo_known:
            if cr[0]:                          # block not needed: drop it
                L.removed = True
                L.hi, L.hi_m = L.lo, float(m[0])
                return
            L.lo_known, L.lo_m = True, float(m[0])
        if cr.any():
            j = int(np.argmin(np.where(cr, mus, np.inf)))
            if mus[j] < L.hi:
                L.hi, L.hi_m = float(mus[j]), float(m[j])
        below = (~cr) & (mus < L.hi)
        if below.any():
            j = int(np.argmax(np.where(below, mus, -np.inf)))
            if mus[j] > L.lo:
                L.lo, L.lo_m = float(mus[j]), float(m[j])

    def _finish_line(self) -> None:
        L = self._line
        on = self._point(L, L.hi)
        self._set_best(on, L.hi_m if L.hi_m is not None else self._best[1])
        if not L.removed:
            off = self._point(L, L.lo)
            sp = self._space
            self._pair = {"on": on, "off": off, "on_margin": L.hi_m,
                          "off_margin": L.lo_m, "gap": L.hi - L.lo, "line": L.kind,
                          "lam_on": L.hi, "lam_off": L.lo,
                          "text_genes_equal": bool(np.array_equal(on[sp.text_slice], off[sp.text_slice])),
                          "text_differs": None, "kind": None,
                          "mag_on": float(sp.magnitude(on)), "mag_off": float(sp.magnitude(off))}
        if L.kind == GLOBAL:
            self._block_lines, self._no_gain = 0, 0
            nxt = self._next_block(None)
        else:
            self._block_lines += 1
            gain = L.removed or (1.0 - L.hi) > self._eps
            self._no_gain = 0 if gain else self._no_gain + 1
            n_active = len(self._active_blocks())
            if self._no_gain >= max(1, n_active) or self._block_lines >= self._max_block_lines:
                nxt = None
            else:
                nxt = self._next_block(L.kind)
        self._line = None
        if nxt is None:
            self._converged = True
        else:
            self._start_line(nxt)

    def _active_blocks(self) -> list[str]:
        g, sp = self._best[0], self._space
        return [b for b, s in sp.blocks if np.any(g[s][sp.magnitude_mask[s]] != 0)]

    def _next_block(self, last: str | None) -> str | None:
        act = self._active_blocks()
        if last is None:  # after the global line; with one block the global line already did it
            return act[0] if len(act) > 1 else None
        others = [b for b in act if b != last]
        return others[0] if others else None

    def _below(self, X: np.ndarray, cap: float) -> np.ndarray:
        """Rescale rows at or above magnitude cap into [lo, hi) * cap (restart region)."""
        lo, hi = self._restart_scale
        cur = np.asarray(self._space.magnitude(X), dtype=float)
        out = X.copy()
        for i in np.flatnonzero(cur >= cap * (1.0 - self._eps)):
            target = cap * self._rng.uniform(lo, hi) * (1.0 - 2 * self._eps)
            out[i] = self._space.scale(X[i], target / max(cur[i], 1e-12))
        return self._space.sanitize(out)

    def _restart(self, x: np.ndarray | None, m: np.ndarray | None, n: int) -> np.ndarray:
        """Guided Phase-1 generation restricted to magnitude < |C|."""
        cap = float(self._space.magnitude(self._best[0]))
        n_fresh = int(round(n * self._explore_frac))
        if x is None or not self._guided:
            X = self._fresh(n)
        else:
            order = np.argsort(m)
            mu = max(2, int(round(n * self._elite_frac)))
            X = np.vstack([self._vary(x[order[:mu]], n - n_fresh), self._fresh(n_fresh)])
        return self._below(X, cap)

    # --------------------------------------------------------- optimizer API
    def update(self) -> None:
        x = self._x_current
        m = self._margin if self._margin is not None else np.asarray(self._fitness[0], dtype=float)
        cr = self._crossing(m)
        rec = {"gen": self._gen, "phase": self._phase, "n_cross": int(cr.sum()),
               "min_margin": float(m.min()), "median_margin": float(np.median(m))}
        n = self._pop_size
        if self._phase == 1:
            if cr.any():
                mag = np.asarray(self._space.magnitude(x), dtype=float)
                i = min(np.flatnonzero(cr), key=lambda k: (mag[k], -m[k]))
                self._set_best(x[i], m[i])
                self._phase = 2
                self._start_line(GLOBAL)
                rec["first_cross"] = True
            elif not self._guided:
                nxt = self._fresh(n)
            else:
                order = np.argsort(m)
                mu = max(2, int(round(n * self._elite_frac)))
                n_fresh = int(round(n * self._explore_frac))
                nxt = np.vstack([self._vary(x[order[:mu]], n - n_fresh), self._fresh(n_fresh)])
        else:
            if self._line is not None:
                rec.update(stage=self._line.kind, lo_before=self._line.lo, hi_before=self._line.hi)
                self._tell_line(m)
                rec.update(lo=self._line.lo, hi=self._line.hi, removed=self._line.removed)
                if self._line.removed or self._line.hi - self._line.lo <= self._eps:
                    self._finish_line()
            else:                                           # guided restart generation
                rec["stage"] = "restart"
                mag = np.asarray(self._space.magnitude(x), dtype=float)
                best_mag = float(self._space.magnitude(self._best[0]))
                better = np.flatnonzero(cr & (mag < best_mag * (1.0 - self._eps)))
                if better.size:
                    i = min(better, key=lambda k: (mag[k], -m[k]))
                    self._set_best(x[i], m[i])
                    self._converged = False
                    self._start_line(GLOBAL)
                    rec["restart_cross"] = True
        if self._phase == 2:
            if self._line is not None:
                nxt = self._line_points(n)
            else:
                # first restart generation after convergence: fresh samples below |C|
                prev = x if rec.get("stage") == "restart" else None
                nxt = self._restart(prev, m if prev is not None else None, n)
            g = self._best[0]
            rec.update(mag=float(self._space.magnitude(g)),
                       mag_image=float(self._space.block_magnitude(g, IMAGE)),
                       mag_text=float(self._space.block_magnitude(g, TEXT)),
                       pair_gap=None if self._pair is None else float(self._pair["gap"]))
        self._trace.append(rec)
        self._gen += 1
        self._x_current = self._space.sanitize(nxt)
        self._margin = None
