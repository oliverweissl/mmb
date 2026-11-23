"""MMBA search: cross first, then minimise (the directed arm, a0).

Phase 1 (cross): generation 0 comes from the shared sparse multi-tier sampler. Each
following generation keeps the individuals with the lowest signed margin
m = sign(g_anchor) * g (m < 0: the predicted answer differs from the nominal one), mutates them (add / remove /
change one edit) and adds a few fresh samples for exploration. Guidance is the margin
only; edit size is not optimised in this phase.

Phase 2 (minimise): as soon as a robust crossing exists, the rest of the budget
shrinks it. Candidates reset chunks of the current best crossing's edits to the
unedited value (ddmin-style: chunk size halves when no chunk can be reset) or move one
edit to another gene (escape local minima). A candidate replaces the incumbent if it
still changes the answer with fewer edits (ties: margin closer to the border). The
result is a 1-minimal crossing C; the non-crossing neighbours C minus one edit are
evaluated along the way, so every run ends with a boundary pair in its trace.

Interface: the tester calls observe_margin(m) for each generation before
assign_fitness(); best_candidates holds the incumbent minimal crossing.
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from mmb.core import OptimizerCandidate

from .discrete_pymoo_optimizer import DiscretePymooOptimizer


class MMBAOptimizer(DiscretePymooOptimizer):
    def __init__(self, *args, tau: float = 0.0, elite_frac: float = 0.34,
                 explore_frac: float = 0.2, move_frac: float = 0.25, guided: bool = True,
                 **kwargs) -> None:
        self._guided = guided  # False (ablation): Phase 1 draws fresh samples only, no margin guidance
        kwargs["algorithm"] = "random"  # only used to draw generation 0 from the sampler
        self._tau = tau
        self._elite_frac = elite_frac
        self._explore_frac = explore_frac
        self._move_frac = move_frac
        self._margin = None
        super().__init__(*args, **kwargs)

    def _init_algorithm(self) -> None:
        super()._init_algorithm()
        self._rng = np.random.default_rng(self._seed)
        self._archive_x: list[np.ndarray] = []
        self._archive_m: list[float] = []
        self._best = None          # (genotype, n_edits, margin)
        self._chunk = None
        self._phase = 1
        self._minimal = False

    # ---------------------------------------------------------------- hooks
    def observe_margin(self, m) -> None:
        self._margin = np.asarray(m, dtype=float)

    @property
    def phase(self) -> int:
        return self._phase

    # ---------------------------------------------------------------- helpers
    def _fresh(self, n: int) -> np.ndarray:
        s = self._algo_params.get("sampling")
        prob = Problem(n_var=self._n_var, n_obj=1, xl=np.zeros(self._n_var),
                       xu=self._gene_bounds - 1, vtype=int)
        if s is not None and hasattr(s, "do"):
            x = s.do(prob, n).get("X")
        else:
            x = self._rng.integers(0, self._gene_bounds, size=(n, self._n_var))
        return self._sanitize(np.asarray(x))

    def _mutate(self, g: np.ndarray, k: int) -> np.ndarray:
        g = g.copy()
        b = self._gene_bounds
        for _ in range(k):
            act = np.nonzero(g)[0]
            ina = np.nonzero((g == 0) & (b > 1))[0]
            r = self._rng.random()
            if (r < 0.5 or len(act) == 0) and len(ina):          # add an edit
                i = self._rng.choice(ina); g[i] = self._rng.integers(1, b[i])
            elif r < 0.7 and len(act):                            # remove an edit
                g[self._rng.choice(act)] = 0
            elif len(act):                                        # change an edit
                i = self._rng.choice(act); g[i] = self._rng.integers(1, max(2, b[i]))
        return g

    def _shrink_candidates(self, n: int) -> np.ndarray:
        C = self._best[0]
        act = list(np.nonzero(C)[0])
        out = []
        n_move = int(round(n * self._move_frac)) if self._minimal or len(act) <= 1 else int(round(n * 0.1))
        k = max(1, self._chunk)
        perm = list(self._rng.permutation(act))
        groups = [perm[i:i + k] for i in range(0, len(perm), k)]
        while len(out) < n - n_move and groups:
            grp = groups.pop(0)
            H = C.copy(); H[grp] = 0; out.append(H)
        while len(out) < n:                                        # move one edit
            H = C.copy()
            if act:
                i = self._rng.choice(act); H[i] = 0
            ina = np.nonzero((H == 0) & (self._gene_bounds > 1))[0]
            if len(ina):
                j = self._rng.choice(ina); H[j] = self._rng.integers(1, self._gene_bounds[j])
            out.append(H)
        return np.stack(out[:n])

    # --------------------------------------------------------- optimizer API
    def update(self) -> None:
        x = self._x_current
        m = self._margin if self._margin is not None else np.asarray(self._fitness[0], dtype=float)
        n_edit = (x != 0).sum(axis=1)
        crossed = m < -self._tau  # behaviour change: the answer differs from the nominal one
        improved = False
        for i in np.nonzero(crossed)[0]:
            cand = (x[i].copy(), int(n_edit[i]), float(m[i]))
            if self._best is None or (cand[1], -cand[2]) < (self._best[1], -self._best[2]):
                self._best = cand; improved = True
        if self._best is not None:
            if self._phase == 1:
                self._phase = 2
                self._chunk = max(1, self._best[1] // 2)
            elif improved:
                self._chunk = max(1, min(self._chunk, self._best[1] // 2))
            elif self._chunk > 1:
                self._chunk //= 2
            else:
                self._minimal = True
            self._best_candidates = [OptimizerCandidate(solution=self._best[0].copy(),
                                                        fitness=[float(self._best[1])] * self._num_objectives)]
            nxt = self._shrink_candidates(self._pop_size)
        elif not self._guided:
            nxt = self._fresh(self._pop_size)
        else:
            order = np.argsort(m)
            mu = max(1, int(round(self._pop_size * self._elite_frac)))
            elite = x[order[:mu]]
            n_fresh = int(round(self._pop_size * self._explore_frac))
            kids = [self._mutate(elite[i % mu], int(self._rng.integers(1, 4)))
                    for i in range(self._pop_size - n_fresh)]
            nxt = np.vstack([np.stack(kids), self._fresh(n_fresh)]) if n_fresh else np.stack(kids)
        self._x_current = self._sanitize(nxt)
        self._margin = None
