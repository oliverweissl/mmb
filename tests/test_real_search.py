"""Real-coded search on synthetic SUTs with a known boundary (CPU only)."""
import numpy as np
import pytest
from pymoo.core.problem import Problem

from mmb.optimizer.deepjanus import DeepJanusOptimizer
from mmb.optimizer.genotype import GenotypeSpace
from mmb.optimizer.mmba import MMBAOptimizer
from mmb.optimizer.real_pymoo_optimizer import RealPymooOptimizer
from mmb.optimizer.sparse_sampling import MagnitudeStratifiedSampling
from mmb.optimizer.synthetic_sut import SyntheticSUT, run_search
from mmb.optimizer.variation import BlockPolynomialMutation, build_crossover

D = 16
SPACE = GenotypeSpace.contract(D)
EPS = 1e-3
POP, GENS = 50, 30


def _problem(sp):
    return Problem(n_var=sp.n_var, n_obj=1, xl=sp.xl, xu=sp.xu, vtype=float)


# ------------------------------------------------------------------ genotype
def test_contract_layout_and_scale():
    sp = SPACE
    assert sp.n_var == D + 2 and sp.image_slice == slice(0, D) and sp.text_slice == slice(D, D + 2)
    g = np.concatenate([np.full(D, 0.5), [0.8, 0.3]])
    h = sp.scale(g, 0.5)
    assert np.allclose(h[:D], 0.25) and h[D] == pytest.approx(0.4) and h[D + 1] == pytest.approx(0.3)
    hb = sp.scale_block(g, 0.0, "text")
    assert np.allclose(hb[:D], 0.5) and hb[D] == 0 and hb[D + 1] == pytest.approx(0.3)
    assert sp.block_magnitude(g, "image") == pytest.approx(0.5)
    assert sp.block_magnitude(g, "text") == pytest.approx(0.8)
    assert sp.magnitude(sp.zero()) == 0
    assert sp.sanitize(np.ones(sp.n_var))[D + 1] < 1.0  # u in [0, 1)


def test_from_manipulator_stub_uses_its_scale():
    class Stub:
        xl = SPACE.xl; xu = SPACE.xu
        image_slice = slice(0, D); text_slice = (D, D + 2)
        calls = 0

        def scale(self, g, lam):
            Stub.calls += 1
            out = g.copy(); out[..., :D + 1] *= lam
            return out

    sp = GenotypeSpace.from_manipulator(Stub())
    assert sp.magnitude_mask[:D + 1].all() and not sp.magnitude_mask[D + 1] and sp.upper_open[D + 1]
    sp.scale(np.ones(sp.n_var), 0.5)
    assert Stub.calls == 1


# ------------------------------------------------------------------ sampler
def test_sampler_stratified_and_shared():
    s = MagnitudeStratifiedSampling(SPACE, seed=3)
    X = s.do(_problem(SPACE), 60).get("X")
    assert X.shape == (60, D + 2)
    assert (X >= SPACE.xl).all() and (X <= SPACE.xu).all() and (X[:, D + 1] < 1).all()
    mi = SPACE.block_magnitude(X, "image"); mt = SPACE.block_magnitude(X, "text")
    modes = (mi > 0).astype(int) + 2 * (mt > 0)
    assert set(np.bincount(modes, minlength=4)[1:]) == {20}     # image / text / both: a third each
    mags = np.sort(np.maximum(mi, mt))
    assert mags[0] < 0.05 and mags[-1] > 0.8 and np.median(mags) < 0.4   # dense at small magnitudes
    X2 = MagnitudeStratifiedSampling(SPACE, seed=3).do(_problem(SPACE), 60).get("X")
    assert np.array_equal(X, X2)


@pytest.mark.parametrize("text", [True, False])
def test_sampler_single_block(text):
    sp = GenotypeSpace.contract(0 if text else D, text=text)
    X = MagnitudeStratifiedSampling(sp, seed=0).do(_problem(sp), 20).get("X")
    assert (sp.magnitude(X) > 0).all()


# ------------------------------------------------------------------ operators
def test_block_mutation_touches_one_block_fair_coin():
    mut = BlockPolynomialMutation(SPACE, prob=1.0)
    X = np.zeros((4000, SPACE.n_var)); X[:, D:] = 0.5   # interior (PM at a bound may return the bound)
    Y = mut._do(_problem(SPACE), X, random_state=np.random.default_rng(0))
    ch_img = (Y[:, :D] != X[:, :D]).any(1); ch_txt = (Y[:, D:] != X[:, D:]).any(1)
    assert not (ch_img & ch_txt).any()                     # one block per offspring
    assert (ch_img | ch_txt).all()                          # at least one gene changed
    assert abs(ch_img.mean() - 0.5) < 0.03                  # 50/50 coin
    n_img = (Y[ch_img, :D] != 0).sum(1)
    assert 1.0 <= n_img.mean() < 2.0                        # ~1/len(block) per gene (+ at least once)
    assert (Y >= SPACE.xl).all() and (Y <= SPACE.xu).all() and (Y[:, D + 1] < 1).all()


def test_shared_operator_parameters():
    opt = RealPymooOptimizer(SPACE, 1, pop_size=10, algorithm="ga", seed=0)
    assert opt.crossover.eta.value == 15 and opt.crossover.prob.value == 0.9
    assert opt.mutation.eta_value == 20
    assert build_crossover().eta.value == 15


# ------------------------------------------------------------------ MMBA
def _final_pair_ok(opt, sut):
    pair = opt.boundary_pair
    assert pair is not None
    assert pair["gap"] <= EPS
    assert sut.margin(pair["on"])[0] < 0 <= sut.margin(pair["off"])[0]
    return pair


def test_mmba_smooth_image_boundary_within_eps():
    sp = GenotypeSpace.contract(D, text=False)
    sut = SyntheticSUT(sp, k_img=3.0)
    opt = MMBAOptimizer(sp, 1, pop_size=POP, seed=0, eps=EPS)
    n = run_search(opt, sut, GENS)
    assert n == POP * GENS == sut.n_calls
    assert opt.phase == 2
    pair = _final_pair_ok(opt, sut)
    assert pair["line"] == "global" and pair["text_genes_equal"]
    opt.annotate_pair(False)
    assert opt.boundary_pair["kind"] == "continuous"
    # the ON point is on the smooth boundary: |m| <= slope * eps
    on_m = sut.margin(pair["on"])[0]
    assert abs(on_m) <= sut.k_img * EPS * np.abs(pair["on"]).max() + 1e-12
    # the first global line found the closed-form crossing lam of its base
    first = [r for r in opt.trace if r.get("stage") == "global"]
    assert first


def test_mmba_global_line_matches_closed_form():
    sp = GenotypeSpace.contract(D, text=False)
    sut = SyntheticSUT(sp, k_img=3.0)
    opt = MMBAOptimizer(sp, 1, pop_size=POP, seed=1, eps=EPS)
    # run until the first global line converges
    base = None
    for _ in range(GENS):
        x = opt.get_x_current(); m = sut.margin(x)
        opt.observe_margin(m); opt.assign_fitness([np.abs(m)])
        line = opt._line
        if line is not None and line.kind == "global" and base is None:
            base = line.base.copy()
        opt.update()
        if base is not None and opt.boundary_pair is not None:
            break
    lam_true = sut.image_crossing_lam(base)
    p = opt.boundary_pair
    assert p["lam_off"] < lam_true <= p["lam_on"] and p["lam_on"] - p["lam_off"] <= EPS


def test_mmba_text_step_boundary_drops_unneeded_image():
    sut = SyntheticSUT(SPACE, k_img=0.0, k_txt=2.0, n_bands=4)   # only text matters, stepwise
    opt = MMBAOptimizer(SPACE, 1, pop_size=POP, seed=2, eps=EPS)
    run_search(opt, sut, GENS)
    pair = _final_pair_ok(opt, sut)
    best = opt.best_candidates[0].solution
    thr = sut.text_threshold                                     # 0.75
    assert SPACE.block_magnitude(best, "image") == 0             # image block dropped
    assert thr <= best[D] <= thr + EPS                           # t just above the step
    assert pair["off"][D] < thr
    # text output steps between OFF and ON (band 2 -> band 3)
    differ = sut.text_score(pair["off"])[0] != sut.text_score(pair["on"])[0]
    opt.annotate_pair(differ)
    assert differ and opt.boundary_pair["kind"] == "text_step"


def test_mmba_mixed_boundary_minimises_magnitude():
    sut = SyntheticSUT(SPACE, k_img=3.0, k_txt=2.0)
    opt = MMBAOptimizer(SPACE, 1, pop_size=POP, seed=0, eps=EPS)
    run_search(opt, sut, GENS)
    _final_pair_ok(opt, sut)
    tr = [r for r in opt.trace if r["phase"] == 2]
    mags = [r["mag"] for r in tr]
    assert mags[-1] <= mags[0] + 1e-12 and all(b <= a + 1e-12 for a, b in zip(mags, mags[1:]))


def test_mmba_deterministic_per_seed():
    def go(seed):
        sut = SyntheticSUT(SPACE)
        opt = MMBAOptimizer(SPACE, 1, pop_size=POP, seed=seed, eps=EPS)
        run_search(opt, sut, 12)
        return opt.best_candidates[0].solution, opt.trace
    a, ta = go(5); b, tb = go(5); c, _ = go(6)
    assert np.array_equal(a, b) and ta == tb
    assert not np.array_equal(a, c)


def test_mmba_unguided_crosses_and_minimises():
    sut = SyntheticSUT(SPACE)
    opt = MMBAOptimizer(SPACE, 1, pop_size=POP, seed=1, guided=False, eps=EPS)
    run_search(opt, sut, GENS)
    assert opt.phase == 2
    _final_pair_ok(opt, sut)


def test_mmba_phase1_guidance_lowers_margin():
    sut = SyntheticSUT(SPACE, base=2.5, k_img=3.0, k_txt=0.5)   # hard to cross at random
    opt = MMBAOptimizer(SPACE, 1, pop_size=POP, seed=0, eps=EPS)
    run_search(opt, sut, GENS)
    p1 = [r["min_margin"] for r in opt.trace if r["phase"] == 1]
    assert len(p1) >= 2 and p1[-1] < p1[0]


# ------------------------------------------------------------------ budget / determinism, all methods
@pytest.mark.parametrize("make", [
    lambda s: RealPymooOptimizer(SPACE, 1, pop_size=POP, algorithm="ga", seed=s),           # MIMICRY-style
    lambda s: RealPymooOptimizer(SPACE, 3, pop_size=POP, algorithm="agemoea2", seed=s),
    lambda s: RealPymooOptimizer(SPACE, 3, pop_size=POP, algorithm="random", seed=s),
    lambda s: DeepJanusOptimizer(SPACE, 1, pop_size=POP, seed=s),
    lambda s: MMBAOptimizer(SPACE, 1, pop_size=POP, seed=s),
])
def test_budget_and_determinism_all_methods(make):
    def go():
        opt = make(7)
        sut = SyntheticSUT(SPACE)
        xs = []
        for _ in range(GENS):
            x = opt.get_x_current(); xs.append(x.copy())
            assert x.shape == (POP, SPACE.n_var)
            m = sut.margin(x)
            if hasattr(opt, "observe_margin"):
                opt.observe_margin(m)
            f = [np.abs(m), SPACE.block_magnitude(x, "image"), SPACE.block_magnitude(x, "text")]
            opt.assign_fitness(f[:opt._num_objectives])
            opt.update()
        return np.stack(xs), sut.n_calls
    a, na = go(); b, nb = go()
    assert na == nb == POP * GENS
    assert np.array_equal(a, b)


def test_generation0_shared_across_methods():
    g0 = [RealPymooOptimizer(SPACE, 1, pop_size=POP, algorithm="ga", seed=4).get_x_current(),
          RealPymooOptimizer(SPACE, 3, pop_size=POP, algorithm="agemoea2", seed=4).get_x_current(),
          DeepJanusOptimizer(SPACE, 1, pop_size=POP, seed=4).get_x_current(),
          MMBAOptimizer(SPACE, 1, pop_size=POP, seed=4).get_x_current()]
    for x in g0[1:]:
        assert np.allclose(np.sort(x, axis=0), np.sort(g0[0], axis=0))


# ------------------------------------------------------------------ C2 decisions (round 2)
def test_weighted_magnitudes_from_manipulator():
    class Stub:
        xl = SPACE.xl; xu = SPACE.xu
        image_slice = slice(0, D); text_slice = slice(D, D + 2)
        w_img, w_txt = 0.5, 2.0

    sp = GenotypeSpace.from_manipulator(Stub())
    g = np.concatenate([np.full(D, 0.4), [0.3, 0.9]])
    assert sp.block_magnitude(g, "image") == pytest.approx(0.2)
    assert sp.block_magnitude(g, "text") == pytest.approx(0.6)
    assert sp.magnitude(g) == pytest.approx(np.hypot(0.2, 0.6))

    class NoWeights:
        xl = SPACE.xl; xu = SPACE.xu
        image_slice = slice(0, D); text_slice = slice(D, D + 2)

    nw = GenotypeSpace.from_manipulator(NoWeights())
    assert nw.w_img == 1.0 and nw.w_txt == 1.0


def test_first_crossing_uses_weighted_magnitude():
    # text is cheap under w_txt = 0.1: the selected first crossing must minimise the weighted norm
    sp = GenotypeSpace.contract(D, w_img=1.0, w_txt=0.1)
    sut = SyntheticSUT(sp, base=1.0, k_img=3.0, k_txt=2.0)
    opt = MMBAOptimizer(sp, 1, pop_size=POP, seed=0, eps=EPS)
    x = opt.get_x_current(); m = sut.margin(x)
    opt.observe_margin(m); opt.assign_fitness([np.abs(m)]); opt.update()
    cr = m < 0
    assert cr.any()
    mags = sp.magnitude(x[cr])
    assert opt.best_candidates[0].fitness[0] == pytest.approx(mags.min())


def test_guided_restart_finds_smaller_crossing_and_respects_cap():
    # text crossing is reachable (t >= 0.75) but a pure-image crossing is cheaper (RMS ~ 1/3)
    sp = GenotypeSpace.contract(D)
    sut = SyntheticSUT(sp, base=1.0, k_img=3.0, k_txt=2.0)
    opt = MMBAOptimizer(sp, 1, pop_size=POP, seed=3, eps=EPS)
    caps = []
    for _ in range(GENS):
        x = opt.get_x_current()
        if opt.phase == 2 and opt._line is None and opt._best is not None:
            caps.append((sp.magnitude(x), sp.magnitude(opt._best[0])))
        m = sut.margin(x)
        opt.observe_margin(m); opt.assign_fitness([np.abs(m)]); opt.update()
    assert sut.n_calls == POP * GENS
    for mags, cap in caps:                       # every restart individual is below |C|
        assert (mags < cap).all()
    restarts = [r for r in opt.trace if r.get("stage") == "restart"]
    assert restarts                               # budget after convergence went to restarts
    assert any(r.get("restart_cross") for r in opt.trace)   # and found a smaller crossing
    mags = [r["mag"] for r in opt.trace if r["phase"] == 2]
    assert all(b <= a + 1e-12 for a, b in zip(mags, mags[1:]))
