import numpy as np
import pytest

from mmb.optimizer import DiscretePymooOptimizer
from mmb.optimizer.algorithms import resolve_algorithm


@pytest.mark.parametrize("name,n_obj", [("agemoea2", 3), ("nsga2", 3), ("ga", 1), ("random", 3), ("hillclimb", 1)])
def test_algorithm_runs_generations(name, n_obj):
    opt = DiscretePymooOptimizer(np.full(8, 5), n_obj, pop_size=10, algorithm=name, seed=0)
    for _ in range(3):
        x = opt.get_x_current()
        assert x.shape == (10, 8) and x.min() >= 0 and x.max() < 5
        f = [x.sum(1).astype(float) + i for i in range(n_obj)]
        opt.assign_fitness(f)
        opt.update()


def test_random_has_no_selection_pressure():
    opt = DiscretePymooOptimizer(np.full(30, 50), 1, pop_size=40, algorithm="random", seed=1)
    means = []
    for _ in range(6):
        x = opt.get_x_current()
        means.append(x.mean())
        opt.assign_fitness([x.sum(1).astype(float)])
        opt.update()
    assert max(means) - min(means) < 5  # a GA would drift towards 0


def test_ga_rejects_multi_objective():
    with pytest.raises(ValueError):
        DiscretePymooOptimizer(np.full(4, 3), 3, algorithm="ga")


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        resolve_algorithm("nope")


def test_hillclimb_descends():
    opt = DiscretePymooOptimizer(np.full(30, 50), 1, pop_size=20, algorithm="hillclimb", seed=0)
    best = []
    for _ in range(15):
        x = opt.get_x_current()
        f = x.sum(1).astype(float)
        best.append(f.min())
        opt.assign_fitness([f])
        opt.update()
    assert min(best[-3:]) < best[0]  # incumbent improves on the minimisation objective


def test_deepjanus_runs_and_builds_archive():
    from mmb.optimizer.deepjanus import DeepJanusOptimizer
    opt = DeepJanusOptimizer(np.full(20, 6), 1, pop_size=12, seed=0, archive_tau=5.0)
    for _ in range(4):
        x = opt.get_x_current()
        assert x.shape == (12, 20)
        opt.assign_fitness([np.abs(x.sum(1) - 40).astype(float)])
        opt.update()
    assert opt.archive_size >= 1
    assert len(opt.best_candidates) >= 1


def test_mmba_crosses_then_minimises_to_key_edits():
    from mmb.optimizer.mmba import MMBAOptimizer
    key = [3, 7, 11]
    opt = MMBAOptimizer(np.full(40, 5), 1, pop_size=30, seed=0)

    def margin(x):
        # crosses (m <= -0.15) only when all key genes are edited; other edits add a little
        return 4.0 - 1.5 * (x[:, key] != 0).sum(1) + 0.01 * (x != 0).sum(1)

    for _ in range(50):
        x = opt.get_x_current()
        m = margin(x)
        opt.observe_margin(m)
        opt.assign_fitness([np.abs(m)])
        opt.update()
    best = opt.best_candidates[0].solution
    assert margin(best[None])[0] < 0
    assert set(np.nonzero(best)[0]) == set(key)      # 1-minimal: exactly the key edits
    assert opt.phase == 2
