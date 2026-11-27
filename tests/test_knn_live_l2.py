import numpy as np

from mmb.manipulator.image.selection import build_codebook_knn_live_l2


def test_live_l2_knn_skips_dead_codes_and_orders_by_l2():
    cb = np.array([[0.0, 0.0], [1.0, 0.0], [1e-5, 0.0], [3.0, 0.0], [1.2, 0.0], [0.0, 2e-5]], dtype=np.float32)
    knn = build_codebook_knn_live_l2(cb, live_norm_min=1e-3)
    live = {1, 3, 4}
    assert knn.shape == (6, 2)
    assert set(np.unique(knn)) <= live                      # dead codes (0, 2, 5) never offered
    assert knn[1].tolist() == [4, 3]                        # nearest live neighbour first, self excluded
    assert knn[2].tolist() == [1, 4]                        # a dead code maps to its nearest live codes
