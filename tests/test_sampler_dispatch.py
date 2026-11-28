"""build_sampler_from_config: the shared stratified-magnitude sampler is the only mode (2026-11-28)."""

import pytest

from mmb.config import SamplingConfig
from mmb.optimizer.sparse_sampling import MagnitudeStratifiedSampling, build_sampler_from_config


def test_default_mode_is_stratified_magnitude():
    cfg = SamplingConfig()
    assert cfg.mode == "stratified_magnitude"
    s = build_sampler_from_config(cfg, seed=3)
    assert isinstance(s, MagnitudeStratifiedSampling)
    assert (s.r_min, s.r_max, s.power) == (cfg.r_min, cfg.r_max, cfg.power)


def test_custom_strata():
    s = build_sampler_from_config(SamplingConfig(r_min=0.1, r_max=0.5, power=1.0), seed=0)
    assert (s.r_min, s.r_max, s.power) == (0.1, 0.5, 1.0)


@pytest.mark.parametrize("mode", ["uniform", "sparse", "sparse_multitier", "sparse_multitier_fps", "bogus"])
def test_removed_modes_rejected(mode):
    with pytest.raises(ValueError, match="stratified_magnitude"):
        SamplingConfig(mode=mode)
