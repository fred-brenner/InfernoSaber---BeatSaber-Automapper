"""Generation intensity shaping and gain, without checkpoints or training."""

from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_pipeline import _postprocess_intensity, infer_beats
from tools.config import config
from training.beat_models.contracts import frame_conditions


@pytest.fixture(autouse=True)
def intensity_defaults(monkeypatch):
    for key, value in {"low_boost": .1, "high_reduction": .1, "scale": 1., "adjustment_strength": .8}.items():
        monkeypatch.setattr(config, f"v2_intensity_{key}", value)


def song(durations):
    durations = np.asarray(durations, np.float32)
    duration = float(durations.sum())
    return SimpleNamespace(metadata={"duration": duration, "bpm": 120.},
                           arrays={"bin_durations": durations, "frame_times": np.arange(0, duration, .01)})


def test_defaults_gently_flatten_and_preserve_average_density():
    raw = np.array([.25, .75], np.float32)
    adjusted, density, report = _postprocess_intensity(song([1., 1.]), raw, 4.)
    np.testing.assert_allclose(adjusted, [.27, .73])
    np.testing.assert_array_equal(raw, [.25, .75])
    assert density == pytest.approx(4.)
    assert report == {"low_boost": .1, "high_reduction": .1, "scale": 1., "adjustment_strength": .8,
                      "requested_nps": 4., "conditioning_nps": density}


def test_asymmetric_adjustments_normalize_a_partial_final_bin(monkeypatch):
    monkeypatch.setattr(config, "v2_intensity_low_boost", .2)
    monkeypatch.setattr(config, "v2_intensity_high_reduction", .1)
    monkeypatch.setattr(config, "v2_intensity_adjustment_strength", 1.)
    adjusted, density, _ = _postprocess_intensity(song([1., 1., .5]), np.array([.1, .3, .6], np.float32), 4.)
    np.testing.assert_allclose(adjusted, [2 / 13, 4 / 13, 7 / 13])
    assert density == pytest.approx(4.)


def test_uniform_local_density_preserves_partial_bins(monkeypatch):
    monkeypatch.setattr(config, "v2_intensity_low_boost", .7)
    monkeypatch.setattr(config, "v2_intensity_high_reduction", .2)
    base = song([1., 1., .5])
    raw = np.array([.4, .4, .2], np.float32)
    adjusted, density, _ = _postprocess_intensity(base, raw, 4.)
    np.testing.assert_allclose(adjusted, raw)
    np.testing.assert_allclose(frame_conditions(base, adjusted, density)[:, 2], .4)


@pytest.mark.parametrize("strength, expected", [(0., [.25, .75]), (.8, [.45, .55]), (1., [.5, .5])])
def test_strength_interpolates_toward_full_target(monkeypatch, strength, expected):
    monkeypatch.setattr(config, "v2_intensity_low_boost", 1.)
    monkeypatch.setattr(config, "v2_intensity_high_reduction", 1.)
    monkeypatch.setattr(config, "v2_intensity_adjustment_strength", strength)
    adjusted, density, _ = _postprocess_intensity(song([1., 1.]), np.array([.25, .75], np.float32), 4.)
    np.testing.assert_allclose(adjusted, expected)
    assert density == pytest.approx(4.)


def test_disabled_adjustment_returns_original_conditioning_exactly(monkeypatch):
    monkeypatch.setattr(config, "v2_intensity_scale", 2.)
    monkeypatch.setattr(config, "v2_intensity_adjustment_strength", 0.)
    base = song([1., 1., .5])
    raw = np.array([.13, .26, .61], np.float32)
    adjusted, density, _ = _postprocess_intensity(base, raw, 3.7)
    np.testing.assert_array_equal(adjusted, raw)
    np.testing.assert_array_equal(frame_conditions(base, adjusted, density), frame_conditions(base, raw, 3.7))
    assert density == 3.7


@pytest.mark.parametrize("scale, expected_nps", [(2., 7.2), (.5, 2.4), (0., .8)])
def test_gain_changes_conditioning_budget_without_cancelling_at_normalization(monkeypatch, scale, expected_nps):
    monkeypatch.setattr(config, "v2_intensity_scale", scale)
    base = song([1., 1.])
    adjusted, density, _ = _postprocess_intensity(base, np.array([.25, .75], np.float32), 4.)
    assert density == pytest.approx(expected_nps)
    assert adjusted.sum() == pytest.approx(1.)
    conditions = frame_conditions(base, adjusted, density)
    assert np.mean(conditions[:, 2]) * 10 == pytest.approx(expected_nps)


def test_shaping_and_gain_share_one_interpolation(monkeypatch):
    monkeypatch.setattr(config, "v2_intensity_low_boost", 1.)
    monkeypatch.setattr(config, "v2_intensity_high_reduction", 1.)
    monkeypatch.setattr(config, "v2_intensity_scale", 2.)
    base = song([1., 1.])
    adjusted, density, _ = _postprocess_intensity(base, np.array([.25, .75], np.float32), 4.)
    # Original local NPS 2 / 6 moves 80% toward the scaled uniform target 8 / 8.
    np.testing.assert_allclose(frame_conditions(base, adjusted, density)[[0, 100], 2] * 10, [6.8, 7.6], rtol=1e-6)


@pytest.mark.parametrize("requested_nps, scale, strength", [(4., 0., 1.), (0., 2., .8)])
def test_zero_effective_density_skips_timing(monkeypatch, requested_nps, scale, strength):
    monkeypatch.setattr(config, "v2_intensity_scale", scale)
    monkeypatch.setattr(config, "v2_intensity_adjustment_strength", strength)
    raw = np.array([.25, .75], np.float32)
    result = infer_beats(song([1., 1.]), requested_nps,
                         {"timing": lambda inputs: pytest.fail("zero density must skip timing")}, intensity=raw)
    assert result["conditioning_nps"] == 0
    assert not result["counts"].any()
    assert not len(result["event_times"])
    assert result["intensity"].sum() == pytest.approx(1.)
    assert np.isfinite(result["conditions"]).all()
    np.testing.assert_array_equal(result["raw_intensity"], raw)
    if scale == 0:
        np.testing.assert_array_equal(result["intensity"], raw)


@pytest.mark.parametrize("key", ["low_boost", "high_reduction", "scale", "adjustment_strength"])
@pytest.mark.parametrize("value", [-.01, float("nan"), float("inf"), True, np.bool_(False), "0.1", None])
def test_invalid_settings_fail_explicitly(monkeypatch, key, value):
    name = f"v2_intensity_{key}"
    monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        _postprocess_intensity(song([1.]), np.array([1.], np.float32), 4.)


@pytest.mark.parametrize("key", ["low_boost", "high_reduction", "adjustment_strength"])
def test_fraction_settings_reject_values_above_one(monkeypatch, key):
    name = f"v2_intensity_{key}"
    monkeypatch.setattr(config, name, 1.01)
    with pytest.raises(ValueError, match=name):
        _postprocess_intensity(song([1.]), np.array([1.], np.float32), 4.)
