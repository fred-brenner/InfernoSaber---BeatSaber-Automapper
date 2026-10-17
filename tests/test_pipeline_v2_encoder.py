"""Retained spectral packing, fixed normalization and cache identity checks."""

import numpy as np
import pytest

from training.pipeline_v2.encoder import FixedEncoder, SAMPLE_RATE, spectrogram_patches


def _audio(duration=2.3):
    time = np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE
    return (.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def _predict(patches):
    means = patches.mean(axis=(1, 2, 3))
    return np.repeat(means[:, None], 16, axis=1).astype(np.float32)


def test_center_patch_matches_existing_legacy_transform_without_import_side_effects():
    # Reproduce process_song's public mathematical operations for a full patch.
    from PIL import Image
    from scipy import signal

    audio = _audio(3.)
    actual = spectrogram_patches(audio, [1.5])[0, :, :, 0]
    center = round(1.5 * SAMPLE_RATE)
    patch = np.abs(audio[center - SAMPLE_RATE:center + SAMPLE_RATE]) * 100
    _, _, power = signal.spectrogram(patch, fs=SAMPLE_RATE, window="hann", nperseg=1480, noverlap=15, detrend=False)
    spectrum = np.log(power.T.astype(np.float32) + 1e-10)
    spectrum -= spectrum.min()
    expected = np.asarray(Image.fromarray(spectrum).resize((24, 20))).T
    np.testing.assert_array_equal(actual, expected)


def test_edges_keep_query_timestamps_and_silence_is_finite_zero():
    encoder = FixedEncoder(np.zeros(74, np.float32), _predict, "test-checkpoint")
    times = np.asarray([0., .001, .0049])
    assert encoder.patches(times).shape == (3, 24, 20, 1)
    assert not encoder.patches(times).any()
    assert not encoder(times).any()
    assert encoder.metadata["silence_scale_fallback"]
    assert encoder.metadata["minimum"] == encoder.metadata["maximum"] == 0


def test_normalization_is_fixed_per_song_independent_of_queries_order_and_batches():
    audio = _audio()
    first = FixedEncoder(audio, _predict, "weights", grid_seconds=.5, batch_size=2)
    second = FixedEncoder(audio, _predict, "weights", grid_seconds=.5, batch_size=7)
    assert first.fingerprint == second.fingerprint
    query = .37
    np.testing.assert_array_equal(first([query]), second([2., query, 0.])[1:2])
    np.testing.assert_array_equal(first.patches([query]), second.patches([0., query])[1:2])
    different = FixedEncoder(audio, _predict, "other-weights")
    assert first.fingerprint != different.fingerprint


def test_cache_uses_rounded_sample_identity_and_empty_query_is_supported():
    calls = []
    def predict(patches):
        calls.append(len(patches))
        return _predict(patches)
    encoder = FixedEncoder(_audio(), predict, "weights")
    encoder([.5, .5, .5 + .1 / SAMPLE_RATE])
    assert calls == [1]
    encoder([.5])
    assert calls == [1]
    assert encoder([]).shape == (0, 16)
    with pytest.raises(ValueError, match="inside"):
        encoder([3.])
    with pytest.raises(ValueError, match="fingerprint"):
        FixedEncoder(_audio(), predict, "")


def test_canonical_duration_keeps_valid_short_native_tail_queries_and_provenance():
    import hashlib

    audio = _audio(3.25)[:-16]
    original = audio.copy()
    native = FixedEncoder(audio, _predict, "weights")
    encoder = FixedEncoder(audio, _predict, "weights", duration=3.25)
    assert len(encoder.samples) == len(audio) == 48084
    np.testing.assert_array_equal(audio, original)
    np.testing.assert_array_equal(encoder.samples, original)
    assert encoder.metadata["decoded_sha256"] == hashlib.sha256(original.astype("<f4").tobytes()).hexdigest()
    assert encoder.metadata["decoded_sample_count"] == len(original)
    assert encoder.metadata["decoded_duration_seconds"] == len(original) / SAMPLE_RATE
    assert encoder.metadata["canonical_duration_seconds"] == 3.25
    assert encoder.metadata["tail_shortfall_samples"] == 16
    assert encoder.metadata["tail_shortfall_seconds"] == 3.25 - len(original) / SAMPLE_RATE
    assert encoder.metadata["query_duration_source"] == "canonical_audio"
    assert native.metadata["query_duration_source"] == "native_decode"
    assert encoder([3.249]).shape == (1, 16)
    with pytest.raises(ValueError, match="inside"):
        native([3.249])
    with pytest.raises(ValueError, match="inside"):
        encoder([3.25])
    with pytest.raises(ValueError, match="inside"):
        spectrogram_patches(audio, [3.25], duration=3.25)
    np.testing.assert_array_equal(spectrogram_patches(audio, [1.5]), spectrogram_patches(audio, [1.5], duration=3.25))
    # Both bounds have the same 0.5-second grid here, so interior normalized
    # patches and the once-per-song scale also remain exactly unchanged.
    np.testing.assert_array_equal(native.patches([1.5]), encoder.patches([1.5]))


def test_canonical_grid_includes_missing_native_tail_and_empty_overlap_is_zero():
    audio = _audio(.1)
    encoder = FixedEncoder(audio, _predict, "weights", duration=2.)
    np.testing.assert_array_equal(encoder.grid, [0., .5, 1., 1.5])
    # A centered patch entirely after the native waveform has no overlap.
    raw = spectrogram_patches(audio, [1.5, 1.9], duration=2.)
    assert raw.shape == (2, 24, 20, 1)
    assert not raw.any()
    assert np.isfinite(encoder([1.5, 1.9])).all()
    # New tail queries never change the normalization already fitted on grid.
    minimum, scale = encoder.minimum, encoder.scale
    encoder([.02, 1.9, .5])
    assert (encoder.minimum, encoder.scale) == (minimum, scale)


@pytest.mark.parametrize("duration", [0., -1., float("inf"), float("nan"), True, "3.25"])
def test_invalid_explicit_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        FixedEncoder(_audio(.1), _predict, "weights", duration=duration)
    with pytest.raises(ValueError, match="duration"):
        spectrogram_patches(_audio(.1), [0.], duration=duration)
