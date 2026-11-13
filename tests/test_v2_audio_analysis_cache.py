"""Per-song audio-analysis cache: decode once, stat-keyed invalidation, parity."""

import os
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from map_creation import sanity_check
from map_creation import v2_postprocessing as post
from tools.config import config


def _tone_bursts(path, seconds=6., rate=44100):
    """Deterministic wav: short tone bursts every 0.5 s on silence."""
    rng = np.random.default_rng(7)
    samples = np.zeros(int(seconds * rate))
    for start in np.arange(.25, seconds - .5, .5):
        t = np.arange(int(.12 * rate)) / rate
        freq = 220 + rng.integers(0, 440)
        envelope = np.exp(-t * 18)
        i0 = int(start * rate)
        samples[i0:i0 + len(t)] += np.sin(2 * np.pi * freq * t) * envelope * .8
    pcm = np.asarray(np.clip(samples, -1, 1) * 32000, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


@pytest.fixture(autouse=True)
def analysis_reset(monkeypatch):
    monkeypatch.setattr(post, "_ANALYSIS", None)
    yield
    post._ANALYSIS = None


@pytest.fixture
def timing_defaults(monkeypatch):
    for key, value in {"min_beat_time": .0625, "min_time_diff": .02,
                       "map_filler_iters": 0, "add_start_end_beats": True,
                       "add_silence_flag": True, "max_speed": 20.,
                       "v2_fill_entry_min_notes": 0, "v2_suppress_start_fill": False,
                       "v2_fill_min_interval_seconds": 0., "v2_fill_respect_silence": True,
                       "v2_onset_fill_flag": True, "v2_onset_fill_strength_quantile": .25,
                       "v2_max_note_gap_seconds": 2., "v2_gap_fill_source": "onsets",
                       "v2_onset_snap_unmatched_policy": "non_silent"}.items():
        monkeypatch.setattr(config, key, value)


def _song(duration):
    return SimpleNamespace(metadata={"duration": duration, "bpm": 120.},
                           arrays={"frame_times": np.arange(int(duration * 100)) / 100})


def _beats(times):
    frames = np.floor(np.asarray(times) * 100 + .5).astype(int)
    counts = np.zeros(int(6 * 100), int)
    counts[frames] = 1
    return {"event_frames": frames, "event_times": np.asarray(times), "counts": counts,
            "frame_probabilities": np.tile([.8, .15, .05], (int(6 * 100), 1))}


def test_legacy_spectrum_decodes_once_per_file(tmp_path, monkeypatch):
    import aubio
    path = _tone_bursts(tmp_path / "song.wav")
    calls = []
    real = aubio.source
    monkeypatch.setattr(aubio, "source", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = post._legacy_timing_features(path)
    second = post._legacy_timing_features(path)
    assert len(calls) == 1
    for a, b in zip(first, second):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_onset_analysis_shared_and_stat_invalidated(tmp_path, monkeypatch):
    import librosa
    path = _tone_bursts(tmp_path / "song.wav")
    calls = []
    real = librosa.load
    monkeypatch.setattr(librosa, "load", lambda *a, **k: calls.append(1) or real(*a, **k))
    # all three librosa consumers share one decode per song
    peaks1 = post._detect_onset_peaks(path)
    peaks2 = post._detect_onsets(path)
    snapped = sanity_check.sanity_check_timing2(None, np.asarray([.3, .8]), audio_path=path)
    assert len(calls) == 1
    np.testing.assert_array_equal(peaks1[0], peaks2)
    assert len(snapped) == 2
    # a changed file invalidates: same path, different size/mtime
    _tone_bursts(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    post._detect_onsets(path)
    assert len(calls) == 2


def test_postprocess_timing_identical_on_cache_hit(tmp_path, timing_defaults):
    path = _tone_bursts(tmp_path / "song.wav", seconds=6.)
    beats = _beats([.75, 1.25, 2.25, 4.25, 4.75])
    cold, cold_report = post.postprocess_timing(_song(6.), beats, path)
    beats = _beats([.75, 1.25, 2.25, 4.25, 4.75])
    warm, warm_report = post.postprocess_timing(_song(6.), beats, path)
    assert cold == warm
    assert cold_report == warm_report


def test_analysis_skips_cache_for_unstattable_path(monkeypatch):
    import librosa
    calls = []
    monkeypatch.setattr(librosa, "load", lambda *a, **k: calls.append(1) or (np.zeros(500), 44100))
    monkeypatch.setattr(librosa.onset, "onset_detect", lambda **k: np.asarray([0], int))
    post._detect_onsets("no-such-file.egg")
    post._detect_onsets("no-such-file.egg")
    assert len(calls) == 2
