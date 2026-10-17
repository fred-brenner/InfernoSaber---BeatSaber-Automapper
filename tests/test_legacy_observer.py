"""Observe the legacy threshold boundary without loading model dependencies."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.legacy_observer import LegacyBeatObserver
from tools.config import config


@pytest.fixture
def timing_config(monkeypatch):
    for name, value in {"max_speed_orig": 20, "thresh_beat": .5,
                        "threshold_start": 1.1, "threshold_end": .7,
                        "beat_spacing": 2, "window": 1}.items():
        monkeypatch.setattr(config, name, value)


def test_observation_forwards_original_arrays_and_copies_raw_boundary(timing_config):
    probabilities = np.array([[.9], [.1], [.8], [.7], [.6], [.1]])
    original_values = probabilities.copy()
    processed = np.array([3.5, 4.0])
    calls = []

    def threshold(values):
        assert values is probabilities
        return (values > config.thresh_beat).astype(int)

    def generate(names, debug_beats):
        calls.append((names, debug_beats))
        binary = generator.apply_first_beat_thresholding(probabilities)
        np.testing.assert_array_equal(binary, (original_values > .5).astype(int))
        binary[:] = 0  # Mirror the later in-place spacing filter.
        return processed

    generator = SimpleNamespace(main=generate, apply_first_beat_thresholding=threshold)
    observer = LegacyBeatObserver(generator)
    names = ["song_0000"]
    assert observer(names) is processed
    assert calls == [(names, True)]
    assert generator.apply_first_beat_thresholding is threshold
    np.testing.assert_array_equal(probabilities, original_values)
    record, = observer.records
    assert record["times_seconds"] == [1.5, 2.0]  # Strictly greater than window.
    assert record["probabilities"] == original_values.reshape(-1).tolist()
    assert record["thresholded_binary"] == [1, 0, 1, 1, 1, 0]
    assert record["name"] == "song_0000"
    assert record["bps_setting"] == 5
    assert record["effective_threshold"] == .5
    assert record["threshold_start"] == 1.1
    assert record["threshold_end"] == .7
    assert record["beat_spacing"] == 2
    assert record["window"] == 1
    assert record["captured"] is True
    assert record["status"] == "ok"


def test_probabilities_are_saved_before_mutating_threshold(timing_config):
    probabilities = np.array([.9, .2, .8, .7])
    before = probabilities.copy()

    def threshold(values):
        binary = (values > .5).astype(int)
        values[:] = -1
        return binary

    def generate(names, debug_beats):
        generator.apply_first_beat_thresholding(probabilities)
        return np.array([1.5])

    generator = SimpleNamespace(main=generate, apply_first_beat_thresholding=threshold)
    observer = LegacyBeatObserver(generator)
    observer(["song_0000"])
    assert observer.records[0]["probabilities"] == before.tolist()
    np.testing.assert_array_equal(probabilities, [-1, -1, -1, -1])


@pytest.mark.parametrize("after_capture", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError("broken"), SystemExit(2)])
def test_exception_restores_wrapper_and_preserves_available_capture(
        timing_config, after_capture, failure):
    def threshold(values):
        return np.ones_like(values)

    def generate(names, debug_beats):
        if after_capture:
            generator.apply_first_beat_thresholding(np.full(5, .9))
        raise failure

    generator = SimpleNamespace(main=generate, apply_first_beat_thresholding=threshold)
    observer = LegacyBeatObserver(generator)
    with pytest.raises(type(failure)) as raised:
        observer(["song_0000"])
    assert raised.value is failure
    assert generator.apply_first_beat_thresholding is threshold
    record, = observer.records
    assert record["status"] == "failed"
    assert record["captured"] is after_capture
    assert record["times_seconds"] == ([1.5, 2.0] if after_capture else [])
    assert record["error"].startswith(type(failure).__name__ + ":")


def test_calls_and_instances_keep_separate_captures(timing_config, monkeypatch):
    probabilities = np.array([.9, .1, .8, .7, .6])

    def threshold(values):
        return (values > config.thresh_beat).astype(int)

    def generate(names, debug_beats):
        generator.apply_first_beat_thresholding(probabilities)
        return np.array([3.0])

    generator = SimpleNamespace(main=generate, apply_first_beat_thresholding=threshold)
    observer = LegacyBeatObserver(generator)
    observer(["song_0000"])
    first = observer.records[0]
    probabilities[:] = .1
    monkeypatch.setattr(config, "max_speed_orig", 28)
    observer(["song_0001"])
    assert first["times_seconds"] == [1.5, 2.0]
    assert observer.records[1]["times_seconds"] == []
    assert observer.records[1]["name"] == "song_0001"
    assert observer.records[1]["bps_setting"] == 7
    assert LegacyBeatObserver(generator).records == []
    assert generator.apply_first_beat_thresholding is threshold


@pytest.mark.parametrize("failure_result", [None, 1])
def test_failure_status_is_forwarded_and_recorded(timing_config, failure_result):
    threshold = lambda values: values
    generator = SimpleNamespace(main=lambda names, debug_beats: failure_result,
                                apply_first_beat_thresholding=threshold)
    observer = LegacyBeatObserver(generator)
    assert observer(["song_0000"]) is failure_result
    assert generator.apply_first_beat_thresholding is threshold
    assert observer.records[0]["status"] == "failed"
    assert observer.records[0]["captured"] is False


def test_nondebug_calls_are_rejected_before_loading_generator():
    observer = LegacyBeatObserver()
    with pytest.raises(ValueError, match="debug_beats=True"):
        observer(["song_0000"], debug_beats=False)
    assert observer.records == []


def test_import_does_not_load_tensorflow_or_generation_pipeline():
    subprocess.run([sys.executable, "-c", """
import sys
from evaluation.legacy_observer import LegacyBeatObserver
assert 'tensorflow' not in sys.modules
assert 'map_creation.gen_beats' not in sys.modules
assert LegacyBeatObserver().records == []
"""], cwd=ROOT_DIR, check=True)
