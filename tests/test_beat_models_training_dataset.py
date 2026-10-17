"""Finite beat-pair inputs and callback lifecycle, without fit or updates."""

from threading import get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from training.beat_models import runner, training_input
from training.beat_models.contracts import CORE, FEATURES, FRAME_SECONDS, Song, TIMING_WINDOW
from training.beat_models.training_input import BatchSource, training_callbacks


@pytest.fixture
def songs():
    selected = []
    for index, duration in enumerate((.31, 1.05, 5.23)):
        frames = np.arange(int(np.ceil(duration / FRAME_SECONDS))) * FRAME_SECONDS
        bins = int(np.ceil(duration))
        durations = np.minimum(1., duration - np.arange(bins)).astype(np.float32)
        shares = np.arange(1, bins + 1, dtype=np.float32)
        shares /= shares.sum()
        classes = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.int32)
        event_frames = np.array([0, len(frames) // 2, len(frames) - 1])
        counts = np.zeros(len(frames), dtype=np.int32)
        counts[event_frames] = np.count_nonzero(classes, axis=1)
        arrays = {"features": np.full((len(frames), FEATURES), index + .25, np.float32),
                  "coarse_features": np.full((bins, FEATURES), index + .5, np.float32),
                  "frame_times": frames, "bin_durations": durations, "intensity": shares,
                  "note_classes": classes, "timing": counts, "event_frames": event_frames}
        selected.append(Song(f"song-{index}", {"duration": duration, "bpm": 120.,
                                              "intensity_target_valid": True}, arrays))
    return selected


def assert_batch_equal(actual, expected):
    actual_inputs, actual_target = actual
    expected_inputs, expected_target = expected
    assert actual_inputs.keys() == expected_inputs.keys()
    for key in actual_inputs:
        np.testing.assert_array_equal(actual_inputs[key], expected_inputs[key])
    np.testing.assert_array_equal(actual_target, expected_target)


def expected_batch(stage, references, profiles):
    inputs, target, mask = runner._batch(stage, references, profiles)
    if stage == "timing":
        target = np.stack((target, mask.astype(np.int32)), axis=-1)
    return inputs, target


@pytest.mark.parametrize("stage", ("intensity", "timing"))
@pytest.mark.parametrize("prefetch", (0, 2))
def test_finite_dataset_matches_existing_samples_and_retains_tail(songs, stage, prefetch):
    references = runner._references(songs, stage)
    source = BatchSource(stage, songs, references, batch_size=2 if stage == "intensity" else 3)
    profiles = {} if stage == "intensity" else runner._profiles(songs)
    dataset = source.dataset(prefetch)
    from training.beat_models.models import tf

    assert int(tf.data.experimental.cardinality(dataset)) == len(source)
    options = dataset.options()
    assert options.deterministic is True
    assert options.autotune.enabled is False
    assert options.experimental_optimization.inject_prefetch is False
    first = list(dataset.as_numpy_iterator())
    second = list(dataset.as_numpy_iterator())
    assert len(first) == len(source)
    assert [len(batch[1]) for batch in first] == ([2, 1] if stage == "intensity" else [3, 1])
    for index, (batch, replay) in enumerate(zip(first, second)):
        selected = references[index * source.batch_size:(index + 1) * source.batch_size]
        assert_batch_equal(batch, expected_batch(stage, selected, profiles))
        assert_batch_equal(batch, replay)
    if stage == "intensity":
        assert dataset.element_spec[0]["audio"].shape.as_list() == [None, None, FEATURES]
        assert [batch[0]["audio"].shape[1] for batch in first] == [2, 6]
    else:
        assert dataset.element_spec[0]["audio"].shape.as_list() == [None, TIMING_WINDOW, FEATURES]
        assert first[-1][1].shape == (1, CORE, 2)
        assert int(first[-1][1][..., 1].sum()) == len(songs[-1].arrays["frame_times"]) - CORE


def test_intensity_never_builds_frame_profiles(songs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Intensity input attempted to build timing conditioning")

    monkeypatch.setattr(runner, "_profiles", forbidden)
    source = BatchSource("intensity", songs, runner._references(songs, "intensity"), 1,
                         mode="predicted", shuffle=True)
    source.prepare_epoch()
    assert len(list(source.dataset().as_numpy_iterator())) == 3


def test_noisy_epoch_order_and_values_match_previous_sequence_rng(songs):
    references = runner._references(songs, "timing")
    source = BatchSource("timing", songs, references, 3, mode="noisy-reference",
                         shuffle=True, seed=29, noise_std=.4)
    order, rng = np.arange(len(references)), np.random.default_rng(29)
    for epoch in range(1, 4):
        rng.shuffle(order)
        profiles = runner._profiles(songs, "noisy-reference", rng=rng, noise_std=.4)
        batches = list(source.iter_batches())
        for batch_index, batch in enumerate(batches):
            chosen = order[batch_index * 3:(batch_index + 1) * 3]
            assert_batch_equal(batch, expected_batch("timing", [references[item] for item in chosen], profiles))
        assert source.epoch == epoch
        if epoch < 3:
            source.prepare_epoch()


def test_iterator_keeps_order_and_profiles_from_its_prepared_epoch(songs):
    references = runner._references(songs, "timing")
    source = BatchSource("timing", songs, references, 1, mode="noisy-reference",
                         shuffle=True, seed=7, noise_std=.7)
    snapshot = source.iter_batches()
    previous_order, previous_profiles = source.order.copy(), source.profiles
    source.prepare_epoch()
    for batch, index in zip(snapshot, previous_order):
        assert_batch_equal(batch, expected_batch("timing", [references[index]], previous_profiles))
    assert source.profiles is not previous_profiles


@pytest.mark.parametrize("mode", ("reference", "predicted"))
def test_fixed_profiles_are_cached_and_predictions_stay_on_caller_thread(songs, mode, monkeypatch):
    calls, prediction_threads = [], []
    original = runner._profiles

    def profiles(*args, **kwargs):
        calls.append(get_ident())
        return original(*args, **kwargs)

    class IntensityPredictor:
        def __call__(self, inputs, training):
            assert training is False
            prediction_threads.append(get_ident())
            durations = inputs["durations"]
            return durations / durations.sum(axis=1, keepdims=True)

    monkeypatch.setattr(runner, "_profiles", profiles)
    source = BatchSource("timing", songs, runner._references(songs, "timing"), 3,
                         mode=mode, intensity_model=IntensityPredictor())
    caller = get_ident()
    for _ in range(2):
        list(source.dataset().as_numpy_iterator())
        source.prepare_epoch()
    assert calls == [caller]
    assert prediction_threads == ([caller] * len(songs) if mode == "predicted" else [])


def test_progress_only_reads_loss_at_log_boundaries_and_records_epoch_times(songs, monkeypatch, capsys):
    from training.beat_models.models import keras

    references = runner._references(songs, "intensity")
    train = BatchSource("intensity", songs, references * 68, 1)
    validation = BatchSource("intensity", songs, references * 35, 1)
    report, now = {}, [10.]
    monkeypatch.setattr(training_input, "perf_counter", lambda: now[0])
    callback = training_callbacks(train, validation, 3, report, log_every_batches=100)
    events = []

    class Checkpoint(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            events.append("checkpoint")

    class Loss:
        calls = 0

        def numpy(self):
            self.calls += 1
            return .25

    loss = Loss()
    callbacks = keras.callbacks.CallbackList([Checkpoint(), callback],
                                              model=SimpleNamespace(stop_training=False))
    callbacks.on_epoch_begin(0)
    now[0] = 15.
    for batch in (15, 31, 47, 63, 79, 95):
        callbacks.on_train_batch_begin(batch)
        callbacks.on_train_batch_end(batch, {"loss": loss})
    assert loss.calls == 0
    callbacks.on_train_batch_begin(111)
    callbacks.on_train_batch_end(111, {"loss": loss})
    assert loss.calls == 1
    now[0] = 30.
    callbacks.on_test_begin()
    now[0] = 33.
    callbacks.on_test_batch_begin(95)
    callbacks.on_test_batch_end(95, {"loss": loss})
    assert loss.calls == 1
    callbacks.on_test_batch_begin(104)
    callbacks.on_test_batch_end(104, {"loss": loss})
    assert loss.calls == 2
    now[0] = 35.
    callbacks.on_test_end()
    now[0] = 37.
    logs = {"loss": loss}
    callbacks.on_epoch_end(0, logs)
    assert events == ["checkpoint"]
    assert train.epoch == validation.epoch == 2
    record = report["epoch_performance"][0]
    assert record["seconds"] == 27
    assert record["train"] == {"seconds": 20., "batches": 204, "examples": 204,
                               "examples_per_second": 10.2}
    assert record["validation"] == {"seconds": 5., "batches": 105, "examples": 105,
                                    "examples_per_second": 21.}
    assert logs == {"loss": loss}
    output = capsys.readouterr().out
    assert "batch 112/204" in output
    assert "batch 105/105" in output
    assert "epoch 1: 27.0s" in output


@pytest.mark.parametrize("stopped,epochs", ((True, 3), (False, 1)))
def test_callback_does_not_prepare_after_stopping_or_final_epoch(songs, stopped, epochs):
    source = BatchSource("intensity", songs, runner._references(songs, "intensity"), 2)
    report = {}
    callback = training_callbacks(source, source, epochs, report, verbose=0)
    callback.set_model(SimpleNamespace(stop_training=stopped))
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0, {})
    assert source.epoch == 1
    assert len(report["epoch_performance"]) == 1


@pytest.mark.parametrize("prefetch", (-1, True, 1.5))
def test_prefetch_rejects_invalid_limits_before_tensorflow(songs, prefetch):
    source = BatchSource("intensity", songs, runner._references(songs, "intensity"), 2)
    with pytest.raises(ValueError, match="Prefetch"):
        source.dataset(prefetch)
