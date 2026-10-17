"""CPU execution/serialization checks; no optimizer is created or applied."""

import numpy as np
import pytest
import tensorflow as tf

from training.beat_models.contracts import (CLASSES, CORE, FEATURES, HISTORY, MELS,
                                            PAD, PATCH, START, TIMING_WINDOW, Song)
from training.beat_models.models import (DurationSoftmax, build_model, intensity_loss,
                                         notes_loss, timing_loss, timing_loss_components)
from training.beat_models.runner import (_gradient_check, _hand_scores, _intensity_scores,
                                         _numpy_outputs, _timing_stage_scores, train_stage)


def sample(stage):
    rng = np.random.default_rng(3)
    if stage == "intensity":
        inputs = {"audio": rng.normal(size=(1, 7, FEATURES)).astype(np.float32),
                  "durations": np.array([[1, 1, 1, 1, 1, 0.13, 0]], dtype=np.float32),
                  "mask": np.array([[True] * 6 + [False]])}
        return inputs, np.array([[0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0]], dtype=np.float32), np.ones(1, dtype=bool)
    if stage == "timing":
        inputs = {"audio": rng.normal(size=(1, TIMING_WINDOW, FEATURES)).astype(np.float32),
                  "conditions": rng.normal(size=(1, TIMING_WINDOW, 4)).astype(np.float32),
                  "mask": np.ones((1, TIMING_WINDOW), dtype=bool)}
        inputs["mask"][:, :254] = False
        target = np.zeros((1, CORE), dtype=np.int32)
        target[0, [0, 23, CORE - 1]] = [1, 2, 1]
        mask = np.ones((1, CORE), dtype=bool)
        mask[:, -10:] = False
        return inputs, target, mask
    history = np.full((2, HISTORY, 2), PAD, dtype=np.int32)
    history[:, 0] = START
    history[1, 1:3] = [[1, 0], [2, 3]]
    mask = np.zeros((2, HISTORY), dtype=bool)
    mask[0, :1] = True
    mask[1, :3] = True
    inputs = {"audio": rng.normal(size=(2, PATCH, MELS, 1)).astype(np.float32),
              "audio_mask": np.ones((2, PATCH), dtype=bool),
              "history": history, "history_age": np.zeros((2, HISTORY, 1), dtype=np.float32),
              "history_mask": mask, "conditions": np.ones((2, 6), dtype=np.float32)}
    return inputs, np.array([[1, 0], [2, 108]], dtype=np.int32), np.ones(2, dtype=bool)


def test_duration_softmax_uses_real_bin_width_and_ignores_padding():
    result = DurationSoftmax()([tf.zeros((1, 3, 1)), tf.constant([[1.0, 0.25, 0.0]]),
                                tf.constant([[True, True, False]])])
    np.testing.assert_allclose(result, [[0.8, 0.2, 0]], atol=1e-6)
    empty = DurationSoftmax()([tf.zeros((1, 3, 1)), tf.zeros((1, 3)), tf.zeros((1, 3), dtype=tf.bool)])
    np.testing.assert_array_equal(empty, np.zeros((1, 3)))


def test_losses_exclude_padded_and_empty_targets_and_normalize_count_rate():
    predicted = tf.constant([[0.25, 0.75]])
    assert float(intensity_loss([[1, 0]], predicted)) == pytest.approx(np.log(4))
    assert float(intensity_loss([[0, 0]], predicted)) == 0
    probabilities = tf.constant([[[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.01, 0.01, 0.98]]])
    loss = timing_loss([[0, 2, 0]], probabilities, [[True, True, False]])
    assert float(loss) == pytest.approx(-np.log(0.8), abs=1e-6)
    # Two valid frames each expecting one note: 100 NPS discrepancy / 10 = 10;
    # Huber(10)=9.5, with auxiliary weight 0.1.
    value = timing_loss([[0, 0]], tf.constant([[[0.25, 0.5, 0.25]] * 2]), [[True, True]])
    assert float(value) == pytest.approx(np.log(4) + 0.95, abs=1e-6)
    assert float(timing_loss([[0, 0]], tf.constant([[[0.25, 0.5, 0.25]] * 2]), [[False, False]])) == 0
    heads = {hand: tf.ones((2, CLASSES)) / CLASSES for hand in ("red", "blue")}
    assert float(notes_loss([[1, 0], [2, 108]], heads, [True, False])) == pytest.approx(np.log(CLASSES))


def test_timing_loss_aggregation_is_invariant_to_batch_partition_and_tail_padding():
    probabilities = tf.constant([[[0.8, 0.1, 0.1]] * 4, [[0.1, 0.8, 0.1]] * 4])
    labels = tf.zeros((2, 4), dtype=tf.int32)
    mask = tf.constant([[True] * 4, [True, False, False, False]])
    whole = [float(value) for value in timing_loss_components(labels, probabilities, mask)]
    partitioned = np.sum([[float(value) for value in timing_loss_components(labels[index:index + 1], probabilities[index:index + 1], mask[index:index + 1])]
                          for index in range(2)], axis=0)
    np.testing.assert_allclose(partitioned, whole, atol=1e-6)
    assert partitioned[0] / partitioned[1] + 0.1 * partitioned[2] / partitioned[3] == pytest.approx(float(timing_loss(labels, probabilities, mask)), abs=1e-6)


def test_teacher_forced_metrics_cover_hands_profiles_and_valid_timing_only():
    hands = _hand_scores([[1, 0], [2, 108], [9, 10]], [[1, 4], [3, 108], [18, 0]])
    assert hands["red"]["class_accuracy"] == pytest.approx(1 / 3)
    assert hands["red"]["position_accuracy"] == pytest.approx(2 / 3)
    assert hands["red"]["direction_accuracy"] == pytest.approx(2 / 3)
    assert hands["blue"]["presence_accuracy"] == pytest.approx(1 / 3)
    assert hands["blue"]["compared_colored_notes"] == 1
    song = Song("metrics", {"duration": .023, "split": "test"}, {
        "frame_times": np.array([0, .01, .02]), "event_frames": np.array([0, 2]),
        "event_times": np.array([.001, .022]), "timing": np.array([1, 0, 2]),
        "note_classes": np.array([[1, 0], [2, 3], [4, 0]]), "intensity": np.array([.25, .75])})
    intensity = _intensity_scores(song, [.5, .5])
    assert intensity["share_mae"] == .25
    assert intensity["bin_count_mae"] == 1
    timing = _timing_stage_scores(song, np.array([1, 2, 2]), np.array([1, 2, 2]), np.array([True, False, True]))
    assert timing["valid_frames"] == 2
    assert timing["evaluated_duration_seconds"] == pytest.approx(.013)
    assert timing["colored_note_count_error"] == 0
    assert timing["timing"]["f1"] == 1


@pytest.mark.parametrize("stage", ["intensity", "timing", "notes"])
def test_stage_forward_gradients_and_safe_serialization_without_updates(stage, tmp_path):
    tf.keras.utils.set_random_seed(3)
    with tf.device("/CPU:0"):
        model = build_model(stage)
        assert 0 < model.count_params() < 1_000_000
        before = [value.copy() for value in model.get_weights()]
        batch = sample(stage)
        report = _gradient_check(model, stage, batch)
        assert report["finite_gradients"]
        assert report["gradient_variables"] == len(model.trainable_variables)
        expected = _numpy_outputs(model(batch[0], training=False))
        for output in expected.values():
            assert np.isfinite(output).all()
            np.testing.assert_allclose(output.sum(axis=-1), 1, atol=1e-6)
        path = tmp_path / f"{stage}.keras"
        model.save(path)
        restored = tf.keras.models.load_model(path, compile=False)
        actual = _numpy_outputs(restored(batch[0], training=False))
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
        for old, current, loaded in zip(before, model.get_weights(), restored.get_weights()):
            np.testing.assert_array_equal(old, current)
            np.testing.assert_array_equal(old, loaded)


def test_timing_predictions_ignore_masked_context_values():
    model = build_model("timing")
    inputs, _, _ = sample("timing")
    expected = model(inputs, training=False).numpy()
    inputs["audio"][~inputs["mask"]] = 10000
    inputs["conditions"][~inputs["mask"]] = -10000
    np.testing.assert_array_equal(model(inputs, training=False), expected)


def test_intensity_valid_predictions_are_independent_of_padded_length():
    model = build_model("intensity")
    inputs, _, _ = sample("intensity")
    unpadded = {key: value[:, :6] for key, value in inputs.items()}
    padded = {key: np.pad(value, [(0, 0), (0, 21)] + ([(0, 0)] if value.ndim == 3 else []))
              for key, value in inputs.items()}
    expected = model(unpadded, training=False).numpy()
    actual = model(padded, training=False).numpy()
    np.testing.assert_allclose(actual[:, :6], expected, atol=1e-6, rtol=1e-6)
    assert not actual[:, 6:].any()


def test_note_predictions_ignore_padded_history_tokens_and_ages():
    model = build_model("notes")
    inputs, _, _ = sample("notes")
    expected = _numpy_outputs(model(inputs, training=False))
    inputs["history"][~inputs["history_mask"]] = 108
    inputs["history_age"][~inputs["history_mask"]] = 10000
    actual = _numpy_outputs(model(inputs, training=False))
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_note_predictions_ignore_masked_audio_frames():
    model = build_model("notes")
    inputs, _, _ = sample("notes")
    inputs["audio_mask"][:, :70] = False
    expected = _numpy_outputs(model(inputs, training=False))
    inputs["audio"][~inputs["audio_mask"]] = 10000
    actual = _numpy_outputs(model(inputs, training=False))
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert model.count_params() == 173450


@pytest.mark.parametrize("stage", ["intensity", "timing", "notes"])
def test_training_dry_run_never_constructs_optimizer_or_calls_fit(stage, tmp_path, monkeypatch):
    from training.beat_models.__main__ import _dummy_data

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted to construct an optimizer or train")

    monkeypatch.setattr(tf.keras.optimizers, "Adam", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    songs, manifest = _dummy_data()
    report = train_stage(songs, manifest, stage, tmp_path / stage, dry_run=True)
    assert report["status"] == "untrained"
    assert report["optimizer_steps"] == 0
    assert np.isfinite(report["preview_loss"])
    if stage == "intensity":
        assert report["intensity_references"]["train"]["valid_songs"] == report["train_examples"]
        assert report["intensity_references"]["validation"]["valid_songs"] == report["validation_examples"]
        assert np.isfinite(report["preview_metrics"]["kl_divergence"])
        assert np.isfinite(report["preview_metrics"]["uniform_rate_loss"])
        assert "uniform_rate_improvement_pct" in report["metric_definitions"]
    assert [path.name for path in (tmp_path / stage).iterdir()] == ["report.json"]
