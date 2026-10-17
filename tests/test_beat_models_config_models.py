"""Configured builders and losses; CPU inference only, without optimizers."""

import numpy as np
import pytest
import tensorflow as tf

from tools.config import config
from training.beat_models.contracts import (CLASSES, CORE, FEATURES, HISTORY, MELS,
                                            PAD, PATCH, START, TIMING_WINDOW)
from training.beat_models.models import build_model, notes_loss, timing_loss, timing_loss_components


@pytest.fixture(autouse=True)
def inference_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Configuration checks must not construct an optimizer or train")

    monkeypatch.setattr(tf.keras.optimizers, "Adam", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    tf.keras.backend.clear_session()
    yield
    tf.keras.backend.clear_session()


def configure(monkeypatch, stage, **values):
    for name, value in values.items():
        monkeypatch.setattr(config, f"{stage}_{name}", value)


def inputs_for(stage):
    rng = np.random.default_rng(3)
    if stage == "intensity":
        return {"audio": rng.normal(size=(1, 7, FEATURES)).astype(np.float32),
                "durations": np.ones((1, 7), dtype=np.float32),
                "mask": np.ones((1, 7), dtype=bool)}
    if stage == "timing":
        return {"audio": rng.normal(size=(1, TIMING_WINDOW, FEATURES)).astype(np.float32),
                "conditions": np.ones((1, TIMING_WINDOW, 4), dtype=np.float32),
                "mask": np.ones((1, TIMING_WINDOW), dtype=bool)}
    history = np.full((1, HISTORY, 2), PAD, dtype=np.int32)
    history[:, 0] = START
    history_mask = np.zeros((1, HISTORY), dtype=bool)
    history_mask[:, 0] = True
    return {"audio": rng.normal(size=(1, PATCH, MELS, 1)).astype(np.float32),
            "audio_mask": np.ones((1, PATCH), dtype=bool), "history": history,
            "history_age": np.zeros((1, HISTORY, 1), dtype=np.float32),
            "history_mask": history_mask, "conditions": np.ones((1, 6), dtype=np.float32)}


def assert_serializes(model, stage, tmp_path):
    inputs = inputs_for(stage)
    expected = model(inputs, training=False)
    path = tmp_path / f"configured_{stage}.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    actual = restored(inputs, training=False)
    if stage != "notes":
        expected, actual = {stage: expected}, {stage: actual}
    for name in expected:
        assert np.isfinite(expected[name]).all()
        np.testing.assert_allclose(np.asarray(expected[name]).sum(axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-6)
    assert restored.count_params() == model.count_params()


@pytest.mark.parametrize("stage,parameters", [("intensity", 60481), ("timing", 180483), ("notes", 173450)])
def test_default_configuration_preserves_parameter_counts(stage, parameters):
    with tf.device("/CPU:0"):
        assert build_model(stage).count_params() == parameters


def test_intensity_builder_reads_configured_temporal_layers(monkeypatch, tmp_path):
    configure(monkeypatch, "intensity", channels=12, kernel_size=5, convolutions_per_block=1,
              dilations=(1, 3), dropout=0.25, activation="tanh")
    with tf.device("/CPU:0"):
        model = build_model("intensity")
        convolutions = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        blocks = [layer for layer in convolutions if layer.kernel_size == (5,)]
        assert len(convolutions) == 4
        assert [layer.filters for layer in blocks] == [12, 12]
        assert [layer.dilation_rate for layer in blocks] == [(1,), (3,)]
        assert model.get_layer("intensity_d1_0_dropout").rate == 0.25
        assert model.get_layer("intensity_d1_0_relu").get_config()["activation"] == "tanh"
        assert_serializes(model, "intensity", tmp_path)


def test_timing_builder_reads_configured_temporal_layers(monkeypatch, tmp_path):
    # One kernel-5 convolution has the same context as two kernel-3 convolutions.
    configure(monkeypatch, "timing", channels=12, kernel_size=5, convolutions_per_block=1,
              dropout=0.3, activation="tanh")
    with tf.device("/CPU:0"):
        model = build_model("timing")
        blocks = [layer for layer in model.layers
                  if isinstance(layer, tf.keras.layers.Conv1D) and layer.kernel_size == (5,)]
        assert len(blocks) == 7
        assert all(layer.filters == 12 for layer in blocks)
        assert model.output_shape == (None, CORE, 3)
        assert model.get_layer("timing_d1_0_dropout").rate == 0.3
        assert model.get_layer("timing_d1_0_relu").get_config()["activation"] == "tanh"
        assert_serializes(model, "timing", tmp_path)


def test_notes_builder_reads_configured_cnn_embeddings_and_history(monkeypatch, tmp_path):
    configure(monkeypatch, "notes", cnn_filters=(8, 12, 16), cnn_kernel_size=5,
              pool_sizes=((2, 2), (2, 2), (5, 4)), audio_dense_units=24, embedding_dim=7,
              gru_units=12, shared_dense_units=20, history_dropout=0.3, activation="tanh")
    with tf.device("/CPU:0"):
        model = build_model("notes")
        convolutions = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        pools = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling2D)]
        assert [layer.filters for layer in convolutions] == [8, 12, 16]
        assert all(layer.kernel_size == (5, 5) for layer in convolutions)
        assert [layer.pool_size for layer in pools] == [(2, 2), (2, 2), (5, 4)]
        assert model.get_layer("audio_features").units == 24
        assert model.get_layer("audio_features").get_config()["activation"] == "tanh"
        assert model.get_layer("hand_embedding").output_dim == 7
        assert model.get_layer("history_gru").input.shape[-1] == 2 * 7 + 1
        assert model.get_layer("history_gru").units == 12
        assert model.get_layer("history_dropout").rate == 0.3
        assert model.get_layer("shared_features").units == 20
        assert all(model.get_layer(hand).units == CLASSES for hand in ("red", "blue"))
        assert_serializes(model, "notes", tmp_path)


@pytest.mark.parametrize("stage,setting,value", [
    ("timing", "chunk_frames", CORE + 1),
    ("timing", "dilations", (1, 2, 4)),
    ("notes", "patch_frames", PATCH + 20),
    ("notes", "history_length", HISTORY + 1),
])
def test_runtime_context_changes_require_reloading_contracts(monkeypatch, stage, setting, value):
    configure(monkeypatch, stage, **{setting: value})
    with pytest.raises(ValueError, match="dimensions changed.*restart"):
        build_model(stage)


@pytest.mark.parametrize("rate_scale,expected_huber", [(20, 8), (200, 0.125)])
def test_timing_loss_reads_configured_weight_huber_delta_and_rate_scale(monkeypatch, rate_scale, expected_huber):
    configure(monkeypatch, "timing", count_loss_weight=0.3, count_huber_delta=2,
              count_rate_scale=rate_scale)
    probabilities = tf.constant([[[0.25, 0.5, 0.25], [0.01, 0.01, 0.98]]])
    target, mask = [[0, 0]], [[True, False]]
    ce, frames, huber, chunks = timing_loss_components(target, probabilities, mask)
    assert float(ce) == pytest.approx(np.log(4))
    assert float(frames) == float(chunks) == 1
    assert float(huber) == pytest.approx(expected_huber)
    assert float(timing_loss(target, probabilities, mask)) == pytest.approx(np.log(4) + 0.3 * expected_huber)


def test_notes_loss_reads_configured_hand_weights_and_mask(monkeypatch):
    configure(monkeypatch, "notes", red_loss_weight=0.8, blue_loss_weight=0.2)
    predicted = {"red": tf.constant([[0.25, 0.25, 0.5], [0.01, 0.49, 0.5]]),
                 "blue": tf.constant([[0.5, 0.25, 0.25], [0.01, 0.49, 0.5]])}
    actual = notes_loss([[0, 0], [0, 0]], predicted, [True, False])
    assert float(actual) == pytest.approx(0.8 * np.log(4) + 0.2 * np.log(2))
