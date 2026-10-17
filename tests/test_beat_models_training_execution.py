"""CPU beat-pair execution checks; optimizer graphs are never executed."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from tools.config import config
from training.beat_models import __main__ as command
from training.beat_models import models, runner
from training.beat_models.artifacts import json_value
from training.beat_models.contracts import CORE


@pytest.fixture(scope="module")
def prepared_data():
    return command._dummy_data()


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    adam = models.keras.optimizers.Adam

    def forbidden(*args, **kwargs):
        pytest.fail("Execution checks must not train models or create production optimizers")

    monkeypatch.setattr(models.keras.optimizers, "get", forbidden)
    monkeypatch.setattr(models.keras.optimizers, "Adam", forbidden)
    monkeypatch.setattr(models.keras.Model, "fit", forbidden)
    monkeypatch.setattr(models.keras.Model, "train_on_batch", forbidden)
    # Only the graph-only test may use this constructor, without any updates.
    yield adam
    models.keras.backend.clear_session()


def _example_batch(prepared_data, stage):
    songs, _ = prepared_data
    references = runner._references(songs, stage)
    selected = references[:3] if stage == "intensity" else references[:2]
    profiles = {} if stage == "intensity" else runner._profiles(songs)
    inputs, target, mask = deepcopy(runner._batch(stage, selected, profiles))
    if stage == "intensity":
        # Explicitly invalid song targets and padded bins must not contribute.
        mask[-1] = False
    else:
        assert mask[0].all() and not mask[-1].all()
        mask[0, 17] = False
    return inputs, target, mask


def _signature(stage, inputs, target, mask):
    def spec(value):
        shape = (None, None, *value.shape[2:]) if stage == "intensity" else (None, *value.shape[1:])
        return tf.TensorSpec(shape, tf.as_dtype(value.dtype))

    return ({key: spec(value) for key, value in inputs.items()}, spec(target),
            tf.TensorSpec((None, *mask.shape[1:]), tf.as_dtype(mask.dtype)))


def _dense(gradient):
    assert gradient is not None
    values = tf.convert_to_tensor(gradient).numpy()
    assert np.isfinite(values).all()
    return values


def _assert_weights_unchanged(model, before):
    assert len(before) == len(model.get_weights())
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("stage", ("intensity", "timing"))
def test_compiled_forward_loss_and_masked_gradients_match_eager_without_updates(prepared_data, stage):
    inputs, target, mask = _example_batch(prepared_data, stage)
    with tf.device("/CPU:0"):
        models.keras.utils.set_random_seed(3)
        model = models.build_model(stage)
        before = model.get_weights()

        def objective(inputs, target, mask):
            with tf.GradientTape() as tape:
                predicted = model(inputs, training=False)
                loss = runner._loss(stage, target, predicted, mask, inputs)
            gradients = tape.gradient(loss, model.trainable_variables)
            components = models.timing_loss_components(target, predicted, mask) if stage == "timing" else ()
            return predicted, loss, gradients, components

        compiled = tf.function(objective, input_signature=_signature(stage, inputs, target, mask),
                               autograph=False)
        eager = objective(inputs, target, mask)
        graph = compiled(inputs, target, mask)
        assert np.isfinite(float(eager[1]))
        np.testing.assert_allclose(graph[0], eager[0], atol=1e-6, rtol=1e-5)
        assert float(graph[1]) == pytest.approx(float(eager[1]), abs=1e-6)
        assert len(eager[2]) == len(graph[2]) == len(model.trainable_variables)
        assert any(np.any(_dense(gradient)) for gradient in eager[2])
        for expected, actual in zip(eager[2], graph[2]):
            np.testing.assert_allclose(_dense(actual), _dense(expected), atol=1e-6, rtol=1e-5)
        if stage == "timing":
            np.testing.assert_allclose(graph[3], eager[3], atol=1e-5, rtol=1e-5)
            ce_sum, frames, huber_sum, chunks = [float(value) for value in graph[3]]
            assert frames == mask.sum() and chunks == len(mask)
            expected_loss = ce_sum / frames + runner.architecture_settings(stage)["count_loss_weight"] * huber_sum / chunks
            assert float(graph[1]) == pytest.approx(expected_loss, abs=1e-6)

        changed = target.copy()
        if stage == "intensity":
            changed[~inputs["mask"]] = 1000
            changed[~mask] = 1000
        else:
            changed[~mask] = (changed[~mask] + 1) % 3
        masked = compiled(inputs, changed, mask)
        assert float(masked[1]) == pytest.approx(float(eager[1]), abs=1e-6)
        for expected, actual in zip(eager[2], masked[2]):
            np.testing.assert_allclose(_dense(actual), _dense(expected), atol=1e-6, rtol=1e-5)

        # A final single-example batch must reuse the graph. Intensity also
        # changes its padded temporal width because song lengths are dynamic.
        songs, _ = prepared_data
        references = runner._references(songs, stage)
        tail = runner._batch(stage, references[1:2], {} if stage == "intensity" else runner._profiles(songs))
        assert len(tail[1]) == 1
        if stage == "intensity":
            assert tail[1].shape[1] != target.shape[1]
        compiled(*tail)
        assert compiled.experimental_get_tracing_count() == 1
        _assert_weights_unchanged(model, before)


@pytest.mark.parametrize("stage", ("intensity", "timing"))
def test_grouped_keras_train_graph_traces_with_zero_optimizer_steps(prepared_data, stage, forbid_training, monkeypatch):
    inputs, target, mask = _example_batch(prepared_data, stage)
    if stage == "timing":
        target = np.stack((target, mask.astype(np.int32)), axis=-1)
    with tf.device("/CPU:0"):
        model = models.build_model(stage)
        before = model.get_weights()
        optimizer = forbid_training(learning_rate=.0005, clipnorm=1.0)
        optimizer.build(model.trainable_variables)
        original_apply = optimizer.apply_gradients

        def graph_only_apply(*args, **kwargs):
            assert not tf.executing_eagerly(), "Optimizer updates may only be traced"
            return original_apply(*args, **kwargs)

        def selected_optimizer(value):
            assert value is optimizer
            return optimizer

        monkeypatch.setattr(optimizer, "apply_gradients", graph_only_apply)
        monkeypatch.setattr(models.keras.optimizers, "get", selected_optimizer)
        model.compile(optimizer=optimizer,
                      loss=models.intensity_objective if stage == "intensity" else models.timing_objective,
                      run_eagerly=False, steps_per_execution=16)
        train_function = model.make_train_function()
        input_spec, _, _ = _signature(stage, inputs, target, mask)
        target_shape = (None, None) if stage == "intensity" else (None, CORE, 2)
        element_spec = (input_spec, tf.TensorSpec(target_shape, tf.as_dtype(target.dtype)))
        # get_concrete_function constructs operations, without an iterator or
        # any execution of the grouped training function.
        graph = train_function.get_concrete_function(tf.data.IteratorSpec(element_spec)).graph
        assert any(operation.type in ("While", "StatelessWhile") for operation in graph.get_operations())
        assert int(optimizer.iterations.numpy()) == 0
        assert int(model._train_counter.numpy()) == 0
        _assert_weights_unchanged(model, before)


@pytest.mark.parametrize("stage,conditioning,compiled", (
    ("intensity", "reference", True),
    ("timing", "reference", True),
    ("timing", "noisy-reference", True),
    ("timing", "predicted", True),
    ("timing", "reference", False),
))
def test_training_entrypoint_wires_bounded_inputs_compile_callbacks_and_reports_without_updates(
        prepared_data, stage, conditioning, compiled, tmp_path, monkeypatch):
    songs, original_manifest = prepared_data
    input_report = {"stage": stage, "mode": "ram", "songs": len(songs), "retained_bytes": 512,
                    "seconds": .125, "integrity_passes": 1}
    manifest = dict(original_manifest, training_input=input_report)
    for key, value in {"prefetch_batches": 2, "compile_steps": compiled,
                       "steps_per_execution": 16, "log_every_batches": 100}.items():
        monkeypatch.setattr(config, f"beat_models_training_{key}", value)
    monkeypatch.setattr(config, f"{stage}_shuffle", False)
    monkeypatch.setattr(config, f"{stage}_verbose", 0)
    monkeypatch.setattr(config, f"{stage}_early_stopping_min_delta", .003)
    if stage == "intensity":
        monkeypatch.setattr(config, "intensity_monitor", "val_kl_divergence")
    selected = runner.training_settings(stage)
    observed = {"profiles": [], "compile": [], "fit": [], "optimizers": [], "upstream": 0}
    original_profiles, original_build = runner._profiles, models.build_model

    def profiles(selected_songs, mode="reference", intensity_model=None, rng=None, noise_std=None):
        result = original_profiles(selected_songs, mode, intensity_model, rng, noise_std)
        observed["profiles"].append((mode, tuple(song.song_id for song in selected_songs),
                                     {key: value.copy() for key, value in result.items()}))
        return result

    def upstream(inputs, training=False):
        assert training is False
        observed["upstream"] += 1
        durations = inputs["durations"]
        return durations / durations.sum(axis=-1, keepdims=True)

    def load_upstream(path, expected_stage, expected_manifest, require_trained=False):
        assert path == "explicit-intensity" and expected_stage == "intensity"
        assert expected_manifest is manifest and require_trained
        return upstream, {"training": {"status": "trained"}}

    def build(selected_stage):
        assert selected_stage == stage
        with tf.device("/CPU:0"):
            model = original_build(selected_stage)
        observed["model"], observed["weights"] = model, model.get_weights()
        return model

    optimizer = SimpleNamespace(iterations=tf.constant(0, tf.int64))

    def optimizer_get(specification):
        observed["optimizers"].append(specification)
        return optimizer

    def compile_spy(model, **kwargs):
        observed["compile"].append(kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Mocked fit must never invoke a training function or optimizer step")

    def fake_fit(model, dataset, *, validation_data, epochs, callbacks, shuffle, verbose):
        assert epochs == 2 and shuffle is False and verbose == 0
        assert isinstance(dataset, tf.data.Dataset) and isinstance(validation_data, tf.data.Dataset)
        names = [type(callback).__name__ for callback in callbacks]
        expected = ["EarlyStopping", "ModelCheckpoint", "TrainingProgress"]
        assert names == (["IntensityMetrics", *expected] if stage == "intensity" else expected)
        early = callbacks[1 if stage == "intensity" else 0]
        checkpoint = callbacks[2 if stage == "intensity" else 1]
        assert early.monitor == checkpoint.monitor == selected["monitor"]
        assert early.restore_best_weights and checkpoint.save_best_only
        assert early.patience == selected["early_stopping_patience"]
        assert early.min_delta == pytest.approx(-selected["early_stopping_min_delta"])
        model.stop_training = False
        callback_list = models.keras.callbacks.CallbackList(
            callbacks, add_history=True, model=model, epochs=epochs,
            steps=int(tf.data.experimental.cardinality(dataset)), verbose=0)
        callback_list.on_train_begin()
        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch)
            snapshots = []
            for split, source in (("train", dataset), ("validation", validation_data)):
                expected_examples = len(runner._references(
                    [song for song in songs if song.metadata["split"] == split], stage))
                assert int(tf.data.experimental.cardinality(source)) == (expected_examples + 3) // 4
                if split == "validation":
                    callback_list.on_test_begin()
                batches = list(source.as_numpy_iterator())
                assert sum(len(target) for _, target in batches) == expected_examples
                assert len(batches[-1][1]) == (expected_examples - 1) % 4 + 1
                for batch_index in range(len(batches)):
                    if split == "train":
                        callback_list.on_train_batch_begin(batch_index)
                        callback_list.on_train_batch_end(batch_index, {"loss": 1.0})
                    else:
                        callback_list.on_test_batch_begin(batch_index)
                        callback_list.on_test_batch_end(batch_index, {"loss": 1.0})
                if split == "validation":
                    callback_list.on_test_end()
                snapshots.append(batches)
            observed["fit"].append(snapshots)
            logs = {"loss": 1.2 / (epoch + 1), "val_loss": 1.5 / (epoch + 1)}
            callback_list.on_epoch_end(epoch, logs)
            if stage == "intensity":
                assert "kl_divergence" in model.history.history
                assert "val_kl_divergence" in model.history.history
        callback_list.on_train_end()
        return model.history

    monkeypatch.setattr(runner, "_profiles", profiles)
    monkeypatch.setattr(runner, "_load_model", load_upstream)
    monkeypatch.setattr(models, "build_model", build)
    monkeypatch.setattr(models.keras.optimizers, "get", optimizer_get)
    monkeypatch.setattr(models.keras.Model, "compile", compile_spy)
    monkeypatch.setattr(models.keras.Model, "fit", fake_fit)
    monkeypatch.setattr(models.keras.Model, "make_train_function", forbidden)
    monkeypatch.setattr(models.keras.Model, "train_step", forbidden)
    output = tmp_path / "run"
    report = runner.train_stage(songs, manifest, stage, output, epochs=2, batch_size=4,
                                conditioning=conditioning,
                                intensity_artifact="explicit-intensity" if conditioning == "predicted" else None)
    assert observed["optimizers"] == [{"class_name": selected["optimizer"],
                                        "config": {"learning_rate": selected["learning_rate"],
                                                   "clipnorm": selected["clipnorm"]}}]
    assert observed["compile"] == [{"optimizer": optimizer,
                                    "loss": models.intensity_objective if stage == "intensity" else models.timing_objective,
                                    "run_eagerly": not compiled,
                                    "steps_per_execution": 16 if compiled else 1}]
    assert report["performance"] == {"prefetch_batches": 2, "compile_steps": compiled,
                                      "steps_per_execution": 16, "log_every_batches": 100,
                                      "effective_steps_per_execution": 16 if compiled else 1}
    assert report["training_input"] == input_report
    assert report["epochs_completed"] == 2 and report["optimizer_steps"] == 0
    assert len(report["epoch_performance"]) == 2
    for epoch, performance in enumerate(report["epoch_performance"], 1):
        assert performance["epoch"] == epoch and performance["seconds"] >= 0
        for split in ("train", "validation"):
            assert performance[split]["examples"] == report[f"{split}_examples"]
            assert performance[split]["batches"] == (report[f"{split}_examples"] + 3) // 4
            assert performance[split]["examples_per_second"] > 0
    assert json.loads((output / "report.json").read_text()) == json_value(report)
    assert json.loads((output / "manifest.json").read_text())["training"] == json_value(report)
    assert (output / "best.keras").is_file()
    assert int(optimizer.iterations.numpy()) == 0
    _assert_weights_unchanged(observed["model"], observed["weights"])
    for split_index in (0, 1):
        changed_conditions = False
        for (first_inputs, first_target), (next_inputs, next_target) in zip(
                observed["fit"][0][split_index], observed["fit"][1][split_index]):
            np.testing.assert_array_equal(first_target, next_target)
            for key in first_inputs:
                if conditioning == "noisy-reference" and split_index == 0 and key == "conditions":
                    changed_conditions |= not np.array_equal(first_inputs[key], next_inputs[key])
                else:
                    np.testing.assert_array_equal(first_inputs[key], next_inputs[key])
        if conditioning == "noisy-reference" and split_index == 0:
            assert changed_conditions
    if stage == "intensity":
        assert observed["profiles"] == []
        assert "val_kl_divergence" in report["history"]
    else:
        modes = [mode for mode, _, _ in observed["profiles"]]
        assert modes.count(conditioning) == 2
        assert len(modes) == (3 if conditioning == "noisy-reference" else 2)
        if conditioning == "noisy-reference":
            noisy = [values for mode, _, values in observed["profiles"] if mode == conditioning]
            assert any(not np.array_equal(noisy[0][song_id], noisy[1][song_id]) for song_id in noisy[0])
        if conditioning == "predicted":
            expected = sum(song.metadata["split"] in ("train", "validation") for song in songs)
            assert observed["upstream"] == expected
