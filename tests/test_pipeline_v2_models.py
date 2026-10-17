"""CPU v2 neural checks: inference, differentiation, and reload; zero updates."""

from copy import deepcopy

import numpy as np
import pytest
import tensorflow as tf

from training.pipeline_v2.contracts import NEW_STAGES
from training.pipeline_v2.models import (build_model, combine_loss_components,
                                         loss_components, stage_loss)
from training.pipeline_v2.runner import (batches, gradient_check, load_stage_artifact,
                                         smoke_models, train_stage)
from training.pipeline_v2.settings import architecture_settings


@pytest.fixture(scope="module")
def dataset():
    from training.pipeline_v2.data import make_dummy_dataset
    return make_dummy_dataset()


def _examples(dataset, stage):
    return next(batches(dataset, stage, batch_size=3))


def _numpy_outputs(model, inputs):
    return {key: np.asarray(value) for key, value in model(inputs, training=False).items()}


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_forward_active_gradients_serialization_without_updates(dataset, stage, tmp_path):
    tf.keras.utils.set_random_seed(3)
    examples = _examples(dataset, stage)
    with tf.device("/CPU:0"):
        model = build_model(stage, dataset.vocabularies)
        before = model.get_weights()
        check = gradient_check(stage, model, examples)
        assert check["finite_gradients"] and check["weights_unchanged"]
        assert check["active_gradient_variables"] > 0
        assert check["gradient_variables"] == len(model.trainable_variables)
        expected = _numpy_outputs(model, examples.inputs)
        dimensions = {"positions": {"red": 13, "blue": 13},
                      "arrows": {"red": 9, "blue": 9},
                      "multiplier": {"red": len(dataset.vocabularies["multiplier"]),
                                     "blue": len(dataset.vocabularies["multiplier"])},
                      "hazards": {"wall": len(dataset.vocabularies["walls"]), "bombs": 12},
                      "arcs": {"attach": 2, "shape": 2, "anchor": 3},
                      "lighting": {"events": len(dataset.vocabularies["lighting"])}}
        assert set(expected) == set(dimensions[stage])
        for head, values in expected.items():
            assert values.shape[-1] == dimensions[stage][head]
            assert np.isfinite(values).all()
            if (stage, head) == ("arcs", "shape"):
                low, high = architecture_settings(stage)["shape_bounds"]
                assert np.all((values >= low) & (values <= high))
            else:
                assert np.all((values >= 0) & (values <= 1))
                if (stage, head) not in (("hazards", "bombs"), ("lighting", "events")):
                    np.testing.assert_allclose(values.sum(axis=-1), 1, atol=1e-6)
        checkpoint = tmp_path / f"{stage}.keras"
        model.save(checkpoint)
        loaded = tf.keras.models.load_model(checkpoint, compile=False)
        actual = _numpy_outputs(loaded, examples.inputs)
        for head in expected:
            np.testing.assert_allclose(actual[head], expected[head], atol=1e-6, rtol=1e-6)
        for old, current, restored in zip(before, model.get_weights(), loaded.get_weights()):
            np.testing.assert_array_equal(current, old)
            np.testing.assert_array_equal(restored, old)


@pytest.mark.parametrize("stage", [stage for stage in NEW_STAGES if stage != "lighting"])
def test_padding_history_ids_values_and_ages_cannot_affect_predictions(dataset, stage):
    examples = _examples(dataset, stage)
    inputs = deepcopy(examples.inputs)
    model = build_model(stage, dataset.vocabularies)
    expected = _numpy_outputs(model, inputs)
    padding = ~inputs["history_mask"]
    inputs["history"][padding] = 100000
    inputs["history_age"][padding] = -100000
    actual = _numpy_outputs(model, inputs)
    for head in expected:
        np.testing.assert_array_equal(actual[head], expected[head])
    inputs["history_mask"][:] = False
    empty = _numpy_outputs(model, inputs)
    inputs["history"][:] = -100000
    inputs["history_age"][:] = 100000
    for head, value in _numpy_outputs(model, inputs).items():
        np.testing.assert_array_equal(value, empty[head])


def test_arc_endpoint_ids_have_no_numeric_influence(dataset):
    examples = _examples(dataset, "arcs")
    inputs = deepcopy(examples.inputs)
    model = build_model("arcs", dataset.vocabularies)
    expected = _numpy_outputs(model, inputs)
    inputs["endpoints"][:] = 999999
    for head, value in _numpy_outputs(model, inputs).items():
        np.testing.assert_array_equal(value, expected[head])


def test_lighting_current_and_history_masks_are_independent(dataset):
    examples = _examples(dataset, "lighting")
    inputs = {key: value[:1].copy() for key, value in examples.inputs.items()}
    inputs["mask"][:] = False
    inputs["mask"][:, :3] = True
    inputs["history_mask"][:] = False
    inputs["history_mask"][:, -2:] = True
    inputs["history"][:, -2:] = 1
    model = build_model("lighting", dataset.vocabularies)
    expected = _numpy_outputs(model, inputs)["events"]
    inputs["audio"][~inputs["mask"]] = 100000
    inputs["gaps"][~inputs["mask"]] = -100000
    inputs["history"][~inputs["history_mask"]] = 100000
    np.testing.assert_array_equal(_numpy_outputs(model, inputs)["events"], expected)
    # A short current block still receives the right-aligned preceding history.
    inputs["history"][:, -2:] = 0
    assert not np.array_equal(_numpy_outputs(model, inputs)["events"], expected)


def test_absent_heads_and_unattached_arc_controls_are_excluded():
    arrows = {"red": tf.constant([[.8] + [.025] * 8]), "blue": tf.ones((1, 9)) / 9}
    target, weights = {"red": np.array([0]), "blue": np.array([-999])}, {"red": np.ones(1), "blue": np.zeros(1)}
    assert float(stage_loss("arrows", target, arrows, weights)) == pytest.approx(-np.log(.8), abs=1e-6)
    arcs = {"attach": tf.constant([[.9, .1]]), "shape": tf.constant([[1., 1.]]),
            "anchor": tf.constant([[.2, .3, .5]])}
    labels = {"attach": np.array([0]), "shape": np.array([[999., 999.]]), "anchor": np.array([-999])}
    masks = {"attach": np.ones(1), "shape": np.zeros(1), "anchor": np.zeros(1)}
    assert float(stage_loss("arcs", labels, arcs, masks)) == pytest.approx(-np.log(.9), abs=1e-6)
    assert float(stage_loss("arcs", labels, arcs, {key: np.zeros(1) for key in masks})) == 0


def test_multilabel_losses_keep_simultaneous_targets_and_ignore_masked_cells():
    wall = tf.constant([[.8, .2]])
    bombs = tf.constant([[.9, .8] + [.1] * 10])
    target = {"wall": np.array([-999]), "bombs": np.array([[1, 1] + [0] * 10], np.float32)}
    weights = {"wall": np.zeros(1), "bombs": np.ones(1)}
    expected = -(11 * np.log(.9) + np.log(.8)) / 12
    assert float(stage_loss("hazards", target, {"wall": wall, "bombs": bombs}, weights)) == pytest.approx(expected, abs=1e-6)
    output = {"events": tf.constant([[[.9, .8], [.1, .1]]])}
    labels = {"events": np.array([[[1, 1], [1, 1]]], np.float32)}
    mask = {"events": np.array([[1, 0]], np.float32)}
    assert float(stage_loss("lighting", labels, output, mask)) == pytest.approx(-(np.log(.9) + np.log(.8)) / 2, abs=1e-6)


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_loss_aggregation_is_independent_of_batch_partition(dataset, stage):
    examples = _examples(dataset, stage)
    model = build_model(stage, dataset.vocabularies)
    predicted = model(examples.inputs, training=False)
    full = loss_components(stage, examples.targets, predicted, examples.weights)
    sums = {head: np.zeros(2) for head in full}
    for index in range(len(examples.times)):
        targets = {key: value[index:index + 1] for key, value in examples.targets.items()}
        weights = {key: value[index:index + 1] for key, value in examples.weights.items()}
        outputs = {key: value[index:index + 1] for key, value in predicted.items()}
        partial = loss_components(stage, targets, outputs, weights)
        for head, pair in partial.items():
            sums[head] += [float(value) for value in pair]
    assert float(combine_loss_components(stage, sums)) == pytest.approx(float(combine_loss_components(stage, full)), abs=1e-6)


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_training_dry_run_never_constructs_optimizer_or_calls_fit(dataset, stage, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted training")

    monkeypatch.setattr(tf.keras.optimizers, "get", forbidden)
    monkeypatch.setattr(tf.keras.optimizers, "Adam", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    report = train_stage(dataset, stage, tmp_path / stage, dry_run=True, batch_size=2)
    assert report["status"] == "untrained"
    assert report["optimizer_steps"] == 0 and report["weights_unchanged"]
    assert np.isfinite(report["preview_loss"])
    assert sorted(path.name for path in (tmp_path / stage).iterdir()) == ["report.json"]


def test_streaming_batches_keep_splits_and_examples(dataset):
    for stage in NEW_STAGES:
        expected = sum(len(song.examples[stage].times) for song in dataset.songs if song.base.metadata["split"] == "train")
        streamed = list(batches(dataset, stage, "train", batch_size=3))
        assert sum(len(batch.times) for batch in streamed) == expected
        assert all(len(batch.times) <= 3 for batch in streamed)


def test_all_stage_smoke_artifacts_explicit_load_and_manifest_compatibility(dataset, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Smoke attempted training")

    monkeypatch.setattr(tf.keras.optimizers, "get", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    report = smoke_models(dataset, tmp_path / "smoke", max_examples=2)
    assert report["optimizer_steps"] == 0
    assert set(report["stages"]) == set(NEW_STAGES)
    for stage in NEW_STAGES:
        selected = tmp_path / "smoke" / stage
        model, manifest = load_stage_artifact(selected, stage, dataset)
        assert manifest["status"] == "untrained"
        assert model.name == stage
        assert report["stages"][stage]["weights_unchanged"]
        assert report["stages"][stage]["roundtrip_max_error"] <= 1e-6
    changed = deepcopy(dataset)
    changed.manifest["encoder_identity"]["checkpoint_fingerprint"] = "different-encoder"
    with pytest.raises(ValueError, match="encoder, vocabulary"):
        load_stage_artifact(tmp_path / "smoke" / "positions", "positions", changed)


def test_training_rejects_structural_dummy_before_any_optimizer(dataset, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dummy training attempted to create an optimizer")

    monkeypatch.setattr(tf.keras.optimizers, "get", forbidden)
    with pytest.raises(ValueError, match="real prepared"):
        train_stage(dataset, "positions", tmp_path / "forbidden", dry_run=False)
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_nonzero_audio_history_and_every_active_head_receive_gradients(dataset, stage):
    """Structural zero latents alone cannot demonstrate learnable audio branches."""
    examples = deepcopy(_examples(dataset, stage))
    rng = np.random.default_rng(21)
    inputs = examples.inputs
    inputs["audio"][:] = rng.uniform(.1, 1., inputs["audio"].shape)
    history_mask = inputs["history_mask"]
    history_mask[:] = False
    history_mask[:, :6] = True
    history = inputs["history"]
    if stage == "lighting":
        history[:] = rng.integers(0, 2, history.shape)
        inputs["mask"][:] = True
        inputs["gaps"][:] = rng.uniform(.1, .5, inputs["gaps"].shape)
    else:
        inputs["history_age"][:] = np.linspace(.6, .1, history.shape[1])[None, :, None]
        if stage in ("positions", "arrows", "multiplier"):
            classes = {"positions": 13, "arrows": 9,
                       "multiplier": len(dataset.vocabularies["multiplier"])}[stage]
            history[:] = rng.integers(0, classes, history.shape)
        elif stage == "hazards":
            history[..., 0] = rng.integers(0, len(dataset.vocabularies["walls"]), history.shape[:2])
            history[..., 1:] = rng.integers(0, 2, history[..., 1:].shape)
        else:
            history[..., 0] = 1
            history[..., 1:3] = [.7, 1.2]
            history[..., 3] = 1
    for values in examples.weights.values():
        values[:] = 1
    if stage == "arcs":
        examples.targets["attach"][:] = 1
        examples.targets["shape"][:] = [.2, 1.8]
        examples.targets["anchor"][:] = 2
    elif stage == "hazards":
        examples.targets["wall"][:] = min(1, len(dataset.vocabularies["walls"]) - 1)
        examples.targets["bombs"][:] = [1, 0] * 6
    elif stage == "lighting":
        examples.targets["events"][:] = rng.integers(0, 2, examples.targets["events"].shape)
    model = build_model(stage, dataset.vocabularies)
    before = model.get_weights()
    with tf.device("/CPU:0"), tf.GradientTape() as tape:
        predictions = model(inputs, training=False)
        loss = stage_loss(stage, examples.targets, predictions, examples.weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    by_variable = {id(variable): gradient for variable, gradient in zip(model.trainable_variables, gradients)}
    for gradient in gradients:
        assert gradient is not None
        value = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
        assert bool(tf.reduce_all(tf.math.is_finite(value)))
    required = [model.get_layer("audio_conv").kernel if stage == "lighting" else model.get_layer("audio_features").kernel]
    recurrent = ["history_lstm", "time_lstm", "combined_lstm"] if stage == "lighting" else ["history_gru"]
    for name in recurrent:
        required.extend([model.get_layer(name).cell.kernel, model.get_layer(name).cell.recurrent_kernel])
    heads = {"positions": ["red", "blue"], "arrows": ["red", "blue"],
             "multiplier": ["red", "blue"], "hazards": ["wall", "bombs"],
             "arcs": ["attach", "bounded_controls", "anchor"], "lighting": ["event_scores"]}
    required.extend(model.get_layer(name).kernel for name in heads[stage])
    for variable in required:
        assert bool(tf.reduce_any(by_variable[id(variable)] != 0)), variable.name
    for old, current in zip(before, model.get_weights()):
        np.testing.assert_array_equal(current, old)


def test_lazy_split_index_avoids_loading_unselected_song_arrays(dataset):
    class IndexedSongs:
        def __init__(self):
            self.records = [{"split": song.base.metadata["split"]} for song in dataset.songs]
            self.loaded = []

        def __getitem__(self, index):
            self.loaded.append(index)
            return dataset.songs[index]

    from training.pipeline_v2.contracts import V2Dataset
    songs = IndexedSongs()
    lazy = V2Dataset(songs, dataset.vocabularies, dataset.manifest)
    list(batches(lazy, "positions", "validation", batch_size=2))
    assert songs.loaded == [index for index, record in enumerate(songs.records) if record["split"] == "validation"]



def test_synthetic_recordings_remain_forbidden_with_claimed_retained_encoder(dataset, tmp_path, monkeypatch):
    claimed = deepcopy(dataset)
    claimed.manifest["status"] = "prepared"
    claimed.manifest["encoder_identity"].update(status="retained", checkpoint_fingerprint="claimed-real")

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic training attempted to construct a model or optimizer")

    from training.pipeline_v2 import models
    monkeypatch.setattr(models, "build_model", forbidden)
    monkeypatch.setattr(tf.keras.optimizers, "get", forbidden)
    with pytest.raises(ValueError, match="Dummy examples"):
        train_stage(claimed, "positions", tmp_path / "forbidden", dry_run=False)
    assert not (tmp_path / "forbidden").exists()
