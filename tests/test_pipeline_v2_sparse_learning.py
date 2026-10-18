"""Positive supervision/metrics and compiled gradients without optimizer updates."""

from copy import deepcopy

import numpy as np
import pytest

from training.pipeline_v2.supervision import class_balance, validate_support


def test_balance_uses_train_only_and_does_not_change_supervision_masks():
    counts = {"train": {"wall": {"positive": 2., "negative": 8.},
                        "bombs": {"positive": [1., 0.], "negative": [500., 10.]}},
              "validation": {"wall": {"positive": 999., "negative": 1.}}}
    before = deepcopy(counts)
    result = class_balance(counts, "hazards", cap=100.)
    assert result["wall"] == {"positive": 2.5, "negative": .625}
    assert result["bombs"]["positive"] == [100., 1.]
    assert before == counts


def test_zero_positive_validation_is_rejected_before_training():
    counts = {split: {"attach": {"positive": positive, "negative": 10.}}
              for split, positive in (("train", 3.), ("validation", 0.))}
    with pytest.raises(ValueError, match="validation has no supervised positive"):
        validate_support(counts, "arcs")


def test_empty_predictions_cannot_pass_positive_metrics():
    import tensorflow as tf
    from training.pipeline_v2.metrics import SparseMetrics
    metric = SparseMetrics("arcs", {}, thresholds=32)
    targets = {"attach": tf.constant([0, 1]), "shape": tf.constant([[0., 0.], [.5, .5]]),
               "anchor": tf.constant([0, 0])}
    predicted = {"attach": tf.constant([[1., 0.], [1., 0.]]), "shape": targets["shape"],
                 "anchor": tf.constant([[1., 0., 0.], [1., 0., 0.]])}
    weights = {"attach": tf.constant([1., 1.]), "shape": tf.constant([0., 1.]), "anchor": tf.constant([0., 1.])}
    metric.update(targets, predicted, weights)
    result = metric.result()
    assert not result["positive_case_pass"]
    assert result["heads"]["attach"]["recall"] == 0
    assert result["heads"]["attach"]["positive"] == 1
    assert result["shape_mae"] == 0


@pytest.mark.parametrize("stage", ["hazards", "lighting", "arcs"])
def test_balanced_eager_compiled_gradients_and_epoch_metrics_without_updates(stage):
    import tensorflow as tf
    from training.pipeline_v2.data import make_dummy_dataset
    from training.pipeline_v2.runner import batches
    from training.pipeline_v2.models import build_model
    from training.pipeline_v2.execution import objective_gradients, StageExecution
    dataset = make_dummy_dataset()
    examples = next(batches(dataset, stage, batch_size=3))
    model = build_model(stage, dataset.vocabularies)
    balance = ({"wall": {"positive": 2., "negative": .5}, "bombs": {"positive": [3.] * 12, "negative": 1.}}
               if stage == "hazards" else {"events": {"positive": [3.] * len(dataset.vocabularies["lighting"]), "negative": 1.}}
               if stage == "lighting" else {"attach": {"positive": 4., "negative": .5}})
    before = model.get_weights()

    def objective(inputs, targets, weights):
        return objective_gradients(stage, model, inputs, targets, weights, balance=balance)

    eager = objective(examples.inputs, examples.targets, examples.weights)
    graph = tf.function(objective)(examples.inputs, examples.targets, examples.weights)
    np.testing.assert_allclose(eager[2], graph[2], rtol=2e-5, atol=1e-6)
    for a, b in zip(eager[3], graph[3]):
        assert a is not None and b is not None
        a, b = tf.convert_to_tensor(a).numpy(), tf.convert_to_tensor(b).numpy()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=3e-6)
    execution = StageExecution(stage, model, examples, compiled=True, balance=balance, vocabularies=dataset.vocabularies)
    execution.reset()
    execution.step(examples)
    result = execution.metrics.result()
    assert np.isfinite(execution.loss()) and np.isfinite(result["pr_auc"])
    assert all(np.array_equal(old, new) for old, new in zip(before, model.get_weights()))


def test_weighted_objective_and_streaming_metrics_are_batch_partition_invariant():
    import tensorflow as tf
    from training.pipeline_v2.models import loss_components, combine_loss_components
    from training.pipeline_v2.metrics import SparseMetrics
    targets = {"wall": tf.constant([0, 1, 0]), "bombs": tf.constant([[0.] * 12, [1.] * 12, [0.] * 12])}
    outputs = {"wall": tf.constant([[.8, .2], [.3, .7], [.9, .1]]),
               "bombs": tf.constant([[.1] * 12, [.8] * 12, [.4] * 12])}
    weights = {"wall": tf.constant([1., 1., 0.]), "bombs": tf.constant([[1.] * 12, [1.] * 12, [0.] * 12])}
    balance = {"wall": {"positive": 4., "negative": .5}, "bombs": {"positive": [3.] * 12, "negative": 1.}}
    full = loss_components("hazards", targets, outputs, weights, balance=balance)
    total = {head: [0., 0.] for head in targets}
    whole, chunks = (SparseMetrics("hazards", {"walls": [{}, {}]}, thresholds=32) for _ in range(2))
    whole.update(targets, outputs, weights)
    for bounds in (slice(0, 1), slice(1, 3)):
        parts = [{key: values[bounds] for key, values in family.items()} for family in (targets, outputs, weights)]
        chunks.update(*parts)
        local = loss_components("hazards", *parts, balance=balance)
        for head, pair in local.items():
            total[head] = [a + b for a, b in zip(total[head], pair)]
    np.testing.assert_allclose(combine_loss_components("hazards", full), combine_loss_components("hazards", total))
    assert whole.result() == chunks.result()


def test_missing_positive_validation_fails_before_model_or_optimizer_construction(tmp_path, monkeypatch):
    from training.pipeline_v2.data import make_dummy_dataset
    from training.pipeline_v2.runner import train_stage
    from training.pipeline_v2 import models
    dataset = make_dummy_dataset(settings={})
    for song in dataset.songs:
        if song.base.metadata["split"] == "validation":
            song.examples["arcs"].targets["attach"].fill(0)
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid supervision reached model/optimizer construction")
    monkeypatch.setattr(models, "build_model", forbidden)
    monkeypatch.setattr(models.keras.optimizers, "get", forbidden)
    with pytest.raises(ValueError, match="validation has no supervised positive"):
        train_stage(dataset, "arcs", tmp_path / "dry-run", dry_run=True)
    assert not (tmp_path / "dry-run").exists()
