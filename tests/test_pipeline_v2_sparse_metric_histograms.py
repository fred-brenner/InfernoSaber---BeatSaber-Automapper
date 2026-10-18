"""Exact Keras PR-count parity without threshold-by-batch comparison tensors."""

from copy import deepcopy

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from training.pipeline_v2.metrics import SparseMetrics


def _assert_auc_equal(actual, reference):
    for name in ("true_positives", "false_positives", "false_negatives", "true_negatives"):
        np.testing.assert_allclose(getattr(actual, name).numpy(), getattr(reference, name).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.result().numpy(), reference.result().numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("threshold_count", [3, 32, 512])
def test_histograms_match_keras_at_threshold_ties_endpoints_and_masked_weighted_chunks(compiled, threshold_count):
    metric = SparseMetrics("arcs", {}, thresholds=threshold_count).auc["attach"]
    reference = tf.keras.metrics.AUC(curve="PR", thresholds=metric.thresholds[1:-1])
    thresholds = np.asarray(metric.thresholds[1:-1], np.float32)
    scores = np.clip(np.r_[0., 1., thresholds,
                           np.nextafter(thresholds, np.float32(-np.inf)),
                           np.nextafter(thresholds, np.float32(np.inf))], 0, 1).astype(np.float32)
    rng = np.random.default_rng(3)
    targets = (rng.random(len(scores)) < .2).astype(np.float32)
    targets[0], targets[1] = 1, 0
    weights = rng.choice([0., .25, .5, 1., 2.], len(scores)).astype(np.float32)
    update = lambda target, score, weight: SparseMetrics.update_auc(metric, target, score, weight)
    if compiled:
        signature = [tf.TensorSpec([None], tf.float32)] * 3
        update = tf.function(update, input_signature=signature, autograph=False)
    for indices in np.array_split(np.arange(len(scores)), 3):
        target, score, weight = (tf.constant(value[indices]) for value in (targets, scores, weights))
        update(target, score, weight)
        reference.update_state(target, score, sample_weight=weight)
        _assert_auc_equal(metric, reference)


@pytest.mark.parametrize("compiled", [False, True])
def test_zero_supervision_and_empty_updates_leave_auc_state_empty(compiled):
    metric = SparseMetrics("arcs", {}, thresholds=32).auc["attach"]
    reference = tf.keras.metrics.AUC(curve="PR", thresholds=metric.thresholds[1:-1])
    update = lambda target, score, weight: SparseMetrics.update_auc(metric, target, score, weight)
    if compiled:
        update = tf.function(update, autograph=False)
    for targets, scores, weights in (([1., 0.], [.2, .9], [0., 0.]), ([], [], [])):
        arrays = [tf.constant(value, tf.float32) for value in (targets, scores, weights)]
        update(*arrays)
        reference.update_state(arrays[0], arrays[1], sample_weight=arrays[2])
    _assert_auc_equal(metric, reference)
    assert float(metric.result()) == 0.


class KerasReferenceMetrics(SparseMetrics):
    @staticmethod
    def update_auc(metric, target, score, weight):
        metric.update_state(tf.reshape(target, [-1]), tf.reshape(score, [-1]), sample_weight=tf.reshape(weight, [-1]))


def _batch(stage):
    rng = np.random.default_rng(3)
    if stage == "lighting":
        vocabulary = {"lighting": [{"type": 0, "value": 0}, {"type": 0, "value": 1}, {"type": 1, "value": 2},
                                    {"type": 8, "value": 0}, {"type": 12, "value": 1}, {"type": 13, "value": 1}]}
        target = (rng.random((3, 4, 6)) < .2).astype(np.float32)
        score = rng.random(target.shape).astype(np.float32)
        weight = rng.choice([0., .5, 1.], target.shape).astype(np.float32)
        weight[-1, -1] = 0  # Padding is excluded per class, independent of predictions.
        return vocabulary, {"events": target}, {"events": score}, {"events": weight}
    if stage == "hazards":
        vocabulary = {"walls": [{"kind": "none"}, {"x": 0}, {"x": 3}]}
        targets = {"wall": rng.integers(0, 3, 11, dtype=np.int32), "bombs": (rng.random((11, 12)) < .1).astype(np.float32)}
        scores = {"wall": rng.dirichlet(np.ones(3), 11).astype(np.float32), "bombs": rng.random((11, 12)).astype(np.float32)}
        weights = {key: rng.choice([0., .5, 1.], target.shape).astype(np.float32) for key, target in targets.items()}
        return vocabulary, targets, scores, weights
    targets = {"attach": rng.integers(0, 2, 11, dtype=np.int32), "shape": rng.random((11, 2)).astype(np.float32),
               "anchor": rng.integers(0, 3, 11, dtype=np.int32)}
    scores = {"attach": rng.dirichlet(np.ones(2), 11).astype(np.float32), "shape": rng.random((11, 2)).astype(np.float32),
              "anchor": rng.dirichlet(np.ones(3), 11).astype(np.float32)}
    weights = {key: rng.choice([0., .5, 1.], 11).astype(np.float32) for key in targets}
    return {}, targets, scores, weights


def _assert_report_equal(actual, reference):
    assert actual.keys() == reference.keys()
    for key, expected in reference.items():
        value = actual[key]
        if isinstance(expected, dict):
            _assert_report_equal(value, expected)
        elif isinstance(expected, (bool, str)) or expected is None:
            assert value == expected
        else:
            np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("stage", ["arcs", "hazards", "lighting"])
@pytest.mark.parametrize("compiled", [False, True])
def test_all_sparse_metric_outputs_match_keras_reference_and_reset(stage, compiled):
    vocabulary, targets, scores, weights = _batch(stage)
    before = deepcopy((targets, scores, weights))
    actual, reference = SparseMetrics(stage, vocabulary, 32), KerasReferenceMetrics(stage, vocabulary, 32)
    update = tf.function(actual.update, autograph=False) if compiled else actual.update
    for start, end in ((0, 1), (1, len(next(iter(targets.values()))))):
        families = [{key: tf.constant(value[start:end]) for key, value in family.items()} for family in (targets, scores, weights)]
        update(*families)
        reference.update(*families)
        _assert_report_equal(actual.result(), reference.result())
    assert actual.result()["pr_auc_thresholds"] == 32
    for previous, current in zip(before, (targets, scores, weights)):
        for key in previous:
            np.testing.assert_array_equal(previous[key], current[key])
    actual.reset()
    reference.reset()
    _assert_report_equal(actual.result(), reference.result())
    assert actual.result()["pr_auc"] == 0.


def test_compiled_auc_update_uses_buckets_without_threshold_batch_tiling():
    metric = SparseMetrics("arcs", {}, 512).auc["attach"]
    update = tf.function(lambda target, score, weight: SparseMetrics.update_auc(metric, target, score, weight),
                         input_signature=[tf.TensorSpec([None], tf.float32)] * 3, autograph=False)
    operations = update.get_concrete_function().graph.get_operations()
    kinds = [operation.type for operation in operations]
    assert "Tile" not in kinds
    assert "LowerBound" in kinds
    assert kinds.count("UnsortedSegmentSum") == 2
    assert all(variable.shape == (512,) for variable in metric.variables)


def test_metric_updates_preserve_trainable_values_and_forward_gradients():
    metrics = SparseMetrics("arcs", {}, 32)
    logits = tf.Variable([-.7, .2, 1.1])
    before = logits.numpy().copy()
    with tf.GradientTape() as tape:
        probabilities = tf.math.sigmoid(logits)
        objective = tf.reduce_sum(probabilities)
        metrics.update_auc(metrics.auc["attach"], tf.constant([0., 1., 1.]), probabilities, tf.ones(3))
    gradient = tape.gradient(objective, logits)
    expected = tf.math.sigmoid(logits).numpy()
    np.testing.assert_allclose(gradient.numpy(), expected * (1 - expected), rtol=1e-6)
    assert np.isfinite(gradient.numpy()).all()
    np.testing.assert_array_equal(logits.numpy(), before)
    assert not any(variable.trainable for variable in metrics.auc["attach"].variables)
