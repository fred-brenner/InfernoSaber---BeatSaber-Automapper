"""CPU execution checks; one Adam is traced only, with zero optimizer steps."""

from copy import deepcopy

import numpy as np
import pytest
import tensorflow as tf

from training.pipeline_v2.contracts import NEW_STAGES, StageExamples
from training.pipeline_v2.execution import StageExecution, objective_gradients
from training.pipeline_v2.models import build_model, stage_loss
from training.pipeline_v2.runner import batches
from training.pipeline_v2.settings import training_settings


@pytest.fixture(scope="module")
def dataset():
    from training.pipeline_v2.data import make_dummy_dataset
    return make_dummy_dataset()


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    adam = tf.keras.optimizers.Adam

    def forbidden(*args, **kwargs):
        pytest.fail("Execution checks must not construct optimizers or train models")

    monkeypatch.setattr(tf.keras.optimizers, "get", forbidden)
    monkeypatch.setattr(tf.keras.optimizers, "Adam", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    # Only the explicit graph-only optimizer test uses the saved constructor.
    return adam


def test_validation_only_execution_returns_masked_loss_without_updates(dataset):
    examples = next(batches(dataset, "positions", batch_size=3))
    with tf.device("/CPU:0"):
        model = build_model("positions", dataset.vocabularies)
        before = model.get_weights()
        expected = float(stage_loss("positions", examples.targets,
                                    model(examples.inputs, training=False), examples.weights))
        execution = StageExecution("positions", model, examples)
        execution.step(examples)
        assert execution.loss() == pytest.approx(expected, abs=1e-6)
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)


def _slice(examples, start, stop):
    return StageExamples(*({key: value[start:stop] for key, value in family.items()}
                           for family in (examples.inputs, examples.targets, examples.weights)),
                         examples.times[start:stop])


def _supervised_examples(dataset, stage):
    examples = deepcopy(next(batches(dataset, stage, batch_size=3)))
    rng = np.random.default_rng(21)
    examples.inputs["audio"][:] = rng.uniform(.1, 1., examples.inputs["audio"].shape)
    for head, target in examples.targets.items():
        if (stage, head) in (("hazards", "bombs"), ("lighting", "events")):
            target[:] = rng.integers(0, 2, target.shape)
        elif (stage, head) == ("arcs", "shape"):
            target[:] = [.2, 1.8]
        else:
            target[:] = 0
        examples.weights[head][:] = 1
        examples.weights[head][-1] = 0
    return examples


def _dense(gradient):
    assert gradient is not None
    value = tf.convert_to_tensor(gradient).numpy()
    assert np.isfinite(value).all()
    return value


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_compiled_forward_objective_and_masked_gradients_match_eager_without_updates(dataset, stage):
    examples = _supervised_examples(dataset, stage)
    with tf.device("/CPU:0"):
        tf.keras.utils.set_random_seed(3)
        model = build_model(stage, dataset.vocabularies)
        before = model.get_weights()

        def objective(inputs, targets, weights):
            return objective_gradients(stage, model, inputs, targets, weights)

        signature = [{key: tf.TensorSpec((None, *value.shape[1:]), tf.as_dtype(value.dtype))
                      for key, value in family.items()}
                     for family in (examples.inputs, examples.targets, examples.weights)]
        compiled = tf.function(objective, input_signature=signature, autograph=False)
        eager = objective(examples.inputs, examples.targets, examples.weights)
        graph = compiled(examples.inputs, examples.targets, examples.weights)
        assert np.isfinite(float(eager[2]))
        assert float(graph[2]) == pytest.approx(float(eager[2]), abs=1e-6)
        for head in eager[0]:
            assert np.isfinite(np.asarray(eager[0][head])).all()
            np.testing.assert_allclose(graph[0][head], eager[0][head], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(graph[1][head], eager[1][head], atol=1e-6, rtol=1e-5)
        assert len(eager[3]) == len(graph[3]) == len(model.trainable_variables)
        assert any(np.any(_dense(gradient) != 0) for gradient in eager[3])
        for expected, actual in zip(eager[3], graph[3]):
            np.testing.assert_allclose(_dense(actual), _dense(expected), atol=1e-6, rtol=1e-5)

        changed = deepcopy(examples.targets)
        for head, target in changed.items():
            target[examples.weights[head] == 0] = 999
        masked = compiled(examples.inputs, changed, examples.weights)
        assert float(masked[2]) == pytest.approx(float(eager[2]), abs=1e-6)
        for expected, actual in zip(eager[3], masked[3]):
            np.testing.assert_allclose(_dense(actual), _dense(expected), atol=1e-6, rtol=1e-5)
        # The final one-row batch reuses the exact same gradient graph.
        tail = _slice(examples, 2, 3)
        compiled(tail.inputs, tail.targets, tail.weights)
        assert compiled.experimental_get_tracing_count() == 1
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_epoch_metrics_preserve_head_masks_across_batch_partitions_and_resets(dataset, stage):
    examples = _supervised_examples(dataset, stage)
    heads = list(examples.weights)
    inactive = heads[0] if stage == "hazards" else heads[-1] if len(heads) > 1 else None
    for index, (head, weight) in enumerate(examples.weights.items()):
        weight[:] = 0
        if head != inactive:
            weight[0] = .25 * (index + 1)
            weight[-1] = 2 * (index + 1)
    with tf.device("/CPU:0"):
        model = build_model(stage, dataset.vocabularies)
        before = model.get_weights()
        expected = float(stage_loss(stage, examples.targets,
                                    model(examples.inputs, training=False), examples.weights))
        for compiled in (False, True):
            execution = StageExecution(stage, model, examples, compiled=compiled)
            with pytest.raises(ValueError, match="No finite"):
                execution.loss()
            for boundaries in ((0, 3), (0, 2, 3), (0, 1, 2, 3)):
                execution.reset()
                for start, stop in zip(boundaries[:-1], boundaries[1:]):
                    execution.step(_slice(examples, start, stop))
                assert execution.loss() == pytest.approx(expected, abs=1e-6)
            assert execution.tracing_counts == {"training": 0, "validation": int(compiled)}
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)


def _probability_examples(probabilities, red_weights):
    values = np.asarray([[probability, *([(1 - probability) / 12] * 12)]
                         for probability in probabilities], np.float32)
    count = len(values)
    return StageExamples(inputs={"red": values, "blue": np.full_like(values, 1 / 13)},
                         targets={"red": np.zeros(count, np.int32), "blue": np.full(count, -999, np.int32)},
                         weights={"red": np.asarray(red_weights, np.float32), "blue": np.zeros(count, np.float32)},
                         times=np.arange(count, dtype=np.float64))


@pytest.mark.parametrize("compiled", (False, True))
def test_epoch_accumulation_retains_small_batches_after_large_weights(compiled):
    # Float32 totals lose the repeated one-unit denominator increments here.
    first = _probability_examples([.5], [2 ** 24])
    following = _probability_examples([.125], [1])
    execution = StageExecution("positions", lambda inputs, training: inputs, first, compiled=compiled)
    with tf.device("/CPU:0"):
        execution.step(first)
        for _ in range(32):
            execution.step(following)
        expected = (np.log(2) * 2 ** 24 + np.log(8) * 32) / (2 ** 24 + 32)
        assert execution.loss() == pytest.approx(expected, abs=1e-7)
        execution.reset()
        execution.step(following)
        assert execution.loss() == pytest.approx(np.log(8), abs=1e-6)
        for values in following.weights.values():
            values[:] = 0
        execution.reset()
        execution.step(following)
        assert execution.loss() == 0


def test_validation_with_explicit_optimizer_never_traces_or_invokes_training(dataset):
    class ForbiddenOptimizer:
        def apply_gradients(self, *args, **kwargs):
            pytest.fail("Validation attempted an optimizer step")

    examples = next(batches(dataset, "positions", batch_size=3))
    with tf.device("/CPU:0"):
        model = build_model("positions", dataset.vocabularies)
        before = model.get_weights()
        execution = StageExecution("positions", model, examples, optimizer=ForbiddenOptimizer())
        execution.step(examples)
        assert np.isfinite(execution.loss())
        assert execution.tracing_counts == {"training": 0, "validation": 1}
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)


@pytest.mark.parametrize("compiled", (False, True))
def test_validation_only_execution_rejects_training_and_nonfinite_objective(dataset, compiled):
    examples = deepcopy(next(batches(dataset, "arcs", batch_size=3)))
    with tf.device("/CPU:0"):
        model = build_model("arcs", dataset.vocabularies)
        before = model.get_weights()
        execution = StageExecution("arcs", model, examples, compiled=compiled)
        with pytest.raises(ValueError, match="explicit optimizer"):
            execution.step(examples, training=True)
        examples.targets["shape"][:] = np.nan
        examples.weights["shape"][:] = 1
        with pytest.raises(tf.errors.InvalidArgumentError, match="Nonfinite arcs validation loss"):
            execution.step(examples)
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)


def test_adam_graph_can_trace_with_all_finite_checks_before_updates_without_execution(dataset, forbid_training):
    examples = next(batches(dataset, "positions", batch_size=3))
    settings = training_settings("positions")
    with tf.device("/CPU:0"):
        model = build_model("positions", dataset.vocabularies)
        before = model.get_weights()
        optimizer = forbid_training(learning_rate=settings["learning_rate"], clipnorm=settings["clipnorm"])
        optimizer.build(model.trainable_variables)
        execution = StageExecution("positions", model, examples, optimizer=optimizer)
        # Tracing builds operations; it never executes this training graph.
        graph = execution._training.get_concrete_function().graph
        checks = {operation for operation in graph.get_operations()
                  if operation.type in ("CheckNumerics", "CheckNumericsV2")}
        updates = [operation for operation in graph.get_operations()
                   if operation.type == "AssignSubVariableOp" and operation.name.startswith("Adam/")]
        assert len(checks) == 1 + len(model.trainable_variables)
        assert updates
        for update in updates:
            ancestors, pending = set(), [update]
            while pending:
                operation = pending.pop()
                if operation not in ancestors:
                    ancestors.add(operation)
                    pending.extend(operation.control_inputs)
                    pending.extend(value.op for value in operation.inputs)
            assert checks <= ancestors
        assert int(optimizer.iterations.numpy()) == 0
        assert execution.tracing_counts == {"training": 1, "validation": 0}
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)


def test_disconnected_variable_is_rejected_before_optimizer_graph_is_built(dataset):
    class ForbiddenOptimizer:
        def apply_gradients(self, *args, **kwargs):
            pytest.fail("Disconnected gradients reached the optimizer")

    examples = next(batches(dataset, "positions", batch_size=3))
    with tf.device("/CPU:0"):
        model = build_model("positions", dataset.vocabularies)
        model.add_weight(name="disconnected", shape=(), trainable=True)
        before = model.get_weights()
        execution = StageExecution("positions", model, examples, optimizer=ForbiddenOptimizer())
        with pytest.raises(ValueError, match="Disconnected positions training variable: disconnected"):
            execution._training.get_concrete_function()
        for old, current in zip(before, model.get_weights()):
            np.testing.assert_array_equal(current, old)
