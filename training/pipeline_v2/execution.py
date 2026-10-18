"""Reusable stage steps with device-side masked epoch metrics.

Import this module only when TensorFlow execution is requested. Metrics belong
to the execution object and are never added to the model's weights.
"""

import tensorflow as tf

from . import models
from .settings import architecture_settings


def objective_gradients(stage, model, inputs, targets, weights, training=False, balance=None):
    """Return predictions, loss components, objective, and gradients; never update."""
    with tf.GradientTape() as tape:
        predicted = model(inputs, training=training)
        components = models.loss_components(stage, targets, predicted, weights, balance=balance)
        loss = models.combine_loss_components(stage, components)
    gradients = tape.gradient(loss, model.trainable_variables)
    return predicted, components, loss, gradients


def _signature(family):
    return {key: tf.TensorSpec((None, *value.shape[1:]), tf.as_dtype(value.dtype))
            for key, value in family.items()}


class StageExecution:
    """Keep one graph per mode across epochs and variable-sized final batches."""

    def __init__(self, stage, model, preview, optimizer=None, compiled=True, balance=None, vocabularies=None):
        self.stage = stage
        heads = architecture_settings(stage)["loss_weights"]
        if set(preview.targets) != set(heads) or set(preview.weights) != set(heads):
            raise ValueError("Stage execution requires every configured target and weight head")
        variables = getattr(model, "trainable_variables", ())
        with tf.device(variables[0].device if variables else None):
            self._totals = {head: tf.Variable(tf.zeros(2, tf.float64), trainable=False,
                                              name=f"{stage}_{head}_loss_totals")
                            for head in heads}
        self._unweighted_totals = {head: tf.Variable(tf.zeros(2, tf.float64), trainable=False)
                                   for head in heads} if balance is not None else self._totals
        self._steps = 0
        self._compiled = compiled
        self.metrics = None
        if vocabularies is not None and stage in ("hazards", "lighting", "arcs"):
            from tools.config import config
            from .metrics import SparseMetrics
            self.metrics = SparseMetrics(stage, vocabularies, config.pipeline_v2_metric_thresholds)
        signature = tuple(_signature(family) for family in (preview.inputs, preview.targets, preview.weights))

        def validation(inputs, targets, weights):
            predicted = model(inputs, training=False)
            components = models.loss_components(stage, targets, predicted, weights)
            loss = models.combine_loss_components(stage, components)
            finite = tf.debugging.assert_all_finite(loss, f"Nonfinite {stage} validation loss")
            with tf.control_dependencies([finite]):
                self._accumulate(components)
                if self._unweighted_totals is not self._totals:
                    self._accumulate(components, self._unweighted_totals)
                if self.metrics is not None:
                    self.metrics.update(targets, predicted, weights)

        self._validation = tf.function(validation, input_signature=signature, autograph=False) if compiled else validation
        self._training = None
        if optimizer is not None:
            def training(inputs, targets, weights):
                predicted, components, loss, gradients = objective_gradients(stage, model, inputs, targets, weights,
                                                                     training=True, balance=balance)
                finite = [tf.debugging.assert_all_finite(loss, f"Nonfinite {stage} training loss")]
                for variable, gradient in zip(model.trainable_variables, gradients):
                    if gradient is None:
                        raise ValueError(f"Disconnected {stage} training variable: {variable.name}")
                    values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                    finite.append(tf.debugging.assert_all_finite(values, f"Nonfinite gradient for {variable.name}"))
                # The optimizer must wait for every check, including sparse values.
                # Merely placing assertions earlier in Python does not order a graph.
                with tf.control_dependencies(finite):
                    update = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                with tf.control_dependencies([] if update is None else [update]):
                    self._accumulate(components)
                    if self._unweighted_totals is not self._totals:
                        ordinary = models.loss_components(stage, targets, predicted, weights)
                        self._accumulate(ordinary, self._unweighted_totals)
                    if self.metrics is not None:
                        self.metrics.update(targets, predicted, weights)

            self._training = tf.function(training, input_signature=signature, autograph=False) if compiled else training

    def _accumulate(self, components, totals=None):
        totals = self._totals if totals is None else totals
        for head, pair in components.items():
            totals[head].assign_add(tf.stack([tf.cast(value, tf.float64) for value in pair]))

    def reset(self):
        """Begin an epoch without recreating functions or touching model weights."""
        for total in (*self._totals.values(), *self._unweighted_totals.values()):
            total.assign(tf.zeros(2, tf.float64))
        self._steps = 0
        if self.metrics is not None:
            self.metrics.reset()

    def step(self, examples, training=False):
        """Run a batch without transferring its objective or metrics to Python."""
        function = self._training if training else self._validation
        if function is None:
            raise ValueError("Training execution requires an explicit optimizer")
        function(examples.inputs, examples.targets, examples.weights)
        self._steps += 1

    def loss(self):
        """Combine all head sums/counts and synchronize once for an epoch result."""
        if not self._steps:
            raise ValueError(f"No finite {self.stage} epoch result")
        result = models.combine_loss_components(self.stage, self._totals)
        result = tf.debugging.assert_all_finite(result, f"Nonfinite {self.stage} epoch result")
        return float(result.numpy())

    def unweighted_loss(self):
        return float(models.combine_loss_components(self.stage, self._unweighted_totals).numpy())

    @property
    def tracing_counts(self):
        """JSON-compatible graph counts, without a device synchronization."""
        return {mode: function.experimental_get_tracing_count() if self._compiled and function is not None else 0
                for mode, function in (("training", self._training), ("validation", self._validation))}
