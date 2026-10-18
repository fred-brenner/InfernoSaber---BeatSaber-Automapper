"""Bounded device-side sparse-object validation metrics; no model state."""

import numpy as np
import tensorflow as tf


class SparseMetrics:
    def __init__(self, stage, vocabularies, thresholds=512):
        if isinstance(thresholds, bool) or not isinstance(thresholds, int) or thresholds < 3:
            raise ValueError("Metric thresholds must be an integer of at least three")
        self.stage = stage
        self.vocabularies = vocabularies
        heads = {"hazards": ("wall", "bombs"), "lighting": ("events",), "arcs": ("attach",)}[stage]
        # Logit-spaced thresholds preserve resolution for rare-object scores.
        points = (1 / (1 + np.exp(-np.linspace(-16, 16, thresholds - 2)))).tolist()
        self.auc = {head: tf.keras.metrics.AUC(curve="PR", thresholds=points, name=f"{stage}_{head}_pr_auc")
                    for head in heads}
        self.counts = {head: tf.Variable(tf.zeros(6, tf.float64), trainable=False) for head in heads}
        self.extra = {}
        for key in (("wall_geometry",) if stage == "hazards" else ("shape_mae", "anchor_accuracy") if stage == "arcs" else ()):
            self.extra[key] = tf.Variable(tf.zeros(2, tf.float64), trainable=False)
        if stage == "arcs":
            for key in ("anchor_reference_counts", "anchor_prediction_counts"):
                self.extra[key] = tf.Variable(tf.zeros(3, tf.float64), trainable=False)
        if stage == "lighting":
            self.types = sorted({int(record["type"]) for record in vocabularies["lighting"]})
            for kind in self.types:
                self.extra[f"type_{kind}"] = tf.Variable(tf.zeros(3, tf.float64), trainable=False)
            self.extra["non_off_light"] = tf.Variable(tf.zeros(3, tf.float64), trainable=False)
            self.extra["off_trigger_counts"] = tf.Variable(tf.zeros(2, tf.float64), trainable=False)

    def reset(self):
        for metric in self.auc.values():
            metric.reset_state()
        for value in (*self.counts.values(), *self.extra.values()):
            value.assign(tf.zeros_like(value))

    @staticmethod
    def weights(weight, values):
        weight = tf.cast(weight, tf.float32)
        while weight.shape.rank < values.shape.rank:
            weight = weight[..., None]
        return tf.broadcast_to(weight, tf.shape(values))

    @staticmethod
    def confusion(target, chosen, weight):
        target, chosen, weight = tf.cast(target, tf.float64), tf.cast(chosen, tf.float64), tf.cast(weight, tf.float64)
        return tf.stack([tf.reduce_sum(target * chosen * weight),
                         tf.reduce_sum((1 - target) * chosen * weight),
                         tf.reduce_sum(target * (1 - chosen) * weight)])

    @staticmethod
    def update_auc(metric, target, score, weight):
        """Accumulate exact threshold counts using O(samples + thresholds) storage.

        Keras uses strict ``score > threshold`` comparisons. Left insertion
        puts ties in the negative bucket, including duplicate float thresholds.
        The existing Keras metric retains its PR interpolation and reset logic.
        """
        dtype = metric.true_positives.dtype
        thresholds = tf.constant(metric.thresholds, dtype=dtype)
        scores = tf.reshape(tf.cast(score, dtype), [-1])
        labels = tf.reshape(tf.cast(target, tf.bool), [-1])
        weights = tf.reshape(tf.cast(weight, dtype), [-1])
        buckets = tf.searchsorted(thresholds, scores, side="left")
        segments = len(metric.thresholds) + 1
        positive = tf.math.unsorted_segment_sum(tf.where(labels, weights, tf.zeros_like(weights)), buckets, segments)
        negative = tf.math.unsorted_segment_sum(tf.where(labels, tf.zeros_like(weights), weights), buckets, segments)
        metric.true_positives.assign_add(tf.math.cumsum(positive, reverse=True)[1:])
        metric.false_positives.assign_add(tf.math.cumsum(negative, reverse=True)[1:])
        metric.false_negatives.assign_add(tf.math.cumsum(positive)[:-1])
        metric.true_negatives.assign_add(tf.math.cumsum(negative)[:-1])

    def update(self, targets, predicted, weights):
        for head, metric in self.auc.items():
            target = tf.cast(targets[head] > 0, tf.float32)
            score = predicted[head]
            if head == "wall":
                score = 1 - score[..., 0]
                chosen = tf.argmax(predicted[head], axis=-1) > 0
            elif head == "attach":
                score = score[..., 1]
                chosen = score >= .5
            else:
                chosen = score >= .5
            weight = self.weights(weights[head], target)
            if self.stage == "lighting":
                selected = tf.zeros_like(chosen)
                for kind in self.types:
                    indices = [i for i, record in enumerate(self.vocabularies["lighting"]) if record["type"] == kind]
                    local = tf.gather(score, indices, axis=-1)
                    best = tf.argmax(local, axis=-1, output_type=tf.int32)
                    category = tf.gather(tf.constant(indices, tf.int32), best)
                    active = tf.reduce_max(local, axis=-1) >= .5
                    projected = tf.one_hot(category, score.shape[-1], dtype=tf.int32) > 0
                    selected |= projected & active[..., None]
                    self.extra[f"type_{kind}"].assign_add(self.confusion(
                        tf.gather(target, indices, axis=-1), tf.gather(selected, indices, axis=-1),
                        tf.gather(weight, indices, axis=-1)))
                chosen = selected
                records = self.vocabularies["lighting"]
                on = [i for i, r in enumerate(records) if 0 <= r["type"] <= 4 and r["value"] > 0]
                self.extra["non_off_light"].assign_add(self.confusion(
                    tf.gather(target, tf.constant(on, tf.int32), axis=-1), tf.gather(chosen, tf.constant(on, tf.int32), axis=-1), tf.gather(weight, tf.constant(on, tf.int32), axis=-1)))
                off = [i for i, r in enumerate(records) if 0 <= r["type"] <= 4 and r["value"] == 0]
                triggers = [i for i, r in enumerate(records) if r["type"] in (8, 9)]
                self.extra["off_trigger_counts"].assign_add(tf.stack([
                    tf.reduce_sum(tf.cast(tf.gather(chosen, tf.constant(idx, tf.int32), axis=-1), tf.float64) *
                                  tf.cast(tf.gather(weight, tf.constant(idx, tf.int32), axis=-1), tf.float64)) for idx in (off, triggers)]))
            self.update_auc(metric, target, score, weight)
            confusion = self.confusion(target, chosen, weight)
            totals = tf.stack([tf.reduce_sum(tf.cast(target * weight, tf.float64)),
                               tf.reduce_sum(tf.cast(weight, tf.float64)),
                               tf.reduce_sum(tf.cast(tf.cast(chosen, tf.float32) * weight, tf.float64))])
            self.counts[head].assign_add(tf.concat([confusion, totals], axis=0))
        if self.stage == "hazards":
            weight = tf.cast(weights["wall"], tf.float64) * tf.cast(targets["wall"] > 0, tf.float64)
            correct = tf.cast(tf.argmax(predicted["wall"], -1, output_type=tf.int32) == tf.cast(targets["wall"], tf.int32), tf.float64)
            self.extra["wall_geometry"].assign_add(tf.stack([tf.reduce_sum(correct * weight), tf.reduce_sum(weight)]))
        if self.stage == "arcs":
            weight = tf.cast(weights["shape"], tf.float64)
            error = tf.reduce_mean(tf.abs(tf.cast(targets["shape"], tf.float64) - tf.cast(predicted["shape"], tf.float64)), -1)
            self.extra["shape_mae"].assign_add(tf.stack([tf.reduce_sum(error * weight), tf.reduce_sum(weight)]))
            weight = tf.cast(weights["anchor"], tf.float64)
            correct = tf.cast(tf.argmax(predicted["anchor"], -1, output_type=tf.int32) == tf.cast(targets["anchor"], tf.int32), tf.float64)
            self.extra["anchor_accuracy"].assign_add(tf.stack([tf.reduce_sum(correct * weight), tf.reduce_sum(weight)]))
            for key, values in (("anchor_reference_counts", targets["anchor"]),
                                ("anchor_prediction_counts", tf.argmax(predicted["anchor"], -1))):
                counts = tf.one_hot(tf.cast(values, tf.int32), 3, dtype=tf.float64) * weight[..., None]
                self.extra[key].assign_add(tf.reduce_sum(counts, axis=0))

    @staticmethod
    def scores(values):
        tp, fp, fn = map(float, values[:3])
        return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.,
                "recall": tp / (tp + fn) if tp + fn else 0., "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}

    def result(self):
        heads = {}
        for head, values in self.counts.items():
            v = values.numpy()
            heads[head] = {**self.scores(v), "pr_auc": float(self.auc[head].result().numpy()),
                           "positive": float(v[3]), "supervised": float(v[4]), "predicted_positive": float(v[5]),
                           "prevalence": float(v[3] / v[4]) if v[4] else 0.,
                           "positive_case_pass": bool(v[0] > 0)}
        result = {"heads": heads, "pr_auc": float(np.mean([h["pr_auc"] for h in heads.values()])),
                  "pr_auc_method": "streaming_logit_threshold_approximation",
                  "pr_auc_thresholds": len(next(iter(self.auc.values())).thresholds),
                  "positive_case_pass": all(h["positive_case_pass"] for h in heads.values())}
        for key, value in self.extra.items():
            v = value.numpy()
            result[key] = v.tolist() if key.endswith("_counts") else self.scores(v) if len(v) == 3 else (v.tolist() if key == "off_trigger_counts" else
                                                           float(v[0] / v[1]) if v[1] else None)
        return result
