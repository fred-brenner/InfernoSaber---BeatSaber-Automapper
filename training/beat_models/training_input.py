"""Bounded beat-pair batches and Keras progress callbacks; never starts training."""

from numbers import Integral
from time import perf_counter

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class BatchSource:
    """Keep song arrays in RAM and prepare profiles outside dataset workers.

    ``epoch`` is the one-based epoch already prepared. Reference and predicted
    profiles are fixed; noisy references are regenerated once per epoch.
    """

    def __init__(self, stage, songs, references, batch_size, mode="reference",
                 shuffle=False, seed=3, noise_std=0.1, intensity_model=None):
        if stage not in ("intensity", "timing"):
            raise ValueError("BatchSource supports intensity and timing")
        if mode not in ("reference", "noisy-reference", "predicted"):
            raise ValueError("Unknown intensity conditioning mode")
        if not isinstance(shuffle, bool):
            raise ValueError("Shuffle must be boolean")
        self.stage = stage
        self.songs, self.references = tuple(songs), tuple(references)
        if not self.songs or not self.references:
            raise ValueError("BatchSource requires nonempty songs and references")
        self.batch_size = _integer(batch_size, "Batch size", 1)
        self.mode, self.shuffle = mode, shuffle
        self.noise_std, self.intensity_model = noise_std, intensity_model
        self.rng = np.random.default_rng(seed)
        self.examples = len(self.references)
        self.order = np.arange(self.examples)
        self.profiles = None
        self.epoch = 0
        self.prepare_epoch()

    def __len__(self):
        return (self.examples + self.batch_size - 1) // self.batch_size

    def prepare_epoch(self):
        """Advance shuffle and noise together, synchronously on the caller."""
        if self.shuffle:
            self.rng.shuffle(self.order)
        if self.stage == "intensity":
            self.profiles = {}
        elif self.profiles is None or self.mode == "noisy-reference":
            from .runner import _profiles

            self.profiles = _profiles(self.songs, self.mode, self.intensity_model,
                                      self.rng, noise_std=self.noise_std)
        self.epoch += 1

    def iter_batches(self):
        """Snapshot this epoch before yielding, preserving partial final batches."""
        order, profiles = self.order.copy(), self.profiles

        def generate():
            from .runner import _batch

            for offset in range(0, len(order), self.batch_size):
                selected = order[offset:offset + self.batch_size]
                inputs, target, mask = _batch(
                    self.stage, [self.references[index] for index in selected], profiles)
                if self.stage == "timing":
                    target = np.stack((target, mask.astype(np.int32)), axis=-1)
                yield inputs, target

        return generate()

    def dataset(self, prefetch_batches=2):
        """Build a finite dataset with an explicit, bounded batch lookahead."""
        prefetch_batches = _integer(prefetch_batches, "Prefetch batch count")
        from . import models
        from .contracts import CORE, FEATURES, TIMING_WINDOW

        tf = models.tf
        if self.stage == "intensity":
            signature = ({"audio": tf.TensorSpec((None, None, FEATURES), tf.float32),
                          "durations": tf.TensorSpec((None, None), tf.float32),
                          "mask": tf.TensorSpec((None, None), tf.bool)},
                         tf.TensorSpec((None, None), tf.float32))
        else:
            signature = ({"audio": tf.TensorSpec((None, TIMING_WINDOW, FEATURES), tf.float32),
                          "conditions": tf.TensorSpec((None, TIMING_WINDOW, 4), tf.float32),
                          "mask": tf.TensorSpec((None, TIMING_WINDOW), tf.bool)},
                         tf.TensorSpec((None, CORE, 2), tf.int32))
        dataset = tf.data.Dataset.from_generator(self.iter_batches, output_signature=signature)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(self)))
        options = tf.data.Options()
        options.deterministic = True
        options.autotune.enabled = False
        options.experimental_optimization.inject_prefetch = False
        dataset = dataset.with_options(options)
        return dataset.prefetch(prefetch_batches) if prefetch_batches else dataset


def training_callbacks(train_source, validation_source, epochs, report,
                       log_every_batches=100, verbose=2):
    """Keep epoch preparation, timing and occasional logging in one callback.

    Place this after early stopping and checkpoint callbacks. It prepares the
    next epoch only when training will continue, and leaves Keras' loss and
    metric aggregation intact.
    """
    epochs = _integer(epochs, "Epoch count", 1)
    log_every_batches = _integer(log_every_batches, "Batch log interval")
    from .models import keras

    class TrainingProgress(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            # Keras can pass device tensors directly to batch hooks. Reading
            # logs only at progress boundaries avoids synchronizing every batch.
            self._supports_tf_logs = True
            self._epoch_start = None
            self._validation_start = None
            self._train_seconds = None
            self._validation_seconds = 0.0
            self._reported = {"train": 0, "validation": 0}

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch = epoch + 1
            self._epoch_start = perf_counter()
            self._train_seconds = None
            self._validation_seconds = 0.0
            self._reported = {"train": 0, "validation": 0}

        def on_test_begin(self, logs=None):
            self._validation_start = perf_counter()
            self._train_seconds = self._validation_start - self._epoch_start

        def on_test_end(self, logs=None):
            self._validation_seconds = perf_counter() - self._validation_start

        def _progress(self, split, batch, source, seconds, logs):
            completed = min(batch + 1, len(source))
            if (not verbose or not log_every_batches
                    or completed // log_every_batches <= self._reported[split]):
                return
            self._reported[split] = completed // log_every_batches
            examples = min(completed * source.batch_size, source.examples)
            loss = (logs or {}).get("loss")
            if hasattr(loss, "numpy"):
                loss = loss.numpy()
            loss_text = "" if loss is None else f", loss={float(loss):.6f}"
            print(f"{train_source.stage} epoch {self._epoch} {split}: "
                  f"batch {completed}/{len(source)}{loss_text}, {seconds:.1f}s, "
                  f"{examples / max(seconds, 1e-9):.1f} examples/s", flush=True)

        def on_train_batch_end(self, batch, logs=None):
            self._progress("train", batch, train_source,
                           perf_counter() - self._epoch_start, logs)

        def on_test_batch_end(self, batch, logs=None):
            self._progress("validation", batch, validation_source,
                           perf_counter() - self._validation_start, logs)

        def on_epoch_end(self, epoch, logs=None):
            seconds = perf_counter() - self._epoch_start
            train_seconds = seconds if self._train_seconds is None else self._train_seconds

            def performance(source, elapsed):
                return {"seconds": elapsed, "batches": len(source), "examples": source.examples,
                        "examples_per_second": source.examples / max(elapsed, 1e-9)}

            record = {"epoch": epoch + 1, "seconds": seconds,
                      "train": performance(train_source, train_seconds),
                      "validation": performance(validation_source, self._validation_seconds)}
            report.setdefault("epoch_performance", []).append(record)
            if verbose:
                print(f"{train_source.stage} epoch {epoch + 1}: {seconds:.1f}s, "
                      f"train {record['train']['examples_per_second']:.1f} examples/s, "
                      f"validation {record['validation']['examples_per_second']:.1f} examples/s",
                      flush=True)
            if epoch + 1 < epochs and not self.model.stop_training:
                train_source.prepare_epoch()
                validation_source.prepare_epoch()

    return TrainingProgress()
