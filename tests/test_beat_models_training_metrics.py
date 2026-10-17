"""Epoch display/report diagnostics, verified without fit or optimizer updates."""

import numpy as np
import pytest

from training.beat_models.contracts import Song
from training.beat_models.training_metrics import (add_intensity_metrics,
                                                   intensity_metrics_callback, intensity_reference)


def reference_songs():
    return [Song("partial", {}, {"intensity": np.array([.5, .5]), "bin_durations": np.array([1, .25])}),
            Song("short", {}, {"intensity": np.array([1.]), "bin_durations": np.array([.1])}),
            Song("empty", {"intensity_target_valid": False},
                 {"intensity": np.zeros(2), "bin_durations": np.ones(2)})]


def test_reference_is_duration_aware_per_song_and_excludes_empty_targets():
    reference = intensity_reference(reference_songs())
    assert reference["valid_songs"] == 2
    assert reference["target_entropy"] == pytest.approx(np.log(2) / 2)
    assert reference["uniform_rate_loss"] == pytest.approx(-.25 * (np.log(.8) + np.log(.2)))
    assert reference["uniform_rate_kl"] == pytest.approx(reference["uniform_rate_loss"] - np.log(2) / 2)
    with pytest.raises(ValueError, match="nonempty"):
        intensity_reference(reference_songs()[-1:])


def test_epoch_metrics_compare_avoidable_loss_and_support_negative_improvement():
    reference = intensity_reference(reference_songs())
    references = {"train": reference, "validation": reference}
    logs = {"loss": reference["target_entropy"], "val_loss": reference["uniform_rate_loss"]}
    add_intensity_metrics(logs, references)
    assert logs["kl_divergence"] == 0
    assert logs["uniform_rate_improvement_pct"] == pytest.approx(100)
    assert logs["val_kl_divergence"] == pytest.approx(reference["uniform_rate_kl"])
    assert logs["val_uniform_rate_improvement_pct"] == pytest.approx(0)
    logs = {"loss": reference["target_entropy"] + 2 * reference["uniform_rate_kl"]}
    add_intensity_metrics(logs, references)
    assert logs["uniform_rate_improvement_pct"] == pytest.approx(-100)
    assert logs["uniform_rate_loss"] == reference["uniform_rate_loss"]


def test_zero_reference_error_omits_undefined_percentage_and_nan_stays_nan():
    reference = intensity_reference(reference_songs()[1:2])
    logs = {"loss": 0}
    add_intensity_metrics(logs, {"train": reference})
    assert logs["kl_divergence"] == 0
    assert "uniform_rate_improvement_pct" not in logs
    logs = {"loss": float("nan")}
    add_intensity_metrics(logs, {"train": reference})
    assert np.isnan(logs["kl_divergence"])


def test_callback_updates_console_and_history_without_training(capsys, monkeypatch):
    from training.beat_models import models

    def forbidden(*args, **kwargs):
        pytest.fail("Metric display attempted to train or construct an optimizer")

    monkeypatch.setattr(models.keras.Model, "fit", forbidden)
    monkeypatch.setattr(models.keras.optimizers, "Adam", forbidden)
    model = models.build_intensity_model()
    before = [value.copy() for value in model.get_weights()]
    reference = intensity_reference(reference_songs())
    callbacks = models.keras.callbacks.CallbackList(
        [intensity_metrics_callback({"train": reference, "validation": reference})],
        add_history=True, add_progbar=True, model=model, verbose=2, epochs=1, steps=1)
    # Simulate callback events only: no batches, model training, or optimizer calls.
    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)
    callbacks.on_train_batch_end(0, {"loss": reference["target_entropy"]})
    callbacks.on_epoch_end(0, {"loss": reference["target_entropy"], "val_loss": reference["uniform_rate_loss"]})
    callbacks.on_train_end()
    output = capsys.readouterr().out
    for name in ("kl_divergence", "uniform_rate_loss", "uniform_rate_improvement_pct",
                 "val_kl_divergence", "val_uniform_rate_loss", "val_uniform_rate_improvement_pct"):
        assert name in output
        assert name in model.history.history
    assert model.history.history["uniform_rate_improvement_pct"] == pytest.approx([100])
    assert model.history.history["val_uniform_rate_improvement_pct"] == pytest.approx([0])
    for old, current in zip(before, model.get_weights()):
        np.testing.assert_array_equal(old, current)
