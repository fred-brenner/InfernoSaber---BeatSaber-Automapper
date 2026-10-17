"""Intensity training diagnostics; importing this module does not load TensorFlow."""

import numpy as np


INTENSITY_METRIC_DEFINITIONS = {
    "kl_divergence": "loss minus mean target entropy; lower is better",
    "uniform_rate_loss": "mean CE against bin_duration / sum(bin_durations)",
    "uniform_rate_improvement_pct": "100 * (uniform_rate_KL - model_KL) / uniform_rate_KL; higher is better; omitted if reference KL <= 1e-7",
}


def intensity_reference(songs, epsilon=1e-7):
    """Average over nonempty songs, giving partial bins their actual duration."""
    entropies, losses = [], []
    for song in songs:
        target = np.asarray(song.arrays["intensity"], dtype=np.float64)
        if not song.metadata.get("intensity_target_valid", bool(target.sum())):
            continue
        durations = np.asarray(song.arrays["bin_durations"], dtype=np.float64)
        uniform = durations / durations.sum()
        entropies.append(-np.sum(target * np.log(np.clip(target, epsilon, 1))))
        losses.append(-np.sum(target * np.log(np.clip(uniform, epsilon, 1))))
    if not losses:
        raise ValueError("Intensity reference requires nonempty target songs")
    entropy, loss = float(np.mean(entropies)), float(np.mean(losses))
    return {"valid_songs": len(losses), "target_entropy": entropy, "uniform_rate_loss": loss,
            "uniform_rate_kl": max(0.0, loss - entropy)}


def add_intensity_metrics(logs, references):
    """Derive epoch metrics from the already computed loss, without extra passes."""
    for split, prefix in (("train", ""), ("validation", "val_")):
        if f"{prefix}loss" not in logs:
            continue
        reference = references[split]
        kl = float(logs[f"{prefix}loss"]) - reference["target_entropy"]
        kl = max(0.0, kl) if np.isfinite(kl) else kl
        logs[f"{prefix}kl_divergence"] = kl
        logs[f"{prefix}uniform_rate_loss"] = reference["uniform_rate_loss"]
        baseline = reference["uniform_rate_kl"]
        if baseline > 1e-7:
            logs[f"{prefix}uniform_rate_improvement_pct"] = 100 * (baseline - kl) / baseline


def intensity_metrics_callback(references):
    """Insert metrics before Keras' automatic History and progress callbacks."""
    from .models import keras

    class IntensityMetrics(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                add_intensity_metrics(logs, references)

    return IntensityMetrics()
