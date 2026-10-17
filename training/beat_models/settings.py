"""Resolve the new models' per-stage defaults from the existing project config."""

import math
from numbers import Integral, Real

from tools.config import config


TRAINING_FIELDS = ("optimizer", "learning_rate", "batch_size", "clipnorm",
                   "early_stopping_patience", "early_stopping_min_delta", "monitor", "monitor_mode",
                   "shuffle", "verbose")
TEMPORAL_FIELDS = ("channels", "kernel_size", "convolutions_per_block", "dilations",
                   "dropout", "activation")
ARCHITECTURE_FIELDS = {
    "intensity": TEMPORAL_FIELDS,
    "timing": TEMPORAL_FIELDS + ("chunk_frames", "count_loss_weight", "count_huber_delta", "count_rate_scale"),
    "notes": ("patch_frames", "history_length", "cnn_filters", "cnn_kernel_size", "pool_sizes",
              "audio_dense_units", "embedding_dim", "gru_units", "shared_dense_units",
              "history_dropout", "activation", "red_loss_weight", "blue_loss_weight"),
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, minimum=0, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (value <= minimum if positive else value < minimum)):
        comparison = ">" if positive else ">="
        raise ValueError(f"{name} must be finite and {comparison} {minimum}")
    return float(value)


def _read(stage, fields):
    if stage not in ARCHITECTURE_FIELDS:
        raise ValueError("Unknown model stage")
    return {name: getattr(config, f"{stage}_{name}") for name in fields}


def training_settings(stage):
    """Read live defaults; callers apply explicit CLI overrides afterwards."""
    values = _read(stage, TRAINING_FIELDS)
    values["epochs"] = getattr(config, f"{stage}_n_epochs")
    values["seed"] = getattr(config, f"{stage}_random_seed")
    for name in ("epochs", "batch_size"):
        values[name] = _integer(values[name], f"{stage}_{name}")
    for name in ("seed", "early_stopping_patience", "verbose"):
        values[name] = _integer(values[name], f"{stage}_{name}", minimum=0)
    values["learning_rate"] = _number(values["learning_rate"], f"{stage}_learning_rate", positive=True)
    if values["clipnorm"] is not None:
        values["clipnorm"] = _number(values["clipnorm"], f"{stage}_clipnorm", positive=True)
    values["early_stopping_min_delta"] = _number(values["early_stopping_min_delta"], f"{stage}_early_stopping_min_delta")
    if not isinstance(values["shuffle"], bool) or values["verbose"] > 2:
        raise ValueError(f"{stage} shuffle must be boolean and verbose must be 0, 1, or 2")
    for name in ("optimizer", "monitor"):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{stage}_{name} must be a nonempty name")
    if values["monitor_mode"] not in ("min", "max", "auto"):
        raise ValueError(f"{stage}_monitor_mode must be min, max, or auto")
    values["conditioning"] = getattr(config, f"{stage}_conditioning") if stage != "intensity" else "reference"
    if values["conditioning"] not in ("reference", "noisy-reference", "predicted"):
        raise ValueError(f"Invalid {stage} intensity conditioning")
    values["intensity_noise_std"] = (_number(getattr(config, f"{stage}_intensity_noise_std"),
                                             f"{stage}_intensity_noise_std") if stage != "intensity" else 0.0)
    return values


def architecture_settings(stage):
    """Return validated widths, context sizes, regularization and loss weights."""
    values = _read(stage, ARCHITECTURE_FIELDS.get(stage, ()))
    if not isinstance(values["activation"], str) or not values["activation"]:
        raise ValueError(f"{stage}_activation must be a nonempty name")
    if stage in ("intensity", "timing"):
        for name in ("channels", "kernel_size", "convolutions_per_block"):
            values[name] = _integer(values[name], f"{stage}_{name}")
        if values["kernel_size"] % 2 != 1:
            raise ValueError(f"{stage}_kernel_size must be odd for centered context")
        values["dilations"] = tuple(_integer(item, f"{stage}_dilations") for item in values["dilations"])
        if not values["dilations"] or len(set(values["dilations"])) != len(values["dilations"]):
            raise ValueError(f"{stage}_dilations must be nonempty and distinct")
        rate = "dropout"
    else:
        for name in ("patch_frames", "history_length", "cnn_kernel_size", "audio_dense_units",
                     "embedding_dim", "gru_units", "shared_dense_units"):
            values[name] = _integer(values[name], f"notes_{name}")
        if values["patch_frames"] % 2:
            raise ValueError("notes_patch_frames must be even for a centered patch")
        values["cnn_filters"] = tuple(_integer(item, "notes_cnn_filters") for item in values["cnn_filters"])
        values["pool_sizes"] = tuple(tuple(_integer(item, "notes_pool_sizes") for item in pair)
                                     for pair in values["pool_sizes"])
        if (not values["cnn_filters"] or len(values["cnn_filters"]) != len(values["pool_sizes"])
                or any(len(pair) != 2 for pair in values["pool_sizes"])):
            raise ValueError("notes_cnn_filters and two-dimensional notes_pool_sizes must match")
        for name in ("red_loss_weight", "blue_loss_weight"):
            values[name] = _number(values[name], f"notes_{name}")
        if values["red_loss_weight"] + values["blue_loss_weight"] <= 0:
            raise ValueError("Notes loss weights cannot both be zero")
        rate = "history_dropout"
    values[rate] = _number(values[rate], f"{stage}_{rate}")
    if values[rate] >= 1:
        raise ValueError(f"{stage}_{rate} must be below one")
    if stage == "timing":
        values["chunk_frames"] = _integer(values["chunk_frames"], "timing_chunk_frames")
        values["count_loss_weight"] = _number(values["count_loss_weight"], "timing_count_loss_weight")
        for name in ("count_huber_delta", "count_rate_scale"):
            values[name] = _number(values[name], f"timing_{name}", positive=True)
    return values


def feature_settings():
    """Keep shared new-model audio settings distinct from legacy preprocessing."""
    names = {"sample_rate": "samplerate_music", "hop": "hop_size", "fft": "fft_size", "mels": "mel_bins"}
    values = {name: _integer(getattr(config, f"beat_models_{suffix}"), f"beat_models_{suffix}")
              for name, suffix in names.items()}
    if values["fft"] % 2:
        raise ValueError("beat_models_fft_size must be even for centered frames")
    return values


def split_fractions():
    """Recording groups share one train/validation/test split across all models."""
    validation = _number(config.beat_models_validation_fraction, "beat_models_validation_fraction")
    test = _number(config.beat_models_test_fraction, "beat_models_test_fraction")
    if validation + test >= 1:
        raise ValueError("Validation and test fractions must leave a nonzero training fraction")
    return 1 - validation - test, validation, test
