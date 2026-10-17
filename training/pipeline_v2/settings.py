"""Resolve v2 defaults from the project's existing configuration module."""

import math
from tools.config import config

STAGES = ("positions", "arrows", "multiplier", "hazards", "lighting", "arcs")


def _stage(stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown v2 stage: {stage}")


def _positive(value, name, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (not zero and value == 0) or (integer and int(value) != value):
        raise ValueError(f"Invalid {name}: {value}")
    return int(value) if integer else float(value)


def architecture_settings(stage):
    _stage(stage)
    fields = ("history_length", "embedding_dim", "audio_dense_units", "gru_units",
              "shared_dense_units", "activation", "history_dropout")
    if stage == "lighting":
        fields = ("block_length", "conv_filters", "time_lstm_units", "history_lstm_units",
                  "combined_lstm_units", "dense_units", "dropout")
    values = {key: getattr(config, f"{stage}_{key}") for key in fields}
    for key, value in values.items():
        if key == "activation":
            if not isinstance(value, str) or not value:
                raise ValueError("Activation must be named")
        elif "dropout" in key:
            if not 0 <= _positive(value, key, zero=True) < 1:
                raise ValueError("Dropout must be below one")
        else:
            values[key] = _positive(value, key, integer=True)
    values["loss_weights"] = dict(getattr(config, f"{stage}_loss_weights"))
    for key, value in values["loss_weights"].items():
        values["loss_weights"][key] = _positive(value, f"{stage} {key} loss", zero=True)
    if not any(values["loss_weights"].values()):
        raise ValueError("At least one loss head must be active")
    if stage == "arcs":
        values["shape_bounds"] = tuple(config.arcs_shape_bounds)
        lo, hi = values["shape_bounds"]
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo < hi:
            raise ValueError("Arc shape bounds must be finite, nonnegative and ordered")
    if stage == "lighting" and values["block_length"] < 3:
        raise ValueError("Lighting block length must be at least three for the retained convolution")
    return values


def training_settings(stage):
    _stage(stage)
    fields = ("optimizer", "learning_rate", "batch_size", "clipnorm", "early_stopping_patience",
              "early_stopping_min_delta", "monitor", "monitor_mode", "shuffle", "verbose")
    result = {key: getattr(config, f"{stage}_{key}") for key in fields}
    result.update(epochs=getattr(config, f"{stage}_n_epochs"), seed=getattr(config, f"{stage}_random_seed"))
    for key in ("batch_size", "epochs"):
        result[key] = _positive(result[key], key, integer=True)
    for key in ("seed", "early_stopping_patience", "verbose"):
        result[key] = _positive(result[key], key, integer=True, zero=True)
    for key in ("learning_rate", "clipnorm"):
        result[key] = _positive(result[key], key)
    result["early_stopping_min_delta"] = _positive(result["early_stopping_min_delta"], "min_delta", zero=True)
    if result["monitor_mode"] not in ("min", "max", "auto") or not isinstance(result["shuffle"], bool):
        raise ValueError("Invalid training monitor or shuffle setting")
    if result["verbose"] > 2:
        raise ValueError("Verbose must be 0, 1 or 2")
    return result


def schedule_settings(overrides=None):
    from .scheduling import schedule_settings as resolve
    return resolve(overrides)


def preparation_settings():
    fields = ("encoder_grid_seconds", "encoder_batch_size", "max_multiplier_templates", "max_wall_templates",
              "max_light_classes", "max_copies", "max_chain_slices", "max_chain_span_seconds",
              "expansion_time_quantum", "chain_span_quantum", "lighting_match_tolerance",
              "bomb_threshold", "lighting_threshold")
    result = {key: getattr(config, f"pipeline_v2_{key}") for key in fields}
    result.update(history_length=config.positions_history_length, lighting_block_length=config.lighting_block_length,
                  arc_shape_bounds=list(config.arcs_shape_bounds), schedule=schedule_settings())
    histories = {getattr(config, f"{stage}_history_length") for stage in STAGES if stage != "lighting"}
    if len(histories) != 1:
        raise ValueError("V2 stages currently share a history length; set all stage history lengths together")
    for key, value in result.items():
        if key in ("schedule", "arc_shape_bounds"):
            continue
        integer = key.startswith("max_") and key != "max_chain_span_seconds"
        integer = integer or key in ("encoder_batch_size", "history_length", "lighting_block_length")
        result[key] = _positive(value, key, integer=integer)
        if key.endswith("threshold") and result[key] > 1:
            raise ValueError(f"{key} must not exceed one")
    return result
