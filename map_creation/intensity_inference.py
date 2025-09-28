"""Inference helpers for the beat intensity model."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from keras.models import load_model

from tools.config import get_config
from tools.config.mapper_selection import get_full_model_path

config = get_config()


@lru_cache(maxsize=1)
def _load_intensity_model():
    """Load and cache the trained intensity model."""

    try:
        model_path = get_full_model_path(config.intensity_model_version)
    except FileNotFoundError:
        if config.verbose_level > 0:
            print(
                "Warning: Could not locate the beat intensity model. "
                "Dynamic speed adaptation will be skipped."
            )
        return None

    return load_model(model_path)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple moving average with edge padding."""

    if window <= 1:
        return values

    window = int(window)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def predict_intensity_ratios(song_windows: np.ndarray) -> Optional[np.ndarray]:
    """Predict beat intensity ratios for the provided spectrogram windows."""

    model = _load_intensity_model()
    if model is None:
        return None

    if song_windows.size == 0:
        return np.zeros((0,), dtype=np.float32)

    predictions = model.predict(song_windows, verbose=0).reshape(-1)
    return predictions.astype(np.float32)


def compute_dynamic_speed_factors(song_windows: np.ndarray) -> Optional[np.ndarray]:
    """Return per-sample speed factors inferred from the intensity model."""

    if not config.use_intensity_model:
        return None

    ratios = predict_intensity_ratios(song_windows)
    if ratios is None:
        return None

    ratios = np.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
    scaled = 1.0 + (ratios - 1.0) * float(config.intensity_response)
    smoothed = _smooth_series(scaled.astype(np.float32), int(config.intensity_smoothing))
    clipped = np.clip(smoothed, config.intensity_speed_min, config.intensity_speed_max)
    return clipped.astype(np.float32)


__all__ = [
    "compute_dynamic_speed_factors",
    "predict_intensity_ratios",
]

