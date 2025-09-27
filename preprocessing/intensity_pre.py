"""Data preparation helpers for beat intensity model training."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from preprocessing.beat_data_helper import load_raw_beat_data, sort_beats_by_time
from preprocessing.music_processing import run_music_preprocessing
from tools.config import get_config
from tools.utils.numpy_shorts import reduce_number_of_songs
from training.helpers import filter_by_bps

config = get_config()


def _remove_indices(values: Sequence[float], indices: Sequence[int]) -> np.ndarray:
    """Return a copy of *values* with the provided *indices* removed."""
    if not indices:
        return np.asarray(values, dtype=np.float32)
    mask = np.ones(len(values), dtype=bool)
    mask[np.asarray(indices, dtype=int)] = False
    return np.asarray(values, dtype=np.float32)[mask]


def _compute_song_average_bps(note_times: np.ndarray) -> float:
    """Compute the average beats-per-second of a song."""
    if note_times.size <= 1:
        return 0.0
    duration = float(note_times[-1] - note_times[0])
    if duration <= 0:
        return 0.0
    return float(note_times.size / duration)


def _compute_window_intensity(note_times: np.ndarray, window_seconds: float) -> np.ndarray:
    """Compute local beat intensity ratios for each note time.

    The returned array contains, for every note time, the ratio between the
    average beats-per-second inside a symmetrical ``window_seconds`` segment
    around the note and the global song average beats-per-second.
    """
    if note_times.size == 0:
        return np.zeros((0, 1), dtype=np.float32)

    avg_bps = _compute_song_average_bps(note_times)
    if avg_bps <= 0:
        return np.zeros((note_times.size, 1), dtype=np.float32)

    half_window = window_seconds / 2.0
    intensities: List[float] = []

    for time_point in note_times:
        start = time_point - half_window
        end = time_point + half_window
        # Count beats within the window using vectorised comparisons.
        beats_in_window = np.count_nonzero((note_times >= start) & (note_times < end))
        window_bps = beats_in_window / window_seconds
        intensities.append(window_bps / avg_bps)

    return np.asarray(intensities, dtype=np.float32).reshape(-1, 1)


def load_intensity_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load spectrogram windows and their beat intensity ratios.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(song_windows, intensity_targets)`` where ``song_windows`` has shape
        ``(n_samples, height, width, channels)`` and ``intensity_targets`` has
        shape ``(n_samples, 1)``.
    """
    # Select maps according to the configured BPM limits.
    name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
    name_ar = reduce_number_of_songs(name_ar, hard_limit=config.intensity_song_limit)

    # Load beat timing information for each song.
    map_dict_notes, _, _ = load_raw_beat_data(name_ar)
    _, time_ar = sort_beats_by_time(map_dict_notes)

    # Prepare spectrogram inputs aligned with each beat time.
    song_ar, rm_index_ar = run_music_preprocessing(
        name_ar,
        time_ar,
        save_file=False,
        song_combined=False,
        channels_last=True,
    )

    windows_list: List[np.ndarray] = []
    intensity_list: List[np.ndarray] = []

    for song_windows, note_times, rm_idx in zip(song_ar, time_ar, rm_index_ar):
        if len(song_windows) == 0 or len(note_times) == 0:
            continue

        cleaned_times = _remove_indices(note_times, rm_idx)
        if cleaned_times.size == 0:
            continue

        # Align windows with cleaned note times.
        if len(song_windows) != cleaned_times.size:
            # Skip inconsistent samples to avoid shape mismatches.
            continue

        intensities = _compute_window_intensity(cleaned_times, config.window)
        if intensities.size == 0:
            continue

        windows_list.append(song_windows.astype(np.float32))
        intensity_list.append(intensities)

    if not windows_list:
        raise RuntimeError(
            "No valid training samples were generated for the intensity dataset."
        )

    song_windows = np.concatenate(windows_list, axis=0).astype(np.float32)
    intensity_targets = np.concatenate(intensity_list, axis=0).astype(np.float32)

    return song_windows, intensity_targets


__all__ = ["load_intensity_dataset"]
