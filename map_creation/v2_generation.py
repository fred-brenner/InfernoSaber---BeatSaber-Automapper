"""Song/package orchestration for all existing generation entry points.

Imports are inert. Models belong to a process; mutable generation settings belong
to one difficulty, and a prepared recording is reused by that process's runtime.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from tools.config import config, paths


def settings_snapshot():
    """Capture values rather than importing/persisting settings in spawned children."""
    allowed = (str, int, float, bool, type(None), list, tuple, dict, np.ndarray, np.generic)
    return {"config": {k: deepcopy(v) for k, v in vars(config).items()
                       if not k.startswith("_") and isinstance(v, allowed)},
            "paths": {k: v for k, v in vars(paths).items()
                      if not k.startswith("_") and isinstance(v, str)}}


def apply_snapshot(snapshot):
    for module, values in ((config, snapshot["config"]), (paths, snapshot["paths"])):
        for key, value in values.items():
            setattr(module, key, deepcopy(value))


@contextmanager
def scoped_settings(snapshot=None):
    previous = settings_snapshot()
    try:
        if snapshot is not None:
            apply_snapshot(snapshot)
        yield
    finally:
        apply_snapshot(previous)


def configure_inference():
    """Set the CPU worker profile before its first TensorFlow import."""
    threads = config.v2_inference_threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("v2_inference_threads must be a positive integer")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(threads)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _difficulty_settings(difficulty):
    """Apply the legacy difficulty adjustments afresh for this one variant."""
    from tools.utils.numpy_shorts import get_factor_from_max_speed

    config.max_speed = config.max_speed_orig = float(difficulty)
    config.add_beat_intensity = config.add_beat_intensity_orig
    config.silence_threshold = config.silence_threshold_orig
    config.jump_speed_offset = config.jump_speed_offset_orig
    if config.add_silence_flag:
        config.add_beat_intensity += 10
        config.silence_threshold = max(.02, (1 - .5 * difficulty / 40) * config.silence_threshold_orig)
    if config.emphasize_beats_flag:
        config.add_beat_intensity = config.add_beat_intensity_orig - (0 if config.add_silence_flag else 10)
    config.obstacle_time_gap = np.asarray(config.obstacle_time_gap_orig, dtype=float) * (1 - difficulty / 80)
    if min(config.obstacle_time_gap) <= .1:
        config.obstacle_time_gap = np.asarray([.1, .25])
    if config.sporty_obstacles:
        config.jump_speed_offset -= .3
        config.add_beat_intensity -= 5
    config.thresh_beat = config.thresh_beat_orig * get_factor_from_max_speed(difficulty, 1.5, .5)
    config.thresh_onbeat = config.thresh_onbeat_orig * get_factor_from_max_speed(difficulty, 1.3, .1)


def _emit(message, logger):
    if logger is None:
        print(message, flush=True)
    else:
        logger(message)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported report value {type(value).__name__}")


def save_report(report, root=None):
    """Keep diagnostics outside the externally consumed map package."""
    root = Path(paths.temp_path if root is None else root) / "v2-generation"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{os.getpid()}.json"
    destination.write_text(json.dumps(report, indent=2, default=_json, allow_nan=False) + "\n", encoding="utf-8")
    return str(destination)



def save_predictions(state):
    """Save numeric inference evidence outside packages, without pickle payloads."""
    root = Path(paths.temp_path) / "v2-generation"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{os.getpid()}-predictions.npz"
    keys = ("raw_intensity", "intensity", "conditioning_nps", "frame_probabilities", "raw_event_frames",
            "raw_event_times", "raw_counts", "event_frames", "event_times", "counts", "raw_model_notes")
    arrays = {key: np.asarray(state[key]) for key in keys if key in state}
    arrays["final_notes"] = np.asarray([[note[key] for key in ("id", "time", "hand", "x", "y", "d")]
                                       for note in state["notes"]], dtype=np.float64).reshape(-1, 6)
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Prediction evidence must contain finite numeric arrays")
    np.savez_compressed(destination, **arrays)
    return str(destination)


def generate_song(name, internal_difficulty=None, logger=None, *, debug_beats=False):
    """Generate one complete single-difficulty package (plus optional Expert).

    ``internal_difficulty`` preserves the public main() convention: four times
    the visible NPS setting. Input conversion/normalization belongs to the caller.
    """
    from map_creation.v2_runtime import get_runtime
    from map_creation.v2_export import write_package

    name = str(name)
    if name.lower().endswith(".egg"):
        name = name[:-4]
    if not name or Path(name).name != name or "\\" in name:
        raise ValueError("Song name must be a local filename stem")
    difficulty = config.max_speed_orig if internal_difficulty is None else internal_difficulty
    if isinstance(difficulty, bool) or not math.isfinite(float(difficulty)) or float(difficulty) < 0:
        raise ValueError("Difficulty must be finite and nonnegative")
    difficulty = float(difficulty)
    audio = Path(paths.songs_pred) / f"{name}.egg"
    if not audio.is_file():
        raise FileNotFoundError(f"Input song missing: {audio}")
    configure_inference()
    started = time.perf_counter()
    runtime = get_runtime()
    prepared = runtime.prepare_audio(audio)
    variants = []
    selections = [(config.general_diff, difficulty)]
    if config.create_expert_flag and config.general_diff != "Expert":
        selections.append(("Expert", difficulty * config.expert_fact))
    for label, value in selections:
        with scoped_settings():
            _difficulty_settings(value)
            _emit(f"{name}: {label}, requested {value / 4:g} retained notes/s", logger)
            result = runtime.generate(prepared, value / 4, logger=logger)
            if debug_beats:
                return np.asarray(result.state["event_times"], dtype=float)
            variants.append({"difficulty": label, "internal_difficulty": value,
                             "automatic_expert": label == "Expert" and config.general_diff != "Expert",
                             "jump_speed_offset": float(config.jump_speed_offset),
                             "prediction_path": save_predictions(result.state),
                             "state": result.state, "diagnostics": result.diagnostics,
                             "stages": result.stages})
    folder = Path(paths.new_map_path) / f"1234_{difficulty:.1f}_{name}"
    package = write_package(folder, name, float(prepared.song.metadata["bpm"]), variants,
                            audio, version=config.bs_mapping_version)
    report = {"status": "ok", "song": name, "internal_difficulty": difficulty,
              "requested_nps": difficulty / 4, "path": str(folder), "package": package,
              "duration_seconds": float(prepared.song.metadata["duration"]),
              "bpm": float(prepared.song.metadata["bpm"]), "pid": os.getpid(),
              "elapsed_seconds": time.perf_counter() - started,
              "variants": [{k: v for k, v in item.items() if k != "state"} for item in variants]}
    report["report_path"] = save_report(report)
    _emit(f"Finished {name} at {difficulty / 4:g} NPS in {report['elapsed_seconds']:.1f}s", logger)
    return report
