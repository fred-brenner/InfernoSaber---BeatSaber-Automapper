"""Song/package orchestration for all existing generation entry points.

Imports are inert. Models belong to a process; mutable generation settings belong
to one difficulty, and a prepared recording is reused by that process's runtime.
"""

from contextlib import contextmanager, nullcontext
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
    existing_names = {"config": set(vars(config)), "paths": set(vars(paths))}
    try:
        if snapshot is not None:
            apply_snapshot(snapshot)
        yield
    finally:
        current = settings_snapshot()
        for name, module in (("config", config), ("paths", paths)):
            for key in current[name].keys() - existing_names[name]:
                delattr(module, key)
        apply_snapshot(previous)


def effective_inference_threads(workers=None):
    """Resolve the per-process TensorFlow thread count for v2 inference.

    ``config.v2_inference_threads`` of 0 (or None) auto-scales as
    ``max(1, min(2, cores // workers))``: batch-1 ~1 MB models gain nothing
    from intraop parallelism, so total TF threads stay at or below the logical
    core count, while up to 2 threads (the previous fixed default) remain
    available when few workers leave idle cores. A positive integer pins the
    count. ``workers`` defaults to ``config.num_workers``.
    """
    value = getattr(config, "v2_inference_threads", 0)
    if value is None:
        value = 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("v2_inference_threads must be 0 (auto) or a positive integer")
    if value:
        return value
    if workers is None:
        workers = getattr(config, "num_workers", 1)
    try:
        valid = (not isinstance(workers, bool) and float(workers).is_integer()
                 and workers >= 1)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Worker count must be a positive integer")
    return max(1, min(2, (os.cpu_count() or 1) // int(workers)))


def configure_inference(workers=None):
    """Set the CPU inference profile before this process's first TensorFlow import."""
    threads = effective_inference_threads(workers)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(threads)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return threads


# MKL and libgomp read their env limit lazily at first parallel use, so a
# spawned worker can still cap them in ``_worker_init`` even though numpy is
# already imported; PyPI-wheel OpenBLAS latches its count at library load,
# which can predate the cap. ``setdefault`` keeps explicit user exports
# authoritative.
_LIBRARY_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def cap_library_threads(threads=None):
    """Cap native library thread pools inside one inference worker process."""
    threads = str(effective_inference_threads() if threads is None else threads)
    for name in _LIBRARY_THREAD_VARS:
        os.environ.setdefault(name, threads)
    # Park idle OpenMP threads instead of letting them spin under many workers.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def _difficulty_scope(difficulty):
    """Reusable scope replaying only this difficulty's config delta.

    The lockstep driver enters a variant's scope per pipeline piece, so a
    full ``scoped_settings`` snapshot per entry would cost more than the
    predicts it protects. This factory resolves ``_difficulty_settings``
    once, then each entry applies just the keys it changed (restoring any
    added or reassigned attribute on exit, like the snapshot scope does).
    In-place mutation of a non-delta value inside a segment is not undone —
    generation code reassigns config keys wholesale instead.

    The delta freezes at factory creation: keys it captured overwrite any
    later ambient edits on every entry, so build scopes after the run's
    bootstrap settings settle rather than reusing them across live changes.
    """
    modules = {"config": config, "paths": paths}
    before = {name: dict(vars(module)) for name, module in modules.items()}
    with scoped_settings():
        _difficulty_settings(difficulty)
        delta = {name: {key: value for key, value in vars(module).items()
                        if key not in before[name] or before[name][key] is not value}
                 for name, module in modules.items()}

    @contextmanager
    def scope():
        saved = {module: dict(vars(module)) for module in modules.values()}
        for name, changes in delta.items():
            for key, value in changes.items():
                setattr(modules[name], key, deepcopy(value))
        try:
            yield
        finally:
            for module, previous in saved.items():
                current = vars(module)
                for key in set(current) - set(previous):
                    delattr(module, key)
                for key, value in previous.items():
                    if key not in current or current[key] is not value:
                        setattr(module, key, value)

    return scope


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


def _song_stem(name):
    """Validate the public song name convention and return its filename stem."""
    name = str(name)
    if name.lower().endswith(".egg"):
        name = name[:-4]
    if not name or Path(name).name != name or "\\" in name:
        raise ValueError("Song name must be a local filename stem")
    return name


def _song_audio(name):
    """Resolve the recorded input audio for a validated song stem."""
    audio = Path(paths.songs_pred) / f"{name}.egg"
    if not audio.is_file():
        raise FileNotFoundError(f"Input song missing: {audio}")
    return audio


def _valid_difficulty(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("Difficulty must be finite and nonnegative")
    return float(value)


def _variant_entry(label, value, scope, result):
    """Snapshot one variant's package payload under its own difficulty scope."""
    with scope():
        return {"difficulty": label, "internal_difficulty": value,
                "automatic_expert": label == "Expert" and config.general_diff != "Expert",
                "jump_speed_offset": float(config.jump_speed_offset),
                "prediction_path": save_predictions(result.state),
                "state": result.state, "diagnostics": result.diagnostics,
                "stages": result.stages}


def _completion_record(name, difficulty, variants, package, folder, prepared, elapsed):
    """The per-difficulty completion record process_song/_combine_maps consume."""
    return {"status": "ok", "song": name, "internal_difficulty": difficulty,
            "requested_nps": difficulty / 4, "path": str(folder), "package": package,
            "duration_seconds": float(prepared.song.metadata["duration"]),
            "bpm": float(prepared.song.metadata["bpm"]), "pid": os.getpid(),
            "elapsed_seconds": elapsed,
            "variants": [{k: v for k, v in item.items() if k != "state"} for item in variants]}


def generate_song(name, internal_difficulty=None, logger=None, *, debug_beats=False):
    """Generate one complete single-difficulty package (plus optional Expert).

    ``internal_difficulty`` preserves the public main() convention: four times
    the visible NPS setting. Input conversion/normalization belongs to the caller.
    """
    from map_creation.v2_runtime import get_runtime
    from map_creation.v2_export import write_package

    name = _song_stem(name)
    difficulty = config.max_speed_orig if internal_difficulty is None else internal_difficulty
    difficulty = _valid_difficulty(difficulty)
    audio = _song_audio(name)
    configure_inference()
    started = time.perf_counter()
    runtime = get_runtime()
    prepared = runtime.prepare_audio(audio)
    variants = []
    selections = [(config.general_diff, difficulty)]
    if config.create_expert_flag and config.general_diff != "Expert":
        selections.append(("Expert", difficulty * config.expert_fact))
    if len(selections) > 1:
        # Batched lockstep decode: each variant still runs its config-dependent
        # pieces under its own replayed difficulty scope.
        scopes = [_difficulty_scope(value) for _, value in selections]
        for label, value in selections:
            _emit(f"{name}: {label}, requested {value / 4:g} retained notes/s", logger)
        results = runtime.generate_batch(prepared, [value / 4 for _, value in selections],
                                         logger=logger, scopes=scopes)
        if debug_beats:
            return np.asarray(results[0].state["event_times"], dtype=float)
        for (label, value), scope, result in zip(selections, scopes, results):
            variants.append(_variant_entry(label, value, scope, result))
    else:
        for label, value in selections:
            with scoped_settings():
                _difficulty_settings(value)
                _emit(f"{name}: {label}, requested {value / 4:g} retained notes/s", logger)
                result = runtime.generate(prepared, value / 4, logger=logger)
                if debug_beats:
                    return np.asarray(result.state["event_times"], dtype=float)
                variants.append(_variant_entry(label, value, nullcontext, result))
    folder = Path(paths.new_map_path) / f"1234_{difficulty:.1f}_{name}"
    package = write_package(folder, name, float(prepared.song.metadata["bpm"]), variants,
                            audio, version=config.bs_mapping_version)
    report = _completion_record(name, difficulty, variants, package, folder, prepared,
                                time.perf_counter() - started)
    report["report_path"] = save_report(report)
    _emit(f"Finished {name} at {difficulty / 4:g} NPS in {report['elapsed_seconds']:.1f}s", logger)
    return report


def generate_song_group(name, difficulties, logger=None, *, debug_beats=False):
    """Generate one package per requested difficulty of one song in one pass.

    ``difficulties`` keeps the public four-times-NPS units. All variants share
    a single ``prepare_audio`` and a single lockstep ``generate_batch``, each
    running its config-dependent pieces under its own replayed difficulty
    scope; every difficulty then gets its own ``1234_<d>_<name>`` package and
    completion record, so a K-difficulty call returns a list of records in
    exactly the shape ``generate_song`` reports. A single-entry list delegates
    to ``generate_song`` so ``create_expert_flag`` twin semantics stay intact;
    true group runs create no Expert twin (matching ``main_multi``'s
    multi-difficulty semantics). A failed per-difficulty package becomes a
    failed record — the shared decode already spent the work, so siblings
    keep their results instead of forcing a whole-song retry.
    """
    from map_creation.v2_runtime import get_runtime
    from map_creation.v2_export import write_package

    name = _song_stem(name)
    difficulties = [_valid_difficulty(value) for value in difficulties]
    if not difficulties:
        raise ValueError("At least one difficulty is required")
    if len(difficulties) == 1:
        result = generate_song(name, difficulties[0], logger=logger, debug_beats=debug_beats)
        return result if debug_beats else [result]
    if len({f"{value:.1f}" for value in difficulties}) != len(difficulties):
        raise ValueError("Difficulties must produce distinct output filenames")
    audio = _song_audio(name)
    configure_inference()
    started = time.perf_counter()
    runtime = get_runtime()
    prepared = runtime.prepare_audio(audio)
    selections = [(config.general_diff, value) for value in difficulties]
    scopes = [_difficulty_scope(value) for _, value in selections]
    for label, value in selections:
        _emit(f"{name}: {label}, requested {value / 4:g} retained notes/s", logger)
    results = runtime.generate_batch(prepared, [value / 4 for _, value in selections],
                                     logger=logger, scopes=scopes)
    if debug_beats:
        return np.asarray(results[0].state["event_times"], dtype=float)
    records = []
    for (label, value), scope, result in zip(selections, scopes, results):
        try:
            variants = [_variant_entry(label, value, scope, result)]
            folder = Path(paths.new_map_path) / f"1234_{value:.1f}_{name}"
            package = write_package(folder, name, float(prepared.song.metadata["bpm"]),
                                    variants, audio, version=config.bs_mapping_version)
        except Exception as error:
            records.append({"status": "failed", "song": name,
                            "internal_difficulty": value,
                            "error": f"{type(error).__name__}: {error}"})
            continue
        records.append(_completion_record(name, value, variants, package, folder, prepared, 0.))
    # The decode produced all difficulties in one shared call: attribute each
    # record an even share of the group wall time — the same convention
    # generate_batch uses for generation_seconds — so summed record elapsed
    # stays at or below the real elapsed.
    share = (time.perf_counter() - started) / len(selections)
    for record in records:
        if record["status"] != "ok":
            continue
        record["elapsed_seconds"] = share
        record["report_path"] = save_report(record)
        _emit(f"Finished {record['song']} at {record['internal_difficulty'] / 4:g} NPS "
              f"in {share:.1f}s (batched share)", logger)
    return records
