"""Benchmark processed beat timings against human-authored map packages.

Run from the repository root with ``python -m evaluation.evaluate_beat_algorithms
--help``. Importing this module does not load models or change configuration.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.human_maps import load_reference


def score_beats(beat_pred, beat_real, tolerance=0.02):
    """Match sorted timestamps one-to-one, including the tolerance boundary.

    Earliest eligible pairs maximize the match count. Their timing error is
    deterministic, but is not a minimum-error assignment. Duplicate predictions
    remain separate proposals; reference onsets are deduplicated by the loader.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    pred = sorted(float(value) for value in beat_pred)
    real = sorted(float(value) for value in beat_real)
    if any(not math.isfinite(value) or value < 0 for value in pred + real):
        raise ValueError("Timestamps must be finite, nonnegative seconds")

    errors = []
    i = j = 0
    while i < len(pred) and j < len(real):
        delta = pred[i] - real[j]
        # Permit floating-point roundoff at an inclusive decimal boundary.
        if abs(delta) <= tolerance or math.isclose(abs(delta), tolerance,
                                                   rel_tol=1e-12, abs_tol=0):
            errors.append(abs(delta))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return _metrics(len(pred), len(real), len(errors), sum(errors))


def _metrics(predicted, reference, matched, absolute_error_sum):
    return {
        "predicted_count": predicted,
        "reference_count": reference,
        "matched_count": matched,
        "false_positives": predicted - matched,
        "false_negatives": reference - matched,
        "precision": matched / predicted if predicted else 0.0,
        "recall": matched / reference if reference else 0.0,
        "f1": 2 * matched / (predicted + reference) if predicted + reference else 0.0,
        "absolute_error_sum_seconds": absolute_error_sum,
        "mean_absolute_error_ms": 1000 * absolute_error_sum / matched if matched else None,
    }


def calculate_beat_accuracy(beat_pred, beat_real, tolerance=0.05):
    """Keep the original precision/recall/F1 helper interface."""
    result = score_beats(beat_pred, beat_real, tolerance)
    return result["precision"], result["recall"], result["f1"]


def discover_packages(roots):
    """Accept individual packages or folders containing map packages."""
    packages = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Map folder does not exist: {root}")
        candidates = [root] + sorted(path for path in root.iterdir() if path.is_dir())
        found = [path for path in candidates
                 if any(item.name.lower() == "info.dat" and item.is_file() for item in path.iterdir())]
        if not found:
            raise ValueError(f"No map packages with Info.dat in: {root}")
        packages.update(found)
    return sorted(packages)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_duration_seconds(path):
    from mutagen import File, MutagenError

    try:
        audio = File(path)
    except MutagenError as exc:
        raise ValueError(f"Could not decode audio metadata: {path}: {exc}") from exc
    if audio is None or not math.isfinite(audio.info.length) or audio.info.length <= 0:
        raise ValueError(f"Could not read a positive audio duration: {path}")
    return float(audio.info.length)


def _settings(config):
    import numpy as np

    # Snapshot settings, leaving module imports and helper functions intact.
    return {key: copy.deepcopy(value) for key, value in vars(config).items()
            if not key.startswith("_") and isinstance(value, (str, bool, int, float,
                                                             list, tuple, type(None), np.ndarray))}


def summarize_results(results, bps_values):
    summaries = []
    for bps in bps_values:
        rows = [row for row in results if row["bps_setting"] == bps]
        valid = [row for row in rows if row["status"] == "ok"]
        summary = _metrics(*(sum(row[key] for row in valid) for key in (
            "predicted_count", "reference_count", "matched_count", "absolute_error_sum_seconds")))
        duration = sum(row["audio_duration_seconds"] for row in valid)
        summary.update({"bps_setting": bps, "successful_runs": len(valid),
                        "failed_runs": len(rows) - len(valid),
                        "audio_duration_seconds": duration,
                        "predicted_density_bps": summary["predicted_count"] / duration if duration else None,
                        "reference_density_bps": summary["reference_count"] / duration if duration else None})
        summaries.append(summary)
    return summaries


def run_benchmark(packages, bps_values, tolerance=0.02, difficulty="ExpertPlus",
                  characteristic="Standard", model_dir=None, seed=3, generate_beats=None):
    """Evaluate each song/BPS pair in a temporary audio workspace.

    Config changes exist only in this process and are reset before every pair
    and on failure. The existing debug path loads only the beat model and never
    reaches map creation or Beat Saber export.
    """
    if not packages or not bps_values:
        raise ValueError("Select at least one map package and BPS setting")
    if any(not math.isfinite(bps) or bps <= 0 for bps in bps_values):
        raise ValueError("BPS settings must be finite and positive")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    score_beats([], [], tolerance)
    import numpy as np
    from tools.config import config, paths
    from tools.config.mapper_selection import get_full_model_path

    settings = _settings(config)
    path_settings = paths.songs_pred, paths.model_path
    numpy_state, random_state = np.random.get_state(), random.getstate()
    report = {"schema_version": 1, "stage": "processed_timing_proposals",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "tolerance_seconds": tolerance, "seed": seed,
              "settings": {key: value.tolist() if hasattr(value, "tolist") else value
                           for key, value in settings.items()},
              "training_overlap": "unknown; no held-out claim",
              "songs": [], "results": []}
    try:
        if generate_beats is None:
            from map_creation.gen_beats import main as generate_beats
        if model_dir is not None:
            paths.model_path = Path(model_dir).expanduser().resolve().as_posix() + "/"
        model_path = Path(get_full_model_path(config.beat_gen_version))
        report["model"] = {"path": str(model_path), "sha256": file_sha256(model_path),
                           "configured_style": config.use_mapper_selection}
        with tempfile.TemporaryDirectory(prefix="infernosaber-beat-benchmark-") as workspace:
            paths.songs_pred = Path(workspace).as_posix() + "/"
            for index, package in enumerate(packages):
                song_id = str(Path(package).resolve())
                try:
                    reference = load_reference(Path(package), difficulty, characteristic)
                    duration = audio_duration_seconds(reference["audio_path"])
                    if any(value > duration for value in reference["times"]):
                        raise ValueError("Reference onset lies beyond the audio duration")
                    metadata = {key: str(value) if isinstance(value, Path) else value
                                for key, value in reference.items() if key != "times"}
                    metadata.update({"song_id": song_id, "audio_duration_seconds": duration,
                                     "reference_times_seconds": reference["times"],
                                     "sha256": {key: file_sha256(reference[key]) for key in
                                                ("audio_path", "info_path", "beatmap_path")}})
                    report["songs"].append(metadata)
                    name = f"song_{index:04d}"
                    staged_audio = Path(workspace) / (name + ".egg")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"Failed input {package}: {exc}", flush=True)
                    for bps in bps_values:
                        report["results"].append({"song_id": song_id, "bps_setting": bps,
                                                  "status": "failed", "error": str(exc)})
                    continue

                for bps in bps_values:
                    for key, value in settings.items():
                        setattr(config, key, copy.deepcopy(value))
                    config.max_speed = config.max_speed_orig = 4 * bps
                    np.random.seed(seed)
                    random.seed(seed)
                    row = {"song_id": song_id, "bps_setting": bps,
                           "max_speed": config.max_speed, "audio_duration_seconds": duration}
                    print(f"Evaluating {reference['song_name']} / {difficulty} / BPS {bps:g}", flush=True)
                    started = time.perf_counter()
                    try:
                        shutil.copy2(reference["audio_path"], staged_audio)
                        predictions = generate_beats([name], debug_beats=True)
                        if np.isscalar(predictions) or predictions is None:
                            raise ValueError(f"Beat generation returned failure status {predictions!r}")
                        predictions = np.asarray(predictions, dtype=float)
                        if predictions.ndim != 1:
                            raise ValueError("Beat generation must return a one-dimensional timing array")
                        row.update(score_beats(predictions, reference["times"], tolerance))
                        row.update({"status": "ok", "prediction_times_seconds": predictions.tolist(),
                                    "predicted_density_bps": len(predictions) / duration,
                                    "reference_density_bps": len(reference["times"]) / duration})
                    except (Exception, SystemExit) as exc:
                        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                        print(f"Failed {reference['song_name']} / BPS {bps:g}: {row['error']}", flush=True)
                    row["effective_timing_settings"] = {
                        key: getattr(config, key) for key in ("max_speed", "max_speed_orig",
                        "thresh_beat", "thresh_onbeat", "silence_threshold", "add_beat_intensity",
                        "map_filler_iters")}
                    row["generation_wall_seconds"] = time.perf_counter() - started
                    report["results"].append(row)
                staged_audio.unlink(missing_ok=True)
    finally:
        for key, value in settings.items():
            setattr(config, key, value)
        paths.songs_pred, paths.model_path = path_settings
        np.random.set_state(numpy_state)
        random.setstate(random_state)
    report["summary_by_bps"] = summarize_results(report["results"], bps_values)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--maps-dir", type=Path, nargs="+",
                         help="Map packages or folders containing packages; originals are read only")
    sources.add_argument("--evaluation-root", type=Path,
                         help="Prepare and benchmark only this folder's bs_map_input packages")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Rebuild evaluation folders and report skips without loading a model")
    parser.add_argument("--bps", type=float, nargs="+", default=[3, 5, 7],
                        help="BPS difficulty settings (default: 3 5 7)")
    parser.add_argument("--difficulty", default="ExpertPlus", help="Exact human difficulty label")
    parser.add_argument("--characteristic", default="Standard")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Matching tolerance in seconds")
    parser.add_argument("--model-dir", type=Path, help="Override the configured model folder in memory")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, help="New JSON report; evaluation-root defaults to results/")
    args = parser.parse_args(argv)
    try:
        if args.prepare_only and args.evaluation_root is None:
            raise ValueError("--prepare-only requires --evaluation-root")
        if args.evaluation_root is None and args.output is None:
            raise ValueError("--maps-dir requires --output")
        root = args.evaluation_root.expanduser().resolve() if args.evaluation_root else None
        output = args.output.expanduser().resolve() if args.output else (
            root / "results" / ("beat-benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"))
        if output.exists():
            raise ValueError("Choose a new report path outside the input map packages")
        preprocessing = None
        if root is not None:
            managed = ("bs_map_input", "maps", "songs_egg", "maps_dict_all", "songs_diff", "archive")
            if any(output.is_relative_to(root / folder) for folder in managed):
                raise ValueError("Save benchmark results outside the source and managed evaluation folders")
            # Validate arguments before refreshing the on-disk evaluation snapshot.
            score_beats([], [], args.tolerance)
            if any(not math.isfinite(bps) or bps <= 0 for bps in args.bps):
                raise ValueError("BPS settings must be finite and positive")
            if not 0 <= args.seed < 2**32:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            from evaluation.prepare_evaluation import prepare_evaluation
            preprocessing = prepare_evaluation(root, args.difficulty, args.characteristic)
            packages = [Path(song["package_path"]) for song in preprocessing["songs"]]
            print(f"Preprocessing: {len(packages)} ready, {len(preprocessing['skipped'])} skipped", flush=True)
            for skipped in preprocessing["skipped"]:
                print(f"Skipped {Path(skipped['package_path']).name}: {skipped['error']}", flush=True)
            print(f"Preprocessing report: {root / 'preprocessing_report.json'}", flush=True)
            if args.prepare_only or not packages:
                return int(bool(preprocessing["skipped"]) or not packages)
        else:
            packages = discover_packages(args.maps_dir)
            if any(output.is_relative_to(package) for package in packages):
                raise ValueError("Choose a new report path outside the input map packages")
        report = run_benchmark(packages, list(dict.fromkeys(args.bps)), args.tolerance,
                               args.difficulty, args.characteristic, args.model_dir, args.seed)
        if preprocessing is not None:
            report["preprocessing"] = preprocessing
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report["command"] = sys.argv if argv is None else [parser.prog, *argv]
    report["runtime"] = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "tensorflow", "tf_keras", "keras", "tcn", "librosa"):
        module = sys.modules.get(name)
        if module is not None:
            report["runtime"][name] = getattr(module, "__version__", "unknown")
    for name in ("TF_USE_LEGACY_KERAS", "CUDA_VISIBLE_DEVICES"):
        report["runtime"][name] = os.environ.get(name)
    if "tensorflow" in sys.modules:
        report["runtime"]["available_devices"] = [
            device.name for device in sys.modules["tensorflow"].config.list_physical_devices()]
    try:
        root = Path(__file__).resolve().parents[1]
        report["code"] = {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
            "source_sha256": {str(path.relative_to(root)): file_sha256(path) for path in
                              (Path(__file__).resolve(), root / "evaluation/human_maps.py",
                               root / "evaluation/prepare_evaluation.py",
                               root / "map_creation/gen_beats.py")},
        }
    except (OSError, subprocess.CalledProcessError):
        report["code"] = {"revision": None, "dirty": None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for row in report["summary_by_bps"]:
        print(f"BPS {row['bps_setting']:g}: P={row['precision']:.3f} "
              f"R={row['recall']:.3f} F1={row['f1']:.3f}; "
              f"{row['successful_runs']} succeeded, {row['failed_runs']} failed")
    print(f"Saved report: {output}")
    return int(any(row["status"] != "ok" for row in report["results"])
               or bool(preprocessing and preprocessing["skipped"]))


if __name__ == "__main__":
    raise SystemExit(main())
