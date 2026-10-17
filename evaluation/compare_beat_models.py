"""Compare trained intensity/timing with legacy beat stages, stopping before notes."""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.evaluate_beat_algorithms import file_sha256, score_beats, summarize_results
from evaluation.human_maps import load_reference


def _scored_row(row, predicted, exact, synchronized, duration, tolerance):
    row = dict(row)
    row.update(score_beats(predicted, synchronized, tolerance))
    row.update({"status": "ok", "audio_duration_seconds": duration,
                "prediction_times_seconds": np.asarray(predicted, dtype=float).tolist(),
                "predicted_density_bps": len(predicted) / duration,
                "reference_density_bps": len(synchronized) / duration,
                "exact_reference_metrics": score_beats(predicted, exact, tolerance),
                "metrics_by_tolerance": {
                    str(value): {"synchronized": score_beats(predicted, synchronized, value),
                                 "exact": score_beats(predicted, exact, value)}
                    for value in sorted({0.01, 0.02, 0.035, 0.05, 0.1, tolerance})}})
    return row


def _pooled(rows):
    rows = list(rows)
    # Reuse the established count pooling, not an average of song-level F1.
    normalized = [dict(row, bps_setting=0) for row in rows]
    summary = summarize_results(normalized, [0])[0]
    summary.pop("bps_setting")
    return summary


def _paired_comparisons(results, bps_values):
    comparisons = []
    for bps in bps_values:
        new = {row["song_id"]: row for row in results if row["backend"] == "intensity_timing"
               and row["scenario"] == "nominal" and row["bps_setting"] == bps and row["status"] == "ok"}
        for backend in ("legacy_raw", "legacy_processed"):
            old = {row["song_id"]: row for row in results if row["backend"] == backend
                   and row["bps_setting"] == bps and row["status"] == "ok"}
            common = sorted(new.keys() & old.keys())
            before, after = _pooled([old[key] for key in common]), _pooled([new[key] for key in common])
            comparisons.append({"bps_setting": bps, "comparator": backend,
                                "paired_songs": len(common), "song_ids": common,
                                "legacy": before, "intensity_timing": after,
                                "f1_difference": after["f1"] - before["f1"],
                                "new_wins": sum(new[key]["f1"] > old[key]["f1"] for key in common),
                                "ties": sum(new[key]["f1"] == old[key]["f1"] for key in common)})
    return comparisons


def _overlap(reference, song, manifests):
    hashes = {"audio": file_sha256(reference["audio_path"]),
              "beatmap": file_sha256(reference["beatmap_path"])}
    return {stage: {"audio_in_checkpoint_sources": any(
                        source.get("role") == "audio" and source.get("sha256") == hashes["audio"]
                        for source in manifest.get("sources", [])),
                    "beatmap_in_checkpoint_sources": any(
                        source.get("role") == "beatmap" and source.get("sha256") == hashes["beatmap"]
                        for source in manifest.get("sources", [])),
                    "decoded_audio_sha256": song.metadata.get("decoded_sha256"),
                    "limitation": "Byte-hash check against all dataset source splits; arbitrary re-encodes require aliases."}
            for stage, manifest in manifests.items()}


def run_comparison(packages, bps_values, tolerance=0.02, difficulty="ExpertPlus",
                   characteristic="Standard", model_dir=None, seed=3, *, backend="compare",
                   intensity_artifact=None, timing_artifact=None, prediction_dir=None,
                   beat_pair=None, legacy_runner=None, legacy_observer=None):
    """Same audio/references, recorded density units, raw decoding; never call notes.

    Numeric settings exercise the legacy BPS control and the new requested colored
    NPS separately. An additional new-model run uses each human map's retained NPS,
    providing only its global count scale, never its local intensity profile.
    """
    if backend not in ("compare", "intensity-timing"):
        raise ValueError("Select compare or intensity-timing")
    if not packages or not bps_values or any(not np.isfinite(value) or value <= 0 for value in bps_values):
        raise ValueError("Select packages and finite positive BPS/NPS settings")
    score_beats([], [], tolerance)
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    from training.beat_models.contracts import REPRESENTATION, synchronize_events

    if beat_pair is None:
        from evaluation.beat_pair import BeatPair
        beat_pair = BeatPair(intensity_artifact, timing_artifact)
    root = Path(prediction_dir) if prediction_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "stage": "beats_only", "backend": backend,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "tolerance_seconds": tolerance, "seed": seed, "representation": REPRESENTATION,
              "reference_policy": "first_onset_inclusive_config_min_time_diff",
              "primary_reference": "synchronized colored-note onsets; same for every backend",
              "shared_input_policy": "Shared preparation failures are excluded from every paired comparison",
              "density_units": {"legacy": "approximate application BPS control",
                                "intensity_timing": "retained colored notes per full decoded second"},
              "comparison_limits": [
                  "Equal numeric controls do not guarantee equal generated density.",
                  "Legacy processed includes existing beat refinement/alignment; new output is raw frame argmax.",
                  "Legacy raw retains the existing threshold and legacy index-to-time convention.",
                  "New reference-density scenario uses human global retained NPS, with predicted local intensity.",
                  "Legacy training overlap is unknown; this is agreement with curated maps, not gameplay quality."],
              "models": {"intensity_timing": beat_pair.metadata}, "songs": [], "results": []}
    legacy = None
    if backend == "compare":
        if legacy_runner is None:
            from evaluation.evaluate_beat_algorithms import run_benchmark
            legacy_runner = run_benchmark
        if legacy_observer is None:
            from evaluation.legacy_observer import LegacyBeatObserver
            legacy_observer = LegacyBeatObserver()
        legacy = legacy_runner(packages, bps_values, tolerance, difficulty, characteristic,
                               model_dir, seed, generate_beats=legacy_observer)
        report["models"]["legacy"] = legacy.get("model")
        report["legacy_settings"] = legacy.get("settings", {})
    observations = {}
    if legacy_observer is not None:
        observations = {(record["name"], record["bps_setting"]): record
                        for record in legacy_observer.records}

    for index, package in enumerate(packages):
        song_id = str(Path(package).resolve())
        try:
            reference = load_reference(package, difficulty, characteristic)
            _, synchronized = synchronize_events(reference["times"], REPRESENTATION["min_time_diff_seconds"])
            started = time.perf_counter()
            song = beat_pair.prepare(reference)
            duration = float(song.metadata["duration"])
            if not np.array_equal(synchronized, song.arrays["event_times"]):
                raise ValueError("Shared synchronized references disagree with model preparation")
            metadata = {key: str(value) if isinstance(value, Path) else value
                        for key, value in reference.items() if key != "times"}
            metadata.update({"song_id": song_id, "audio_duration_seconds": duration,
                             "reference_times_seconds": synchronized.tolist(),
                             "exact_reference_times_seconds": reference["times"],
                             "original_colored_notes": len(song.arrays["original_notes"]),
                             "retained_colored_notes": len(song.arrays["retained_indices"]),
                             "synchronized_onsets": len(synchronized),
                             "reference_retained_nps": len(song.arrays["retained_indices"]) / duration,
                             "preparation_wall_seconds": time.perf_counter() - started,
                             "sha256": {key: file_sha256(reference[key]) for key in
                                        ("audio_path", "info_path", "beatmap_path")},
                             "overlap": _overlap(reference, song, getattr(beat_pair, "manifests", {}))})
            report["songs"].append(metadata)
        except (Exception, SystemExit) as error:
            print(f"Skipped beat comparison {package}: {error}", flush=True)
            for value in [*bps_values, None]:
                report["results"].append({"song_id": song_id, "backend": "intensity_timing",
                    "scenario": "nominal" if value is not None else "reference_density",
                    "bps_setting": value, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            # A failed common input is excluded from both sides of the paired comparison.
            continue

        if legacy is not None:
            for old in (item for item in legacy["results"] if item["song_id"] == song_id):
                old = dict(old, backend="legacy_processed", scenario="nominal")
                if old["status"] == "ok":
                    old = _scored_row(old, old["prediction_times_seconds"], reference["times"],
                                      synchronized, duration, tolerance)
                report["results"].append(old)
                captured = observations.get((f"song_{index:04d}", old["bps_setting"]))
                if captured is not None and captured.get("captured", True):
                    raw = {"song_id": song_id, "backend": "legacy_raw", "scenario": "nominal",
                           "bps_setting": old["bps_setting"],
                           "effective_threshold": captured["effective_threshold"],
                           "timebase": {key: captured[key] for key in ("beat_spacing", "window")}}
                    raw = _scored_row(raw, captured["times_seconds"], reference["times"],
                                      synchronized, duration, tolerance)
                    if root is not None:
                        path = root / f"legacy_{index:02d}_{old['bps_setting']:g}.npz"
                        np.savez_compressed(path, probabilities=captured["probabilities"],
                                            times_seconds=captured["times_seconds"])
                        raw["arrays"] = {"file": str(path), "sha256": file_sha256(path)}
                    report["results"].append(raw)
                else:
                    report["results"].append({"song_id": song_id, "backend": "legacy_raw",
                        "scenario": "nominal", "bps_setting": old["bps_setting"], "status": "failed",
                        "error": "Legacy generator did not reach the thresholded model output"})

        scenarios = [("nominal", value, value) for value in bps_values]
        scenarios.append(("reference_density", None, metadata["reference_retained_nps"]))
        for scenario, value, density in scenarios:
            row = {"song_id": song_id, "backend": "intensity_timing", "scenario": scenario,
                   "bps_setting": value, "requested_nps": density}
            print(f"Evaluating intensity+timing: {reference['song_name']} / NPS {density:g} / {scenario}", flush=True)
            started = time.perf_counter()
            try:
                prediction = beat_pair.predict(song, density)
                row = _scored_row(row, prediction["times_seconds"], reference["times"],
                                  synchronized, duration, tolerance)
                shares = np.asarray(prediction["intensity_shares"])
                error = np.abs(shares - song.arrays["intensity"])
                colored = int(np.asarray(prediction["counts"]).sum())
                row.update({"colored_note_count": colored, "colored_nps": colored / duration,
                            "requested_nps_error": colored / duration - density,
                            "bpm_input": prediction["bpm"], "bpm_source": "Info.dat",
                            "intensity_share_mae": float(error.mean()),
                            "intensity_total_variation": float(error.sum() / 2),
                            "class_argmax_counts": np.bincount(prediction["counts"], minlength=3).tolist(),
                            "mean_frame_probabilities": np.asarray(prediction["frame_probabilities"]).mean(axis=0).tolist()})
                if root is not None:
                    label = f"nps_{value:g}" if value is not None else "reference_density"
                    path = root / f"pair_{index:02d}_{label}.npz"
                    np.savez_compressed(path, **{key: item for key, item in prediction.items()
                                               if isinstance(item, np.ndarray)})
                    row["arrays"] = {"file": str(path), "sha256": file_sha256(path)}
            except (Exception, SystemExit) as error:
                row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                print(f"Failed intensity+timing: {row['error']}", flush=True)
            row["generation_wall_seconds"] = time.perf_counter() - started
            report["results"].append(row)
    report["summary_by_bps"] = []
    for name in ("legacy_raw", "legacy_processed", "intensity_timing"):
        selected = [row for row in report["results"] if row["backend"] == name and row["scenario"] == "nominal"]
        if selected:
            report["summary_by_bps"].extend(dict(summary, backend=name)
                                             for summary in summarize_results(selected, bps_values))
    report["reference_density_summary"] = _pooled(
        [row for row in report["results"] if row["scenario"] == "reference_density"])
    report["paired_comparisons"] = _paired_comparisons(report["results"], bps_values) if legacy else []
    return report
