"""Controlled ablation comparisons for v2 legacy timing/note postprocessing.

Each arm reruns production generation on the same prepared recording with the
same seeded note decoder, changing only which legacy timing/note
postprocessing steps run and whether the learned multiplier expands anchors.
This evaluates raw model output, heuristic additions and the multiplier
separately, including the frequency and type of corrections each arm needs
(fixed upstream of the hard invariants). Default invocation only describes
the comparison; --execute performs generation and writes a report.

2026-10-31: the comparison justified skipping ``LEGACY_NOTE_MUTATORS`` in
production when ``config.v2_cleanup_note_heuristics`` is False (the default);
``full`` remains the pre-reduction all-steps baseline.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace

import numpy as np


# Step names match v2_postprocessing._TIMING_STEPS / _NOTE_STEPS.
TIMING_HEURISTIC_STEPS = ("high_difficulty_fill", "bpm_alignment", "onset_snap",
                        "start_end_fill", "density_fill", "silence_removal",
                        "max_gap_fill")
NOTE_HEURISTIC_STEPS = ("mirroring", "legacy_note_flow")
NOTE_FIX_STEPS = ("same_direction_recut",)
NOTE_MINIMAL = NOTE_HEURISTIC_STEPS + NOTE_FIX_STEPS

ARMS = {
    "full": {"timing_skip": (), "notes_skip": (), "multiplier": True,
             "summary": "all legacy postprocessing steps (pre-reduction baseline)"},
    "timing_minimal": {"timing_skip": TIMING_HEURISTIC_STEPS, "notes_skip": (),
                       "multiplier": True,
                       "summary": "raw model timing; only the minimum-spacing merge"},
    "timing_no_fill": {"timing_skip": ("high_difficulty_fill", "start_end_fill",
                                       "density_fill", "max_gap_fill"),
                       "notes_skip": (), "multiplier": True,
                       "summary": "no legacy density fills; alignment/snap/silence kept"},
    "timing_no_alignment": {"timing_skip": ("bpm_alignment", "onset_snap"),
                            "notes_skip": (), "multiplier": True,
                            "summary": "no BPM alignment or onset snapping; fills/silence kept"},
    "timing_no_silence": {"timing_skip": ("silence_removal",), "notes_skip": (),
                          "multiplier": True,
                          "summary": "no silence removal; fills/alignment/snap kept"},
    "notes_minimal": {"timing_skip": (), "notes_skip": NOTE_MINIMAL, "multiplier": True,
                      "summary": "raw model notes; structural labeling and clearance only"},
    "notes_no_additions": {"timing_skip": (), "notes_skip": ("mirroring", "flow_additions"),
                           "multiplier": True,
                           "summary": "no added notes (mirroring/emphasis/breaks); flow mutations kept"},
    "no_multiplier": {"timing_skip": (), "notes_skip": (), "multiplier": False,
                      "summary": "no learned expansion; all other cleanup kept"},
    "raw": {"timing_skip": TIMING_HEURISTIC_STEPS, "notes_skip": NOTE_MINIMAL,
            "multiplier": False,
            "summary": "pure model output plus hard invariants"},
    "raw_multiplier": {"timing_skip": TIMING_HEURISTIC_STEPS, "notes_skip": NOTE_MINIMAL,
                       "multiplier": True,
                       "summary": "raw model stream plus learned multiplier only"},
}


def arm_postprocessors(timing_skip=(), notes_skip=()):
    """Production postprocessors with the arm's steps omitted."""
    from map_creation import v2_postprocessing as base

    def timing(song, beats, audio_path, logger=None):
        return base.postprocess_timing(song, beats, audio_path, logger=logger,
                                       skip=timing_skip)

    def notes(values, event_times, logger=None):
        return base.postprocess_notes(values, event_times, logger=logger,
                                      skip=notes_skip)

    return SimpleNamespace(postprocess_timing=timing, postprocess_notes=notes,
                           validate_expansions=base.validate_expansions,
                           validate_hazards=base.validate_hazards,
                           validate_arcs=base.validate_arcs,
                           enforce_note_flow_and_clearance=base.enforce_note_flow_and_clearance)


def _none_multiplier(predictor):
    """Force the NONE expansion class while keeping the call and history honest."""
    def call(inputs):
        value = predictor.predict(inputs, verbose=0) if hasattr(predictor, "predict") else predictor(inputs)
        output = {key: np.asarray(item) for key, item in value.items()}
        for hand in ("red", "blue"):
            replaced = np.zeros_like(output[hand])
            replaced[..., 0] = 1.
            output[hand] = replaced
        return output
    call.status = getattr(predictor, "status", "trained")
    return call


def detect_onsets(audio_path):
    """Onset times from the same librosa detector the legacy snap step uses."""
    import librosa
    from tools.config import config
    factor = config.thresh_onbeat / config.thresh_onbeat_orig
    pre_max = 1 if int(5 * factor) <= 0 else int(5 * factor)
    post_max = 1 if int(5 * factor) <= 0 else int(5 * factor)
    pre_avg = 1 if int(7 * factor) <= 0 else int(7 * factor)
    post_avg = 1 if int(8 * factor) <= 0 else int(8 * factor)
    wait = int(8 * factor)
    y, sr = librosa.load(str(audio_path), sr=44100)
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=512,
                                        backtrack=False, pre_max=pre_max, post_max=post_max,
                                        pre_avg=pre_avg, post_avg=post_avg,
                                        delta=config.thresh_onbeat, wait=wait)
    return librosa.frames_to_time(frames, sr=sr, hop_length=512)


def _nearest(times, reference):
    """Distance from each entry of ``times`` to the nearest ``reference`` entry."""
    times, reference = np.asarray(times, np.float64), np.asarray(reference, np.float64)
    if not len(times):
        return np.empty(0)
    if not len(reference):
        return np.full(len(times), np.inf)
    index = np.clip(np.searchsorted(reference, times), 0, len(reference) - 1)
    before = np.clip(index - 1, 0, len(reference) - 1)
    return np.minimum(np.abs(times - reference[index]), np.abs(times - reference[before]))


def _distance_stats(distances, within=(.02, .05)):
    distances = np.asarray(distances, np.float64)
    distances = distances[np.isfinite(distances)]
    if not len(distances):
        return None
    stats = {"n": int(len(distances)), "mean": float(distances.mean()),
             "median": float(np.median(distances)),
             "p90": float(np.quantile(distances, .9)), "max": float(distances.max())}
    for window in within:
        stats[f"within_{window:g}s"] = float((distances <= window).mean())
    return stats


def _event_metrics(state, bpm, onsets, tolerance):
    """Raw-vs-final event stream, snap distance and musical alignment."""
    raw = np.asarray(state.get("raw_event_times", ()), np.float64)
    final = np.asarray(state["event_times"], np.float64)
    metrics = {"raw_events": int(len(raw)), "events": int(len(final)),
               "retained_notes": int(np.asarray(state["counts"], np.int64).sum()),
               "raw_events_dropped": int((_nearest(raw, final) > tolerance).sum()),
               "events_added": int((_nearest(final, raw) > tolerance).sum()),
               "snap_to_raw_seconds": _distance_stats(_nearest(final, raw))}
    if len(final) and onsets is not None and len(onsets):
        metrics["onset_distance_seconds"] = _distance_stats(_nearest(final, onsets))
    if len(final) and np.isfinite(bpm) and bpm > 0:
        period = 60. / float(bpm)
        phase = np.mod(final, period)
        anchor = float(np.median(phase))
        anchored = np.abs(np.mod(final - anchor + period / 2, period) - period / 2)
        metrics["bpm_grid_seconds"] = {"phase_zero": _distance_stats(np.minimum(phase, period - phase), (.02,)),
                                       "anchored": _distance_stats(anchored, (.02,)),
                                       "anchor_phase_seconds": anchor}
    return metrics


def _note_metrics(notes, duration):
    kinds = Counter(note.get("kind", "unknown") for note in notes)
    ordered = sorted(notes, key=lambda note: (note["time"], note["id"]))
    ioi, last = [], {}
    for note in ordered:
        if note["hand"] in last:
            ioi.append(note["time"] - last[note["hand"]])
        last[note["hand"]] = note["time"]
    directions = np.asarray([note["d"] for note in ordered], np.int64) \
        if ordered else np.empty(0, np.int64)
    return {"notes": len(ordered),
            "notes_per_second": len(ordered) / duration if duration else None,
            "unique_onsets": len({note["time"] for note in ordered}),
            "kinds": dict(kinds),
            "dots_share": float((directions == 8).mean()) if len(directions) else 0.,
            "direction_histogram": {str(d): int((directions == d).sum()) for d in range(9)},
            "same_hand_ioi_seconds": _distance_stats(np.asarray(ioi), within=())}


def _correction_metrics(diagnostics):
    """Per-stage operation deltas and the corrections the invariants needed."""
    timing = diagnostics.get("timing_postprocessing", {})
    notes = diagnostics.get("note_postprocessing", {})
    clearance = notes.get("note_clearance") or {}
    post_expansion = diagnostics.get("note_clearance") or {}
    hazard = diagnostics.get("hazard_clearance") or {}
    hazard_filters = hazard.get("application_filters") or hazard
    arcs = diagnostics.get("arc_clearance") or {}
    return {"timing_operations": timing.get("operations", []),
            "timing_skipped": timing.get("skipped_steps", []),
            "note_operations": notes.get("operations", []),
            "note_skipped": notes.get("skipped_steps", []),
            "pre_clearance": notes.get("pre_clearance"),
            "note_clearance": clearance or None,
            "post_expansion_clearance": post_expansion or None,
            "residual": ((post_expansion or clearance) or {}).get("residual"),
            "raw_decisions": diagnostics.get("raw_decisions"),
            "rejected_expansions": len(diagnostics.get("rejected_expansions", [])),
            "rejected_walls": hazard_filters.get("rejected_walls"),
            "rejected_bombs": hazard_filters.get("rejected_bombs"),
            "rejected_arcs": arcs.get("rejected_arcs"),
            "final_counts": diagnostics.get("final_counts")}


def _sum_counts(rows, key):
    total = Counter()
    for row in rows:
        values = row.get(key) or {}
        total.update({name: value for name, value in values.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return dict(total)


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _aggregate(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["arm"], []).append(record)
    summary = {}
    for arm, rows in grouped.items():
        summary[arm] = {
            "runs": len(rows),
            "raw_events_total": sum(row["events"]["raw_events"] for row in rows),
            "events_total": sum(row["events"]["events"] for row in rows),
            "events_added_total": sum(row["events"]["events_added"] for row in rows),
            "raw_events_dropped_total": sum(row["events"]["raw_events_dropped"] for row in rows),
            "retained_notes_total": sum(row["events"]["retained_notes"] for row in rows),
            "notes_total": sum(row["notes"]["notes"] for row in rows),
            "unique_onsets_total": sum(row["notes"]["unique_onsets"] for row in rows),
            "note_kinds_total": _sum_counts([row["notes"] for row in rows], "kinds"),
            "notes_per_second_mean": _mean(row["notes"]["notes_per_second"] for row in rows),
            "dots_share_mean": _mean(row["notes"]["dots_share"] for row in rows),
            "snap_median_seconds_mean": _mean((row["events"].get("snap_to_raw_seconds") or {}).get("median")
                                              for row in rows),
            "onset_median_seconds_mean": _mean((row["events"].get("onset_distance_seconds") or {}).get("median")
                                               for row in rows),
            "onset_within_0.05s_mean": _mean((row["events"].get("onset_distance_seconds") or {}).get("within_0.05s")
                                              for row in rows),
            "bpm_grid_median_seconds_mean": _mean((row["events"].get("bpm_grid_seconds") or {}).get("anchored", {}).get("median")
                                                  for row in rows),
            "pre_clearance_total": _sum_counts([row["corrections"] for row in rows], "pre_clearance"),
            "clearance_fixes_total": _sum_counts([row["corrections"] for row in rows], "note_clearance"),
            "residual_violations_total": _sum_counts([row["corrections"] for row in rows], "residual"),
            "rejected_expansions_total": sum(row["corrections"]["rejected_expansions"] for row in rows),
            "generation_seconds_mean": _mean(row["elapsed_seconds"] for row in rows)}
    return summary


def _run_metrics(result, bpm, onsets, duration):
    from tools.config import config
    state = result.state
    return {"events": _event_metrics(state, bpm, onsets, float(config.min_time_diff)),
            "notes": _note_metrics(state["notes"], duration),
            "corrections": _correction_metrics(result.diagnostics),
            "stage_seconds": {name: round(stage.get("elapsed_seconds", 0.), 4)
                              for name, stage in result.stages.items()}}


def _note_arrays(state):
    kinds = {"retained": 0, "heuristic": 1, "copy": 2}
    return np.asarray([[note["time"], note["hand"], note["x"], note["y"], note["d"],
                        kinds.get(note.get("kind"), 3)] for note in state["notes"]],
                      np.float64).reshape(-1, 6)


def run_comparison(songs=None, nps_values=(5.,), arm_names=None, seed=0,
                   bundle=None, output=None, max_songs=None, logger=None):
    """Generate every requested (song, difficulty, arm) and collect metrics."""
    from map_creation import v2_pipeline
    from map_creation.v2_generation import (configure_inference, scoped_settings,
                                            _difficulty_settings)
    from map_creation.v2_runtime import get_runtime
    from tools.config import paths

    configure_inference()
    runtime = get_runtime(bundle)
    root = Path(paths.songs_pred)
    names = sorted(path.stem for path in root.glob("*.egg")) if not songs else [str(name) for name in songs]
    if max_songs is not None:
        names = names[:max_songs]
    missing = [name for name in names if not (root / f"{name}.egg").is_file()]
    if missing:
        raise FileNotFoundError(f"Input songs missing under {root}: {missing}")
    arms = ARMS if arm_names is None else {name: ARMS[name] for name in arm_names}
    destination = Path(output)
    (destination / "arrays").mkdir(parents=True, exist_ok=True)

    def _json(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported report value {type(value).__name__}")

    records = []
    for song_index, name in enumerate(names):
        prepared = runtime.prepare_audio(root / f"{name}.egg")
        bpm = float(prepared.song.metadata["bpm"])
        duration = float(prepared.song.metadata["duration"])
        for diff_index, nps in enumerate(nps_values):
            internal = float(nps) * 4
            with scoped_settings():
                _difficulty_settings(internal)
                onsets = detect_onsets(prepared.audio_path)
                seed_value = int(seed) * 100_019 + song_index * 1_009 + diff_index
                for arm_name, arm in arms.items():
                    predictors = dict(runtime.predictors)
                    if not arm["multiplier"]:
                        predictors["multiplier"] = _none_multiplier(predictors["multiplier"])
                    random.seed(seed_value)
                    started = perf_counter()
                    result = v2_pipeline.generate(
                        prepared.song, float(nps), predictors, prepared.encoder,
                        runtime.settings, runtime.vocabularies,
                        lighting_generator=runtime.lighting,
                        audio_path=prepared.audio_path,
                        postprocessors=arm_postprocessors(arm["timing_skip"], arm["notes_skip"]),
                        rng=np.random.default_rng(seed_value),
                        intensity=prepared.intensity, logger=logger)
                    prepared.intensity = result.state["raw_intensity"].copy()
                    metrics = _run_metrics(result, bpm, onsets, duration)
                    record = {"song": name, "bpm": bpm, "duration": duration,
                              "nps": float(nps), "internal_difficulty": internal,
                              "arm": arm_name, "seed": seed_value,
                              "elapsed_seconds": perf_counter() - started, **metrics}
                    array_path = destination / "arrays" / f"{name}__nps{nps:g}__{arm_name}.npz"
                    np.savez_compressed(array_path,
                                        raw_event_times=result.state["raw_event_times"],
                                        event_times=result.state["event_times"],
                                        counts=result.state["counts"],
                                        notes=_note_arrays(result.state))
                    record["arrays"] = str(array_path.relative_to(destination))
                    records.append(record)
                    with open(destination / "records.jsonl", "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, default=_json) + "\n")

    report = {"kind": "v2_postprocessing_comparison", "schema_version": 1,
              "seed": int(seed), "bundle": str(runtime.root),
              "arms": {name: {**{key: (list(value) if isinstance(value, tuple) else value)
                                    for key, value in arm.items()}}
                       for name, arm in arms.items()},
              "songs": names, "nps_values": [float(nps) for nps in nps_values],
              "records": records, "aggregate": _aggregate(records)}
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, default=_json, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--songs", nargs="*", default=None,
                         help="Song stems in paths.songs_pred; default: every .egg file")
    command.add_argument("--max-songs", type=int, default=None,
                         help="Cap the number of songs after selection")
    command.add_argument("--nps", default="5",
                         help="Comma-separated retained-notes-per-second requests "
                              "(app difficulty values; internal difficulty is 4x)")
    command.add_argument("--arms", default=None,
                         help=f"Comma-separated subset of: {', '.join(ARMS)}")
    command.add_argument("--seed", type=int, default=0,
                         help="Shared decoder seed; arms on one (song, nps) stay paired")
    command.add_argument("--bundle", default=None,
                         help="Explicit v2 bundle path (default: config v2_model_bundle_path)")
    command.add_argument("--output", required=True, help="A new comparison report directory")
    action = command.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true",
                        help="Display the plan without loading data or models (the default)")
    action.add_argument("--execute", action="store_true",
                        help="Run the generation arms and write the report")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        nps_values = tuple(float(item) for item in args.nps.split(",") if item.strip())
    except ValueError:
        command.error("--nps must be comma-separated numbers")
    if not nps_values or any(not np.isfinite(nps) or nps < 0 for nps in nps_values):
        command.error("--nps values must be finite and nonnegative")
    arm_names = None
    if args.arms:
        arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
        unknown = set(arm_names) - set(ARMS)
        if unknown:
            command.error(f"Unknown arms: {sorted(unknown)}; choose from {', '.join(ARMS)}")
    if not args.execute:
        plan = {"status": "planned", "songs": args.songs or "all .egg files in paths.songs_pred",
                "max_songs": args.max_songs, "nps_values": nps_values,
                "arms": {name: ARMS[name]["summary"] for name in (arm_names or list(ARMS))},
                "seed": args.seed, "bundle": args.bundle or "config default",
                "output": args.output, "execution_requested": False,
                "instruction": "Add --execute only when a controlled comparison run is requested."}
        print(json.dumps(plan, indent=2))
        return plan
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        command.error("--output must be a new or empty directory")
    report = run_comparison(songs=args.songs, nps_values=nps_values, arm_names=arm_names,
                            seed=args.seed, bundle=args.bundle, output=args.output,
                            max_songs=args.max_songs)
    print(f"V2 postprocessing comparison saved in {destination}")
    for arm, row in report["aggregate"].items():
        nps_mean = row["notes_per_second_mean"]
        print(f"  {arm}: {row['runs']} runs, {row['notes_total']} notes "
              f"({row['note_kinds_total']}), NPS {f'{nps_mean:.3f}' if nps_mean is not None else 'n/a'}, "
              f"residual {sum((row['residual_violations_total'] or {}).values())}")
    return report


if __name__ == "__main__":
    main()
