"""Legacy timing/flow adapters and conservative clearance for v2 objects.

All timestamps remain elapsed seconds. Legacy note mutators run once, before
stable note IDs and expansion relationships are assigned. Clearance only drops
added objects; it never moves an upstream colored note.
"""

import math
import random
from collections import Counter

import numpy as np

from tools.config import config
from map_creation.artificial_mod import gimme_more_notes
from map_creation.bpm_optimizer import align_beats_on_bpm
from map_creation.note_postprocessing import remove_double_notes
from map_creation.sanity_check import (calc_note_pos, calc_note_speed,
                                       fill_map_times_scale, remove_silent_times,
                                       sanity_check_notes, sanity_check_timing2)


def _base(song):
    return getattr(song, "base", song)


def _report(logger, message):
    if logger:
        logger(message)


def _legacy_timing_features(audio_path):
    """Reuse the legacy spectrum and frame conventions without pitch inference."""
    import aubio
    from PIL import Image
    from beat_prediction.find_beats import get_silent_times
    from preprocessing.music_processing import log_specgram
    from training.beat_models.audio_logging import filter_vorbis_timestamp_warning

    blocks = []
    with filter_vorbis_timestamp_warning():
        source = aubio.source(str(audio_path), channels=1, samplerate=config.samplerate_music)
        try:
            while True:
                samples, read = source()
                blocks.append(np.asarray(samples).copy())
                if read < source.hop_size:
                    break
        finally:
            del source
    audio = np.concatenate(blocks)
    # The legacy spectrum needs one full window, including for a tiny song.
    minimum = int(round(35.608 * config.samplerate_music / 1000))
    if len(audio) < minimum:
        audio = np.pad(audio, (0, minimum - len(audio)))
    _, spectrum = log_specgram(audio, config.samplerate_music, 35.608)
    pitch_frames = len(blocks)
    spectrum = np.asarray(Image.fromarray(spectrum.T).resize((pitch_frames, spectrum.shape[1])))

    def scaled(values):
        values = np.asarray(values, dtype=np.float64)
        values = values - values.min()
        peak = values.max()
        return values / peak if peak else np.zeros_like(values)

    volume = scaled(spectrum.sum(axis=0))
    onset = np.r_[0., scaled(np.maximum(np.diff(spectrum.T, axis=0), 0).sum(axis=1))] if pitch_frames > 1 else np.zeros(1)
    times = np.arange(pitch_frames) * config.hop_size / config.samplerate_music
    silent = get_silent_times(spectrum, times) if config.add_silence_flag else np.empty(0)
    length = config.tcn_len

    def windows(values):
        values = np.pad(values, (0, max(0, length + 1 - len(values))), mode="edge")
        return np.lib.stride_tricks.sliding_window_view(values, length)[:-1]

    return windows(volume), windows(onset), silent


def _start_end_times(times, volume, onset):
    if not config.add_start_end_beats or not len(times):
        return times
    energy = onset.max(axis=1) + onset.mean(axis=1) + 3 * (volume.max(axis=1) + volume.mean(axis=1))
    threshold = energy.mean()
    start, end = times[0] * config.beat_spacing, times[-1] * config.beat_spacing
    added = []
    for index, value in enumerate(energy):
        if index < start or index > end:
            if value > threshold:
                threshold = value + .1
                added.append(index / config.beat_spacing)
            else:
                threshold -= .1
    return np.sort(np.r_[times, added])


def _merge_candidates(candidates, duration):
    """Earliest-onset grouping, preserving the largest count, capped at two."""
    merged = []
    for candidate in sorted(candidates, key=lambda item: item["time"]):
        time = float(candidate["time"])
        if not np.isfinite(time) or not 0 <= time < duration:
            continue
        candidate = {**candidate, "time": time, "count": min(2, max(1, int(candidate["count"])))}
        if merged and time - merged[-1]["time"] <= config.min_time_diff + 1e-9:
            merged[-1]["count"] = max(merged[-1]["count"], candidate["count"])
        else:
            merged.append(candidate)
    return merged


def postprocess_timing(song, beats, audio_path, logger=None):
    """Apply legacy timing cleanup while carrying count labels through moves."""
    base = _base(song)
    duration, bpm = float(base.metadata["duration"]), float(base.metadata["bpm"])
    frame_times = np.asarray(base.arrays["frame_times"])
    probabilities = np.asarray(beats["frame_probabilities"])
    original = [{"time": float(time), "frame": int(frame), "count": int(beats["counts"][frame])}
                for frame, time in zip(beats["event_frames"], beats["event_times"])]
    operations = []
    diagnostics = {"raw_events": len(original), "raw_retained_notes": sum(item["count"] for item in original),
                   "operations": operations}

    def record_operation(stage, previous, current, moved_times=0):
        operations.append({"stage": stage, "before_events": len(previous), "after_events": len(current),
                           "event_count_delta": len(current) - len(previous),
                           "before_retained_notes": sum(item["count"] for item in previous),
                           "after_retained_notes": sum(item["count"] for item in current),
                           "moved_event_times": moved_times})
    # No nonzero timing proposals means there is no anchor to fill. In
    # particular, requested zero density must remain empty.
    if not original:
        return [], {**diagnostics, "events": 0, "retained_notes": 0, "merged_events": 0}
    candidates = []
    spacing_removed, merged_events = 0, 0
    for candidate in original:
        if not candidates or candidate["time"] - candidates[-1]["time"] >= config.min_beat_time - 1e-9:
            candidates.append(candidate.copy())
        elif candidate["time"] - candidates[-1]["time"] <= config.min_time_diff + 1e-9:
            candidates[-1]["count"] = max(candidates[-1]["count"], candidate["count"])
            merged_events += 1
        else:
            spacing_removed += 1
    record_operation("minimum_spacing", original, candidates)
    volume, onset, silent_times = _legacy_timing_features(audio_path)

    def replace_times(stage, times, moving=False, drop_zero=False):
        nonlocal candidates, merged_events
        previous = candidates
        moved_times = 0
        if moving:
            if len(times) != len(previous):
                raise ValueError("A timing move must preserve its count-label correspondence")
            candidates = [{**item, "time": float(time)} for item, time in zip(previous, times)]
            moved_times = sum(item["time"] != float(time) for item, time in zip(previous, times))
        else:
            counts = {item["time"]: item for item in previous}
            candidates = []
            for time in times:
                if float(time) in counts:
                    candidates.append(counts[float(time)].copy())
                else:
                    frame = int(np.argmin(abs(frame_times - time)))
                    count = 1 + int(np.argmax(probabilities[frame, 1:3]))
                    candidates.append({"time": float(time), "frame": frame, "count": count})
        if drop_zero:
            candidates = [item for item in candidates if item["time"] > 0]
        before = len(candidates)
        candidates = _merge_candidates(candidates, duration)
        merged_events += before - len(candidates)
        record_operation(stage, previous, candidates, moved_times)

    def times():
        return np.asarray([item["time"] for item in candidates], np.float64)

    iterations = max(0, int(config.map_filler_iters))
    for pass_index, (threshold, index) in enumerate(((5.5 * 4, int(iterations / 2) + 1), (8 * 4, iterations - 1))):
        if candidates and iterations and config.max_speed >= threshold:
            replace_times(f"high_difficulty_fill_{pass_index}", fill_map_times_scale(times(), min(iterations - 1, max(0, index))))
    align = config.use_fixed_bpm is None or config.use_fixed_bpm <= 0
    if candidates and align:
        replace_times("initial_bpm_alignment", align_beats_on_bpm(times(), bpm), moving=True)
    if candidates:
        replace_times("onset_snap", sanity_check_timing2(None, times(), audio_path=audio_path), moving=True, drop_zero=True)
    if candidates:
        replace_times("start_end_fill", _start_end_times(times(), volume, onset))
    for index in range(iterations):
        if not candidates or len(candidates) > config.max_speed * 10 / 40 * candidates[-1]["time"] * config.add_beat_intensity / 100:
            break
        replace_times(f"density_fill_{index}", fill_map_times_scale(times(), index))
    if candidates and config.add_silence_flag:
        replace_times("silence_removal", remove_silent_times(times(), silent_times))
    if candidates and align:
        replace_times("final_bpm_alignment", align_beats_on_bpm(times(), bpm), moving=True)
    diagnostics.update(events=len(candidates), retained_notes=sum(item["count"] for item in candidates),
                       merged_events=merged_events, minimum_spacing_removed=spacing_removed,
                       timing_policy="legacy_cleanup_count_preserving")
    _report(logger, f"Timing cleanup: {len(original)} → {len(candidates)} events")
    return candidates, diagnostics


def postprocess_notes(notes, event_times, logger=None):
    """Run the shared legacy flow pass, then establish stable retained anchors."""
    from training.beat_models.data import reduce_notes

    operations = []

    def differences(current):
        def counts(items, fields):
            return Counter(tuple(item[key] for key in fields) for item in items)

        original_counts = counts(notes, ("time", "hand"))
        final_counts = counts(current, ("time", "hand"))
        original_geometry = counts(notes, ("time", "hand", "x", "y", "d"))
        final_geometry = counts(current, ("time", "hand", "x", "y", "d"))
        return {"note_count_differences": {"added_notes": sum((final_counts - original_counts).values()),
                                           "removed_notes": sum((original_counts - final_counts).values()),
                                           "basis": "per_elapsed_time_and_hand_multiplicity"},
                "geometry_record_differences": {"added_records": sum((final_geometry - original_geometry).values()),
                                                "removed_records": sum((original_geometry - final_geometry).values()),
                                                "semantics": "multiset differences include position/direction changes; not identity moves"}}

    times = np.asarray(sorted(set(float(time) for time in event_times)), np.float64)
    if not len(times) or not notes:
        return [], {"input_notes": len(notes), "retained_notes": 0, "heuristic_notes": 0, "notes": 0,
                    "operations": operations, **differences([])}
    groups = {time: [] for time in times}
    for note in notes:
        groups.setdefault(float(note["time"]), []).extend([int(note[key]) for key in ("x", "y", "hand", "d")])
    times = np.asarray(sorted(groups), np.float64)
    flattened = [groups[time] for time in times]

    def grouped_counts(values):
        return Counter((float(time), group[offset + 2]) for time, group in zip(times, values)
                       for offset in range(0, len(group), 4))

    def record_operation(stage, previous, current):
        before, after = sum(previous.values()), sum(current.values())
        operations.append({"stage": stage, "before_notes": before, "after_notes": after,
                           "delta_notes": after - before, "added_notes": sum((current - previous).values()),
                           "removed_notes": sum((previous - current).values())})

    def apply_operation(stage, transform):
        nonlocal flattened
        previous = grouped_counts(flattened)
        flattened = transform(flattened)
        record_operation(stage, previous, grouped_counts(flattened))

    if config.gimme_more_notes_flag:
        apply_operation("mirroring", gimme_more_notes)
    if config.single_notes_only_flag:
        apply_operation("initial_single_note_filter", remove_double_notes)
    apply_operation("legacy_note_flow", lambda values: sanity_check_notes(values, times))
    # Emphasis and mirroring must not undo the user's single-note selection.
    if config.single_notes_only_flag:
        apply_operation("final_single_note_filter", remove_double_notes)
    structured, source = [], []
    seen = set()
    for event, (time, values) in enumerate(zip(times, flattened)):
        for offset in range(0, len(values), 4):
            x, y, hand, direction = map(int, values[offset:offset + 4])
            key = (time, x, y, hand)
            if key in seen:
                continue
            seen.add(key)
            if not (0 <= x < 4 and 0 <= y < 3 and hand in (0, 1) and 0 <= direction <= 8):
                raise ValueError("Legacy note cleanup produced invalid colored-note geometry")
            identity = len(structured)
            structured.append({"id": identity, "time": float(time), "hand": hand, "x": x, "y": y,
                               "d": direction, "kind": "heuristic"})
            source.append([identity, 0, time, hand, x, y, direction, 0])
    record_operation("duplicate_removal", grouped_counts(flattened),
                     Counter((item["time"], item["hand"]) for item in structured))
    if structured:
        # reduce_notes implements the training cut projection and source-order
        # tie policy; group explicitly so times are never merged a second time.
        event_ids = np.searchsorted(times, [item["time"] for item in structured])
        selected, _ = reduce_notes(np.asarray(source), event_ids)
        for index in selected:
            structured[int(index)]["kind"] = "retained"
    retained = sum(note["kind"] == "retained" for note in structured)
    diagnostics = {"input_notes": len(notes), "retained_notes": retained,
                   "heuristic_notes": len(structured) - retained, "notes": len(structured),
                   "ids_assigned_after_cleanup": True, "times_changed": False,
                   "operations": operations, **differences(structured)}
    _report(logger, f"Note flow: {len(notes)} → {len(structured)} notes ({retained} retained anchors)")
    return structured, diagnostics


def _occupied_notes(state):
    """Sample the full chain sweep as well as its ordinary colored head."""
    occupied = list(state.get("notes", []))
    for chain in state.get("chains", []):
        span = chain["end_time"] - chain["time"]
        steps = max(int(chain["slices"]) - 1, int(math.ceil(span / .01)), 1)
        for index in range(1, steps + 1):
            alpha = index / steps
            occupied.append({"time": chain["time"] + alpha * span, "hand": chain["hand"],
                             "x": chain["x"] + alpha * (chain["tx"] - chain["x"]),
                             "y": chain["y"] + alpha * (chain["ty"] - chain["y"]),
                             "d": chain["d"], "chain_id": chain["id"], "anchor_id": chain["anchor_id"]})
    return occupied


def _same_cell(first, second):
    return abs(first["x"] - second["x"]) < .5 and abs(first["y"] - second["y"]) < .5


def validate_expansions(state, song, logger=None):
    """Drop conflicting learned copies/chains while retaining upstream notes."""
    originals = [note for note in state["notes"] if note.get("kind") != "copy"]
    kept = list(originals)
    prior_chains = [chain for chain in state.get("chains", []) if chain.get("_legacy_expansion_checked")]
    chain_sweep = _occupied_notes({"notes": [], "chains": prior_chains})
    removed_copies = 0
    for note in sorted((note for note in state["notes"] if note.get("kind") == "copy"), key=lambda item: (item["time"], item["id"])):
        if any(abs(note["time"] - other["time"]) <= config.min_time_diff and _same_cell(note, other)
               for other in (*kept, *chain_sweep)):
            removed_copies += 1
        else:
            kept.append(note)
    state["notes"] = sorted(kept, key=lambda item: item["id"])
    note_ids = {note["id"] for note in kept}
    chains = state.get("chains", [])
    state["chains"] = []
    for chain in chains:
        if chain["anchor_id"] not in note_ids:
            continue
        if chain.get("_legacy_expansion_checked"):
            state["chains"].append(chain)
            continue
        samples = _occupied_notes({"notes": [], "chains": [chain]})
        occupied = _occupied_notes(state)
        if any(abs(sample["time"] - note["time"]) <= config.min_time_diff and _same_cell(sample, note)
               for sample in samples for note in occupied if note.get("id") != chain["anchor_id"]):
            continue
        chain["_legacy_expansion_checked"] = True
        state["chains"].append(chain)
    diagnostics = {"rejected_copies": removed_copies, "rejected_chains": len(chains) - len(state["chains"])}
    _report(logger, f"Expansion clearance: rejected {removed_copies} copies and {diagnostics['rejected_chains']} chains")
    return diagnostics


def _wall_contains(wall, point):
    return (wall["x"] - .5 <= point["x"] <= wall["x"] + wall["width"] - .5
            and wall["y"] - .5 <= point["y"] <= wall["y"] + wall["height"] - .5)


def _integer_geometry(record, keys):
    return all(isinstance(record[key], (int, float, np.integer, np.floating))
               and not isinstance(record[key], (bool, np.bool_)) and np.isfinite(record[key])
               and int(record[key]) == record[key] for key in keys)


def _swing_cells(note):
    return [{"x": x, "y": y} for x, y in calc_note_pos([note["x"], note["y"], note["hand"], note.get("d", 8)])]


def validate_hazards(state, song, logger=None):
    """Filter model hazards with legacy margins, geometry modes and clearance."""
    duration = float(_base(song).metadata["duration"])
    occupied = sorted(_occupied_notes(state), key=lambda note: note["time"])
    times = np.asarray([note["time"] for note in occupied])
    before, after = map(float, config.obstacle_time_gap)
    positions = config.sport_obstacle_positions if config.sporty_obstacles else config.norm_obstacle_positions
    allowed_lanes = {lane for side in positions for lane in side}
    allowed_types = config.sport_obstacle_allowed_types if config.sporty_obstacles else config.norm_obstacle_allowed_types
    walls = []
    original_walls, original_bombs = list(state.get("walls", [])), list(state.get("bombs", []))
    for wall in sorted(original_walls, key=lambda item: item["time"]):
        start, end = wall["time"], wall["end_time"]
        if not config.add_obstacle_flag or not 0 <= start < end <= duration or end - start < config.obstacle_min_duration:
            continue
        if not _integer_geometry(wall, ("x", "y", "width", "height")):
            continue
        if (wall["width"] <= 0 or wall["height"] <= 0 or wall["height"] > config.max_obstacle_height
                or wall["x"] < 0 or wall["y"] < 0
                or wall["x"] + wall["width"] > 4 or wall["y"] + wall["height"] > 5):
            continue
        lanes = {lane for lane in range(4) if wall["x"] <= lane < wall["x"] + wall["width"]}
        crouch = config.sporty_obstacles and wall["y"] == 2 and wall["width"] == config.obstacle_crouch_width
        if not lanes or not lanes.issubset(allowed_lanes) or (wall["y"] not in allowed_types and not crouch):
            continue
        # Legacy sporty mode avoids an opening crouch wall and overlapping
        # opposite inner walls that provide no movement/recovery gap.
        if config.sporty_obstacles and {1, 2}.issubset(lanes) and wall["y"] >= 2 and start < 10:
            continue
        if config.sporty_obstacles and lanes.intersection({1, 2}):
            recovery = max(before, after)
            if any(other["time"] - recovery < end and other["end_time"] + recovery > start
                   and {1, 2}.issubset(lanes | {lane for lane in range(4)
                                                if other["x"] <= lane < other["x"] + other["width"]})
                   for other in walls):
                continue
        if sum(other["end_time"] > start for other in walls) >= config.obstacle_max_count:
            continue
        lo = int(np.searchsorted(times, start - before, side="left"))
        hi = int(np.searchsorted(times, end + after, side="right"))
        if any(_wall_contains(wall, note) for note in occupied[lo:hi]):
            continue
        walls.append(wall)
    bombs = []
    radius = max(config.min_time_diff, .1)
    for bomb in sorted(original_bombs, key=lambda item: item["time"]):
        if not _integer_geometry(bomb, ("x", "y")):
            continue
        if not (0 <= bomb["time"] < duration and 0 <= bomb["x"] < 4 and 0 <= bomb["y"] < 3):
            continue
        lo = int(np.searchsorted(times, bomb["time"] - radius, side="left"))
        hi = int(np.searchsorted(times, bomb["time"] + radius, side="right"))
        if any(_same_cell(bomb, cell) for note in occupied[lo:hi] for cell in _swing_cells(note)):
            continue
        if any(wall["time"] <= bomb["time"] <= wall["end_time"] and _wall_contains(wall, bomb) for wall in walls):
            continue
        if any(abs(bomb["time"] - other["time"]) <= config.min_time_diff and _same_cell(bomb, other) for other in bombs):
            continue
        bombs.append(bomb)
    state.update(walls=walls, bombs=bombs)
    diagnostics = {"rejected_walls": len(original_walls) - len(walls), "rejected_bombs": len(original_bombs) - len(bombs)}
    _report(logger, f"Hazard clearance: {len(walls)} walls and {len(bombs)} bombs accepted")
    return diagnostics


def validate_arcs(state, song, logger=None, *, settings=None, probability=None):
    """Accept only existing endpoints and clear, configured arc connections."""
    notes = {note["id"]: note for note in state["notes"]}
    occupied = _occupied_notes(state)
    original = state.get("arcs", [])
    options = settings if settings is not None else state.get("_arc_options", {})
    from training.pipeline_v2.policies import repaired
    from training.pipeline_v2.arc_policy import eligible_gap
    accepted = []
    for arc in original:
        if not config.add_slider_flag:
            continue
        if arc.get("_legacy_clearance_checked"):
            accepted.append(arc)
            continue
        ids = arc.get("endpoint_ids", [])
        if len(ids) != 2 or any(identity not in notes for identity in ids):
            continue
        head, tail = (notes[identity] for identity in ids)
        gap = tail["time"] - head["time"]
        gap_allowed = (eligible_gap(gap, options) if repaired(options, "arcs") else
                       config.slider_time_gap[0] < gap < config.slider_time_gap[1])
        if head["hand"] != tail["hand"] or not gap_allowed:
            continue
        if config.slider_movement_minimum > 0:
            speed = calc_note_speed([head[key] for key in ("x", "y", "hand", "d")],
                                    [tail[key] for key in ("x", "y", "hand", "d")], 1, cdf_lr=1.5)
            if speed < config.slider_movement_minimum:
                continue
        # A chain already owns its head's swing through its final slice.
        if any(chain["hand"] == head["hand"] and chain["time"] < tail["time"] and chain["end_time"] > head["time"]
               for chain in state.get("chains", [])):
            continue
        steps = max(1, int(math.ceil(gap / .01)))
        samples = [{"time": head["time"] + alpha * gap, "x": head["x"] + alpha * (tail["x"] - head["x"]),
                    "y": head["y"] + alpha * (tail["y"] - head["y"])} for alpha in np.linspace(0, 1, steps + 1)]
        if any(abs(sample["time"] - bomb["time"]) <= .1 and _same_cell(sample, bomb)
               for sample in samples for bomb in state.get("bombs", [])):
            continue
        if any(wall["time"] - .1 <= sample["time"] <= wall["end_time"] + .1 and _wall_contains(wall, sample)
               for sample in samples for wall in state.get("walls", [])):
            continue
        neighbors = [note for note in occupied if note.get("id") not in ids
                     and head["time"] - config.min_time_diff <= note["time"] <= tail["time"] + config.min_time_diff]
        if any(abs(sample["time"] - note["time"]) <= config.min_time_diff and _same_cell(sample, note)
               for sample in samples for note in neighbors):
            continue
        chance = config.slider_probability if probability is None else probability
        if 0 <= chance < 1 and random.random() > chance:
            continue
        arc["_legacy_clearance_checked"] = True
        accepted.append(arc)
    state["arcs"] = accepted
    diagnostics = {"rejected_arcs": len(original) - len(accepted), "synthetic_start_arc": False,
                   "clearance_geometry": "sampled_endpoint_segment"}
    _report(logger, f"Arc clearance: {len(accepted)} existing-endpoint arcs accepted")
    return diagnostics
