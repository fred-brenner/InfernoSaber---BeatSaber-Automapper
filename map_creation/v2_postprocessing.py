"""Legacy timing/flow adapters and conservative clearance for v2 objects.

All timestamps remain elapsed seconds. Legacy note mutators run once, before
stable note IDs and expansion relationships are assigned. Clearance resolves
same-time same-cell and swing-path conflicts; it prefers mutating heuristic or
copied notes and only touches retained model notes when nothing else resolves
the conflict. A start-of-map floor drops early notes (``apply_intro_floor``;
hazards get their own floor inside ``validate_hazards``) and a best-effort
diagonal cap recuts excess diagonal cuts toward flow-legal adjacent cardinals
— both run after note cleanup and onset snap, before the retained snapshot.

Cut-direction convention: ``get_cut_dir_xy`` yields the approach vector — the
hand enters from ``pos + step`` and exits through ``pos - step``. Both adjacent
cells along the arrow axis belong to the swing volume and must stay free of
simultaneous other-hand notes (same occupancy model as ``calc_note_pos``).
"""

import math
import random
from collections import Counter

import numpy as np

from tools.config import config
from map_creation.bpm_optimizer import align_beats_on_bpm
from map_creation.note_postprocessing import remove_double_notes
from map_creation.sanity_check import (calc_note_pos, calc_note_speed,
                                       fill_map_times_scale, get_cut_dir_xy, remove_silent_times,
                                       sanity_check_notes, sanity_check_timing2)


def _base(song):
    return getattr(song, "base", song)


def _report(logger, message):
    if logger and config.verbose_level > 1:
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


def _section_entries(times, pause_seconds):
    """Map start plus every time that follows a note-free gap of ``pause_seconds``."""
    times = np.sort(np.asarray(times, np.float64))
    if not len(times):
        return times
    return times[np.r_[True, np.diff(times) >= pause_seconds]]


def _blocked_intervals(entries, delay):
    """Post-entry windows where subdivision fills are gated; empty when off."""
    if delay <= 0:
        return []
    return [(float(entry), float(entry) + delay) for entry in entries]


def _silent_spans(silent_times):
    """Merge silent frame times into ``(lo, hi)`` spans, closing sub-4-hop gaps."""
    times = np.sort(np.asarray(silent_times, np.float64))
    if not len(times):
        return []
    max_gap = 4 * config.hop_size / config.samplerate_music
    spans, start = [], float(times[0])
    for index in range(1, len(times)):
        if times[index] - times[index - 1] >= max_gap:
            spans.append((start, float(times[index - 1])))
            start = float(times[index])
    return spans + [(start, float(times[-1]))]


def _inside_spans(time, spans):
    """True when ``time`` sits inside any ``(lo, hi)`` span, boundaries included."""
    return any(lo <= time <= hi for lo, hi in spans)


def _filter_fill_times(previous, new_times, blocked, silent_spans=()):
    """Drop fill insertions inside ``blocked`` windows or ``silent_spans``; model times always pass."""
    if not blocked and not silent_spans:
        return np.asarray(new_times, np.float64)
    kept = {float(time) for time in previous}
    return np.asarray([float(time) for time in new_times
                       if float(time) in kept
                       or not (any(start <= float(time) < end for start, end in blocked)
                               or _inside_spans(float(time), silent_spans))],
                      np.float64)


_TIMING_STEPS = ("minimum_spacing", "high_difficulty_fill", "bpm_alignment", "onset_snap",
                 "start_end_fill", "silence_removal", "density_fill", "max_gap_fill")


def postprocess_timing(song, beats, audio_path, logger=None, skip=()):
    """Apply legacy timing cleanup while carrying count labels through moves.

    Step order follows ``_TIMING_STEPS``: ``minimum_spacing``,
    ``high_difficulty_fill``, ``bpm_alignment``, ``onset_snap``,
    ``start_end_fill``, ``silence_removal``, ``density_fill``,
    ``max_gap_fill`` (the one ``bpm_alignment`` name covers the bookending
    initial/final alignments). Subdivision fills keep insertions out of
    post-entry ramp windows and detected silent spans; unmatched onset-snap
    events follow ``v2_onset_snap_unmatched_policy``; ``max_gap_fill``
    backstops any remaining dead zone from ``v2_gap_fill_source``.

    ``skip`` names steps from ``_TIMING_STEPS`` to omit for controlled
    comparisons. Legacy audio features are decoded lazily — skipping every
    step that needs them avoids the legacy analysis entirely.
    """
    unknown = set(skip) - set(_TIMING_STEPS)
    if unknown:
        raise ValueError(f"Unknown timing postprocessing steps: {sorted(unknown)}")
    base = _base(song)
    duration, bpm = float(base.metadata["duration"]), float(base.metadata["bpm"])
    frame_times = np.asarray(base.arrays["frame_times"])
    probabilities = np.asarray(beats["frame_probabilities"])
    original = [{"time": float(time), "frame": int(frame), "count": int(beats["counts"][frame])}
                for frame, time in zip(beats["event_frames"], beats["event_times"])]
    operations = []
    diagnostics = {"raw_events": len(original), "raw_retained_notes": sum(item["count"] for item in original),
                   "operations": operations, "skipped_steps": sorted(skip)}

    def record_operation(stage, previous, current, moved_times=0):
        operations.append({"stage": stage, "before_events": len(previous), "after_events": len(current),
                           "event_count_delta": len(current) - len(previous),
                           "before_retained_notes": sum(item["count"] for item in previous),
                           "after_retained_notes": sum(item["count"] for item in current),
                           "moved_event_times": moved_times})
    # No nonzero timing proposals means there is no anchor to fill. In
    # particular, requested zero density must remain empty.
    if not original:
        return [], {**diagnostics, "events": 0, "retained_notes": 0, "merged_events": 0,
                    "gap_fill_source_used": None}
    candidates = []
    spacing_removed, merged_events = 0, 0
    if "minimum_spacing" in skip:
        candidates = [item.copy() for item in original]
    else:
        for candidate in original:
            if not candidates or candidate["time"] - candidates[-1]["time"] >= config.min_beat_time - 1e-9:
                candidates.append(candidate.copy())
            elif candidate["time"] - candidates[-1]["time"] <= config.min_time_diff + 1e-9:
                candidates[-1]["count"] = max(candidates[-1]["count"], candidate["count"])
                merged_events += 1
            else:
                spacing_removed += 1
        record_operation("minimum_spacing", original, candidates)
    features = None

    def legacy_features():
        nonlocal features
        if features is None:
            features = _legacy_timing_features(audio_path)
        return features

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

    def fill_blocked():
        pause = float(getattr(config, "v2_fill_entry_pause_beats", 4.0) or 0.) * 60. / bpm
        delay = float(getattr(config, "v2_fill_entry_delay_seconds", 0.) or 0.)
        entries = _section_entries(times(), pause) if pause > 0 else np.asarray([times()[0]])
        return _blocked_intervals(entries, delay)

    detected_silence = None
    gap_fill_source_used = None

    def silent_spans(force=False):
        """Detected silent spans, decoded once and only when fills/snap gate on them.

        The cached detection keys on ``add_silence_flag`` alone so call order
        cannot leak the gate state; ``v2_fill_respect_silence`` applies per
        call. ``force`` bypasses that fill gate — the onset-snap unmatched
        policy must still cull events inside detected silence."""
        nonlocal detected_silence
        if detected_silence is None:
            detected_silence = (_silent_spans(legacy_features()[2])
                                if config.add_silence_flag else [])
        return detected_silence if force or config.v2_fill_respect_silence else []

    iterations = max(0, int(config.map_filler_iters))
    if "high_difficulty_fill" not in skip:
        for pass_index, (threshold, index) in enumerate(((5.5 * 4, int(iterations / 2) + 1), (8 * 4, iterations - 1))):
            if candidates and iterations and config.max_speed >= threshold:
                replace_times(f"high_difficulty_fill_{pass_index}",
                              _filter_fill_times(times(), fill_map_times_scale(times(), min(iterations - 1, max(0, index))),
                                                 fill_blocked(), silent_spans=silent_spans()))
    align = "bpm_alignment" not in skip
    if candidates and align:
        replace_times("initial_bpm_alignment", align_beats_on_bpm(times(), bpm), moving=True)
    if candidates and "onset_snap" not in skip:
        snapped = sanity_check_timing2(None, times(), audio_path=audio_path)
        policy = getattr(config, "v2_onset_snap_unmatched_policy", "always")
        if policy not in ("always", "non_silent", "never"):
            raise ValueError(f"Unknown v2_onset_snap_unmatched_policy: {policy!r}")
        if policy != "always":
            spans = silent_spans(force=True) if policy == "non_silent" else ()
            snapped = np.asarray([original if marker <= 0 and (policy == "never" or not _inside_spans(original, spans))
                                  else marker
                                  for original, marker in zip(times(), snapped)])
        replace_times("onset_snap", snapped, moving=True, drop_zero=True)
    if candidates and "start_end_fill" not in skip:
        volume, onset, _ = legacy_features()
        replace_times("start_end_fill", _start_end_times(times(), volume, onset))
    if candidates and config.add_silence_flag and "silence_removal" not in skip:
        *_, silent_times = legacy_features()
        replace_times("silence_removal", remove_silent_times(times(), silent_times))
    if "density_fill" not in skip:
        for index in range(iterations):
            if not candidates or len(candidates) > config.max_speed * 10 / 40 * candidates[-1]["time"] * config.add_beat_intensity / 100:
                break
            replace_times(f"density_fill_{index}",
                          _filter_fill_times(times(), fill_map_times_scale(times(), index), fill_blocked(),
                                             silent_spans=silent_spans()))
    if candidates and "max_gap_fill" not in skip:
        max_gap = float(getattr(config, "v2_max_note_gap_seconds", 0.) or 0.)
        if max_gap > 0:
            edge = float(getattr(config, "v2_gap_fill_edge_seconds", .5) or 0.)
            source = getattr(config, "v2_gap_fill_source", "onsets")
            current, inserted, pool, spans = times(), [], None, ()
            for a, b in zip(current, current[1:]):
                if b - a <= max_gap:
                    continue
                if pool is None:
                    # decode candidates and silence only when a dead zone exists
                    used = {"source": source}
                    pool = _gap_candidate_times(source, audio_path, bpm, duration, used)
                    gap_fill_source_used = used["source"]
                    spans = silent_spans()
                for t in pool[(pool > a + edge) & (pool < b - edge)]:
                    if _inside_spans(t, spans) or (inserted and t - inserted[-1] < config.min_beat_time):
                        continue
                    inserted.append(float(t))
            if inserted:
                replace_times("max_gap_fill", np.sort(np.r_[current, inserted]))
    if candidates and align:
        replace_times("final_bpm_alignment", align_beats_on_bpm(times(), bpm), moving=True)
    diagnostics.update(events=len(candidates), retained_notes=sum(item["count"] for item in candidates),
                       merged_events=merged_events, minimum_spacing_removed=spacing_removed,
                       timing_policy="legacy_cleanup_count_preserving",
                       gap_fill_source_used=gap_fill_source_used)
    _report(logger, f"Timing cleanup: {len(original)} -> {len(candidates)} events")
    return candidates, diagnostics


# Exact reversals keep consecutive same-hand swings alternating; dots are
# directionless and never need a recut.
_REVERSE_DIRECTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}
# Direction index → 45°-step angle (right=0, counterclockwise).
_DIRECTION_ANGLE = {3: 0, 5: 1, 0: 2, 4: 3, 2: 4, 6: 5, 1: 6, 7: 7}
# Direction index → axis group; recuts prefer staying inside the original's
# group so the fix does not reshape the model's direction marginals.
_DIRECTION_GROUP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3}
# Direction index → one-cell (dx, dy) offset. ``get_cut_dir_xy`` yields the
# approach vector: the hand enters from ``pos + step`` and exits through
# ``pos - step``, so both adjacent cells along the axis belong to the swing
# volume (same occupancy model as ``calc_note_pos``).
_DIRECTION_STEP = {direction: get_cut_dir_xy(direction) for direction in range(8)}
_STEP_DIRECTION = {step: direction for direction, step in _DIRECTION_STEP.items()}


def _swept_cells(x, y, direction):
    """In-grid cells the saber passes through on this cut: approach and exit."""
    step = _DIRECTION_STEP.get(direction)
    if step is None:
        return ()
    return tuple((x + sign * step[0], y + sign * step[1]) for sign in (1, -1)
                 if 0 <= x + sign * step[0] < 4 and 0 <= y + sign * step[1] < 3)


def _dot_virtual_direction(previous_d, px, py, x, y):
    """Implied direction of a dot note from geometry.

    After the previous cut the hand rests near its exit cell ``pos - step``;
    the dot is then hit by whatever motion reaches it from there. A dot on the
    exit cell continues the same stroke; a dot in the return zone inherits the
    pendulum reversal. ``None`` when there is no previous direction."""
    if previous_d is None or previous_d == 8:
        return None
    dx, dy = _DIRECTION_STEP[previous_d]
    mx, my = x - px + dx, y - py + dy
    if mx == 0 and my == 0:
        return _REVERSE_DIRECTION[previous_d]
    return _STEP_DIRECTION[(-int(np.sign(mx)), -int(np.sign(my)))]


def _v2_mirroring(values):
    """``gimme_more_notes`` variant that also mirrors the cut direction.

    The legacy pass only mirrors lane and hand, so a copied inward cut became
    an unmotivated outward/inward mismatch for the other hand. Mirroring the
    horizontal step (dy preserved) keeps doubled sections symmetric like human
    maps. v1 keeps using the untouched legacy function."""
    probability = config.gimme_more_notes_prob
    if probability <= 0:
        return values
    extra = config.max_speed / 4 - 5
    if extra > 0:
        probability += 0.1 * 0.1 * extra
    for index, section in enumerate(values):
        if probability < random.random() or len(section) < 4 or len(set(section[2::4])) != 1:
            continue
        mirrored = []
        for offset in range(0, len(section), 4):
            x, y, hand, direction = map(int, section[offset:offset + 4])
            mirror_hand = 1 - hand if hand in (0, 1) else hand
            if direction == 8:
                mirror_direction = 8
            else:
                dx, dy = _DIRECTION_STEP[direction]
                mirror_direction = _STEP_DIRECTION[(-dx, dy)]
            mirrored.extend([3 - x, y, mirror_hand, mirror_direction])
        values[index] = section + mirrored
    return values


def _step_distance(a, b):
    """Smallest angle between two cut directions in 45° steps (0–4); dots count as free."""
    if a == 8 or b == 8:
        return 4
    pa, pb = _DIRECTION_ANGLE[a], _DIRECTION_ANGLE[b]
    return min((pa - pb) % 8, (pb - pa) % 8)


def _swing_clear(origin, forbidden, direction):
    """No swept cell of ``direction`` from ``origin`` sits in ``forbidden``."""
    return (origin is None or not forbidden
            or not (set(_swept_cells(origin[0], origin[1], direction)) & forbidden))


def _rank_flow_candidates(original, candidates, others):
    """Reversal-preferring order over pre-filtered directions; 8 = give up."""
    if not candidates:
        return 8
    same_group = [d for d in candidates if _DIRECTION_GROUP[d] == _DIRECTION_GROUP[original]]
    pool = same_group or candidates
    return min(pool, key=lambda d: (-min((_step_distance(d, other) for other in others), default=8),
                                    _step_distance(d, original), d))


def _flow_legal(direction, previous, x, y, time, same_stroke_gap):
    """Strict ≥135° rule against a (time, x, y, effective direction) predecessor.

    Near-simultaneous pairs share one stroke and are exempt. With
    ``v2_flow_min_turn_steps`` ≤ 2 the legacy rule returns: ≥135°, or 90°
    when the hand physically travels at least two cells."""
    if previous is None:
        return True
    ptime, px, py, pdirection = previous
    if pdirection is None or time - ptime <= same_stroke_gap:
        return True
    steps = _step_distance(direction, pdirection)
    if steps <= 1:
        return False
    # Missing/None selects the strict default; an explicit 0 already means
    # <= 2 and must not be coerced back to 3.
    configured = getattr(config, "v2_flow_min_turn_steps", None)
    min_turn = 3 if configured is None else int(configured)
    if min_turn <= 2:
        return steps >= 3 or (steps == 2 and abs(x - px) + abs(y - py) >= 2)
    return steps >= min_turn


def _flow_direction(original, previous_direction, blocked=(), swing_from=None, forbidden_cells=None):
    """Nearest reversal of ``original`` that keeps ≥135° from ``previous_direction``.

    Human flow resolves repeats with a swing back (~180°), so the candidate
    furthest from ``previous_direction`` wins; ties fall back to the smallest
    rotation away from ``original``. This keeps the forced fix inside the
    reversal vocabulary instead of manufacturing side cuts the model never
    asked for. ``blocked`` lists further directions the replacement must stay
    ≥135° away from (used when recutting the earlier note of a pair). With
    ``swing_from``/``forbidden_cells`` a candidate is rejected when one of its
    swept cells is occupied by a simultaneous other-hand note."""
    others = (previous_direction, *blocked)
    candidates = [d for d in range(8)
                  if all(_step_distance(d, other) >= 3 for other in others)
                  and _swing_clear(swing_from, forbidden_cells, d)]
    return _rank_flow_candidates(original, candidates, others)


def _recut_same_direction_repeats(flattened, times, same_stroke_gap, flipped=None):
    """Rotate a note that repeats its predecessor's cut within ±45°.

    Consecutive same-hand notes *farther* than ``same_stroke_gap`` seconds need
    a ≥135° direction change (``v2_flow_min_turn_steps`` ≤ 2 restores the
    legacy 90°-with-two-cell-travel rule). Near-simultaneous pairs share one
    stroke and are exempt; dots never need a recut themselves but pass their
    geometry-implied direction on to the next note's check.
    ``flipped``, when given, is incremented per rewritten note.
    """
    avoid_swept = bool(getattr(config, "v2_cleanup_swing_clash", True))
    last = {}
    for time, group in zip(times, flattened):
        for offset in range(0, len(group), 4):
            x, y, hand, direction = map(int, group[offset:offset + 4])
            previous = last.get(hand)
            if previous is not None:
                ptime, px, py, pdirection = previous
                if direction != 8 and not _flow_legal(direction, previous, x, y, time, same_stroke_gap):
                    forbidden = ({(int(group[o]), int(group[o + 1])) for o in range(0, len(group), 4)
                                  if int(group[o + 2]) != hand} if avoid_swept else None)
                    direction = _flow_direction(direction, pdirection,
                                                swing_from=(x, y), forbidden_cells=forbidden)
                    group[offset + 3] = direction
                    if flipped is not None:
                        flipped["n"] += 1
            # dots get their implied direction from geometry, alternating the pendulum
            last[hand] = (time, x, y, direction if direction != 8 else
                          (_dot_virtual_direction(pdirection, px, py, x, y) if previous else None))
    return flattened


def recut_same_direction_notes(notes, chains=None, protect_retained=True):
    """Flip any note repeating its same-hand predecessor's cut direction.

    Covers heuristic additions, dedup survivors, and multiplier copies that
    the stream-level pass cannot see. With ``protect_retained`` a retained
    model note (contract-fixed downstream) is never flipped; a repeat against
    a non-retained predecessor is then resolved by recutting that earlier
    note against both neighbors instead. Chain heads keep their swing
    consistent with a flipped anchor. Returns the number of flipped notes.
    """
    anchors, flipped, last = {note["id"]: note for note in notes}, 0, {}
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    same_stroke_gap = float(getattr(config, "v2_cleanup_same_dir_gap", 0.1) or 0.)
    if getattr(config, "v2_cleanup_swing_clash", True):
        eps = float(getattr(config, "min_time_diff", .02))
        window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
        other_cells = _other_hand_cells(ordered, eps, window)
    else:
        other_cells = {}
    for note in ordered:
        history = last.setdefault(note["hand"], [None, None])
        previous = history[1]
        prior = ((previous["time"], previous["x"], previous["y"], previous["d"])
                 if previous is not None else None)
        if note["d"] != 8 and not _flow_legal(note["d"], prior, note["x"], note["y"], note["time"], same_stroke_gap):
            # prefer recutting a heuristic predecessor over a retained model
            # note: dedup/mirroring artifacts created most violating pairs
            if previous.get("kind") != "retained" and not previous.get("_virtual") \
                    and note.get("kind") == "retained":
                before = history[0]
                blocked = (before["d"],) if before is not None and before["d"] != 8 else ()
                previous["d"] = _flow_direction(previous["d"], note["d"], blocked,
                                                swing_from=(previous["x"], previous["y"]),
                                                forbidden_cells=other_cells.get(previous["id"]))
                flipped += 1
            elif not protect_retained or note.get("kind") != "retained":
                note["d"] = _flow_direction(note["d"], previous["d"], swing_from=(note["x"], note["y"]),
                                            forbidden_cells=other_cells.get(note["id"]))
                flipped += 1
        # dots get their implied direction from geometry for the next pair;
        # the wrapper is virtual-only and must never become a recut target
        effective = note
        if note["d"] == 8 and previous is not None:
            virtual = _dot_virtual_direction(previous["d"], previous["x"], previous["y"],
                                             note["x"], note["y"])
            effective = {**note, "d": virtual if virtual is not None else 8, "_virtual": True}
        history[:] = [previous, effective]
    for chain in chains or []:
        anchor = anchors.get(chain.get("anchor_id"))
        if anchor is not None:
            chain["d"] = anchor["d"]
    return flipped


def _simultaneous_pairs(ordered, window):
    """All ordered note pairs at most ``window`` seconds apart, regardless of hand."""
    for index, note in enumerate(ordered):
        for other in ordered[index + 1:]:
            if other["time"] - note["time"] > window:
                break
            yield note, other


def _other_hand_cells(ordered, eps, window):
    """Note id → cells of other-hand notes standing during its swing.

    A swing at ``t`` clips an uncut other-hand block whose own hit time lies
    inside ``[t - eps, t + window]``: simultaneous-or-earlier notes inside the
    ``eps`` grace count both ways, later notes are still standing (forward
    horizon only — a note struck before the swing is already cleared)."""
    cells = {note["id"]: set() for note in ordered}
    for first, second in _simultaneous_pairs(ordered, window):
        if first["hand"] != second["hand"]:
            cells[first["id"]].add((second["x"], second["y"]))
            if second["time"] - first["time"] <= eps:
                cells[second["id"]].add((first["x"], first["y"]))
    return cells


def _history_context(ordered, eps):
    """Per note: same-hand predecessor (time, x, y, virtual direction through
    dots) strictly before its simultaneous group, plus for each dot its implied
    direction derived from the predecessor's exit geometry."""
    pred, incoming = {}, {}
    last_eff = {}
    start = 0
    for note in ordered:
        while ordered[start]["time"] < note["time"] - eps:
            earlier = ordered[start]
            before = last_eff.get(earlier["hand"])
            if earlier["d"] != 8:
                effective = earlier["d"]
            elif before is not None and before[3] is not None:
                effective = _dot_virtual_direction(before[3], before[1], before[2],
                                                   earlier["x"], earlier["y"])
            else:
                effective = None
            last_eff[earlier["hand"]] = (earlier["time"], earlier["x"], earlier["y"], effective)
            start += 1
        pred[note["id"]] = last_eff.get(note["hand"])
        before = pred[note["id"]]
        incoming[note["id"]] = note["d"] if note["d"] != 8 else (
            _dot_virtual_direction(before[3], before[1], before[2], note["x"], note["y"])
            if before is not None and before[3] is not None else None)
    return pred, incoming


def _off_side(note):
    """The note sits on the opposite hand's half of the grid."""
    return (note["x"] >= 2) == (note["hand"] == 0)


def _relocation_cells(note, ordered, eps, window, incoming, dropped):
    """Free cells for ``note``: unoccupied at its own time, outside every
    earlier-or-simultaneous other-hand swing volume, and not swinging into a
    later-standing other-hand note itself."""
    occupied, swung, later = set(), set(), set()
    for other in ordered:
        if other["id"] == note["id"] or other["id"] in dropped:
            continue
        dt = other["time"] - note["time"]
        if abs(dt) <= eps:
            occupied.add((other["x"], other["y"]))
        if other["hand"] != note["hand"]:
            direction = other["d"] if other["d"] != 8 else incoming.get(other["id"])
            if -window <= dt <= eps and direction is not None and direction != 8:
                swung.update(_swept_cells(other["x"], other["y"], direction))
                swung.add((other["x"], other["y"]))
            if -eps <= dt <= window:
                later.add((other["x"], other["y"]))
    cells = []
    for x in range(4):
        for y in range(3):
            if (x, y) in occupied or (x, y) in swung:
                continue
            direction = note["d"] if note["d"] != 8 else incoming.get(note["id"])
            own = {(x, y)} | set(_swept_cells(x, y, direction)) \
                if direction is not None and direction != 8 else {(x, y)}
            if own & later:
                continue
            cells.append((x, y))
    cells.sort(key=lambda cell: ((cell[0] >= 2) != (note["hand"] == 1),
                                 abs(cell[0] - note["x"]) + abs(cell[1] - note["y"]), cell[0], cell[1]))
    return cells


def _resolve_clashes(notes, protect_retained, check_cells, check_swing, counters):
    """One pass over simultaneous-position conflicts, mutating ``notes``.

    Same-time same-cell notes are relocated to a clear cell (preferring the
    off-side, non-retained one) or dropped when nothing is free; same-hand
    duplicates are dropped. A note whose swing volume (approach cell, own
    cell and exit cell) is occupied by an uncut other-hand note — struck at
    the same instant or up to ``window`` seconds later — is resolved by
    recutting the swinger, then relocating the target, then the swinger,
    then dropping the target. Returns the number of changed notes."""
    eps = float(getattr(config, "min_time_diff", .02))
    window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    pred, incoming = _history_context(ordered, eps)
    other_cells = _other_hand_cells(ordered, eps, window)
    dropped, changed = set(), 0
    alive = lambda note: note["id"] not in dropped
    movable = lambda note: not (protect_retained and note.get("kind") == "retained")

    def drop(note):
        nonlocal changed
        dropped.add(note["id"])
        counters["dropped"] += 1
        changed += 1

    if check_cells:
        for first, second in _simultaneous_pairs(ordered, eps):
            if not alive(first) or not alive(second) or (first["x"], first["y"]) != (second["x"], second["y"]):
                continue
            if first["hand"] == second["hand"]:
                drop(first if movable(first) and not movable(second) else second)
                counters["cell_dropped"] += 1
                continue
            movers = sorted((first, second), key=lambda note: (not movable(note), not _off_side(note), note["id"]))
            for mover in movers:
                cells = _relocation_cells(mover, ordered, eps, window, incoming, dropped)
                if cells:
                    mover["x"], mover["y"] = cells[0]
                    counters["cell_relocated"] += 1
                    changed += 1
                    break
            else:
                drop(movers[0])
                counters["cell_dropped"] += 1
    if check_swing:
        partners = {note["id"]: [] for note in ordered}
        for first, second in _simultaneous_pairs(ordered, window):
            if first["hand"] != second["hand"]:
                partners[first["id"]].append(second)
                if second["time"] - first["time"] <= eps:
                    partners[second["id"]].append(first)
        for swinger in ordered:
            if not alive(swinger):
                continue
            direction = swinger["d"] if swinger["d"] != 8 else incoming.get(swinger["id"])
            swept = set(_swept_cells(swinger["x"], swinger["y"], direction)) \
                if direction is not None and direction != 8 else set()
            for target in partners[swinger["id"]]:
                if not alive(target):
                    continue
                position = (target["x"], target["y"])
                on_cell = position == (swinger["x"], swinger["y"])
                if position not in swept and not (on_cell and target["time"] - swinger["time"] > eps):
                    continue
                resolved = False
                previous = pred.get(swinger["id"])
                if swinger["d"] != 8 and movable(swinger) and not on_cell:
                    others = (() if previous is None or previous[3] is None
                              or swinger["time"] - previous[0] <= gap else (previous[3],))
                    candidates = [d for d in range(8)
                                  if _flow_legal(d, previous, swinger["x"], swinger["y"], swinger["time"], gap)
                                  and _swing_clear((swinger["x"], swinger["y"]),
                                                   other_cells.get(swinger["id"]), d)]
                    new = _rank_flow_candidates(swinger["d"], candidates, others)
                    if new != 8:
                        swinger["d"] = new
                        counters["swing_recut"] += 1
                        changed += 1
                        resolved = True
                if not resolved:
                    for mover in (target, swinger):
                        if not movable(mover):
                            continue
                        cells = _relocation_cells(mover, ordered, eps, window, incoming, dropped)
                        if cells:
                            mover["x"], mover["y"] = cells[0]
                            counters["swing_relocated"] += 1
                            changed += 1
                            resolved = True
                            break
                if not resolved:
                    drop(min((swinger, target), key=lambda n: (not movable(n), n is swinger)))
                    counters["swing_dropped"] += 1
                break
    if dropped:
        notes[:] = [note for note in notes if note["id"] not in dropped]
    return changed


def clearance_violations(notes):
    """Pure count of remaining hard-rule violations: same-cell pairs, swing-into
    pairs and same-hand flow repeats."""
    eps = float(getattr(config, "min_time_diff", .02))
    window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    pred, incoming = _history_context(ordered, eps)
    same_cell = sum(1 for first, second in _simultaneous_pairs(ordered, eps)
                    if (first["x"], first["y"]) == (second["x"], second["y"]))
    partners = {note["id"]: [] for note in ordered}
    for first, second in _simultaneous_pairs(ordered, window):
        if first["hand"] != second["hand"]:
            partners[first["id"]].append(second)
            if second["time"] - first["time"] <= eps:
                partners[second["id"]].append(first)
    swing_into = 0
    for swinger in ordered:
        direction = swinger["d"] if swinger["d"] != 8 else incoming.get(swinger["id"])
        swept = set(_swept_cells(swinger["x"], swinger["y"], direction)) \
            if direction is not None and direction != 8 else set()
        for target in partners[swinger["id"]]:
            position = (target["x"], target["y"])
            swing_into += int(position in swept
                              or (position == (swinger["x"], swinger["y"])
                                  and target["time"] - swinger["time"] > eps))
    repeats = sum(1 for note in ordered
                  if note["d"] != 8 and not _flow_legal(note["d"], pred.get(note["id"]),
                                                      note["x"], note["y"], note["time"], gap))
    return {"same_cell_pairs": same_cell, "swing_into_pairs": swing_into, "flow_repeats": repeats}


def enforce_note_flow_and_clearance(notes, chains=None, protect_retained=True):
    """Fixpoint of the ≥135° recut and swing-clearance.

    A swing at ``t`` sweeps ``pos + step → pos → pos - step``; an other-hand
    note standing on any of those cells at that instant — same time, or
    struck up to ``v2_cleanup_swing_window`` seconds later — is a clash.
    Both passes can feed back into each other (a flow recut may point at an
    occupied swept cell; a relocation changes travel distances), so they repeat
    until stable, bounded at eight rounds. Retained notes are never touched
    while any alternative fix exists when ``protect_retained`` is set.
    Returns a stats dict including residual violation counts."""
    stats = {"flow_recuts": 0, "cell_relocated": 0, "cell_dropped": 0,
             "swing_recut": 0, "swing_relocated": 0, "swing_dropped": 0}
    flow = float(getattr(config, "v2_cleanup_same_dir_recut", 0.) or 0.) > 0
    check_cells = bool(getattr(config, "v2_cleanup_cell_clash", True))
    check_swing = bool(getattr(config, "v2_cleanup_swing_clash", True))
    for _ in range(8):
        changed = 0
        if flow:
            flips = recut_same_direction_notes(notes, chains, protect_retained)
            stats["flow_recuts"] += flips
            changed += flips
        if check_cells or check_swing:
            counters = {"dropped": 0, "cell_dropped": 0, "cell_relocated": 0,
                        "swing_recut": 0, "swing_relocated": 0, "swing_dropped": 0}
            changed += _resolve_clashes(notes, protect_retained, check_cells, check_swing, counters)
            for key in ("cell_dropped", "cell_relocated", "swing_recut", "swing_relocated", "swing_dropped"):
                stats[key] += counters[key]
        if not changed:
            break
    anchors = {note["id"]: note for note in notes}
    for chain in chains or []:
        anchor = anchors.get(chain.get("anchor_id"))
        if anchor is not None:
            chain["x"], chain["y"], chain["d"] = anchor["x"], anchor["y"], anchor["d"]
    stats["residual"] = clearance_violations(notes)
    return stats


def _detect_onsets(audio_path):
    """Detected onset peaks in seconds — same librosa recipe as the legacy snap."""
    import librosa
    factor = config.thresh_onbeat / config.thresh_onbeat_orig
    frames = lambda count: max(1, int(count * factor))
    samples, rate = librosa.load(str(audio_path), sr=44100)
    envelope = librosa.onset.onset_strength(y=samples, sr=rate, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=envelope, sr=rate, hop_length=512,
                                        backtrack=False, pre_max=frames(5), post_max=frames(5),
                                        pre_avg=frames(7), post_avg=frames(8),
                                        delta=config.thresh_onbeat,
                                        wait=frames(int(getattr(config, "v2_notes_onset_snap_wait", 4))))
    return np.asarray(librosa.frames_to_time(onsets, sr=rate, hop_length=512), np.float64)


def _v1_beat_times(audio_path):
    """Beat times (seconds) predicted by the trained v1 beat-TCN, inference only.

    The bundle lives at ``model/<v2_gap_fill_v1_bundle>/``; ``scoped_settings``
    restores the temporary ``paths.model_path`` override afterwards so v1
    config/path state never leaks into the v2 run."""
    from pathlib import Path

    from evaluation.legacy_generator import beat_proposals
    from map_creation.v2_generation import scoped_settings
    from tools.config import paths

    name = Path(audio_path).stem
    with scoped_settings():
        paths.model_path = paths.dir_path + f"model/{config.v2_gap_fill_v1_bundle}/"
        return np.asarray(beat_proposals([name])["timing_ar"], np.float64)


def _gap_candidate_times(source, audio_path, bpm, duration, used=None):
    """Insertion-time pool for the max-gap safety net, selected by ``source``.

    ``"onsets"`` reuses the librosa spectral-flux peaks and ``"bpm_grid"`` the
    whole-beat lattice; ``"v1_beat"`` runs the trained v1 beat-TCN and falls
    back to detected onsets when its bundle is missing or inference fails.
    ``used``, when given, receives the actually decoded source as
    ``used["source"]`` so callers can record the fallback."""
    if source == "onsets":
        return _detect_onsets(audio_path)
    if source == "bpm_grid":
        step = 60. / bpm
        return np.arange(step, duration, step)
    if source == "v1_beat":
        try:
            return _v1_beat_times(audio_path)
        except Exception as error:
            print(f"Warning: v1_beat gap fill unavailable ({error}); using detected onsets")
            if used is not None:
                used["source"] = "onsets"
            return _detect_onsets(audio_path)
    raise ValueError(f"Unknown v2_gap_fill_source: {source!r}")


def _snap_violated(note, t_new, near, siblings, index, effective, eps, window, gap):
    """Hard-clearance check for ``note`` moved to ``t_new``.

    ``near`` carries every other note inside the swing window at their current
    times; ``siblings``/``index`` describe the same-hand sequence for the flow
    pair checks. Mirrors ``clearance_violations``: same-cell simultaneity,
    both swing directions and same-hand repeats.
    """
    cell = (note["x"], note["y"])
    own_dir = effective.get(note["id"])
    swept = lambda other: (set() if effective.get(other["id"]) is None
                           else set(_swept_cells(other["x"], other["y"], effective[other["id"]])))
    own = set() if own_dir is None else set(_swept_cells(note["x"], note["y"], own_dir))
    for other in near:
        dt = other["time"] - t_new
        if abs(dt) > window:
            continue
        occupied = (other["x"], other["y"]) == cell
        if abs(dt) <= eps:
            if occupied:
                return True
            if other["hand"] != note["hand"] and (cell in swept(other) or (other["x"], other["y"]) in own):
                return True
        elif other["hand"] != note["hand"]:
            clash = cell in swept(other) if dt < 0 else (other["x"], other["y"]) in own
            if clash or occupied:
                return True
    if note["d"] != 8:
        previous = next((siblings[i] for i in range(index - 1, -1, -1)
                         if siblings[i]["time"] < t_new - eps), None)
        if previous is not None and not _flow_legal(
                note["d"], (previous["time"], previous["x"], previous["y"], effective.get(previous["id"])),
                note["x"], note["y"], t_new, gap):
            return True
    following = next((siblings[i] for i in range(index + 1, len(siblings))
                      if siblings[i]["time"] > t_new + eps), None)
    if following is not None and following["d"] != 8 and not _flow_legal(
            following["d"], (t_new, note["x"], note["y"], own_dir),
            following["x"], following["y"], following["time"], gap):
        return True
    return False


def snap_note_times(notes, chains, audio_path, duration, ratio=None, logger=None):
    """Snap each note onto its nearest detected onset within a bounded window.

    The window is ``ratio`` of the note's own same-hand inter-onset interval
    (the smaller gap to its same-hand neighbours — per-note, per-hand), so a
    note can never reach another note's clearance zone. A candidate move is
    applied only when ``_snap_violated`` finds no same-cell, swing or
    same-hand-flow conflict — the same rules the clearance fixpoint resolves.
    Chain spans move with their anchor note. Returns diagnostics."""
    stats = {"notes": len(notes), "moved": 0, "blocked": 0, "no_onset": 0,
             "mean_shift_ms": 0., "max_shift_ms": 0.}
    if audio_path is None or not notes:
        return stats
    ratio = float(getattr(config, "v2_notes_onset_snap_ioi_ratio", 0.15) if ratio is None else ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        return stats
    onsets = _detect_onsets(audio_path)
    if not len(onsets):
        return stats
    from bisect import bisect_left, insort
    eps = float(getattr(config, "min_time_diff", .02))
    window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    _, effective = _history_context(ordered, eps)
    per_hand = {hand: [note for note in ordered if note["hand"] == hand] for hand in (0, 1)}
    index = {note["id"]: i for siblings in per_hand.values() for i, note in enumerate(siblings)}
    live = sorted((note["time"], note["id"]) for note in ordered)
    lookup = {note["id"]: note for note in ordered}
    anchored = {note["id"]: [chain for chain in chains if chain.get("anchor_id") == note["id"]]
                for note in ordered}
    shifts = []
    for note in ordered:
        siblings, i = per_hand[note["hand"]], index[note["id"]]
        ioi = min(note["time"] - siblings[i - 1]["time"] if i else math.inf,
                  siblings[i + 1]["time"] - note["time"] if i + 1 < len(siblings) else math.inf)
        if not np.isfinite(ioi) or ioi <= 0:
            continue
        bound = ratio * ioi
        near_onsets = onsets[np.abs(onsets - note["time"]) <= bound + 1e-9]
        if not len(near_onsets):
            stats["no_onset"] += 1
            continue
        t_new = float(near_onsets[np.argmin(np.abs(near_onsets - note["time"]))])
        if t_new == note["time"]:
            continue
        lo = bisect_left(live, (t_new - window - 1e-9, -1))
        hi = bisect_left(live, (t_new + window + 1e-9, math.inf))
        near = [lookup[other] for _, other in live[lo:hi] if other != note["id"]]
        delta = t_new - note["time"]
        if any(chain["end_time"] + delta >= duration for chain in anchored[note["id"]]) \
                or _snap_violated(note, t_new, near, siblings, i, effective, eps, window, gap):
            stats["blocked"] += 1
            continue
        live.remove((note["time"], note["id"]))
        insort(live, (t_new, note["id"]))
        note["time"] = t_new
        for chain in anchored[note["id"]]:
            chain["time"] += delta
            chain["end_time"] += delta
        shifts.append(delta)
        stats["moved"] += 1
    if shifts:
        stats["mean_shift_ms"] = float(np.mean(np.abs(shifts)) * 1e3)
        stats["max_shift_ms"] = float(np.max(np.abs(shifts)) * 1e3)
    _report(logger, f"Onset snap: {stats['moved']} notes moved, "
                    f"{stats['blocked']} blocked, {stats['no_onset']} without onset in range")
    return stats


def apply_intro_floor(notes, cutoff_seconds):
    """Drop notes starting before ``cutoff_seconds`` in place; return the count."""
    before = len(notes)
    notes[:] = [note for note in notes if note["time"] >= cutoff_seconds]
    return before - len(notes)


# Diagonal direction → its two adjacent cardinals; recuts stay inside the
# narrowest direction change so the fix reshapes the model's cut as little as
# possible.
_DIAGONAL_CARDINALS = {4: (0, 2), 5: (0, 3), 6: (1, 2), 7: (1, 3)}


def apply_diagonal_cap(notes, chains=None, rng=None):
    """Cap diagonal cuts (d 4-7) at ``v2_diagonal_max_fraction`` of all notes.

    ``excess`` diagonals — ``len(diagonals) - int(cap * len(notes))``, dots
    included in the base count — are drawn from ``rng`` (a numpy Generator, so
    a generation seed reproduces the selection) and recut to an adjacent
    cardinal. A candidate must pass ``_flow_legal`` against both same-hand
    neighbours and ``_swing_clear`` against simultaneous other-hand cells;
    neighbours are read with their *current* direction, so chosen notes are
    processed in time order and earlier recuts constrain later ones (dots use
    their ``incoming`` implied direction, except a dot successor whose implied
    direction follows this note and is recomputed per candidate). The cheapest
    legal cut by the previous cut's exit-cell → candidate approach-cell
    Manhattan distance wins (no predecessor leaves all costs equal); ties
    prefer the vertical cardinal. When no cardinal is legal the diagonal stays —
    never dropped to satisfy a style metric. Chain heads follow their
    anchor's direction like ``recut_same_direction_notes``. Returns run
    statistics; ``>= 1`` disables the cap (negative clamps to 0)."""
    rng = np.random.default_rng() if rng is None else rng
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    diagonals = [note for note in ordered if note["d"] in (4, 5, 6, 7)]
    stats = {"diagonal_before": len(diagonals), "diagonal_after": len(diagonals),
             "recut": 0, "kept_over_cap": 0,
             "fraction_after": len(diagonals) / len(ordered) if ordered else 0.}
    cap = max(float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.), 0.)
    excess = max(0, len(diagonals) - int(cap * len(notes)))
    if ordered and cap < 1 and excess:
        eps = float(getattr(config, "min_time_diff", .02))
        window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
        gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
        _, incoming = _history_context(ordered, eps)
        other_cells = _other_hand_cells(ordered, eps, window)
        per_hand = {}
        for note in ordered:
            per_hand.setdefault(note["hand"], []).append(note)
        position = {note["id"]: index for siblings in per_hand.values()
                    for index, note in enumerate(siblings)}

        def effective(other):
            return other["d"] if other["d"] != 8 else incoming.get(other["id"])

        chosen = sorted((diagonals[index]
                         for index in rng.choice(len(diagonals), size=excess, replace=False)),
                        key=lambda item: (item["time"], item["id"]))
        for note in chosen:
            siblings, index = per_hand[note["hand"]], position[note["id"]]
            time, x, y = note["time"], note["x"], note["y"]
            before = index - 1
            while before >= 0 and siblings[before]["time"] >= time - eps:
                before -= 1
            previous = siblings[before] if before >= 0 else None
            prior = ((previous["time"], previous["x"], previous["y"], effective(previous))
                     if previous is not None else None)
            after = index + 1
            while after < len(siblings) and siblings[after]["time"] <= time + eps:
                after += 1
            successor = siblings[after] if after < len(siblings) else None

            def cost(candidate):
                # Spec metric (todo_after_playtest.md case 3): cells the hand
                # must travel from the previous same-hand cut's exit cell
                # (``pos - step``; a dot exits through its implied direction's
                # cell, or its own cell when no direction is implied) to this
                # candidate's approach cell (``pos + step``). No predecessor
                # leaves every candidate at the same cost.
                if previous is None:
                    return 0
                implied = prior[3]
                if implied is None:
                    exit_x, exit_y = previous["x"], previous["y"]
                else:
                    pstep = _DIRECTION_STEP[implied]
                    exit_x, exit_y = previous["x"] - pstep[0], previous["y"] - pstep[1]
                step = _DIRECTION_STEP[candidate]
                return abs(exit_x - (x + step[0])) + abs(exit_y - (y + step[1]))

            def legal(candidate):
                if not _flow_legal(candidate, prior, x, y, time, gap):
                    return False
                if successor is not None:
                    if successor["d"] != 8:
                        next_d = successor["d"]
                    else:
                        # the dot's implied direction follows the last note of
                        # this note's simultaneous group — usually this note
                        earlier = siblings[after - 1]
                        base = candidate if earlier is note else effective(earlier)
                        next_d = _dot_virtual_direction(base, earlier["x"], earlier["y"],
                                                        successor["x"], successor["y"])
                    if next_d is not None and not _flow_legal(
                            next_d, (time, x, y, candidate),
                            successor["x"], successor["y"], successor["time"], gap):
                        return False
                return _swing_clear((x, y), other_cells.get(note["id"]), candidate)

            candidates = [candidate for candidate in _DIAGONAL_CARDINALS[note["d"]] if legal(candidate)]
            if not candidates:
                stats["kept_over_cap"] += 1
                continue
            note["d"] = min(candidates, key=lambda candidate: (cost(candidate), candidate not in (0, 1)))
            stats["recut"] += 1
        stats["diagonal_after"] = sum(note["d"] in (4, 5, 6, 7) for note in ordered)
        stats["fraction_after"] = stats["diagonal_after"] / len(ordered)
    anchors = {note["id"]: note for note in notes}
    for chain in chains or []:
        anchor = anchors.get(chain.get("anchor_id"))
        if anchor is not None:
            chain["d"] = anchor["d"]
    return stats


_NOTE_STEPS = ("mirroring", "initial_single_note_filter", "legacy_note_flow",
               "flow_additions", "same_direction_recut", "final_single_note_filter",
               "clearance")

# Legacy v1 note mutators that may be bypassed in production: the controlled
# comparisons in evaluation/evaluate_v2_postprocessing.py showed the flow pass
# culls ~half of sampled model notes (realized NPS ~0.5x requested) while the
# structural pass plus the clearance fixpoint already reach zero residual
# violations. ``flow_additions`` only exists inside the legacy flow pass.
LEGACY_NOTE_MUTATORS = ("mirroring", "legacy_note_flow")


def postprocess_notes(notes, event_times, logger=None, skip=()):
    """Run the shared legacy flow pass, then establish stable retained anchors.

    ``skip`` names steps from ``_NOTE_STEPS`` to omit for controlled
    comparisons: mirroring, legacy_note_flow and the in-flow adders
    (``flow_additions``: emphasis, breaks, added dots) are legacy heuristics;
    the same-direction recut and the clearance fixpoint are invariant fixes;
    dedup/retained-anchor labeling is structural and always runs.
    """
    unknown = set(skip) - set(_NOTE_STEPS)
    if unknown:
        raise ValueError(f"Unknown note postprocessing steps: {sorted(unknown)}")
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
                    "operations": operations, "skipped_steps": sorted(skip),
                    "pre_clearance": {"same_cell_pairs": 0, "swing_into_pairs": 0, "flow_repeats": 0},
                    "note_clearance": None, "same_direction_recut_notes": None,
                    "same_direction_recut_stream_notes": 0, **differences([])}
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

    if config.gimme_more_notes_flag and "mirroring" not in skip:
        apply_operation("mirroring", _v2_mirroring)
    if config.single_notes_only_flag and "initial_single_note_filter" not in skip:
        apply_operation("initial_single_note_filter", remove_double_notes)
    def relaxed_flow(values):
        # Sampled v2 model output already carries human-like variety; the
        # legacy homogenizers are relaxed only for this call, leaving the v1
        # generator's cleanup behavior unchanged.
        overrides = {"allow_dot_notes": getattr(config, "v2_cleanup_allow_dots", config.allow_dot_notes),
                     "flow_model_flag": getattr(config, "v2_cleanup_flow_recut", config.flow_model_flag),
                     "shift_blocks_left_right_flag": getattr(config, "v2_cleanup_lane_shift",
                                                           config.shift_blocks_left_right_flag),
                     "sanity_direction_recut_flag": getattr(config, "v2_cleanup_sanity_recut",
                                                          config.sanity_direction_recut_flag),
                     "add_dot_notes": getattr(config, "v2_cleanup_add_dots", config.add_dot_notes),
                     "shift_blocks_middle_flag": getattr(config, "v2_cleanup_middle_shift",
                                                         config.shift_blocks_middle_flag),
                     "shift_blocks_up_down_flag": getattr(config, "v2_cleanup_updown_shift",
                                                          config.shift_blocks_up_down_flag),
                     "correct_notes_all_flag": getattr(config, "v2_cleanup_cross_hand_adjust",
                                                       config.correct_notes_all_flag)}
        if "flow_additions" in skip:
            overrides.update({"emphasize_beats_flag": False, "add_breaks_flag": False,
                              "add_dot_notes": 0})
        previous = {key: getattr(config, key) for key in overrides}
        try:
            for key, value in overrides.items():
                setattr(config, key, value)
            return sanity_check_notes(values, times)
        finally:
            for key, value in previous.items():
                setattr(config, key, value)

    if "legacy_note_flow" not in skip:
        apply_operation("legacy_note_flow", relaxed_flow)
    flat_flips = {"n": 0}
    if float(getattr(config, "v2_cleanup_same_dir_recut", 0.) or 0.) > 0 and "same_direction_recut" not in skip:
        gap = float(getattr(config, "v2_cleanup_same_dir_gap", 0.1) or 0.)
        apply_operation("same_direction_recut",
                        lambda values: _recut_same_direction_repeats(values, times, gap, flat_flips))
    # Emphasis and mirroring must not undo the user's single-note selection.
    if config.single_notes_only_flag and "final_single_note_filter" not in skip:
        apply_operation("final_single_note_filter", remove_double_notes)
    structured, source = [], []
    last_kept, same_stroke_gap = {}, float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    for event, (time, values) in enumerate(zip(times, flattened)):
        per_hand = {}
        for offset in range(0, len(values), 4):
            x, y, hand, direction = map(int, values[offset:offset + 4])
            if not (0 <= x < 4 and 0 <= y < 3 and hand in (0, 1) and 0 <= direction <= 8):
                raise ValueError("Legacy note cleanup produced invalid colored-note geometry")
            per_hand.setdefault(hand, []).append((x, y, direction))
        for hand in sorted(per_hand):
            options = per_hand[hand]
            previous = last_kept.get(hand)
            # with same-hand candidates, keep the one that flows best against
            # the predecessor instead of always the first surviving cleanup
            if len(options) > 1 and previous is not None and previous[3] is not None \
                    and time - previous[0] > same_stroke_gap:
                def rank(option):
                    od = option[2]
                    if od == 8:
                        # disallowed dots rank below legal cuts but above illegal
                        # repeats; the survivor is recut to a direction below.
                        return (0, 0) if config.allow_dot_notes else (1, 0)
                    steps = _step_distance(od, previous[3])
                    illegal = not _flow_legal(option[2], previous, option[0], option[1],
                                              time, same_stroke_gap)
                    return illegal, -steps
                x, y, direction = min(enumerate(options), key=lambda item: (*rank(item[1]), item[0]))[1]
            else:
                x, y, direction = options[0]
            effective = direction if direction != 8 else (
                _dot_virtual_direction(previous[3], previous[1], previous[2], x, y)
                if previous is not None and previous[3] is not None else None)
            if direction == 8 and not config.allow_dot_notes:
                # Model dots need a real cut direction: the geometry-implied
                # one, or the legacy first-note vertical fallback.
                direction = effective if effective is not None else \
                    (0 if y >= config.first_note_layer_threshold else 1)
                effective = direction
            last_kept[hand] = (time, x, y, effective)
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
    # Corrections-needed measure: violations the fixpoint receives per arm.
    pre_clearance = clearance_violations(structured) if structured else \
        {"same_cell_pairs": 0, "swing_into_pairs": 0, "flow_repeats": 0}
    clearance = (enforce_note_flow_and_clearance(structured, protect_retained=False)
                 if "clearance" not in skip else None)
    diagnostics = {"input_notes": len(notes), "retained_notes": retained,
                   "same_direction_recut_notes": clearance["flow_recuts"] if clearance else None,
                   "same_direction_recut_stream_notes": flat_flips["n"],
                   "pre_clearance": pre_clearance,
                   "note_clearance": clearance,
                   "skipped_steps": sorted(skip),
                   "heuristic_notes": len(structured) - retained, "notes": len(structured),
                   "ids_assigned_after_cleanup": True, "times_changed": False,
                   "operations": operations, **differences(structured)}
    _report(logger, f"Note flow: {len(notes)} -> {len(structured)} notes ({retained} retained anchors)")
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


def _within_cell_gap(ax, ay, bx, by, cell_gap):
    """Chebyshev cell margin; ``cell_gap`` 0 is same-cell, 1 adds all 8 neighbours."""
    return abs(ax - bx) < cell_gap + .5 and abs(ay - by) < cell_gap + .5


def _same_cell(first, second):
    return _within_cell_gap(first["x"], first["y"], second["x"], second["y"], 0)


def _bomb_safe_cells(occupied_window, cell_gap):
    """Grid cells staying ``cell_gap`` rings clear of every windowed swing."""
    free = {(x, y) for x in range(4) for y in range(3)}
    for note in occupied_window:
        for cell in _swing_cells(note):
            free = {item for item in free
                    if not _within_cell_gap(item[0], item[1], cell["x"], cell["y"], cell_gap)}
        if not free:
            break
    return free


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
    """Filter model hazards with legacy margins, geometry modes and clearance.

    ``v2_map_intro_no_hazards_seconds`` additionally keeps the song opening
    free of walls and bombs (generated and calm-effect hazards alike); those
    rejections are reported under ``rejected_intro``."""
    duration = float(_base(song).metadata["duration"])
    occupied = sorted(_occupied_notes(state), key=lambda note: note["time"])
    times = np.asarray([note["time"] for note in occupied])
    before, after = map(float, config.obstacle_time_gap)
    intro_floor = float(getattr(config, "v2_map_intro_no_hazards_seconds", 0.) or 0.)
    rejected_intro = 0
    positions = config.sport_obstacle_positions if config.sporty_obstacles else config.norm_obstacle_positions
    allowed_lanes = {lane for side in positions for lane in side}
    allowed_types = config.sport_obstacle_allowed_types if config.sporty_obstacles else config.norm_obstacle_allowed_types
    walls = []
    original_walls, original_bombs = list(state.get("walls", [])), list(state.get("bombs", []))
    for wall in sorted(original_walls, key=lambda item: item["time"]):
        start, end = wall["time"], wall["end_time"]
        if (not config.add_obstacle_flag or not 0 <= start < end <= duration
                or end - start < config.obstacle_min_duration or start < intro_floor):
            rejected_intro += int(start < intro_floor)
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
        if any(_wall_contains(wall, cell) for note in occupied[lo:hi]
               for cell in _swing_cells(note)):
            continue
        walls.append(wall)
    bombs = []
    radius = max(float(getattr(config, "v2_bomb_note_time_gap", .8) or 0.), config.min_time_diff)
    cell_gap = max(float(getattr(config, "v2_bomb_note_cell_gap", 1) or 0.), 0.)
    for bomb in sorted(original_bombs, key=lambda item: item["time"]):
        if not config.add_bombs_flag or not _integer_geometry(bomb, ("x", "y")):
            continue
        if (not (0 <= bomb["time"] < duration and 0 <= bomb["x"] < 4 and 0 <= bomb["y"] < 3)
                or bomb["time"] < intro_floor):
            rejected_intro += int(bomb["time"] < intro_floor)
            continue
        lo = int(np.searchsorted(times, bomb["time"] - radius, side="left"))
        hi = int(np.searchsorted(times, bomb["time"] + radius, side="right"))
        if any(_within_cell_gap(bomb["x"], bomb["y"], cell["x"], cell["y"], cell_gap)
               for note in occupied[lo:hi] for cell in _swing_cells(note)):
            continue
        if any(wall["time"] <= bomb["time"] <= wall["end_time"] and _wall_contains(wall, bomb) for wall in walls):
            continue
        if any(abs(bomb["time"] - other["time"]) <= config.min_time_diff and _same_cell(bomb, other) for other in bombs):
            continue
        bombs.append(bomb)
    state.update(walls=walls, bombs=bombs)
    diagnostics = {"rejected_walls": len(original_walls) - len(walls),
                   "rejected_bombs": len(original_bombs) - len(bombs),
                   "rejected_intro": rejected_intro}
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
    synthetic = 0
    for arc in original:
        if not config.add_slider_flag:
            continue
        if arc.get("_legacy_clearance_checked"):
            accepted.append(arc)
            continue
        ids = arc.get("endpoint_ids", [])
        head = arc.get("synthetic_head")
        if head is not None:
            # Guidance arc into the first note: skips gap/movement/probability
            # like the legacy turbo start but keeps occupancy clearance.
            if not _synthetic_arc_ok(arc, head, ids, notes, occupied, state):
                continue
            arc["_legacy_clearance_checked"] = True
            synthetic += 1
            accepted.append(arc)
            continue
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
    diagnostics = {"rejected_arcs": len(original) - len(accepted), "synthetic_start_arc": synthetic,
                   "clearance_geometry": "sampled_endpoint_segment"}
    _report(logger, f"Arc clearance: {len(accepted)} existing-endpoint arcs accepted")
    return diagnostics


def _synthetic_arc_ok(arc, head, ids, notes, occupied, state):
    """Validate a turbo-start record; geometry stays arc data, never a note."""
    if (len(ids) != 1 or ids[0] not in notes
            or not _integer_geometry(head, ("x", "y", "d"))
            or not (0 <= head["x"] < 4 and 0 <= head["y"] < 3 and 0 <= head["d"] < 9)):
        return False
    tail = notes[ids[0]]
    end = arc.get("end_time", -1)
    if (tail["hand"] != arc.get("hand") or tail["time"] != end
            or not 0 <= arc.get("time", -1) < end
            or head["x"] != tail["x"] or head["y"] != tail["y"] or head["d"] != tail["d"]):
        return False
    if any(chain["hand"] == arc["hand"] and chain["time"] < arc["end_time"] and chain["end_time"] > arc["time"]
           for chain in state.get("chains", [])):
        return False
    gap = arc["end_time"] - arc["time"]
    steps = max(1, int(math.ceil(gap / .01)))
    samples = [{"time": arc["time"] + alpha * gap, "x": head["x"], "y": head["y"]}
               for alpha in np.linspace(0, 1, steps + 1)]
    if any(abs(sample["time"] - bomb["time"]) <= .1 and _same_cell(sample, bomb)
           for sample in samples for bomb in state.get("bombs", [])):
        return False
    if any(wall["time"] - .1 <= sample["time"] <= wall["end_time"] + .1 and _wall_contains(wall, sample)
           for sample in samples for wall in state.get("walls", [])):
        return False
    neighbors = [note for note in occupied if note.get("id") not in ids
                 and arc["time"] - config.min_time_diff <= note["time"] <= arc["end_time"] + config.min_time_diff]
    if any(abs(sample["time"] - note["time"]) <= config.min_time_diff and _same_cell(sample, note)
           for sample in samples for note in neighbors):
        return False
    return True


def _arc_sample_points(state):
    """Sampled (time, x, y) positions along accepted arcs.

    Straight endpoint-segment sampling — the same convention ``validate_arcs``
    uses for its bomb/wall/note clearance, so decorative objects skip the same
    volume arcs already reserve.
    """
    notes = {note["id"]: note for note in state.get("notes", [])}
    points = []
    for arc in state.get("arcs", []):
        ids = arc.get("endpoint_ids", [])
        head = arc.get("synthetic_head")
        if head is not None:
            if len(ids) != 1 or ids[0] not in notes:
                continue
            tail, head_pos = notes[ids[0]], head
        else:
            if len(ids) != 2 or any(identity not in notes for identity in ids):
                continue
            head_pos, tail = (notes[identity] for identity in ids)
        gap = arc["end_time"] - arc["time"]
        steps = max(1, int(math.ceil(gap / .01)))
        for alpha in np.linspace(0, 1, steps + 1):
            points.append({"time": arc["time"] + alpha * gap,
                           "x": head_pos["x"] + alpha * (tail["x"] - head_pos["x"]),
                           "y": head_pos["y"] + alpha * (tail["y"] - head_pos["y"])})
    return points


def calm_area_pulses(state, song):
    """Beat-grid pulse times inside note-free stretches.

    A gap between consecutive gameplay objects — notes as points, chains as
    their full span — qualifies when its note-free span reaches
    ``v2_calm_effects_min_gap_seconds``. ``v2_calm_effects_edge_seconds`` keeps
    pulses clear of the boundary objects so real silence keeps its empty
    space. Pulse times sit on the bpm beat grid anchored at zero (the same
    grid ``align_beats_on_bpm`` and the export factor use) with
    ``v2_calm_effects_division`` pulses per beat. Returns (pulses,
    diagnostics); each pulse carries its per-gap ``index`` for alternating
    placement.
    """
    base = _base(song)
    duration, bpm = float(base.metadata["duration"]), float(base.metadata["bpm"])
    min_gap = max(float(getattr(config, "v2_calm_effects_min_gap_seconds", 4.) or 0.), 0.)
    edge = max(float(getattr(config, "v2_calm_effects_edge_seconds", 1.) or 0.), 0.)
    start_seconds = max(float(getattr(config, "v2_calm_effects_start_seconds", 1.5) or 0.), 0.)
    division = max(int(getattr(config, "v2_calm_effects_division", 1) or 1), 1)
    diagnostics = {"min_gap_seconds": min_gap, "edge_seconds": edge, "division": division,
                   "start_seconds": start_seconds, "bpm": bpm, "gaps": 0, "pulses": 0}
    if not state.get("notes") or not np.isfinite([duration, bpm]).all() or bpm <= 0:
        return [], diagnostics
    occupied = [[float(note["time"]), float(note["time"])] for note in state["notes"]]
    occupied += [[float(chain["time"]), float(chain["end_time"])] for chain in state.get("chains", [])]
    occupied.sort()
    merged = []
    for start, end in occupied:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    intervals, cursor, touched = [], 0., False
    for start, end in merged:
        intervals.append((cursor, start, touched))
        cursor, touched = max(cursor, end), True
    intervals.append((cursor, duration, touched))
    step = 60. / bpm / division
    pulses = []
    for low, high, occupied_low in intervals:
        if high - low < min_gap - 1e-9:
            continue
        interior = (max(low + edge if occupied_low else low, start_seconds),
                    high - edge if high < duration else high)
        grid = np.arange(math.ceil(interior[0] / step - 1e-9) * step, interior[1], step)
        grid = grid[grid >= interior[0] - 1e-9]
        for index, time in enumerate(grid):
            pulses.append({"time": float(time), "index": index, "gap": diagnostics["gaps"]})
        diagnostics["gaps"] += 1
    diagnostics["pulses"] = len(pulses)
    return pulses, diagnostics


# Per-gap placement themes; each maps a pulse's within-gap index to a
# (wall_lanes, bomb_lane, flash_types, top_slab) motif or ``None`` for a rest
# pulse. Deterministic rotation keeps regeneration reproducible while each
# calm gap gets its own look.
_CALM_THEMES = ("alternate", "pairs", "stagger", "phrase", "rest4", "sparse", "twin")


def _calm_theme(gap_index):
    themes = _CALM_THEMES if getattr(config, "v2_calm_effects_variety", True) else _CALM_THEMES[:1]
    return themes[gap_index % len(themes)]


def _calm_motif(theme, index):
    """Resolve one pulse to (wall_lanes, bomb_lane, flash_types, top_slab) or None.

    ``alternate`` mirrors sides every pulse; ``pairs`` and ``phrase`` hold a
    side for two or four pulses; ``stagger`` steps between full pillars and
    top slabs; ``rest4`` breathes every fourth pulse; ``sparse`` only plays
    even indices; ``twin`` flashes both edge lanes on every fourth pulse as an
    accent frame. All placements stay on the safe edge lanes/bottom corners.
    """
    side = index % 2
    if theme == "pairs":
        side = (index // 2) % 2
    elif theme == "phrase":
        side = (index // 4) % 2
    elif theme == "sparse":
        if index % 2:
            return None
        side = (index // 2) % 2
    elif theme == "rest4" and index % 4 == 3:
        return None
    accent = theme == "twin" and index % 4 == 0
    return ((0, 3) if accent else (3 * side,),
            None if accent else 3 - 3 * side,
            (2, 3) if accent else (2 + side,),
            theme == "stagger" and index % 2 == 1)


def add_calm_effects(state, song, pulses):
    """Place decorative edge-lane wall flashes and bottom-corner bombs.

    Each gap resolves its ``_calm_theme`` motif per pulse — side patterns,
    rest beats, twin accents and pillar/slab steps — while bombs keep the
    bottom corner opposite the wall. Elements skip cells an accepted arc
    already sweeps through; ``validate_hazards`` then applies the legacy
    margins, allowed types/positions and note clearance. Placement is visual
    only — no element forces movement.
    """
    counts = {"walls": 0, "bombs": 0}
    if not pulses:
        return counts
    base = _base(song)
    duration, bpm = float(base.metadata["duration"]), float(base.metadata["bpm"])
    # quarter-beat flashes with a micro-epsilon over the validator's minimum
    # duration — float subtraction would otherwise dip just below it.
    wall_seconds = max(float(getattr(config, "v2_calm_effects_wall_beats", .25)) * 60. / bpm,
                       config.obstacle_min_duration + 1e-6)
    want_walls = bool(getattr(config, "v2_calm_effects_walls", True)) and config.add_obstacle_flag
    want_bombs = bool(getattr(config, "v2_calm_effects_bombs", True))
    arcs = _arc_sample_points(state) if (want_walls or want_bombs) else []
    themes = {}
    for pulse in pulses:
        theme = themes.setdefault(pulse["gap"], _calm_theme(pulse["gap"]))
        motif = _calm_motif(theme, pulse["index"])
        if motif is None:
            continue
        wall_lanes, bomb_lane, _, slab = motif
        time = pulse["time"]
        if want_walls:
            for lane in wall_lanes:
                wall = {"kind": "wall", "x": lane, "y": 2 if slab else 0, "width": 1,
                        "height": 1 if slab else 3,
                        "time": time, "end_time": min(time + wall_seconds, duration)}
                if not any(wall["time"] - .1 <= point["time"] <= wall["end_time"] + .1
                           and _wall_contains(wall, point) for point in arcs):
                    state["walls"].append(wall)
                    counts["walls"] += 1
        if want_bombs and bomb_lane is not None:
            bomb = {"time": time, "x": bomb_lane, "y": 0}
            if not any(abs(point["time"] - time) <= .1 and _same_cell(point, bomb)
                       for point in arcs):
                state["bombs"].append(bomb)
                counts["bombs"] += 1
    state["walls"].sort(key=lambda wall: (wall["time"], wall["x"]))
    state["bombs"].sort(key=lambda bomb: (bomb["time"], bomb["x"]))
    return counts


def add_calm_lighting(state, pulses):
    """Append laser flashes at calm pulses following each gap's motif.

    Existing model events keep priority: a pulse at a time that already has an
    event of the same type is skipped. Value 6 is a red flash — short by
    design so the space between beats stays empty. Rest pulses stay dark.
    """
    if not pulses or not getattr(config, "v2_calm_effects_lighting", True):
        return 0
    from bisect import insort
    eps = float(getattr(config, "min_time_diff", .02))
    events = state["lighting"]
    by_type = {}
    for event in events:
        by_type.setdefault(event["type"], []).append(event["time"])
    added, themes = 0, {}
    for pulse in pulses:
        theme = themes.setdefault(pulse["gap"], _calm_theme(pulse["gap"]))
        motif = _calm_motif(theme, pulse["index"])
        if motif is None:
            continue
        for event_type in motif[2]:
            times = by_type.setdefault(event_type, [])
            location = int(np.searchsorted(times, pulse["time"]))
            if location < len(times) and times[location] - pulse["time"] <= eps:
                continue
            if location and pulse["time"] - times[location - 1] <= eps:
                continue
            events.append({"time": pulse["time"], "type": event_type, "value": 6})
            insort(times, pulse["time"])
            added += 1
    events.sort(key=lambda event: event["time"])
    return added
