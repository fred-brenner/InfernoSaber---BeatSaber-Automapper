"""Legacy timing/flow adapters and conservative clearance for v2 objects.

All timestamps remain elapsed seconds. Legacy note mutators run once, before
stable note IDs and expansion relationships are assigned. Clearance resolves
same-time same-cell and swing-path conflicts; it prefers mutating heuristic or
copied notes and only touches retained model notes when nothing else resolves
the conflict. A start-of-map floor drops early notes (``apply_intro_floor``;
hazards get their own floor inside ``validate_hazards``), a fast-run direction
simplifier collapses chaotic same-hand bursts to ping-pong with optional
counter-direction recuts (``apply_fast_direction_rules``), and a best-effort
diagonal cap recuts excess diagonal cuts toward flow-legal adjacent cardinals
— all run after note cleanup and onset snap, before the retained snapshot.
A movement-speed cap (``apply_speed_cap``) runs once before the retained
snapshot, while every note is still mutable, and again after expansion and
the clearance fixpoint with retained notes frozen; the v1 gate it restores
lived inside the skipped legacy flow pass. A counter-direction sweep
(``apply_fast_counter_sweep``) then recuts non-retained notes expansion left
inside fast pairs it never saw.

Cut-direction convention: ``get_cut_dir_xy`` yields the approach vector — the
hand enters from ``pos + step`` and exits through ``pos - step``. Both adjacent
cells along the arrow axis belong to the swing volume and must stay free of
simultaneous other-hand notes (same occupancy model as ``calc_note_pos``).
"""

import bisect
import math
import random
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from tools.config import config
from tools.utils.numpy_shorts import get_factor_from_max_speed
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


def _blocked_intervals(entries, delay, times=(), min_notes=0):
    """Post-entry windows where subdivision fills are gated; empty when off.

    Each entry blocks ``[entry, entry + delay)`` union ``[entry, t_k)`` with
    ``t_k`` the ``min_notes``-th kept candidate at/after the entry — map end
    when fewer remain."""
    if delay <= 0 and min_notes <= 0:
        return []
    times = np.sort(np.asarray(times, np.float64))
    intervals = []
    for entry in entries:
        end = float(entry) + max(0., float(delay))
        if min_notes > 0:
            index = int(np.searchsorted(times, float(entry))) + min_notes - 1
            end = max(end, float(times[index]) if index < len(times) else math.inf)
        if end > float(entry):
            intervals.append((float(entry), end))
    return intervals


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


def _fill_min_interval():
    """Difficulty-scaled spacing floor for inserted fill times; 0 disables.

    ``v2_fill_min_interval_seconds`` is calibrated at ``v2_fill_min_interval_nps``
    and scales inversely with the map's NPS (``max_speed/4`` in internal
    units), clamped to ``[v2_fill_min_interval_min_seconds, base]`` — high-NPS
    maps keep tighter fills, easy maps never exceed the base.
    """
    base = float(getattr(config, "v2_fill_min_interval_seconds", 0.) or 0.)
    if base <= 0:
        return 0.
    nps = max(float(config.max_speed) / 4., 1e-6)
    ref = float(getattr(config, "v2_fill_min_interval_nps", 5.) or 5.)
    floor = float(getattr(config, "v2_fill_min_interval_min_seconds", .035) or 0.)
    return min(base, max(floor, base * ref / nps))


def _filter_fill_times(previous, new_times, blocked, silent_spans=(), min_interval=0.):
    """Drop fill insertions inside ``blocked`` windows or ``silent_spans``; model times always pass.

    New insertions also keep ``min_interval`` spacing from every existing and
    already-accepted time, so fills cannot create sub-playable event streams.
    """
    if not blocked and not silent_spans and min_interval <= 0:
        return np.asarray(new_times, np.float64)
    kept = {float(time) for time in previous}
    accepted = sorted(kept)
    out = []
    for time in new_times:
        time = float(time)
        if time in kept:
            out.append(time)
            continue
        if any(start <= time < end for start, end in blocked) \
                or _inside_spans(time, silent_spans):
            continue
        if min_interval > 0:
            index = bisect.bisect_left(accepted, time)
            near = accepted[max(0, index - 1):index + 1]
            if near and min(abs(time - other) for other in near) < min_interval:
                continue
        out.append(time)
        bisect.insort(accepted, time)
    return np.asarray(out, np.float64)


_TIMING_STEPS = ("minimum_spacing", "high_difficulty_fill", "bpm_alignment", "onset_snap",
                 "start_end_fill", "silence_removal", "density_fill", "max_gap_fill",
                 "onset_coverage")


def postprocess_timing(song, beats, audio_path, logger=None, skip=()):
    """Apply legacy timing cleanup while carrying count labels through moves.

    Step order follows ``_TIMING_STEPS``: ``minimum_spacing``,
    ``high_difficulty_fill``, ``bpm_alignment``, ``onset_snap``,
    ``start_end_fill``, ``silence_removal``, ``density_fill``,
    ``max_gap_fill``, ``onset_coverage`` (the one ``bpm_alignment`` name
    covers the bookending initial/final alignments). Subdivision fills keep
    insertions out of post-entry ramp windows — ``v2_fill_entry_delay_seconds``
    union a ``v2_fill_entry_min_notes`` kept-candidate count gate — and, gated
    by ``v2_fill_respect_silence``, detected silent spans; unmatched onset-snap
    events follow ``v2_onset_snap_unmatched_policy``; ``max_gap_fill``
    backstops any remaining dead zone from ``v2_gap_fill_source``, always
    keeping inserts out of detected silent spans (independent of the fill
    gate). ``start_end_fill`` drops its start-segment inserts while
    ``v2_suppress_start_fill`` and the intro note floor are both active.
    Every insertion channel keeps the difficulty-scaled
    ``v2_fill_min_interval_seconds`` spacing floor, and ``onset_coverage``
    appends strong detected onsets the model missed, after the final
    alignment so inserts stay on the onset.

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
                    "gap_fill_source_used": None, "entry_blocked_windows": 0,
                    "start_fill_suppressed": 0, "onset_coverage_added": 0}
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

    entry_blocked_windows = 0
    start_fill_suppressed = 0
    min_interval = _fill_min_interval()

    def _entry_intervals(min_notes):
        pause = float(getattr(config, "v2_fill_entry_pause_beats", 4.0) or 0.) * 60. / bpm
        delay = float(getattr(config, "v2_fill_entry_delay_seconds", 0.) or 0.)
        # pause <= 0 disables pause detection: only the map start is an entry
        current = times()
        entries = _section_entries(current, pause) if pause > 0 else np.asarray([current[0]])
        return _blocked_intervals(entries, delay, current, min_notes)

    def fill_blocked():
        nonlocal entry_blocked_windows
        intervals = _entry_intervals(int(getattr(config, "v2_fill_entry_min_notes", 0) or 0))
        entry_blocked_windows = len(intervals)
        return intervals

    def coverage_blocked():
        # Onset-anchored inserts are real audio hits, one per onset and already
        # spaced by the fill floor — they keep the post-entry delay but not the
        # N-note runway, which exists to throttle synthetic subdivision fills.
        return _entry_intervals(0)

    detected_silence = None
    gap_fill_source_used = None

    def silent_spans(force=False):
        """Detected silent spans, decoded once and only when fills/snap gate on them.

        The cached detection keys on ``add_silence_flag`` alone so call order
        cannot leak the gate state; ``v2_fill_respect_silence`` applies per
        call. ``force`` bypasses that fill gate — the onset-snap unmatched
        policy and the max-gap net must still cull events inside detected
        silence. ``detected_silence`` stays undecoded until a gated or forced
        call, so fills alone under ``v2_fill_respect_silence=False`` never
        trigger the legacy decode."""
        nonlocal detected_silence
        if not force and not config.v2_fill_respect_silence:
            return []
        if detected_silence is None:
            detected_silence = (_silent_spans(legacy_features()[2])
                                if config.add_silence_flag else [])
        return detected_silence

    iterations = max(0, int(config.map_filler_iters))
    if "high_difficulty_fill" not in skip:
        for pass_index, (threshold, index) in enumerate(((5.5 * 4, int(iterations / 2) + 1), (8 * 4, iterations - 1))):
            if candidates and iterations and config.max_speed >= threshold:
                replace_times(f"high_difficulty_fill_{pass_index}",
                              _filter_fill_times(times(), fill_map_times_scale(times(), min(iterations - 1, max(0, index))),
                                                 fill_blocked(), silent_spans=silent_spans(),
                                                 min_interval=min_interval))
    align = "bpm_alignment" not in skip
    if candidates and align:
        replace_times("initial_bpm_alignment", align_beats_on_bpm(times(), bpm), moving=True)
    if candidates and "onset_snap" not in skip:
        policy = getattr(config, "v2_onset_snap_unmatched_policy", "always")
        if policy not in ("always", "non_silent", "never"):
            raise ValueError(f"Unknown v2_onset_snap_unmatched_policy: {policy!r}")
        snapped = sanity_check_timing2(None, times(), audio_path=audio_path)
        if policy != "always":
            spans = silent_spans(force=True) if policy == "non_silent" else ()
            snapped = np.asarray([original if marker <= 0 and (policy == "never" or not _inside_spans(original, spans))
                                  else marker
                                  for original, marker in zip(times(), snapped)])
        replace_times("onset_snap", snapped, moving=True, drop_zero=True)
    if candidates and "start_end_fill" not in skip:
        volume, onset, _ = legacy_features()
        current = times()
        proposed = _start_end_times(current, volume, onset)
        if getattr(config, "v2_suppress_start_fill", False) \
                and float(getattr(config, "v2_map_intro_no_notes_seconds", 0.) or 0.) > 0:
            # only added events sit before the first event — the end segment stays
            start_fill_suppressed = int(np.count_nonzero(proposed < current[0]))
            proposed = proposed[proposed >= current[0]]
        added = proposed[~np.isin(proposed, current)]
        if added.size and min_interval > 0:
            proposed = np.sort(np.r_[current, _filter_fill_times(
                current, added, (), min_interval=min_interval)])
        replace_times("start_end_fill", proposed)
    if candidates and config.add_silence_flag and "silence_removal" not in skip:
        *_, silent_times = legacy_features()
        replace_times("silence_removal", remove_silent_times(times(), silent_times))
    if "density_fill" not in skip:
        for index in range(iterations):
            if not candidates or len(candidates) > config.max_speed * 10 / 40 * candidates[-1]["time"] * config.add_beat_intensity / 100:
                break
            replace_times(f"density_fill_{index}",
                          _filter_fill_times(times(), fill_map_times_scale(times(), index), fill_blocked(),
                                             silent_spans=silent_spans(), min_interval=min_interval))
    if candidates and "max_gap_fill" not in skip:
        max_gap = float(getattr(config, "v2_max_note_gap_seconds", 0.) or 0.)
        if max_gap > 0:
            source = getattr(config, "v2_gap_fill_source", "onsets")
            if source not in ("onsets", "bpm_grid", "v1_beat"):
                raise ValueError(f"Unknown v2_gap_fill_source: {source!r}")
            if audio_path is None and source != "bpm_grid":
                raise ValueError(f"v2_gap_fill_source {source!r} requires an audio_path")
            # the edge alone keeps inserts min_beat_time off the bounding events
            edge = max(float(getattr(config, "v2_gap_fill_edge_seconds", .5) or 0.),
                       config.min_beat_time)
            current, inserted, pool, spans = times(), [], None, ()
            for a, b in zip(current, current[1:]):
                if b - a <= max_gap:
                    continue
                if pool is None:
                    # decode candidates and silence only when a dead zone exists
                    used = {"source": source}
                    pool = _gap_candidate_times(source, audio_path, bpm, duration, used)
                    gap_fill_source_used = used["source"]
                    # detected silence is a detection property, not a fill gate
                    spans = silent_spans(force=True)
                for t in pool[(pool > a + edge) & (pool < b - edge)]:
                    if _inside_spans(t, spans) \
                            or (inserted and t - inserted[-1] < max(config.min_beat_time, min_interval)):
                        continue
                    inserted.append(float(t))
            if inserted:
                replace_times("max_gap_fill", np.sort(np.r_[current, inserted]))
    if candidates and align:
        replace_times("final_bpm_alignment", align_beats_on_bpm(times(), bpm), moving=True)
    onset_coverage_added = 0
    onset_coverage_loud_admitted = 0
    if candidates and "onset_coverage" not in skip \
            and getattr(config, "v2_onset_fill_flag", False) and audio_path is not None:
        peaks, strengths, loudness = _detect_onset_peaks(audio_path)
        if len(peaks):
            quantile = min(max(float(getattr(config, "v2_onset_fill_strength_quantile", .5) or 0.), 0.), 1.)
            admitted = strengths >= np.quantile(strengths, quantile)
            loud_quantile = min(max(float(getattr(config, "v2_onset_fill_loudness_quantile", 0.) or 0.), 0.), 1.)
            if loud_quantile > 0:
                low = min(max(float(getattr(config, "v2_onset_fill_loudness_strength_quantile", .25) or 0.), 0.), 1.)
                loud = loudness >= np.quantile(loudness, loud_quantile)
                tier2 = loud & (strengths >= np.quantile(strengths, low))
                onset_coverage_loud_admitted = int(np.sum(tier2 & ~admitted))
                admitted |= tier2
            peaks = peaks[admitted]
            window = float(getattr(config, "v2_onset_fill_window_seconds", .12) or .12)
            current = times()
            index = np.searchsorted(current, peaks)
            missed = []
            for peak, i in zip(peaks, index):
                near = current[max(0, i - 1):i + 1]
                if not len(near) or np.min(np.abs(near - peak)) > window:
                    missed.append(float(peak))
            kept = _filter_fill_times(current, missed, coverage_blocked(),
                                      silent_spans=silent_spans(force=True),
                                      min_interval=min_interval)
            if len(kept):
                onset_coverage_added = len(kept)
                replace_times("onset_coverage", np.sort(np.r_[current, kept]))
    diagnostics.update(events=len(candidates), retained_notes=sum(item["count"] for item in candidates),
                       merged_events=merged_events, minimum_spacing_removed=spacing_removed,
                       timing_policy="legacy_cleanup_count_preserving",
                       gap_fill_source_used=gap_fill_source_used,
                       entry_blocked_windows=entry_blocked_windows,
                       start_fill_suppressed=start_fill_suppressed,
                       onset_coverage_added=onset_coverage_added,
                       onset_coverage_loud_admitted=onset_coverage_loud_admitted)
    _report(logger, f"Timing cleanup: {len(original)} -> {len(candidates)} events")
    return candidates, diagnostics


# Exact reversals keep consecutive same-hand swings alternating; dots are
# directionless and never need a recut.
_REVERSE_DIRECTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}
# Direction index → 45°-step angle (right=0, counterclockwise).
_DIRECTION_ANGLE = {3: 0, 5: 1, 0: 2, 4: 3, 2: 4, 6: 5, 1: 6, 7: 7}
_ANGLE_DIRECTION = {angle: direction for direction, angle in _DIRECTION_ANGLE.items()}
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


_LATE_TURNS_READY = ContextVar("v2_late_turns_ready", default=True)


@contextmanager
def playability_generation_phase():
    """Keep early selection strict; restore the caller's audit phase on exit."""
    token = _LATE_TURNS_READY.set(False)
    try:
        yield
    finally:
        _LATE_TURNS_READY.reset(token)


def _slow_flow_enabled():
    return (getattr(config, "v2_playability_enabled", False)
            and (not getattr(config, "v2_playability_late_turns", False) or _LATE_TURNS_READY.get())
            and (float(getattr(config, "v2_playability_slow90_seconds", 0.) or 0.) > 0
                 or float(getattr(config, "v2_playability_slow45_seconds", 0.) or 0.) > 0))


def _slow_turn_legal(direction, previous, x, y, time, steps):
    """Eligibility comes from the current incoming gap and swing geometry."""
    if not _slow_flow_enabled() or previous is None or previous[3] in (None, 8):
        return False
    ptime, px, py, pdirection = previous
    threshold = float(getattr(config, "v2_playability_slow90_seconds" if steps == 2
                              else "v2_playability_slow45_seconds", 0.) or 0.)
    if steps not in (1, 2) or threshold <= 0 or time - ptime + 1e-9 < threshold:
        return False
    if steps == 2:
        return abs(x - px) + abs(y - py) >= 2
    # Hand travel begins at the previous cut's exit. A positive projection
    # onto the proposed cut's travel vector excludes an approach against it.
    pdx, pdy = _DIRECTION_STEP[pdirection]
    dx, dy = _DIRECTION_STEP[direction]
    return (x - px + pdx) * -dx + (y - py + pdy) * -dy > 0


def _flow_successors(seq, start, alive=None):
    """Only the dot run and its next arrow can depend on a recut."""
    for index in range(start, len(seq)):
        note = seq[index]
        if alive is not None and note["id"] not in alive:
            continue
        yield note
        if note["d"] != 8:
            break


def _flow_edges_legal(direction, note, previous=None, following=(), gap=None):
    """Validate a recut's incoming and outgoing context, including dot runs."""
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.) if gap is None else gap
    if not _flow_legal(direction, previous, note["x"], note["y"], note["time"], gap):
        return False
    prior = (note["time"], note["x"], note["y"], direction)
    successors = (following,) if isinstance(following, dict) else following
    for successor in successors:
        if successor["d"] != 8:
            return _flow_legal(successor["d"], prior, successor["x"], successor["y"],
                               successor["time"], gap)
        implied = _dot_virtual_direction(prior[3], prior[1], prior[2], successor["x"], successor["y"])
        prior = (successor["time"], successor["x"], successor["y"], implied)
    return True


def _flow_legal(direction, previous, x, y, time, same_stroke_gap):
    """Shared edge rule against a (time, x, y, effective direction) predecessor.

    Near-simultaneous pairs share one stroke and are exempt. With
    ``v2_flow_min_turn_steps`` ≤ 2 the legacy rule returns: ≥135°, or 90°
    when the hand physically travels at least two cells. Opt-in slow-turn
    experiments use the current incoming gap and physical approach, with
    strict flow retained below their threshold. No note stores permission."""
    if previous is None:
        return True
    ptime, px, py, pdirection = previous
    if pdirection is None or time - ptime <= same_stroke_gap:
        return True
    steps = _step_distance(direction, pdirection)
    if _slow_turn_legal(direction, previous, x, y, time, steps):
        return True
    if steps <= 1:
        return False
    # Missing/None selects the strict default; an explicit 0 already means
    # <= 2 and must not be coerced back to 3.
    configured = getattr(config, "v2_flow_min_turn_steps", None)
    min_turn = 3 if configured is None else int(configured)
    if min_turn <= 2:
        return steps >= 3 or (steps == 2 and abs(x - px) + abs(y - py) >= 2)
    return steps >= min_turn


def _flow_direction(original, previous_direction, blocked=(), swing_from=None,
                    forbidden_cells=None, exclude=(), *, note=None, previous=None,
                    following=(), same_stroke_gap=None):
    """Nearest reversal of ``original`` that keeps ≥135° from ``previous_direction``.

    Human flow resolves repeats with a swing back (~180°), so the candidate
    furthest from ``previous_direction`` wins; ties fall back to the smallest
    rotation away from ``original``. This keeps the forced fix inside the
    reversal vocabulary instead of manufacturing side cuts the model never
    asked for. ``blocked`` lists further directions the replacement must stay
    ≥135° away from (used when recutting the earlier note of a pair) and
    ``exclude`` removes directions outright (e.g. the counter family inside a
    fast pair). With ``swing_from``/``forbidden_cells`` a candidate is
    rejected when one of its swept cells is occupied by a simultaneous
    other-hand note. With contextual slow flow enabled, ``note``, ``previous``
    and ``following`` validate the complete affected incoming/outgoing edges
    using the same predicate as snaps, audits and cap feasibility."""
    others = (previous_direction, *blocked)
    candidates = [d for d in range(8) if d not in exclude
                  and (_flow_edges_legal(d, note, previous, following, same_stroke_gap)
                       if note is not None and _slow_flow_enabled() else
                       all(_step_distance(d, other) >= 3 for other in others))
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
                                                swing_from=(x, y), forbidden_cells=forbidden,
                                                note={"time": time, "x": x, "y": y},
                                                previous=previous, same_stroke_gap=same_stroke_gap)
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
    slow_context = _slow_flow_enabled()
    per_hand = ({hand: [note for note in ordered if note["hand"] == hand] for hand in (0, 1)}
                if slow_context else {})
    sequence_position = {note["id"]: index for seq in per_hand.values() for index, note in enumerate(seq)}
    for note in ordered:
        following_notes = (tuple(_flow_successors(per_hand[note["hand"]], sequence_position[note["id"]] + 1))
                           if slow_context else ())
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
                                                forbidden_cells=other_cells.get(previous["id"]),
                                                note=previous, previous=((before["time"], before["x"], before["y"], before["d"])
                                                                         if before is not None else None),
                                                following=(note, *following_notes))
                flipped += 1
            elif not protect_retained or note.get("kind") != "retained":
                note["d"] = _flow_direction(note["d"], previous["d"], swing_from=(note["x"], note["y"]),
                                            forbidden_cells=other_cells.get(note["id"]),
                                            note=note, previous=prior, following=following_notes)
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


def _detect_onset_peaks(audio_path):
    """Detected onset peaks in seconds with envelope strengths and local loudness.

    Loudness is RMS energy smoothed over v2_onset_fill_loudness_smooth_seconds
    sampled at each peak frame — it separates "audible hit inside a loud
    section" from "transient inside a quiet passage" for the coverage tiers."""
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
    times = np.asarray(librosa.frames_to_time(onsets, sr=rate, hop_length=512), np.float64)
    rms = librosa.feature.rms(y=samples, hop_length=512)[0]
    width = max(1, int(float(getattr(config, "v2_onset_fill_loudness_smooth_seconds", 1.) or 1.)
                       * rate / 512))
    smooth = np.convolve(rms, np.ones(width) / width, mode="same")
    loudness = np.asarray(smooth[np.clip(onsets, 0, len(smooth) - 1)], np.float64)
    return times, np.asarray(envelope[onsets], np.float64), loudness


def _detect_onsets(audio_path):
    """Detected onset peaks in seconds — same librosa recipe as the legacy snap."""
    return _detect_onset_peaks(audio_path)[0]


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


# Row → cut families fighting the note's own position: a top-row down-cut or
# bottom-row up-cut reads as "reach the other way" at speed.
_COUNTER_FAMILY = {0: (0, 4, 5), 2: (1, 6, 7)}


def _counter_direction(note):
    """The cut opposes the note's row: top-row down-family / bottom-row up-family."""
    return note["d"] in _COUNTER_FAMILY.get(note["y"], ())


def apply_fast_direction_rules(notes, rng=None, logger=None):
    """Collapse chaotic directions inside fast same-hand runs to ping-pong.

    A same-hand consecutive pair closer than ``v2_fast_gap_seconds`` (<= 0
    disables both rules) is a fast pair; runs extend transitively through
    consecutive fast pairs. Inside a run, each note's allowed direction set is
    its predecessor's direction plus the 180° reverse, union the ±45°
    family admitted per note with probability
    ``min(1, expected / local)`` — the reverse's own ±45° neighbours (the
    135°/225° off-reversals, the only ±45° cuts ``_flow_legal`` admits
    between separated strokes) plus the predecessor's ±45° neighbours,
    which only same-stroke pairs can play. ``expected`` is the mean
    ``calc_note_speed`` over *every* same-hand pair in the map (the per-map
    average at this NPS) and ``local`` the pair's own cost, so a pair moving
    at the map average keeps the ±45° family almost always while an
    extreme burst converges to strict ping-pong. A note outside its set is
    recut to an allowed direction that stays ``_flow_legal`` against both
    same-hand neighbours and ``_swing_clear`` of simultaneous other-hand
    cells; when nothing is legal the note keeps its direction
    (legality-first, like the diagonal cap).
    With ``v2_fast_counter_direction = "recut"`` ("off"/"keep" disables), run
    members whose cut opposes their row (top-row down-family d 1/6/7,
    bottom-row up-family d 0/4/5) are recut first — preferring the allowed
    pool, then any legal non-counter direction. Dots are untouched and pass
    their geometry-implied direction on, the convention of
    ``_recut_same_direction_repeats``. Neighbours are read with their current
    direction, so earlier recuts constrain later ones. Returns run stats."""
    rng = np.random.default_rng() if rng is None else rng
    stats = {"fast_pairs": 0, "fast_simplified": 0,
             "fast_counter_recuts": 0, "expected_speed": 0.}
    threshold = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    if threshold <= 0 or len(notes) < 2:
        return stats
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    per_hand = {hand: [note for note in ordered if note["hand"] == hand]
                for hand in (0, 1)}
    speeds, fast, runs = [], {}, set()
    for seq in per_hand.values():
        for previous, current in zip(seq, seq[1:]):
            speed = calc_note_speed([previous[key] for key in ("x", "y", "hand", "d")],
                                    [current[key] for key in ("x", "y", "hand", "d")],
                                    current["time"] - previous["time"])
            speeds.append(speed)
            if current["time"] - previous["time"] < threshold:
                fast[current["id"]] = speed
                runs.update((previous["id"], current["id"]))
    expected = float(np.mean(speeds)) if speeds else 0.
    stats["expected_speed"] = expected
    stats["fast_pairs"] = len(fast)
    if expected <= 0 or not fast:
        return stats
    eps = float(getattr(config, "min_time_diff", .02))
    window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    cells = _other_hand_cells(ordered, eps, window)
    counter_mode = getattr(config, "v2_fast_counter_direction", "recut") == "recut"

    for seq in per_hand.values():
        previous = None  # (time, x, y, effective direction through dots)
        for index, note in enumerate(seq):
            time, x, y = note["time"], note["x"], note["y"]
            pred_d = previous[3] if previous is not None else None
            allowed = None
            local = fast.get(note["id"])
            if local is not None and pred_d is not None and note["d"] != 8:
                allowed = {pred_d, _REVERSE_DIRECTION[pred_d]}
                if rng.uniform() < (min(1., expected / local) if local > 0 else 1.):
                    angle = _DIRECTION_ANGLE[pred_d]
                    allowed |= {_ANGLE_DIRECTION[(angle + step) % 8]
                                for step in (-1, 1, 3, 5)}
            counter = counter_mode and note["id"] in runs and _counter_direction(note)
            if counter or (allowed is not None and note["d"] != 8
                           and note["d"] not in allowed):
                successor = seq[index + 1] if index + 1 < len(seq) else None

                def legal(candidate):
                    if not _flow_legal(candidate, previous, x, y, time, gap):
                        return False
                    if successor is not None:
                        next_d = successor["d"] if successor["d"] != 8 else \
                            _dot_virtual_direction(candidate, x, y,
                                                   successor["x"], successor["y"])
                        if next_d is not None and not _flow_legal(
                                next_d, (time, x, y, candidate),
                                successor["x"], successor["y"], successor["time"], gap):
                            return False
                    return _swing_clear((x, y), cells.get(note["id"]), candidate)

                def recut(pool):
                    candidates = [d for d in pool if legal(d)]
                    if not candidates:
                        return False
                    note["d"] = _rank_flow_candidates(
                        note["d"], candidates, () if pred_d is None else (pred_d,))
                    return True

                if counter:
                    pool = [d for d in range(8) if d not in _COUNTER_FAMILY[y]]
                    preferred = [d for d in (allowed or pool) if d in pool]
                    if recut(preferred) or recut(pool):
                        stats["fast_counter_recuts"] += 1
                elif recut(sorted(allowed)):
                    stats["fast_simplified"] += 1
            # dots get their implied direction from geometry, alternating the pendulum
            effective = note["d"] if note["d"] != 8 else (
                _dot_virtual_direction(previous[3], previous[1], previous[2], x, y)
                if previous is not None else None)
            previous = (time, x, y, effective)
    _report(logger, f"Fast directions: {stats['fast_pairs']} pairs, "
                    f"{stats['fast_simplified']} simplified, "
                    f"{stats['fast_counter_recuts']} counter recuts "
                    f"(expected speed {expected:.1f})")
    return stats


def apply_fast_counter_sweep(notes, rng=None, logger=None):
    """Recut counter directions inside fast pairs formed after the snapshot.

    Multiplier copies inherit their anchor's cut, so expansion — and the
    speed cap's deletions — can leave fast same-hand pairs the pre-snapshot
    ``apply_fast_direction_rules`` never saw. Only non-retained notes are
    eligible: retained ones are contract-fixed against the snapshot
    ``generate`` verifies at the end. A counter-directional member (the
    same ``_counter_direction`` predicate) is recut to a legal direction
    with the counter family excluded from the pool; when nothing is legal
    the note keeps its cut (legality-first, like the other passes). The
    ping-pong/±45° simplification does not rerun — counter only. Dots are
    untouched and pass their geometry-implied direction on. Returns run
    stats."""
    stats = {"fast_pairs_checked": 0, "fast_counter_recuts": 0}
    threshold = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    if (threshold <= 0 or len(notes) < 2
            or getattr(config, "v2_fast_counter_direction", "recut") != "recut"):
        return stats
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    per_hand = {hand: [note for note in ordered if note["hand"] == hand]
                for hand in (0, 1)}
    fast = set()
    for seq in per_hand.values():
        for previous, current in zip(seq, seq[1:]):
            if current["time"] - previous["time"] < threshold:
                stats["fast_pairs_checked"] += 1
                fast.update((previous["id"], current["id"]))
    if not fast:
        return stats
    eps = float(getattr(config, "min_time_diff", .02))
    window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    cells = _other_hand_cells(ordered, eps, window)
    for seq in per_hand.values():
        previous = None  # (time, x, y, effective direction through dots)
        for index, note in enumerate(seq):
            time, x, y = note["time"], note["x"], note["y"]
            if (note["id"] in fast and note.get("kind") != "retained"
                    and _counter_direction(note)):
                successor = seq[index + 1] if index + 1 < len(seq) else None

                def legal(candidate):
                    if not _flow_legal(candidate, previous, x, y, time, gap):
                        return False
                    if successor is not None:
                        next_d = successor["d"] if successor["d"] != 8 else \
                            _dot_virtual_direction(candidate, x, y,
                                                   successor["x"], successor["y"])
                        if next_d is not None and not _flow_legal(
                                next_d, (time, x, y, candidate),
                                successor["x"], successor["y"], successor["time"], gap):
                            return False
                    return _swing_clear((x, y), cells.get(note["id"]), candidate)

                candidates = [d for d in range(8)
                              if d not in _COUNTER_FAMILY.get(y, ()) and legal(d)]
                if candidates:
                    pred_d = previous[3] if previous is not None else None
                    note["d"] = _rank_flow_candidates(
                        note["d"], candidates, () if pred_d is None else (pred_d,))
                    stats["fast_counter_recuts"] += 1
            # dots get their implied direction from geometry, alternating the pendulum
            effective = note["d"] if note["d"] != 8 else (
                _dot_virtual_direction(previous[3], previous[1], previous[2], x, y)
                if previous is not None else None)
            previous = (time, x, y, effective)
    _report(logger, f"Fast counter sweep: {stats['fast_pairs_checked']} pairs, "
                    f"{stats['fast_counter_recuts']} counter recuts")
    return stats


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
    count = len(notes)
    excess = max(0, len(diagonals) - int(cap * count))
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


def apply_note_emphasis(state, song, *, rng=None, logger=None):
    """Optional v1-style slow-swing stacks; see ``v2_note_emphasis``."""
    from map_creation.v2_note_emphasis import apply_note_emphasis as _impl
    return _impl(state, song, rng=rng, logger=logger)


def speed_cap_scale():
    """Effective v2 speed-cap scale.

    ``v2_speed_cap_scale`` is the reference at 100% beat intensity; the app's
    Beat Intensity (%) field scales it linearly (100 -> x1, 50 -> x0.5), so
    one dial governs note density and allowed movement speed together. Reads
    ``add_beat_intensity_orig`` — the user-visible value before per-variant
    silence/emphasize/sporty adjustments, which shift the fill budget only."""
    scale = float(getattr(config, "v2_speed_cap_scale", 1.) or 0.)
    intensity = getattr(config, "add_beat_intensity_orig", None)
    if intensity is None:
        intensity = getattr(config, "add_beat_intensity", None)
    if intensity is not None:
        scale *= float(intensity) / 100.
    return scale


def repair_playability_directions(notes, cap, **kwargs):
    """Explicit arrow-only experiment, also usable on portable archived passages."""
    from map_creation.v2_playability_directions import repair_directions
    return repair_directions(notes, cap, **kwargs)


def finalize_playability_reach(notes, chains=None, *, protect_retained=True, cap=None, hard_only=False, guard=None):
    """Apply the opt-in reach policy and keep retained expansion anchors intact."""
    if not (getattr(config, "v2_playability_enabled", False)
            and (getattr(config, "v2_playability_reach", False)
                 or getattr(config, "v2_playability_priority", False))):
        return {}
    from map_creation.v2_playability_reach import repair_reach
    if cap is None:
        scale = speed_cap_scale()
        ordered = sorted(notes, key=lambda note: (note["time"], note["id"]))
        count = max(int(getattr(config, "v2_speed_cap_start_notes", 30) or 0), 0)
        first = float(getattr(config, "v2_speed_cap_start_factor", .35) or 0.)
        ramp = np.linspace(first, 1., count) if count else ()
        limits = {}
        for index, note in enumerate(ordered):
            speed = config.max_speed * (ramp[index] if index < count else 1.)
            limits[note["id"]] = (speed * get_factor_from_max_speed(speed, .5, 1) * scale
                                  if scale > 0 else float("inf"))
        cap = lambda note: limits[note["id"]]
    if getattr(config, "v2_playability_priority", False):
        from map_creation.v2_playability_priority import repair_priority
        stats = repair_priority(notes, cap, protect_retained=protect_retained)
    else:
        stats = repair_reach(notes, cap, protect_retained=protect_retained, hard_only=hard_only, guard=guard)
    anchors = {note["id"]: note for note in notes}
    for chain in chains or []:
        anchor = anchors.get(chain.get("anchor_id"))
        if anchor is not None:
            chain["x"], chain["y"], chain["d"] = anchor["x"], anchor["y"], anchor["d"]
    return stats


def _commit_decoder_recovery(notes, cap, stats):
    """Commit supported decoder alternatives locally, without model-history ripple."""
    if getattr(config, "v2_playability_priority", False):
        return
    if not (getattr(config, "v2_playability_enabled", False)
            and getattr(config, "v2_playability_reach", False)
            and getattr(config, "v2_playability_recovery_balanced", False)):
        return
    from map_creation.v2_playability_policy import PatternGuard
    from map_creation.v2_playability_recovery import repair_recovery
    guard = PatternGuard(notes)
    stats.update(repair_recovery(notes, cap, guard=guard))
    stats["recovery_pattern_guard"] = guard.report()


def _apply_playability_experiments(notes, cap, stats, isolated_ids=()):
    if not getattr(config, "v2_playability_enabled", False):
        return
    if getattr(config, "v2_playability_late_turns", False):
        _LATE_TURNS_READY.set(True)
    identity = sorted((n["id"], n["time"], n["hand"]) for n in notes)
    if getattr(config, "v2_playability_priority", False):
        import hashlib
        import json
        from map_creation.v2_playability_priority import repair_priority
        native = sorted(tuple(n[k] for k in ("id", "time", "hand", "x", "y", "d")) for n in notes)
        stats["priority_input_signature_sha256"] = hashlib.sha256(
            json.dumps(native, separators=(",", ":")).encode()).hexdigest()
        stats.update(repair_priority(notes, cap, protect_retained=False))
        stats["playability_preserved"] = identity == sorted((n["id"], n["time"], n["hand"]) for n in notes)
        if not stats["playability_preserved"]:
            raise RuntimeError("Priority policy changed note identity or rhythm")
        return
    guard = None
    if getattr(config, "v2_playability_reach", False):
        mandatory = finalize_playability_reach(notes, protect_retained=False, cap=cap, hard_only=True)
        stats.update({"mandatory_"+key: value for key, value in mandatory.items()})
        from map_creation.v2_playability_policy import PatternGuard
        guard = PatternGuard(notes)
    geometry = (getattr(config, "v2_playability_m2", False),
                getattr(config, "v2_playability_m3", False),
                getattr(config, "v2_playability_m4", False))
    if any(geometry):
        from map_creation.v2_playability_geometry import repair_geometry
        stats.update(repair_geometry(notes, cap, sightline=geometry[0],
                                     coordination=geometry[1], interference=geometry[2], guard=guard))
    initial_diagonals = sum(n["d"] in (4, 5, 6, 7) for n in notes)
    diagonal_limit = initial_diagonals + math.floor(len(notes) * .01 + 1e-9)
    if getattr(config, "v2_playability_reach", False):
        from map_creation.v2_playability_reach import diagonal_fraction_limit, note_span_nps
        arrow_count = sum(note["d"] != 8 for note in notes)
        diagonal_limit = max(initial_diagonals,
            min(initial_diagonals + math.floor(len(notes)*.01 + 1e-9),
                math.floor(arrow_count * diagonal_fraction_limit(note_span_nps(notes)) + 1e-9)))

    def budget():
        return (diagonal_limit - sum(n["d"] in (4, 5, 6, 7) for n in notes)
                if getattr(config, "v2_playability_m1", False) else None)

    if getattr(config, "v2_playability_m1", False):
        stats.update(repair_playability_directions(notes, cap, diagonal_budget=budget(), guard=guard))
    if getattr(config, "v2_playability_isolated", False):
        from map_creation.v2_playability_directions import repair_isolated
        stats.update(repair_isolated(notes, cap, eligible_ids=isolated_ids, diagonal_budget=budget(), guard=guard))
    if getattr(config, "v2_playability_late_turns", False):
        from map_creation.v2_playability_directions import repair_late_turns
        stats.update(repair_late_turns(notes, cap, diagonal_budget=budget(), guard=guard))
    if getattr(config, "v2_playability_dwell", False):
        from map_creation.v2_playability_directions import repair_dwell
        stats.update(repair_dwell(notes, cap, diagonal_budget=budget(), guard=guard))
    if getattr(config, "v2_playability_reach", False):
        stats.update(finalize_playability_reach(notes, protect_retained=False, cap=cap, guard=guard))
        if any(geometry):
            # Joint arrow/cell repair changes travel axes and visibility.
            # One bounded, reach-safe refinement uses the settled arrows.
            settled = repair_geometry(notes, cap, sightline=geometry[0],
                                      coordination=geometry[1], interference=geometry[2], guard=guard)
            stats.update({"settled_" + name: value for name, value in settled.items()})
    if guard is not None:
        stats.update(guard.report())
    stats["playability_preserved"] = identity == sorted((n["id"], n["time"], n["hand"]) for n in notes)
    if not stats["playability_preserved"]:
        raise RuntimeError("Playability experiment changed note identity or rhythm")


def apply_speed_cap(notes, chains=None, rng=None, logger=None, drop_retained=True):
    """Drop a note of any same-hand pair above the v1 movement cap.

    The legacy ``correct_notes`` gate lives inside ``legacy_note_flow``, which
    ``v2_cleanup_note_heuristics = False`` skips, so nothing else bounds
    travel per second. The pair cost is ``calc_note_speed`` on consecutive
    kept same-hand notes; the cap composes like v1 — ``ramped = max_speed *
    ramp`` with ``ramp`` the linear ``v2_speed_cap_start_factor`` → 1 ramp
    over the first ``v2_speed_cap_start_notes`` combined chronological
    positions keyed on the later note (v1's ``decr_speed_val``/
    ``decr_speed_range`` start ramp without its end half), then
    ``cap = ramped * get_factor_from_max_speed(ramped, .5, 1) *
    v2_speed_cap_scale`` (nonpositive scale disables the pass). Evaluating
    the factor on the ramped value tightens early positions further than
    scaling the base cap. Dropping the later note (v1 dropped the whole
    timestep) keeps the other hand's; anchored copies and chains go with it
    so no expansion is orphaned. Each deletion re-checks the newly-adjacent
    pair for speed and ``_flow_legal`` — an illegal successor is recut via
    ``_flow_direction`` or dropped when no legal recut exists — iterating
    to a fixpoint bounded at twice the note count (deletion-only, so it
    always converges). ``drop_retained=False`` freezes retained notes —
    production snapshots them before multiplier expansion and verifies them
    at the end of ``generate``: a violating pair whose later note is
    retained is resolved through the earlier one instead — a mutable
    earlier note drops on overspeed, and a revealed flow violation recuts
    or drops it — so only a retained→retained pair is counted skipped
    (``speed_cap_retained_skipped`` for overspeed, ``speed_cap_flow_skipped``
    for flow, once per note and reason) and the frozen note still becomes
    the predecessor for the next check. Copies and heuristic notes still
    drop or recut normally. A recut inside a fast pair
    (``v2_fast_gap_seconds``) keeps the counter-direction family out of its
    candidate pool, so the pass cannot create what
    ``apply_fast_counter_sweep`` would need to fix again. Returns run
    statistics; ``speed_cap_value`` is the effective base cap before the
    start ramp."""
    scale = speed_cap_scale()
    value = config.max_speed * get_factor_from_max_speed(config.max_speed, .5, 1) * scale
    stats = {"speed_cap_dropped": 0, "speed_cap_chains_dropped": 0,
             "speed_cap_recuts": 0, "speed_cap_retained_skipped": 0,
             "speed_cap_flow_skipped": 0, "speed_cap_value": value}
    if drop_retained and getattr(config, "v2_playability_enabled", False):
        stats["speed_cap_enabled"] = scale > 0
    if scale <= 0 or len(notes) < 2:
        if drop_retained:
            _commit_decoder_recovery(notes, lambda note: float("inf"), stats)
            _apply_playability_experiments(notes, lambda note: float("inf"), stats)
            anchors = {note["id"]: note for note in notes}
            for chain in chains or []:
                anchor = anchors.get(chain.get("anchor_id"))
                if anchor is not None:
                    chain["x"], chain["y"], chain["d"] = anchor["x"], anchor["y"], anchor["d"]
        return stats
    ordered = sorted(notes, key=lambda item: (item["time"], item["id"]))
    position = {note["id"]: index for index, note in enumerate(ordered)}
    start_notes = max(int(getattr(config, "v2_speed_cap_start_notes", 30) or 0), 0)
    start_factor = float(getattr(config, "v2_speed_cap_start_factor", .35) or 0.)
    ramp = np.linspace(start_factor, 1., start_notes) if start_notes else ()

    def cap_at_position(index):
        # v1 composition: the ramp scales max_speed before the factor is
        # evaluated, so early positions tighten beyond a scaled base cap.
        ramped = config.max_speed * (ramp[index] if index < start_notes else 1.)
        return ramped * get_factor_from_max_speed(ramped, .5, 1) * scale

    def cap(note):
        return cap_at_position(position[note["id"]])

    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    eps = float(getattr(config, "min_time_diff", .02))
    window = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    alive = {note["id"] for note in ordered}
    spared = set()  # (stat key, note id) already counted under drop_retained=False
    copies_of = {}
    for note in ordered:
        if note.get("kind") == "copy" and note.get("anchor_id") is not None:
            copies_of.setdefault(note["anchor_id"], []).append(note["id"])
    # Initial fast-pair membership; it only shrinks as notes drop, so it is a
    # conservative gate for the counter-family exclusion below.
    fast_ids = set()
    fast_gap = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    if getattr(config, "v2_fast_counter_direction", "recut") == "recut" and fast_gap > 0:
        for hand in (0, 1):
            seq = [note for note in ordered if note["hand"] == hand]
            for first, second in zip(seq, seq[1:]):
                if second["time"] - first["time"] < fast_gap:
                    fast_ids.update((first["id"], second["id"]))
    needs_flow = set()
    isolated_ids = set()
    for _ in range(max(1, 2 * len(ordered))):
        changed = False
        for hand in (0, 1):
            seq = [note for note in ordered if note["hand"] == hand and note["id"] in alive]
            cells = _other_hand_cells([note for note in ordered if note["id"] in alive],
                                    eps, window)
            seq_pos = {note["id"]: index for index, note in enumerate(seq)}

            def effective(index):
                note = seq[index]
                if note["d"] != 8:
                    return note["d"]
                previous = index - 1
                while previous >= 0 and seq[previous]["id"] not in alive:
                    previous -= 1
                if previous < 0:
                    return None
                implied = effective(previous)
                return (None if implied is None
                        else _dot_virtual_direction(implied, seq[previous]["x"], seq[previous]["y"],
                                                    note["x"], note["y"]))

            def mark_followers(identity):
                # a deletion or recut changes the implied direction of a dot
                # run, so flag the following alive notes through its dots
                index = seq_pos.get(identity)
                if index is None:
                    return
                index += 1
                while index < len(seq) and seq[index]["id"] not in alive:
                    index += 1
                while index < len(seq):
                    needs_flow.add(seq[index]["id"])
                    if seq[index]["d"] != 8:
                        break
                    index += 1
                    while index < len(seq) and seq[index]["id"] not in alive:
                        index += 1

            def drop(note):
                for victim in (note["id"], *copies_of.get(note["id"], ())):
                    if victim not in alive:
                        continue
                    alive.discard(victim)
                    stats["speed_cap_dropped"] += 1
                    mark_followers(victim)

            def frozen(note):
                # True when the note must survive untouched: a retained
                # note under drop_retained=False.
                return not drop_retained and note.get("kind") == "retained"

            def skip(note, key):
                # Count a retained->retained pair once per frozen later note
                # and reason (overspeed vs revealed flow violation).
                if (key, note["id"]) not in spared:
                    spared.add((key, note["id"]))
                    stats[key] += 1

            def counter_excluded(note):
                # Recuts inside a fast pair keep the counter family out of
                # the candidate pool.
                return _COUNTER_FAMILY.get(note["y"], ()) if note["id"] in fast_ids else ()

            index = 0
            while index < len(seq):
                if seq[index]["id"] not in alive:
                    index -= 1
                    while index >= 0 and seq[index]["id"] not in alive:
                        index -= 1
                    if index < 0:
                        index = 0
                        while index < len(seq) and seq[index]["id"] not in alive:
                            index += 1
                    continue
                following = index + 1
                while following < len(seq) and seq[following]["id"] not in alive:
                    following += 1
                if following >= len(seq):
                    break
                previous, current = seq[index], seq[following]
                speed = calc_note_speed([previous[key] for key in ("x", "y", "hand", "d")],
                                        [current[key] for key in ("x", "y", "hand", "d")],
                                        current["time"] - previous["time"])
                if speed > cap(current):
                    if frozen(current):
                        if frozen(previous):
                            # retained->retained overspeed: unresolvable
                            skip(current, "speed_cap_retained_skipped")
                            index = following
                            continue
                        # the earlier note is equally responsible; with the
                        # later one frozen it is the only mutable choice
                        drop(previous)
                        changed = True
                        continue
                    drop(current)
                    changed = True
                    continue
                if current["id"] in needs_flow:
                    implied = effective(index)
                    prior = (previous["time"], previous["x"], previous["y"], implied)
                    if current["d"] != 8 and not _flow_legal(
                            current["d"], prior, current["x"], current["y"],
                            current["time"], gap):
                        if frozen(current):
                            if frozen(previous):
                                # retained->retained flow: unresolvable; the
                                # mark is consumed once, on the skip
                                needs_flow.discard(current["id"])
                                skip(current, "speed_cap_flow_skipped")
                                index = following
                                continue
                            # the mutable earlier note carries the fix: recut
                            # it against the frozen successor and its own
                            # predecessor, or drop it when nothing is legal
                            earlier = index - 1
                            while earlier >= 0 and seq[earlier]["id"] not in alive:
                                earlier -= 1
                            blocked = ()
                            if earlier >= 0:
                                implied_before = effective(earlier)
                                if implied_before is not None:
                                    blocked = (implied_before,)
                            changed = True
                            recut = _flow_direction(
                                previous["d"], current["d"], blocked,
                                swing_from=(previous["x"], previous["y"]),
                                forbidden_cells=cells.get(previous["id"]),
                                exclude=counter_excluded(previous), note=previous,
                                previous=((seq[earlier]["time"], seq[earlier]["x"], seq[earlier]["y"], effective(earlier))
                                          if earlier >= 0 else None),
                                following=(_flow_successors(seq, following, alive) if _slow_flow_enabled() else ()))
                            if recut == 8:
                                drop(previous)
                            else:
                                previous["d"] = recut
                                isolated_ids.add(previous["id"])
                                stats["speed_cap_recuts"] += 1
                                mark_followers(previous["id"])
                            continue
                        needs_flow.discard(current["id"])
                        recut = _flow_direction(current["d"], implied,
                                                swing_from=(current["x"], current["y"]),
                                                forbidden_cells=cells.get(current["id"]),
                                                exclude=counter_excluded(current), note=current, previous=prior,
                                                following=(_flow_successors(seq, following + 1, alive) if _slow_flow_enabled() else ()))
                        changed = True
                        if recut == 8:
                            drop(current)
                        else:
                            current["d"] = recut
                            isolated_ids.add(current["id"])
                            stats["speed_cap_recuts"] += 1
                            mark_followers(current["id"])
                        continue
                    needs_flow.discard(current["id"])
                index = following
        if not changed:
            break
    notes[:] = [note for note in notes if note["id"] in alive]
    if chains:
        kept = len(chains)
        chains[:] = [chain for chain in chains if chain["anchor_id"] in alive]
        stats["speed_cap_chains_dropped"] = kept - len(chains)
    if drop_retained:
        repair_cap = cap
        if getattr(config, "v2_playability_enabled", False):
            # The frozen pass builds a fresh ramp after primary deletions.
            # Only retained predecessors are guaranteed to survive until it;
            # mutable heuristics may disappear and copies may add later slots.
            # Their retained-only index is therefore a conservative lower
            # bound for a rising ramp. Include the base cap for a custom
            # decreasing ramp, and retain each original per-ID budget too.
            frozen_position, retained_before = {}, 0
            for note in ordered:
                if note["id"] not in alive:
                    continue
                frozen_position[note["id"]] = retained_before
                retained_before += note.get("kind") == "retained"

            def repair_cap(note):
                return min(cap(note), cap_at_position(frozen_position[note["id"]]), value)

        _commit_decoder_recovery(notes, repair_cap, stats)
        _apply_playability_experiments(notes, repair_cap, stats, isolated_ids)
    anchors = {note["id"]: note for note in notes}
    for chain in chains or []:
        anchor = anchors.get(chain.get("anchor_id"))
        if anchor is not None:
            chain["x"], chain["y"], chain["d"] = anchor["x"], anchor["y"], anchor["d"]
    skipped = []
    if stats["speed_cap_retained_skipped"]:
        skipped.append(f"{stats['speed_cap_retained_skipped']} retained overspeed")
    if stats["speed_cap_flow_skipped"]:
        skipped.append(f"{stats['speed_cap_flow_skipped']} retained flow")
    spared_note = f", {' + '.join(skipped)} skipped" if skipped else ""
    _report(logger, f"Speed cap: {stats['speed_cap_dropped']} notes and "
                    f"{stats['speed_cap_chains_dropped']} chains dropped, "
                    f"{stats['speed_cap_recuts']} recuts{spared_note} (cap {value:.1f})")
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
    decoded_directions = ({(float(n["time"]), n["hand"], n["x"], n["y"]): n["d"] for n in notes}
                          if getattr(config, "v2_playability_enabled", False)
                          and (getattr(config, "v2_playability_dwell", False)
                               or getattr(config, "v2_playability_late_turns", False)) else {})

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
            decoded = decoded_directions.get((float(time), hand, x, y))
            if decoded is not None:
                structured[-1]["_decoded_d"] = decoded
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
