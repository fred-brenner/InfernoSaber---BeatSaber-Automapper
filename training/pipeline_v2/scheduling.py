"""Deterministic elapsed-second query proposals, independent of model outputs.

These are candidate schedules, not clearance or playability validators. Every
schedule records its parameters; proposed times are never silently snapped.
"""

import math

import numpy as np


SCHEDULE_DEFAULTS = {
    "wall_min_gap_seconds": .8,
    "wall_margin_before_seconds": .3,
    "wall_margin_after_seconds": .3,
    "wall_strobe_grid_beats": 1.,
    "wall_strobe_duration_seconds": .1,
    "wall_strobe_min_gap_seconds": .3,
    "clearance_seconds": .1,
    "bomb_grid_beats": .5,
    "bomb_min_gap_seconds": .3,
    "lighting_grid_beats": .5,
    "lighting_onset_threshold": 1.,
    "arc_min_gap_seconds": .15,
    "arc_max_gap_seconds": .8,
    "max_candidates": 10000,
}


def schedule_settings(overrides=None):
    """Read live project defaults, accepting explicit recorded run overrides."""
    from tools.config import config

    selected = {key: getattr(config, f"pipeline_v2_{key}", value)
                for key, value in SCHEDULE_DEFAULTS.items()}
    selected.update({key: value for key, value in (overrides or {}).items() if key in selected})
    for key, value in selected.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid scheduler parameter {key}")
        if value < 0 or (key in ("bomb_grid_beats", "lighting_grid_beats", "max_candidates", "wall_strobe_duration_seconds") and value == 0):
            raise ValueError(f"Invalid scheduler parameter {key}")
    if int(selected["max_candidates"]) != selected["max_candidates"]:
        raise ValueError("max_candidates must be a positive integer")
    selected["max_candidates"] = int(selected["max_candidates"])
    if selected["arc_max_gap_seconds"] < selected["arc_min_gap_seconds"]:
        raise ValueError("Arc maximum gap must be at least its minimum")
    return selected


def _timeline(duration, bpm):
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("Schedules require finite positive duration and BPM")


def _bounded(candidates, settings):
    if len(candidates) > settings["max_candidates"]:
        raise ValueError(f"Candidate schedule has {len(candidates)} queries; exceeds configured "
                         f"max_candidates={settings['max_candidates']}; no queries were discarded")
    return candidates


def _grid(duration, bpm, beat_step, maximum):
    step = 60. * beat_step / bpm
    count = int(math.ceil(duration / step))
    if count > maximum:
        raise ValueError(f"Candidate grid has {count} queries; exceeds configured "
                         f"max_candidates={maximum}; no queries were discarded")
    return np.arange(count, dtype=np.float64) * step


def hazard_candidates(notes, duration, bpm, settings=None):
    """Propose walls in whole-map gaps and bombs on a separate BPM grid.

    Wall margins follow the legacy gap-rule convention. Colored-note and chain
    clearance remains an explicit review boundary in the concept pipeline.
    """
    selected = schedule_settings(settings)
    _timeline(duration, bpm)
    times = sorted({float(note["time"]) for note in notes if 0 <= note["time"] < duration})
    boundaries = [0., *times, float(duration)]
    candidates = []
    for previous, following in zip(boundaries, boundaries[1:]):
        if following - previous < selected["wall_min_gap_seconds"]:
            continue
        start = previous + selected["wall_margin_before_seconds"]
        end = following - selected["wall_margin_after_seconds"]
        if end > start:
            candidates.append({"kind": "wall", "time": start, "end_time": end,
                               "source_gap": [previous, following]})
    if selected["wall_strobe_grid_beats"] > 0:
        for time in _grid(duration, bpm, selected["wall_strobe_grid_beats"], selected["max_candidates"]):
            end = float(time) + selected["wall_strobe_duration_seconds"]
            if end >= duration:
                continue
            left = int(np.searchsorted(times, float(time) - selected["wall_strobe_min_gap_seconds"], side="left"))
            right = int(np.searchsorted(times, end + selected["wall_strobe_min_gap_seconds"], side="right"))
            if left == right:
                candidates.append({"kind": "wall", "time": float(time), "end_time": end, "strobe": True})
    for time in _grid(duration, bpm, selected["bomb_grid_beats"], selected["max_candidates"]):
        index = int(np.searchsorted(times, time))
        neighbors = times[max(0, index - 1):index + 1]
        if all(abs(float(time) - other) >= selected["bomb_min_gap_seconds"] for other in neighbors):
            candidates.append({"kind": "bomb", "time": float(time), "end_time": float(time)})
    candidates.sort(key=lambda candidate: (candidate["time"], candidate["kind"]))
    return _bounded(candidates, selected)


def lighting_candidates(song, duration=None, bpm=None, settings=None):
    """Union of an independent BPM grid and local audio-onset peaks."""
    base = getattr(song, "base", song)
    duration = base.metadata["duration"] if duration is None else duration
    bpm = base.metadata["bpm"] if bpm is None else bpm
    selected = schedule_settings(settings)
    _timeline(duration, bpm)
    proposed = {round(float(time), 9): {"kind": "lighting", "time": float(time),
                                      "grid": True, "onset": False}
                for time in _grid(duration, bpm, selected["lighting_grid_beats"], selected["max_candidates"])}
    audio = base.arrays.get("features", base.arrays.get("audio"))
    frame_times = base.arrays.get("frame_times")
    if audio is not None and frame_times is not None:
        onset = np.asarray(audio)[:, -1]
        for index in range(1, len(onset) - 1):
            if (onset[index] >= selected["lighting_onset_threshold"]
                    and onset[index] > onset[index - 1] and onset[index] >= onset[index + 1]):
                time = float(frame_times[index])
                if 0 <= time < duration:
                    record = proposed.setdefault(round(time, 9), {"kind": "lighting", "time": time,
                                                                 "grid": False, "onset": False})
                    record["onset"] = True
    return _bounded(sorted(proposed.values(), key=lambda candidate: candidate["time"]), selected)


def arc_candidates(notes, settings=None):
    """Adjacent existing same-hand endpoints; never invent an endpoint."""
    selected = schedule_settings(settings)
    candidates = []
    for hand in (0, 1):
        ordered = sorted((note for note in notes if note["hand"] == hand),
                         key=lambda note: (note["time"], note["id"]))
        for head, tail in zip(ordered, ordered[1:]):
            gap = float(tail["time"] - head["time"])
            if selected["arc_min_gap_seconds"] <= gap <= selected["arc_max_gap_seconds"] and gap > 0:
                candidates.append({"kind": "arc", "time": float(head["time"]),
                                   "end_time": float(tail["time"]), "hand": hand,
                                   "endpoint_ids": [head["id"], tail["id"]], "gap": gap})
    candidates.sort(key=lambda candidate: (candidate["time"], candidate["hand"]))
    return _bounded(candidates, selected)
