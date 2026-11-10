"""Bounded chronological best-effort note postprocessing.

Each pass visits time-ordered notes once and mutates only x/y/d of base
notes. Unresolvable notes keep their incumbent and are reported. The strict
optimizer remains available through the search adapters; nothing here
reopens earlier decisions or rescans whole phrases.
"""
import math
import time as clock

import numpy as np

from tools.config import config
from tools.utils.numpy_shorts import get_factor_from_max_speed
from map_creation import v2_playability_priority_metrics as metrics
from map_creation.v2_forward_clearance import LocalNoteIndex
from map_creation.v2_forward_directions import enforce_directions

_TOL = 1e-9
_MAX_PROPOSALS = 8


def _bounds(hand):
    return (0, 2) if hand == 0 else (1, 3)


def _eps():
    return float(getattr(config, "min_time_diff", .02))


def _ordered(notes):
    return sorted(notes, key=lambda n: (float(n["time"]), n["id"]))


def _clone(note, x=None, y=None, d=None):
    clone = dict(note)
    if x is not None:
        clone["x"] = x
    if y is not None:
        clone["y"] = y
    if d is not None:
        clone["d"] = d
    return clone


def _default_cap(notes):
    from map_creation import v2_postprocessing as post
    scale = post.speed_cap_scale()
    ordered = _ordered(notes)
    count = max(int(getattr(config, "v2_speed_cap_start_notes", 30) or 0), 0)
    first = float(getattr(config, "v2_speed_cap_start_factor", .35) or 0.)
    ramp = np.linspace(first, 1., count) if count else ()
    limits = {}
    for index, note in enumerate(ordered):
        speed = config.max_speed * (ramp[index] if index < count else 1.)
        limits[note["id"]] = (speed * get_factor_from_max_speed(speed, .5, 1) * scale
                              if scale > 0 else float("inf"))
    return lambda note: limits[note["id"]]


def normalize_notes(notes, event_times, logger=None):
    from map_creation import v2_postprocessing as base
    return base.postprocess_notes(notes, event_times, logger=logger,
                                skip=(*base.LEGACY_NOTE_MUTATORS,
                                      "flow_additions", "same_direction_recut",
                                      "clearance"))


def _direction_pass(notes, index):
    from map_creation import v2_postprocessing as post
    stats = {"attempted": 0, "changed": 0, "blocked": 0, "unresolved": 0}
    gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    recut_flow = float(getattr(config, "v2_cleanup_same_dir_recut", .5) or 0.) > 0.
    fast_gap = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    recut_counter = (fast_gap > 0.
                     and getattr(config, "v2_fast_counter_direction", "recut") == "recut")
    fraction = float(getattr(config, "v2_diagonal_max_fraction", .1))
    priority = bool(getattr(config, "v2_playability_priority", False))
    diagonals = [sum(n["d"] in (4, 5, 6, 7) for n in notes)]
    state = {}
    for note in notes:
        prev = state.get(note["hand"])
        if note["d"] == 8:
            if prev is not None:
                prev["d"] = post._dot_virtual_direction(
                    prev["d"], prev["x"], prev["y"], note["x"], note["y"])
                prev["x"], prev["y"] = note["x"], note["y"]
                prev["time"] = note["time"]
                prev["axis"] = None
                prev["last_change"] = None
                prev["ordinal"] += 1
            continue
        axis = metrics.AXIS[note["d"]]
        over = diagonals[0] > len(notes) * fraction + _TOL
        if prev is None:
            prior = None
            ordinal = 0
            fast = False
            churn_bad = False
        else:
            ordinal = prev["ordinal"] + 1
            prior = (prev["time"], prev["x"], prev["y"], prev["d"])
            fast = fast_gap > 0. and note["time"] - prev["time"] < fast_gap
            churn_bad = (priority and prev["axis"] is not None and axis != prev["axis"]
                         and prev["last_change"] is not None
                         and ordinal - prev["last_change"] <= 2)
        flow_bad = (recut_flow and prior is not None
                    and not post._flow_legal(note["d"], prior, note["x"], note["y"],
                                             note["time"], gap))
        counter_bad = recut_counter and fast and post._counter_direction(note)
        diag_bad = over and note["d"] in (4, 5, 6, 7)
        if flow_bad or counter_bad or churn_bad or diag_bad:
            stats["attempted"] += 1
            prev_d = prev["d"] if prev is not None else None
            candidates = [note["d"]]
            if prev_d is not None:
                candidates.append(post._REVERSE_DIRECTION[prev_d])
            candidates += sorted(range(8), key=lambda d: (
                int(over and d in (4, 5, 6, 7)),
                post._step_distance(d, note["d"]),
                int(d in (2, 3)), d))
            seen, ordered = set(), []
            for d in candidates:
                if d not in seen:
                    seen.add(d)
                    ordered.append(d)
                if len(ordered) >= _MAX_PROPOSALS:
                    break
            pool = []
            for d in ordered:
                trial = _clone(note, d=d)
                if (recut_flow
                        and not post._flow_legal(d, prior, note["x"], note["y"],
                                                 note["time"], gap)):
                    continue
                if recut_counter and fast and post._counter_direction(trial):
                    continue
                if not index.accepts(trial):
                    stats["blocked"] += 1
                    continue
                pool.append(d)
            if over:
                cardinals = [d for d in pool if d not in (4, 5, 6, 7)]
                if cardinals:
                    pool = cardinals
                elif not (flow_bad or counter_bad or churn_bad):
                    pool = []
            if churn_bad and prev["axis"] is not None:
                same_axis = [d for d in pool if metrics.AXIS[d] == prev["axis"]]
                if same_axis:
                    pool = same_axis
            if pool and pool[0] != note["d"]:
                d = pool[0]
                if note["d"] in (4, 5, 6, 7):
                    diagonals[0] -= 1
                if d in (4, 5, 6, 7):
                    diagonals[0] += 1
                note["d"] = d
                axis = metrics.AXIS[d]
                stats["changed"] += 1
            else:
                stats["unresolved"] += 1
        if prev is not None:
            if axis != prev["axis"]:
                if prev["axis"] is not None:
                    prev["last_change"] = ordinal
                prev["axis"] = axis
            prev.update(d=note["d"], x=note["x"], y=note["y"],
                        time=note["time"], ordinal=ordinal)
        else:
            state[note["hand"]] = {"d": note["d"], "x": note["x"], "y": note["y"],
                                   "time": note["time"], "axis": metrics.AXIS[note["d"]],
                                   "last_change": None, "ordinal": 0}
    return stats


def _arrival(previous_group, notes, candidate):
    from map_creation.v2_postprocessing import calc_note_speed
    speeds = []
    for p in previous_group:
        prior = notes[p]
        speeds.append(calc_note_speed(
            [prior["x"], prior["y"], prior["hand"], prior["d"]],
            [candidate["x"], candidate["y"], candidate["hand"], candidate["d"]],
            candidate["time"] - prior["time"]))
    return min(speeds) if speeds else 0.


def _reach_cost(note, previous_group, notes, cap, scale):
    from map_creation.v2_playability_reach import forbidden_cut
    step = metrics.STEP.get(note["d"], (0, 0))
    low, high = _bounds(note["hand"])
    dx = max(low - (note["x"] + step[0]), note["x"] + step[0] - high, 0)
    dy = max(-(note["y"] + step[1]), note["y"] + step[1] - 2, 0)
    over = max(0., _arrival(previous_group, notes, note) - cap(note)) if scale > 0 else 0.
    return (int(forbidden_cut(note) or max(dx, dy) >= 2), over, max(dx, dy))


def _reach_pass(notes, index, cap):
    from map_creation.v2_playability_policy import SpeedGroups
    from map_creation import v2_postprocessing as post
    stats = {"attempted": 0, "changed": 0, "blocked": 0, "unresolved": 0}
    if not (getattr(config, "v2_playability_priority", False)
            or getattr(config, "v2_playability_reach", False)):
        return stats
    groups = SpeedGroups(notes, cap, _eps())
    scale = post.speed_cap_scale()
    for i, note in enumerate(notes):
        group = groups.previous.get(i, [])
        base_cost = _reach_cost(note, group, notes, cap, scale)
        if all(value <= _TOL for value in base_cost):
            continue
        stats["attempted"] += 1
        low, high = _bounds(note["hand"])
        prior = notes[group[0]] if group else None
        center = 1 if note["hand"] == 0 else 2
        ref_x = prior["x"] if prior is not None else center
        ref_y = prior["y"] if prior is not None else 1
        step = metrics.STEP.get(note["d"], (0, 0))
        clamp_x = min(max(note["x"], low - step[0]), high - step[0])
        clamp_y = min(max(note["y"], -step[1]), 2 - step[1])
        proposals = [(note["x"], note["y"]),
                     (min(max(note["x"], low), high), min(max(note["y"], 0), 2)),
                     (clamp_x, clamp_y),
                     (note["x"] + int(np.sign(ref_x - note["x"])), note["y"]),
                     (note["x"], note["y"] + int(np.sign(ref_y - note["y"]))),
                     (note["x"] + int(np.sign(center - note["x"])), note["y"]),
                     (note["x"], note["y"] + int(np.sign(1 - note["y"]))),
                     (center, 1)]
        seen, accepted = set(), False
        for x, y in proposals[:_MAX_PROPOSALS]:
            if (x, y) in seen or not (low <= x <= high and 0 <= y <= 2):
                continue
            seen.add((x, y))
            trial = _clone(note, x=x, y=y)
            if _reach_cost(trial, group, notes, cap, scale) >= base_cost:
                continue
            if not index.accepts(trial):
                stats["blocked"] += 1
                continue
            note["x"], note["y"] = x, y
            accepted = True
            break
        stats["changed" if accepted else "unresolved"] += 1
    return stats


def _collision_pass(notes, index):
    stats = {"attempted": 0, "changed": 0, "blocked": 0, "unresolved": 0}
    for note in notes:
        if not index.conflicts(note):
            continue
        stats["attempted"] += 1
        low, high = _bounds(note["hand"])
        center = 1 if note["hand"] == 0 else 2
        other_cols = [x for x in range(low, high + 1) if x != note["x"]]
        proposals = [(note["x"] - 1, note["y"]), (note["x"] + 1, note["y"]),
                     (note["x"], note["y"] - 1), (note["x"], note["y"] + 1),
                     (center, 1)] + [(x, note["y"]) for x in other_cols]
        accepted = False
        seen = set()
        for x, y in proposals[:_MAX_PROPOSALS]:
            if ((x, y) in seen or (x, y) == (note["x"], note["y"])
                    or not (low <= x <= high and 0 <= y <= 2)):
                continue
            seen.add((x, y))
            if index.accepts(_clone(note, x=x, y=y)):
                note["x"], note["y"] = x, y
                accepted = True
                break
        stats["changed" if accepted else "unresolved"] += 1
    return stats


def _repeated_pass(notes, index, rng):
    stats = {"attempted": 0, "changed": 0, "blocked": 0, "unresolved": 0}
    previous = {}
    for note in notes:
        hand = note["hand"]
        prior = previous.get(hand)
        previous[hand] = note
        if (prior is None or note["d"] not in metrics.STEP
                or note["time"] - prior["time"] <= _eps() + _TOL
                or (note["x"], note["y"]) != (prior["x"], prior["y"])):
            continue
        same_onset = [n for n in index.nearby(note, window=_eps())
                      if n["hand"] == hand
                      and abs(n["time"] - note["time"]) <= _eps() + _TOL]
        if len(same_onset) != 1:
            continue
        stats["attempted"] += 1
        dx, dy = metrics.STEP[note["d"]]
        x = note["x"] if dy else note["x"] - dx
        y = note["y"] - dy
        low, high = _bounds(hand)
        if not (low <= x <= high and 0 <= y <= 2):
            stats["unresolved"] += 1
            continue
        trial = _clone(note, x=x, y=y)
        if not index.accepts(trial):
            stats["blocked"] += 1
            continue
        if y == 2 and float(rng.random()) >= .5:
            stats["blocked"] += 1
            continue
        note["x"], note["y"] = x, y
        stats["changed"] += 1
    return stats


def _visibility_pass(notes, index):
    from map_creation import v2_playability_visibility as visibility
    stats = {"attempted": 0, "changed": 0, "blocked": 0, "unresolved": 0}
    if not getattr(config, "v2_playability_priority", False):
        return stats
    span = (max(n["time"] for n in notes) - min(n["time"] for n in notes)
            if notes else 0.)
    nps = len(notes) / span if span > 0 else 0.
    costs = visibility.visibility_costs(notes, nps=nps)
    if not costs["supported"]:
        return stats
    cutoff = costs["cutoff"]
    by_id = {n["id"]: n for n in notes}
    for event in costs["events"]:
        severity = math.fsum(
            weight for note_id, weight in zip(event["note_ids"], event["weights"])
            if visibility.central_top(by_id.get(note_id, {"x": -1, "y": -1})))
        if severity <= cutoff:
            continue
        for note_id in event["group_ids"]:
            note = by_id.get(note_id)
            if note is None or not visibility.central_top(note):
                continue
            stats["attempted"] += 1
            low, high = _bounds(note["hand"])
            outer = low if note["hand"] == 0 else high
            accepted = False
            for x, y in ((note["x"], 1), (outer, note["y"])):
                trial = _clone(note, x=x, y=y)
                if visibility.central_top(trial) or not index.accepts(trial):
                    continue
                note["x"], note["y"] = x, y
                accepted = True
                break
            stats["changed" if accepted else "unresolved"] += 1
    return stats


def clear_center_view(notes, *, rng=None, index=None):
    import random
    rng = random if rng is None else rng
    index = LocalNoteIndex(notes) if index is None else index
    stats = {"moved": 0, "blocked": 0, "draws": 0}
    for note in _ordered(list(notes)):
        eligible = ((note["hand"] == 0 and note["x"] == 1)
                    or (note["hand"] == 1 and note["x"] == 2))
        weight = {0: 0., 1: 1., 2: .25}.get(note["y"], 0.)
        if not eligible or weight <= 0.:
            continue
        stats["draws"] += 1
        if float(rng.random()) >= .8 * weight:
            continue
        x = note["x"] - 1 if note["hand"] == 0 else note["x"] + 1
        low, high = _bounds(note["hand"])
        if not (low <= x <= high) or not index.accepts(_clone(note, x=x)):
            stats["blocked"] += 1
            continue
        note["x"] = x
        stats["moved"] += 1
    return stats


def repair_notes(notes, *, rng=None, cap=None, logger=None):
    import random
    from map_creation import v2_postprocessing as post
    rng = random if rng is None else rng
    view = _ordered(notes)
    index = LocalNoteIndex(view)
    cap = _default_cap(notes) if cap is None else cap
    report = {}
    start = clock.monotonic()
    with post.playability_generation_phase():
        report["direction"] = _direction_pass(view, index)
        report["reach"] = _reach_pass(view, index, cap)
        report["collision"] = _collision_pass(view, index)
        report["repeated"] = _repeated_pass(view, index, rng)
        report["visibility"] = _visibility_pass(view, index)
        report["center"] = clear_center_view(notes, rng=rng, index=index)
        report["hard_direction"] = enforce_directions(notes, index)
    report["seconds"] = clock.monotonic() - start
    return report
