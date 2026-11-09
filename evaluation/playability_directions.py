"""Directional diagnostics for the offline playability scorer."""

from itertools import groupby, pairwise
from math import atan2, floor, pi, ulp
from statistics import median
from typing import NamedTuple


FAST_REVERSAL_SECONDS = 0.25


def movement_axis_diagnostics(prepared):
    """Changes between grid displacements, rounded to the nearest 45° axis.

    Opposite displacements share an axis. A stationary step ends the observed
    trajectory. A run has a measurable dwell only when a new axis follows it.
    Dots carry a grid position and therefore participate in this diagnostic.
    """
    angles = {"0": 0, "45": 0, "90": 0}
    run_moves, run_seconds = [], []
    for hand in (0, 1):
        previous_axis = None
        moves = 0
        for note in prepared:
            if note["hand"] != hand:
                continue
            pred = note["pred"]
            if pred is None:
                continue
            dx, dy = note["x"] - pred["x"], note["y"] - pred["y"]
            if dx == dy == 0:
                previous_axis = None
                continue
            axis = floor(atan2(dy, dx) * 4 / pi + 0.5) % 4
            if previous_axis is not None:
                steps = abs(axis - previous_axis)
                degrees = 45 * min(steps, 4 - steps)
                angles[str(degrees)] += 1
                if axis != previous_axis:
                    run_moves.append(moves)
                    run_seconds.append(pred["t"] - start)
            if axis != previous_axis:
                start = pred["t"]
                moves = 0
            previous_axis = axis
            moves += 1
    return {"angle_counts": angles, "completed_runs": len(run_moves),
            "median_run_moves": median(run_moves) if run_moves else None,
            "median_run_seconds": median(run_seconds) if run_seconds else None}


# Clockwise or counterclockwise order gives the same unsigned angle. Opposite
# directions are four steps apart and have the same undirected axis.
_DIRECTION_POSITION = {0: 0, 4: 1, 2: 2, 6: 3, 1: 4, 7: 5, 3: 6, 5: 7}
_ANGLES = ("0", "45", "90", ">=135")


class _Turn(NamedTuple):
    chain: tuple  # (hand, dot-free sequence number)
    index: int    # arrival index within that sequence
    time: float
    gap: float
    angle: str


def _cut_events(prepared, hand):
    """Observe turns and completed runs on one cut axis within each arrow sequence."""
    turns, runs = [], []
    notes = (n for n in prepared if n["hand"] == hand)
    for sequence, (directed, group) in enumerate(groupby(notes, lambda n: n["d"] != 8)):
        if not directed:
            continue
        arrows = list(group)
        start_of_axis = 0
        for index, (before, after) in enumerate(pairwise(arrows), 1):
            a = _DIRECTION_POSITION[before["d"]]
            b = _DIRECTION_POSITION[after["d"]]
            steps = abs(b - a)
            angle = _ANGLES[min(steps, 8 - steps, 3)]
            turns.append(_Turn((hand, sequence), index, after["t"], after["t"] - before["t"], angle))
            if a % 4 != b % 4:
                runs.append((index - start_of_axis, after["t"] - arrows[start_of_axis]["t"]))
                start_of_axis = index
    return turns, runs


def _summary(values):
    return {"count": len(values), "median": float(median(values)) if values else None}


def _intervals(turns):
    """Time and transition count to the next selected turn, grouped by the starting angle."""
    return [(a.angle, b.time - a.time, b.index - a.index)
            for a, b in pairwise(turns) if a.chain == b.chain]


def _interval_summary(intervals):
    return {"seconds": _summary([seconds for _, seconds, _ in intervals]),
            "transitions": _summary([steps for _, _, steps in intervals])}


def _cut_diagnostics(turns, runs, minutes):
    # The per-hand lists are concatenated in hand order. The chain check in
    # _intervals keeps pooled results from connecting the two hands or crossing dots.
    changes = _intervals([turn for turn in turns if turn.angle != "0"])
    angles = {}
    for angle in _ANGLES:
        selected = [turn for turn in turns if turn.angle == angle]
        recurrence = _interval_summary(_intervals(selected))
        next_change = _interval_summary([interval for interval in changes if interval[0] == angle])
        angles[angle] = {"count": len(selected),
                         "share": len(selected) / len(turns) if turns else None,
                         "per_minute": len(selected) / minutes if turns and minutes > 0 else None,
                         "gap_seconds": _summary([turn.gap for turn in selected]),
                         "interval_seconds": recurrence["seconds"],
                         "interval_transitions": recurrence["transitions"],
                         "next_change_seconds": next_change["seconds"],
                         "next_change_transitions": next_change["transitions"]}
    non_reversals = [turn for turn in turns if turn.angle != ">=135"]
    return {"angles": angles, "gap_seconds": _summary([turn.gap for turn in turns]),
            "non_reversal_intervals": _interval_summary(_intervals(non_reversals)),
            "direction_change_intervals": _interval_summary(changes),
            "completed_cut_axis_runs": {"notes": _summary([length for length, _ in runs]),
                                        "seconds": _summary([seconds for _, seconds in runs])}}


def m5_direction_variety(prepared, minutes):
    """Fraction of required-arrow pairs turning <135° and fast reversals per map-minute.

    Dot notes break cut sequences. Diagnostics include observed inter-arrival and
    recurrence times in seconds and same-hand transitions. Each angle bucket's
    next_change fields measure the time to a turn of any nonzero angle.
    """
    per_hand = [_cut_events(prepared, hand) for hand in (0, 1)]
    turns = per_hand[0][0] + per_hand[1][0]
    runs = per_hand[0][1] + per_hand[1][1]
    diagnostics = _cut_diagnostics(turns, runs, minutes)
    reversals = diagnostics["angles"][">=135"]["count"]
    # Beat-to-second conversion may round an exact quarter slightly downward.
    fast = sum(turn.angle == ">=135" and turn.gap < FAST_REVERSAL_SECONDS - ulp(turn.time)
               for turn in turns)
    return {"non_reversal_share": (len(turns) - reversals) / len(turns) if turns else None,
            "transitions": len(turns), "fast_reversals": fast,
            "fast_reversals_per_minute": fast / minutes if turns and minutes > 0 else None,
            "movement_axis": movement_axis_diagnostics(prepared),
            "diagnostics": {"overall": diagnostics,
                            "per_hand": {str(hand): _cut_diagnostics(hand_turns, hand_runs, minutes)
                                         for hand, (hand_turns, hand_runs) in enumerate(per_hand)}}}
