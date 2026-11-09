"""Frozen human phrase-placement tails, independent of arrows and runtime config.

Times are elapsed seconds. Each hand contributes one rolling phrase per onset
after four onsets, using at most eight 20ms-anchored onset groups in four
seconds. Stack placements share one onset's weight; dots keep their real cell.
Only calibrated excess has cost, so isolated and normal-range top/outer notes
receive no reward for becoming centered. No source corpus is read at runtime.
"""

from collections import deque
import json
from pathlib import Path


_REFERENCE = json.loads(Path(__file__).with_name("v2_playability_placement_reference.json").read_text())
_BANDS = ("<3", "3-5", "5-7", "7+")
_METRICS = ("top_row_exposure", "outer_lane_exposure", "opposite_side_exposure",
            "lane_concentration", "row_concentration")
_GROUP_SECONDS = .02
_WINDOW_SECONDS = 4.
_WINDOW_EVENTS = 8
_MIN_EVENTS = 4
_TOL = 1e-9


def _groups(ordered):
    groups = [[], []]
    for i, note in enumerate(ordered):
        hand = note["hand"]
        if not groups[hand] or note["time"]-groups[hand][-1][0] > _GROUP_SECONDS+_TOL:
            groups[hand].append((note["time"], []))
        groups[hand][-1][1].append((note.get("id", i), note))
    return groups


def _phrases(groups, *, details):
    """Bounded per-onset work; fixed time/event denominators survive cell edits."""
    events = []
    for hand, hand_groups in enumerate(groups):
        recent = deque()
        for time, members in hand_groups:
            lanes, rows = [0.]*4, [0.]*3
            weight = 1./len(members)
            opposite = 0.
            for _, note in members:
                lanes[note["x"]] += weight
                rows[note["y"]] += weight
                opposite += weight*int(note["x"] >= 2 if hand == 0 else note["x"] <= 1)
            recent.append((time, members, lanes, rows, opposite))
            while len(recent) > _WINDOW_EVENTS or recent[0][0] < time-_WINDOW_SECONDS-_TOL:
                recent.popleft()
            if len(recent) < _MIN_EVENTS:
                continue
            count = len(recent)
            lane = [sum(event[2][x] for event in recent)/count for x in range(4)]
            row = [sum(event[3][y] for event in recent)/count for y in range(3)]
            values = (row[2], lane[0]+lane[3], sum(event[4] for event in recent)/count,
                      max(lane), max(row))
            identities = [identity for event in recent for identity, _ in event[1]] if details else None
            events.append((time, hand, values, identities))
    events.sort(key=lambda event: (event[0], event[1]))
    return events


def _prepare(notes, nps):
    ordered = sorted(notes, key=lambda note: (note["time"], note.get("id", 0)))
    groups = _groups(ordered)
    if nps is None:
        span = ordered[-1]["time"]-ordered[0]["time"] if ordered else 0.
        nps = sum(map(len, groups))/span if span > 0 else 0.
    nps = float(nps)
    band = _BANDS[sum(nps >= boundary for boundary in (3, 5, 7))]
    cutoffs = [_REFERENCE["metrics"][metric]["cutoffs_by_nps_band"][band] for metric in _METRICS]
    return groups, nps, band, cutoffs


def placement_costs(notes, nps=None):
    """JSON-safe per-component excess and all eligible rolling phrase events.

    ``nps`` freezes the caller's full-map retained note-span density band. The
    optional default derives the same 20ms per-hand count from ``notes``. A
    component is flagged only on strict ``severity > cutoff``; its normalized
    excess is zero throughout the allowed human range. Opposite side means
    columns 2/3 for the left hand and columns 0/1 for the right hand.
    """
    groups, nps, band, cutoffs = _prepare(notes, nps)
    events = _phrases(groups, details=True)
    metrics = {}
    for index, metric in enumerate(_METRICS):
        cutoff = cutoffs[index]
        values = [event[2][index] for event in events]
        metrics[metric] = {
            "budget": _REFERENCE["metrics"][metric]["budget"], "cutoff": cutoff,
            "eligible_events": len(values), "flagged_events": sum(value > cutoff for value in values),
            "severities": values, "event_note_ids": [event[3] for event in events],
            "excess": sum(max(0., value-cutoff)/max(1., cutoff) for value in values),
        }
    return {"reference_version": _REFERENCE["version"], "nps": nps, "nps_band": band,
            "metrics": metrics, "total_excess": sum(metric["excess"] for metric in metrics.values())}


def local_placement_cost(notes, nps):
    """Finite scalar with the same zero-inside-reference rule as full diagnostics."""
    groups, _, _, cutoffs = _prepare(notes, nps)
    events = _phrases(groups, details=False)
    return sum(sum(max(0., event[2][index]-cutoff)/max(1., cutoff) for event in events)
               for index, cutoff in enumerate(cutoffs))
