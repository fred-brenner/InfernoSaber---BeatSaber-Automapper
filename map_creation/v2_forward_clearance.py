"""Local time-window collision index for the bounded forward passes.

Times index notes once; positions and directions are read live so accepted
edits are visible immediately. The physical predicate guards concrete
same-cell, swept-cell and cut-segment conflicts only.
"""
from bisect import bisect_left, bisect_right
import math

from tools.config import config
from map_creation import v2_playability_priority_metrics as metrics

_TOL = 1e-9


def _horizon():
    return max(float(getattr(config, "min_time_diff", .02)),
               float(getattr(config, "v2_cleanup_swing_window", .15) or 0.),
               metrics.FAST_SECONDS)


def _eps():
    return float(getattr(config, "min_time_diff", .02))


def _segment(note):
    point = ((float(note["x"]), float(note["y"])),
             (float(note["x"]), float(note["y"])))
    if note.get("kind") == "chain_sample":
        return point
    step = metrics.STEP.get(note["d"])
    if step is None:
        return point
    sx, sy = step
    return ((note["x"] + sx, note["y"] + sy),
            (note["x"] - sx, note["y"] - sy))


def _swept(note):
    if note.get("kind") == "chain_sample":
        return {(float(note["x"]), float(note["y"]))}
    from map_creation.v2_postprocessing import _swept_cells
    cells = set(_swept_cells(note["x"], note["y"], note["d"]))
    cells.add((note["x"], note["y"]))
    return cells


def _point_in_cells(px, py, cells):
    return any(abs(px - cx) <= .5 + _TOL and abs(py - cy) <= .5 + _TOL
               for cx, cy in cells)


def _legal_geometry(note):
    try:
        x, y = float(note["x"]), float(note["y"])
        time = float(note["time"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(time)
            and 0 <= x < 4 and 0 <= y < 3
            and note.get("hand") in (0, 1)
            and note.get("d") in range(9)):
        return False
    if note.get("kind") != "chain_sample" and not (x.is_integer() and y.is_integer()):
        return False
    return True


def _same_cell(note, other):
    return (abs(float(note["x"]) - float(other["x"])) <= _TOL
            and abs(float(note["y"]) - float(other["y"])) <= _TOL)


class LocalNoteIndex:
    """Sorted live references; physical conflicts only, no cached coordinates."""

    def __init__(self, notes=()):
        self._seq = 0
        self.entries = []
        self.times = []
        self.add_many(notes)

    def add_many(self, records):
        for note in records:
            self.entries.append((float(note["time"]), self._seq, note))
            self._seq += 1
        self.entries.sort(key=lambda entry: (entry[0], entry[1]))
        self.times = [entry[0] for entry in self.entries]

    def nearby(self, note, window=None):
        window = _horizon() if window is None else float(window)
        time = float(note["time"])
        lo = bisect_left(self.times, time - window - _TOL)
        hi = bisect_right(self.times, time + window + _TOL)
        return [entry[2] for entry in self.entries[lo:hi]]

    def _conflicts(self, note, other, ignore_ids):
        if other is note or other["id"] == note.get("id"):
            return False
        if other["id"] in ignore_ids:
            return False
        note_sample = note.get("kind") == "chain_sample"
        other_sample = other.get("kind") == "chain_sample"
        if (note_sample and note.get("anchor_id") is not None
                and other["id"] == note["anchor_id"]):
            return False
        if (other_sample and other.get("anchor_id") is not None
                and note.get("id") == other["anchor_id"]):
            return False
        note_chain = note.get("chain_id") if note_sample else None
        other_chain = other.get("chain_id") if other_sample else None
        if (note_chain is not None and other_chain is not None
                and note_chain == other_chain):
            return False
        gap = abs(float(note["time"]) - float(other["time"]))
        if _same_cell(note, other) and gap <= _eps() + _TOL:
            return True
        if note["hand"] == other["hand"]:
            return False
        swing = max(float(getattr(config, "v2_cleanup_swing_window", .15) or 0.),
                    _eps())
        if gap <= swing + _TOL:
            if (_point_in_cells(float(note["x"]), float(note["y"]), _swept(other))
                    or _point_in_cells(float(other["x"]), float(other["y"]), _swept(note))):
                return True
        if gap <= metrics.FAST_SECONDS + _TOL:
            if metrics._segment_distance(_segment(note), _segment(other)) <= metrics.GEOMETRY_TOLERANCE:
                return True
        return False

    def conflicts(self, note, ignore_ids=()):
        ignore = set(ignore_ids)
        return any(self._conflicts(note, other, ignore)
                   for other in self.nearby(note))

    def accepts(self, note, ignore_ids=()):
        return _legal_geometry(note) and not self.conflicts(note, ignore_ids)

    def accepts_many(self, proposals, ignore_ids=()):
        proposals = list(proposals)
        ids = set()
        for note in proposals:
            if note.get("id") in ids or not _legal_geometry(note):
                return False
            ids.add(note.get("id"))
        ignore = set(ignore_ids)
        probe = LocalNoteIndex(proposals)
        for note in proposals:
            if any(self._conflicts(note, other, ignore)
                   for other in probe.nearby(note)):
                return False
            if any(self._conflicts(note, other, ignore)
                   for other in self.nearby(note)):
                return False
        return True
