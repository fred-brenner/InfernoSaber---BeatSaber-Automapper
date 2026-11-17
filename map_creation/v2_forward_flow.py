import bisect
import math
from collections.abc import Sequence

from tools.config import config

_MAX_CANDIDATES = 12
_WAVE_NOTES = 8
_WAVE_PHASES = 4
_RUN_GAP = .4
_RECOVERY_GAP = .6
_TOL = 1e-9


def _eps():
    return float(getattr(config, "min_time_diff", .02))


def _bounds(hand):
    return (0, 2) if hand == 0 else (1, 3)


def _outside(value, low, high):
    return max(low - value, value - high, 0)


def enabled():
    return (bool(getattr(config, "v2_pattern_flow", False))
            and getattr(config, "v2_note_postprocessing_mode", "forward") == "forward"
            and getattr(config, "v2_playability_enabled", False)
            and (getattr(config, "v2_playability_priority", False)
                 or getattr(config, "v2_playability_reach", False))
            and not getattr(config, "v2_cleanup_note_heuristics", False))


def prepared_rebound(note, previous):
    from map_creation import v2_postprocessing as post
    from map_creation import v2_playability_priority_metrics as metrics
    if (previous is None or note.get("d") not in range(8)
            or previous.get("d") not in range(8)
            or previous.get("hand", note["hand"]) != note["hand"]):
        return False
    dt = float(note["time"]) - float(previous["time"])
    if not (_eps() < dt <= _RECOVERY_GAP + _TOL):
        return False
    if post._step_distance(int(note["d"]), int(previous["d"])) < 3:
        return False
    sx, sy = metrics.STEP[note["d"]]
    entry_x, entry_y = note["x"] + sx, note["y"] + sy
    px, py = metrics.STEP[previous["d"]]
    exit_x, exit_y = previous["x"] - px, previous["y"] - py
    low, high = _bounds(note["hand"])
    crossed = []
    if entry_x < low:
        crossed.append(exit_x < low)
    elif entry_x > high:
        crossed.append(exit_x > high)
    if entry_y < 0:
        crossed.append(exit_y < 0)
    elif entry_y > 2:
        crossed.append(exit_y > 2)
    if not crossed or not all(crossed):
        return False
    return (entry_x - exit_x) ** 2 + (entry_y - exit_y) ** 2 <= 1 + _TOL


def _terminal_accent(note, following, group):
    from map_creation import v2_playability_priority_metrics as metrics
    if (following is not None
            and float(following["time"]) - float(note["time"]) < _RECOVERY_GAP - _TOL):
        return False
    if (len(group) not in (2, 3)
            or note["id"] not in {member["id"] for member in group}):
        return False
    if any(member["hand"] != note["hand"] or member["d"] != note["d"]
           for member in group) or note["d"] not in range(8):
        return False
    cells = sorted({(member["x"], member["y"]) for member in group})
    if len(cells) != len(group):
        return False
    sx, sy = metrics.STEP[note["d"]]
    if sx == 0:
        cells.sort(key=lambda cell: cell[1])
        return all(b[0] == a[0] and b[1] - a[1] == 1
                   for a, b in zip(cells, cells[1:]))
    if sy == 0:
        cells.sort(key=lambda cell: cell[0])
        return all(b[0] - a[0] == 1 and b[1] == a[1]
                   for a, b in zip(cells, cells[1:]))
    cells.sort(key=lambda cell: cell[0] * sx + cell[1] * sy)
    return all(b[0] - a[0] == sx and b[1] - a[1] == sy
               for a, b in zip(cells, cells[1:]))


def style_cost(note, previous, following, group=()):
    from map_creation import v2_playability_priority_metrics as metrics
    incoming = (math.inf if previous is None
                else float(note["time"]) - float(previous["time"]))
    outgoing = (math.inf if following is None
                else float(following["time"]) - float(note["time"]))
    fastest = min(incoming, outgoing)
    center = 0.
    if (note["x"] in (1, 2) and note["y"] == 1
            and not _terminal_accent(note, following, group)):
        center = 12. if fastest < .4 else 6.
    horizontal = 0.
    if note["d"] in (2, 3):
        horizontal = 6. if fastest < .4 else 1.
    rebound = prepared_rebound(note, previous)
    boundary = 0.
    if note["d"] in range(8) and not rebound:
        sx, sy = metrics.STEP[note["d"]]
        low, high = _bounds(note["hand"])
        excursion = max(_outside(note["x"] + sx, low, high),
                        _outside(note["y"] + sy, 0, 2))
        if excursion > 0.:
            boundary = excursion * (1. + .4 / (.4 + incoming))
    upper = 0.
    if (note["x"] in (1, 2) and note["y"] == 2 and note["d"] in (1, 2, 3, 6, 7)
            and not rebound):
        upper = 1.
    return (center, horizontal, boundary, upper)


class _MemberView(Sequence):
    __slots__ = ("members", "lo", "hi")

    def __init__(self, members, lo, hi):
        self.members, self.lo, self.hi = members, lo, hi

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.members[self.lo + index]


class FlowContext:
    def __init__(self, notes):
        eps = _eps()
        self._previous = {}
        self._following = {}
        self._group = {}
        self._groups = {}
        self._members = {0: [], 1: []}
        self._pos = {}
        for hand in (0, 1):
            members = sorted((n for n in notes
                              if n["hand"] == hand
                              and n.get("kind") != "chain_sample"),
                             key=lambda n: (float(n["time"]), n["id"]))
            times = [float(n["time"]) for n in members]
            self._members[hand] = members
            for index, note in enumerate(members):
                self._pos[note["id"]] = index
                t = float(note["time"])
                lo = bisect.bisect_left(times, t - eps)
                hi = bisect.bisect_right(times, t + eps)
                self._previous[note["id"]] = members[lo - 1] if lo else None
                self._following[note["id"]] = (members[hi]
                                               if hi < len(members) else None)
                key = (hand, lo, hi)
                group = self._groups.get(key)
                if group is None:
                    group = _MemberView(members, lo, hi)
                    self._groups[key] = group
                self._group[note["id"]] = group

    def previous(self, note):
        return self._previous.get(note["id"])

    def following(self, note):
        return self._following.get(note["id"])

    def group(self, note):
        return self._group.get(note["id"], ())


def _tuple_note(previous, hand):
    if previous is None or previous[3] is None:
        return None
    return {"time": previous[0], "x": previous[1], "y": previous[2],
            "d": previous[3], "hand": hand}


def _pred_tuple(context, note):
    from map_creation import v2_postprocessing as post
    previous = context.previous(note)
    if previous is None:
        return None
    if previous["d"] != 8:
        return (float(previous["time"]), previous["x"], previous["y"],
                previous["d"])
    members = context._members[note["hand"]]
    base = context._pos[previous["id"]]
    run = []
    while base >= 0 and members[base]["d"] == 8:
        run.append(base)
        base -= 1
    if base < 0:
        eff, px, py = None, None, None
    else:
        eff, px, py = members[base]["d"], members[base]["x"], members[base]["y"]
    for j in reversed(run):
        eff = post._dot_virtual_direction(eff, px, py,
                                          members[j]["x"], members[j]["y"])
        px, py = members[j]["x"], members[j]["y"]
    return (float(previous["time"]), previous["x"], previous["y"], eff)


def _arrival(previous_members, note):
    from map_creation.v2_postprocessing import calc_note_speed
    speeds = [calc_note_speed(
        [p["x"], p["y"], p["hand"], p["d"]],
        [note["x"], note["y"], note["hand"], note["d"]],
        float(note["time"]) - float(p["time"])) for p in previous_members]
    return min(speeds) if speeds else 0.


def _violations(members):
    from map_creation import v2_postprocessing as post
    from map_creation.v2_forward_directions import _fast_ids
    eps = _eps()
    fast = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    counter_on = (fast > 0.
                  and getattr(config, "v2_fast_counter_direction", "recut") == "recut")
    flow_on = (getattr(config, "v2_cleanup_same_dir_recut", 0) or 0) > 0
    if not (flow_on or counter_on):
        return set()
    ordered = sorted(members, key=lambda n: (float(n["time"]), n["id"]))
    pred, _ = post._history_context(ordered, eps)
    fast_ids = _fast_ids(ordered, eps, fast)
    bad = set()
    for note in ordered:
        if note["d"] == 8:
            continue
        if (flow_on
                and not post._flow_legal(note["d"], pred.get(note["id"]),
                                         note["x"], note["y"], note["time"],
                                         eps)):
            bad.add(note["id"])
            continue
        if (counter_on and note["id"] in fast_ids
                and post._counter_direction(note)
                and not prepared_rebound(
                    note, _tuple_note(pred.get(note["id"]), note["hand"]))):
            bad.add(note["id"])
    return bad


def _center_crossing(chain):
    t0, t1 = 0., 1.
    x, y = float(chain["x"]), float(chain["y"])
    for start, span, lo, hi in ((x, float(chain["tx"]) - x, .5, 2.5),
                                (y, float(chain["ty"]) - y, .5, 1.5)):
        if abs(span) <= _TOL:
            if not lo - _TOL <= start <= hi + _TOL:
                return False
            continue
        a, b = (lo - start) / span, (hi - start) / span
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1 + _TOL:
            return False
    return t0 <= t1 + _TOL


def check_additions(before, proposed):
    if not proposed:
        return None
    merged = sorted([*before, *proposed],
                    key=lambda n: (float(n["time"]), n["id"]))
    prior = FlowContext(before)
    after = FlowContext(merged)
    for note in proposed:
        if any(style_cost(note, after.previous(note), after.following(note),
                          after.group(note))):
            return "pattern_flow"
    for note in before:
        old = style_cost(note, prior.previous(note), prior.following(note),
                         prior.group(note))
        new = style_cost(note, after.previous(note), after.following(note),
                         after.group(note))
        if any(n_ > o_ + _TOL for n_, o_ in zip(new, old)):
            return "pattern_flow"
    return None


def audit_patterns(notes):
    context = FlowContext(notes)
    counts = {"eligible": 0, "center_middle": 0, "horizontal": 0,
              "unprepared_boundary": 0, "upper_center_down": 0}
    keys = ("center_middle", "horizontal", "unprepared_boundary",
            "upper_center_down")
    for note in notes:
        if note.get("kind") == "chain_sample":
            continue
        counts["eligible"] += 1
        costs = style_cost(note, context.previous(note),
                           context.following(note), context.group(note))
        for key, value in zip(keys, costs):
            counts[key] += int(value > _TOL)
    return counts


def repair_patterns(notes, index, cap):
    from map_creation import v2_forward_patterns
    return v2_forward_patterns.repair_patterns(notes, index, cap)
