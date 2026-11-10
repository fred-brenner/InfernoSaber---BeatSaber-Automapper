"""Bounded hard same-hand direction predicates and the final repair pass.

Flow repeats (``_flow_legal`` under ``v2_cleanup_same_dir_recut``) and fast
row-counter cuts (``_counter_direction`` inside a same-hand pair strictly
closer than ``v2_fast_gap_seconds``) are enforced hard at the end of the
forward pass: every violation is repaired within 9 cells x 8 arrows = 72
bounded proposals. When no non-worsening candidate exists the single
offending base note is dropped as a bounded last resort (never dots);
survivors keep ids, times, hands, metadata and caller order.
"""
import bisect
from collections import deque

from tools.config import config

_TOL = 1e-9
_MAX_PROPOSALS = 72


def _gates():
    eps = float(getattr(config, "min_time_diff", .02))
    flow_on = (getattr(config, "v2_cleanup_same_dir_recut", 0) or 0) > 0
    same_stroke = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
    fast_gap = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    counter_on = (fast_gap > 0.
                  and getattr(config, "v2_fast_counter_direction", "recut") == "recut")
    reach_on = (getattr(config, "v2_playability_enabled", False)
                and (getattr(config, "v2_playability_priority", False)
                     or getattr(config, "v2_playability_reach", False)))
    return eps, flow_on, same_stroke, fast_gap, counter_on, reach_on


def _ordered(notes):
    return sorted(notes, key=lambda n: (n["time"], n["id"]))


def _fast_ids(ordered, eps, fast):
    """Ids of notes inside a same-hand pair whose gap is eps < dt < fast.

    Each note is fast when its nearest strictly-earlier or strictly-later
    distinct-onset same-hand neighbor lies inside the fast gap — later
    same-onset stack members count through the stack's own predecessor.
    """
    ids = set()
    if fast <= 0.:
        return ids
    by_hand = {0: [], 1: []}
    for note in ordered:
        by_hand[note["hand"]].append(note)
    for members in by_hand.values():
        total = len(members)
        lo_ptr = hi_ptr = 0
        for i, note in enumerate(members):
            t = note["time"]
            while lo_ptr < total and members[lo_ptr]["time"] < t - eps:
                lo_ptr += 1
            if lo_ptr > 0 and t - members[lo_ptr - 1]["time"] < fast:
                ids.add(note["id"])
                continue
            if hi_ptr <= i:
                hi_ptr = i + 1
            while hi_ptr < total and members[hi_ptr]["time"] <= t + eps:
                hi_ptr += 1
            if hi_ptr < total and members[hi_ptr]["time"] - t < fast:
                ids.add(note["id"])
    return ids


def _findings(ordered, eps, flow_on, gap, fast, counter_on):
    """Enabled (flow, fast-counter) violation ids on a sorted same-context list."""
    from map_creation import v2_postprocessing as post
    pred, _ = post._history_context(ordered, eps)
    fast_ids = _fast_ids(ordered, eps, fast)
    flow = [note["id"] for note in ordered
            if flow_on and note["d"] != 8
            and not post._flow_legal(note["d"], pred.get(note["id"]),
                                     note["x"], note["y"], note["time"], gap)]
    counter = [note["id"] for note in ordered
               if counter_on and note["id"] in fast_ids
               and post._counter_direction(note)]
    return flow, counter


def audit_directions(notes):
    eps, flow_on, gap, fast, counter_on, _ = _gates()
    flow, counter = _findings(_ordered(notes), eps, flow_on, gap, fast, counter_on)
    return {"flow": flow, "fast_counter": counter}


def _arrow_order(note, pred):
    from map_creation import v2_postprocessing as post
    original = note["d"]
    arrows = [original]
    if pred is not None and pred[3] is not None:
        reverse = post._REVERSE_DIRECTION[pred[3]]
        if reverse != original:
            arrows.append(reverse)
    arrows += sorted((d for d in range(8) if d not in set(arrows)),
                     key=lambda d: (post._step_distance(d, original),
                                    d in (4, 5, 6, 7), d))
    return arrows


def enforce_directions(notes, index):
    """Chronologically repair every enabled flow/fast-counter violation.

    Tries the incumbent cell then the rest of the note's 3x3 hand area by
    Manhattan distance, with bounded ordered arrows per cell. A candidate is
    accepted only when its local conflict-id set does not grow past the
    incumbent's — existing conflicts may persist, new ones may not. When all
    bounded candidates fail, the single offending base note is dropped as a
    last resort and reported under ``dropped``/``dropped_notes``.
    """
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_reach import forbidden_cut
    eps, flow_on, gap, fast, counter_on, reach_on = _gates()
    stats = {"attempted": 0, "proposals": 0, "changed": 0, "relocated": 0,
             "dropped": 0, "dropped_notes": []}
    if not notes or not (flow_on or counter_on):
        return stats
    view = _ordered(notes)
    fast_ids = _fast_ids(view, eps, fast)
    bounds = {0: (0, 2), 1: (1, 3)}
    diagonals = [sum(note["d"] in (4, 5, 6, 7) for note in view)]
    total = len(view)
    fraction = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
    pending = {0: deque(), 1: deque()}
    last_eff = {0: None, 1: None}
    dropped = set()

    def pred_of(note):
        hand = note["hand"]
        while pending[hand] and pending[hand][0]["time"] < note["time"] - eps:
            earlier = pending[hand].popleft()
            before = last_eff[hand]
            if earlier["d"] != 8:
                eff = earlier["d"]
            elif before is not None and before[3] is not None:
                eff = post._dot_virtual_direction(before[3], before[1], before[2],
                                                  earlier["x"], earlier["y"])
            else:
                eff = None
            last_eff[hand] = (earlier["time"], earlier["x"], earlier["y"], eff)
        return last_eff[hand]

    for note in view:
        if note["d"] == 8:
            pending[note["hand"]].append(note)
            continue
        pred = pred_of(note)
        flow_bad = (flow_on
                    and not post._flow_legal(note["d"], pred, note["x"], note["y"],
                                             note["time"], gap))
        counter_bad = (counter_on and note["id"] in fast_ids
                       and post._counter_direction(note))
        if flow_bad or counter_bad:
            stats["attempted"] += 1
            base_cell = (note["x"], note["y"])
            baseline = index.conflict_ids(note)
            low, high = bounds[note["hand"]]
            cells = [base_cell] + sorted(
                [(x, y) for x in range(low, high + 1) for y in range(3)
                 if (x, y) != base_cell],
                key=lambda c: (abs(c[0] - base_cell[0]) + abs(c[1] - base_cell[1]),
                               c[1], c[0]))
            tried = 0
            accepted = False
            for x, y in cells:
                if tried >= _MAX_PROPOSALS:
                    break
                arrows = _arrow_order(note, pred)
                if diagonals[0] > total * fraction + _TOL:
                    arrows = ([d for d in arrows if d not in (4, 5, 6, 7)]
                              + [d for d in arrows if d in (4, 5, 6, 7)])
                for d in arrows:
                    if tried >= _MAX_PROPOSALS:
                        break
                    tried += 1
                    stats["proposals"] += 1
                    trial = {"id": note["id"], "time": note["time"],
                             "hand": note["hand"], "x": x, "y": y, "d": d}
                    if (flow_on
                            and not post._flow_legal(d, pred, x, y,
                                                     note["time"], gap)):
                        continue
                    if (counter_on and note["id"] in fast_ids
                            and post._counter_direction(trial)):
                        continue
                    if reach_on and forbidden_cut(trial):
                        continue
                    if not index.conflict_ids(trial) <= baseline:
                        continue
                    if note["d"] in (4, 5, 6, 7):
                        diagonals[0] -= 1
                    if d in (4, 5, 6, 7):
                        diagonals[0] += 1
                    note["x"], note["y"], note["d"] = x, y, d
                    stats["changed"] += 1
                    stats["relocated"] += int((x, y) != base_cell)
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                dropped.add(note["id"])
                index.remove_ids({note["id"]})
                stats["dropped"] += 1
                stats["dropped_notes"].append(
                    {"id": note["id"], "time": note["time"],
                     "proposals": tried, "flow": flow_bad,
                     "fast_counter": counter_bad})
                continue
        pending[note["hand"]].append(note)
    if dropped:
        notes[:] = [note for note in notes if note["id"] not in dropped]
        survivors = _ordered(notes)
        flow, counter = _findings(survivors, eps, flow_on, gap, fast, counter_on)
        if flow or counter:
            raise ValueError(
                f"Forward direction audit residuals after drops: "
                f"flow={flow[:5]}, fast_counter={counter[:5]}")
    return stats


class DirectionGuard:
    """Bounded direction check for appended copies/emphasis stacks.

    Sorted per-hand member and arrow references allow a bisected context:
    the closest earlier arrow onset plus its direct same-onset siblings, the
    first subsequent arrow onset plus its direct siblings and every dot
    between. Only that slice is audited; a proposal is rejected when it
    violates or creates a new violation on an incumbent.
    """

    def __init__(self, notes):
        self.members = {0: [], 1: []}
        self.times = {0: [], 1: []}
        self.arrows = {0: [], 1: []}
        self.arrow_times = {0: [], 1: []}
        self.ids = set()
        for note in _ordered(n for n in notes
                             if n.get("kind") != "chain_sample"):
            self._insert(note)

    def _insert(self, note):
        hand = note["hand"]
        key = (note["time"], note["id"])
        index = bisect.bisect(self.times[hand], key)
        self.times[hand].insert(index, key)
        self.members[hand].insert(index, note)
        if note["d"] != 8:
            index = bisect.bisect(self.arrow_times[hand], key)
            self.arrow_times[hand].insert(index, key)
            self.arrows[hand].insert(index, note)
        self.ids.add(note["id"])

    def add_many(self, proposed):
        for note in proposed:
            if note["id"] not in self.ids and note.get("kind") != "chain_sample":
                self._insert(note)

    def _window(self, hand, lo, hi, eps):
        members, arrows = self.members[hand], self.arrow_times[hand]
        start, end = 0, len(members)
        before = bisect.bisect_left(arrows, (lo, float("-inf"))) - 1
        if before >= 0:
            anchor = arrows[before][0]
            start = bisect.bisect_left(self.times[hand],
                                       (anchor - eps, float("-inf")))
        after = bisect.bisect(arrows, (hi, float("inf")))
        if after < len(arrows):
            anchor = arrows[after][0]
            end = bisect.bisect(self.times[hand], (anchor + eps, float("inf")))
        return members[start:end]

    def check(self, proposed):
        eps, flow_on, gap, fast, counter_on, _ = _gates()
        if not proposed or not (flow_on or counter_on):
            return None
        expand = max(eps, fast)
        for hand in {n["hand"] for n in proposed}:
            batch = [n for n in proposed if n["hand"] == hand]
            lo = min(n["time"] for n in batch) - expand
            hi = max(n["time"] for n in batch) + expand
            window = self._window(hand, lo, hi, eps)
            merged = sorted([*window, *batch], key=lambda n: (n["time"], n["id"]))
            old_flow, old_counter = _findings(window, eps, flow_on, gap,
                                              fast, counter_on)
            new_flow, new_counter = _findings(merged, eps, flow_on, gap,
                                              fast, counter_on)
            if ({*new_flow} - {*old_flow}) or ({*new_counter} - {*old_counter}):
                return "direction"
        return None
