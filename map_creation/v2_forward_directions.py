"""Bounded hard same-hand direction predicates and the final repair pass.

Flow repeats (``_flow_legal`` under ``v2_cleanup_same_dir_recut``) and fast
row-counter cuts (``_counter_direction`` inside a same-hand pair strictly
closer than ``v2_fast_gap_seconds``) are enforced hard at the end of the
forward pass: detection is one vectorized pass per hand (``_detect``) and
only flagged notes enter a chronological repair worklist, each within
9 cells x 8 arrows = 72 bounded proposals. Only true same-onset stacks
(inside ``min_time_diff``) stay exempt from the >=135° turn — every
distinct-onset same-hand pair is audited. When no non-worsening candidate
exists the single offending base note is dropped as a bounded last resort
(never dots), and ``v2_hard_direction_policy = "drop"`` removes flagged
notes directly without the proposal search; survivors keep ids, times,
hands, metadata and caller order. A separate chronological scan drops
later notes that still occupy a survivor's cell — same hand within
``v2_cleanup_same_dir_gap``, opposite hand within
``v2_cleanup_swing_window`` — without any candidate search.
"""
import bisect
import heapq
import math

import numpy as np

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
    same_hand_overlap = max(eps, float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0))
    cross_overlap = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0))
    return (eps, flow_on, same_stroke, fast_gap, counter_on, reach_on,
            same_hand_overlap, cross_overlap)


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


def _detect(notes):
    """Strict hard-rule violation ids from one vectorized pass per hand.

    Mirrors the in-phase ``_flow_legal``/``_counter_direction`` audit
    exactly: the predecessor is the last same-hand note strictly before
    ``time - min_time_diff`` (``searchsorted`` on the sorted per-hand
    times), dots inherit ``_dot_virtual_direction`` in member order (the
    only sequential loop — dots are rare), a distinct-onset pair violates
    flow below ``v2_flow_min_turn_steps`` 45°-steps with the legacy
    >=2-cell-travel exception when it is configured <= 2, and fast
    membership mirrors the ``_fast_ids`` strictly-earlier/later windows.
    Returns id sets ``{"flow", "counter"}``; slow-turn leniency is not part
    of the strict core and overlap is deliberately left to ``_overlap_scan``.
    """
    from map_creation import v2_postprocessing as post
    eps, flow_on, _, fast, counter_on, _, _, _ = _gates()
    flagged = {"flow": set(), "counter": set()}
    if not notes or not (flow_on or counter_on):
        return flagged
    ordered = _ordered(notes)
    configured = getattr(config, "v2_flow_min_turn_steps", None)
    min_turn = 3 if configured is None else int(configured)
    angle = np.asarray([post._DIRECTION_ANGLE[d] for d in range(8)] + [0])
    for hand in (0, 1):
        members = [note for note in ordered if note["hand"] == hand]
        if not members:
            continue
        ids = [note["id"] for note in members]
        times = np.asarray([float(note["time"]) for note in members])
        xs = np.asarray([float(note["x"]) for note in members])
        ys = np.asarray([float(note["y"]) for note in members])
        ds = np.asarray([int(note["d"]) for note in members])
        total = len(members)
        effective = np.where(ds != 8, ds, -1)
        for i in np.flatnonzero(ds == 8):
            if i and effective[i - 1] >= 0:
                implied = post._dot_virtual_direction(
                    int(effective[i - 1]), float(xs[i - 1]), float(ys[i - 1]),
                    float(xs[i]), float(ys[i]))
                if implied is not None:
                    effective[i] = implied
        before = np.searchsorted(times, times - eps, side="left") - 1
        has_pred = before >= 0
        pred_eff = np.full(total, -1, dtype=int)
        pred_eff[has_pred] = effective[before[has_pred]]
        if flow_on:
            known = has_pred & (pred_eff >= 0)
            pa = angle[np.clip(ds, 0, 8)]
            pb = angle[np.clip(pred_eff, 0, 8)]
            steps = np.minimum((pa - pb) % 8, (pb - pa) % 8)
            if min_turn <= 2:
                travel = (np.abs(xs - xs[np.clip(before, 0, None)])
                          + np.abs(ys - ys[np.clip(before, 0, None)]))
                illegal = (steps <= 1) | ((steps == 2) & (travel < 2.))
            else:
                illegal = steps < min_turn
            mask = (ds != 8) & known & illegal
            flagged["flow"].update(ids[i] for i in np.flatnonzero(mask))
        if counter_on:
            lo, hi = before + 1, np.searchsorted(times, times + eps,
                                                 side="right")
            near = np.zeros(total, dtype=bool)
            earlier = lo > 0
            near[earlier] = (times[earlier]
                             - times[lo[earlier] - 1]) < fast
            later = hi < total
            near[later] |= (times[hi[later]] - times[later]) < fast
            counter = (near
                       & (((ys == 0) & np.isin(ds, (0, 4, 5)))
                          | ((ys == 2) & np.isin(ds, (1, 6, 7)))))
            flagged["counter"].update(ids[i] for i in np.flatnonzero(counter))
    return flagged


def audit_directions(notes):
    eps, flow_on, _, fast, counter_on, _, _, _ = _gates()
    ordered = _ordered(notes)
    flagged = _detect(notes)
    flow_ids, counter_ids = flagged["flow"], flagged["counter"]
    if flow_ids:
        from map_creation import v2_postprocessing as post
        if post._slow_flow_enabled():
            # Strict ids stay exact: outside the generation phase the
            # slow-turn leniency may still legalize a flagged pair.
            pred, _ = post._history_context(ordered, eps)
            by_id = {note["id"]: note for note in ordered}
            flow_ids = {i for i in flow_ids
                        if not post._flow_legal(by_id[i]["d"], pred.get(i),
                                                by_id[i]["x"], by_id[i]["y"],
                                                by_id[i]["time"], eps)}
    return {"flow": [note["id"] for note in ordered if note["id"] in flow_ids],
            "fast_counter": [note["id"] for note in ordered
                             if note["id"] in counter_ids]}


def _overlap_scan(ordered, same_hand_gap, cross_gap):
    """Ids of notes that still occupy a survivor's cell inside the windows.

    One chronological pass tracks the last survivor per hand and the last
    survivor per cell. A note is flagged when the last same-hand survivor
    sits on its ``(x, y)`` no earlier than ``same_hand_gap`` ago, or the
    last survivor on its cell is the opposite hand no earlier than
    ``cross_gap`` ago. Flagged notes never update the survivor maps, so a
    chain of repeats collapses onto the first survivor and equal-time pairs
    deterministically drop the later ``(time, id)`` arrival.
    """
    flagged = []
    last_by_hand, last_by_cell = {}, {}
    for note in ordered:
        cell = (note["x"], note["y"])
        previous = last_by_hand.get(note["hand"])
        occupied = last_by_cell.get(cell)
        if (previous is not None
                and (previous["x"], previous["y"]) == cell
                and note["time"] - previous["time"] <= same_hand_gap + _TOL):
            flagged.append(note["id"])
            continue
        if (occupied is not None and occupied["hand"] != note["hand"]
                and note["time"] - occupied["time"] <= cross_gap + _TOL):
            flagged.append(note["id"])
            continue
        last_by_hand[note["hand"]] = note
        last_by_cell[cell] = note
    return flagged


def audit_overlaps(notes):
    """Residual same-cell overlaps under the same enablement as the repair."""
    _, flow_on, _, _, counter_on, _, same_hand, cross = _gates()
    flagged = (_overlap_scan(_ordered(notes), same_hand, cross)
               if flow_on or counter_on else [])
    return {"overlap": flagged}


def enforce_overlaps(notes, index):
    """Drop the later note of each residual same-cell overlap, no search.

    Runs before the direction repair so removals merge cleanly: every note
    flagged by ``_overlap_scan`` is removed through ``index.remove_ids``
    and reported under ``dropped``/``dropped_notes`` with ``proposals`` 0
    and reason ``"overlap"``. Bounded O(n log n) — removal only, notes are
    never moved here.
    """
    _, flow_on, _, _, counter_on, _, same_hand, cross = _gates()
    stats = {"checked": 0, "dropped": 0, "dropped_notes": []}
    if not notes or not (flow_on or counter_on):
        return stats
    view = _ordered(notes)
    stats["checked"] = len(view)
    flagged = _overlap_scan(view, same_hand, cross)
    if not flagged:
        return stats
    dropped = set(flagged)
    for note in view:
        if note["id"] in dropped:
            index.remove_ids({note["id"]})
            stats["dropped_notes"].append(
                {"id": note["id"], "time": note["time"],
                 "proposals": 0, "reason": "overlap"})
    stats["dropped"] = len(dropped)
    notes[:] = [note for note in notes if note["id"] not in dropped]
    return stats


def _overlap_conflicts(index, note, ignore_ids, same_hand, cross):
    """A live neighbor already sits on ``note``'s cell inside the windows."""
    cell = (note["x"], note["y"])
    for other in index.nearby(note, window=max(same_hand, cross)):
        if other["id"] in ignore_ids or other["id"] == note["id"]:
            continue
        if (other["x"], other["y"]) != cell:
            continue
        dt = abs(float(other["time"]) - float(note["time"]))
        if other["hand"] == note["hand"]:
            if dt <= same_hand + _TOL:
                return True
        elif dt <= cross + _TOL:
            return True
    return False


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

    ``_detect`` flags the violations once over per-hand arrays; only those
    notes enter the worklist — clean notes are never touched in Python.
    Each flagged note is re-verified against the live same-hand survivor
    context (a prior repair may already have fixed it), then tried on the
    incumbent cell and the rest of its 3x3 hand area by Manhattan distance,
    with bounded ordered arrows per cell. A candidate is accepted only when
    its local conflict-id set does not grow past the incumbent's — existing
    conflicts may persist, new ones may not. When all bounded candidates
    fail the single offending base note is dropped as a last resort and
    reported under ``dropped``/``dropped_notes``;
    ``v2_hard_direction_policy = "drop"`` skips the search and drops every
    still-violating note directly. Every accepted edit or drop rechecks the
    same-hand survivors whose predecessor it could have changed, so a
    repair breaking its successor is caught inside this call.
    """
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_reach import forbidden_cut
    eps, flow_on, _, fast, counter_on, reach_on, same_hand, cross = _gates()
    stats = {"attempted": 0, "proposals": 0, "changed": 0, "relocated": 0,
             "dropped": 0, "dropped_notes": []}
    if not notes or not (flow_on or counter_on):
        return stats
    view = _ordered(notes)
    detected = _detect(view)
    fast_ids = _fast_ids(view, eps, fast)
    bounds = {0: (0, 2), 1: (1, 3)}
    diagonals = [sum(note["d"] in (4, 5, 6, 7) for note in view)]
    total = len(view)
    fraction = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
    drop_only = getattr(config, "v2_hard_direction_policy", "repair") == "drop"
    members = {hand: [note for note in view if note["hand"] == hand]
               for hand in (0, 1)}
    times = {hand: [float(note["time"]) for note in members[hand]]
             for hand in (0, 1)}
    alive = {hand: [True] * len(members[hand]) for hand in (0, 1)}
    pos = {note["id"]: i for hand in (0, 1)
           for i, note in enumerate(members[hand])}
    by_id = {note["id"]: note for note in view}
    dropped = set()
    repaired = set()
    done = set()
    heap, in_heap = [], set()
    for nid in detected["flow"] | detected["counter"]:
        heapq.heappush(heap, (by_id[nid]["time"], nid))
        in_heap.add(nid)

    def live_pred(note):
        """Last same-hand survivor strictly before ``time - eps`` plus its
        effective direction (dot back-propagation over live values)."""
        hand = note["hand"]
        seq, ts, al = members[hand], times[hand], alive[hand]
        p = bisect.bisect_left(ts, note["time"] - eps) - 1
        while p >= 0 and not al[p]:
            p -= 1
        if p < 0:
            return None
        base, run = p, []
        while base >= 0 and (not al[base] or seq[base]["d"] == 8):
            if al[base]:
                run.append(base)
            base -= 1
        if base < 0:
            eff, px, py = None, None, None
        else:
            eff, px, py = seq[base]["d"], seq[base]["x"], seq[base]["y"]
        for j in reversed(run):
            eff = post._dot_virtual_direction(eff, px, py,
                                              seq[j]["x"], seq[j]["y"])
            px, py = seq[j]["x"], seq[j]["y"]
        return (seq[p]["time"], seq[p]["x"], seq[p]["y"], eff)

    def violates(note, pred):
        """Enabled (flow, fast-counter) violations under the live context."""
        flow_bad = (flow_on and note["d"] != 8
                    and not post._flow_legal(note["d"], pred, note["x"],
                                             note["y"], note["time"], eps))
        counter_bad = (counter_on and note["id"] in fast_ids
                       and post._counter_direction(note))
        return flow_bad, counter_bad

    def recheck_successors(hand, p):
        """Flag still-violating survivors whose predecessor context the
        mutation at member ``p`` could have changed (itself plus its run of
        consecutive dot successors, bounded by the next survivor arrow)."""
        seq, ts, al = members[hand], times[hand], alive[hand]
        boundary = p + 1
        while (boundary < len(seq)
               and (not al[boundary] or seq[boundary]["d"] == 8)):
            boundary += 1
        limit = ts[boundary] + eps + _TOL if boundary < len(seq) else math.inf
        for j in range(p + 1, len(seq)):
            if ts[j] > limit:
                break
            if not al[j] or seq[j]["d"] == 8:
                continue
            nid = seq[j]["id"]
            if nid in in_heap or (nid in done and nid not in repaired):
                continue
            flow_bad, counter_bad = violates(seq[j], live_pred(seq[j]))
            if flow_bad or counter_bad:
                heapq.heappush(heap, (seq[j]["time"], nid))
                in_heap.add(nid)

    def drop(note, tried, flow_bad, counter_bad):
        dropped.add(note["id"])
        index.remove_ids({note["id"]})
        alive[note["hand"]][pos[note["id"]]] = False
        stats["dropped"] += 1
        stats["dropped_notes"].append(
            {"id": note["id"], "time": note["time"], "proposals": tried,
             "flow": flow_bad, "fast_counter": counter_bad})
        recheck_successors(note["hand"], pos[note["id"]])

    while heap:
        _, nid = heapq.heappop(heap)
        in_heap.discard(nid)
        if nid in done and nid not in repaired:
            continue
        note = by_id[nid]
        hand = note["hand"]
        pred = live_pred(note)
        flow_bad, counter_bad = violates(note, pred)
        if nid in repaired:
            # Bounded second chance: a note repaired once that still flags
            # is dropped directly, never re-searched. Chronological order
            # makes this unreachable — every mutation affecting a note's
            # context lands before the note itself is popped.
            if flow_bad or counter_bad:
                drop(note, 0, flow_bad, counter_bad)
                repaired.discard(nid)
            done.add(nid)
            continue
        if not (flow_bad or counter_bad):
            done.add(nid)
            continue
        stats["attempted"] += 1
        tried = 0
        accepted = False
        if not drop_only:
            base_cell = (note["x"], note["y"])
            baseline = index.conflict_ids(note)
            low, high = bounds[note["hand"]]
            cells = [base_cell] + sorted(
                [(x, y) for x in range(low, high + 1) for y in range(3)
                 if (x, y) != base_cell],
                key=lambda c: (abs(c[0] - base_cell[0]) + abs(c[1] - base_cell[1]),
                               c[1], c[0]))
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
                                                     note["time"], eps)):
                        continue
                    if (counter_on and note["id"] in fast_ids
                            and post._counter_direction(trial)):
                        continue
                    if reach_on and forbidden_cut(trial):
                        continue
                    if _overlap_conflicts(index, trial, dropped | getattr(index, "_dropped", frozenset()),
                                          same_hand, cross):
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
        if accepted:
            repaired.add(nid)
            recheck_successors(hand, pos[nid])
        else:
            drop(note, tried, flow_bad, counter_bad)
        done.add(nid)
    if dropped:
        notes[:] = [note for note in notes if note["id"] not in dropped]
        survivors = _ordered(notes)
        flow, counter = _findings(survivors, eps, flow_on, eps, fast, counter_on)
        if flow or counter:
            raise ValueError(
                f"Forward direction audit residuals after drops: "
                f"flow={flow[:5]}, fast_counter={counter[:5]}")
    return stats


class DirectionGuard:
    """Bounded direction/overlap check for appended copies/emphasis stacks.

    Sorted per-hand member and arrow references allow a bisected context:
    the closest earlier arrow onset plus its direct same-onset siblings, the
    first subsequent arrow onset plus its direct siblings and every dot
    between. Only that slice is audited; a proposal is rejected when it
    violates or creates a new violation on an incumbent, or when it shares
    a cell with a member or sibling proposal inside the same-hand or
    cross-hand overlap windows.
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

    def _overlap_blocked(self, note, proposed, same_hand, cross):
        """``note`` shares a cell inside an overlap window with member/proposed."""
        cell = (note["x"], note["y"])
        for hand, gap in ((note["hand"], same_hand), (1 - note["hand"], cross)):
            times, members = self.times[hand], self.members[hand]
            lo = bisect.bisect_left(times, (note["time"] - gap - _TOL, float("-inf")))
            hi = bisect.bisect(times, (note["time"] + gap + _TOL, float("inf")))
            if any((other["x"], other["y"]) == cell for other in members[lo:hi]):
                return True
        for other in proposed:
            if other is note or (other["x"], other["y"]) != cell:
                continue
            gap = same_hand if other["hand"] == note["hand"] else cross
            if abs(other["time"] - note["time"]) <= gap + _TOL:
                return True
        return False

    def check(self, proposed):
        (eps, flow_on, _, fast, counter_on, _,
         same_hand, cross) = _gates()
        if not proposed or not (flow_on or counter_on):
            return None
        expand = max(eps, fast)
        for hand in {n["hand"] for n in proposed}:
            batch = [n for n in proposed if n["hand"] == hand]
            lo = min(n["time"] for n in batch) - expand
            hi = max(n["time"] for n in batch) + expand
            window = self._window(hand, lo, hi, eps)
            merged = sorted([*window, *batch], key=lambda n: (n["time"], n["id"]))
            old_flow, old_counter = _findings(window, eps, flow_on, eps,
                                              fast, counter_on)
            new_flow, new_counter = _findings(merged, eps, flow_on, eps,
                                              fast, counter_on)
            if ({*new_flow} - {*old_flow}) or ({*new_counter} - {*old_counter}):
                return "direction"
        if any(self._overlap_blocked(note, proposed, same_hand, cross)
               for note in proposed):
            return "overlap"
        return None
