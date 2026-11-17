import time as clock

import numpy as np

from tools.config import config
from map_creation import v2_forward_flow as flow

_TOL = flow._TOL

_RED_CYCLE = ((1, 2, 0), (0, 0, 6), (0, 2, 0), (1, 0, 7))
_MIRROR_D = {2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}


def _cycle(hand):
    if hand == 0:
        return _RED_CYCLE
    return tuple((3 - x, y, _MIRROR_D.get(d, d)) for x, y, d in _RED_CYCLE)


def _gates():
    eps = flow._eps()
    fast = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
    counter_on = (fast > 0.
                  and getattr(config, "v2_fast_counter_direction", "recut") == "recut")
    flow_on = (getattr(config, "v2_cleanup_same_dir_recut", 0) or 0) > 0
    same_hand = max(eps, float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.))
    cross = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
    return eps, fast, counter_on, flow_on, same_hand, cross


def _single(group):
    return len(group) == 1 and group[0]["d"] != 8


def _group_gap(first, second):
    return float(second[0]["time"]) - float(first[0]["time"])


def _hand_groups(context, hand):
    eps = flow._eps()
    groups = []
    for note in context._members[hand]:
        if groups and float(note["time"]) - float(groups[-1][-1]["time"]) <= eps + _TOL:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def _regular(chunk):
    gaps = [_group_gap(a, b) for a, b in zip(chunk, chunk[1:])]
    median = float(np.median(gaps))
    tolerance = max(.015, .15 * median)
    return all(abs(gap - median) <= tolerance + _TOL for gap in gaps)


def _mixed(chunk):
    return sum(int(group[0]["d"] in (2, 3, 4, 5, 6, 7)) for group in chunk)


def _aggregate(context, notes):
    total = [0., 0., 0., 0.]
    for note in notes:
        cost = flow.style_cost(note, context.previous(note),
                               context.following(note), context.group(note))
        for i, value in enumerate(cost):
            total[i] += value
    return tuple(total)


def _speed_ok(edges, cap):
    for previous_members, note, incumbent in edges:
        if flow._arrival(previous_members, note) > max(cap(note), incumbent) + _TOL:
            return False
    return True


def _prepare(context, index, notes):
    hand = notes[0]["hand"]
    members = context._members[hand]
    first, last = notes[0], notes[-1]
    succ = context.following(last)
    succ_members = list(context.group(succ)) if succ is not None else []
    boundary = []
    for edge in (context.previous(first), succ):
        if edge is not None:
            boundary.extend(context.group(edge))
    p0, p1 = context._pos[first["id"]], context._pos[last["id"]]
    arrival = {}
    for m, note in enumerate(notes):
        edge = context.previous(note) if m == 0 else notes[m - 1]
        previous_members = context.group(edge) if edge is not None else ()
        arrival[note["id"]] = flow._arrival(previous_members, note)
    return {"first": first, "last": last, "succ_members": succ_members,
            "succ_dot": any(member["d"] == 8 for member in succ_members),
            "pred_t": flow._pred_tuple(context, first),
            "slice": members[max(0, p0 - 2):p1 + 3],
            "before_viol": flow._violations(members[max(0, p0 - 2):p1 + 3]),
            "boundary": boundary,
            "base_boundary": {
                n["id"]: flow.style_cost(n, context.previous(n),
                                         context.following(n),
                                         context.group(n)) for n in boundary},
            "agg_before": _aggregate(context, notes),
            "conflicts": {n["id"]: index.conflict_ids(n) for n in notes},
            "arrival": arrival,
            "succ_base": {n["id"]: flow._arrival((last,), n)
                          for n in succ_members}}


def _check_chunk(context, index, cap, notes, pre, saved, strict):
    from map_creation import v2_postprocessing as post
    from map_creation.v2_forward_directions import _overlap_conflicts
    from map_creation.v2_playability_reach import forbidden_cut
    eps, fast, counter_on, flow_on, same_hand, cross = _gates()
    first, last = pre["first"], pre["last"]
    if flow_on:
        if not post._flow_legal(first["d"], pre["pred_t"], first["x"], first["y"],
                                float(first["time"]), eps):
            return None
        for a, b in zip(notes, notes[1:]):
            if not post._flow_legal(
                    b["d"], (float(a["time"]), a["x"], a["y"], a["d"]),
                    b["x"], b["y"], float(b["time"]), eps):
                return None
        for member in pre["succ_members"]:
            if not post._flow_legal(
                    member["d"],
                    (float(last["time"]), last["x"], last["y"], last["d"]),
                    member["x"], member["y"], float(member["time"]), eps):
                return None
    if not flow._violations(pre["slice"]) <= pre["before_viol"]:
        return None
    ignore = getattr(index, "_dropped", frozenset())
    for note in notes:
        if (forbidden_cut(note)
                or _overlap_conflicts(index, note, ignore, same_hand, cross)
                or not index.conflict_ids(note) <= pre["conflicts"][note["id"]]):
            return None
    agg = _aggregate(context, notes)
    if agg > pre["agg_before"] or (strict and not agg < pre["agg_before"]):
        return None
    for member in pre["boundary"]:
        new = flow.style_cost(member, context.previous(member),
                              context.following(member), context.group(member))
        if any(n_ > o_ + _TOL
               for n_, o_ in zip(new, pre["base_boundary"][member["id"]])):
            return None
    edges = []
    for m, note in enumerate(notes):
        edge = context.previous(note) if m == 0 else notes[m - 1]
        previous_members = context.group(edge) if edge is not None else ()
        edges.append((previous_members, note, pre["arrival"][note["id"]]))
    for member in pre["succ_members"]:
        edges.append(((last,), member, pre["succ_base"][member["id"]]))
    if not _speed_ok(edges, cap):
        return None
    displacement = sum(abs(n["x"] - ox) + abs(n["y"] - oy)
                       for n, (ox, oy, _d) in zip(notes, saved))
    deviation = sum(post._step_distance(n["d"], od) if od != 8 else 0
                    for n, (_ox, _oy, od) in zip(notes, saved))
    return (*agg, displacement, deviation)


def _lane_attempt(context, index, cap, hand, notes, report):
    pre = _prepare(context, index, notes)
    if pre["succ_dot"]:
        return
    saved = [(n["x"], n["y"], n["d"]) for n in notes]
    applied = False
    try:
        for note in notes:
            note["x"] = 0 if hand == 0 else 3
        applied = (_check_chunk(context, index, cap, notes, pre, saved,
                                strict=True) is not None)
    finally:
        if not applied:
            for note, (x, y, d) in zip(notes, saved):
                note["x"], note["y"], note["d"] = x, y, d
    if applied:
        report["lane_applied"] += 1
        report["lane_notes"] += len(notes)


def _lane_runs(context, index, cap, report):
    for hand in (0, 1):
        groups = _hand_groups(context, hand)
        i = 0
        while i < len(groups):
            if not _single(groups[i]) or groups[i][0]["x"] not in (1, 2):
                i += 1
                continue
            j = i
            while (j + 1 < len(groups) and _single(groups[j + 1])
                    and groups[j + 1][0]["x"] == groups[i][0]["x"]
                    and _group_gap(groups[j], groups[j + 1]) <= flow._RECOVERY_GAP + _TOL):
                j += 1
            run = groups[i:j + 1]
            i = j + 1
            if len(run) < 2 or not any(group[0]["y"] == 1 for group in run):
                continue
            report["lane_runs"] += 1
            report["lane_proposals"] += 1
            _lane_attempt(context, index, cap, hand,
                          [group[0] for group in run], report)


def _wave_hand(context, index, cap, hand, report):
    groups = _hand_groups(context, hand)
    i = 0
    while i < len(groups):
        if not _single(groups[i]):
            i += 1
            continue
        j = i
        while (j + 1 < len(groups) and _single(groups[j + 1])
                and .1 - _TOL <= _group_gap(groups[j], groups[j + 1]) <= flow._RUN_GAP + _TOL):
            j += 1
        k = i
        while k <= j:
            chunk = groups[k:min(k + flow._WAVE_NOTES, j + 1)]
            k += len(chunk)
            if len(chunk) < 4 or not _regular(chunk) or _mixed(chunk) < 2:
                continue
            report["wave_windows"] += 1
            _wave_attempt(context, index, cap, hand, chunk, report)
        i = j + 1


def _wave_attempt(context, index, cap, hand, chunk, report):
    notes = [group[0] for group in chunk]
    pre = _prepare(context, index, notes)
    if pre["succ_dot"]:
        return
    saved = [(n["x"], n["y"], n["d"]) for n in notes]
    cycle = _cycle(hand)
    first_d = notes[0]["d"]
    if first_d in (0, 4, 5):
        prefer = {0, 2}
    elif first_d in (1, 6, 7):
        prefer = {1, 3}
    else:
        prefer = set()
    phases = sorted(range(flow._WAVE_PHASES),
                    key=lambda p: (p not in prefer, p))
    best = None
    for order, phase in enumerate(phases):
        report["wave_proposals"] += 1
        key = snapshot = None
        try:
            for m, note in enumerate(notes):
                note["x"], note["y"], note["d"] = cycle[(phase + m) % flow._WAVE_PHASES]
            key = _check_chunk(context, index, cap, notes, pre, saved,
                               strict=False)
            if key is not None:
                snapshot = [(n["x"], n["y"], n["d"]) for n in notes]
        finally:
            for note, (x, y, d) in zip(notes, saved):
                note["x"], note["y"], note["d"] = x, y, d
        if snapshot is not None:
            rank = (*key, order)
            if best is None or rank < best[0]:
                best = (rank, snapshot)
    if best is None or best[1] == saved:
        return
    for note, (x, y, d) in zip(notes, best[1]):
        note["x"], note["y"], note["d"] = x, y, d
    report["wave_applied"] += 1
    report["wave_notes"] += len(notes)


def _repair_note(context, index, cap, note, group, base, report):
    from map_creation import v2_postprocessing as post
    from map_creation.v2_forward_directions import _overlap_conflicts
    from map_creation.v2_playability_reach import forbidden_cut
    eps, fast, counter_on, flow_on, same_hand, cross = _gates()
    previous = context.previous(note)
    following = context.following(note)
    x0, y0, d0 = note["x"], note["y"], note["d"]
    low, high = flow._bounds(note["hand"])
    pred_t = flow._pred_tuple(context, note)
    pred_dict = flow._tuple_note(pred_t, note["hand"])
    fast_member = ((following is not None
                    and float(following["time"]) - float(note["time"]) < fast)
                   or (previous is not None
                       and float(note["time"]) - float(previous["time"]) < fast))
    arrows = range(8) if d0 != 8 else (8,)
    ranked = []
    for x in range(low, high + 1):
        for y in range(3):
            for d in arrows:
                if (x, y, d) == (x0, y0, d0):
                    continue
                trial = {"id": note["id"], "time": note["time"],
                         "hand": note["hand"], "x": x, "y": y, "d": d}
                if d != 8:
                    if (flow_on
                            and not post._flow_legal(d, pred_t, x, y,
                                                     float(note["time"]), eps)):
                        continue
                    if (flow_on and following is not None
                            and following["d"] != 8
                            and not post._flow_legal(
                                following["d"],
                                (float(note["time"]), x, y, d),
                                following["x"], following["y"],
                                float(following["time"]), eps)):
                        continue
                    if forbidden_cut(trial):
                        continue
                    if (counter_on and fast_member
                            and post._counter_direction(trial)
                            and not flow.prepared_rebound(trial, pred_dict)):
                        continue
                cost = flow.style_cost(trial, previous, following, group)
                if not cost < base:
                    continue
                deviation = (post._step_distance(d, d0)
                             if d != 8 and d0 != 8 else 0)
                ranked.append((cost, abs(x - x0) + abs(y - y0), deviation,
                               x, y, d))
    ranked.sort()
    if not ranked:
        return False
    members = context._members[note["hand"]]
    p = context._pos[note["id"]]
    slice_notes = members[max(0, p - 2):p + 3]
    before_viol = flow._violations(slice_notes)
    base_following = (flow.style_cost(following, context.previous(following),
                                      context.following(following),
                                      context.group(following))
                      if following is not None else None)
    previous_members = context.group(previous) if previous is not None else ()
    incumbent_arrival = flow._arrival(previous_members, note)
    succ_arrival = (flow._arrival((note,), following)
                    if following is not None else None)
    baseline = index.conflict_ids(note)
    ignore = getattr(index, "_dropped", frozenset())
    for cost, _disp, _dev, x, y, d in ranked[:flow._MAX_CANDIDATES]:
        report["proposals"] += 1
        accepted = False
        try:
            note["x"], note["y"], note["d"] = x, y, d
            accepted = (
                not _overlap_conflicts(index, note, ignore, same_hand, cross)
                and index.conflict_ids(note) <= baseline
                and flow._violations(slice_notes) <= before_viol
                and (base_following is None
                     or not any(n_ > o_ + _TOL for n_, o_ in zip(
                         flow.style_cost(following, context.previous(following),
                                         context.following(following),
                                         context.group(following)),
                         base_following)))
                and flow._arrival(previous_members, note)
                <= max(cap(note), incumbent_arrival) + _TOL
                and (following is None
                     or flow._arrival((note,), following)
                     <= max(cap(following), succ_arrival) + _TOL))
        finally:
            if not accepted:
                note["x"], note["y"], note["d"] = x0, y0, d0
        if accepted:
            return True
    return False


def _local_repair(context, index, cap, report):
    ordered = sorted((n for members in context._members.values()
                      for n in members),
                     key=lambda n: (float(n["time"]), n["id"]))
    seen = set()
    for note in ordered:
        group = context.group(note)
        key = group[0]["id"] if group else note["id"]
        if key in seen:
            continue
        seen.add(key)
        if len(group) != 1:
            continue
        base = flow.style_cost(note, context.previous(note),
                               context.following(note), group)
        if not any(base):
            continue
        previous = context.previous(note)
        following = context.following(note)
        if ((previous is not None
                 and (previous["d"] == 8 or len(context.group(previous)) != 1))
                or (following is not None
                    and (following["d"] == 8
                         or len(context.group(following)) != 1))):
            report["skipped"] += 1
            continue
        report["attempted"] += 1
        if _repair_note(context, index, cap, note, group, base, report):
            report["changed"] += 1
        else:
            report["blocked"] += 1


def repair_patterns(notes, index, cap):
    start = clock.monotonic()
    report = {"attempted": 0, "changed": 0, "blocked": 0, "skipped": 0,
              "proposals": 0, "lane_runs": 0, "lane_proposals": 0,
              "lane_applied": 0, "lane_notes": 0, "wave_windows": 0,
              "wave_proposals": 0, "wave_applied": 0, "wave_notes": 0,
              "residual": {}, "seconds": 0.}
    if not flow.enabled() or not notes:
        report["seconds"] = clock.monotonic() - start
        return report
    context = flow.FlowContext(notes)
    _lane_runs(context, index, cap, report)
    for hand in (0, 1):
        _wave_hand(context, index, cap, hand, report)
    _local_repair(context, index, cap, report)
    report["residual"] = flow.audit_patterns(notes)
    report["seconds"] = clock.monotonic() - start
    return report
