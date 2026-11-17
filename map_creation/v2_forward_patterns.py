import time as clock

import numpy as np

from tools.config import config
from map_creation import v2_forward_flow as flow
from map_creation import v2_forward_lanes as lanes

_TOL = flow._TOL

_RED_CYCLE = ((1, 2, 0), (0, 0, 6), (0, 2, 0), (1, 0, 7))
_MIRROR_D = {2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
_PHRASE_CYCLES = (
    ("outer_bounce", ((0, 1, 0), (0, 0, 1))),
    ("bottom_rebound", ((1, 0, 0), (1, 0, 1))),
    ("diagonal_rocker", ((0, 1, 4), (1, 0, 1))),
    ("fan", ((0, 1, 4), (1, 0, 1), (1, 2, 0), (1, 0, 1))),
)
_PHRASE_TRIALS = 12
_REJECTION_EXAMPLES = 3


def _upper_down(note):
    return note["x"] in (1, 2) and note["y"] == 2 and note["d"] in (1, 6, 7)


def _reject(report, family, reason, notes):
    if report is not None:
        key = f"{family}:{reason}"
        counts = report.setdefault("rejections", {})
        examples = report.setdefault("rejection_examples", {})
        counts[key] = counts.get(key, 0) + 1
        samples = examples.setdefault(key, [])
        if len(samples) < _REJECTION_EXAMPLES:
            samples.append({"id": notes[0]["id"], "time": notes[0]["time"],
                            "length": len(notes)})
    return None


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


def _check_chunk(context, index, cap, notes, pre, saved, strict, report=None,
                 family="wave"):
    from map_creation import v2_postprocessing as post
    from map_creation.v2_forward_directions import _overlap_conflicts
    from map_creation.v2_playability_reach import forbidden_cut
    eps, fast, counter_on, flow_on, same_hand, cross = _gates()
    first, last = pre["first"], pre["last"]
    if flow_on:
        if not post._flow_legal(first["d"], pre["pred_t"], first["x"], first["y"],
                                float(first["time"]), eps):
            return _reject(report, family, "entry_flow", notes)
        for a, b in zip(notes, notes[1:]):
            if not post._flow_legal(
                    b["d"], (float(a["time"]), a["x"], a["y"], a["d"]),
                    b["x"], b["y"], float(b["time"]), eps):
                return _reject(report, family, "internal_flow", notes)
        for member in pre["succ_members"]:
            if not post._flow_legal(
                    member["d"],
                    (float(last["time"]), last["x"], last["y"], last["d"]),
                    member["x"], member["y"], float(member["time"]), eps):
                return _reject(report, family, "exit_flow", notes)
    if not flow._violations(pre["slice"]) <= pre["before_viol"]:
        return _reject(report, family, "direction", notes)
    ignore = getattr(index, "_dropped", frozenset())
    for note in notes:
        if forbidden_cut(note):
            return _reject(report, family, "forbidden", notes)
        if _overlap_conflicts(index, note, ignore, same_hand, cross):
            return _reject(report, family, "overlap", notes)
        if not index.conflict_ids(note) <= pre["conflicts"][note["id"]]:
            return _reject(report, family, "collision", notes)
    agg = _aggregate(context, notes)
    if agg > pre["agg_before"] or (strict and not agg < pre["agg_before"]):
        return _reject(report, family, "style", notes)
    if sum(_upper_down(note) for note in notes) > sum(
            _upper_down({"x": x, "y": y, "d": d}) for x, y, d in saved):
        return _reject(report, family, "upper_down", notes)
    for member in pre["boundary"]:
        new = flow.style_cost(member, context.previous(member),
                              context.following(member), context.group(member))
        if any(n_ > o_ + _TOL
               for n_, o_ in zip(new, pre["base_boundary"][member["id"]])):
            return _reject(report, family, "neighbor_style", notes)
    edges = []
    for m, note in enumerate(notes):
        edge = context.previous(note) if m == 0 else notes[m - 1]
        previous_members = context.group(edge) if edge is not None else ()
        edges.append((previous_members, note, pre["arrival"][note["id"]]))
    for member in pre["succ_members"]:
        edges.append(((last,), member, pre["succ_base"][member["id"]]))
    if not _speed_ok(edges, cap):
        return _reject(report, family, "speed", notes)
    displacement = sum(abs(n["x"] - ox) + abs(n["y"] - oy)
                       for n, (ox, oy, _d) in zip(notes, saved))
    deviation = sum(post._step_distance(n["d"], od) if od != 8 else 0
                    for n, (_ox, _oy, od) in zip(notes, saved))
    return (*agg, lanes.repetition_cost(context, notes), displacement, deviation)


def _lane_attempt(context, index, cap, hand, notes, report):
    pre = _prepare(context, index, notes)
    if pre["succ_dot"]:
        _reject(report, "lane", "skipped_context", notes)
        return
    saved = [(n["x"], n["y"], n["d"]) for n in notes]
    applied = False
    try:
        for note in notes:
            note["x"] = 0 if hand == 0 else 3
        applied = (_check_chunk(context, index, cap, notes, pre, saved,
                                strict=True, report=report, family="lane") is not None)
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
    cycle = _cycle(hand)
    best = None
    lengths = sorted({len(chunk), max(4, len(chunk) - 2), 4}, reverse=True)
    for length in (n for n in lengths if n <= len(chunk)):
        notes = [group[0] for group in chunk[:length]]
        pre = _prepare(context, index, notes)
        if pre["succ_dot"]:
            _reject(report, "wave", "skipped_context", notes)
            continue
        saved = [(n["x"], n["y"], n["d"]) for n in notes]
        for phase in range(flow._WAVE_PHASES):
            report["wave_proposals"] += 1
            key = snapshot = None
            try:
                for m, note in enumerate(notes):
                    note["x"], note["y"], note["d"] = cycle[(phase + m) % flow._WAVE_PHASES]
                key = _check_chunk(context, index, cap, notes, pre, saved,
                                   strict=False, report=report, family="wave")
                if key is not None:
                    snapshot = [(n["x"], n["y"], n["d"]) for n in notes]
            finally:
                for note, (x, y, d) in zip(notes, saved):
                    note["x"], note["y"], note["d"] = x, y, d
            if snapshot is not None:
                rank = (*key, -length, phase)
                if best is None or rank < best[0]:
                    best = (rank, notes, saved, snapshot)
    if best is None or best[2] == best[3]:
        return
    _, notes, saved, snapshot = best
    for note, (x, y, d) in zip(notes, snapshot):
        note["x"], note["y"], note["d"] = x, y, d
    report["wave_applied"] += 1
    report["wave_notes"] += sum(a != b for a, b in zip(saved, snapshot))


def _phrase_candidates(notes, hand, regular):
    saved = tuple((n["x"], n["y"], n["d"]) for n in notes)
    low, high = flow._bounds(hand)
    shift = -1 if hand == 0 else 1
    translated = tuple((x + shift, y, d) for x, y, d in saved)
    lowered = tuple((x, 0 if x in (1, 2) and y == 1 else y, d)
                    for x, y, d in saved)
    candidates = []
    if all(low <= x <= high for x, _, _ in translated):
        candidates.append(("translate", translated))
    candidates.append(("lower_center", lowered))
    if regular:
        for name, cycle in _PHRASE_CYCLES:
            if hand == 1:
                cycle = tuple((3 - x, y, _MIRROR_D.get(d, d)) for x, y, d in cycle)
            for phase in range(len(cycle)):
                geometry = tuple(cycle[(phase + i) % len(cycle)]
                                 for i in range(len(notes)))
                candidates.append((name, geometry))
    seen = {saved}
    for name, geometry in candidates:
        if geometry not in seen:
            seen.add(geometry)
            yield name, geometry


def _phrase_hand(context, index, cap, hand, report):
    groups = _hand_groups(context, hand)
    i = 0
    while i < len(groups):
        group = groups[i]
        if not _single(group) or len(context.group(group[0])) != 1:
            i += 1
            continue
        j = i + 1
        while (j < len(groups) and _single(groups[j])
               and len(context.group(groups[j][0])) == 1
               and flow._eps() < _group_gap(groups[j - 1], groups[j])
               <= flow._RUN_GAP + _TOL):
            j += 1
        for start in range(i, j, 4):
            chunk = groups[start:min(start + 4, j)]
            if len(chunk) < 2:
                continue
            notes = [group[0] for group in chunk]
            if not any(_aggregate(context, notes)):
                continue
            report["phrase_windows"] += 1
            previous = context.previous(notes[0])
            following = context.following(notes[-1])
            if any(edge is not None
                   and (edge["d"] == 8 or len(context.group(edge)) != 1)
                   for edge in (previous, following)):
                _reject(report, "phrase", "skipped_context", notes)
                continue
            candidates = list(_phrase_candidates(notes, hand, _regular(chunk)))
            if not candidates:
                _reject(report, "phrase", "no_candidate", notes)
                continue
            pre = _prepare(context, index, notes)
            saved = [(n["x"], n["y"], n["d"]) for n in notes]
            before_repetition = lanes.repetition_cost(context, notes)
            best = None
            for order, (name, geometry) in enumerate(candidates[:_PHRASE_TRIALS]):
                report["phrase_proposals"] += 1
                key = None
                try:
                    for note, (x, y, d) in zip(notes, geometry):
                        note["x"], note["y"], note["d"] = x, y, d
                    key = _check_chunk(context, index, cap, notes, pre, saved,
                                       strict=True, report=report, family="phrase")
                    if key is not None and key[4] > before_repetition:
                        key = _reject(report, "phrase", "lane_repetition", notes)
                finally:
                    for note, (x, y, d) in zip(notes, saved):
                        note["x"], note["y"], note["d"] = x, y, d
                if key is not None:
                    rank = (*key, order)
                    if best is None or rank < best[0]:
                        best = (rank, name, geometry)
            if best is not None:
                _, name, geometry = best
                for note, (x, y, d) in zip(notes, geometry):
                    note["x"], note["y"], note["d"] = x, y, d
                report["phrase_applied"] += 1
                report["phrase_notes"] += sum(a != b for a, b in zip(saved, geometry))
                motifs = report["phrase_motifs"]
                motifs[name] = motifs.get(name, 0) + 1
        i = j


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
    repetition_by_x = {x: lanes.repetition_cost(context, [dict(note, x=x)])
                       for x in range(low, high + 1)}
    for x in range(low, high + 1):
        for y in range(3):
            for d in arrows:
                if (x, y, d) == (x0, y0, d0):
                    continue
                trial = {"id": note["id"], "time": note["time"],
                         "hand": note["hand"], "x": x, "y": y, "d": d}
                if _upper_down(trial) > _upper_down(note):
                    continue
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
                ranked.append((cost, repetition_by_x[x], abs(x - x0) + abs(y - y0),
                               deviation, abs(x - x0), x, y, d))
    ranked.sort()
    if not ranked:
        _reject(report, "local", "no_candidate", [note])
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
    for cost, _rep, _disp, _dev, _dx, x, y, d in ranked[:flow._MAX_CANDIDATES]:
        report["proposals"] += 1
        accepted = False
        try:
            note["x"], note["y"], note["d"] = x, y, d
            reason = None
            if _overlap_conflicts(index, note, ignore, same_hand, cross):
                reason = "overlap"
            elif not index.conflict_ids(note) <= baseline:
                reason = "collision"
            elif not flow._violations(slice_notes) <= before_viol:
                reason = "direction"
            elif (base_following is not None
                  and any(n_ > o_ + _TOL for n_, o_ in zip(
                      flow.style_cost(following, context.previous(following),
                                      context.following(following),
                                      context.group(following)),
                      base_following))):
                reason = "neighbor_style"
            elif (flow._arrival(previous_members, note)
                  > max(cap(note), incumbent_arrival) + _TOL
                  or (following is not None
                      and flow._arrival((note,), following)
                      > max(cap(following), succ_arrival) + _TOL)):
                reason = "speed"
            if reason is not None:
                _reject(report, "local", reason, [note])
            else:
                accepted = True
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
            _reject(report, "local", "skipped_context", [note])
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
              "phrase_windows": 0, "phrase_proposals": 0, "phrase_applied": 0,
              "phrase_notes": 0, "phrase_motifs": {},
              "lateral_windows": 0, "lateral_proposals": 0, "lateral_applied": 0,
              "lateral_notes": 0, "lateral_motifs": {},
              "rejections": {}, "rejection_examples": {},
              "residual": {}, "seconds": 0.}
    if not flow.enabled() or not notes:
        report["seconds"] = clock.monotonic() - start
        return report
    context = flow.FlowContext(notes)
    _lane_runs(context, index, cap, report)
    for hand in (0, 1):
        _wave_hand(context, index, cap, hand, report)
    for hand in (0, 1):
        _phrase_hand(context, index, cap, hand, report)
    _local_repair(context, index, cap, report)
    lanes.repair_lanes(context, index, cap, report)
    report["residual"] = flow.audit_patterns(notes)
    report["seconds"] = clock.monotonic() - start
    return report
