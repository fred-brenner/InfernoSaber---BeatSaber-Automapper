from map_creation import v2_forward_flow as flow


_LATERAL_CYCLES = (
    ("adjacent_rebound", ((0, 0, 1), (1, 0, 0))),
    ("adjacent_rocker", ((0, 1, 4), (1, 0, 1))),
)


def repetition_cost(context, notes):
    replacements = {note["id"]: note for note in notes}
    total = 0
    for hand in (0, 1):
        affected = [context._pos[note["id"]] for note in notes
                    if note["hand"] == hand]
        if not affected:
            continue
        members = context._members[hand]
        lo, hi = max(0, min(affected) - 2), min(len(members), max(affected) + 3)
        for i in range(lo, hi - 2):
            original = members[i:i + 3]
            if not any(note["id"] in replacements for note in original):
                continue
            triple = [replacements.get(note["id"], note) for note in original]
            if (any(len(context.group(note)) != 1 for note in original)
                    or any(note["d"] == 8 for note in triple)):
                continue
            gaps = [float(b["time"]) - float(a["time"])
                    for a, b in zip(triple, triple[1:])]
            if (all(flow._eps() < gap <= flow._RUN_GAP + flow._TOL for gap in gaps)
                    and triple[0]["x"] == triple[1]["x"] == triple[2]["x"]):
                total += 1
    return total


def _candidates(notes, hand, regular, mirror):
    saved = tuple((n["x"], n["y"], n["d"]) for n in notes)
    outer = (0, 1) if hand == 0 else (3, 2)
    candidates = []
    for phase in range(2):
        candidates.append(("adjacent_lanes", tuple(
            (outer[(i + phase) % 2], n["y"], n["d"])
            for i, n in enumerate(notes))))
    for phase in range(2):
        geometry = []
        for i, note in enumerate(notes):
            x, y, d = outer[(i + phase) % 2], note["y"], note["d"]
            if x == outer[1]:
                if y == 1 and d in (0, 4, 5):
                    y = 2
                elif y in (1, 2) and d in (1, 6, 7):
                    y = 0
            geometry.append((x, y, d))
        candidates.append(("adjacent_layers", tuple(geometry)))
    if regular:
        for name, cycle in _LATERAL_CYCLES:
            if hand == 1:
                cycle = tuple((3 - x, y, mirror.get(d, d)) for x, y, d in cycle)
            for phase in range(len(cycle)):
                candidates.append((name, tuple(
                    cycle[(phase + i) % len(cycle)] for i in range(len(notes)))))
    for phase in range(2):
        geometry = list(saved)
        for i in range(1, len(notes) - 1):
            geometry[i] = (outer[(i + phase) % 2], 0, notes[i]["d"])
        candidates.append(("compact_bottom", tuple(geometry)))
    seen = {saved}
    for name, geometry in candidates:
        if geometry not in seen:
            seen.add(geometry)
            yield name, geometry


def _attempt(context, index, cap, notes, chunk, hand, report, patterns):
    pre = patterns._prepare(context, index, notes)
    saved = [(n["x"], n["y"], n["d"]) for n in notes]
    before = repetition_cost(context, notes)
    best = None
    for order, (name, geometry) in enumerate(
            _candidates(notes, hand, patterns._regular(chunk), patterns._MIRROR_D)):
        report["lateral_proposals"] += 1
        key = None
        try:
            for note, (x, y, d) in zip(notes, geometry):
                note["x"], note["y"], note["d"] = x, y, d
            key = patterns._check_chunk(context, index, cap, notes, pre, saved,
                                        strict=False, report=report, family="lateral")
            if key is not None and key[4] >= before:
                key = patterns._reject(report, "lateral", "lane_repetition", notes)
            if key is not None and any(
                    new > old + flow._TOL for new, old in zip(key[:4], pre["agg_before"])):
                key = patterns._reject(report, "lateral", "style_regression", notes)
        finally:
            for note, (x, y, d) in zip(notes, saved):
                note["x"], note["y"], note["d"] = x, y, d
        if key is not None:
            rank = (key[4], *key, order)
            if best is None or rank < best[0]:
                best = (rank, name, geometry)
    if best is None:
        return
    _, name, geometry = best
    for note, (x, y, d) in zip(notes, geometry):
        note["x"], note["y"], note["d"] = x, y, d
    report["lateral_applied"] += 1
    report["lateral_notes"] += sum(a != b for a, b in zip(saved, geometry))
    motifs = report["lateral_motifs"]
    motifs[name] = motifs.get(name, 0) + 1


def repair_lanes(context, index, cap, report):
    from map_creation import v2_forward_patterns as patterns

    for hand in (0, 1):
        groups = patterns._hand_groups(context, hand)
        i = 0
        while i < len(groups):
            if (not patterns._single(groups[i])
                    or len(context.group(groups[i][0])) != 1):
                i += 1
                continue
            j = i + 1
            while (j < len(groups) and patterns._single(groups[j])
                   and len(context.group(groups[j][0])) == 1
                   and groups[j][0]["x"] == groups[i][0]["x"]
                   and .1 - flow._TOL <= patterns._group_gap(groups[j - 1], groups[j])
                   <= flow._RUN_GAP + flow._TOL):
                j += 1
            start = i
            while j - start >= 4:
                chunk = groups[start:min(start + 8, j)]
                if len(chunk) > 4 and not patterns._regular(chunk):
                    chunk = chunk[:4]
                start += len(chunk)
                notes = [group[0] for group in chunk]
                report["lateral_windows"] += 1
                previous, following = context.previous(notes[0]), context.following(notes[-1])
                if any(edge is not None
                       and (edge["d"] == 8 or len(context.group(edge)) != 1)
                       for edge in (previous, following)):
                    patterns._reject(report, "lateral", "skipped_context", notes)
                    continue
                _attempt(context, index, cap, notes, chunk, hand, report, patterns)
            i = j
