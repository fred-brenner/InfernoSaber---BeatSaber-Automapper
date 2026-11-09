"""Frozen human reach tails and bounded joint cell/arrow repair.

All times are elapsed seconds. The reference is a small versioned source asset;
inference never imports the offline evaluator, reads human maps, or uses RNG.
Endpoint/travel severities follow the November 8 human-only calibration, with
exact-time stacks paired best-case and dots receiving no invented endpoints.
"""

from bisect import bisect_left, bisect_right
from collections import ChainMap
import json
import math
from pathlib import Path

from tools.config import config


_REFERENCE = json.loads(Path(__file__).with_name("v2_playability_reach_reference.json").read_text())
_TAU = _REFERENCE["tau_seconds"]
_BANDS = ("<3", "3-5", "5-7", "7+")
_STEP = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0),
         4: (1, -1), 5: (-1, -1), 6: (1, 1), 7: (-1, 1)}
_METRICS = ("nominal_approach_excursion", "vertical_preparation_excursion",
            "nominal_followthrough_excursion", "vertical_followthrough_excursion",
            "nominal_center_reach", "recovery_adjusted_travel", "repeated_excursion_exposure")
_TOL, _GROUP_EPS = 1e-9, 1e-7


def forbidden_cut(note, direction=None, x=None, y=None):
    """The independent zero-allowance far-opposite inward-cut policy."""
    direction = note["d"] if direction is None else direction
    x = note["x"] if x is None else x
    return ((note["hand"] == 0 and x == 3 and direction in (2, 4, 6))
            or (note["hand"] == 1 and x == 0 and direction in (3, 5, 7)))


def calibrated_cutoff(metric, nps):
    """Fixed supported human stratum; a caller freezes NPS before repair."""
    band = _BANDS[sum(nps >= bound for bound in (3, 5, 7))]
    return _REFERENCE["metrics"][metric]["cutoffs_by_nps_band"][band]


def diagonal_fraction_limit(nps):
    """Human per-map arrow diagonal-share p95, frozen by retained NPS band."""
    band = _BANDS[sum(nps >= bound for bound in (3,5,7))]
    return _REFERENCE["diagonal_mix"]["p95_by_nps_band"][band]


def _outside(value, low, high):
    return max(low-value, value-high, 0.)


def _bases(note):
    low, high = (0, 2) if note["hand"] == 0 else (1, 3)
    x, y = note["x"], note["y"]
    center = _outside(x, low, high)
    if note["d"] == 8:
        return (0., 0., 0., 0., center)
    sx, sy = _STEP[note["d"]]
    return (_outside(x+sx, low, high), _outside(y+sy, 0, 2),
            _outside(x-sx, low, high), _outside(y-sy, 0, 2), center)


def _excess(metric, severity, nps):
    cutoff = calibrated_cutoff(metric, nps)
    # An ordinary human-range value has exactly zero optimization reward.
    return max(0., severity-cutoff)/max(1., cutoff)


def _travel(previous, note, gap):
    psx, psy = _STEP[previous["d"]]
    sx, sy = _STEP[note["d"]]
    return math.hypot(note["x"]+sx-previous["x"]+psx,
                      note["y"]+sy-previous["y"]+psy)/(gap+_TAU)


def local_reach_components(previous_note, note, nps):
    """Independent preparation/center/exit and incoming travel tail excess.

    The caller supplies a distinct-time same-hand predecessor. Stacked-group
    best pairing and cumulative exposure require the full ``reach_costs`` API.
    A dot contributes its center only, with no virtual travel endpoints.
    """
    gap = math.inf if previous_note is None else note["time"]-previous_note["time"]
    multiplier = 1.+_TAU/(_TAU+gap)
    values = _bases(note)
    keys = range(5) if note["d"] != 8 else (4,)
    result = {_METRICS[k]:_excess(_METRICS[k],values[k]*multiplier,nps) for k in keys}
    if (previous_note is not None and gap > _GROUP_EPS
            and previous_note["d"] != 8 and note["d"] != 8):
        result["recovery_adjusted_travel"] = _excess("recovery_adjusted_travel",_travel(previous_note,note,gap),nps)
    return result


def local_reach_cost(previous_note, note, nps):
    """Scalar counterpart to ``local_reach_components`` for local ranking."""
    return sum(local_reach_components(previous_note,note,nps).values())


def _measured_nps(ordered):
    if not ordered:
        return 0.
    retained, last = 0, {}
    for n in ordered:
        if n["hand"] not in last or n["time"]-last[n["hand"]] > .02+_TOL:
            retained += 1
            last[n["hand"]] = n["time"]
    span = ordered[-1]["time"]-ordered[0]["time"]
    return retained/span if span > 0 else 0.


def note_span_nps(notes):
    """Retained per-hand 20ms-group notes divided by elapsed note span."""
    return _measured_nps(sorted(notes,key=lambda n:(n["time"],n.get("id",0))))


def reach_costs(notes, nps=None):
    """Pure JSON-able exact calibrated event diagnostics for a raw note list."""
    ordered = sorted(notes, key=lambda n: (n["time"], n.get("id", 0)))
    nps = _measured_nps(ordered) if nps is None else float(nps)
    values = {key: [] for key in _METRICS}
    event_ids = {key: [] for key in _METRICS}
    endpoint_values = {}
    groups = [[], []]
    for i, note in enumerate(ordered):
        hand = note["hand"]
        if not groups[hand] or abs(note["time"]-ordered[groups[hand][-1][0]]["time"]) > _GROUP_EPS:
            groups[hand].append([])
        groups[hand][-1].append(i)
    for hand_groups in groups:
        previous, load = None, 0.
        for current in hand_groups:
            gap = math.inf if previous is None else ordered[current[0]]["time"]-ordered[previous[0]]["time"]
            multiplier = 1.+_TAU/(_TAU+gap)
            arrows = [i for i in current if ordered[i]["d"] != 8]
            for i in current:
                endpoint_values[i] = [v*multiplier for v in _bases(ordered[i])]
            if previous is not None and arrows:
                prior_arrows = [i for i in previous if ordered[i]["d"] != 8]
                if prior_arrows:
                    values["recovery_adjusted_travel"].append(min(
                        _travel(ordered[a], ordered[b], gap) for a in prior_arrows for b in arrows))
                    event_ids["recovery_adjusted_travel"].append([ordered[i].get("id", i) for i in arrows])
            load *= math.exp(-gap/_TAU) if math.isfinite(gap) else 0.
            if arrows:
                load += max(max(_bases(ordered[i])[:4]) for i in arrows)
                values["repeated_excursion_exposure"].append(load)
                event_ids["repeated_excursion_exposure"].append([ordered[i].get("id", i) for i in arrows])
            previous = current
    for i,note in enumerate(ordered):
        for k in (range(5) if note["d"] != 8 else (4,)):
            key = _METRICS[k]
            values[key].append(endpoint_values[i][k])
            event_ids[key].append([note.get("id",i)])
    metrics = {}
    for key in _METRICS:
        cutoff = calibrated_cutoff(key, nps)
        flags = [value > cutoff for value in values[key]]
        metrics[key] = {"budget": _REFERENCE["metrics"][key]["budget"], "cutoff": cutoff,
                        "eligible_events": len(flags), "flagged_events": sum(flags),
                        "severities": values[key], "event_note_ids": event_ids[key],
                        "excess": sum(_excess(key, v, nps) for v in values[key])}
    banned = [n.get("id", i) for i, n in enumerate(ordered) if forbidden_cut(n)]
    return {"reference_version": _REFERENCE["version"], "nps": nps,
            "nps_band": _BANDS[sum(nps >= b for b in (3, 5, 7))],
            "hard_bans": len(banned), "hard_ban_ids": banned, "metrics": metrics,
            "total_excess": sum(m["excess"] for m in metrics.values())}


class _Window:
    """Fixed event neighborhoods for local calibrated geometry/style ranking."""

    def __init__(self, notes, nps, post):
        self.notes,self.nps,self.post = notes,nps,post
        times = [n["time"] for n in notes]
        self.previous,self.kept,self.occluders,self.pairs = {},[],{},[]
        for hand in (0,1):
            anchor = None
            for i,n in enumerate(notes):
                if n["hand"] != hand or (anchor is not None and n["time"]-times[anchor] <= .02+_TOL):
                    continue
                if anchor is not None:
                    self.previous[i] = anchor
                self.kept.append(i)
                anchor = i
        self.kept.sort()
        kept = set(self.kept)
        for i in self.kept:
            self.occluders[i] = [j for j in reversed(range(bisect_left(times,times[i]-.4-_TOL),i))
                                  if j in kept and .02 < times[i]-times[j] <= .4]
            if notes[i]["hand"] == 0:
                lo,hi = bisect_left(times,times[i]-.25-_TOL),bisect_right(times,times[i]+.25+_TOL)
                self.pairs.extend((i,j) for j in range(lo,hi)
                                  if j in kept and notes[j]["hand"] == 1 and abs(times[i]-times[j]) <= .25)
        self.baseline_m2 = self.geometry()[0]

    def lateral_excess(self):
        """M1 arrival stress has no optimization reward below its human tail."""
        cutoff = calibrated_cutoff("m1",self.nps)
        total = 0.
        for i,p in self.previous.items():
            direction = self.notes[i]["d"]
            stress = 1. if direction in (2,3) else .5 if direction in (4,5,6,7) else 0.
            gap = self.notes[i]["time"]-self.notes[p]["time"]
            total += max(0.,stress*max(0.,1.-gap/.5)-cutoff)/max(1.,cutoff)
        return total

    def geometry(self, *, raw=False):
        from map_creation.v2_playability_costs import calibrated_limits, interference_severity, occlusion_severity
        from map_creation.v2_playability_geometry import _corridor
        limits = calibrated_limits(self.nps)
        effective,volumes = {},{}
        for i in self.kept:
            n,p = self.notes[i],self.previous.get(i)
            before = None if p is None else self.notes[p]
            direction = n["d"]
            if direction == 8:
                direction = (self.post._dot_virtual_direction(effective[p],before["x"],before["y"],n["x"],n["y"])
                             if p is not None and effective[p] is not None else None)
            effective[i] = direction
            volume = {(n["x"],n["y"])} | set(self.post._swept_cells(n["x"],n["y"],direction))
            if p is not None and n["time"]-before["time"] <= .4+_TOL:
                sx,sy = _STEP.get(effective[p],(0,0))
                dx,dy = _STEP.get(direction,(0,0))
                volume |= _corridor((before["x"]-sx,before["y"]-sy),(n["x"]+dx,n["y"]+dy))
            volumes[i] = volume
        m2 = 0.
        for i in self.kept:
            n = self.notes[i]
            before = next((j for j in self.occluders[i] if abs(n["x"]-self.notes[j]["x"]) <= 1),None)
            if before is not None:
                severity = occlusion_severity(n,self.notes[before],n["time"]-self.notes[before]["time"],
                                              effective[i],effective[before])
                m2 += severity if raw else max(0.,severity-limits["m2"])/max(1.,limits["m2"])
        m4 = 0.
        for i,j in self.pairs:
            severity = interference_severity(self.notes[i],self.notes[j],abs(self.notes[i]["time"]-self.notes[j]["time"]),
                                              volumes[i],volumes[j])
            m4 += severity if raw else max(0.,severity-limits["m4"])/max(1.,limits["m4"])
        return m2,m4


class _Search:
    """Cache time neighborhoods; update virtual dots only along affected edges."""

    def __init__(self, notes, cap, nps, protect_retained):
        from map_creation import v2_postprocessing as post
        self.post, self.notes, self.cap, self.nps = post, notes, cap, nps
        from map_creation.v2_playability_policy import SpeedGroups
        self.speed_groups = SpeedGroups(notes, cap, float(getattr(config, "min_time_diff", .02)))
        self.times = [n["time"] for n in notes]
        self.eps = float(getattr(config, "min_time_diff", .02))
        self.swing = max(self.eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
        self.gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
        self.previous, self.following, self.flow_previous, self.children = {}, {}, {}, {}
        self.effective, self.fast, self.near = {}, set(), {}
        self.committed = {i:(n["x"],n["y"],n["d"]) for i,n in enumerate(notes)}
        self.diagonals = sum(n["d"] in (4,5,6,7) for n in notes)
        arrow_count = sum(n["d"] != 8 for n in notes)
        self.diagonal_limit = max(self.diagonals,math.floor(arrow_count*diagonal_fraction_limit(nps)+_TOL))
        self.limit_diagonals = False
        self.mutable = {i for i,n in enumerate(notes) if not (protect_retained and n.get("kind") == "retained")}
        fast = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
        for hand in (0, 1):
            seq = [i for i,n in enumerate(notes) if n["hand"] == hand]
            times = [self.times[i] for i in seq]
            for offset, i in enumerate(seq):
                if offset:
                    p = seq[offset-1]
                    self.previous[i], self.following[p] = p, i
                    if self.times[i]-self.times[p] < fast:
                        self.fast.update((p, i))
                before = bisect_left(times, self.times[i]-self.eps)-1
                if before >= 0:
                    p = seq[before]
                    self.flow_previous[i] = p
                    self.children.setdefault(p, []).append(i)
        for i,n in enumerate(notes):
            p = self.flow_previous.get(i)
            self.effective[i] = n["d"] if n["d"] != 8 else self._dot(i, p, self.effective)
            self.near[i] = list(range(bisect_left(self.times, self.times[i]-self.swing-_TOL),
                                      bisect_right(self.times, self.times[i]+self.swing+_TOL)))

    def _dot(self, i, p, effective):
        if p is None or effective[p] is None:
            return None
        a,b = self.notes[p],self.notes[i]
        return self.post._dot_virtual_direction(effective[p], a["x"], a["y"], b["x"], b["y"])

    def affected(self, changed):
        affected, pending = set(changed), list(changed)
        while pending:
            for i in self.children.get(pending.pop(), ()):
                if i not in affected:
                    affected.add(i)
                    if self.notes[i]["d"] == 8:
                        pending.append(i)
        return affected

    def legal(self, changed):
        post = self.post
        if self.limit_diagonals:
            delta = sum(int(self.notes[i]["d"] in (4,5,6,7))-int(self.committed[i][2] in (4,5,6,7)) for i in changed)
            if self.diagonals+delta > self.diagonal_limit:
                return False
        affected = self.affected(changed)
        effective = ChainMap({}, self.effective)
        for i in sorted(affected):
            n,p = self.notes[i],self.flow_previous.get(i)
            effective[i] = n["d"] if n["d"] != 8 else self._dot(i,p,effective)
        for i in affected:
            n,p = self.notes[i],self.flow_previous.get(i)
            prior = None if p is None else (self.times[p], self.notes[p]["x"], self.notes[p]["y"], effective[p])
            if n["d"] != 8 and not post._flow_legal(n["d"],prior,n["x"],n["y"],n["time"],self.gap):
                return False
            if i in changed:
                if forbidden_cut(n):
                    return False
                if (i in self.fast and getattr(config,"v2_fast_counter_direction","recut") == "recut"
                        and post._counter_direction(n)):
                    return False
            for j in self.near[i]:
                if j == i:
                    continue
                other,dt = self.notes[j],self.times[j]-self.times[i]
                same = (n["x"],n["y"]) == (other["x"],other["y"])
                if abs(dt) <= self.eps and same:
                    return False
                if n["hand"] == other["hand"]:
                    continue
                if -self.eps <= dt <= self.swing:
                    if (other["x"],other["y"]) in post._swept_cells(n["x"],n["y"],effective[i]) or (same and dt > self.eps):
                        return False
                if -self.swing <= dt <= self.eps:
                    if (n["x"],n["y"]) in post._swept_cells(other["x"],other["y"],effective[j]) or (same and -dt > self.eps):
                        return False
        if not self.speed_groups.legal(changed):
            return False
        return True

    def options(self, i, *, broad=False):
        n = self.notes[i]
        original = (n["x"],n["y"],n["d"])
        cells = [(x,y) for x in range(4) for y in range(3)
                 if broad or abs(x-n["x"])+abs(y-n["y"]) <= 1]
        options = [(x,y,d) for x,y in cells for d in ((8,) if n["d"] == 8 else range(8))
                   if not forbidden_cut(n,d,x,y)
                   and not (i in self.fast and getattr(config,"v2_fast_counter_direction","recut") == "recut"
                            and d in self.post._COUNTER_FAMILY.get(y,()))]
        # The incumbent/permitted nearest alternatives precede broad changes.
        return sorted(options,key=lambda p:(p != original,
                      int(p[2] != original[2])+int(p[:2] != original[:2]),
                      abs(p[0]-original[0])+abs(p[1]-original[1]),
                      self.post._step_distance(p[2],original[2]),p))

    def install(self, edits):
        for i,(x,y,d) in edits.items():
            self.notes[i]["x"],self.notes[i]["y"],self.notes[i]["d"] = x,y,d

    def commit(self, edits):
        self.install(edits)
        self.diagonals += sum(int(edits[i][2] in (4,5,6,7))-int(self.committed[i][2] in (4,5,6,7)) for i in edits)
        self.committed.update(edits)
        for i in sorted(self.affected(edits)):
            n = self.notes[i]
            self.effective[i] = n["d"] if n["d"] != 8 else self._dot(i,self.flow_previous.get(i),self.effective)

    def context(self, i, participants=()):
        ids = self.affected([i,*participants])
        lo,hi = min(self.times[j] for j in ids),max(self.times[j] for j in ids)
        # The bounded window retains several human recovery time constants.
        notes = self.notes[bisect_left(self.times,lo-2.):bisect_right(self.times,hi+2.)]
        return _Window(notes,self.nps,self.post)

    def rank(self, context, edits, original):
        from map_creation.v2_playability_directions import window_style_penalty
        costs = reach_costs(context.notes,self.nps)
        m2,crossing = context.geometry()
        if edits and m2 > context.baseline_m2+_TOL:
            return None
        style = window_style_penalty(context.notes,self.nps)
        changes = sum(edits[i] != original[i] for i in edits)
        distance = sum(abs(edits[i][0]-original[i][0])+abs(edits[i][1]-original[i][1])
                       +.25*self.post._step_distance(edits[i][2],original[i][2]) for i in edits)
        return costs["hard_bans"],costs["total_excess"]+crossing+context.lateral_excess()+.12*style,changes,distance


def _repair_hard_reverse(state, report):
    """First feasible repair; completed later blocks freeze except bounded retry."""
    notes = state.notes
    groups = []
    for i, note in enumerate(notes):
        if not groups or abs(note["time"]-notes[groups[-1][0]]["time"]) > _GROUP_EPS:
            groups.append([])
        groups[-1].append(i)
    frozen, completed = set(), []

    def options(i):
        current = (notes[i]["x"], notes[i]["y"], notes[i]["d"])
        return sorted(state.options(i, broad=True), key=lambda v: (
            v != current, v[2] != current[2], state.post._step_distance(v[2], current[2]),
            abs(v[0]-current[0])+abs(v[1]-current[1]), v))

    def try_edits(edits):
        current = {i: (notes[i]["x"], notes[i]["y"], notes[i]["d"]) for i in edits}
        report["reach_candidates"] += 1
        state.install(edits)
        if state.legal(edits):
            state.commit(edits)
            return True
        state.install(current)
        return False

    def joint(i, allowed):
        participants = set(state.near[i]) | {j for j in (state.previous.get(i), state.following.get(i)) if j is not None}
        for j in tuple(participants):
            participants.update(k for k in (state.previous.get(j), state.following.get(j)) if k is not None)
        participants = sorted((j for j in participants if j != i and j in allowed and j in state.mutable and j not in frozen),
                              key=lambda j: (abs(state.times[j]-state.times[i]), j))[:8]
        for j in participants:
            for first in options(i):
                for second in options(j)[:32]:
                    if try_edits({i: first, j: second}):
                        return True
        return False

    for group in reversed(groups):
        allowed = set(range(max(group)+1)) & state.mutable
        for i in reversed(group):
            if not forbidden_cut(notes[i]):
                continue
            if i not in state.mutable:
                raise ValueError("forbidden reach cut in frozen retained note")
            if any(try_edits({i: option}) for option in options(i)):
                continue
            report["reach_broadened_windows"] += 1
            if joint(i, allowed):
                continue
            repaired = False
            # Reopen the nearest completed groups only; preserve all others.
            for width in range(1, min(3, len(completed))+1):
                released = {j for block in completed[-width:] for j in block} & state.mutable
                frozen.difference_update(released)
                repaired = joint(i, allowed | released)
                frozen.update(released)
                if repaired:
                    report["reach_backtracks"].append(dict(note_id=notes[i]["id"], time=notes[i]["time"], later_groups=width))
                    break
            if not repaired:
                raise ValueError(f"infeasible reach repair near note {notes[i]['id']} at {notes[i]['time']:.6f}s")
        frozen.update(group)
        completed.append(group)


def repair_reach(notes, cap, *, nps=None, protect_retained=False, hard_only=False, guard=None):
    """Count-preserving mandatory repair followed by guarded optional tails.

    Hard repair tries the original cut on nearby cells first and freezes later
    groups in reverse traversal. A bounded two-note retry can reopen at most
    three later groups. Infeasibility rolls the complete invocation back.
    """
    ordered = sorted(notes, key=lambda n: (n["time"], n["id"]))
    nps = _measured_nps(ordered) if nps is None else float(nps)
    before = reach_costs(ordered, nps)
    initial_diagonals = sum(n["d"] in (4, 5, 6, 7) for n in ordered)
    diagonal_limit = max(initial_diagonals, math.floor(sum(n["d"] != 8 for n in ordered)*diagonal_fraction_limit(nps)+_TOL))
    report = {"reach_hard_before": before["hard_bans"], "reach_hard_after": before["hard_bans"],
              "reach_hard_bans_before": before["hard_bans"], "reach_hard_bans_after": before["hard_bans"],
              "reach_changed": 0, "reach_candidates": 0, "reach_broadened_windows": 0, "reach_backtracks": [],
              "reach_excess_before": before["total_excess"], "reach_excess_after": before["total_excess"],
              "reach_diagonal_net": 0, "reach_diagonal_limit": diagonal_limit, "reach_diagonal_limit_exceeded": False,
              "reach_reference_band": before["nps_band"], "reach_reference_version": _REFERENCE["version"]}
    if protect_retained and any(forbidden_cut(n) and n.get("kind") == "retained" for n in ordered):
        raise ValueError("forbidden reach cut in frozen retained note")
    if not ordered:
        return report
    original = {i: (n["x"], n["y"], n["d"]) for i, n in enumerate(ordered)}
    state = _Search(ordered, cap, nps, protect_retained)
    checkpoint = guard.checkpoint() if guard is not None else None
    clearance_before = state.post.clearance_violations(ordered)
    speed_before = state.speed_groups.violations(range(len(ordered)))
    try:
        _repair_hard_reverse(state, report)
        if guard is not None:
            guard.commit({n["id"] for i, n in enumerate(ordered) if original[i] != state.committed[i]})
        if not protect_retained and not hard_only:
            from map_creation.v2_playability_placement import placement_costs
            placement = placement_costs(ordered, nps) if guard is not None else None
            tail_ids = {identity for metric in before["metrics"].values()
                        for value, ids in zip(metric["severities"], metric["event_note_ids"])
                        if value > metric["cutoff"] for identity in ids}
            if placement is not None:
                tail_ids.update(identity for metric in placement["metrics"].values()
                                for value, ids in zip(metric["severities"], metric["event_note_ids"])
                                if value > metric["cutoff"] for identity in ids)
            state.limit_diagonals = True
            for i in reversed(range(len(ordered))):
                if i not in state.mutable or ordered[i]["id"] not in tail_ids:
                    continue
                context = state.context(i)
                def cost():
                    rank = state.rank(context, {}, original)
                    place = placement_costs(context.notes, nps)["total_excess"] if placement is not None else 0.
                    return rank[0], rank[1]+place
                incumbent = cost()
                current = {i: (ordered[i]["x"], ordered[i]["y"], ordered[i]["d"])}
                for option in state.options(i):
                    if option == current[i]:
                        continue
                    report["reach_candidates"] += 1
                    state.install({i: option})
                    if (state.legal({i}) and cost() < incumbent
                            and (guard is None or guard.allows({ordered[i]["id"]}))):
                        state.commit({i: option})
                        if guard is not None:
                            guard.commit({ordered[i]["id"]})
                        break
                    state.install(current)
        after = reach_costs(ordered, nps)
        if after["hard_bans"]:
            raise ValueError("reach repair left forbidden cuts")
        clearance_after = state.post.clearance_violations(ordered)
        speed_after = state.speed_groups.violations(range(len(ordered)))
        report.update(reach_clearance_before=clearance_before, reach_clearance_after=clearance_after,
                      reach_speed_violations_before=sorted(speed_before), reach_speed_violations_after=sorted(speed_after))
        if any(clearance_after[k] > clearance_before[k] for k in clearance_before) or speed_after-speed_before:
            raise ValueError("reach repair introduced a physical violation")
    except Exception:
        state.install(original)
        if checkpoint is not None:
            guard.restore(checkpoint)
        raise
    report.update(reach_hard_after=0, reach_hard_bans_after=0,
                  reach_changed=sum((n["x"], n["y"], n["d"]) != original[i] for i, n in enumerate(ordered)),
                  reach_diagonal_net=sum(n["d"] in (4, 5, 6, 7) for n in ordered)-initial_diagonals,
                  reach_diagonal_limit_exceeded=sum(n["d"] in (4, 5, 6, 7) for n in ordered)>state.diagonal_limit,
                  reach_excess_after=after["total_excess"])
    return report
