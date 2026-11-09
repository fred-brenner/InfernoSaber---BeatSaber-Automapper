"""Bounded, opt-in cell repairs before retained v2 notes are frozen.

The generation costs below use local note geometry, without importing the
corpus scorer. Times, identities, hands and arrows stay fixed. Each note moves
at most one cell per invocation; an incumbent wins every tie. Neighborhoods,
travel vectors and scatter matrices are cached so a proposal does not rescore
the complete song. The original primary-pass arrival cap is supplied by caller.
"""

from bisect import bisect_left, bisect_right
from collections import ChainMap
from itertools import product

import numpy as np

from tools.config import config
from map_creation import v2_postprocessing as post
from map_creation import v2_playability_costs as combined_costs

# Frozen offline human M3 medians by note-span NPS (<3, 3–5, 5–7, >=7).
# 51 curated highest-Standard maps, eligible support 7 / 27 / 7 / 10.
# playability-m5-by-app-20261107/full-report.json.gz SHA256:
# 45f5fa5677dc6780f268f054e3894a47e7f9abe619d702072eacd3ba32c80f9a
# All bands have >=5 eligible human maps; no adjacent-band fallback was needed.
_M3_REFERENCE = (.08701159823732912, .15450233640937738,
                 .1304347475082495, .09284348342647153)
_EPS, _VIS, _PAIR, _RESET, _SEGMENT = .02, .4, .25, .4, .5
_TOL = 1e-9
_MIRROR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}


def _corridor(a, b):
    """Grid-clamped supercover between predecessor exit and note approach."""
    x, y = (min(3, max(0, int(a[0]))), min(2, max(0, int(a[1]))))
    x1, y1 = min(3, max(0, int(b[0]))), min(2, max(0, int(b[1])))
    dx, dy = x1-x, y1-y
    nx, ny, sx, sy = abs(dx), abs(dy), int(np.sign(dx)), int(np.sign(dy))
    cells, ix, iy = {(x, y)}, 0, 0
    while ix < nx or iy < ny:
        decision = (1+2*ix)*ny-(1+2*iy)*nx
        if decision == 0:
            x, y, ix, iy = x+sx, y+sy, ix+1, iy+1
            cells.update(((x-sx, y), (x, y-sy), (x, y)))
        elif decision < 0:
            x, ix = x+sx, ix+1
            cells.add((x, y))
        else:
            y, iy = y+sy, iy+1
            cells.add((x, y))
    return cells


def _axis(scatter, count):
    if count < 2:
        return None
    values, vectors = np.linalg.eigh(scatter)
    total = float(values.sum())
    return vectors[:, 1] if total > 0 and values[1]/total >= .6 else None


def _divergence(scatters, counts):
    left, right = (_axis(scatters[h], counts[h]) for h in (0, 1))
    if left is None or right is None:
        return None
    return float(min(abs(left[0]*right[1]-left[1]*right[0]),
                     abs(left[0]*right[1]+left[1]*right[0])))


class _Geometry:
    """Immutable time neighborhoods with incremental position-dependent costs."""

    def __init__(self, notes, cap, guard=None):
        self.notes, self.cap, self.guard = notes, cap, guard
        from map_creation.v2_playability_policy import SpeedGroups
        self.speed_groups = SpeedGroups(notes, cap, float(getattr(config, "min_time_diff", .02)))
        self.combined = bool(getattr(config, "v2_playability_reach", False))
        self.times = [n["time"] for n in notes]
        self.eps = float(getattr(config, "min_time_diff", .02))
        self.window = max(self.eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0.))
        self.gap = float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0.)
        self.previous, self.following, self.flow_previous = {}, {}, {}
        self.metric_previous, self.metric_following, self.kept = {}, {}, []
        for hand in (0, 1):
            seq = [i for i, n in enumerate(notes) if n["hand"] == hand]
            times = [self.times[i] for i in seq]
            anchor = None
            for offset, i in enumerate(seq):
                if offset:
                    self.previous[i] = seq[offset-1]
                    self.following[seq[offset-1]] = i
                before = bisect_left(times, self.times[i]-self.eps)-1
                if before >= 0:
                    self.flow_previous[i] = seq[before]
                if anchor is None or self.times[i]-self.times[anchor] > _EPS+_TOL:
                    if anchor is not None:
                        self.metric_previous[i] = anchor
                        self.metric_following[anchor] = i
                    self.kept.append(i)
                    anchor = i
        self.children = {}
        for previous in (self.flow_previous, self.metric_previous):
            children = {}
            for child, parent in previous.items():
                children.setdefault(parent, []).append(child)
            self.children[id(previous)] = children
        self.kept.sort()
        self.kept_set = set(self.kept)
        span = self.times[-1]-self.times[0] if self.times else 0.
        self.nps = len(self.kept)/span if span > 0 else 0.
        self.limits = combined_costs.calibrated_limits(self.nps) if self.combined else None
        self.effective = self._directions(self.flow_previous)
        self.metric_effective = self._directions(self.metric_previous)
        self.near, self.occluders, self.partners = {}, {}, {}
        for i in range(len(notes)):
            self.near[i] = list(range(bisect_left(self.times, self.times[i]-max(self.window, self.eps)-_TOL),
                                      bisect_right(self.times, self.times[i]+max(self.window, self.eps)+_TOL)))
        for i in self.kept:
            lo, hi = bisect_left(self.times, self.times[i]-_VIS-_TOL), bisect_left(self.times, self.times[i])
            self.occluders[i] = [j for j in reversed(range(lo, hi))
                                  if j in self.kept_set and _EPS < self.times[i]-self.times[j] <= _VIS]
            self.partners[i] = []
        # Build each pair once using the actual subtraction, then attach it
        # to both notes. Floating t +/- window bounds can differ by one ULP;
        # independent bisects previously created one-sided cache entries.
        for i in self.kept:
            hi = bisect_right(self.times, self.times[i]+_PAIR+_TOL)
            for j in range(i+1, hi):
                if (j in self.kept_set and notes[j]["hand"] != notes[i]["hand"]
                        and self.times[j]-self.times[i] <= _PAIR):
                    self.partners[i].append(j)
                    self.partners[j].append(i)
        self.volumes = {i: self._volume(i, self.metric_effective) for i in self.kept}
        self.m2 = {i: self._occlusion(i) for i in self.kept}
        self.m4 = {(i, j): self._interference(i, j, self.volumes)
                   for i in self.kept if notes[i]["hand"] == 0 for j in self.partners[i]}
        self.segment_for, self.segments = {}, []
        for i in self.kept:
            if not self.segments or self.times[i]-self.times[self.segments[-1][-1]] > _SEGMENT:
                self.segments.append([])
            self.segments[-1].append(i)
            self.segment_for[i] = len(self.segments)-1
        self.edges, self.scatter, self.counts, self.support = {}, [], [], []
        for segment_id, segment in enumerate(self.segments):
            matrices, counts, support = [np.zeros((2, 2)), np.zeros((2, 2))], [0, 0], [0, 0]
            for hand in (0, 1):
                seq = [i for i in segment if notes[i]["hand"] == hand]
                for a, b in zip(seq, seq[1:]):
                    v = self._vector(a, b)
                    self.edges[b] = (a, hand, segment_id, v)
                    matrices[hand] += np.outer(v, v)
                    counts[hand] += 1
                    support[hand] += bool(np.any(v))
            self.scatter.append(matrices)
            self.counts.append(counts)
            self.support.append(support)
        self.m3 = [_divergence(s, c) for s, c in zip(self.scatter, self.counts)]
        self.m3_weight = sum(min(self.counts[s]) for s, value in enumerate(self.m3) if value is not None)
        self.m3_numerator = sum(value*min(self.counts[s]) for s, value in enumerate(self.m3) if value is not None)
        self.reach = {}
        self.reach_components = {}
        self.proposal_reach_delta = 0.
        if self.combined:
            from map_creation.v2_playability_reach import local_reach_components
            self.local_reach_components = local_reach_components
            self.reach_components = {i: self._reach_components(i) for i in range(len(notes))}
            self.reach = {i: sum(values.values()) for i, values in self.reach_components.items()}

    def _reach_components(self, i):
        p = self.previous.get(i)
        # A stack has ambiguous physical pairing. Never invent a zero-gap
        # endpoint transition; its position still receives calibrated cost.
        while p is not None and self.times[i]-self.times[p] <= 1e-7:
            p = self.previous.get(p)
        return self.local_reach_components(None if p is None else self.notes[p], self.notes[i], self.nps)

    def _directions(self, previous):
        effective = {}
        for i, note in enumerate(self.notes):
            if note["d"] != 8:
                effective[i] = note["d"]
            else:
                p = previous.get(i)
                effective[i] = self._dot(i, p, effective)
        return effective

    def _dot(self, i, p, effective):
        if p is None or effective.get(p) is None:
            return None
        n, before = self.notes[i], self.notes[p]
        return post._dot_virtual_direction(effective[p], before["x"], before["y"], n["x"], n["y"])

    def _volume(self, i, effective):
        note, direction = self.notes[i], effective[i]
        x, y = note["x"], note["y"]
        volume = {(x, y)} | set(post._swept_cells(x, y, direction))
        p = self.metric_previous.get(i)
        if p is not None and self.times[i]-self.times[p] <= _RESET+_TOL:
            before = self.notes[p]
            sx, sy = post._DIRECTION_STEP.get(effective[p], (0, 0))
            dx, dy = post._DIRECTION_STEP.get(direction, (0, 0))
            volume |= _corridor((before["x"]-sx, before["y"]-sy), (x+dx, y+dy))
        return volume

    def _occlusion(self, i):
        note = self.notes[i]
        for j in self.occluders[i]:
            before = self.notes[j]
            dx = abs(note["x"]-before["x"])
            if dx <= 1:
                if self.combined:
                    return combined_costs.occlusion_severity(note, before, self.times[i]-self.times[j])
                return max(0, note["y"]-before["y"])*(1-dx/3)*(1-(self.times[i]-self.times[j])/_VIS)
        return 0.

    def _interference(self, a, b, volumes):
        left, right = self.notes[a], self.notes[b]
        dt = abs(self.times[a]-self.times[b])
        if self.combined:
            return combined_costs.interference_severity(left, right, dt, volumes[a], volumes[b])
        if (dt <= _EPS+_TOL and left["x"]+right["x"] == 3 and left["y"] == right["y"]
                and right["d"] == _MIRROR[left["d"]]):
            return 0.
        return (len(volumes[a] & volumes[b])+max(0, left["x"]-right["x"]))*max(0, 1-dt/_PAIR)

    def _vector(self, a, b):
        return np.array([self.notes[b]["x"]-self.notes[a]["x"],
                         self.notes[b]["y"]-self.notes[a]["y"]], dtype=float)

    def _affected(self, changed, previous):
        # Traverse only descendants through dots plus the first outgoing
        # arrow edge. An unchanged arrow stops geometry propagation.
        affected, pending = set(changed), list(changed)
        children = self.children[id(previous)]
        while pending:
            for i in children.get(pending.pop(), ()):
                if i in affected:
                    continue
                affected.add(i)
                if self.notes[i]["d"] == 8:
                    pending.append(i)
        return affected

    def _legal(self, changed, effective):
        if self.combined:
            from map_creation.v2_playability_reach import forbidden_cut
            if any(forbidden_cut(self.notes[i]) for i in changed):
                return False
        affected = self._affected(changed, self.flow_previous)
        if not self.speed_groups.legal(changed):
            return False
        for i in affected:
            note, p = self.notes[i], self.flow_previous.get(i)
            prior = None if p is None else (self.times[p], self.notes[p]["x"], self.notes[p]["y"], effective[p])
            if note["d"] != 8 and not post._flow_legal(note["d"], prior, note["x"], note["y"], note["time"], self.gap):
                return False
            if i in changed and getattr(config, "v2_fast_counter_direction", "recut") == "recut":
                fast = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
                adjacent = [j for j in (self.previous.get(i), self.following.get(i)) if j is not None]
                if (any(abs(self.times[i]-self.times[j]) < fast for j in adjacent)
                        and post._counter_direction(note)):
                    return False
            for j in self.near[i]:
                if j == i:
                    continue
                other, dt = self.notes[j], self.times[j]-self.times[i]
                same_cell = (note["x"], note["y"]) == (other["x"], other["y"])
                if abs(dt) <= self.eps and same_cell:
                    return False
                if note["hand"] == other["hand"]:
                    continue
                if -self.eps <= dt <= self.window:
                    if ((other["x"], other["y"]) in post._swept_cells(note["x"], note["y"], effective[i])
                            or (same_cell and dt > self.eps)):
                        return False
                if -self.window <= dt <= self.eps:
                    if ((note["x"], note["y"]) in post._swept_cells(other["x"], other["y"], effective[j])
                            or (same_cell and -dt > self.eps)):
                        return False
        return True

    def proposal(self, edits, *, commit=False):
        self.proposal_reach_delta = 0.
        changed = set(edits)
        old = {i: (self.notes[i]["x"], self.notes[i]["y"]) for i in changed}
        for i, (x, y) in edits.items():
            self.notes[i]["x"], self.notes[i]["y"] = x, y
        committed = False
        try:
            affected = self._affected(changed, self.flow_previous)
            effective = ChainMap({}, self.effective)
            for i in sorted(affected):
                if self.notes[i]["d"] == 8:
                    effective[i] = self._dot(i, self.flow_previous.get(i), effective)
            if not self._legal(changed, effective):
                return None
            metric_affected = self._affected(changed & self.kept_set, self.metric_previous)
            metric_eff = ChainMap({}, self.metric_effective)
            for i in sorted(metric_affected):
                if self.notes[i]["d"] == 8:
                    metric_eff[i] = self._dot(i, self.metric_previous.get(i), metric_eff)
            volume_ids = metric_affected | {self.metric_following[i] for i in metric_affected if i in self.metric_following}
            volumes = ChainMap({}, self.volumes)
            volumes.update({i: self._volume(i, metric_eff) for i in volume_ids})
            m2_ids = {j for i in changed for j in range(i, bisect_right(self.times, self.times[i]+_VIS+_TOL)) if j in self.kept_set}
            m2 = {i: self._occlusion(i) for i in m2_ids}
            pair_ids = {(i, j) if self.notes[i]["hand"] == 0 else (j, i)
                        for i in volume_ids for j in self.partners[i]}
            m4 = {pair: self._interference(*pair, volumes) for pair in pair_ids}
            edge_ids = (changed | {self.metric_following[i] for i in changed if i in self.metric_following}) & self.edges.keys()
            segment_ids = {self.edges[i][2] for i in edge_ids}
            scatters = {s: [a.copy() for a in self.scatter[s]] for s in segment_ids}
            support = {s: self.support[s][:] for s in segment_ids}
            vectors = {}
            for i in edge_ids:
                a, hand, s, previous = self.edges[i]
                v = self._vector(a, i)
                vectors[i] = v
                scatters[s][hand] += np.outer(v, v)-np.outer(previous, previous)
                support[s][hand] += int(bool(np.any(v)))-int(bool(np.any(previous)))
            m3 = {s: _divergence(scatters[s], self.counts[s]) for s in segment_ids}
            # Preserve eligibility and nonzero displacement support on each
            # hand: becoming stationary cannot manufacture an M3 improvement.
            for s in segment_ids:
                if (any(support[s][h] < self.support[s][h] for h in (0, 1))
                        or (self.m3[s] is not None and m3[s] is None)):
                    return None
            deltas = (sum(m2[i]-self.m2[i] for i in m2),
                      sum(m4[p]-self.m4[p] for p in m4),
                      sum(((m3[s] or 0)-(self.m3[s] or 0))*min(self.counts[s]) for s in m3))
            m3_delta = deltas[2]
            m3_weight_delta = sum(min(self.counts[s])*((m3[s] is not None)-(self.m3[s] is not None)) for s in m3)
            if self.combined:
                old_scalar = self.m3_numerator/max(1, self.m3_weight)
                new_scalar = (self.m3_numerator+m3_delta)/max(1, self.m3_weight+m3_weight_delta)
                low, high = self.limits["m3_normal"]
                # A clear sightline is not worth flattening otherwise human
                # bimanual motion. Already-low streams may rise naturally,
                # but cannot drift still farther below the human lower quartile.
                if new_scalar < min(old_scalar, low)-_TOL:
                    return None
                event_cost = (lambda value, cutoff: max(0., value-cutoff)/(1+cutoff)) if self.guard is not None else combined_costs.soft_event_cost
                deltas = (sum(event_cost(m2[i], self.limits["m2"])-event_cost(self.m2[i], self.limits["m2"]) for i in m2),
                          sum(event_cost(m4[p], self.limits["m4"])-event_cost(self.m4[p], self.limits["m4"]) for p in m4),
                          (sum(max(0., (m3[s] or 0.)-combined_costs.m3_segment_cutoff(self.nps))
                               -max(0., (self.m3[s] or 0.)-combined_costs.m3_segment_cutoff(self.nps)) for s in segment_ids)
                           if self.guard is not None else max(0., new_scalar-high)-max(0., old_scalar-high)))
                reach_ids = changed | {self.following[i] for i in changed if i in self.following}
                # Include outgoing stacked arrivals that resolve to the same
                # predecessor. The time-bounded local refresh never walks the
                # whole song for a cell proposal.
                pending = list(reach_ids)
                while pending:
                    i = pending.pop()
                    j = self.following.get(i)
                    if j is not None and j not in reach_ids and self.times[j]-self.times[i] <= 1e-7:
                        reach_ids.add(j)
                        pending.append(j)
                reach_components = {i: self._reach_components(i) for i in reach_ids}
                reach = {i: sum(values.values()) for i, values in reach_components.items()}
                self.proposal_reach_delta = sum(reach[i]-self.reach[i] for i in reach_ids)
                # Visibility/path moves must not manufacture another extreme
                # preparation or travel event on an incoming/outgoing edge.
                if any(value > self.reach_components[i].get(metric, 0.)+_TOL
                       for i in reach_ids for metric, value in reach_components[i].items()):
                    return None
            if self.guard is not None and not self.guard.allows({self.notes[i]["id"] for i in edits}):
                return None
            if any(delta > _TOL for delta in deltas):
                return None
            if commit:
                self.effective.update(effective.maps[0])
                self.metric_effective.update(metric_eff.maps[0])
                self.volumes.update(volumes.maps[0])
                self.m2.update(m2)
                self.m4.update(m4)
                for s in segment_ids:
                    self.scatter[s], self.support[s], self.m3[s] = scatters[s], support[s], m3[s]
                for i, v in vectors.items():
                    self.edges[i] = (*self.edges[i][:3], v)
                self.m3_numerator += m3_delta
                self.m3_weight += m3_weight_delta
                if self.combined:
                    self.reach.update(reach)
                    self.reach_components.update(reach_components)
                if self.guard is not None:
                    self.guard.commit({self.notes[i]["id"] for i in edits})
                committed = True
            return deltas
        finally:
            if not committed:
                for i, (x, y) in old.items():
                    self.notes[i]["x"], self.notes[i]["y"] = x, y

    def costs(self):
        weight = sum(min(self.counts[s]) for s, value in enumerate(self.m3) if value is not None)
        result = {"m2": sum(self.m2.values())/max(1, len(self.kept)),
                "m4_severity": sum(self.m4.values()),
                "m3": sum(value*min(self.counts[s]) for s, value in enumerate(self.m3)
                           if value is not None)/weight if weight else 0.,
                "m3_segments": sum(value is not None for value in self.m3),
                "m3_nonzero_vectors": [sum(s[h] for s in self.support) for h in (0, 1)],
                "stationary_moves": sum(sum(c)-sum(v) for c, v in zip(self.counts, self.support)),
                "hand_notes": [sum(n["hand"] == h for n in self.notes) for h in (0, 1)],
                "lane_counts": [sum(n["x"] == x for n in self.notes) for x in range(4)],
                "row_counts": [sum(n["y"] == y for n in self.notes) for y in range(3)],
                "crossed_hand_pairs": sum(self.notes[a]["x"] > self.notes[b]["x"] for a, b in self.m4),
                "mirrored_doubles": sum(abs(self.times[a]-self.times[b]) <= _EPS+_TOL
                                        and (not self.combined or self.notes[a]["x"] <= self.notes[b]["x"])
                                        and self.notes[a]["x"]+self.notes[b]["x"] == 3
                                        and self.notes[a]["y"] == self.notes[b]["y"]
                                        and self.notes[b]["d"] == _MIRROR[self.notes[a]["d"]]
                                        for a, b in self.m4)}
        if self.combined:
            result["geometry_version"] = combined_costs.VERSION
            result["m2_tail_events"] = sum(v > self.limits["m2"] for v in self.m2.values())
            result["m4_tail_events"] = sum(v > self.limits["m4"] for v in self.m4.values())
            result["m3_normal_range"] = self.limits["m3_normal"]
            result["m3_excess"] = max(0., result["m3"]-self.limits["m3_normal"][1])
            result["reach_local_excess"] = sum(self.reach.values())
            result["reach_component_excess"] = {
                metric: sum(values.get(metric, 0.) for values in self.reach_components.values())
                for metric in sorted(set().union(*(values.keys() for values in self.reach_components.values())))}
        return result


def _neighbors(note):
    x, y = note["x"], note["y"]
    return [(a, b) for a, b in ((x-1, y), (x, y-1), (x, y+1), (x+1, y))
            if 0 <= a < 4 and 0 <= b < 3]


def repair_geometry(notes, cap, *, sightline=False, coordination=False, interference=False, guard=None):
    """Repair surviving notes in place; return JSON-able diagnostics.

    Each component requires a strict intended-cost improvement with no increase
    in either other geometric cost. Every candidate checks production incoming
    and outgoing flow/speed, occupancy, both swing directions and dot history.
    """
    report = {"geometry_changed": 0, "geometry_sightline_changed": 0,
              "geometry_interference_changed": 0, "geometry_coordination_changed": 0,
              "geometry_candidates": 0}
    if not (sightline or coordination or interference) or len(notes) < 2:
        return report
    # Onset snapping can invert event IDs across hands. Every bisect cache
    # uses chronological order, while the caller's list and identities stay
    # untouched; the sorted list shares only the mutable note dictionaries.
    notes = sorted(notes, key=lambda note: (note["time"], note["id"]))
    original_positions = [(note, note["x"], note["y"]) for note in notes]
    checkpoint = guard.checkpoint() if guard is not None and hasattr(guard, "checkpoint") else None
    clearance_before = post.clearance_violations(notes)
    report["geometry_clearance_before"] = clearance_before
    geometry = _Geometry(notes, cap, guard=guard)
    if geometry.combined:
        from map_creation.v2_playability_reach import forbidden_cut
        forbidden_before = {n["id"] for n in notes if forbidden_cut(n)}
        report["geometry_forbidden_before"] = len(forbidden_before)
    report["geometry_before"] = geometry.costs()
    span = notes[-1]["time"]-notes[0]["time"]
    nps = len(geometry.kept)/span if span > 0 else 0.
    threshold = (geometry.limits["m3_normal"][1] if geometry.combined
                 else _M3_REFERENCE[sum(nps >= bound for bound in (3, 5, 7))])
    if guard is not None:
        threshold = combined_costs.m3_segment_cutoff(nps)
    report["geometry_coordination_threshold"] = threshold
    used = set()

    def choose(proposals, target, name):
        best, best_score = None, None
        for edits in proposals:
            if not edits or used.intersection(edits):
                continue
            report["geometry_candidates"] += 1
            deltas = geometry.proposal(edits)
            if deltas is None or deltas[target] >= -_TOL:
                continue
            reach_delta = geometry.proposal_reach_delta if geometry.combined else 0.
            rank = (deltas[target]+reach_delta, reach_delta, sum(deltas), len(edits))
            if best_score is None or rank < best_score:
                best, best_score = edits, rank
        if best is not None:
            geometry.proposal(best, commit=True)
            used.update(best)
            report["geometry_changed"] += len(best)
            report[f"geometry_{name}_changed"] += len(best)
            return True
        return False

    if sightline:
        for i in geometry.kept:
            if (geometry.m2[i] <= geometry.limits["m2"] if geometry.combined
                    else geometry.m2[i] < .5-_TOL):
                continue
            occluder = next((j for j in geometry.occluders[i]
                             if abs(notes[i]["x"]-notes[j]["x"]) <= 1), None)
            participants = [i] if occluder is None else [i, occluder]
            choose(({j: cell} for j in participants for cell in _neighbors(notes[j])), 0, "sightline")
    if interference:
        for a, b in list(geometry.m4):
            if geometry.m4[(a, b)] <= (geometry.limits["m4"] if guard is not None and geometry.combined else _TOL):
                continue
            # Include predecessors: a legal standing-block layout can still
            # have an intersecting travel corridor into the next note.
            participants = sorted({a, b} | {geometry.metric_previous[i] for i in (a, b)
                                            if i in geometry.metric_previous})
            changed = choose(({j: cell} for j in participants for cell in _neighbors(notes[j])), 1, "interference")
            if geometry.combined and not changed and notes[a]["x"] > notes[b]["x"]:
                # A crossed double cannot uncross one note at a time when
                # the partner occupies its destination. Try the bounded pair
                # together; each note still moves only one cell in this pass.
                choose(({a: first, b: second} for first in _neighbors(notes[a])
                        for second in _neighbors(notes[b])), 1, "interference")
    if coordination:
        for s, segment in enumerate(geometry.segments):
            if geometry.combined and guard is None and geometry.m3_numerator/max(1, geometry.m3_weight) <= threshold+_TOL:
                break
            if geometry.m3[s] is None or geometry.m3[s] <= threshold+_TOL:
                continue
            for offset in range(0, len(segment), 4):
                if geometry.combined and guard is None and geometry.m3_numerator/max(1, geometry.m3_weight) <= threshold+_TOL:
                    break
                window = [i for i in segment[offset:offset+4] if i not in used]
                if geometry.m3[s] <= threshold+_TOL:
                    break
                if not window:
                    continue
                singles = ({i: cell} for i in window for cell in _neighbors(notes[i]))
                if choose(singles, 2, "coordination"):
                    continue
                # Joint edits can clear a collision that makes an isolated
                # move illegal. Four notes => at most 5**4-1 alternatives.
                options = [[None]+_neighbors(notes[i]) for i in window]
                choose(({i: cell for i, cell in zip(window, cells) if cell is not None}
                        for cells in product(*options)), 2, "coordination")
    clearance_after = post.clearance_violations(notes)
    report["geometry_rolled_back"] = any(clearance_after[key] > clearance_before[key]
                                        for key in clearance_before)
    if geometry.combined:
        forbidden_after = {n["id"] for n in notes if forbidden_cut(n)}
        report["geometry_rolled_back"] |= bool(forbidden_after-forbidden_before)
        report["geometry_forbidden_after"] = len(forbidden_after)
    if report["geometry_rolled_back"]:
        # Full-stream production checks are independent of local caches. A
        # missed edge must not reach the retained snapshot: revert the entire
        # position experiment atomically, preserving every original record.
        report["geometry_attempted_clearance"] = clearance_after
        for note, x, y in original_positions:
            note["x"], note["y"] = x, y
        if checkpoint is not None:
            guard.restore(checkpoint)
        for key in ("geometry_changed", "geometry_sightline_changed",
                    "geometry_interference_changed", "geometry_coordination_changed"):
            report[key] = 0
        report["geometry_after"] = report["geometry_before"]
        clearance_after = clearance_before
        if geometry.combined:
            report["geometry_attempted_forbidden_after"] = len(forbidden_after)
            report["geometry_forbidden_after"] = len(forbidden_before)
    else:
        report["geometry_after"] = geometry.costs()
    report["geometry_clearance_after"] = clearance_after
    return report
