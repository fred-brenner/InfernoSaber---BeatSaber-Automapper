"""Shared local acceptance for the complete experimental playability policy.

Optional changes compare with the last accepted stream, so a later repair
cannot spend an earlier pass's improvement. Mandatory hard repair is separate.
The live dictionaries are shared with all repair modules; snapshots are private.
"""

from bisect import bisect_left, bisect_right
from functools import lru_cache

from tools.config import config


@lru_cache(maxsize=32768)
def _cached_speed(previous, current, gap, calculate, parameters):
    return calculate(list(previous), list(current), gap)


class SpeedGroups:
    """Distinct-time same-hand arrival budgets; simultaneous stacks share travel."""
    def __init__(self, notes, cap, eps):
        self.notes, self.cap = notes, cap
        self.previous, self.partners, self.following = {}, {}, {}
        for hand in (0, 1):
            groups = []
            for i, n in enumerate(notes):
                if n['hand'] != hand:
                    continue
                if not groups or n['time']-notes[groups[-1][0]]['time'] > eps+1e-9:
                    groups.append([])
                groups[-1].append(i)
            for offset, group in enumerate(groups):
                for i in group:
                    self.partners[i] = group
                    self.previous[i] = groups[offset-1] if offset else []
                    self.following[i] = groups[offset+1] if offset+1 < len(groups) else []

    def legal(self, changed):
        return not self.violations(changed)

    def violations(self, changed):
        from map_creation.v2_postprocessing import calc_note_speed
        parameters = (config.max_speed, config.reaction_time,
                      config.reaction_time_fact, config.cdf, config.cdf_lr)
        arrivals = {j for i in changed for j in self.partners[i]+self.following[i]}
        violations = set()
        for i in arrivals:
            previous = self.previous[i]
            if not previous:
                continue
            n = self.notes[i]
            current = tuple(n[k] for k in ('x', 'y', 'hand', 'd'))
            speed = min(_cached_speed(tuple(self.notes[p][k] for k in ('x', 'y', 'hand', 'd')),
                                     current, n['time']-self.notes[p]['time'],
                                     calc_note_speed, parameters) for p in previous)
            if speed > self.cap(n)+1e-9:
                violations.add(n['id'])
        return violations


class PatternGuard:
    def __init__(self, notes, nps=None):
        from map_creation.v2_playability_reach import note_span_nps
        self.notes = sorted(notes, key=lambda n: (n['time'], n['id']))
        self.index = {n['id']: i for i, n in enumerate(self.notes)}
        if len(self.index) != len(self.notes):
            raise ValueError('duplicate note identity in pattern guard')
        self.times = [n['time'] for n in self.notes]
        self.accepted = [dict(n) for n in self.notes]
        self.nps = note_span_nps(self.notes) if nps is None else float(nps)
        from map_creation.v2_playability_directions import direction_tail_costs
        self.direction_costs = direction_tail_costs(self.notes, self.nps)
        self.coordination_costs = _coordination_features(self.notes, self.nps)
        self.checked = self.rejected = self.commits = 0
        self.reasons = {}

    def _indices(self, identities):
        try:
            return sorted(self.index[i] for i in identities)
        except KeyError as exc:
            raise ValueError('unknown note identity in pattern guard') from exc

    def _windows(self, indices):
        # Both short complaint passages and longer exposure remain visible.
        windows = set()
        for i in indices:
            for radius in (1., 4.):
                windows.add((bisect_left(self.times, self.times[i]-radius),
                             bisect_right(self.times, self.times[i]+radius)))
        return sorted(windows)

    def allows(self, changed_ids):
        indices = self._indices(changed_ids)
        if not any(self.notes[i] != self.accepted[i] for i in indices):
            return True
        self.checked += 1
        if any((self.notes[i]['hand'], self.notes[i]['d']) != (self.accepted[i]['hand'], self.accepted[i]['d']) for i in indices):
            # Direction runs may span far beyond a short geometry window.
            # Full per-hand tails have fixed eligible-pair denominators.
            from map_creation.v2_playability_directions import direction_tail_costs
            proposed = direction_tail_costs(self.notes, self.nps)
            for hand, value in proposed['per_hand'].items():
                if value['score'] > self.direction_costs['per_hand'][hand]['score']+1e-8:
                    key = 'direction_full_hand'+hand
                    self.rejected += 1
                    self.reasons[key] = self.reasons.get(key, 0)+1
                    return False
        if any((self.notes[i]['hand'], self.notes[i]['x'], self.notes[i]['y']) != (self.accepted[i]['hand'], self.accepted[i]['x'], self.accepted[i]['y']) for i in indices):
            proposed = _coordination_features(self.notes, self.nps)
            for key, value in proposed.items():
                if value > self.coordination_costs[key]+1e-8:
                    reason = 'global_'+key
                    self.rejected += 1
                    self.reasons[reason] = self.reasons.get(reason, 0)+1
                    return False
        for lo, hi in self._windows(indices):
            old = _features(self.accepted[lo:hi], self.nps)
            new = _features(self.notes[lo:hi], self.nps)
            for key, value in new.items():
                # A rare sustained-horizontal tail may spend ordinary M4
                # variation while keeping the calibrated M4 tail unchanged.
                # A comfortable phrase has no such exception.
                if (key == 'interference' and new['m1_run'] < old['m1_run']-1e-8
                        and new['lateral'] < old['lateral']-1e-8
                        and new['interference_tail'] <= old['interference_tail']+1e-8):
                    continue
                # A bounded diagonal bridge can rotate a sustained harmful
                # horizontal phrase. It must reduce load and keep dwell tails.
                if (key == 'axis_fast45' and new['lateral'] < old['lateral']-1e-8
                        and value <= old[key]+2 and new['direction_tail'] <= old['direction_tail']+1e-8):
                    continue
                if value > old[key] + 1e-8:
                    self.rejected += 1
                    self.reasons[key] = self.reasons.get(key, 0) + 1
                    return False
        return True

    def commit(self, changed_ids):
        indices = self._indices(changed_ids)
        if any((self.notes[i]['hand'], self.notes[i]['d']) != (self.accepted[i]['hand'], self.accepted[i]['d']) for i in indices):
            from map_creation.v2_playability_directions import direction_tail_costs
            self.direction_costs = direction_tail_costs(self.notes, self.nps)
        if any((self.notes[i]["hand"], self.notes[i]["x"], self.notes[i]["y"]) != (self.accepted[i]["hand"], self.accepted[i]["x"], self.accepted[i]["y"]) for i in indices):
            self.coordination_costs = _coordination_features(self.notes, self.nps)
        for i in indices:
            self.accepted[i] = dict(self.notes[i])
        self.commits += 1

    def checkpoint(self):
        return [dict(n) for n in self.accepted], self.commits, self.direction_costs, self.coordination_costs

    def restore(self, checkpoint):
        self.accepted, self.commits, self.direction_costs, self.coordination_costs = checkpoint

    def report(self):
        return dict(pattern_guard_checked=self.checked, pattern_guard_rejected=self.rejected,
                    pattern_guard_commits=self.commits, pattern_guard_reasons=dict(self.reasons))


def _features(notes, nps):
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_reach import _Window, _travel, reach_costs
    from map_creation.v2_playability_placement import placement_costs
    from map_creation.v2_playability_costs import calibrated_limits
    from map_creation.v2_playability_geometry import _divergence
    import numpy as np

    result = dict(lateral=0., travel=0., peak_travel=0., axis_fast45=0., axis_fast90=0., detours=0.)
    for hand in (0, 1):
        seq = []
        for n in notes:
            if n['hand'] == hand and (not seq or n['time']-seq[-1]['time'] > .02+1e-9):
                seq.append(n)
        for a, b in zip(seq, seq[1:]):
            gap = b['time']-a['time']
            if b['d'] == 8:
                continue
            lateral = 1. if b['d'] in (2, 3) else .5 if b['d'] in (4, 5, 6, 7) else 0.
            result['lateral'] += lateral*max(0., 1.-gap/.5)
            if a['d'] == 8:
                continue
            # Recovery-adjusted physical movement, also below the severe tail.
            travel = _travel(a, b, gap)
            result['travel'] += travel
            result['peak_travel'] = max(result['peak_travel'], travel)
            aa, ba = post._DIRECTION_ANGLE[a['d']] % 4, post._DIRECTION_ANGLE[b['d']] % 4
            steps = min((ba-aa) % 4, (aa-ba) % 4)
            if gap < .5-1e-9 and steps:
                result['axis_fast45' if steps == 1 else 'axis_fast90'] += 1
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            if 8 not in (a['d'], b['d'], c['d']):
                axes = [post._DIRECTION_ANGLE[n['d']] % 4 for n in (a, b, c)]
                if axes[0] == axes[2] != axes[1] and min(b['time']-a['time'], c['time']-b['time']) < .5:
                    result['detours'] += 1
    context = _Window(notes, nps, post)
    result['occlusion'], result['interference'] = context.geometry(raw=True)
    result['interference_tail'] = context.geometry()[1]
    from map_creation.v2_playability_lateral import consecutive_horizontal_load
    result['m1_run'] = consecutive_horizontal_load(notes, nps)['total_excess']
    reach = reach_costs(notes, nps)
    result.update({'reach_'+key: value['excess'] for key, value in reach['metrics'].items()})
    placement = placement_costs(notes, nps)
    result.update({'placement_'+key: value['excess'] for key, value in placement['metrics'].items()})
    result.update(_coordination_features(notes, nps))
    from map_creation.v2_playability_directions import direction_tail_costs
    result['direction_tail'] = direction_tail_costs(notes, nps)['score']
    return result


def _coordination_features(notes, nps):
    from map_creation.v2_playability_costs import calibrated_limits
    from map_creation.v2_playability_geometry import _divergence
    import numpy as np
    result = {}
    # Same raw displacement, dot placement and activity segmentation as M3.
    from map_creation.v2_playability_costs import m3_segment_cutoff
    segments = []
    last = {}
    for n in notes:
        if n['hand'] in last and n['time']-last[n['hand']] <= .02+1e-9:
            continue
        last[n['hand']] = n['time']
        if not segments or n['time']-segments[-1][-1]['time'] > .5:
            segments.append([])
        segments[-1].append(n)
    numerator = weight = excess = eligible = 0.
    for segment in segments:
        scatters, counts = [], []
        for hand in (0, 1):
            seq = [n for n in segment if n['hand'] == hand]
            coords = np.array([(n["x"], n["y"]) for n in seq], dtype=float).reshape(-1, 2)
            vv = np.diff(coords, axis=0)
            counts.append(len(vv))
            scatters.append(vv.T @ vv)
        divergence = _divergence(scatters, counts)
        if divergence is not None:
            eligible += 1
            excess += max(0., divergence-m3_segment_cutoff(nps))
            w = min(counts)
            numerator += divergence*w
            weight += w
    low, high = calibrated_limits(nps)["m3_normal"]
    result['coordination_high'] = excess
    result['coordination_low'] = max(0., low-numerator/weight) if weight else 0.
    result['coordination_support'] = -eligible
    result["coordination_map_high"] = max(0., numerator/weight-high) if weight else 0.
    return result
