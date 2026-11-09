"""Bounded, count-preserving priority playability repairs.

Hard crossing and preparation findings are never grandfathered. Failure to
find a legal assignment rolls back the entire call.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import lru_cache
import math

from map_creation import v2_playability_priority_metrics as metrics
from map_creation.v2_playability_reach import _Search, _Window, note_span_nps
from map_creation.v2_playability_policy import SpeedGroups
from tools.config import config

# A deterministic 3-second local work bucket, including prefilter attempts.
HARD_CLUSTER_SECONDS = 3.
HARD_CLUSTER_BUDGET = 600_000
VISIBILITY_CLUSTER_BUDGET = 160_000
VISIBILITY_MAX_PASSES = 12
VISIBILITY_EXCEPTION_POLICY = 'priority-visibility-rare-residual-v1-20261109'
VISIBILITY_EXCEPTION_RATE = .03


@lru_cache(maxsize=16)
def _soft_bounds(nps=None):
    import json
    from pathlib import Path
    source = json.loads(Path(__file__).with_name('v2_playability_priority_reference.json').read_text())
    if nps is None:
        reference = source['global_metrics']
    else:
        band = ('<3' if nps < 3 else '3-5' if nps < 5 else '5-7' if nps < 7 else '7+')
        reference = source['bands'][band]['metrics']
    return {'preparation': reference['preparation']['cutoff'],
            'axis': reference['axis']['cutoff'],
            'crossing': reference['slow_crossing']['cutoff'],
            **{key: reference[key]['cutoff'] for key in (
                'preparation_excursion', 'inner_inward', 'vertical_preparation', 'outer_placement')}}


def _soft_improves(score, incumbent, *, edit_count=1):
    """Higher priorities precede placement/M2, then recovery travel."""
    tol = 1e-8
    # Small isolated changes are soft costs; larger new local churn or slow
    # interference must not be bought by a single otherwise-attractive edit.
    if (score[1]-incumbent[1] > .5*max(1, edit_count)+tol
            or score[2]-incumbent[2] > max(1, edit_count)+tol):
        return False
    gain = ((incumbent[0]-score[0]) + 5.*(incumbent[1]-score[1])
            + 2.*(incumbent[2]-score[2]))
    if gain > 1. or (score[2] < incumbent[2]-tol and gain >= -tol):
        return True
    same_primary = all(abs(score[i]-incumbent[i]) <= tol for i in range(3))
    if score[0] <= incumbent[0]+tol and gain >= -tol and score[3] < incumbent[3]-.1:
        return True
    return same_primary and abs(score[3]-incumbent[3]) <= tol and score[4] < incumbent[4]-tol


def _classify_final_visibility(events, baseline, cutoff):
    """Classify freshly measured residuals without weakening repair guards.

    Visibility remains a soft priority: a rare new event may exceed the human
    cutoff by one onset-group occupancy quantum, provided it does not worsen
    an already bad original event. Bursts and over-rate hands still fail.
    """
    events = list(events)
    eligible = {hand: sum(e['hand'] == hand for e in events) for hand in (0, 1)}
    pending = {hand: [] for hand in (0, 1)}
    regressions, exceptions = [], []
    for event in events:
        original = baseline.get(event['id'], 0.)
        base = max(original, cutoff)
        if event['severity'] <= base+1e-9:
            continue
        ceiling = cutoff+1./event['onsets']
        if original > cutoff+1e-9 or event['severity'] > ceiling+1e-9:
            regressions.append(event['id'])
            continue
        pending[event['hand']].append(dict(
            id=event['id'], time=event['time'], hand=event['hand'], group_ids=event['group_ids'],
            original=original, base=base, severity=event['severity'], onsets=event['onsets'],
            ceiling=ceiling, cutoff=cutoff, policy_version=VISIBILITY_EXCEPTION_POLICY))
    budgets = {}
    for hand in (0, 1):
        maximum = math.floor(VISIBILITY_EXCEPTION_RATE*eligible[hand])
        accepted = pending[hand] if len(pending[hand]) <= maximum else []
        if not accepted:
            regressions.extend(e['id'] for e in pending[hand])
        exceptions.extend(accepted)
        budgets[str(hand)] = dict(eligible_events=eligible[hand], maximum_exceptions=maximum,
                                  actual_exceptions=len(accepted))
    exceptions.sort(key=lambda e: (e['time'], e['hand'], e['id']))
    return dict(visibility_exception_policy=VISIBILITY_EXCEPTION_POLICY,
                visibility_exceptions=exceptions, visibility_exception_budget_by_hand=budgets,
                visibility_regressions=regressions)


def audit_priority(notes):
    """Return the versioned, full-stream priority event audit."""
    return metrics.priority_metrics(notes)


def _clearance_findings(notes):
    """Identity-level counterpart to production clearance_violations."""
    from map_creation import v2_postprocessing as post
    eps = float(getattr(config, 'min_time_diff', .02))
    window = max(eps, float(getattr(config, 'v2_cleanup_swing_window', .15) or 0.))
    gap = float(getattr(config, 'v2_cleanup_same_dir_gap', .1) or 0.)
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    pred, incoming = post._history_context(ordered, eps)
    result = {'same_cell_pairs': set(), 'swing_into_pairs': set(), 'flow_repeats': set()}
    for first, second in post._simultaneous_pairs(ordered, eps):
        if (first['x'], first['y']) == (second['x'], second['y']):
            result['same_cell_pairs'].add((first['id'], second['id']))
    for first, second in post._simultaneous_pairs(ordered, window):
        if first['hand'] == second['hand']:
            continue
        pairs = [(first, second)]
        if second['time']-first['time'] <= eps:
            pairs.append((second, first))
        for swinger, target in pairs:
            direction = swinger['d'] if swinger['d'] != 8 else incoming.get(swinger['id'])
            swept = set(post._swept_cells(swinger['x'], swinger['y'], direction)) if direction is not None and direction != 8 else set()
            position = (target['x'], target['y'])
            if position in swept or (position == (swinger['x'], swinger['y']) and target['time']-swinger['time'] > eps):
                result['swing_into_pairs'].add((swinger['id'], target['id']))
    for n in ordered:
        if n['d'] != 8 and not post._flow_legal(n['d'], pred.get(n['id']), n['x'], n['y'], n['time'], gap):
            result['flow_repeats'].add(n['id'])
    return result


class _PrioritySearch:
    def __init__(self, notes, cap, protect_retained):
        self.notes = sorted(notes, key=lambda n: (n['time'], n['hand'], n['id']))
        self.index = {n['id']: i for i, n in enumerate(self.notes)}
        if len(self.index) != len(self.notes):
            raise ValueError('duplicate note identity')
        self.times = [n['time'] for n in self.notes]
        self.nps = note_span_nps(self.notes)
        self.state = _Search(self.notes, cap, self.nps, protect_retained)
        self.state.limit_diagonals = True
        self.paths = metrics.note_paths(self.notes)
        self.groups = {}
        self.location = {}
        for hand in (0, 1):
            groups = list(metrics._onset_groups(self.notes, hand))
            self.groups[hand] = groups
            for j, group in enumerate(groups):
                for n in group:
                    self.location[n['id']] = (hand, j)
        self.pairs = {}
        self.pair_index = defaultdict(set)
        for i, n in enumerate(self.notes):
            end = bisect_right(self.times, n['time'] + metrics.PATH_SECONDS + metrics.TIME_TOLERANCE)
            for other in self.notes[i+1:end]:
                if n['hand'] == other['hand']:
                    continue
                left, right = (n, other) if n['hand'] == 0 else (other, n)
                key = (left['id'], right['id'])
                self.pairs[key] = metrics.crossing_pair_event(left, right, self.paths[key[0]], self.paths[key[1]])
                self.pair_index[key[0]].add(key)
                self.pair_index[key[1]].add(key)
        self.hard_pairs = {k for k, e in self.pairs.items() if e['hard']}
        self.prep_hard = {e['id'] for e in metrics.preparation_events(self.notes) if e['hard']}
        self.candidates = 0
        self.hard_work = defaultdict(int)
        self.hard_budget_active = False
        self.hard_banned = set()
        self.hard_backtracks = 0
        self.baseline_speed = self.state.speed_groups.violations(range(len(self.notes)))
        # Preserve the actual baseline speed, not merely its violation identity.
        from map_creation.v2_postprocessing import calc_note_speed
        speed = self.state.speed_groups
        caps = {}
        self.requested_caps = {}
        for i, n in enumerate(self.notes):
            allowed = cap(n)
            self.requested_caps[n['id']] = allowed
            previous = speed.previous[i]
            if previous:
                arrival = min(calc_note_speed([self.notes[p][k] for k in ('x', 'y', 'hand', 'd')],
                            [n[k] for k in ('x', 'y', 'hand', 'd')], n['time']-self.notes[p]['time'])
                              for p in previous)
                allowed = max(allowed, arrival)
            caps[n['id']] = allowed
        self.effective_caps = caps
        self.state.speed_groups = SpeedGroups(self.notes, lambda n: caps[n['id']],
                                              float(getattr(config, 'min_time_diff', .02)))
        self._refresh_soft()
        self._refresh_visibility(protect_retained)

    def _refresh_visibility(self, protect_retained):
        from map_creation.v2_playability_visibility import visibility_costs
        self.visibility = visibility_costs(self.notes, self.nps)
        self.visibility_events = {e['id']: dict(e) for e in self.visibility['events']}
        baseline = ([n for n in self.notes if n.get('kind') == 'retained']
                    if protect_retained else self.notes)
        reference = visibility_costs(baseline, self.nps)
        by_member = {identity: e['severity'] for e in reference['events'] for identity in e['group_ids']}
        self.visibility_baseline = {anchor: max((by_member.get(identity, 0.) for identity in event['group_ids']), default=0.)
                                    for anchor, event in self.visibility_events.items()}
        self.visibility_dependents = defaultdict(list)
        for anchor, event in self.visibility_events.items():
            for identity, weight in zip(event['note_ids'], event['weights']):
                self.visibility_dependents[self.index[identity]].append((anchor, weight))
        self.visibility_work = defaultdict(int)

    def visibility_deltas(self, edits):
        """Exact fixed-window changes, including stacked-note fractional weights."""
        from map_creation.v2_playability_visibility import central_top
        changes = defaultdict(float)
        for i, (x, y, _) in edits.items():
            old = self.state.committed[i]
            change = int(central_top({'x': x, 'y': y})) - int(central_top({'x': old[0], 'y': old[1]}))
            if change:
                for anchor, weight in self.visibility_dependents[i]:
                    changes[anchor] += change*weight
        return {anchor: (self.visibility_events[anchor]['severity'],
                         self.visibility_events[anchor]['severity']+delta)
                for anchor, delta in changes.items() if abs(delta) > 1e-12}

    def visibility_allowed(self, edits, deltas=None):
        cutoff = self.visibility['cutoff']
        if deltas is None:
            deltas = self.visibility_deltas(edits)
        return all(new <= max(old, cutoff)+1e-9 for old, new in deltas.values())

    def visibility_regressions(self):
        cutoff = self.visibility['cutoff']
        return [anchor for anchor, event in self.visibility_events.items()
                if event['severity'] > max(self.visibility_baseline[anchor], cutoff)+1e-9]

    def _refresh_soft(self):
        """Cache accepted event identities and exact incoming exposure states."""
        self.prep_events = {e['id']: e for e in metrics.preparation_events(self.notes)}
        self.axis_events = {e['id']: e for e in metrics.axis_change_events(self.notes)}
        self.exposure = {}
        self.axis_change_groups = {}
        for hand in (0, 1):
            self.axis_change_groups[hand] = [j for j, group in enumerate(self.groups[hand])
                if any(self.axis_events[n['id']]['magnitude_degrees'] for n in group)]
            prior_time, exposure = None, 0.
            for offset, group in enumerate(self.groups[hand]):
                time = group[0]['time']
                gap = None if prior_time is None else time-prior_time
                load = 0. if gap is None else exposure*math.exp(-gap/metrics.EXPOSURE_TAU)
                self.exposure[hand, offset] = (load, gap)
                awkward = any(self.prep_events[n['id']]['eligible'] and
                    (self.prep_events[n['id']]['components']['excursion'] +
                     self.prep_events[n['id']]['components']['specific']) > 0 for n in group)
                exposure, prior_time = load + int(awkward), time

    def soft_deltas(self, edits, trial):
        """Exact changed event totals, with cached exposure and cadence context."""
        original = {i: (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d']) for i in edits}
        result = {'preparation': {}, 'axis': {}, 'crossing': {}}
        self.state.install(edits)
        try:
            for hand in (0, 1):
                changed = [self.location[self.notes[i]['id']][1] for i in edits if self.notes[i]['hand'] == hand]
                if not changed:
                    continue
                groups = self.groups[hand]
                first, last = min(changed), max(changed)
                load, gap = self.exposure[hand, first]
                prior = None
                for offset in range(first, len(groups)):
                    group = groups[offset]
                    time = group[0]['time']
                    if prior is not None:
                        gap = time-prior
                        load = next_exposure*math.exp(-gap/metrics.EXPOSURE_TAU)
                    standalone = metrics.preparation_events(group)
                    speed = 0. if gap is None else metrics.EXPOSURE_TAU/(metrics.EXPOSURE_TAU+gap)
                    awkward = False
                    for event in standalone:
                        raw = event['total']
                        value = raw*(1.+speed+load)
                        old = self.prep_events[event['id']]['total']
                        if abs(value-old) > 1e-11:
                            result['preparation'][event['id']] = (old, value)
                        awkward |= event['eligible'] and raw > 0
                    next_exposure = load + int(awkward)
                    prior = time
                    if offset >= last and offset+1 < len(groups):
                        next_gap = groups[offset+1][0]['time']-time
                        next_load = next_exposure*math.exp(-next_gap/metrics.EXPOSURE_TAU)
                        if abs(next_load-self.exposure[hand, offset+1][0]) <= 1e-12:
                            break
                changes = self.axis_change_groups[hand]
                previous_changes = changes[:bisect_left(changes, first)]
                following_changes = changes[bisect_right(changes, last):]
                start = max(0, previous_changes[-3]-1) if len(previous_changes) >= 3 else 0
                end = min(len(groups), max(last+2, following_changes[2]+1 if len(following_changes) >= 3 else
                                           following_changes[-1]+1 if following_changes else last+2))
                seq = [n for group in groups[start:end] for n in group]
                for event in metrics.axis_change_events(seq):
                    if self.location[event['id']][1] < first:
                        continue
                    old = self.axis_events[event['id']]['total']
                    if abs(event['total']-old) > 1e-11:
                        result['axis'][event['id']] = (old, event['total'])
            for key, event in trial[1].items():
                old = self.pairs[key]['total']
                if abs(old-event['total']) > 1e-11:
                    result['crossing'][key] = (old, event['total'])
        finally:
            self.state.install(original)
        return result

    def soft_allowed(self, edits, trial, deltas=None):
        """Do not create or worsen an individually severe priority event."""
        if not self.visibility_allowed(edits):
            return False
        bounds = _soft_bounds(self.nps)
        if deltas is None:
            deltas = self.soft_deltas(edits, trial)
        if not all(new <= max(old, bounds[name])+1e-9
                   for name, entries in deltas.items()
                   for old, new in entries.values()):
            return False

        def specific(note, event):
            components = event['components']
            raw = components['excursion'] + components['specific']
            factor = event['total']/raw if raw else 0.
            dx, dy = metrics.STEP.get(note['d'], (0, 0))
            inward = int((note['hand'] == 0 and note['x'] == 1 and dx == -1)
                         or (note['hand'] == 1 and note['x'] == 2 and dx == 1))
            vertical = int((note['y'] == 2 and dy == 1) or (note['y'] == 0 and dy == -1))
            return {'preparation_excursion': event['excursion']*factor,
                    'inner_inward': inward*factor, 'vertical_preparation': vertical*factor,
                    'outer_placement': components['placement']*factor}

        identities = set(deltas['preparation']) | {self.notes[i]['id'] for i in edits}
        for identity in identities:
            i = self.index[identity]
            old_note = self.notes[i]
            old_event = self.prep_events[identity]
            candidate = dict(old_note)
            if i in edits:
                candidate.update(zip(('x', 'y', 'd'), edits[i]))
            new_event = metrics.preparation_events([candidate])[0]
            new_event['total'] = deltas['preparation'].get(identity, (0., old_event['total']))[1]
            old_specific, new_specific = specific(old_note, old_event), specific(candidate, new_event)
            if any(new_specific[key] > max(old_specific[key], bounds[key])+1e-9 for key in old_specific):
                return False
        return True

    def _affected_paths(self, changed):
        targets = set()
        for i in changed:
            n = self.notes[i]
            hand, offset = self.location[n['id']]
            groups = self.groups[hand]
            targets.update(x['id'] for x in groups[offset])
            for group in groups[offset+1:]:
                targets.update(x['id'] for x in group)
                # A dot anchor depends on prior effective direction.
                if group[-1]['d'] != 8:
                    break
        updated = {}
        for hand in (0, 1):
            indices = [self.location[k][1] for k in targets if self.location[k][0] == hand]
            if not indices:
                continue
            lo, hi = min(indices), max(indices)
            groups = self.groups[hand]
            start = max(0, lo-1)
            while start > 0 and groups[start][-1]['d'] == 8:
                start -= 1
            fragment = [n for group in groups[start:hi+1] for n in group]
            computed = metrics.note_paths(fragment)
            updated.update({k: computed[k] for k in targets if k in computed})
        return updated

    def _flow_valid(self, changed):
        # _history_context is the production authority. Its stack handling is
        # not equivalent to the old cached search's dot propagation.
        from map_creation import v2_postprocessing as post
        eps = float(getattr(config, 'min_time_diff', .02))
        gap = float(getattr(config, 'v2_cleanup_same_dir_gap', .1) or 0.)
        for hand in (0, 1):
            offsets = [self.location[self.notes[i]['id']][1] for i in changed
                       if self.notes[i]['hand'] == hand]
            if not offsets:
                continue
            groups = self.groups[hand]
            lo, hi = min(offsets), max(offsets)
            while hi+1 < len(groups):
                hi += 1
                if groups[hi][-1]['d'] != 8:
                    break
            start = max(0, lo-1)
            while start > 0 and groups[start][-1]['d'] == 8:
                start -= 1
            seq = [n for group in groups[start:hi+1] for n in group]
            pred, _ = post._history_context(seq, eps)
            for group in groups[lo:hi+1]:
                for n in group:
                    if n['d'] != 8 and not post._flow_legal(n['d'], pred.get(n['id']), n['x'], n['y'], n['time'], gap):
                        return False
        return True

    def _hard_work(self, edits):
        if not self.hard_budget_active:
            return
        buckets = {math.floor(self.notes[i]['time'] / HARD_CLUSTER_SECONDS) for i in edits}
        for bucket in buckets:
            self.hard_work[bucket] += 1
            if self.hard_work[bucket] > HARD_CLUSTER_BUDGET:
                raise ValueError(f'priority hard budget exhausted in time bucket {bucket}: '
                                 f'{self.hard_work[bucket]} work units')

    def trial(self, edits, *, hard_only=False):
        if hard_only:
            self._hard_work(edits)
            fingerprint = tuple(sorted((i, tuple(v)) for i, v in edits.items()
                if tuple(v) != (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d'])))
            if self.hard_budget_active and fingerprint in self.hard_banned:
                return None
        self.candidates += 1
        original = {i: (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d']) for i in edits}
        self.state.install(edits)
        try:
            if not self.state.legal(edits) or not self._flow_valid(edits):
                return None
            paths = self._affected_paths(edits)
            keys = set().union(*(self.pair_index[k] for k in paths)) if paths else set()
            if hard_only:
                keys = {k for k in keys if self.pairs[k]['dt'] <= metrics.FAST_SECONDS + metrics.TIME_TOLERANCE}
            new_events = {}
            for key in keys:
                left, right = (self.notes[self.index[k]] for k in key)
                new_events[key] = metrics.crossing_pair_event(left, right,
                    paths.get(key[0], self.paths[key[0]]), paths.get(key[1], self.paths[key[1]]))
            hard = (self.hard_pairs - keys) | {k for k, e in new_events.items() if e['hard']}
            prep = (self.prep_hard - {self.notes[i]['id'] for i in edits}) | {
                e['id'] for e in metrics.preparation_events([self.notes[i] for i in edits]) if e['hard']}
            return paths, new_events, hard, prep
        finally:
            self.state.install(original)

    def _commit_cells(self, edits):
        """Keep rolling visibility exact for accepted moves and hard rollbacks."""
        visibility = self.visibility_deltas(edits)
        self.state.commit(edits)
        for anchor, (_, value) in visibility.items():
            self.visibility_events[anchor]['severity'] = value

    def commit(self, edits, trial):
        paths, events, self.hard_pairs, self.prep_hard = trial
        self._commit_cells(edits)
        self.paths.update(paths)
        keys = set().union(*(self.pair_index[k] for k in paths)) if paths else set()
        for key in keys - events.keys():
            left, right = (self.notes[self.index[k]] for k in key)
            events[key] = metrics.crossing_pair_event(left, right, self.paths[key[0]], self.paths[key[1]])
        self.pairs.update(events)
        if not self.hard_budget_active:
            self._refresh_soft()

    def separate_repeated_positions(self, rng=None):
        """One guarded cut-direction step for exact same-hand position repeats."""
        import random
        rng = random if rng is None else rng
        previous = {}
        changed = 0
        for i, note in enumerate(self.notes):
            prior = previous.get(note['hand'])
            previous[note['hand']] = note
            hand, group = self.location[note['id']]
            if (prior is None or len(self.groups[hand][group]) != 1
                    or i not in self.state.mutable or note['d'] not in metrics.STEP
                    or note['time']-prior['time'] <= self.state.eps
                    or (note['x'], note['y']) != (prior['x'], prior['y'])):
                continue
            # STEP points to cut preparation; motion follows its opposite.
            dx, dy = metrics.STEP[note['d']]
            x = note['x'] if dy else note['x']-dx
            y = note['y']-dy
            low, high = (0, 2) if note['hand'] == 0 else (1, 3)
            if not (low <= x <= high and 0 <= y <= 2):
                continue
            edits = {i: (x, y, note['d'])}
            trial = self.trial(edits)
            if (trial is None or trial[2] or trial[3]
                    or not self.soft_allowed(edits, trial)):
                continue
            if y == 2 and rng.random() >= .5:
                continue
            self.commit(edits, trial)
            changed += 1
        return changed

    def _soft_score(self, fragment, pair_keys, updates=None):
        from map_creation.v2_playability_placement import local_placement_cost
        from map_creation.v2_playability_costs import occlusion_severity, calibrated_limits
        from map_creation.v2_playability_reach import _travel, calibrated_cutoff
        prep = sum(e['total'] for e in metrics.preparation_events(fragment))
        axis = sum(e['total'] for e in metrics.axis_change_events(fragment))
        updates = updates or {}
        near = sum(updates.get(k, self.pairs[k])['total'] for k in pair_keys)
        placement = local_placement_cost(fragment, self.nps)
        cutoff = calibrated_limits(self.nps)['m2']
        m2 = travel = 0.
        previous = {}
        for i, n in enumerate(fragment):
            prior = previous.get(n['hand'])
            if prior and n['time']-prior['time'] > .02 and 8 not in (n['d'], prior['d']):
                travel += max(0., _travel(prior, n, n['time']-prior['time'])
                              - calibrated_cutoff('recovery_adjusted_travel', self.nps))
            previous[n['hand']] = n
            for before in reversed(fragment[:i]):
                dt = n['time']-before['time']
                if dt > .4:
                    break
                if dt > .02 and abs(n['x']-before['x']) <= 1:
                    m2 += max(0., occlusion_severity(n, before, dt)-cutoff) / max(1., cutoff)
                    break
        return prep, axis, near, placement + m2, travel

    def _repair_reach_runs(self, target_ids=None):
        """Recenter short repeated outer-lane phrases with coherent cuts."""
        from itertools import islice, product
        changed = 0
        runs = []
        for hand in (0, 1):
            current = []
            for group in self.groups[hand]:
                n = group[0]
                eligible = (len(group) == 1 and n['d'] != 8 and self.index[n['id']] in self.state.mutable
                            and n['x'] == (0 if hand == 0 else 3))
                if eligible and current and n['time']-self.notes[current[-1]]['time'] > .6:
                    if len(current) >= 3:
                        runs.append((hand, current))
                    current = []
                if eligible:
                    current.append(self.index[n['id']])
                else:
                    if len(current) >= 3:
                        runs.append((hand, current))
                    current = []
            if len(current) >= 3:
                runs.append((hand, current))
        for hand, run in runs:
            if target_ids is not None and not any(self.notes[i]['id'] in target_ids for i in run):
                continue
            # Separate long passages into bounded local windows; every window
            # keeps the outside neighbors as fixed flow and speed constraints.
            for offset in range(0, len(run), 6):
                indices = run[offset:offset+6]
                if len(indices) < 3:
                    continue
                original = {i: (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d']) for i in indices}
                lo = bisect_left(self.times, self.times[indices[0]]-4.)
                hi = bisect_right(self.times, self.times[indices[-1]]+4.)
                fragment = self.notes[lo:hi]
                ids = {n['id'] for n in fragment}
                keys = {key for n in fragment for key in self.pair_index[n['id']]
                        if key[0] in ids or key[1] in ids}
                baseline = self._soft_score(fragment, keys)
                incumbent, best = baseline, None
                patterns = [tuple(original[i][2] for i in indices)]
                for pair in ((0, 1), (1, 0), (4, 7), (7, 4), (5, 6), (6, 5)):
                    patterns.append(tuple(pair[k % 2] for k in range(len(indices))))
                rows = [tuple(dict.fromkeys((original[i][1], 1, 0, 2))) for i in indices]
                for directions in dict.fromkeys(patterns):
                    for ys in islice(product(*rows), 128):
                        edits = {i: (1 if hand == 0 else 2, y, d)
                                 for i, y, d in zip(indices, ys, directions)}
                        trial = self.trial(edits)
                        if trial is None or trial[2] or trial[3]:
                            continue
                        deltas = self.soft_deltas(edits, trial)
                        if (sum(new-old for old, new in deltas['preparation'].values()) >= -1e-8
                                or not self.soft_allowed(edits, trial, deltas)):
                            continue
                        self.state.install(edits)
                        try:
                            local = self._soft_score(fragment, keys, trial[1])
                            score = tuple(baseline[k] + sum(new-old for old, new in deltas[name].values())
                                          for k, name in enumerate(('preparation', 'axis', 'crossing'))) + local[3:]
                        finally:
                            self.state.install(original)
                        if _soft_improves(score, incumbent, edit_count=sum(edits[i] != original[i] for i in edits)):
                            incumbent, best = score, (edits, trial)
                if best is not None:
                    self.commit(*best)
                    changed += sum(best[0][i] != original[i] for i in indices)
        return changed

    def _repair_joint_reach(self, target_ids):
        """Bounded adjacent same-hand cell and arrow repair for awkward reach."""
        from itertools import islice, product
        changed = 0
        for identity in sorted(target_ids, key=lambda k: (self.notes[self.index[k]]['time'], str(k)), reverse=True):
            i = self.index[identity]
            if i not in self.state.mutable or self.notes[i]['d'] == 8:
                continue
            note = self.notes[i]
            neighbors = [j for j in (self.state.previous.get(i), self.state.following.get(i))
                         if j is not None and j in self.state.mutable and self.notes[j]['d'] != 8
                         and self.state.eps < abs(self.times[j]-self.times[i]) <= .6]
            for j in neighbors:
                other = self.notes[j]
                original = {k: (self.notes[k]['x'], self.notes[k]['y'], self.notes[k]['d']) for k in (i, j)}
                # Only neighboring inner columns are proposed for the target;
                # the companion can retain its column or use the vacated one.
                first = [o for o in self.state.options(i, broad=True)
                         if o[0] in (1, 2) and abs(o[0]-note['x']) == 1
                         and abs(o[1]-note['y']) <= 1]
                second = [o for o in self.state.options(j, broad=True)
                          if o[0] in (other['x'], note['x']) and abs(o[1]-other['y']) <= 1]
                if not first or not second:
                    continue
                lo = bisect_left(self.times, min(self.times[i], self.times[j])-4.)
                hi = bisect_right(self.times, max(self.times[i], self.times[j])+4.)
                fragment = self.notes[lo:hi]
                ids = {n['id'] for n in fragment}
                keys = {key for n in fragment for key in self.pair_index[n['id']]
                        if key[0] in ids or key[1] in ids}
                baseline = self._soft_score(fragment, keys)
                incumbent = baseline
                best = None
                for a, b in islice(product(first, second), 384):
                    edits = {i: a, j: b}
                    if b == original[j]:
                        continue
                    trial = self.trial(edits)
                    if trial is None or trial[2] or trial[3]:
                        continue
                    deltas = self.soft_deltas(edits, trial)
                    if (sum(new-old for old, new in deltas['preparation'].values()) >= -1e-8
                            or not self.soft_allowed(edits, trial, deltas)):
                        continue
                    self.state.install(edits)
                    try:
                        local = self._soft_score(fragment, keys, trial[1])
                        score = tuple(baseline[k] + sum(new-old for old, new in deltas[name].values())
                                      for k, name in enumerate(('preparation', 'axis', 'crossing'))) + local[3:]
                    finally:
                        self.state.install(original)
                    if _soft_improves(score, incumbent, edit_count=sum(edits[k] != original[k] for k in edits)):
                        incumbent, best = score, (edits, trial)
                if best is not None:
                    self.commit(*best)
                    changed += sum(best[0][k] != original[k] for k in best[0])
                    break
        return changed

    def repair_soft(self):
        """Refine flagged local exposures; all accepted moves retain hard legality."""
        prep = {e['id'] for e in metrics.preparation_events(self.notes) if e['total'] >= 2.}
        axis = {e['id'] for e in metrics.axis_change_events(self.notes)
                if e['components']['frequency'] + e['components']['irregularity'] >= 1.}
        for key, event in self.pairs.items():
            if event['total'] > 1e-9:
                prep.update(key)
                prep.update(v for v in event['predecessor_ids'] if v is not None)
        from map_creation.v2_playability_placement import placement_costs
        placement = placement_costs(self.notes, self.nps)
        for m in placement['metrics'].values():
            for severity, identities in zip(m['severities'], m['event_note_ids']):
                if severity > m['cutoff'] + 1e-9:
                    prep.update(identities)
        from map_creation.v2_playability_costs import occlusion_severity, calibrated_limits
        cutoff = calibrated_limits(self.nps)['m2']
        for i, n in enumerate(self.notes):
            for j in range(i-1, -1, -1):
                before = self.notes[j]
                dt = n['time']-before['time']
                if dt > .4:
                    break
                if dt > .02 and abs(n['x']-before['x']) <= 1:
                    if n['hand'] != before['hand'] and occlusion_severity(n, before, dt) > cutoff + 1e-9:
                        prep.update((n['id'], before['id']))
                    break
        run_changed = self._repair_reach_runs()
        reach_joint_changed = 0
        for _ in range(2):
            reach_targets = {e['id'] for e in self.prep_events.values()
                             if e['total'] >= 2. and (e['components']['excursion'] > 0
                                or e['components']['specific'] > 0)}
            repaired = self._repair_joint_reach(reach_targets)
            reach_joint_changed += repaired
            if not repaired:
                break
        changed = run_changed + reach_joint_changed
        for i in reversed(range(len(self.notes))):
            if i not in self.state.mutable or self.notes[i]['id'] not in prep | axis:
                continue
            t = self.notes[i]['time']
            lo = bisect_left(self.times, t-4.)
            hi = bisect_right(self.times, t+4.)
            fragment = self.notes[lo:hi]
            fragment_ids = {n['id'] for n in fragment}
            pair_keys = {key for n in fragment for key in self.pair_index[n['id']]
                         if key[0] in fragment_ids or key[1] in fragment_ids}
            incumbent = self._soft_score(fragment, pair_keys)
            baseline_score = incumbent
            original = (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d'])
            best = None
            for option in self.state.options(i)[:32]:
                if option == original:
                    continue
                edits = {i: option}
                trial = self.trial(edits)
                if trial is None or trial[2] or trial[3]:
                    continue
                deltas = self.soft_deltas(edits, trial)
                if not self.soft_allowed(edits, trial, deltas):
                    continue
                self.state.install(edits)
                try:
                    local_score = self._soft_score(fragment, pair_keys, trial[1])
                    score = tuple(baseline_score[k] + sum(new-old for old, new in deltas[name].values())
                                  for k, name in enumerate(('preparation', 'axis', 'crossing'))) + local_score[3:]
                finally:
                    self.state.install({i: original})
                if _soft_improves(score, incumbent):
                    incumbent, best = score, (edits, trial)
            if best:
                self.commit(*best)
                changed += 1
        joint_changed = 0
        # Stabilize coupled adjacent arrows when neither single recut can
        # preserve the surrounding flow.  Cells stay fixed in this pass.
        for i in reversed(range(len(self.notes))):
            n = self.notes[i]
            if n['id'] not in axis or i not in self.state.mutable or n['d'] == 8:
                continue
            j = self.state.previous.get(i)
            if j is None or j not in self.state.mutable or self.notes[j]['d'] == 8:
                continue
            t = n['time']
            lo, hi = bisect_left(self.times, t-4.), bisect_right(self.times, t+4.)
            fragment = self.notes[lo:hi]
            ids = {v['id'] for v in fragment}
            keys = {key for v in fragment for key in self.pair_index[v['id']]
                    if key[0] in ids or key[1] in ids}
            incumbent = self._soft_score(fragment, keys)
            baseline_score = incumbent
            original = {k: (self.notes[k]['x'], self.notes[k]['y'], self.notes[k]['d']) for k in (i, j)}
            best = None
            for first in range(8):
                for second in range(8):
                    edits = {i: (*original[i][:2], first), j: (*original[j][:2], second)}
                    if all(edits[k] == original[k] for k in edits):
                        continue
                    trial = self.trial(edits)
                    if trial is None or trial[2] or trial[3]:
                        continue
                    deltas = self.soft_deltas(edits, trial)
                    if not self.soft_allowed(edits, trial, deltas):
                        continue
                    self.state.install(edits)
                    try:
                        local_score = self._soft_score(fragment, keys, trial[1])
                        score = tuple(baseline_score[k] + sum(new-old for old, new in deltas[name].values())
                                      for k, name in enumerate(('preparation', 'axis', 'crossing'))) + local_score[3:]
                    finally:
                        self.state.install(original)
                    if score[1] < incumbent[1]-1e-8 and _soft_improves(score, incumbent, edit_count=sum(edits[k] != original[k] for k in edits)):
                        incumbent, best = score, (edits, trial)
            if best:
                self.commit(*best)
                count = sum(best[0][k] != original[k] for k in best[0])
                changed += count
                joint_changed += count
        return changed, joint_changed + reach_joint_changed + run_changed

    def repair_visibility(self):
        """Reduce local obstruction with bounded joint cell/arrow edits.

        Every affected rolling phrase is guarded, including neighboring phrases
        beyond the target. The primary per-event preparation, cadence and
        crossing tail guards remain active. Isolated top notes have no reward.
        """
        from itertools import islice, product
        from map_creation.v2_playability_visibility import central_top
        cutoff = self.visibility['cutoff']
        changed = 0
        buckets = sorted({math.floor(e['time']/HARD_CLUSTER_SECONDS) for e in self.visibility_events.values()
                          if e['severity'] > cutoff+1e-9})
        for bucket in buckets:
            deferred = set()
            for _ in range(VISIBILITY_MAX_PASSES):
                flagged = [e for anchor, e in self.visibility_events.items()
                           if anchor not in deferred and math.floor(e['time']/HARD_CLUSTER_SECONDS) == bucket
                           and e['severity'] > cutoff+1e-9
                           and any(self.index[k] in self.state.mutable and central_top(self.notes[self.index[k]])
                                   for k in e['note_ids'])]
                if not flagged:
                    break
                event = max(flagged, key=lambda e: (
                    e['severity']-max(cutoff, self.visibility_baseline[e['id']]),
                    e['severity']-cutoff, -e['time']))
                targets = [k for k in event['note_ids'] if central_top(self.notes[self.index[k]])]
                best = None

                def examine(edits):
                    nonlocal best
                    work_buckets = {math.floor(self.times[i]/HARD_CLUSTER_SECONDS) for i in edits}
                    if any(self.visibility_work[b] >= VISIBILITY_CLUSTER_BUDGET for b in work_buckets):
                        return
                    for work_bucket in work_buckets:
                        self.visibility_work[work_bucket] += 1
                    visibility = self.visibility_deltas(edits)
                    if not self.visibility_allowed(edits, visibility):
                        return
                    gain = sum(max(0., old-cutoff)-max(0., new-cutoff) for old, new in visibility.values())
                    if gain <= 1e-9:
                        return
                    trial = self.trial(edits)
                    if trial is None or trial[2] or trial[3]:
                        return
                    deltas = self.soft_deltas(edits, trial)
                    if not self.soft_allowed(edits, trial, deltas):
                        return
                    primary = {key: sum(new-old for old, new in values.values())
                               for key, values in deltas.items()}
                    count = sum(option != self.state.committed[i] for i, option in edits.items())
                    distance = sum(abs(option[0]-self.notes[i]['x'])+abs(option[1]-self.notes[i]['y'])
                                   for i, option in edits.items())
                    # Primary weights select among equally useful visibility
                    # repairs, and every individually severe event stays guarded.
                    rank = (-gain, primary['preparation']+5.*primary['axis']+2.*primary['crossing'], count, distance,
                            tuple(sorted(edits.items())))
                    if best is None or rank < best[0]:
                        best = rank, dict(edits), trial

                for identity in sorted(targets, key=lambda k: (self.times[self.index[k]], str(k))):
                    i = self.index[identity]
                    if i not in self.state.mutable:
                        continue
                    options = [o for o in self.state.options(i, broad=True)
                               if not central_top({'x': o[0], 'y': o[1]})][:24]
                    for option in options:
                        examine({i: option})
                    neighbors = [j for j in range(bisect_left(self.times, self.times[i]-.6),
                                                 bisect_right(self.times, self.times[i]+.6))
                                 if j != i and j in self.state.mutable]
                    neighbors.sort(key=lambda j: (abs(self.times[j]-self.times[i]), j))
                    for j in neighbors[:12]:
                        for first, second in islice(product(options, self.state.options(j, broad=True)[:32]), 384):
                            examine({i: first, j: second})
                if best is None:
                    deferred.add(event['id'])
                    continue
                _, edits, trial = best
                count = sum(option != self.state.committed[i] for i, option in edits.items())
                self.commit(edits, trial)
                changed += count
                deferred.clear()
        return changed

    def _sequence_escape(self, target, *, hand, radius, allow_bridge=False):
        """Search a small same-hand phrase against fixed opposite-hand paths.

        Only complete candidates are accepted by the normal full-boundary trial.
        The beam retains up to three histories per ending cut; it is not a proof
        of infeasibility when no path is found.
        """
        from map_creation import v2_postprocessing as post
        if target not in self.hard_pairs:
            return None
        seq = [i for i, n in enumerate(self.notes) if n['hand'] == hand]
        identity = next((k for k in target if self.notes[self.index[k]]['hand'] == hand), None)
        if identity is None:
            return None
        centre = seq.index(self.index[identity])
        lo, hi = max(1, centre-radius), min(len(seq)-1, centre+radius+1)
        selected = seq[lo:hi]
        if not selected or any(i not in self.state.mutable for i in selected):
            return None
        context = [seq[lo-1], *selected, seq[hi]]
        if any(self.notes[i]['d'] == 8 for i in context):
            return None
        if any(self.times[b]-self.times[a] <= self.state.eps+1e-9 for a, b in zip(context, context[1:])):
            return None
        base_hard = set(self.hard_pairs)
        base_prep = set(self.prep_hard)
        work = 0

        def permitted(i, option):
            x, y, direction = option
            n = self.notes[i]
            for j in self.state.near[i]:
                if j == i:
                    continue
                other = self.notes[j]
                dt = other['time']-n['time']
                same = (x, y) == (other['x'], other['y'])
                if abs(dt) <= self.state.eps and same:
                    return False
                if other['hand'] == hand:
                    continue
                if -self.state.eps <= dt <= self.state.swing:
                    if (other['x'], other['y']) in post._swept_cells(x, y, direction) or (same and dt > self.state.eps):
                        return False
                if -self.state.swing <= dt <= self.state.eps:
                    if (x, y) in post._swept_cells(other['x'], other['y'], self.state.effective[j]) or (same and -dt > self.state.eps):
                        return False
            return True

        def path_safe(i, previous, option):
            n = dict(self.notes[i])
            n.update(zip(('x', 'y', 'd'), option))
            path = metrics.note_paths([previous, n])[n['id']]
            for key in self.pair_index[n['id']]:
                if self.pairs[key]['dt'] > metrics.FAST_SECONDS + metrics.TIME_TOLERANCE:
                    continue
                left, right = (self.notes[self.index[k]] for k in key)
                if hand == 0:
                    left = n
                else:
                    right = n
                event = metrics.crossing_pair_event(left, right,
                    path if hand == 0 else self.paths[key[0]],
                    path if hand == 1 else self.paths[key[1]])
                if event['hard'] and (key not in base_hard or key == target):
                    return False
            return True

        # (edits, movement, assignment, previous note); candidates are ordered
        # deterministically and never mutate the live stream during expansion.
        states = [(0, 0, {}, self.notes[seq[lo-1]])]
        for i in selected:
            n = self.notes[i]
            original = (n['x'], n['y'], n['d'])
            options = []
            for option in self.state.options(i, broad=True):
                work += 1
                self._hard_work({i: option})
                if work > 30000:
                    return None
                if permitted(i, option):
                    options.append(option)
            candidates = []
            for count, distance, edits, previous in states:
                for option in options:
                    work += 1
                    self._hard_work({i: option})
                    if work > 30000:
                        return None
                    x, y, direction = option
                    prior = (previous['time'], previous['x'], previous['y'], previous['d'])
                    if not post._flow_legal(direction, prior, x, y, n['time'], self.state.gap):
                        continue
                    speed = post.calc_note_speed([previous[k] for k in ('x','y','hand','d')],
                        [x, y, hand, direction], n['time']-previous['time'])
                    if speed > self.state.speed_groups.cap(n)+1e-9 or not path_safe(i, previous, option):
                        continue
                    updated = dict(n)
                    updated.update(zip(('x', 'y', 'd'), option))
                    assignment = dict(edits)
                    assignment[i] = option
                    change = int(option != original)
                    candidates.append((count+change, distance+abs(x-original[0])+abs(y-original[1])
                                       +int(direction != original[2]), assignment, updated))
            candidates.sort(key=lambda item: (item[0], item[1], tuple(item[2].values())))
            retained, counts = [], defaultdict(int)
            for item in candidates:
                last = item[3]
                key = (last['x'], last['y'], last['d'])
                if counts[key] >= 3:
                    continue
                counts[key] += 1
                retained.append(item)
                if len(retained) >= 128:
                    break
            states = retained
            if not states:
                return None
        for _, _, edits, _ in states:
            trial = self.trial(edits, hard_only=True)
            if trial is None or target in trial[2] or not trial[3] <= base_prep:
                continue
            if trial[2] <= base_hard-{target}:
                return edits, trial
            if allow_bridge and len(trial[2]) == len(base_hard) and trial[2]-base_hard:
                return edits, trial
        return None

    def _sequence_bridge(self, target):
        """Explore a same-count bridge on a private copy, then validate atomically."""
        from copy import deepcopy
        base_hard, base_prep = set(self.hard_pairs), set(self.prep_hard)
        for radius in (1, 2, 3, 4):
            for hand in (0, 1):
                first = self._sequence_escape(target, hand=hand, radius=radius, allow_bridge=True)
                if first is None or len(first[1][2]) < len(base_hard):
                    continue
                new_pairs = first[1][2]-base_hard
                if not new_pairs:
                    continue
                clone = _PrioritySearch(deepcopy(self.notes), lambda n: self.effective_caps[n['id']], False)
                clone.state.diagonal_limit = self.state.diagonal_limit
                clone.state.mutable = set(self.state.mutable)
                clone.hard_work = self.hard_work
                clone.hard_budget_active = self.hard_budget_active
                clone.commit(*first)
                for new_target in sorted(new_pairs):
                    for next_radius in (1, 2, 3, 4):
                        for next_hand in (1-hand, hand):
                            second = clone._sequence_escape(new_target, hand=next_hand, radius=next_radius)
                            if second is None:
                                continue
                            combined = dict(first[0])
                            combined.update(second[0])
                            trial = self.trial(combined, hard_only=True)
                            if (trial is not None and trial[2] <= base_hard-{target}
                                    and trial[3] <= base_prep):
                                return combined, trial
        return None

    def repair_hard(self):
        # Only accept a reduction in the finite hard-event count. Every
        # candidate is evaluated locally, including affected successor paths.
        failed = set()
        history = []
        self.hard_budget_active = True
        while self.prep_hard or self.hard_pairs:
            hard_before = len(self.prep_hard) + len(self.hard_pairs)
            participants = set(self.prep_hard)
            for left, right in self.hard_pairs:
                participants.update((left, right))
                participants.update(k for k in (self.paths[left]['predecessor_id'], self.paths[right]['predecessor_id']) if k is not None)
            ordered = sorted((self.index[k] for k in participants if self.index[k] in self.state.mutable and self.index[k] not in failed), reverse=True)
            best = None
            for i in ordered:
                current = (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d'])
                for option in self.state.options(i, broad=True):
                    if option == current:
                        continue
                    edits = {i: option}
                    trial = self.trial(edits, hard_only=True)
                    if trial is None:
                        continue
                    after = len(trial[2]) + len(trial[3])
                    if after < hard_before:
                        # Incumbent-first, local improvement with deterministic ordering.
                        best = edits, trial
                        break
                if best:
                    break
                failed.add(i)
            if best is None:
                # Reopen a pair around the most recent unresolved crossing.
                # Intermediate states need not be legal; the joint assignment must be.
                candidate_pairs = []
                for identity in sorted(self.prep_hard if not self.hard_pairs else (), key=lambda v: self.notes[self.index[v]]['time'], reverse=True):
                    i = self.index[identity]
                    neighbors = set(self.state.near[i])
                    neighbors.update(j for j in (self.state.previous.get(i), self.state.following.get(i)) if j is not None)
                    candidate_pairs.extend((i, j) for j in sorted(neighbors, key=lambda j: (abs(self.times[j]-self.times[i]), j)) if j != i)
                for key in sorted(self.hard_pairs, key=lambda k: max(self.notes[self.index[v]]['time'] for v in k), reverse=True)[:8]:
                    first, second = (self.index[v] for v in key)
                    candidate_pairs.append((first, second))
                    for i in (first, second):
                        for j in (self.state.previous.get(i), self.state.following.get(i)):
                            if j is not None and j != i:
                                candidate_pairs.append((i, j))
                    # Moving one endpoint can expose the other hand's next
                    # cut after its immediate neighbor. Check that bounded
                    # second-neighbor frontier atomically before wider search.
                    for endpoint, other in ((first, second), (second, first)):
                        for adjacent in (self.state.previous, self.state.following):
                            neighbor = adjacent.get(endpoint)
                            partner = adjacent.get(neighbor) if neighbor is not None else None
                            if partner is not None and partner != other:
                                candidate_pairs.append((other, partner))
                    first_pred = self.paths[key[0]]['predecessor_id']
                    second_pred = self.paths[key[1]]['predecessor_id']
                    if (first not in self.state.mutable or second not in self.state.mutable) and first_pred is not None and second_pred is not None:
                        candidate_pairs.append((self.index[first_pred], self.index[second_pred]))
                for first, second in dict.fromkeys(candidate_pairs):
                    if first not in self.state.mutable or second not in self.state.mutable:
                        continue
                    first_options = self.state.options(first, broad=True)
                    second_options = self.state.options(second, broad=True)
                    from itertools import islice, product
                    for left, right in islice(product(first_options, second_options), 2048):
                        edits = {first: left, second: right}
                        trial = self.trial(edits, hard_only=True)
                        if trial is not None and len(trial[2])+len(trial[3]) < hard_before:
                            best = edits, trial
                            break
                    if best:
                        break
            if best is None:
                for key in sorted(self.hard_pairs, key=lambda k: max(self.notes[self.index[v]]['time'] for v in k), reverse=True)[:3]:
                    for radius in (1, 2, 3, 4):
                        for hand in (0, 1):
                            best = self._sequence_escape(key, hand=hand, radius=radius)
                            if best is not None:
                                break
                        if best is not None:
                            break
                    if best is not None:
                        break
            if best is None:
                for key in sorted(self.hard_pairs, key=lambda k: max(self.notes[self.index[v]]['time'] for v in k), reverse=True)[:3]:
                    best = self._sequence_bridge(key)
                    if best is not None:
                        break
            if best is None:
                # Bounded three-note escape for a crossing coupled to a
                # same-hand incoming/outgoing edge. Only complete triples are
                # checked; intermediate assignments may be physically invalid.
                for key in sorted(self.hard_pairs, key=lambda k: max(self.notes[self.index[v]]['time'] for v in k), reverse=True)[:4]:
                    first, second = (self.index[v] for v in key)
                    if first not in self.state.mutable or second not in self.state.mutable:
                        continue
                    neighbors = []
                    for i in (first, second):
                        neighbors.extend((self.state.previous.get(i), self.state.following.get(i)))
                    for third in dict.fromkeys(neighbors):
                        if third is None or third in (first, second) or third not in self.state.mutable:
                            continue
                        for one in self.state.options(first, broad=True)[:40]:
                            for two in self.state.options(second, broad=True)[:40]:
                                for three in self.state.options(third, broad=True)[:40]:
                                    edits = {first: one, second: two, third: three}
                                    trial = self.trial(edits, hard_only=True)
                                    if trial is not None and len(trial[2])+len(trial[3]) < hard_before:
                                        best = edits, trial
                                        break
                                if best:
                                    break
                            if best:
                                break
                        if best:
                            break
                    if best:
                        break
            if best is None:
                # Pair each hand with an adjacent same-hand note, prune on
                # real same-hand flow/speed, then test cross-hand combinations.
                from itertools import product
                sequence_pairs = list(sorted(self.hard_pairs, key=lambda k: max(self.notes[self.index[v]]['time'] for v in k), reverse=True)[:2])
                for identity in sorted(self.prep_hard, key=lambda v: self.notes[self.index[v]]['time'], reverse=True):
                    i = self.index[identity]
                    neighbors = [j for j in self.state.near[i] if self.notes[j]['hand'] != self.notes[i]['hand']]
                    neighbors.sort(key=lambda j: (abs(self.times[j]-self.times[i]), j))
                    sequence_pairs.extend((identity, self.notes[j]['id']) for j in neighbors[:2])
                for key in sequence_pairs:
                    first, second = (self.index[v] for v in key)
                    for edge in ('previous', 'following'):
                        adjacent = getattr(self.state, edge)
                        third, fourth = adjacent.get(first), adjacent.get(second)
                        indices = (first, second, third, fourth)
                        if None in indices or len(set(indices)) != 4 or not set(indices) <= self.state.mutable:
                            continue
                        alternatives = []
                        for pair in ((first, third), (second, fourth)):
                            original = {i: (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d']) for i in pair}
                            feasible = []
                            for values in product(*(self.state.options(i, broad=True) for i in pair)):
                                edits = dict(zip(pair, values))
                                self._hard_work(edits)
                                self.state.install(edits)
                                try:
                                    if self.state.speed_groups.legal(edits) and self._flow_valid(edits):
                                        feasible.append(edits)
                                finally:
                                    self.state.install(original)
                                if len(feasible) >= 128:
                                    break
                            alternatives.append(feasible)
                        for left, right in product(*alternatives):
                            edits = {**left, **right}
                            trial = self.trial(edits, hard_only=True)
                            if trial is not None and len(trial[2])+len(trial[3]) < hard_before:
                                best = edits, trial
                                break
                        if best:
                            break
                    if best:
                        break
            if best is None:
                times = [self.notes[self.index[k]]['time'] for k in participants]
                offset = next((j for j in range(len(history)-1, -1, -1)
                               if any(abs(self.times[i]-t) <= 1.5 for i in history[j][0] for t in times)), None)
                if offset is None:
                    raise ValueError(f'priority hard search exhausted: {hard_before} findings, ids {sorted(participants, key=str)}')
                old, installed = history.pop(offset)
                self._hard_work(old)
                fingerprint = tuple(sorted((i, tuple(installed[i])) for i in old if old[i] != installed[i]))
                self.hard_banned.add(fingerprint)
                self._commit_cells(old)
                paths = self._affected_paths(old)
                self.paths.update(paths)
                keys = set().union(*(self.pair_index[k] for k in paths)) if paths else set()
                self.hard_pairs.difference_update(keys)
                for key in keys:
                    left, right = (self.notes[self.index[k]] for k in key)
                    event = metrics.crossing_pair_event(left, right, self.paths[key[0]], self.paths[key[1]])
                    self.pairs[key] = event
                    if event['hard']:
                        self.hard_pairs.add(key)
                identities = {self.notes[i]['id'] for i in old}
                self.prep_hard.difference_update(identities)
                self.prep_hard.update(event['id'] for event in metrics.preparation_events([self.notes[i] for i in old]) if event['hard'])
                self.hard_backtracks += 1
                failed.clear()
                continue
            edited = set(best[0])
            prior = {i: (self.notes[i]['x'], self.notes[i]['y'], self.notes[i]['d']) for i in edited}
            affected = self._affected_paths(edited)
            affected_times = [self.notes[self.index[k]]['time'] for k in affected]
            self.commit(*best)
            history.append((prior, dict(best[0])))
            failed = {i for i in failed if all(abs(self.times[i]-t) > 3. for t in affected_times)}


def repair_priority(notes, cap, *, protect_retained=False):
    """Repair hard priorities while preserving identity, rhythm and note count."""
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_visibility import visibility_costs
    original = [(n, n['x'], n['y'], n['d']) for n in notes]
    before = audit_priority(notes)
    clearance_before = post.clearance_violations(notes)
    clearance_identities = _clearance_findings(notes)
    try:
        state = _PrioritySearch(notes, cap, protect_retained)
        state.repair_hard()
        state.hard_budget_active = False
        state._refresh_soft()
        soft_changed, soft_joint_changed = state.repair_soft()
        visibility_changed = state.repair_visibility()
        repeat_position_changed = state.separate_repeated_positions()
        after = audit_priority(notes)
        # Rebuild acceptance evidence from live cells. An incremental cache
        # must never be the final authority for an original pre-hard budget.
        visibility_after = visibility_costs(notes, state.nps)
        visibility_events = {e['id']: e for e in visibility_after['events']}
        visibility_audit = _classify_final_visibility(visibility_events.values(),
                                                     state.visibility_baseline, visibility_after['cutoff'])
        if visibility_audit['visibility_regressions']:
            raise ValueError(f"priority visibility budget failed for onset groups {visibility_audit['visibility_regressions'][:10]}")
        clearance_after = post.clearance_violations(notes)
        speed_after = state.state.speed_groups.violations(range(len(state.notes)))
        if after['crossing']['hard'] or after['preparation']['hard']:
            raise ValueError('priority hard audit failed')
        clearance_new = _clearance_findings(notes)
        if (any(clearance_new[k] - clearance_identities[k] for k in clearance_new)
                or any(clearance_after[k] > clearance_before[k] for k in clearance_before) or speed_after):
            raise ValueError('priority physical audit failed')
        return dict(visibility_audit, priority_preserved=True, priority_hard_before=before['crossing']['hard'] + before['preparation']['hard'],
                    priority_hard_after=after['crossing']['hard'] + after['preparation']['hard'], soft_changed=soft_changed, soft_joint_changed=soft_joint_changed, candidates=state.candidates,
                    repeat_position_changed=repeat_position_changed,
                    visibility_changed=visibility_changed, visibility_before=state.visibility,
                    visibility_after=visibility_after,
                    visibility_limits=[dict(id=anchor, time=e['time'], group_ids=e['group_ids'],
                        baseline=state.visibility_baseline.get(anchor, 0.),
                        limit=max(state.visibility_baseline.get(anchor, 0.), visibility_after['cutoff']), severity=e['severity'])
                        for anchor, e in visibility_events.items()],
                    visibility_work_by_bucket={str(k): v for k, v in sorted(state.visibility_work.items())},
                    changed=sum((n['x'], n['y'], n['d']) != (x, y, d) for n, x, y, d in original),
                    clearance_before=clearance_before, clearance_after=clearance_after,
                    baseline_speed_violations=sorted(state.baseline_speed, key=str),
                    hard_backtracks=state.hard_backtracks, hard_work_by_bucket={str(k): v for k, v in sorted(state.hard_work.items())},
                    priority_speed_limits=[dict(id=n['id'], time=n['time'], hand=n['hand'], x=n['x'], y=n['y'], d=n['d'],
                        requested_cap=state.requested_caps[n['id']] if math.isfinite(state.requested_caps[n['id']]) else None,
                        effective_cap=state.effective_caps[n['id']] if math.isfinite(state.effective_caps[n['id']]) else None)
                        for n in state.notes])
    except Exception:
        for n, x, y, d in original:
            n['x'], n['y'], n['d'] = x, y, d
        raise
