"""Bounded opt-in arrow repairs using production flow and movement constraints.

The standalone entry point also supports archived passages. ``cap`` is keyed
by note identity by the primary speed pass; a deleted note never shifts a
survivor's budget. No model, audio, corpus or random state is read here.
"""
import math
from functools import lru_cache
from bisect import bisect_left, bisect_right

from tools.config import config


def _horizontal(first, fd, second, sd):
    return int(0 < second['time'] - first['time'] <= .25 + 1e-9
               and {fd, sd} == {2, 3})


def _edge_cost(first, fd, second, sd):
    stress = (1. if sd in (2, 3) else .5 if sd in (4, 5, 6, 7) else 0.)
    return (_horizontal(first, fd, second, sd),
            stress * max(0., 1. - (second['time'] - first['time']) / .5))


def _combined_mode():
    return bool(getattr(config, 'v2_playability_reach', False))


def _map_nps(notes):
    if _combined_mode():
        from map_creation.v2_playability_reach import note_span_nps
        return note_span_nps(notes)
    span = notes[-1]['time'] - notes[0]['time'] if len(notes) > 1 else 0.
    return len(notes) / span if span > 0 else 0.


def _budget_note_count(notes):
    return sum(n['d'] != 8 for n in notes) if _combined_mode() else len(notes)


def _diagonal_growth_budget(total_notes, diagonal_count, nps):
    if _combined_mode():
        from map_creation.v2_playability_reach import diagonal_fraction_limit
        return min(max(0, math.floor(total_notes * diagonal_fraction_limit(nps) + 1e-9) - diagonal_count),
                   max(0, math.floor(total_notes * .01 + 1e-9)))
    return max(0, math.floor(total_notes * .01 + 1e-9))


def axis_run_events(notes):
    """Observe completed axes and censored tails without joining hands or dots.

    Actual 135-degree cuts change the undirected axis by 45 degrees; both
    angles remain available. Incoming recovery and time to the next axis
    change are distinct from the completed run's full duration. End-of-map
    and dot tails provide lower bounds only, never fabricated completions.
    """
    from map_creation import v2_postprocessing as pp
    eps = float(getattr(config, 'min_time_diff', .02))
    result = dict(completed=[], censored=[], transitions=0, fast_reversals=0,
                  per_hand_transitions={0: 0, 1: 0})
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    for hand in (0, 1):
        previous, start, count, anchor = None, None, 0, None
        last_event = None
        for note in (n for n in ordered if n['hand'] == hand):
            if anchor is not None and note['time'] - anchor <= eps + 1e-9:
                continue
            anchor = note['time']
            if note['d'] == 8:
                if previous is not None:
                    result['censored'].append(dict(hand=hand, notes=count,
                        seconds=previous['time']-start, start_time=start, end_time=previous['time']))
                previous, start, count, last_event = None, None, 0, None
                continue
            if previous is None:
                previous, start, count = note, note['time'], 1
                continue
            gap = note['time'] - previous['time']
            steps = pp._step_distance(previous['d'], note['d'])
            result['transitions'] += 1
            result['per_hand_transitions'][hand] += 1
            result['fast_reversals'] += int(steps >= 3 and gap < .25 - math.ulp(note['time']))
            before_axis = pp._DIRECTION_ANGLE[previous['d']] % 4
            after_axis = pp._DIRECTION_ANGLE[note['d']] % 4
            angle = min((before_axis-after_axis) % 4, (after_axis-before_axis) % 4) * 45
            if angle:
                event = dict(hand=hand, ending_angle=angle, actual_degrees=steps*45,
                    notes=count, seconds=note['time']-start, incoming_seconds=gap,
                    start_time=start, change_time=note['time'], next_change_seconds=None)
                if last_event is not None:
                    last_event['next_change_seconds'] = note['time']-last_event['change_time']
                result['completed'].append(event)
                last_event, start, count = event, note['time'], 0
            count += 1
            previous = note
        if previous is not None:
            result['censored'].append(dict(hand=hand, notes=count,
                seconds=previous['time']-start, start_time=start, end_time=previous['time']))
    return result


def _hand_style(seq, references):
    """Distribution distance plus explicitly censored terminal axis runs.

    A missing angle is a coverage deficit, not a fabricated completed run.
    Dot-separated single arrows have no eligible pair and earn no verdict.
    """
    import numpy as np
    from map_creation import v2_postprocessing as pp
    eps = float(getattr(config, 'min_time_diff', .02))
    kept, anchor = [], None
    for note in seq:
        if anchor is None or note['time'] - anchor > eps + 1e-9:
            kept.append(note)
            anchor = note['time']
    eligible = sum(a['d'] != 8 and b['d'] != 8 for a, b in zip(kept, kept[1:]))
    runs = _axis_runs(kept)
    if not eligible:
        return 0., runs
    total_reference = sum(ref['runs'] for ref in references.values())
    score = 0.
    for angle, observations in runs.items():
        ref = references[angle]
        weight = ref['runs'] / total_reference
        if not observations:
            score += weight * min(2., eligible / max(1., ref['notes'][1]))
            continue
        lengths = np.asarray([r[0] for r in observations])
        durations = np.asarray([r[1] for r in observations])
        score += weight * (float(np.mean(abs(np.quantile(lengths, [.25, .5, .75]) - ref['notes'])
                                         / np.maximum(ref['notes'], 1.)))
                           + float(np.mean(abs(np.quantile(durations, [.25, .5, .75]) - ref['ms'])
                                           / np.maximum(ref['ms'], 250.)))
                           + abs(float(np.mean(lengths == 1)) - ref['one_note'])
                           + abs(float(np.mean(lengths >= 5)) - ref['five_or_more']))
    total = sum(len(v) for v in runs.values())
    if total:
        score += abs(len(runs[90]) / total - references[90]['runs'] / total_reference)
    # A never-ending axis cannot improve by hiding all completed observations.
    # These lower bounds are censored tails, not extra dwell observations.
    upper_notes = sum(ref['runs'] / total_reference * ref['notes'][2] for ref in references.values())
    upper_ms = sum(ref['runs'] / total_reference * ref['ms'][2] for ref in references.values())
    tails, previous, start, count = [], None, None, 0
    for note in kept:
        axis = None if note['d'] == 8 else pp._DIRECTION_ANGLE[note['d']] % 4
        if axis is None:
            if count >= 2:
                tails.append((count, last_time - start))
            previous, start, count = None, None, 0
        else:
            if previous is None or axis != previous:
                start, count = note['time'], 0
            count += 1
            last_time, previous = note['time'], axis
    if count >= 2:
        tails.append((count, last_time - start))
    for count, duration in tails:
        score += min(4., max(0., count / max(1., upper_notes) - 1.))
        score += min(4., max(0., duration * 1000. / max(250., upper_ms) - 1.))
    return score, runs


def direction_interval_reference(nps, hand, angle):
    """Frozen joint two-tail event bounds; historical quartile API is separate."""
    from map_creation.v2_playability_reference import nps_band
    return _direction_reference()[f'{nps_band(nps)}|{hand}|{angle}']


@lru_cache(maxsize=1)
def _direction_reference():
    import json
    from pathlib import Path
    return json.loads(Path(__file__).with_name('v2_playability_direction_reference.json').read_text())['strata']


def direction_tail_costs(notes, nps=None):
    """Finite normal-range M5 cost with explicit completed/censored coverage.

    The <=10% joint human event budget covers short and long run lengths,
    durations and incoming gaps together. Actual 135-degree cuts count as
    45-degree axis changes. Censored tails can exceed only upper bounds.
    No penalty is assigned merely for missing a particular ending angle.
    """
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    nps = _map_nps(ordered) if nps is None else nps
    events = axis_run_events(ordered)
    result = dict(score=0., completed_runs=len(events['completed']),
        eligible_transitions=events['transitions'], fast_reversals=events['fast_reversals'],
        fast_axis_changes=0, short_runs=0, long_runs=0, censored_long_runs=0,
        flagged_runs=0, per_hand={})
    for hand in (0, 1):
        observed = [e for e in events['completed'] if e['hand'] == hand]
        penalty = 0.
        for event in observed:
            reference = direction_interval_reference(nps, hand, event['ending_angle'])
            result['fast_axis_changes'] += int(event['incoming_seconds'] < .5 - 1e-9)
            if not reference['supported']:
                continue
            low, high = False, False
            for field, (lower, upper) in reference['intervals'].items():
                value = event[field]
                low |= value < lower - 1e-9
                high |= value > upper + 1e-9
                penalty += max(0., lower-value-1e-9) / max(lower, .02)
                penalty += max(0., value-upper-1e-9) / max(upper, .02)
            result['short_runs'] += int(low)
            result['long_runs'] += int(high)
            result['flagged_runs'] += int(low or high)
        references = [direction_interval_reference(nps, hand, angle) for angle in (45, 90)]
        supported = [r for r in references if r['supported']]
        for tail in (e for e in events['censored'] if e['hand'] == hand and e['notes'] >= 2):
            if not supported:
                continue
            # An unfinished run has no ending angle. Use the more permissive
            # supported upper bound, so an unknown ending cannot invent harm.
            bounds = {field: max(r['intervals'][field][1] for r in supported)
                      for field in ('notes', 'seconds')}
            excess = sum(max(0., tail[field]-upper-1e-9) / max(upper, .02)
                         for field, upper in bounds.items())
            result['censored_long_runs'] += int(excess > 0.)
            penalty += excess
        # Fixed eligible arrow pairs cannot be diluted by inventing extra
        # normal completions while leaving an offending run unchanged.
        penalty /= max(1, events['per_hand_transitions'][hand])
        result['per_hand'][str(hand)] = dict(score=penalty, completed_runs=len(observed))
        result['score'] += penalty
    return result


def window_style_penalty(seq, nps):
    """Frozen normal-range M5 tail cost for a bounded joint-search window."""
    if _combined_mode():
        return direction_tail_costs(seq, nps)['score']
    from map_creation.v2_playability_reference import dwell_reference
    ordered = sorted(seq, key=lambda n: (n['time'], n['id']))
    return sum(_hand_style([n for n in ordered if n['hand'] == hand],
        {angle: dwell_reference(nps, hand, angle) for angle in (45, 90)})[0]
        for hand in (0, 1))


def _combined_window_path(seq, start, stop, cap, cells, fast_ids, diagonal_allowance,
                          *, forced, require_improvement, nps, guard=None, joint_cells=None):
    """Bounded style-diverse search with cached local arrow-edge legality."""
    from map_creation import v2_postprocessing as pp
    from map_creation.v2_playability_reach import calibrated_cutoff, forbidden_cut, local_reach_cost
    from map_creation.v2_playability_reference import dwell_reference
    gap = float(getattr(config, 'v2_cleanup_same_dir_gap', .1) or 0.)
    end = min(stop + 1, len(seq))
    window = seq[start - 1:end]
    mutable = stop - start
    max_diagonals = min(mutable, sum(n['d'] in (4, 5, 6, 7) for n in seq[start:stop]) + diagonal_allowance)
    cutoff = calibrated_cutoff('m1', nps)
    excess = lambda a, ad, b, bd: max(0., _edge_cost(a, ad, b, bd)[1] - cutoff)
    incumbent_excess = sum(excess(a, a['d'], b, b['d']) for a, b in zip(window, window[1:]))
    incumbent_reach = sum(local_reach_cost(a, b, nps) for a, b in zip(window, window[1:]))
    from map_creation.v2_playability_lateral import consecutive_horizontal_load
    run_cost = lambda ns: consecutive_horizontal_load(ns, nps)['total_excess']
    incumbent_run = run_cost(window)
    incumbent_total = incumbent_excess + incumbent_reach + incumbent_run
    if require_improvement and incumbent_excess+incumbent_run <= 1e-9:
        return None
    refs = {angle: dwell_reference(nps, window[0]['hand'], angle) for angle in (45, 90)}

    @lru_cache(maxsize=None)
    def partial_rank(state):
        cost, edits, path = state
        previous = pp._DIRECTION_ANGLE[path[0]] % 4
        run_start, run_notes, completed, distance = 0, 1, 0, 0.
        for i, d in enumerate(path[1:], 1):
            axis = pp._DIRECTION_ANGLE[d] % 4
            if axis != previous:
                angle = min((axis - previous) % 4, (previous - axis) % 4) * 45
                ref = refs[angle]
                distance += abs(run_notes - ref['notes'][1]) / max(1., ref['notes'][1])
                distance += abs((window[i]['time'] - window[run_start]['time']) * 1000. - ref['ms'][1]) / max(250., ref['ms'][1])
                run_start, run_notes, completed = i, 0, completed + 1
            run_notes += 1
            previous = axis
        upper = max(ref['notes'][2] for ref in refs.values())
        tail = max(0., run_notes / max(1., upper) - 1.)
        prefix = [{**n, 'd': d} for n, d in zip(window, path)]
        return cost + run_cost(prefix) + .12 * (distance / max(1, completed) + tail), edits, path

    options = {(window[0]['d'], 0): [(0., 0, (window[0]['d'],))]}
    for index, note in enumerate(window[1:], 1):
        previous = window[index - 1]
        fixed = start + index - 1 >= stop
        candidates = ((note['d'],) if fixed else
                      (forced[note['id']],) if forced and note['id'] in forced else range(8))
        next_options = {}
        # Each geometric edge is evaluated once, independently of path history.
        edges = {}
        arrival_cap = cap(note)
        for direction in candidates:
            if forbidden_cut(note, direction=direction):
                continue
            if not pp._swing_clear((note['x'], note['y']), cells.get(note['id']), direction):
                continue
            if not fixed and note['id'] in fast_ids and direction in pp._COUNTER_FAMILY.get(note['y'], ()):
                continue
            for prior_direction in {key[0] for key in options}:
                prior = (previous['time'], previous['x'], previous['y'], prior_direction)
                if not pp._flow_legal(direction, prior, note['x'], note['y'], note['time'], gap):
                    continue
                speed = pp.calc_note_speed([previous['x'], previous['y'], previous['hand'], prior_direction],
                                           [note['x'], note['y'], note['hand'], direction],
                                           note['time'] - previous['time'])
                if speed <= arrival_cap + 1e-9:
                    prior_note, current_note = {**previous, 'd': prior_direction}, {**note, 'd': direction}
                    reach = local_reach_cost(prior_note, current_note, nps)
                    edges[prior_direction, direction] = excess(previous, prior_direction, note, direction) + reach
        for (prior_direction, diagonal_count), states in options.items():
            for direction in candidates:
                edge = edges.get((prior_direction, direction))
                count = diagonal_count + int(not fixed and direction in (4, 5, 6, 7))
                if edge is None or count > max_diagonals:
                    continue
                key = direction, count
                for cost, edits, path in states:
                    if guard is not None and incumbent_run > 0 and len(path) >= 2:
                        axes = [pp._DIRECTION_ANGLE[d] % 4 for d in (*path[-2:], direction)]
                        original_axes = [pp._DIRECTION_ANGLE[n['d']] % 4 for n in window[index-2:index+1]]
                        if (axes[0] == axes[2] != axes[1]
                                and not (original_axes[0] == original_axes[2] != original_axes[1])):
                            continue
                    next_options.setdefault(key, []).append((cost + edge, edits + int(not fixed and direction != note['d']),
                                                             path + (direction,)))
        options = {}
        for key, states in next_options.items():
            ranked = sorted(states, key=partial_rank)
            # Preserve a minimal alternative if broad rotations fail reach.
            minimal = min(states, key=lambda state: (state[1], partial_rank(state)))
            kept = ranked[:3]
            if minimal not in kept:
                kept.append(minimal)
            elif len(ranked) > 3:
                kept.append(ranked[3])
            options[key] = kept
        if not options:
            return None
    # Score only bounded context, never the complete hand stream inside DP.
    context_start, context_stop = max(0, start - 8), min(len(seq), stop + 9)
    context = [dict(n) for n in seq[context_start:context_stop]]
    incumbent_style = window_style_penalty(context, nps)
    candidates = [state for states in options.values() for state in states]
    if guard is not None and joint_cells is not None and incumbent_run > 0:
        # Keep small adjacent diagonal alternatives that a load-optimal beam
        # may prune before their jointly feasible cell adjustment is known.
        original_path = tuple(n['d'] for n in window)
        for index in range(1, mutable):
            for first, second in ((4, 7), (7, 4), (5, 6), (6, 5)):
                path = list(original_path)
                path[index:index+2] = first, second
                if sum(d in (4, 5, 6, 7) for d in path[1:1+mutable]) > max_diagonals:
                    continue
                legal = True
                for offset in range(1, len(window)):
                    a, b = window[offset-1:offset+1]
                    ad, bd = path[offset-1:offset+1]
                    prior = (a['time'], a['x'], a['y'], ad)
                    if not pp._flow_legal(bd, prior, b['x'], b['y'], b['time'], gap):
                        legal = False
                        break
                if legal:
                    edits = sum(d != n['d'] for n, d in zip(window, path))
                    candidates.append((0., edits, tuple(path)))
    candidates.sort(key=lambda state: (state[1] > 2, partial_rank(state)))
    best, best_cells, bridge_checks = None, {}, 0
    for cost, edits, path in candidates:
        if forced:
            candidate_reach = sum(local_reach_cost({**a, 'd': ad}, {**b, 'd': bd}, nps)
                                  for a, ad, b, bd in zip(window, path, window[1:], path[1:]))
            if candidate_reach > incumbent_reach + 1e-9:
                continue
        for offset, direction in enumerate(path[1:1 + mutable], start - context_start):
            context[offset]['d'] = direction
        style = window_style_penalty(context, nps)
        actual = seq[start:stop]
        original_cuts = [n['d'] for n in actual]
        original_cells = {n['id']: (n['x'], n['y']) for n in actual}
        candidate_cells = {}
        try:
            for note, direction in zip(actual, path[1:1+mutable]):
                note['d'] = direction
            changed = {n['id'] for n, d in zip(actual, original_cuts) if n['d'] != d}
            physical = None
            physical_ids = set()
            if guard is not None and changed:
                from map_creation.v2_playability_reach import _Search
                physical = _Search(guard.notes, cap, nps, False)
                physical_ids = {i for i, n in enumerate(guard.notes) if n['id'] in changed}
            approved = (not changed or guard is None
                        or (physical.legal(physical_ids) and guard.allows(changed)))
            if not approved and joint_cells is not None and bridge_checks < 32 and run_cost(window) < incumbent_run:
                from map_creation.v2_playability_lateral_bridge import find_lateral_bridge
                bridge_checks += 1
                candidate_cells = find_lateral_bridge(guard.notes, cap, {n['id'] for n in actual}, guard)
                if candidate_cells is not None:
                    for note in actual:
                        if note['id'] in candidate_cells:
                            note['x'], note['y'] = candidate_cells[note['id']]
                    changed.update(candidate_cells)
                    physical_ids = {i for i, n in enumerate(guard.notes) if n['id'] in changed}
                    approved = physical.legal(physical_ids) and guard.allows(changed)
            if not approved:
                continue
            repeated = run_cost(window)
            total_cost = repeated + sum(excess(a, a['d'], b, b['d'])+local_reach_cost(a, b, nps)
                                        for a, b in zip(window, window[1:]))
        finally:
            for note, direction in zip(actual, original_cuts):
                note['d'] = direction
                note['x'], note['y'] = original_cells[note['id']]
        rank = total_cost + .12 * style + .002 * edits, edits, path
        if require_improvement and (total_cost >= incumbent_total-1e-9
                or rank[0] >= incumbent_total+.12*incumbent_style-1e-9):
            continue
        if best is None or rank < best[0]:
            best = rank, total_cost, path
            best_cells = dict(candidate_cells or {})
            if guard is not None and joint_cells is not None:
                break
    if best is None:
        return None
    if require_improvement and (best[1] >= incumbent_total - 1e-9
                                or best[0][0] >= incumbent_total + .12 * incumbent_style - 1e-9):
        return None
    if joint_cells is not None:
        joint_cells.update(best_cells)
    return best[2][1:1 + mutable]


def _window_path(seq, start, stop, cap, cells, fast_ids, diagonal_allowance,
                 *, forced=None, require_improvement=True, nps=None, guard=None, joint_cells=None):
    """Legacy exact DP or combined style-diverse search; <=16 mutable arrows."""
    if _combined_mode():
        span = seq[-1]['time'] - seq[0]['time'] if len(seq) > 1 else 0.
        nps = len(seq) / span if nps is None and span > 0 else (nps or 0.)
        return _combined_window_path(seq, start, stop, cap, cells, fast_ids, diagonal_allowance,
                                     forced=forced, require_improvement=require_improvement, nps=nps, guard=guard, joint_cells=joint_cells)
    from map_creation import v2_postprocessing as pp
    gap = float(getattr(config, 'v2_cleanup_same_dir_gap', .1) or 0.)
    # Include the unchanged outgoing arrow, so its arrival cap and flow are
    # part of the search. Callers keep dot-adjacent arrows fixed.
    end = min(stop + 1, len(seq))
    window = seq[start - 1:end]
    mutable = stop - start
    existing_diagonals = sum(n['d'] in (4, 5, 6, 7) for n in seq[start:stop])
    max_diagonals = min(mutable, existing_diagonals + diagonal_allowance)
    options = {(window[0]['d'], 0): ((0, 0., 0), (window[0]['d'],))}
    for index, note in enumerate(window[1:], 1):
        previous = window[index - 1]
        fixed = start + index - 1 >= stop
        candidates = ((note['d'],) if fixed else
                      (forced[note['id']],) if forced and note['id'] in forced else range(8))
        next_options = {}
        for direction in candidates:
            if not pp._swing_clear((note['x'], note['y']), cells.get(note['id']), direction):
                continue
            if not fixed and note['id'] in fast_ids and direction in pp._COUNTER_FAMILY.get(note['y'], ()):
                continue
            for (prior_direction, diagonal_count), (cost, path) in options.items():
                count = diagonal_count + int(not fixed and direction in (4, 5, 6, 7))
                if count > max_diagonals:
                    continue
                prior = (previous['time'], previous['x'], previous['y'], prior_direction)
                if not pp._flow_legal(direction, prior, note['x'], note['y'], note['time'], gap):
                    continue
                speed = pp.calc_note_speed([previous['x'], previous['y'], previous['hand'], prior_direction],
                                           [note['x'], note['y'], note['hand'], direction],
                                           note['time'] - previous['time'])
                if speed > cap(note) + 1e-9:
                    continue
                lateral, stress = _edge_cost(previous, prior_direction, note, direction)
                candidate = ((cost[0] + lateral, cost[1] + stress,
                              cost[2] + int(direction != note['d'])), path + (direction,))
                key = (direction, count)
                if key not in next_options or candidate < next_options[key]:
                    next_options[key] = candidate
        options = next_options
        if not options:
            return None
    if not options:
        return None
    best_cost, path = min(options.values())
    incumbent = (0, 0., 0)
    for first, second in zip(window, window[1:]):
        lateral, stress = _edge_cost(first, first['d'], second, second['d'])
        incumbent = (incumbent[0] + lateral, incumbent[1] + stress, 0)
    if require_improvement and best_cost >= incumbent:
        return None
    return path[1:1 + mutable]


def repair_directions(notes, cap, *, total_notes=None, diagonal_count=None, diagonal_budget=None, guard=None):
    """Repair fast horizontal runs with fixed identity, timing and hands.

    Legacy arms change arrows only. Complete guarded repair may jointly move
    a bounded set of nearby cells to avoid introducing new reach excursions.

    Runs require three exact reversals with gaps <=250 ms. Fixed outside
    arrows bound deterministic overlapping windows of at most 16 mutable
    arrows. Stacks and dots split search, and arrows next to a dot remain
    fixed to prevent changes propagating through virtual dot directions.
    ``total_notes``/``diagonal_count`` carry the original whole-map budget
    when explicitly operating on a portable archived passage.
    """
    from map_creation import v2_postprocessing as pp
    stats = {'m1_windows': 0, 'm1_windows_improved': 0, 'm1_direction_changed': 0,
             'm1_diagonal_bridges': 0, 'm1_diagonal_net': 0, 'm1_cell_changed': 0}
    if len(notes) < 4:
        return stats
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    nps = _map_nps(ordered)
    initial = {n['id']: n['d'] for n in ordered}
    initial_cells = {n['id']: (n['x'], n['y']) for n in ordered}
    by_id = {n['id']: n for n in ordered}
    initial_diagonals = sum(n['d'] in (4, 5, 6, 7) for n in notes)
    total_notes = _budget_note_count(notes) if total_notes is None else total_notes
    diagonal_count = initial_diagonals if diagonal_count is None else diagonal_count
    # Legacy permits 1% net growth; combined uses the frozen human arrow
    # fraction and preserves inputs that already exceed that fraction.
    growth_budget = (_diagonal_growth_budget(total_notes, diagonal_count, nps)
                     if diagonal_budget is None else diagonal_budget)
    eps = float(getattr(config, 'min_time_diff', .02))
    swing = max(eps, float(getattr(config, 'v2_cleanup_swing_window', .15) or 0.))
    cells = pp._other_hand_cells(ordered, eps, swing)
    before_clearance = pp.clearance_violations(notes)
    fast_gap = float(getattr(config, 'v2_fast_gap_seconds', .2) or 0.)
    per_hand = [[n for n in ordered if n['hand'] == hand] for hand in (0, 1)]
    fast_ids = set()
    if getattr(config, 'v2_fast_counter_direction', 'recut') == 'recut':
        for seq in per_hand:
            for first, second in zip(seq, seq[1:]):
                if second['time'] - first['time'] < fast_gap:
                    fast_ids.update((first['id'], second['id']))
    for seq in per_hand:
        style_before = window_style_penalty(seq, nps) if _combined_mode() else None
        protected = set()
        for index, note in enumerate(seq):
            if note['d'] == 8:
                protected.update(n['id'] for n in seq[max(0, index - 1):index + 2])
        for first, second in zip(seq, seq[1:]):
            if second['time'] - first['time'] <= eps + 1e-9:
                protected.update((first['id'], second['id']))
        def horizontal_spans(lo, hi):
            spans, begin = [], None
            for index in range(lo+1, hi+2):
                active = (index <= hi and not ({seq[index-1]['id'], seq[index]['id']} & protected)
                          and _horizontal(seq[index-1], seq[index-1]['d'], seq[index], seq[index]['d']))
                if active and begin is None:
                    begin = index-1
                if not active and begin is not None:
                    if index-1-begin >= 3:
                        spans.append((begin, index-1))
                    begin = None
            return spans
        runs = horizontal_spans(0, len(seq)-1)
        pending = (sorted(runs, key=lambda span: (span[1]-span[0], span[0]))
                   if guard is not None else list(reversed(runs)))
        while pending:
            first, last = pending.pop()
            starts = list(range(first+1, last+1, 14))
            for start in (reversed(starts) if guard is not None else starts):
                stop = min(start+16, last+1)
                stats['m1_windows'] += 1
                current_delta = sum(n['d'] in (4, 5, 6, 7) for n in notes) - initial_diagonals
                cell_edits = {}
                path = _window_path(seq, start, stop, cap, cells, fast_ids,
                                    growth_budget - current_delta, nps=nps, guard=guard, joint_cells=cell_edits)
                if path is None:
                    continue
                previous = [n['d'] for n in seq[start:stop]]
                previous_cells = {identity: (by_id[identity]['x'], by_id[identity]['y']) for identity in cell_edits}
                for identity, xy in cell_edits.items():
                    by_id[identity]['x'], by_id[identity]['y'] = xy
                for note, direction in zip(seq[start:stop], path):
                    note['d'] = direction
                changed_ids = {n['id'] for n, d in zip(seq[start:stop], previous) if n['d'] != d} | set(cell_edits)
                residual = pp.clearance_violations(notes)
                style_after = window_style_penalty(seq, nps) if style_before is not None else None
                if (any(residual[key] > before_clearance[key] for key in residual)
                        or (style_before is not None and style_after > style_before + 1e-9)
                        or (guard is not None and changed_ids and not guard.allows(changed_ids))):
                    for note, direction in zip(seq[start:stop], previous):
                        note['d'] = direction
                    for identity, xy in previous_cells.items():
                        by_id[identity]['x'], by_id[identity]['y'] = xy
                    continue
                if guard is not None and changed_ids:
                    guard.commit(changed_ids)
                if cell_edits:
                    cells = pp._other_hand_cells(ordered, eps, swing)
                style_before = style_after
                stats['m1_windows_improved'] += 1
                if guard is not None:
                    # Revisit only shorter remaining pieces of this run.
                    # Every accepted step removes at least one horizontal
                    # arrival, so work is bounded by the original run length.
                    pending.extend(span for span in horizontal_spans(first, last)
                                   if span[1]-span[0] < last-first)
                    break
    stats['m1_direction_changed'] = sum(n['d'] != initial[n['id']] for n in notes)
    stats['m1_diagonal_bridges'] = sum(n['d'] in (4, 5, 6, 7) and initial[n['id']] not in (4, 5, 6, 7)
                                       for n in notes)
    stats['m1_cell_changed'] = sum((n['x'], n['y']) != initial_cells[n['id']] for n in notes)
    stats['m1_diagonal_net'] = sum(n['d'] in (4, 5, 6, 7) for n in notes) - initial_diagonals
    return stats


def _axis_runs(seq):
    """Completed cut-axis runs by ending angle; dots break and stacks collapse."""
    from map_creation import v2_postprocessing as pp
    eps = float(getattr(config, 'min_time_diff', .02))
    runs = {45: [], 90: []}
    previous = None
    run_start = None
    run_notes = 0
    anchor_time = None
    for note in seq:
        if anchor_time is not None and note['time'] - anchor_time <= eps + 1e-9:
            continue
        anchor_time = note['time']
        if note['d'] == 8:
            previous, run_start, run_notes = None, None, 0
            continue
        axis = pp._DIRECTION_ANGLE[note['d']] % 4
        if previous is not None and axis != previous:
            steps = min((axis - previous) % 4, (previous - axis) % 4)
            runs[steps * 45].append((run_notes, (note['time'] - run_start) * 1000.))
            run_start, run_notes = note['time'], 0
        elif previous is None:
            run_start = note['time']
        run_notes += 1
        previous = axis
    return runs


def _dwell_distance(seq, references):
    import numpy as np
    runs = _axis_runs(seq)
    score = 0.
    for angle, observations in runs.items():
        if not observations:
            # Missing observed changes are not a perfect distribution match.
            # A finite penalty lets an actual decoder proposal restore an
            # eligible long-run tail; completed-run coverage is also guarded.
            score += 4. if len(seq) >= 2 else 0.
            continue
        target = references[angle]
        lengths = np.asarray([r[0] for r in observations])
        durations = np.asarray([r[1] for r in observations])
        score += float(np.mean(abs(np.quantile(lengths, [.25, .5, .75]) - target['notes'])
                               / np.maximum(target['notes'], 1.)))
        score += float(np.mean(abs(np.quantile(durations, [.25, .5, .75]) - target['ms'])
                               / np.maximum(target['ms'], 250.)))
        score += abs(float(np.mean(lengths == 1)) - target['one_note'])
        score += abs(float(np.mean(lengths >= 5)) - target['five_or_more'])
    total = sum(len(v) for v in runs.values())
    if total:
        expected = references[90]['runs'] / sum(ref['runs'] for ref in references.values())
        score += abs(len(runs[90]) / total - expected)
    return score, runs


def _dwell_harmful_ids(seq, nps):
    """Spend bounded search on observed tail regions, preserving normal prefixes."""
    events = axis_run_events(seq)
    ranges = []
    for event in events['completed']:
        reference = direction_interval_reference(nps, event['hand'], event['ending_angle'])
        if reference['supported'] and any(event[field] < lo-1e-9 or event[field] > hi+1e-9
                for field, (lo, hi) in reference['intervals'].items()):
            ranges.append((event['start_time'], event['change_time']))
    for event in events['censored']:
        references = [direction_interval_reference(nps, event['hand'], angle) for angle in (45, 90)]
        supported = [r for r in references if r['supported']]
        if supported and any(event[field] > max(r['intervals'][field][1] for r in supported)+1e-9
                             for field in ('notes', 'seconds')):
            ranges.append((event['start_time'], event['end_time']))
    times = [n['time'] for n in seq]
    return {n['id'] for lo, hi in ranges
            for n in seq[bisect_left(times, lo):bisect_right(times, hi)]}


def repair_dwell(notes, cap, *, decoded_directions=None, diagonal_budget=None, guard=None):
    """Separate conservative dwell arm, using frozen supported human strata.

    Smooth one-note 45-degree detours or restore an actual decoder proposal
    inside a >=5-note axis run. Single-arrow proposals reuse the bounded
    direction search and keep both outside neighbors fixed. Only strict
    distribution improvements with no M1-stress increase are committed. A
    previously observed ending angle cannot disappear merely to avoid its
    score. The reference band is frozen from the pre-repair note-span NPS.
    Combined mode also restores feasible shorter decoded turns, respects
    calibrated reach tails, and keeps unfinished axis tails in its score.
    """
    from map_creation import v2_postprocessing as pp
    from map_creation.v2_playability_reference import dwell_reference, nps_band
    combined = _combined_mode()
    stats = {'dwell_direction_changed': 0, 'dwell_distance_before': 0.,
             'dwell_distance_after': 0., 'dwell_candidates': 0, 'dwell_pooled_strata': []}
    if len(notes) < 3:
        return stats
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    nps = _map_nps(ordered)
    stats['dwell_reference_band'] = nps_band(nps)
    initial = {n['id']: n['d'] for n in notes}
    decoded_directions = ({n['id']: n['_decoded_d'] for n in notes if '_decoded_d' in n}
                          if decoded_directions is None else decoded_directions)
    eps = float(getattr(config, 'min_time_diff', .02))
    swing = max(eps, float(getattr(config, 'v2_cleanup_swing_window', .15) or 0.))
    cells = pp._other_hand_cells(ordered, eps, swing)
    baseline_clearance = pp.clearance_violations(notes)
    diagonal_before = sum(n['d'] in (4, 5, 6, 7) for n in notes)
    budget = (_diagonal_growth_budget(_budget_note_count(notes), diagonal_before, nps)
              if diagonal_budget is None else diagonal_budget)
    for hand in (0, 1):
        seq = [n for n in ordered if n['hand'] == hand]
        refs = {angle: dwell_reference(nps, hand, angle) for angle in (45, 90)}
        stats['dwell_pooled_strata'].extend({'hand': hand, 'ending_angle': angle,
                                           'bands': list(ref['source_bands']), 'packages': ref['packages']}
                                          for angle, ref in refs.items() if len(ref['source_bands']) > 1)
        distance = ((lambda seq, refs: (window_style_penalty(seq, nps), _axis_runs(seq)))
                    if combined else _dwell_distance)
        score, runs = distance(seq, refs)
        stats['dwell_distance_before'] += score
        fast_ids = set()
        if getattr(config, 'v2_fast_counter_direction', 'recut') == 'recut':
            fast_gap = float(getattr(config, 'v2_fast_gap_seconds', .2) or 0.)
            for first, second in zip(seq, seq[1:]):
                if second['time'] - first['time'] < fast_gap:
                    fast_ids.update((first['id'], second['id']))
        checked = 0
        harmful_ids = _dwell_harmful_ids(seq, nps) if combined else None
        for index in range(1, len(seq) - 1):
            before, note, after = seq[index - 1:index + 2]
            if harmful_ids is not None and note['id'] not in harmful_ids:
                continue
            if any(n['d'] == 8 for n in (before, note, after)):
                continue
            if min(note['time'] - before['time'], after['time'] - note['time']) <= eps + 1e-9:
                continue
            axes = [pp._DIRECTION_ANGLE[n['d']] % 4 for n in (before, note, after)]
            difference = min((axes[1] - axes[0]) % 4, (axes[0] - axes[1]) % 4)
            detour = axes[0] == axes[2] and difference == 1
            if detour:
                proposals = [d for d in range(8) if pp._DIRECTION_ANGLE[d] % 4 == axes[0]]
            else:
                decoded = decoded_directions.get(note['id'])
                if decoded is None or decoded == note['d'] or decoded == 8:
                    continue
                left, right = index, index
                while left > 0 and seq[left - 1]['d'] != 8 and pp._DIRECTION_ANGLE[seq[left - 1]['d']] % 4 == axes[1]:
                    left -= 1
                while right + 1 < len(seq) and seq[right + 1]['d'] != 8 and pp._DIRECTION_ANGLE[seq[right + 1]['d']] % 4 == axes[1]:
                    right += 1
                if ((not combined and right - left + 1 < 5)
                        or pp._DIRECTION_ANGLE[decoded] % 4 == axes[1]):
                    continue
                proposals = [decoded]
            if checked >= 64:
                break
            checked += 1
            incumbent = note['d']
            best, best_score, best_runs = incumbent, score, runs
            for proposed in proposals:
                stats['dwell_candidates'] += 1
                delta = sum(n['d'] in (4, 5, 6, 7) for n in notes) - diagonal_before
                path = _window_path(seq, index, index + 1, cap, cells, fast_ids,
                                    budget - delta, forced={note['id']: proposed}, require_improvement=False, nps=nps, guard=guard)
                if path is None:
                    continue
                old_cost = tuple(sum(_edge_cost(a, a['d'], b, b['d'])[part]
                                     for a, b in ((before, note), (note, after))) for part in (0, 1))
                note['d'] = proposed
                new_cost = tuple(sum(_edge_cost(a, a['d'], b, b['d'])[part]
                                     for a, b in ((before, note), (note, after))) for part in (0, 1))
                candidate_score, candidate_runs = distance(seq, refs)
                loses_coverage = any(runs[angle] and not candidate_runs[angle] for angle in (45, 90))
                residual = pp.clearance_violations(notes)
                legal = all(residual[key] <= baseline_clearance[key] for key in residual)
                if (legal and not loses_coverage and new_cost[0] <= old_cost[0] and new_cost[1] <= old_cost[1] + 1e-9
                        and candidate_score < best_score - 1e-9
                        and (guard is None or guard.allows({note['id']}))):
                    best, best_score, best_runs = proposed, candidate_score, candidate_runs
                note['d'] = incumbent
            note['d'], score, runs = best, best_score, best_runs
            if best != incumbent:
                if guard is not None:
                    guard.commit({note['id']})
                if combined:
                    harmful_ids = _dwell_harmful_ids(seq, nps)
        stats['dwell_distance_after'] += score
    stats['dwell_direction_changed'] = sum(n['d'] != initial[n['id']] for n in notes)
    return stats


def repair_isolated(notes, cap, *, eligible_ids, total_notes=None, diagonal_budget=None, guard=None):
    """Re-rank only actual primary-cap recuts with both neighbors fixed.

    This separate arm uses existing feasible alternatives, never changes the
    shared legacy recut ranking, and consumes no randomness. Missing context,
    dots, stacks, exhausted diagonal budget or tied cost keep the incumbent.
    """
    from map_creation import v2_postprocessing as pp
    eligible_ids = set(eligible_ids)
    stats = {'isolated_candidates': 0, 'isolated_direction_changed': 0,
             'isolated_horizontal_before': 0, 'isolated_horizontal_after': 0}
    if not eligible_ids:
        return stats
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    nps = _map_nps(ordered)
    eps = float(getattr(config, 'min_time_diff', .02))
    swing = max(eps, float(getattr(config, 'v2_cleanup_swing_window', .15) or 0.))
    cells = pp._other_hand_cells(ordered, eps, swing)
    baseline = pp.clearance_violations(notes)
    diagonal_before = sum(n['d'] in (4, 5, 6, 7) for n in notes)
    budget = (_diagonal_growth_budget(_budget_note_count(notes) if total_notes is None else total_notes, diagonal_before, nps)
              if diagonal_budget is None else diagonal_budget)
    for hand in (0, 1):
        seq = [n for n in ordered if n['hand'] == hand]
        fast_ids = set()
        if getattr(config, 'v2_fast_counter_direction', 'recut') == 'recut':
            fast_gap = float(getattr(config, 'v2_fast_gap_seconds', .2) or 0.)
            for first, second in zip(seq, seq[1:]):
                if second['time'] - first['time'] < fast_gap:
                    fast_ids.update((first['id'], second['id']))
        for index in range(1, len(seq) - 1):
            before, note, after = seq[index - 1:index + 2]
            if note['id'] not in eligible_ids or note['d'] not in (2, 3):
                continue
            gaps = note['time'] - before['time'], after['time'] - note['time']
            if min(gaps) <= eps + 1e-9 or min(gaps) > .25 + 1e-9 or before['d'] == 8 or after['d'] == 8:
                continue
            stats['isolated_candidates'] += 1
            stats['isolated_horizontal_before'] += 1
            delta = sum(n['d'] in (4, 5, 6, 7) for n in notes) - diagonal_before
            path = _window_path(seq, index, index + 1, cap, cells, fast_ids, budget - delta, nps=nps, guard=guard)
            if path is not None:
                previous = note['d']
                note['d'] = path[0]
                residual = pp.clearance_violations(notes)
                if (any(residual[key] > baseline[key] for key in residual)
                        or (guard is not None and note['d'] != previous and not guard.allows({note['id']}))):
                    note['d'] = previous
                elif note['d'] != previous:
                    if guard is not None:
                        guard.commit({note['id']})
                    stats['isolated_direction_changed'] += 1
            stats['isolated_horizontal_after'] += int(note['d'] in (2, 3))
    return stats


def repair_late_turns(notes, cap, *, diagonal_budget=None, guard=None):
    """Restore feasible decoded 45/90-degree turns after primary selection.

    The first 30 surviving combined positions and the configured ramp remain
    fixed. Only the original arrow is proposed: this arm cannot add, delete,
    dot, retime or move notes. Both affected arrival budgets, outgoing flow,
    fast-row bounds and other-hand occupancy are checked by the same bounded
    direction search. Contextual slow-flow validation remains active later.
    """
    from map_creation import v2_postprocessing as pp
    stats = {'late_turn_candidates': 0, 'late_turn_direction_changed': 0,
             'late_turn45_changed': 0, 'late_turn90_changed': 0}
    ordered = sorted(notes, key=lambda n: (n['time'], n['id']))
    nps = _map_nps(ordered)
    minimum = max(30, int(getattr(config, 'v2_speed_cap_start_notes', 30) or 0))
    late_ids = {n['id'] for n in ordered[minimum:]}
    eps = float(getattr(config, 'min_time_diff', .02))
    swing = max(eps, float(getattr(config, 'v2_cleanup_swing_window', .15) or 0.))
    cells = pp._other_hand_cells(ordered, eps, swing)
    baseline = pp.clearance_violations(notes)
    diagonal_before = sum(n['d'] in (4, 5, 6, 7) for n in notes)
    if _combined_mode() and diagonal_budget is None:
        diagonal_budget = _diagonal_growth_budget(_budget_note_count(notes), diagonal_before, nps)
    for hand in (0, 1):
        seq = [n for n in ordered if n['hand'] == hand]
        fast_ids = set()
        if getattr(config, 'v2_fast_counter_direction', 'recut') == 'recut':
            fast_gap = float(getattr(config, 'v2_fast_gap_seconds', .2) or 0.)
            for first, second in zip(seq, seq[1:]):
                if second['time'] - first['time'] < fast_gap:
                    fast_ids.update((first['id'], second['id']))
        for index in range(1, len(seq) - 1):
            before, note, after = seq[index - 1:index + 2]
            decoded = note.get('_decoded_d')
            if note['id'] not in late_ids or decoded in (None, 8, note['d']):
                continue
            if before['d'] == 8 or after['d'] == 8 or min(note['time'] - before['time'], after['time'] - note['time']) <= eps + 1e-9:
                continue
            steps = pp._step_distance(before['d'], decoded)
            prior = (before['time'], before['x'], before['y'], before['d'])
            if not pp._slow_turn_legal(decoded, prior, note['x'], note['y'], note['time'], steps):
                continue
            stats['late_turn_candidates'] += 1
            # Restore only the decoded alternative. This arm does not impose
            # the M1 style ranking on a deliberately slower musical phrase.
            delta = sum(n['d'] in (4, 5, 6, 7) for n in notes) - diagonal_before
            allowance = 1 if diagonal_budget is None else diagonal_budget - delta
            path = _window_path(seq, index, index + 1, cap, cells, fast_ids, allowance,
                                forced={note['id']: decoded}, require_improvement=False, nps=nps)
            if path is None:
                continue
            previous = note['d']
            note['d'] = decoded
            residual = pp.clearance_violations(notes)
            if (any(residual[key] > baseline[key] for key in residual)
                    or (guard is not None and not guard.allows({note['id']}))):
                note['d'] = previous
                continue
            if guard is not None:
                guard.commit({note['id']})
            stats['late_turn_direction_changed'] += 1
            stats[f'late_turn{steps * 45}_changed'] += 1
    return stats
