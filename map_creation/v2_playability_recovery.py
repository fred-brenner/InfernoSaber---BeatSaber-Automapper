"""Deferred supported hand recovery on the unchanged primary survivor stream.

Model position/arrow histories are never modified. A chosen surviving single
or adjacent supported pair may change hands/cells/cuts; all other note
dictionaries and every onset remain fixed.
"""
from bisect import bisect_left, bisect_right
import hashlib
import json
import math

from tools.config import config


def primary_signature(notes):
    """Canonical primary geometry hash, independent of private metadata/scalars."""
    rows = sorted([int(n['id']), float(n['time']), int(n['hand']), int(n['x']),
                   int(n['y']), int(n['d'])] for n in notes)
    return hashlib.sha256(json.dumps(rows, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def repair_recovery(notes, cap, guard=None):
    """Commit bounded local recoveries only after actual fast-reversal reduction.

    Candidates originate from the original position model, not from an
    altered autoregressive trajectory. Both former/new hand arrival groups,
    outgoing flow and whole-stream clearance are recomputed for every trial.
    The shared guard retains all normal-range direction/geometry constraints.
    """
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_directions import axis_run_events
    from map_creation.v2_playability_policy import PatternGuard, SpeedGroups
    from map_creation.v2_playability_reach import forbidden_cut
    ordered = sorted(notes, key=lambda n:(n['time'], n['id']))
    eps = float(getattr(config, 'min_time_diff', .02))
    support_ratio = float(getattr(config, 'v2_playability_recovery_ratio', .9))
    identity = [(n['id'], n['time']) for n in ordered]
    times = [n['time'] for n in ordered]
    double_ids = {n['id'] for i, n in enumerate(ordered)
        if (i > 0 and n['time']-ordered[i-1]['time'] <= eps+1e-9)
        or (i+1 < len(ordered) and ordered[i+1]['time']-n['time'] <= eps+1e-9)}
    original_doubles = [(n['id'], n['hand']) for n in ordered if n['id'] in double_ids]
    fast = axis_run_events(ordered)['fast_reversals']
    stats = dict(recovery_candidates=0, recovery_proposals=0, recovery_hand_swaps=0,
        recovery_rejected_physical=0, recovery_rejected_guard=0, recovery_changed_ids=[],
        recovery_joint_candidates=0, recovery_joint_swaps=0, recovery_joint_changed_ids=[],
        recovery_fast_reversals_before=fast, recovery_fast_reversals_after=fast,
        recovery_primary_signature_sha256=primary_signature(ordered),
        recovery_preserved_onsets_count_doubles=True)
    if guard is None:
        guard = PatternGuard(ordered)
    clearance = post.clearance_violations(ordered)
    speed_groups = SpeedGroups(ordered, cap, eps)
    speed_violations = speed_groups.violations(range(len(ordered)))
    last_recovery = -math.inf
    recovery_times = []
    mirror = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6}
    considered = 0
    for note_index, note in enumerate(ordered):
        alternatives = note.get('_recovery_alternatives', ())
        if not alternatives or note['d'] == 8 or note.get('kind') == 'copy':
            continue
        stats['recovery_candidates'] += 1
        if considered >= 128 or stats['recovery_hand_swaps'] >= 8:
            continue
        time, hand = note['time'], note['hand']
        if time-last_recovery < .5-1e-9 or note['id'] in double_ids:
            continue
        past = bisect_left(times, time-eps)
        previous = next((ordered[i] for i in range(past-1, -1, -1) if ordered[i]['hand'] == hand), None)
        other = 1-hand
        other_previous = next((ordered[i] for i in range(past-1, -1, -1) if ordered[i]['hand'] == other), None)
        following = next((ordered[i] for i in range(bisect_right(times, time+eps), len(ordered))
                          if ordered[i]['hand'] == other), None)
        if (previous is None or previous['d'] == 8 or other_previous is None
                or time-previous['time'] >= .25-math.ulp(time)
                or post._step_distance(previous['d'], note['d']) < 3
                or time-other_previous['time'] < .25-1e-9
                or (following is not None and following['time']-time < .25-1e-9)):
            continue
        recent = ordered[bisect_left(times, time-2.):past]
        if sum(n['hand'] == hand for n in recent) < sum(n['hand'] == other for n in recent)+2:
            continue
        considered += 1
        former_arrivals = set(speed_groups.partners[note_index]+speed_groups.following[note_index])
        original = {k:note[k] for k in ('hand', 'x', 'y', 'd')}
        directions = list(dict.fromkeys([original['d'], mirror[original['d']],
            *sorted(range(8), key=lambda d:(post._step_distance(original['d'], d),
                                           d in (2, 3), d))]))
        accepted = False
        for alternative in alternatives[:3]:
            if (alternative.get('hand') != other or alternative.get('support', 0.) < support_ratio
                    or not 0 <= alternative.get('x', -1) < 4 or not 0 <= alternative.get('y', -1) < 3
                    or ((alternative['x'] < 2) != (other == 0))
                    or (alternative['x']-other_previous['x'])**2
                        +(alternative['y']-other_previous['y'])**2 > 2):
                continue
            for direction in directions:
                stats['recovery_proposals'] += 1
                note.update(hand=other, x=alternative['x'], y=alternative['y'], d=direction)
                if forbidden_cut(note):
                    note.update(original)
                    stats['recovery_rejected_physical'] += 1
                    continue
                proposed_fast = axis_run_events(ordered)['fast_reversals']
                residual = post.clearance_violations(ordered)
                # Rebuild groups after the hand move: the former hand's new
                # outgoing edge is absent from the old topology.
                if proposed_fast >= fast or any(residual[k] > clearance[k] for k in residual):
                    stats['recovery_rejected_physical'] += 1
                    note.update(original)
                    continue
                proposed_groups = SpeedGroups(ordered, cap, eps)
                proposed_speed = proposed_groups.violations(range(len(ordered)))
                affected = former_arrivals | {note_index}
                physical = (not (proposed_speed-speed_violations) and proposed_groups.legal(affected))
                if not physical:
                    stats['recovery_rejected_physical'] += 1
                    note.update(original)
                    continue
                if not guard.allows({note['id']}):
                    stats['recovery_rejected_guard'] += 1
                    note.update(original)
                    continue
                guard.commit({note['id']})
                fast, clearance, speed_violations = proposed_fast, residual, proposed_speed
                speed_groups = proposed_groups
                last_recovery = time
                recovery_times.append(time)
                stats['recovery_hand_swaps'] += 1
                stats['recovery_changed_ids'].append(note['id'])
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            note.update(original)
    # One interior insertion/removal often breaks alternating cut parity.
    # Adjacent supported opposite-hand singles can trade timing slots while
    # retaining each hand's cut order and total count.
    pair_candidates = 0
    for first_index in range(len(ordered)-1):
        if stats['recovery_hand_swaps'] > 6 or pair_candidates >= 128 or not fast:
            break
        second_index = first_index+1
        first, second = ordered[first_index], ordered[second_index]
        changed = {first['id'], second['id']}
        if (first['hand'] == second['hand'] or second['time']-first['time'] > .5+1e-9
                or changed & (double_ids | set(stats['recovery_changed_ids']))
                or any(n['d'] == 8 or n.get('kind') == 'copy'
                       or not n.get('_recovery_alternatives') for n in (first, second))
                or any(abs(n['time']-time) < .5-1e-9
                       for n in (first, second) for time in recovery_times)):
            continue
        pair_indices = {first_index, second_index}
        incident_fast = False
        former_arrivals = set(pair_indices)
        for index in pair_indices:
            note = ordered[index]
            former_arrivals.update(speed_groups.partners[index]+speed_groups.following[index])
            for previous in speed_groups.previous[index]:
                prior = ordered[previous]
                incident_fast |= (prior['d'] != 8 and note['d'] != 8
                    and note['time']-prior['time'] < .25-math.ulp(note['time'])
                    and post._step_distance(prior['d'], note['d']) >= 3)
            for following in speed_groups.following[index]:
                after = ordered[following]
                incident_fast |= (after['d'] != 8 and note['d'] != 8
                    and after['time']-note['time'] < .25-math.ulp(after['time'])
                    and post._step_distance(note['d'], after['d']) >= 3)
        if not incident_fast:
            continue
        recent = ordered[bisect_left(times, first['time']-2.):first_index]
        counts = [sum(n['hand'] == h for n in recent) for h in (0, 1)]
        imbalance = counts[first['hand']]-counts[second['hand']]
        if abs(imbalance-1) > abs(imbalance+1):
            continue
        alternatives = []
        for note, partner in ((first, second), (second, first)):
            other = partner['hand']
            prior = next((ordered[i] for i in range(len(ordered)-1, -1, -1)
                          if i not in pair_indices and ordered[i]['hand'] == other
                          and ordered[i]['time'] < note['time']-eps), None)
            after = next((n for i, n in enumerate(ordered)
                          if i not in pair_indices and n['hand'] == other
                          and n['time'] > note['time']+eps), None)
            if (prior is None or note['time']-prior['time'] < .25-1e-9
                    or (after is not None and after['time']-note['time'] < .25-1e-9)):
                alternatives.append([])
                continue
            alternatives.append([a for a in note['_recovery_alternatives'][:3]
                if a.get('hand') == other and a.get('support', 0.) >= support_ratio
                and 0 <= a.get('x', -1) < 4 and 0 <= a.get('y', -1) < 3
                and ((a['x'] < 2) == (other == 0))
                and (a['x']-prior['x'])**2+(a['y']-prior['y'])**2 <= 2])
        if not all(alternatives):
            continue
        pair_candidates += 1
        stats['recovery_joint_candidates'] += 1
        originals = [{key:n[key] for key in ('hand','x','y','d')} for n in (first,second)]
        cuts = list(dict.fromkeys([(originals[1]['d'], originals[0]['d']),
                                  (mirror[originals[1]['d']], mirror[originals[0]['d']])]))
        accepted = False
        for first_alternative in alternatives[0]:
            for second_alternative in alternatives[1]:
                for first_cut, second_cut in cuts:
                    stats['recovery_proposals'] += 1
                    for n, a, d in ((first,first_alternative,first_cut),
                                    (second,second_alternative,second_cut)):
                        n.update(hand=a['hand'], x=a['x'], y=a['y'], d=d)
                    proposed_fast = axis_run_events(ordered)['fast_reversals']
                    residual = post.clearance_violations(ordered)
                    physical = (not any(forbidden_cut(n) for n in (first, second))
                        and proposed_fast < fast
                        and all(residual[k] <= clearance[k] for k in residual))
                    if physical:
                        proposed_groups = SpeedGroups(ordered, cap, eps)
                        proposed_speed = proposed_groups.violations(range(len(ordered)))
                        physical = (not (proposed_speed-speed_violations)
                                    and proposed_groups.legal(former_arrivals))
                    if not physical:
                        stats['recovery_rejected_physical'] += 1
                    elif not guard.allows(changed):
                        stats['recovery_rejected_guard'] += 1
                    else:
                        guard.commit(changed)
                        fast, clearance, speed_violations = proposed_fast, residual, proposed_speed
                        speed_groups = proposed_groups
                        recovery_times.extend((first['time'],second['time']))
                        stats['recovery_hand_swaps'] += 2
                        stats['recovery_joint_swaps'] += 1
                        ids = [first['id'],second['id']]
                        stats['recovery_changed_ids'].extend(ids)
                        stats['recovery_joint_changed_ids'].extend(ids)
                        accepted = True
                    if accepted:
                        break
                    for n, original in zip((first,second),originals):
                        n.update(original)
                if accepted:
                    break
            if accepted:
                break
        if not accepted:
            for n, original in zip((first,second),originals):
                n.update(original)
    stats['recovery_fast_reversals_after'] = fast
    stats['recovery_unresolved_candidates'] = stats['recovery_candidates']-stats['recovery_hand_swaps']
    stats['recovery_preserved_onsets_count_doubles'] = (
        identity == [(n['id'], n['time']) for n in ordered]
        and original_doubles == [(n['id'], n['hand']) for n in ordered if n['id'] in double_ids])
    if not stats['recovery_preserved_onsets_count_doubles']:
        raise RuntimeError('Deferred recovery changed note identity, timing, count or a double')
    return stats
