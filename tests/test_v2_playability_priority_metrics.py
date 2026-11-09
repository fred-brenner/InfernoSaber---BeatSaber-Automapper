import json
import pytest


def note(i, t, x=1, y=1, hand=0, d=0):
    return dict(id=i, time=t, x=x, y=y, hand=hand, d=d)


def test_preparation_edges_mirrors_and_diagonal():
    from map_creation.v2_playability_priority_metrics import preparation_events
    notes = [note(1, 0, 3, 1, 0, 2), note(2, 1, 0, 1, 1, 3),
             note(3, 2, 2, 2, 0, 6), note(4, 3, 1, 1, 0, 3),
             note(5, 4, 2, 1, 1, 2), note(6, 5, 1, 2, 0, 1),
             note(7, 6, 1, 0, 0, 0)]
    events = preparation_events(notes)
    assert [e['hard'] for e in events[:2]] == [True, True]
    assert events[2]['offsets'] == [1, 1] and events[2]['excursion'] == 1 and events[2]['hard'] is False
    assert all(e['components']['specific'] > 0 for e in events[3:])


def test_preparation_no_followthrough_cost_and_speed_repetition():
    from map_creation.v2_playability_priority_metrics import preparation_events
    # Up cut at top has an in-area approach but an out-of-grid exit.
    isolated = preparation_events([note(0, 0, 1, 2, d=0)])[0]
    assert isolated['total'] == 0
    slow = preparation_events([note(0, 0, 1, 1, d=3), note(1, 2, 1, 1, d=3)])[-1]
    fast = preparation_events([note(0, 0, 1, 1, d=3), note(1, .1, 1, 1, d=3)])[-1]
    assert fast['total'] > slow['total'] > 0
    dot = preparation_events([note(0, 0, 3, 2, d=8)])[0]
    assert not dot['eligible'] and dot['total'] == 0


@pytest.mark.parametrize('dt,hard', [(0, True), (.249, True), (.25, True), (.251, False), (1.499, False), (1.5, False)])
def test_crossing_time_boundaries(dt, hard):
    from map_creation.v2_playability_priority_metrics import crossing_events
    events = crossing_events([note(0, 0, 2, 1, 0, 0), note(1, dt, 1, 1, 1, 2)])
    assert len(events) == 1 and events[0]['detected']
    assert events[0]['hard'] is hard
    assert events[0]['total'] > 0 if dt < 1.5 else events[0]['total'] == 0


def test_crossing_mirrors_lookback_and_beyond_window():
    from map_creation.v2_playability_priority_metrics import crossing_events
    notes = [note(0, 0, 0, 1, 0, 0), note(1, 1, 2, 1, 0, 0),
             note(2, 1, 1, 1, 1, 0)]
    pair = next(e for e in crossing_events(notes) if e['ids'] == [1, 2])
    assert pair['hard'] and pair['distance'] == 0
    assert crossing_events([note(0, 0), note(1, 1.501, hand=1)]) == []
    # Symmetry is not an exemption when their swept cells overlap.
    pair = crossing_events([note(0, 0, 1, 1, 0, 2), note(1, 0, 2, 1, 1, 3)])[0]
    assert pair['hard']


def test_axis_magnitudes_reversal_cadence_and_long_tail():
    from map_creation.v2_playability_priority_metrics import axis_change_events
    def measure(ds):
        return axis_change_events([note(i, i, d=d) for i, d in enumerate(ds)])
    assert measure([0, 1])[-1]['total'] == 0  # reverse, same axis
    a = measure([0, 4])[-1]
    b = measure([0, 2])[-1]
    assert b['components']['base'] == 2 * a['components']['base']
    regular = measure([0, 4, 0, 4, 0])
    irregular = measure([0, 4, 0, 0, 4])
    assert regular[2]['components']['frequency'] > 0
    assert irregular[-1]['components']['irregularity'] > regular[-1]['components']['irregularity']
    assert sum(e['total'] for e in measure([0, 4] + [4]*100)) == a['total']


def test_dots_and_stacks_are_deterministic_and_json_compatible():
    from map_creation.v2_playability_priority_metrics import priority_metrics
    notes = [note(0, 0, d=0), note(1, 1, d=8), note(2, 2, d=2),
             note(3, 2, x=2, d=4), note(4, 3, d=4)]
    result = priority_metrics(notes)
    json.dumps(result, allow_nan=False)
    assert result == priority_metrics(list(reversed(notes)))
    assert not next(e for e in result['axis']['events'] if e['id'] == 2)['eligible']
    assert sum(e['eligible'] for e in result['axis']['events'] if e['time'] == 2) == 0


def test_axis_irregularity_needs_two_interchange_intervals():
    from map_creation.v2_playability_priority_metrics import axis_change_events
    events = axis_change_events([note(i, i, d=d) for i, d in enumerate([0, 0, 0, 4, 0])])
    assert events[-1]['components']['frequency'] > 0
    assert events[-1]['components']['irregularity'] == 0


def test_outer_dot_has_placement_cost_but_no_preparation_endpoint():
    from map_creation.v2_playability_priority_metrics import preparation_events
    event = preparation_events([note(0, 0, x=0, hand=0, d=8)])[0]
    assert not event['eligible'] and event['excursion'] == 0
    assert event['components']['specific'] > 0


def test_crossing_volume_uses_followthrough_without_standalone_penalty():
    from map_creation.v2_playability_priority_metrics import crossing_events
    pair = crossing_events([note(0, 0, 0, 1, 0, 3), note(1, 0, 1, 1, 1, 8)])[0]
    assert pair['detected'] and pair['hard']


def test_crossing_dot_uses_previous_cut_to_construct_its_swing():
    from map_creation.v2_playability_priority_metrics import crossing_events
    # Prior up cut at (0, 1) leaves an exit at (0, 2). The dot at (1, 2)
    # is approached horizontally; its swept path touches the other hand.
    notes = [note(0, 0, 0, 1, 0, 0), note(1, .5, 1, 2, 0, 8),
             note(2, .5, 2, 2, 1, 8)]
    pair = next(e for e in crossing_events(notes) if e['ids'] == [1, 2])
    assert pair['hard'] and pair['distance'] == 0


def test_stacked_excursions_do_not_inflate_later_repetition():
    from map_creation.v2_playability_priority_metrics import preparation_events
    single = [note(0, 0, 0, 1, d=3), note(2, .1, 0, 1, d=3)]
    stack = [note(0, 0, 0, 1, d=3), note(1, 0, 0, 1, d=3), note(2, .1, 0, 1, d=3)]
    assert preparation_events(single)[-1]['total'] == preparation_events(stack)[-1]['total']


def test_review_axis_example_and_per_hand_totals():
    from map_creation.v2_playability_priority_metrics import priority_metrics
    changes = [0, 45, 0, 90, 0, 0, 45, 0, 0, 0, 45, 0, 45, 0, 0]
    axes, current = [], 0
    for change in changes:
        current = (current + change // 45) % 4
        axes.append([3, 5, 0, 4][current])
    result = priority_metrics([note(i, i, d=d) for i, d in enumerate(axes)])['axis']
    assert [e['magnitude_degrees'] for e in result['events']] == changes
    assert result['per_hand']['0']['total'] == result['total']
    assert result['per_hand']['1']['eligible'] == 0


def test_axis_irregularity_is_local_and_slow_changes_are_cheap():
    from map_creation.v2_playability_priority_metrics import axis_change_events
    def last(intervals):
        ds = [0]
        for index, interval in enumerate(intervals):
            ds.extend([ds[-1]] * (interval - 1))
            ds.append(4 if index % 2 == 0 else 0)
        return axis_change_events([note(i, i, d=d) for i, d in enumerate(ds)])[-1]
    assert last([1, 2, 1])['components']['irregularity'] > last([1, 20, 30])['components']['irregularity']
    event = last([1, 2, 1])
    assert event['interval_notes'] == 1 and event['previous_change_time'] is not None
    assert event['previous_id'] is not None and event['interval_seconds'] == 1


def test_crossing_roundoff_and_near_simultaneous_stack():
    from map_creation.v2_playability_priority_metrics import crossing_events
    pair = crossing_events([note(0, 1, 2, 1, 0), note(1, 1.25 + 1e-12, 1, 1, 1, 2)])[0]
    assert pair['hard']
    pair = crossing_events([note(0, 1, 2, 1, 0), note(1, 2.5 + 1e-12, 1, 1, 1, 2)])[0]
    assert pair['detected'] and pair['total'] == 0
    notes = [note(0, 0, 0, 1, 0), note(1, 0, 1, 1, 0),
             note(2, 5e-8, 2, 1, 0), note(3, 5e-8, 3, 1, 1)]
    pair = next(e for e in crossing_events(notes) if e['ids'] == [2, 3])
    assert pair['predecessor_ids'] == [None, None]


def test_irregularity_decays_with_time_between_changes():
    from map_creation.v2_playability_priority_metrics import axis_change_events
    ds = [0, 4, 4, 0, 4]
    def cost(step):
        return axis_change_events([note(i, i*step, d=d) for i, d in enumerate(ds)])[-1]['components']['irregularity']
    assert cost(5) < cost(.1)


def test_repeated_specific_preparation_compounds_without_excursion():
    from map_creation.v2_playability_priority_metrics import preparation_events
    repeated = [note(i, .1*i, x=1, y=1, d=3) for i in range(3)]
    event = preparation_events(repeated)[-1]
    assert event['excursion'] == 0 and event['components']['specific'] > 0
    assert event['components']['repetition'] > 0


def test_stacked_members_share_previous_group_context_independent_of_id_order():
    from map_creation.v2_playability_priority_metrics import preparation_events
    def events(swap):
        return preparation_events([note(0, 0, x=1, d=3),
            note(2 if swap else 1, .1, x=1, d=3),
            note(1 if swap else 2, .1, x=0, d=3), note(3, .2, x=1, d=3)])
    a, b = events(False), events(True)
    for left, right in ((a[1], b[2]), (a[2], b[1])):
        assert left['components'] == right['components']
        assert left['components']['speed'] > 0 and left['components']['repetition'] > 0
    assert a[-1]['components'] == b[-1]['components']


def test_outer_dot_placement_has_own_eligible_denominator():
    from map_creation.v2_playability_priority_metrics import priority_metrics
    result = priority_metrics([note(0, 0, x=0, d=8), note(1, 1, x=3, hand=1, d=8)])['preparation']
    assert result['preparation_eligible'] == 0
    assert result['placement_eligible'] == 2
    assert all(e['placement_eligible'] and not e['preparation_eligible'] for e in result['events'])
    assert all(e['components']['placement'] > 0 for e in result['events'])


def test_cross_hand_interleaving_cannot_split_a_same_hand_onset():
    from map_creation.v2_playability_priority_metrics import crossing_events, STACK_EPS
    notes = [note(0, 0, x=3, hand=1, d=0),
             note(1, .9*STACK_EPS, x=0, hand=0, d=0),
             note(2, 1.8*STACK_EPS, x=1, hand=0, d=0)]
    events = crossing_events(notes)
    pair = next(e for e in events if e['ids'] == [2, 0])
    assert pair['predecessor_ids'][0] is None


def test_anchored_same_hand_chain_keeps_distinct_later_group():
    from map_creation.v2_playability_priority_metrics import crossing_events, axis_change_events, STACK_EPS
    notes = [note(0, 0, x=0, d=0), note(1, .9*STACK_EPS, x=1, d=4),
             note(2, 1.8*STACK_EPS, x=2, d=0), note(3, .1, x=2, d=4),
             note(4, .1, x=3, hand=1, d=0)]
    axis = axis_change_events(notes)
    assert next(e for e in axis if e['id'] == 3)['eligible']
    assert next(e for e in axis if e['id'] == 3)['previous_id'] == 2
    pair = next(e for e in crossing_events(notes) if e['ids'] == [3, 4])
    assert pair['predecessor_ids'][0] == 2


def test_saved_disjoint_paths_do_not_hard_fail():
    from pathlib import Path
    from map_creation.v2_playability_priority_metrics import crossing_events
    saved = json.loads(Path('.superpowers/sdd/2026-11-08-priority-playability/cell-overlap-disjoint-path-example.json').read_text())
    pair = next(e for e in crossing_events(saved['notes'] + saved['predecessors']) if e['ids'] == [5, 6])
    assert not pair['hard'] and not pair['detected']
    assert pair['distance'] > 0


@pytest.mark.parametrize('a,b,hit', [
    ([[0, 0], [1, 0]], [[.999999, -1], [.999999, 1]], True),
    ([[0, 0], [1, 0]], [[1, 0], [2, 1]], True),
    ([[0, 0], [2, 0]], [[1, 0], [3, 0]], True),
    ([[0, 0], [1, 0]], [[.5, .1], [1.5, .1]], False),
    ([[0, 0], [1, 0]], [[0, 4], [1, 4]], False),
    ([[0, 0], [0, 0]], [[-1, 0], [1, 0]], True),
])
def test_pair_geometry_actual_intersections(a, b, hit):
    from map_creation.v2_playability_priority_metrics import crossing_pair_event
    event = crossing_pair_event(note(0, 0), note(1, 0, hand=1), [a], [b])
    assert event['detected'] is hit and event['hard'] is hit
    assert event['distance'] == pytest.approx(0) if hit else event['distance'] > 0


def test_near_miss_is_soft_but_far_path_has_no_cost():
    from map_creation.v2_playability_priority_metrics import crossing_pair_event
    a = note(0, 0)
    b = note(1, 0, hand=1)
    near = crossing_pair_event(a, b, [[[0, 0], [1, 0]]], [[[0, .1], [1, .1]]])
    far = crossing_pair_event(a, b, [[[0, 0], [1, 0]]], [[[0, 4], [1, 4]]])
    assert 0 < near['total'] and not near['hard']
    assert far['total'] == 0 and not far['detected']


def test_note_paths_preserve_unclamped_endpoints_and_predecessor():
    from map_creation.v2_playability_priority_metrics import note_paths
    paths = note_paths([note(0, 0, 0, 1, d=2), note(1, 1.5, 0, 1, d=3)])
    assert paths[0]['segments'] == [[[1, 1], [-1, 1]]]
    assert paths[1]['predecessor_id'] == 0
    assert paths[1]['segments'][0] == [[-1, 1], [-1, 1]]


def test_path_lookback_limit_and_stack_share_distinct_predecessor():
    from map_creation.v2_playability_priority_metrics import note_paths, STACK_EPS
    notes = [note(0, 0, 0, 1), note(1, 1.5, 1, 1),
             note(2, 1.5 + .5*STACK_EPS, 2, 1), note(3, 3.01, 1, 1)]
    paths = note_paths(notes)
    assert paths[1]['predecessor_id'] == paths[2]['predecessor_id'] == 0
    assert len(paths[1]['segments']) == 2
    assert paths[3]['predecessor_id'] == 2 and len(paths[3]['segments']) == 1


def test_intersection_symmetry_and_ineligible_pair():
    from map_creation.v2_playability_priority_metrics import crossing_pair_event
    a, b = note(0, 0), note(1, 1.501, hand=1)
    path_a, path_b = [[[0, 0], [2, 1]]], [[[0, 1], [2, 0]]]
    event = crossing_pair_event(a, b, path_a, path_b)
    reverse = crossing_pair_event(b, a, path_b, path_a)
    assert event['detected'] and reverse['detected']
    assert not event['eligible'] and not event['hard'] and event['total'] == 0
