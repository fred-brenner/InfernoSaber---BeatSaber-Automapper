"""Priority policy preserves rhythm and repairs hard path/preparation faults."""
from copy import deepcopy
import importlib
import importlib.util
import json
import math

import pytest

from map_creation import v2_postprocessing as post
from map_creation.v2_playability_priority_metrics import crossing_events, preparation_events
from tools.config import config


def policy():
    name = 'map_creation.v2_playability_priority'
    assert importlib.util.find_spec(name) is not None, 'priority repair is missing'
    return importlib.import_module(name)


def note(identity, time, hand, x, y, direction, **extra):
    return dict(id=identity, time=time, hand=hand, x=x, y=y, d=direction,
                custom={'identity': identity}, **extra)


def signature(notes):
    return [{k: v for k, v in n.items() if k not in ('x', 'y', 'd')} for n in notes]


def hard_events(notes):
    return ([e for e in preparation_events(notes) if e['hard']],
            [e for e in crossing_events(notes) if e['hard']])


@pytest.fixture(autouse=True)
def strict_flow(monkeypatch):
    monkeypatch.setattr(config, 'v2_flow_min_turn_steps', 3)
    monkeypatch.setattr(config, 'min_time_diff', .02)
    monkeypatch.setattr(config, 'v2_cleanup_same_dir_gap', .1)
    monkeypatch.setattr(config, 'v2_cleanup_swing_window', .15)
    monkeypatch.setattr(config, 'v2_fast_counter_direction', 'recut')
    monkeypatch.setattr(config, 'v2_fast_gap_seconds', .2)


def test_comfortable_unchanged_axis_stream_is_preserved():
    # Ordinary fast reversals and a long run are not repair targets.
    notes = [note(i, i*.3, 0, (1, 2)[(i//2) % 2], (1, 0, 2, 1)[i % 4], i % 2) for i in range(18)]
    before = deepcopy(notes)
    assert hard_events(notes) == ([], [])
    result = policy().repair_priority(notes, lambda n: math.inf)
    assert notes == before
    json.dumps(result, allow_nan=False)


def test_forbidden_preparation_repaired_without_changing_identity_or_timing():
    notes = [note(8, 1., 0, 3, 1, 2), note(9, 3., 1, 2, 1, 0)]
    before = deepcopy(notes)
    assert hard_events(notes)[0]
    policy().repair_priority(notes, lambda n: math.inf)
    assert signature(notes) == signature(before)
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())


def test_fast_crossing_is_repaired_using_real_paths():
    notes = [note(0, 1., 0, 1, 1, 3), note(1, 1., 1, 2, 1, 2)]
    before = deepcopy(notes)
    assert hard_events(notes)[1]
    policy().repair_priority(notes, lambda n: math.inf)
    assert signature(notes) == signature(before)
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())


def test_immutable_hard_fault_fails_and_rolls_back():
    notes = [note(0, 1., 0, 3, 1, 2, kind='retained')]
    before = deepcopy(notes)
    assert hard_events(notes)[0]
    with pytest.raises((ValueError, RuntimeError)):
        policy().repair_priority(notes, lambda n: math.inf, protect_retained=True)
    assert notes == before


def test_repeated_awkward_preparation_improves_locally():
    notes = [note(i, i*.3, 0, 0, 1, i % 2) for i in range(10)]
    from map_creation.v2_playability_placement import placement_costs
    before = sum(e['total'] for e in preparation_events(notes))
    placement_before = placement_costs(notes)['total_excess']
    policy().repair_priority(notes, lambda n: math.inf)
    assert sum(e['total'] for e in preparation_events(notes)) < before
    assert placement_costs(notes)['total_excess'] < placement_before
    assert hard_events(notes) == ([], [])


def test_frequent_axis_churn_is_reduced_without_forcing_variety():
    from map_creation.v2_playability_priority_metrics import axis_change_events
    notes = [note(i, i*.3, 0, 1, 1, (0, 7)[i % 2]) for i in range(10)]
    before = sum(e['components']['frequency'] + e['components']['irregularity']
                 for e in axis_change_events(notes))
    assert before > 0
    policy().repair_priority(notes, lambda n: math.inf)
    after = sum(e['components']['frequency'] + e['components']['irregularity']
                for e in axis_change_events(notes))
    assert after < before
    assert hard_events(notes) == ([], [])


def test_saved_r7_passage_clears_hard_findings_with_identity_and_caps():
    from pathlib import Path
    data = json.loads((Path(__file__).parent / 'fixtures/v2_r7_horizontal_passage.json').read_text())
    notes = deepcopy(data['notes'])
    before_signature = signature(notes)
    baseline_clearance = post.clearance_violations(notes)
    assert hard_events(notes)[1]
    result = policy().repair_priority(notes, lambda n: data['arrival_caps'][str(n['id'])])
    assert signature(notes) == before_signature
    assert hard_events(notes) == ([], [])
    assert result['priority_preserved'] and result['priority_hard_after'] == 0
    assert all(post.clearance_violations(notes)[key] <= value
               for key, value in baseline_clearance.items())


def test_preexisting_speed_exception_cannot_get_worse():
    from map_creation.sanity_check import calc_note_speed
    notes = [note(0, 0., 0, 1, 1, 0), note(1, 1., 0, 3, 1, 2)]
    def speed():
        return calc_note_speed([notes[0][k] for k in ('x', 'y', 'hand', 'd')],
                               [notes[1][k] for k in ('x', 'y', 'hand', 'd')], 1.)
    before = speed()
    assert before > .5
    policy().repair_priority(notes, lambda n: .5)
    assert speed() <= before + 1e-9
    assert hard_events(notes) == ([], [])


def test_issue_after_sixty_four_notes_is_not_skipped():
    notes = [note(i, i*2., 0, 1, 1, i % 2, kind='retained') for i in range(70)]
    notes.append(note(70, 142., 0, 3, 1, 2))
    original = deepcopy(notes)
    policy().repair_priority(notes, lambda n: math.inf, protect_retained=True)
    assert notes[:70] == original[:70]
    assert hard_events(notes) == ([], [])


def test_dots_and_simultaneous_stack_keep_identity_and_direction_kind():
    notes = [note(0, 0., 0, 1, 1, 0, kind='retained'),
             note(1, 2., 0, 0, 0, 8), note(2, 2., 0, 2, 0, 8),
             note(3, 4., 0, 3, 1, 2)]
    before = deepcopy(notes)
    policy().repair_priority(notes, lambda n: math.inf, protect_retained=True)
    assert notes[0] == before[0]
    assert signature(notes) == signature(before)
    assert [n['d'] == 8 for n in notes] == [n['d'] == 8 for n in before]
    assert hard_events(notes) == ([], [])


def test_irregular_dense_arrows_support_joint_sequence_repair():
    from map_creation.v2_playability_priority_metrics import axis_change_events
    directions = [0, 6, 3, 6, 3, 4, 3, 2, 5]
    notes = [note(i, i*.3, 0, 1, 1, d) for i, d in enumerate(directions)]
    before = sum(e['total'] for e in axis_change_events(notes))
    signature_before = signature(notes)
    result = policy().repair_priority(notes, lambda n: math.inf)
    assert result['soft_joint_changed'] >= 2
    assert sum(e['total'] for e in axis_change_events(notes)) < before
    assert signature(notes) == signature_before
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())


def test_visibility_tail_is_actively_refined():
    from map_creation.v2_playability_costs import occlusion_severity
    notes = [note(0, 0., 0, 1, 0, 1), note(1, .3, 1, 1, 2, 0),
             note(2, 10., 0, 1, 1, 0)]
    before = occlusion_severity(notes[1], notes[0], .3)
    policy().repair_priority(notes, lambda n: math.inf)
    assert occlusion_severity(notes[1], notes[0], .3) < before
    assert hard_events(notes) == ([], [])


def test_unrepairable_retained_fault_rolls_back_earlier_successful_edit():
    notes = [note(0, 0., 0, 3, 1, 2, kind='retained'),
             note(1, 3., 1, 0, 1, 3)]
    before = deepcopy(notes)
    with pytest.raises((ValueError, RuntimeError)):
        policy().repair_priority(notes, lambda n: math.inf, protect_retained=True)
    assert notes == before


def test_repair_is_deterministic_for_reordered_input():
    notes = [note(0, 1., 0, 1, 1, 3), note(1, 1., 1, 2, 1, 2),
             note(2, 3., 0, 3, 1, 2)]
    first, second = deepcopy(notes), list(reversed(deepcopy(notes)))
    policy().repair_priority(first, lambda n: math.inf)
    policy().repair_priority(second, lambda n: math.inf)
    assert sorted(first, key=lambda n: n['id']) == sorted(second, key=lambda n: n['id'])


def test_isolated_slow_crossing_is_actively_improved():
    notes = [note(0, 0., 0, 1, 1, 2), note(1, .5, 1, 2, 1, 3)]
    before = sum(e['total'] for e in crossing_events(notes))
    assert before > 0 and not hard_events(notes)[1]
    policy().repair_priority(notes, lambda n: math.inf)
    assert sum(e['total'] for e in crossing_events(notes)) < before - .1
    assert hard_events(notes) == ([], [])


def test_exhausted_cluster_budget_is_explicit_and_atomic(monkeypatch):
    module = policy()
    monkeypatch.setattr(module, 'HARD_CLUSTER_BUDGET', 0)
    notes = [note(0, 1., 0, 1, 1, 3), note(1, 1., 1, 2, 1, 2)]
    before = deepcopy(notes)
    with pytest.raises(ValueError, match='budget'):
        module.repair_priority(notes, lambda n: math.inf)
    assert notes == before


def test_soft_deltas_match_full_events_beyond_four_seconds_and_straddling_pairs():
    from map_creation.v2_playability_priority_metrics import priority_metrics
    notes = [note(i, t, h, x, y, d) for i, (t, h, x, y, d) in enumerate([
        (0., 0, 0, 1, 1), (2., 0, 0, 1, 0), (4.5, 0, 1, 1, 7),
        (5., 1, 3, 1, 0), (8.7, 0, 1, 1, 0), (9.2, 1, 2, 1, 1),
        (12.5, 0, 1, 1, 7), (16.6, 0, 1, 1, 0)])]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    edits = {state.index[2]: (1, 1, 1)}
    trial = state.trial(edits)
    assert trial is not None
    before = priority_metrics(notes)
    deltas = state.soft_deltas(edits, trial)
    state.state.install(edits)
    try:
        after = priority_metrics(notes)
    finally:
        state.state.install({state.index[2]: (1, 1, 7)})
    for category in ('preparation', 'axis', 'crossing'):
        key = (lambda e: tuple(e['ids'])) if category == 'crossing' else (lambda e: e['id'])
        old = {key(e): e['total'] for e in before[category]['events']}
        new = {key(e): e['total'] for e in after[category]['events']}
        changed = {k for k in old if abs(new[k] - old[k]) > 1e-9}
        assert changed <= set(deltas[category])
        for k in changed:
            assert deltas[category][k] == pytest.approx((old[k], new[k]))


def test_travel_breaks_only_equivalent_primary_and_medium_scores():
    better = policy()._soft_improves
    baseline = (2., 1., .5, 1., 5.)
    assert better((2., 1., .5, 1., 4.), baseline)
    assert not better((2., 1., .5, 1., 6.), baseline)
    assert not better((2.1, 1., .5, .8, 0.), baseline)


def test_final_speed_limits_receipt_includes_baseline_allowances_and_final_cells():
    notes = [note(0, 0., 0, 1, 1, 0), note(1, 1., 0, 3, 1, 2)]
    result = policy().repair_priority(notes, lambda n: .5)
    entries = {e['id']: e for e in result['priority_speed_limits']}
    assert entries[1]['requested_cap'] == .5
    assert entries[1]['effective_cap'] > .5
    assert all((entries[n['id']]['x'], entries[n['id']]['y'], entries[n['id']]['d']) ==
               (n['x'], n['y'], n['d']) for n in notes)
    json.dumps(result, allow_nan=False)


def test_incremental_soft_deltas_match_full_audit_for_stacks_dots_and_joint_edits():
    from map_creation.v2_playability_priority_metrics import priority_metrics
    notes = [note(i, t, h, x, y, d) for i, (t, h, x, y, d) in enumerate([
        (0., 0, 0, 1, 0), (.1, 1, 3, 1, 1), (2., 0, 1, 1, 8),
        (2., 0, 2, 2, 8), (4.5, 0, 1, 1, 1), (5.0, 1, 2, 1, 0),
        (8., 0, 1, 1, 0), (9., 1, 3, 1, 1), (13., 0, 1, 1, 1)])]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    before = priority_metrics(notes)
    physical_before = policy()._clearance_findings(notes)
    proposals = [{state.index[1]: (3, 2, 1)},
                 {state.index[1]: (3, 2, 1), state.index[5]: (2, 2, 0)}]
    checked = 0
    for edits in proposals:
        trial = state.trial(edits)
        if trial is None:
            continue
        checked += 1
        deltas = state.soft_deltas(edits, trial)
        original = {i: (state.notes[i]['x'], state.notes[i]['y'], state.notes[i]['d']) for i in edits}
        state.state.install(edits)
        try:
            after = priority_metrics(notes)
            from map_creation.v2_playability_priority_metrics import note_paths
            full_paths = note_paths(notes)
            physical_after = policy()._clearance_findings(notes)
            assert all(physical_after[k] <= physical_before[k] for k in physical_after)
            for identity, path in trial[0].items():
                assert path == full_paths[identity]
            full_pairs = {tuple(e['ids']): e for e in after['crossing']['events']}
            for key, event in trial[1].items():
                assert event == full_pairs[key]
        finally:
            state.state.install(original)
        for category in ('preparation', 'axis', 'crossing'):
            key = (lambda e: tuple(e['ids'])) if category == 'crossing' else (lambda e: e['id'])
            old = {key(e): e['total'] for e in before[category]['events']}
            new = {key(e): e['total'] for e in after[category]['events']}
            changed = {k for k in old if abs(new[k]-old[k]) > 1e-9}
            assert changed <= set(deltas[category])
            for k in changed:
                assert deltas[category][k] == pytest.approx((old[k], new[k]))
    assert checked == len(proposals)


def test_slow_crossing_near_end_of_window_can_improve():
    notes = [note(0, 0., 0, 1, 1, 2), note(1, 1.49, 1, 2, 1, 3)]
    before = sum(e['total'] for e in crossing_events(notes))
    assert before > 0
    policy().repair_priority(notes, lambda n: math.inf)
    assert sum(e['total'] for e in crossing_events(notes)) < before - 1e-6


def test_existing_clearance_exception_identity_is_preserved():
    notes = [note(0, 0., 0, 1, 1, 0), note(1, 0., 0, 1, 1, 1),
             note(2, 4., 0, 3, 1, 2)]
    before = policy()._clearance_findings(notes)
    assert before['same_cell_pairs'] == {(0, 1)}
    policy().repair_priority(notes, lambda n: math.inf)
    after = policy()._clearance_findings(notes)
    assert all(after[k] <= before[k] for k in before)
    assert hard_events(notes) == ([], [])


def test_retained_crossing_can_jointly_repair_both_mutable_predecessors():
    notes = [note(0, 0., 0, 1, 2, 1, kind='new'),
             note(1, .03, 1, 3, 1, 3, kind='new'),
             note(2, .4, 0, 3, 1, 0, kind='retained'),
             note(3, .42, 1, 2, 0, 4, kind='retained')]
    before = deepcopy(notes)
    search = policy()._PrioritySearch(notes, lambda n: 44.16, True)
    assert search.hard_pairs and not search.prep_hard
    for i in (0, 1):
        for option in search.state.options(i, broad=True):
            trial = search.trial({i: option})
            assert trial is None or trial[2] or trial[3]
    policy().repair_priority(notes, lambda n: 44.16, protect_retained=True)
    assert notes[2:] == before[2:]
    assert all(notes[i] != before[i] for i in (0, 1))
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())


def test_soft_guard_rejects_new_locally_severe_preparation_event():
    notes = [note(i, i*.3, 0, 0, 1, i % 2) for i in range(12)]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    edits = {state.index[7]: (0, 1, 7)}
    trial = state.trial(edits)
    assert trial is not None and not trial[2] and not trial[3]
    deltas = state.soft_deltas(edits, trial)
    assert deltas['preparation'][7][1] > policy()._soft_bounds()['preparation']
    assert not state.soft_allowed(edits, trial)


def test_isolated_preparation_ban_can_repair_jointly_with_neighbor():
    rows = [(676,75.91764,1,3,1,5), (677,76.04535,0,1,0,0), (678,76.17039,1,3,0,6),
            (679,76.24851,0,2,1,1), (680,76.40476,0,0,2,4),
            (681,76.48288,1,0,0,3), (682,76.63913,1,0,0,2),
            (683,76.71725,0,3,1,1), (684,76.79538,1,2,0,7),
            (685,76.87350,0,1,2,4), (686,76.95162,1,1,0,0),
            (687,77.02975,0,0,2,3)]
    notes = [note(*row) for row in rows]
    search = policy()._PrioritySearch(notes, lambda n: 44.16, False)
    assert search.prep_hard == {681} and not search.hard_pairs
    for option in search.state.options(search.index[681], broad=True):
        trial = search.trial({search.index[681]: option})
        assert trial is None or trial[2] or trial[3]
    before = signature(notes)
    policy().repair_priority(notes, lambda n: 44.16)
    assert hard_events(notes) == ([], [])
    assert signature(notes) == before


def test_tiny_slow_crossing_gain_cannot_buy_large_preparation_harm():
    before = (0., 0., .008, 0., 0.)
    assert not policy()._soft_improves((9., 0., 0., 0., 0.), before)
    assert policy()._soft_improves((0., 0., 0., 0., 0.), before)


def test_joint_edit_accounts_for_slow_pair_outside_other_notes_time_window():
    notes = [note(0, 0., 0, 1, 1, 2), note(1, .5, 1, 2, 1, 3),
             note(2, 10., 0, 1, 1, 3)]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    edits = {state.index[0]: (1, 0, 2), state.index[2]: (1, 2, 3)}
    trial = state.trial(edits)
    assert trial is not None
    deltas = state.soft_deltas(edits, trial)
    assert (0, 1) in deltas['crossing']
    assert deltas['crossing'][(0, 1)][1] < deltas['crossing'][(0, 1)][0]


def test_saved_angry_too_crossing_resolves_with_bounded_same_hand_sequence(monkeypatch):
    # Captured pre-policy seed 3465051232, after the hard search stalls.
    rows = [(707,108.00337510547205,0,1,2,1),(708,108.2398185941043,0,0,1,0),
            (709,108.2398185941043,1,2,1,1),(710,108.37838682458826,0,0,1,1),
            (711,108.37838682458826,1,3,1,0),(712,108.56589268414638,0,1,1,5),
            (713,108.75339854370449,0,2,1,1),(714,108.75339854370449,1,3,0,6),
            (716,108.87840245007656,1,1,1,0),(718,108.99446712018141,1,1,2,6),
            (719,109.06590830963468,0,2,2,4),(720,109.12841026282071,1,1,1,3),
            (721,109.22216319259977,0,0,0,7),(722,109.31591612237882,1,3,1,2),
            (723,109.40966905215788,0,1,2,0),(724,109.50342198193694,1,3,0,3),
            (725,109.56335600907029,0,0,1,6),(728,109.71771439286049,1,1,0,2),
            (731,109.87843370105315,0,0,1,5),(732,109.93487528344671,1,1,0,3)]
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_late_turns', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    notes = [note(*row) for row in rows]
    before = deepcopy(notes)
    search = policy()._PrioritySearch(notes, lambda n: 44.16, False)
    search.state.diagonal_limit = 579
    assert (719, 718) in search.hard_pairs
    result = search._sequence_escape((719, 718), hand=1, radius=3)
    assert result is not None
    edits, trial = result
    search.commit(edits, trial)
    assert (719, 718) not in search.hard_pairs
    assert signature(notes) == signature(before)
    assert not any(post.clearance_violations(notes).values())
    assert not search.state.speed_groups.violations(range(len(notes)))


def test_saved_angry_too_crossing_accepts_atomic_two_phrase_bridge(monkeypatch):
    rows = [(379,52.31413481671302,1,1,0,1),(380,52.40788774649208,0,0,2,0),
            (381,52.50164067627113,1,3,1,0),(382,52.55836734693877,0,0,1,1),
            (383,52.60878688173291,1,3,1,6),(386,52.76950618992558,0,0,1,5),
            (387,52.82307929265647,1,2,1,3),(388,52.87665239538735,0,0,1,6),
            (389,52.92988662131519,1,3,2,2),(390,53.001656301759425,0,0,1,5),
            (392,53.06415825494547,1,3,0,3),(394,53.18916216131754,0,1,1,1),
            (397,53.31301587301587,0,1,2,0),(398,53.37666802087565,1,2,0,6),
            (399,53.43916997406169,0,2,1,7),(400,53.50167192724773,1,0,1,0),
            (402,53.62648526077098,1,2,0,7),(405,53.777414965986395,1,1,1,0),
            (408,53.92668520891278,1,2,0,7),(409,54.00168755273602,0,2,2,4),
            (410,54.09544048251508,1,3,2,4),(411,54.18919341229413,0,1,0,1),
            (412,54.28294634207319,1,3,0,1),(413,54.376699271852246,0,0,2,0),
            (414,54.470452201631296,1,1,2,0),(415,54.564205131410354,0,0,1,1),
            (416,54.564205131410354,1,2,1,1)]
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_late_turns', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    notes = [note(*row) for row in rows]
    before = deepcopy(notes)
    search = policy()._PrioritySearch(notes, lambda n: 44.16, False)
    search.state.diagonal_limit = 579
    assert (399, 398) in search.hard_pairs
    original_hard = set(search.hard_pairs)
    result = search._sequence_bridge((399, 398))
    assert result is not None
    assert notes == before  # Both proposed phrases stay speculative until commit.
    edits, trial = result
    assert len(trial[2]) < len(original_hard) and not trial[3]
    search.commit(edits, trial)
    assert not any(post.clearance_violations(notes).values())
    assert not search.state.speed_groups.violations(range(len(notes)))
    assert signature(notes) == signature(before)


def test_small_isolated_axis_and_slow_path_trade_can_fix_large_preparation():
    # A small first axis change and a slow-path penalty are soft costs.
    before = (10., 0., 0., 0., 0.)
    after = (4., .25, .1, 0., 0.)
    assert policy()._soft_improves(after, before)
    # The reverse trade must not use a tiny slow-path gain to create bad reach.
    assert not policy()._soft_improves((9., 0., 0., 0., 0.), (0., 0., .008, 0., 0.))


def test_soft_guard_cannot_add_specific_inner_inward_tail():
    lanes = [0, 2, 2, 0, 1, 2, 1, 2, 2, 0]
    notes = [note(i, i*.3, 0, x, 1, i % 2) for i, x in enumerate(lanes)]
    search = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    edits = {search.index[4]: (1, 1, 5)}
    trial = search.trial(edits)
    assert trial is not None and not trial[2] and not trial[3]
    assert not search.soft_allowed(edits, trial)


def test_recovery_travel_is_subordinate_to_large_preparation_gain():
    before = (10., 0., 0., 0., 1.)
    after = (0., 0., 0., 0., 100.)
    assert policy()._soft_improves(after, before)


def test_small_distributed_slow_penalty_can_trade_for_strong_reach_gain():
    # Real BWB candidate had -17.7 preparation, +0.45 axis and +0.86 slow path.
    assert policy()._soft_improves((2.3, .45, .86, 0., 0.), (20., 0., 0., 0., 0.))


def test_saved_bwb_reach_trap_can_repair_joint_cells_and_arrows(monkeypatch):
    # R3 priority post-policy passage; all affected recorded caps are 24.
    rows = [(298,103.72353741496599,1,2,1,1),(299,103.82802721088436,0,0,1,3),
            (300,103.94412698412698,1,2,1,0),(301,104.06022675736962,0,0,1,2),
            (302,104.16471655328799,1,1,0,6),(303,104.28081632653061,0,0,1,3),
            (304,104.38530612244898,1,2,0,5),(305,104.5014058956916,0,0,1,6),
            (306,104.61750566893424,1,3,1,6),(307,104.72199546485261,0,0,0,5),
            (308,104.83809523809524,1,2,0,5),(310,105.04358191164418,0,1,1,6),
            (311,105.04358191164418,1,3,1,1),(313,105.22237949787676,1,3,1,0),
            (315,105.40117708410935,0,1,1,0),(316,105.40117708410935,1,2,0,1)]
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_late_turns', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    notes = [note(*row) for row in rows]
    original = deepcopy(notes)
    search = policy()._PrioritySearch(notes, lambda n: 24., False)
    assert not search.hard_pairs and not search.prep_hard
    before = sum(e['total'] for e in search.prep_events.values())
    changed = search._repair_joint_reach({306})
    assert changed >= 2
    assert sum(e['total'] for e in search.prep_events.values()) < before
    assert signature(notes) == signature(original)
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())
    assert not search.state.speed_groups.violations(range(len(notes)))


def test_outer_lane_run_can_be_recentered_as_one_coherent_phrase():
    notes = [note(i, i*.35, 0, 0, 1, (3, 2)[i % 2]) for i in range(4)]
    original = deepcopy(notes)
    search = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    before = sum(e['total'] for e in search.prep_events.values())
    assert search._repair_reach_runs() >= 3
    assert sum(e['total'] for e in search.prep_events.values()) < before
    assert sum(n['x'] == 0 for n in notes) < 4
    assert hard_events(notes) == ([], [])
    assert signature(notes) == signature(original)
    assert not any(post.clearance_violations(notes).values())


def test_phrase_can_spend_several_small_axis_changes_for_large_reach_gain():
    before = (30., 0., 0., 0., 0.)
    after = (10., 1., 1.5, 0., 0.)
    assert policy()._soft_improves(after, before, edit_count=4)
    assert not policy()._soft_improves(after, before, edit_count=1)


def test_visibility_guard_rejects_local_tail_even_when_primary_cost_improves():
    from map_creation.v2_playability_visibility import visibility_costs
    notes = [note(i, i*.15, 0, 1, 2 if i < 4 else 1, (2, 3)[i % 2])
             for i in range(10)]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    edits = {state.index[4]: (1, 2, 2)}
    before = visibility_costs(notes, state.nps)
    assert before['flagged_events'] > 0
    assert not state.visibility_allowed(edits)
    # This guard is independent of _soft_improves' primary-gain early return.
    assert not state.soft_allowed(edits, (None, {}, None, None),
                                  {'preparation': {}, 'axis': {}, 'crossing': {}})


def test_visibility_cached_deltas_match_full_events_for_stacks_and_joint_edits():
    from map_creation.v2_playability_visibility import visibility_costs
    notes = [note(i, t, hand, x, y, d) for i, (t, hand, x, y, d) in enumerate([
        (0., 0, 1, 2, 8), (.01, 0, 0, 1, 8), (.2, 0, 1, 1, 2),
        (.35, 1, 2, 2, 0), (.4, 0, 1, 2, 3), (.6, 0, 1, 1, 2),
        (.8, 0, 2, 2, 3), (1., 0, 0, 1, 2), (1.2, 0, 1, 2, 3),
        (1.8, 0, 1, 1, 2), (2., 0, 1, 1, 3)])]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    edits = {state.index[0]: (1, 1, 8), state.index[5]: (1, 2, 2)}
    before = {e['id']: e['severity'] for e in visibility_costs(notes, state.nps)['events']}
    deltas = state.visibility_deltas(edits)
    original = {i: state.state.committed[i] for i in edits}
    state.state.install(edits)
    try:
        after = {e['id']: e['severity'] for e in visibility_costs(notes, state.nps)['events']}
    finally:
        state.state.install(original)
    changed = {identity for identity in before if abs(before[identity]-after[identity]) > 1e-9}
    assert changed <= set(deltas)
    for identity in changed:
        assert deltas[identity] == pytest.approx((before[identity], after[identity]))


def test_dedicated_visibility_repair_reduces_sustained_upper_center_excess():
    from map_creation.v2_playability_visibility import visibility_costs
    notes = [note(i, i*.2, 0, 1, 2, (2, 3)[i % 2]) for i in range(10)]
    original = deepcopy(notes)
    state = policy()._PrioritySearch(notes, lambda n: 44.16, False)
    before = visibility_costs(notes, state.nps)
    assert before['flagged_events'] > 0
    assert state.repair_visibility() > 0
    after = visibility_costs(notes, state.nps)
    assert after['excess'] < before['excess']
    assert signature(notes) == signature(original)
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())
    assert not state.state.speed_groups.violations(range(len(notes)))


def test_protected_visibility_budget_matches_retained_endpoint_members_in_stacks():
    notes = [note(i, i*.15, 0, 1, 2, i % 2, kind='retained') for i in range(8)]
    # A lower-ID copy becomes the group anchor; its retained partner still
    # identifies the inherited exception from the retained-only reference.
    notes.append(note(-1, 1.05, 0, 2, 2, 8, kind='copy', anchor_id=7))
    state = policy()._PrioritySearch(notes, lambda n: math.inf, True)
    endpoint = next(e for e in state.visibility_events.values() if -1 in e['group_ids'])
    assert endpoint['id'] == -1
    assert state.visibility_baseline[endpoint['id']] == pytest.approx(1.)
    assert endpoint['id'] not in state.visibility_regressions()


def test_protected_new_onset_has_no_inherited_visibility_exception():
    notes = [note(i, i*.15, 0, 1, 2, i % 2, kind='retained') for i in range(8)]
    notes.append(note(20, 1.2, 0, 1, 2, 8, kind='copy', anchor_id=7))
    state = policy()._PrioritySearch(notes, lambda n: math.inf, True)
    assert state.visibility_baseline[20] == 0.
    assert 20 in state.visibility_regressions()


def test_saved_night_upper_center_passage_clears_visibility_tail_with_real_caps(monkeypatch):
    # Frozen R4 post-policy passage, seed3328878355. Recorded requested and
    # effective caps are44.16 for every captured ID; this is not an infinite cap.
    from map_creation.v2_playability_visibility import visibility_costs
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_late_turns', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    rows = [
        (704, 71.5871201814059, 1, 3, 2, 2),
        (705, 71.70321995464853, 0, 1, 2, 6),
        (706, 71.81931972789116, 1, 3, 2, 5),
        (707, 71.94702947845805, 0, 0, 1, 3),
        (708, 72.05151927437642, 1, 2, 0, 6),
        (709, 72.10796664127653, 0, 0, 1, 2),
        (710, 72.19083900226758, 1, 3, 1, 0),
        (711, 72.28371882086168, 0, 0, 1, 3),
        (712, 72.41142857142857, 1, 2, 2, 7),
        (713, 72.5275283446712, 0, 1, 0, 2),
        (714, 72.64362811791383, 1, 2, 2, 0),
        (715, 72.7329544344837, 0, 0, 1, 5),
        (716, 72.8110779086346, 1, 2, 1, 1),
        (717, 72.88920138278549, 0, 0, 1, 6),
        (718, 72.96732485693639, 1, 2, 1, 0),
        (719, 73.04544833108729, 0, 0, 1, 5),
        (720, 73.12357180523819, 1, 2, 0, 6),
        (721, 73.22412698412698, 0, 0, 1, 6),
        (723, 73.34022675736962, 0, 1, 1, 0),
        (724, 73.43606570184176, 1, 3, 0, 3),
        (725, 73.51418917599266, 0, 1, 1, 1),
        (726, 73.58403628117914, 1, 2, 2, 4),
        (727, 73.7094978613699, 0, 1, 1, 0),
        (728, 73.8162358276644, 1, 2, 2, 7),
        (729, 73.93233560090702, 0, 1, 1, 1),
        (730, 74.04843537414966, 1, 3, 1, 0),
        (731, 74.13917696919982, 0, 1, 1, 0),
        (732, 74.21730044335072, 1, 2, 0, 1),
        (733, 74.28063492063492, 0, 1, 1, 1),
        (734, 74.39673469387755, 1, 3, 1, 0),
        (735, 74.45167086580341, 0, 1, 1, 0),
        (737, 74.52444444444444, 1, 2, 0, 1),
        (738, 74.62893424036281, 1, 2, 1, 0),
        (739, 74.76416476240699, 0, 0, 1, 6),
        (740, 74.88134997363333, 1, 2, 1, 1),
        (741, 74.99853518485968, 0, 1, 2, 5),
        (742, 75.07665865901056, 1, 3, 1, 0),
        (743, 75.15478213316146, 0, 1, 2, 2),
        (744, 75.22104308390023, 1, 2, 0, 1),
        (745, 75.33714285714285, 0, 1, 2, 3),
        (746, 75.45324263038549, 1, 2, 0, 0),
        (747, 75.58446124099139, 0, 2, 2, 4),
        (748, 75.68544217687075, 1, 2, 0, 1),
        (749, 75.80154195011337, 0, 0, 0, 7),
        (750, 75.91764172335601, 1, 3, 1, 5),
        (751, 76.0453514739229, 0, 1, 2, 2),
        (752, 76.1703872971231, 1, 3, 0, 6),
        (753, 76.248510771274, 0, 2, 2, 3),
        (755, 76.40475771957578, 0, 1, 2, 2),
        (756, 76.48288119372668, 1, 2, 0, 3),
        (759, 76.63912814202848, 1, 0, 0, 2),
        (760, 76.71725161617937, 0, 1, 2, 3),
        (761, 76.79537509033027, 1, 2, 0, 7),
        (762, 76.87349856448117, 0, 1, 2, 2),
        (763, 76.95162203863205, 1, 1, 0, 2),
        (764, 77.02974551278295, 0, 0, 2, 3),
        (765, 77.10786898693385, 1, 2, 0, 7),
        (766, 77.21795918367347, 0, 1, 2, 4),
        (767, 77.34223940938654, 1, 3, 1, 0),
        (768, 77.57660983183922, 0, 0, 0, 7),
        (769, 77.682358276644, 1, 2, 0, 1),
        (770, 77.8109802542919, 0, 1, 1, 0),
        (771, 77.91455782312926, 1, 2, 1, 0),
        (772, 78.04535067674459, 0, 1, 1, 1),
        (773, 78.27972109919729, 1, 3, 1, 1),
        (774, 78.37895691609977, 0, 1, 1, 0),
        (775, 78.43596804749907, 1, 3, 1, 5),
        (776, 78.51409152164997, 0, 1, 0, 6),
        (777, 78.63127673287632, 1, 3, 0, 6),
        (778, 78.74846194410266, 0, 1, 2, 5),
        (779, 78.82658541825356, 1, 3, 1, 5),
        (780, 78.90470889240444, 0, 1, 1, 1),
        (781, 78.97106575963718, 1, 2, 0, 1),
    ]
    notes = [note(*row) for row in rows]
    original = deepcopy(notes)
    state = policy()._PrioritySearch(notes, lambda n:44.16, False)
    before = visibility_costs(notes, state.nps)
    local_before = [e for e in before['events'] if 74<=e['time']<=78]
    assert max(e['severity'] for e in local_before) == pytest.approx(.875)
    assert sum(e['flagged'] for e in local_before) == 11
    prep_before = {e['id']:e['total'] for e in preparation_events(notes)}
    assert state.repair_visibility() > 0
    after = visibility_costs(notes, state.nps)
    local_after = [e for e in after['events'] if 74<=e['time']<=78]
    assert max(e['severity'] for e in local_after) <= after['cutoff']+1e-9
    assert not any(e['flagged'] for e in local_after)
    assert signature(notes) == signature(original)
    assert hard_events(notes) == ([], [])
    assert not any(post.clearance_violations(notes).values())
    assert not state.state.speed_groups.violations(range(len(notes)))
    assert all(e['total']<=max(prep_before[e['id']],policy()._soft_bounds(state.nps)['preparation'])+1e-9
               for e in preparation_events(notes))


def test_unrepairable_new_copy_visibility_rolls_back_without_touching_retained_notes():
    notes = [note(i, i*.15, 0, 1, 2, i % 2, kind='retained') for i in range(8)]
    notes.append(note(20, 1.2, 0, 1, 2, 8, kind='copy', anchor_id=7))
    original = deepcopy(notes)
    with pytest.raises(ValueError, match='visibility budget'):
        policy().repair_priority(notes, lambda n: math.inf, protect_retained=True)
    assert notes == original


def test_hard_backtrack_restores_visibility_cache_and_keeps_original_limits(monkeypatch):
    from map_creation.v2_playability_visibility import visibility_costs
    notes = [note(i, i*.2, 0, 1, 2 if i == 0 else 1, (2, 3)[i % 2])
             for i in range(8)]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    original_limits = dict(state.visibility_baseline)
    # Drive the real hard-search history/restore branch without a large,
    # unrelated infeasible map fixture. The actual assignments and caches
    # are real; the pending findings only select that uncommon search path.
    state.prep_hard = {0, 1}
    original_option = (1, 2, 2)
    lowered_option = (1, 1, 2)
    real_trial, real_work = state.trial, state._hard_work
    accepted = False

    def options(i, *, broad=False):
        current = state.state.committed[i]
        return [current, lowered_option] if i == 0 else [current]

    def trial(edits, *, hard_only=False):
        nonlocal accepted
        if not accepted and edits == {0: lowered_option}:
            accepted = True
            return real_trial(edits, hard_only=hard_only)
        return None

    def work(edits):
        if edits == {0: original_option}:
            state.prep_hard.clear()
        real_work(edits)

    monkeypatch.setattr(state.state, 'options', options)
    monkeypatch.setattr(state, 'trial', trial)
    monkeypatch.setattr(state, '_hard_work', work)
    state.repair_hard()
    assert state.hard_backtracks == 1
    state.hard_budget_active = False
    state._refresh_soft()
    actual = {e['id']: e['severity'] for e in visibility_costs(notes, state.nps)['events']}
    assert {k: e['severity'] for k, e in state.visibility_events.items()} == pytest.approx(actual)
    assert state.visibility_baseline == original_limits
    assert not state.visibility_allowed({1: (1, 2, 3)})


def test_post_hard_visibility_refresh_does_not_grandfather_new_excess():
    notes = [note(i, i*.2, 0, 1, 2 if i == 0 else 1, (2, 3)[i % 2])
             for i in range(8)]
    state = policy()._PrioritySearch(notes, lambda n: math.inf, False)
    original_limits = dict(state.visibility_baseline)
    state.hard_budget_active = True
    edits = {2: (1, 2, 2)}
    trial = state.trial(edits, hard_only=True)
    assert trial is not None
    state.commit(edits, trial)
    state.hard_budget_active = False
    state._refresh_soft()
    assert state.visibility_baseline == original_limits
    assert 3 in state.visibility_regressions()


def test_final_visibility_audit_recomputes_events_instead_of_trusting_cache(monkeypatch):
    notes = [note(i, i*.2, 0, 1, 1, (2, 3)[i % 2]) for i in range(8)]
    original = deepcopy(notes)
    monkeypatch.setattr(policy()._PrioritySearch, 'repair_soft', lambda self: (0, 0))

    def stale_final_pass(self):
        # Fault injection: legal live cells change while their event cache
        # stays stale. The final acceptance audit must still reject them.
        self.notes[self.index[0]]['y'] = 2
        self.notes[self.index[1]]['y'] = 2
        return 2

    monkeypatch.setattr(policy()._PrioritySearch, 'repair_visibility', stale_final_pass)
    with pytest.raises(ValueError, match='visibility budget'):
        policy().repair_priority(notes, lambda n: math.inf)
    assert notes == original


def test_saved_horizon_crossing_repairs_with_second_successor_inside_pair_budget(monkeypatch):
    # Exact Horizon8 stalled geometry, seed1243713497, caps44.16. The right
    # target's single move exposes the left hand's second successor2548;
    # both cells must move atomically. Keep the surrounding boundary fixed.
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_late_turns', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    monkeypatch.setattr(policy(), 'HARD_CLUSTER_BUDGET', 20_000)
    rows = [
        (2534, 270.80384414810396, 1, 3, 2, 2),
        (2535, 270.87834314236756, 0, 0, 0, 1),
        (2538, 271.05714072860013, 1, 2, 2, 5),
        (2540, 271.2359383148327, 1, 1, 0, 2),
        (2541, 271.325337107949, 0, 3, 1, 0),
        (2542, 271.4180498866213, 1, 1, 2, 3),
        (2543, 271.5041346941816, 0, 2, 0, 7),
        (2545, 271.59353348729786, 1, 2, 2, 2),
        (2546, 271.68293228041415, 0, 0, 0, 2),
        (2547, 271.7723310735305, 1, 1, 2, 5),
        (2548, 271.8361873543278, 0, 0, 2, 5),
        (2549, 271.9000436351252, 1, 3, 1, 6),
        (2551, 272.02775619671985, 1, 3, 1, 5),
        (2552, 272.0916124775172, 0, 2, 2, 2),
        (2554, 272.21932503911194, 0, 0, 1, 3),
        (2555, 272.21932503911194, 1, 3, 1, 6),
    ]
    notes = [note(*row) for row in rows]
    original = deepcopy(notes)
    state = policy()._PrioritySearch(notes, lambda n:44.16, False)
    state.state.mutable = {i for i,n in enumerate(state.notes) if 270.8<=n['time']<=272.2}
    assert state.hard_pairs == {(2543,2545)} and not state.prep_hard
    witness = {state.index[2545]:(1,2,2), state.index[2548]:(2,1,5)}
    assert state.state.following[state.state.following[state.index[2543]]] == state.index[2548]
    for i, option in witness.items():
        trial = state.trial({i:option})
        assert trial is None or trial[2] or trial[3]
    trial = state.trial(witness)
    assert trial is not None and not trial[2] and not trial[3]
    state.repair_hard()
    assert hard_events(notes) == ([],[])
    assert not any(post.clearance_violations(notes).values())
    assert not state.state.speed_groups.violations(range(len(notes)))
    assert signature(notes) == signature(original)
    frozen = {n['id']:n for n in original if n['time']>272.2}
    assert {n['id']:n for n in notes if n['id'] in frozen} == frozen
    assert max(state.hard_work.values()) <= 20_000


def _rare_visibility_case(history=100, hand=0, top_after=2):
    from map_creation.v2_playability_visibility import visibility_costs
    offset = hand*1000
    before = [note(offset+i, i*.1, hand, 1+hand, 1, 8) for i in range(history)]
    before += [note(offset+history+i, 14.+i*.2, hand, 1+hand, 1, 8) for i in range(5)]
    after = deepcopy(before)
    for n in after[history:history+top_after]:
        n['y'] = 2
    old = visibility_costs(before, 7.4)
    current = visibility_costs(after, 7.4)
    return before, after, current, {e['id']:e['severity'] for e in old['events']}


def test_final_visibility_allows_only_documented_rare_single_quantum_tails():
    _, _, current, baseline = _rare_visibility_case()
    audit = policy()._classify_final_visibility(current['events'], baseline, current['cutoff'])
    assert not audit['visibility_regressions']
    assert {e['id'] for e in audit['visibility_exceptions']} == {103,104}
    assert [e['severity'] for e in audit['visibility_exceptions']] == pytest.approx([.5,.4])
    assert audit['visibility_exception_budget_by_hand']['0'] == dict(
        eligible_events=99, maximum_exceptions=2, actual_exceptions=2)
    for event in audit['visibility_exceptions']:
        assert event['original'] == 0.
        assert event['base'] == current['cutoff']
        assert event['ceiling'] == pytest.approx(current['cutoff']+1/event['onsets'])
        assert event['policy_version'] == audit['visibility_exception_policy']


@pytest.mark.parametrize(('history','maximum'), [(34,0),(50,1)])
def test_final_visibility_exception_budget_is_per_hand_and_fails_over_rate(history,maximum):
    _, _, left, left_baseline = _rare_visibility_case(history=history)
    _, _, right, right_baseline = _rare_visibility_case(hand=1, top_after=0)
    audit = policy()._classify_final_visibility(left['events']+right['events'],
                                               left_baseline|right_baseline, left['cutoff'])
    assert set(audit['visibility_regressions']) == {history+3,history+4}
    assert not audit['visibility_exceptions']
    assert audit['visibility_exception_budget_by_hand']['0']['maximum_exceptions'] == maximum
    assert audit['visibility_exception_budget_by_hand']['1']['maximum_exceptions'] == 2


def test_final_visibility_rejects_bursts_beyond_one_group_quantum():
    from map_creation.v2_playability_visibility import visibility_costs
    before, after, _, baseline = _rare_visibility_case(top_after=5)
    # Extend the burst to the known seven-of-eight central-top case.
    before += [note(105+i, 15.+i*.2, 0, 1, 1, 8) for i in range(3)]
    after += [note(105+i, 15.+i*.2, 0, 1, 2 if i < 2 else 1, 8) for i in range(3)]
    current = visibility_costs(after, 7.4)
    baseline = {e['id']:e['severity'] for e in visibility_costs(before, 7.4)['events']}
    assert current['events'][-1]['severity'] == pytest.approx(.875)
    audit = policy()._classify_final_visibility(current['events'], baseline, current['cutoff'])
    assert 107 in audit['visibility_regressions']
    assert not audit['visibility_exceptions']


def test_final_visibility_never_excuses_worsening_an_existing_tail():
    _, _, current, baseline = _rare_visibility_case()
    # Fractional stack occupancy can produce a pre-existing .45 event;
    # .5 is inside the one-group ceiling but still worsens that bad event.
    baseline[103] = .45
    audit = policy()._classify_final_visibility(current['events'], baseline, current['cutoff'])
    assert audit['visibility_regressions'] == [103]
    assert {e['id'] for e in audit['visibility_exceptions']} == {104}


def test_soft_candidates_cannot_create_even_final_exception_eligible_tails():
    before, _, _, _ = _rare_visibility_case()
    state = policy()._PrioritySearch(before, lambda n:math.inf, False)
    assert not state.visibility_allowed({state.index[100]:(1,2,8),state.index[101]:(1,2,8)})


def test_rare_final_exception_uses_fresh_cells_and_keeps_original_limits(monkeypatch):
    notes, _, _, _ = _rare_visibility_case()
    monkeypatch.setattr(policy()._PrioritySearch, 'repair_soft', lambda self:(0,0))

    def stale_final_pass(self):
        self.notes[self.index[100]]['y'] = 2
        self.notes[self.index[101]]['y'] = 2
        return 2

    monkeypatch.setattr(policy()._PrioritySearch, 'repair_visibility', stale_final_pass)
    result = policy().repair_priority(notes, lambda n:math.inf)
    assert not result['visibility_regressions']
    assert {e['id'] for e in result['visibility_exceptions']} == {103,104}
    limits = {e['id']:e for e in result['visibility_limits']}
    assert limits[103]['baseline'] == 0.
    assert limits[103]['limit'] == result['visibility_after']['cutoff']
    assert limits[103]['severity'] == pytest.approx(.5)
    assert limits[103]['severity'] > limits[103]['limit']
