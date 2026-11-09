"""Completed-axis events use human tail budgets without erasing normal variety."""
from copy import deepcopy

import pytest

from map_creation import v2_playability_directions as directions
from tools.config import config


def _notes(arrows, gap=.5, *, hand=0):
    return [dict(id=i, time=1.+i*gap, hand=hand, x=hand*3, y=1, d=d)
            for i, d in enumerate(arrows)]


def test_direction_calibration_joint_tails_respect_budget_and_pool_sparse_maps():
    from evaluation.playability_direction_calibration import fit_direction_reference
    events = [dict(package=f'map{i % 5}', band='3-5', hand=0, ending_angle=45,
                   notes=1, seconds=.5 if 0 < i < 99 else (0.05 if i == 0 else 9.),
                   incoming_seconds=.5) for i in range(100)]
    reference = fit_direction_reference(events, budget=.1)
    direct = reference['strata']['3-5|0|45']
    assert direct['intervals']['seconds'] == [.5, .5]
    assert direct['empirical_flagged_share'] == .02
    assert direct['empirical_flagged_share'] <= direct['budget']
    pooled = reference['strata']['5-7|0|45']
    assert pooled['source_bands'] == ['3-5', '5-7']
    assert pooled['packages'] == 5
    assert pooled['events'] == 100


def test_axis_events_fold_actual135_but_retain_incoming_and_next_change_times():
    events = directions.axis_run_events(_notes([0, 6, 1, 2], gap=.2))
    completed = events['completed']
    assert [e['ending_angle'] for e in completed] == [45, 45, 90]
    assert [e['actual_degrees'] for e in completed] == [135, 45, 90]
    assert [e['notes'] for e in completed] == [1, 1, 1]
    assert [e['seconds'] for e in completed] == pytest.approx([.2, .2, .2])
    assert [e['incoming_seconds'] for e in completed] == pytest.approx([.2, .2, .2])
    assert [e['next_change_seconds'] for e in completed[:2]] == pytest.approx([.2, .2])
    assert completed[-1]['next_change_seconds'] is None
    assert events['fast_reversals'] == 1
    assert events['transitions'] == 3


def test_axis_events_keep_hands_separate_collapse_stacks_and_censor_dot_boundaries():
    notes = _notes([0, 1, 8, 2, 3], gap=.5)
    notes.append(dict(id=20, time=1.51, hand=0, x=0, y=1, d=6))
    notes.extend(_notes([0, 2], gap=.75, hand=1))
    events = directions.axis_run_events(notes)
    assert [(e['hand'], e['ending_angle'], e['notes']) for e in events['completed']] == [(1, 90, 1)]
    assert [(e['hand'], e['notes']) for e in events['censored']] == [(0, 2), (0, 2), (1, 1)]
    assert events['transitions'] == 3
    assert events['fast_reversals'] == 0


@pytest.mark.parametrize('hand', [0, 1])
def test_combined_dwell_preserves_normal_short_runs_even_with_smoother_alternative(monkeypatch, hand):
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    notes = _notes([0, 7]*8, gap=.5, hand=hand)
    before = deepcopy(notes)
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=1)
    assert notes == before
    assert result['dwell_direction_changed'] == 0
    assert result['dwell_distance_before'] == 0.


def test_frozen_axis_tails_see_fast135_churn_and_censored_long_runs():
    short = directions.direction_tail_costs(_notes([0, 7]*10, gap=.1), nps=4.)
    normal = directions.direction_tail_costs(_notes([0, 7]*10, gap=.5), nps=4.)
    long = directions.direction_tail_costs(_notes([0, 1]*40, gap=.5), nps=4.)
    assert short['short_runs'] > 0
    assert short['fast_axis_changes'] == 19
    assert short['score'] > 0.
    assert normal['score'] == 0.
    assert long['censored_long_runs'] == 1
    assert long['score'] > 0.


def test_combined_directions_have_one_percent_growth_ceiling(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    assert directions._diagonal_growth_budget(200, 20, 4.) == 2
    assert directions._diagonal_growth_budget(40, 4, 4.) == 0


@pytest.mark.parametrize('repair', ['dwell', 'directions', 'isolated', 'late_turns'])
def test_direction_passes_respect_external_pattern_guard(monkeypatch, repair):
    import json
    from pathlib import Path
    monkeypatch.setattr(config, 'v2_playability_reach', False)
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_flow_min_turn_steps', 3)
    kwargs = {}
    if repair in ('directions', 'isolated'):
        passage = json.loads((Path(__file__).parent/'fixtures/v2_r7_horizontal_passage.json').read_text())
        notes = deepcopy(passage['notes'])
        cap = lambda n: passage['arrival_caps'][str(n['id'])]
        kwargs = dict(total_notes=1480)
        if repair == 'isolated':
            kwargs['eligible_ids'] = {276, 282}
    elif repair == 'dwell':
        notes = _notes([0, 7]*8, gap=.5)
        cap = lambda n: float('inf')
    else:
        notes = [dict(id=i, time=1.+i*.5, hand=0, x=2*(i%2), y=1, d=i%2) for i in range(40)]
        notes[35]['_decoded_d'] = 2
        cap = lambda n: float('inf')
    before = deepcopy(notes)
    unguarded = deepcopy(notes)
    function = getattr(directions, 'repair_'+repair)
    function(unguarded, cap, **kwargs)
    assert unguarded != before, 'Fixture must exercise a feasible proposed edit'
    frozen = {n['id']: n['d'] for n in notes}
    class FreezeDirections:
        def allows(self, identities):
            return all(n['d'] == frozen[n['id']] for n in notes if n['id'] in identities)
        def commit(self, identities):
            assert self.allows(identities), 'A rejected arrow was committed'
    function(notes, cap, guard=FreezeDirections(), **kwargs)
    assert notes == before


@pytest.mark.parametrize('hand', [0, 1])
def test_combined_dwell_can_shorten_overlong_censored_run_with_supported_decoder(monkeypatch, hand):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    notes = [dict(id=i, time=1.+i*.5, hand=hand, x=1+hand, y=2*(i%2), d=i%2) for i in range(100)]
    notes[40]['_decoded_d'] = 2 if hand == 0 else 3
    before = deepcopy(notes)
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert notes[40]['d'] == before[40]['_decoded_d']
    assert result['dwell_direction_changed'] == 1
    assert result['dwell_distance_after'] < result['dwell_distance_before']
    assert [{k:v for k,v in n.items() if k != 'd'} for n in notes] == [
        {k:v for k,v in n.items() if k != 'd'} for n in before]


def test_combined_dwell_smooths_abnormally_fast_actual135_axis_churn(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    notes = _notes([0, 7]*12, gap=.15)
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert result['dwell_direction_changed'] > 0
    assert result['dwell_distance_after'] < result['dwell_distance_before']


def test_adding_normal_axis_changes_cannot_dilute_unchanged_short_tail():
    before = _notes([0, 7, 4, 7, 4, 7, 4, 7, 4, 7], gap=.5)
    for i, note in enumerate(before[1:], 1):
        note['time'] = 1.1+(i-1)*.5
    after = deepcopy(before)
    after[5]['d'] = 0
    old = directions.direction_tail_costs(before, nps=4.)
    new = directions.direction_tail_costs(after, nps=4.)
    assert old['short_runs'] == new['short_runs'] == 1
    assert new['completed_runs'] > old['completed_runs']
    assert new['score'] == pytest.approx(old['score'])


def test_dwell_repair_keeps_both_previously_observed_ending_angles(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    notes = _notes([0, 7]*5+[0], gap=.15)
    notes.extend([dict(id=11, time=3.2, hand=0, x=2, y=1, d=2),
                  dict(id=12, time=3.7, hand=0, x=2, y=1, d=3)])
    before_angles = {e['ending_angle'] for e in directions.axis_run_events(notes)['completed']}
    assert before_angles == {45, 90}
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert result['dwell_direction_changed'] > 0
    assert {e['ending_angle'] for e in directions.axis_run_events(notes)['completed']} == {45, 90}


@pytest.mark.parametrize('guarded', [False, True])
def test_normal_dwell_prefix_cannot_exhaust_search_before_supported_long_tail(monkeypatch, guarded):
    from map_creation.v2_playability_policy import PatternGuard
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    notes = [dict(id=i, time=1.+i*.5, hand=0, x=1, y=2*(i%2),
                  d=(0 if i%2 == 0 else 7) if i < 140 else i%2) for i in range(240)]
    for identity, (x, y) in {179:(0, 0), 180:(1, 1), 181:(3, 0)}.items():
        notes[identity].update(x=x, y=y)
    notes[180]['_decoded_d'] = 3
    before = deepcopy(notes)
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0,
                                     guard=PatternGuard(notes) if guarded else None)
    assert notes[:140] == before[:140]
    assert notes[180]['d'] == 3
    assert result['dwell_direction_changed'] == 1
    assert result['dwell_candidates'] <= 2
    assert result['dwell_distance_after'] < result['dwell_distance_before']
