from copy import deepcopy

import pytest


def note(i, time, x=1, y=1, d=0, hand=0):
    return dict(id=i, time=time, x=x, y=y, d=d, hand=hand)


def guard(notes):
    from map_creation.v2_playability_policy import PatternGuard
    return PatternGuard(notes, nps=4.)


def test_optional_repair_rejects_moderate_load_even_below_tail_cutoff():
    notes = [note(0, 1., d=0), note(1, 1.25, d=1), note(2, 1.5, d=0)]
    policy = guard(notes)
    notes[1]['d'] = 7
    assert not policy.allows({1})


def test_arrow_arrival_after_dot_still_counts_moderate_lateral_load():
    notes = [note(0, 1., d=8), note(1, 1.3, d=0)]
    policy = guard(notes)
    notes[1]['d'] = 4
    assert not policy.allows({1})


def test_optional_repair_cannot_add_fast_axis_churn_to_a_comfortable_stream():
    notes = [note(i, 1.+i*.3, d=0 if i % 2 == 0 else 1) for i in range(6)]
    policy = guard(notes)
    notes[2]['d'] = 4
    assert not policy.allows({2})


def test_same_cuts_cannot_hide_a_larger_travel_jump():
    notes = [note(i, 1.+i*.3, d=0 if i % 2 == 0 else 1) for i in range(6)]
    policy = guard(notes)
    notes[2]['x'] = 3
    assert not policy.allows({2})


def test_committed_improvement_is_the_next_pass_baseline():
    notes = [note(i, 1.+i*.3, d=2 if i % 2 == 0 else 3) for i in range(6)]
    policy = guard(notes)
    for n in notes:
        n['d'] = 0 if n['id'] % 2 == 0 else 1
    assert policy.allows(set(range(6)))
    policy.commit(set(range(6)))
    notes[2]['d'] = 2
    assert not policy.allows({2})


def test_guard_keeps_original_identity_and_handles_unsorted_empty_input():
    notes = [note(8, 3.), note(3, 1., d=1), note(7, 2.)]
    original = deepcopy(notes)
    policy = guard(notes)
    assert policy.allows({8})
    assert notes == original
    assert guard([]).allows(set())
    with pytest.raises(ValueError, match='identity'):
        policy.allows({99})


def test_coordination_uses_segmented_raw_displacements_and_dot_positions():
    from map_creation.v2_playability_policy import _features
    notes = []
    for segment in range(8):
        for i in range(5):
            t = 1.+2*segment+.3*i
            notes.extend([note(len(notes), t, 0, 2*(i % 2), i % 2),
                          note(len(notes)+1, t, 2+(i % 2) if segment == 0 else 3,
                               1 if segment == 0 else 2*(i % 2), i % 2, hand=1)])
    assert _features(notes, 6.)['coordination_low'] == pytest.approx(0.)
    dots = [note(0,1.,0,0,8), note(1,1.,2,0,8,hand=1),
            note(2,1.2,0,1,8), note(3,1.2,3,0,8,hand=1),
            note(4,1.4,0,2,8), note(5,1.4,2,0,8,hand=1)]
    assert _features(dots, 4.)['coordination_high'] > 0.


def test_local_reach_change_cannot_hide_an_overlong_full_axis_run():
    from map_creation.v2_playability_policy import PatternGuard
    from map_creation.v2_playability_directions import direction_tail_costs
    notes = [note(i, 1.+i, y=2*(i % 2), d=i % 2) for i in range(60)]
    notes[30]['d'] = 2
    policy = PatternGuard(notes, nps=1.)
    before = direction_tail_costs(notes, nps=1.)['score']
    notes[30]['d'] = 0
    assert direction_tail_costs(notes, nps=1.)['score'] > before
    assert not policy.allows({30})


def test_global_coordination_cannot_worsen_behind_unchanged_segment_tails(monkeypatch):
    from map_creation import v2_playability_policy as policy
    notes = []
    for i in range(8):
        notes.extend([note(2*i, 1.+i*.3, 0, 2*(i % 2), i % 2),
                      note(2*i+1, 1.+i*.3, 2, 2*(i % 2), i % 2, hand=1)])
    guard = policy.PatternGuard(notes, nps=4.)
    # Isolate the whole-map M3 contract from the independent local checks.
    monkeypatch.setattr(policy, '_features', lambda notes, nps: {'unchanged': 0.})
    for n in notes:
        if n['hand'] == 1:
            n['x'], n['y'] = 2+(n['id']//2) % 2, (n['id']//2) % 2
    assert not guard.allows({n['id'] for n in notes if n['hand'] == 1})


def test_deferred_hand_commit_refreshes_and_restores_both_global_caches():
    from map_creation.v2_playability_policy import PatternGuard, _coordination_features
    from map_creation.v2_playability_directions import direction_tail_costs
    notes = [note(i, 1.+i, y=2*(i % 2), d=i % 2) for i in range(60)]
    notes[30]['d'] = 2
    policy = PatternGuard(notes, nps=1.)
    checkpoint = policy.checkpoint()
    notes[30]['hand'] = 1
    policy.commit({30})
    assert policy.direction_costs == direction_tail_costs(notes, nps=1.)
    assert policy.coordination_costs == _coordination_features(notes, nps=1.)
    notes[30]['hand'] = 0
    policy.restore(checkpoint)
    assert policy.direction_costs == direction_tail_costs(notes, nps=1.)


def test_complete_policy_relaxes_the_actual_reported_repeated_horizontal_passage(monkeypatch):
    from map_creation.v2_playability_lateral import consecutive_horizontal_load
    import json
    from pathlib import Path
    from evaluation.playability_experiments import experiment_policy
    from map_creation import v2_playability_directions as directions
    from map_creation.v2_playability_policy import PatternGuard
    fixture = json.loads((Path(__file__).parent/'fixtures/v2_r7_horizontal_passage.json').read_text())
    notes = fixture['notes']
    cap = lambda n: fixture['arrival_caps'][str(n['id'])]
    monkeypatch.setattr(directions, '_map_nps', lambda notes: 4.5)
    before = sum(n['d'] in (2, 3) for n in notes)
    before_run = consecutive_horizontal_load(notes, nps=4.5)
    with experiment_policy('combined'):
        result = directions.repair_directions(notes, cap, total_notes=1480,
            diagonal_count=210, diagonal_budget=14, guard=PatternGuard(notes, nps=4.5))
    assert result['m1_windows_improved'] > 0
    assert sum(n['d'] in (2, 3) for n in notes) < before
    after_run = consecutive_horizontal_load(notes, nps=4.5)
    assert after_run['max_reversals'] <= 5
    assert after_run['total_excess'] < before_run['total_excess']*.2
