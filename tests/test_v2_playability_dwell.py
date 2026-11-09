"""Frozen human strata and conservative completed-axis-run proposals."""
from copy import deepcopy

import pytest

from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture(autouse=True)
def legacy_profile(monkeypatch):
    """Exercise historical arms independently of the promoted production policy."""
    from evaluation.playability_experiments import SWITCHES
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    for field, value in SWITCHES.items():
        monkeypatch.setattr(config, 'v2_playability_' + field, value)


def test_dwell_reference_pools_sparse_left_and_right90_bands():
    from map_creation.v2_playability_reference import dwell_reference
    for hand in (0, 1):
        sparse = dwell_reference(5.5, hand, 90)
        assert sparse['source_bands'] == ('3-5', '5-7')
        assert sparse['packages'] >= 5
        supported = dwell_reference(5.5, hand, 45)
        assert supported['source_bands'] == ('5-7',)
        assert supported['packages'] == 7
    assert dwell_reference(2.99, 0, 45)['source_bands'] == ('<3',)
    assert dwell_reference(7., 1, 90)['source_bands'] == ('7+',)


def test_dwell_smooths_repeated_one_note45_detours_without_rhythm_or_stress_change(monkeypatch):
    from map_creation import v2_playability_directions as directions
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    notes = [dict(id=i, time=1.+i*.5, hand=0, x=0, y=1, d=(0 if i%2 == 0 else 7))
             for i in range(15)]
    before = deepcopy(notes)
    repair = getattr(directions, 'repair_dwell', None)
    assert callable(repair), 'The separate dwell experiment is missing'
    result = repair(notes, lambda n: float('inf'))
    assert result['dwell_direction_changed'] > 0
    assert result['dwell_distance_after'] < result['dwell_distance_before']
    assert [{k:v for k,v in n.items() if k != 'd'} for n in notes] == [{k:v for k,v in n.items() if k != 'd'} for n in before]
    assert post.clearance_violations(notes) == {'same_cell_pairs': 0, 'swing_into_pairs': 0, 'flow_repeats': 0}
    assert notes[0]['d'] == before[0]['d'] and notes[-1]['d'] == before[-1]['d']
    replay = deepcopy(before)
    assert repair(replay, lambda n: float('inf')) == result
    assert replay == notes


def test_dwell_does_not_invent_changes_in_an_uninterrupted_vertical_control():
    from map_creation import v2_playability_directions as directions
    notes = [dict(id=i, time=1.+i*.5, hand=0, x=0, y=1, d=i%2) for i in range(15)]
    before = deepcopy(notes)
    repair = getattr(directions, 'repair_dwell', None)
    assert callable(repair), 'The separate dwell experiment is missing'
    assert repair(notes, lambda n: float('inf'))['dwell_direction_changed'] == 0
    assert notes == before


def test_dwell_arm_runs_separately_and_is_audited(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', False, raising=False)
    monkeypatch.setattr(config, 'v2_playability_dwell', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)
    notes = [dict(id=i, time=1.+i*.5, hand=0, x=0, y=1, d=(0 if i%2 == 0 else 7))
             for i in range(15)]
    stats = post.apply_speed_cap(notes)
    assert stats['dwell_direction_changed'] > 0
    assert stats['playability_preserved']
    assert stats['speed_cap_enabled'] is False
    assert 'm1_direction_changed' not in stats


def test_dwell_keeps_original_decoder_proposal_for_later_feasibility(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_dwell', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', 0., raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', 0., raising=False)
    monkeypatch.setattr(config, 'v2_cleanup_same_dir_recut', 1.)
    raw = [dict(id=0, time=1., hand=0, x=0, y=0, d=0),
           dict(id=1, time=1.2, hand=0, x=2, y=0, d=2)]
    notes, _ = post.postprocess_notes(raw, [1., 1.2], skip=tuple(s for s in post._NOTE_STEPS if s != 'same_direction_recut'))
    assert notes[1]['d'] != 2
    assert notes[1]['_decoded_d'] == 2


def test_dwell_restores_supported_decoded_change_in_a_long_unobserved_run(monkeypatch):
    from map_creation.v2_playability_directions import repair_dwell
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5, raising=False)
    notes = [dict(id=i, time=1.+i*.5, hand=0, x=2*(i%2), y=1, d=i%2)
             for i in range(15)]
    notes[3]['_decoded_d'] = 2
    result = repair_dwell(notes, lambda n: float('inf'))
    assert notes[3]['d'] == 2
    assert result['dwell_direction_changed'] == 1
    assert result['dwell_distance_after'] < result['dwell_distance_before']
    assert post.clearance_violations(notes)['flow_repeats'] == 0
