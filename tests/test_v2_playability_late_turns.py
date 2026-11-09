"""Restore actual decoder turns after strict primary density selection."""
from copy import deepcopy

import pytest

from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    from evaluation.playability_experiments import SWITCHES
    for field, value in SWITCHES.items():
        monkeypatch.setattr(config, 'v2_playability_' + field, value)
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_late_turns', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_start_notes', 30)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)


def _notes():
    notes = [dict(id=i, time=1.+i*.5, hand=0, x=2*(i%2), y=1, d=i%2) for i in range(40)]
    notes[5]['_decoded_d'] = 2
    notes[35]['_decoded_d'] = 2
    return notes


def _phase():
    phase = getattr(post, 'playability_generation_phase', None)
    assert callable(phase), 'Late-turn phase context is missing'
    return phase()


def test_late_turns_keep_early_cleanup_strict_and_restore_only_beyond_ramp():
    notes = _notes()
    before = deepcopy(notes)
    with _phase():
        assert not post._flow_legal(2, (1., 0, 0, 0), 2, 0, 1.5, .1)
        result = post.apply_speed_cap(notes)
        assert post._flow_legal(2, (1., 0, 0, 0), 2, 0, 1.5, .1)
    assert notes[5]['d'] == 1
    assert notes[35]['d'] == 2
    assert result['late_turn_direction_changed'] == 1
    assert result['playability_preserved']
    assert [{k:v for k,v in n.items() if k != 'd'} for n in notes] == [{k:v for k,v in n.items() if k != 'd'} for n in before]
    assert not any(post.clearance_violations(notes).values())


def test_late_turn_rejects_outgoing_swing_collision():
    notes = _notes()
    target = notes[35]
    notes.append(dict(id=100, time=target['time']+.05, hand=1, x=1, y=1, d=0))
    with _phase():
        result = post.apply_speed_cap(notes)
    assert target['d'] == 1
    assert result['late_turn_direction_changed'] == 0


def test_late_turn_phase_resets_even_on_failure():
    with pytest.raises(ValueError):
        with _phase():
            assert not post._flow_legal(2, (1., 0, 0, 0), 2, 0, 1.5, .1)
            raise ValueError('fixture failure')
    assert post._flow_legal(2, (1., 0, 0, 0), 2, 0, 1.5, .1)
