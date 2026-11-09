"""Incoming-gap small-turn eligibility is recomputed at every flow consumer."""
from copy import deepcopy

import pytest

from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture(autouse=True)
def slow_policy(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65, raising=False)
    monkeypatch.setattr(config, 'v2_flow_min_turn_steps', 3)


@pytest.mark.parametrize('threshold', [.35, .5])
@pytest.mark.parametrize('hand', [0, 1])
def test_slow90_requires_incoming_gap_and_two_cell_travel(monkeypatch, threshold, hand):
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', threshold)
    # Horizontal reflection keeps the turn and two-cell displacement legal.
    px, x, direction = (0, 2, 2) if hand == 0 else (3, 1, 3)
    prior = (1., px, 0, 0)
    assert post._flow_legal(direction, prior, x, 0, 1. + threshold, .1)
    assert not post._flow_legal(direction, prior, x, 0, 1. + threshold - .0001, .1)
    assert not post._flow_legal(direction, prior, px + (1 if hand == 0 else -1), 0, 2., .1)


@pytest.mark.parametrize('threshold', [.5, .65])
@pytest.mark.parametrize('hand', [0, 1])
def test_slow45_requires_physical_approach_and_gap(monkeypatch, threshold, hand):
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', threshold)
    px, x, direction = (0, 2, 5) if hand == 0 else (3, 1, 4)
    prior = (1., px, 0, 0)
    assert post._flow_legal(direction, prior, x, 1, 1. + threshold, .1)
    assert not post._flow_legal(direction, prior, x, 1, 1. + threshold - .0001, .1)
    # Same angle and long pause, but the hand approaches against the cut.
    assert not post._flow_legal(direction, (1., x, 2, 0), px, 0, 2., .1)
    assert not post._flow_legal(0, prior, x, 2, 3., .1)


def test_master_off_and_zero_threshold_preserve_strict_flow(monkeypatch):
    prior = (1., 0, 0, 0)
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    assert not post._flow_legal(2, prior, 2, 0, 2., .1)
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', 0.)
    assert not post._flow_legal(2, prior, 2, 0, 2., .1)
    assert post._flow_legal(2, prior, 2, 0, 1.05, .1)


def test_stream_and_structured_cleanup_preserve_decoded_slow_turn():
    values = [[0, 0, 0, 0], [2, 0, 0, 2]]
    assert post._recut_same_direction_repeats(deepcopy(values), [1., 1.5], .1) == values
    notes = [dict(id=0, time=1., hand=0, x=0, y=0, d=0),
             dict(id=1, time=1.5, hand=0, x=2, y=0, d=2)]
    before = deepcopy(notes)
    assert post.recut_same_direction_notes(notes, protect_retained=False) == 0
    assert notes == before
    assert post.clearance_violations(notes)['flow_repeats'] == 0


def test_snap_cannot_cross_slow_reset_threshold():
    notes = [dict(id=0, time=1., hand=0, x=0, y=0, d=0),
             dict(id=1, time=1.5, hand=0, x=2, y=0, d=2)]
    effective = {0: 0, 1: 2}
    assert not post._snap_violated(notes[1], 1.5, [], notes, 1, effective, .02, .15, .1)
    assert post._snap_violated(notes[1], 1.4999, [], notes, 1, effective, .02, .15, .1)


def test_full_edge_context_recut_rejects_newly_unsafe_outgoing_turn():
    note = dict(id=1, time=1.5, hand=0, x=2, y=0, d=2)
    previous = (1., 0, 0, 0)
    following = dict(id=2, time=1.7, hand=0, x=2, y=1, d=0)
    # d2 is an eligible slow arrival, but its fast 90-degree exit is illegal.
    direction = post._flow_direction(2, 0, note=note, previous=previous, following=following)
    assert direction != 2
    assert post._flow_legal(direction, previous, 2, 0, 1.5, .1)
    assert post._flow_legal(0, (1.5, 2, 0, direction), 2, 1, 1.7, .1)


def test_deletion_recomputes_small_turn_against_new_predecessor(monkeypatch):
    monkeypatch.setattr(config, 'max_speed', 20.)
    monkeypatch.setattr(config, 'v2_speed_cap_start_notes', 0)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 2.)
    notes = [dict(id=0, time=1., hand=0, x=0, y=0, d=0),
             dict(id=1, time=1.001, hand=0, x=2, y=0, d=2),
             dict(id=2, time=1.6, hand=0, x=0, y=0, d=0)]
    # Before deletion the last90-degree edge has enough incoming recovery.
    assert post._flow_legal(0, (1.001, 2, 0, 2), 0, 0, 1.6, .1)
    stats = post.apply_speed_cap(notes)
    assert stats['speed_cap_dropped'] == 1
    assert [n['id'] for n in notes] == [0, 2]
    assert notes[-1]['d'] != 0
    assert post.clearance_violations(notes)['flow_repeats'] == 0


def test_slow_turn_cannot_bypass_opposite_hand_swing_clearance():
    notes = [dict(id=0, time=1., hand=0, x=0, y=0, d=0),
             dict(id=1, time=1.5, hand=0, x=2, y=0, d=2),
             dict(id=2, time=1.55, hand=1, x=1, y=0, d=0)]
    residual = post.clearance_violations(notes)
    assert residual['flow_repeats'] == 0
    assert residual['swing_into_pairs'] > 0
    post.enforce_note_flow_and_clearance(notes, protect_retained=False)
    assert not any(post.clearance_violations(notes).values())
