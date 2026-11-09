"""Exact position repeats move along the cut, within existing priority guards."""
from copy import deepcopy
import math
from unittest.mock import Mock

import pytest

from map_creation.v2_playability_priority import _PrioritySearch


def note(i, hand, x, y, d, time=None, **extra):
    return dict(id=i, time=float(i) if time is None else time, hand=hand,
                x=x, y=y, d=d, **extra)


def search(notes, cap=math.inf, protect=False):
    return _PrioritySearch(notes, lambda n: cap, protect)


@pytest.mark.parametrize('direction,opposite,target', [
    (0, 1, (1, 2)), (1, 0, (1, 0)), (2, 3, (0, 1)), (3, 2, (2, 1)),
    (4, 7, (1, 2)), (5, 6, (1, 2)), (6, 5, (1, 0)), (7, 4, (1, 0))])
def test_cut_direction_one_cell_and_diagonals_vertical(direction, opposite, target):
    notes = [note(0, 0, 1, 1, opposite), note(1, 0, 1, 1, direction)]
    original = deepcopy(notes)
    state = search(notes)
    # Isolate direction mapping from separate calibrated soft costs.
    state.soft_allowed = lambda *args: True
    assert state.separate_repeated_positions(Mock(random=lambda: .1)) == 1
    assert (notes[1]['x'], notes[1]['y']) == target
    assert notes[0] == original[0]
    assert all({k:v for k,v in n.items() if k not in ('x','y')} ==
               {k:v for k,v in o.items() if k not in ('x','y')}
               for n,o in zip(notes,original))


def test_live_previous_position_is_used():
    notes = [note(i, 0, 1, 1, i % 2) for i in range(4)]
    state = search(notes)
    assert state.separate_repeated_positions(Mock(random=lambda: .1)) == 2
    assert [n['y'] for n in notes] == [1,0,1,0]


@pytest.mark.parametrize('draw,changed', [(.499,1),(.5,0),(.9,0)])
def test_top_row_coin_flip(draw,changed):
    notes = [note(0,0,1,1,1),note(1,0,1,1,0)]
    assert search(notes).separate_repeated_positions(Mock(random=lambda: draw)) == changed
    assert notes[1]['y'] == (2 if changed else 1)


@pytest.mark.parametrize('hand,x,y,d',[(0,0,1,2),(0,2,1,3),(1,1,1,2),(1,3,1,3),(0,1,0,1),(0,1,2,0)])
def test_hand_grid_edges_do_not_move(hand,x,y,d):
    notes = [note(0,hand,x,y,8),note(1,hand,x,y,d)]
    before = deepcopy(notes)
    assert search(notes).separate_repeated_positions() == 0
    assert notes == before


def test_dots_distinct_positions_simultaneous_and_protected_retained_stay():
    for notes,protect in [
        ([note(0,0,1,1,0),note(1,0,1,1,8)],False),
        ([note(0,0,1,2,0),note(1,0,1,1,1)],False),
        ([note(0,0,1,1,0,time=1),note(1,0,1,1,1,time=1)],False),
        ([note(0,0,1,1,0),note(1,0,1,1,1,kind='retained')],True),
    ]:
        before = deepcopy(notes)
        assert search(notes,protect=protect).separate_repeated_positions() == 0
        assert notes == before


def test_collision_is_rejected_without_drawing_top_row_randomness():
    notes = [note(0,0,1,1,1),note(1,0,1,1,0),note(2,1,1,2,8,time=1)]
    before=deepcopy(notes); rng=Mock()
    assert search(notes).separate_repeated_positions(rng) == 0
    assert notes == before
    rng.random.assert_not_called()


def test_soft_guard_rejection_is_not_committed():
    notes = [note(0,0,1,1,0),note(1,0,1,1,1)]
    before=deepcopy(notes); state=search(notes)
    state.soft_allowed=lambda *args: False
    assert state.separate_repeated_positions() == 0
    assert notes == before


def test_repeat_pass_is_wired_before_final_audit(monkeypatch):
    from map_creation import v2_playability_priority as policy
    calls=[]
    monkeypatch.setattr(policy._PrioritySearch,'separate_repeated_positions',lambda self: calls.append(1) or 0)
    result=policy.repair_priority([note(0,0,1,1,0)],lambda n: math.inf)
    assert calls == [1]
    assert result['repeat_position_changed'] == 0


def test_each_hand_has_its_own_predecessor():
    notes=[note(0,0,1,1,0),note(1,1,1,1,1)]
    before=deepcopy(notes)
    assert search(notes).separate_repeated_positions() == 0
    assert notes == before


def test_speed_guard_cannot_be_bypassed(monkeypatch):
    notes=[note(0,0,1,1,0),note(1,0,1,1,1)]
    before=deepcopy(notes); state=search(notes)
    monkeypatch.setattr(state.state.speed_groups,'legal',lambda changed: False)
    assert state.separate_repeated_positions() == 0
    assert notes == before


def test_first_member_of_stack_is_not_moved():
    notes=[note(0,0,1,1,0),note(1,0,1,1,1),note(2,0,2,1,1,time=1)]
    before=deepcopy(notes)
    assert search(notes).separate_repeated_positions() == 0
    assert notes == before


def test_actual_local_visibility_tail_cannot_worsen():
    notes=[note(i,0,1,2,8) for i in range(6)]
    notes += [note(6,0,1,1,1),note(7,0,1,1,0)]
    # Keep the eight onsets in the M2 window.
    for n in notes:
        n['time'] *= .2
    state=search(notes); before=deepcopy(notes); rng=Mock()
    assert state.separate_repeated_positions(rng) == 0
    assert notes == before
    rng.random.assert_not_called()
