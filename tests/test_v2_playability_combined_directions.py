"""Combined arrow choices retain human variation within local hard bounds."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from map_creation import v2_playability_directions as directions
from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture(autouse=True)
def combined_policy(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_reach', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5, raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65, raising=False)
    monkeypatch.setattr(config, 'v2_flow_min_turn_steps', 3)


def _note(identity, time, x, direction, *, hand=0, y=1):
    return dict(id=identity, time=time, hand=hand, x=x, y=y, d=direction)


@pytest.mark.parametrize('hand,x,forbidden,neighbor', [
    (0, 3, 2, 3), (0, 3, 4, 7), (0, 3, 6, 5),
    (1, 0, 3, 2), (1, 0, 5, 6), (1, 0, 7, 4)])
def test_forced_direction_cannot_create_mirrored_forbidden_cut(hand, x, forbidden, neighbor):
    notes = [_note(0, 1., 1, neighbor, hand=hand), _note(1, 2., x, 0, hand=hand),
             _note(2, 3., 1, neighbor, hand=hand)]
    before = deepcopy(notes)
    path = directions._window_path(notes, 1, 2, lambda n: float('inf'), {}, set(), 10,
                                   forced={1: forbidden}, require_improvement=False)
    assert path is None
    assert notes == before


def test_style_has_no_verdict_without_eligible_arrow_pair_and_sees_unfinished_tail():
    style = getattr(directions, 'window_style_penalty', None)
    assert callable(style), 'The combined local style objective is missing'
    assert style([_note(0, 1., 0, 8), _note(1, 2., 0, 0)], 2.) == 0.
    vertical = [_note(i, 1.+i*.5, 0, i % 2) for i in range(80)]
    varied = [_note(i, 1.+i*.5, 0, [0, 1, 4, 7, 0, 1, 2, 3][i % 8]) for i in range(20)]
    assert style(vertical, 2.) > style(varied, 2.) == 0.
    assert style(list(reversed(varied)), 2.) == style(varied, 2.)


def test_combined_dwell_preserves_human_normal_four_note_run():
    notes = [_note(0, 1., 1, 0, y=0), _note(1, 1.5, 1, 1, y=2), _note(2, 2., 1, 0, y=0),
             _note(3, 2.5, 1, 1, y=2)]
    notes[1]['_decoded_d'] = 2
    before = deepcopy(notes)
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert notes == before
    assert result['dwell_direction_changed'] == 0
    assert result['dwell_distance_before'] == result['dwell_distance_after'] == 0.
    assert [{k: v for k, v in n.items() if k != 'd'} for n in notes] == [
        {k: v for k, v in n.items() if k != 'd'} for n in before]
    assert not any(post.clearance_violations(notes).values())


def test_combined_dwell_checks_outgoing_arrival_cap():
    notes = [_note(0, 1., 0, 0), _note(1, 1.5, 2, 1), _note(2, 2., 0, 0),
             _note(3, 2.5, 2, 1), _note(4, 3., 0, 0)]
    notes[1]['_decoded_d'] = 2
    before = deepcopy(notes)
    directions.repair_dwell(notes, lambda n: 0. if n['id'] == 2 else float('inf'))
    assert notes == before


@pytest.mark.parametrize('gap,restored', [(.4999, False), (.5, True)])
def test_combined_dwell_keeps_original_slow90_boundary(gap, restored):
    notes = [_note(i, 1.+i*gap, 1, i%2, y=2*(i%2)) for i in range(100)]
    notes[40]['_decoded_d'] = 2
    directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert (notes[40]['d'] == 2) is restored


def test_combined_dwell_cannot_restore_slow_arrival_with_fast_illegal_exit():
    notes = [_note(0, 1., 0, 0), _note(1, 1.5, 2, 1), _note(2, 1.7, 0, 0),
             _note(3, 2.2, 2, 1)]
    notes[1]['_decoded_d'] = 2
    before = deepcopy(notes)
    directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert notes == before


@pytest.mark.parametrize('gap,eligible', [(.6499, False), (.65, True)])
def test_combined_diagonal_keeps_slow45_boundary_and_shared_budget(gap, eligible):
    notes = [_note(0, 1., 0, 0, y=0), _note(1, 1.+gap, 2, 1, y=0),
             _note(2, 1.+2*gap, 0, 0, y=2), _note(3, 1.+3*gap, 2, 1, y=0)]
    notes[1]['_decoded_d'] = 5
    before = deepcopy(notes)
    path = directions._window_path(notes, 1, 2, lambda n: float('inf'), {}, set(), 1,
                                   forced={1: 5}, require_improvement=False)
    assert (path == (5,)) is eligible
    directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=0)
    assert notes == before


def test_combined_late_turn_uses_shared_one_percent_budget_by_default():
    notes = [_note(i, 1.+i*.7, 2*(i % 2), i % 2, y=0) for i in range(100)]
    notes[36]['y'] = 2
    notes[35]['_decoded_d'] = 5
    before = deepcopy(notes)
    result = directions.repair_late_turns(notes, lambda n: float('inf'))
    assert notes[35]['d'] == 5
    assert result['late_turn_direction_changed'] == 1
    assert [{k: v for k, v in n.items() if k != 'd'} for n in notes] == [
        {k: v for k, v in n.items() if k != 'd'} for n in before]


def test_dwell_does_not_restore_decoded_arrow_that_increases_calibrated_preparation():
    notes = [_note(i, 1.+i*.7, 0, i % 2, y=0) for i in range(4)]
    notes[1]['_decoded_d'] = 7
    before = deepcopy(notes)
    result = directions.repair_dwell(notes, lambda n: float('inf'), diagonal_budget=1)
    assert notes == before
    assert result['dwell_direction_changed'] == 0


def test_combined_late_turn_does_not_restore_forbidden_decoded_cut():
    notes = [_note(i, 1.+i*.5, (0 if i % 2 == 0 else 3), i % 2) for i in range(40)]
    notes[35]['_decoded_d'] = 2
    result = directions.repair_late_turns(notes, lambda n: float('inf'))
    assert notes[35]['d'] == 1
    assert result['late_turn_direction_changed'] == 0


def test_combined_dwell_does_not_create_turns_when_decoder_is_uniform():
    notes = [_note(i, 1.+i*.5, 0, i % 2) for i in range(80)]
    before = deepcopy(notes)
    result = directions.repair_dwell(notes, lambda n: float('inf'))
    assert notes == before
    assert result['dwell_direction_changed'] == 0
    assert result['dwell_distance_before'] > 0.


def test_calibrated_m1_keeps_a_normal_horizontal_recut():
    # At .4 seconds the horizontal contribution is .2, below every
    # frozen human cutoff; a diagonal alternative is not an improvement.
    notes = [_note(0, 1., 0, 2), _note(1, 1.4, 2, 3), _note(2, 1.8, 0, 2)]
    before = deepcopy(notes)
    path = directions._window_path(notes, 1, 2, lambda n: float('inf'), {}, set(), 1)
    assert path is None
    assert notes == before


def test_fast_m1_repair_cannot_replace_a_uniform_horizontal_stream_with_uniform_vertical():
    notes = [_note(i, 1.+i*.2, 0, 2+i % 2) for i in range(200)]
    from map_creation.v2_playability_policy import PatternGuard
    result = directions.repair_directions(notes, lambda n: float('inf'), diagonal_budget=2,
                                          guard=PatternGuard(notes))
    assert sum(n['d'] in (0, 1) for n in notes) < 180
    assert result['m1_diagonal_net'] <= 2


def test_combined_r7_retains_a_feasible_direction_repair():
    passage = json.loads((Path(__file__).parent/'fixtures/v2_r7_horizontal_passage.json').read_text())
    notes = deepcopy(passage['notes'])
    before = deepcopy(notes)
    result = directions.repair_directions(notes, lambda n: passage['arrival_caps'][str(n['id'])],
                                          total_notes=1480, diagonal_budget=14)
    assert result['m1_direction_changed'] > 0
    assert sum({a['d'], b['d']} == {2, 3} for a, b in zip(
        [n for n in notes if n['hand'] == 1], [n for n in notes if n['hand'] == 1][1:])) < 12
    assert not any(post.clearance_violations(notes).values())
    assert [{k: v for k, v in n.items() if k != 'd'} for n in notes] == [
        {k: v for k, v in n.items() if k != 'd'} for n in before]
