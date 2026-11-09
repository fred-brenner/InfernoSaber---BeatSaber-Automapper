"""Portable direction experiments: no audio, weights or human corpus at runtime."""
from copy import deepcopy
import json
from pathlib import Path

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


@pytest.fixture
def passage(monkeypatch):
    value = json.loads((Path(__file__).parent / 'fixtures/v2_r7_horizontal_passage.json').read_text())
    monkeypatch.setattr(config, 'max_speed', 20.)
    monkeypatch.setattr(config, 'v2_flow_min_turn_steps', 3)
    monkeypatch.setattr(config, 'v2_fast_gap_seconds', .2)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', 0., raising=False)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', 0., raising=False)
    return value


def _horizontal_count(notes):
    count = 0
    for hand in (0, 1):
        seq = sorted((n for n in notes if n['hand'] == hand), key=lambda n: n['time'])
        count += sum(0 < b['time'] - a['time'] <= .25 and {a['d'], b['d']} == {2, 3}
                     for a, b in zip(seq, seq[1:]))
    return count


def _repair(notes, cap, **kwargs):
    repair = getattr(post, 'repair_playability_directions', None)
    assert callable(repair), 'The bounded production direction repair is missing'
    return repair(notes, cap, **kwargs)


def test_r7_witness_and_isolated_middle_are_pinned(passage):
    notes = passage['notes']
    assert _horizontal_count([n for n in notes if n['hand'] == 1]) == 12
    witness = [{**n, 'd': passage['witness'].get(str(n['id']), n['d'])} for n in notes]
    assert post.clearance_violations(witness) == {'same_cell_pairs': 0, 'swing_into_pairs': 0, 'flow_repeats': 0}
    right = sorted((n for n in witness if n['hand'] == 1), key=lambda n: n['time'])
    assert max(post.calc_note_speed([a[k] for k in ('x', 'y', 'hand', 'd')],
                                    [b[k] for k in ('x', 'y', 'hand', 'd')], b['time'] - a['time'])
               for a, b in zip(right, right[1:])) == pytest.approx(22.31066509, abs=.00001)
    # With the two horizontal neighbors fixed, no isolated nonhorizontal edit
    # can satisfy flow, movement and other-hand clearance together.
    for direction in (0, 1, 4, 5, 6, 7):
        trial = [{**n, 'd': direction} if n['id'] == 282 else n for n in notes]
        pair = sorted((n for n in trial if n['hand'] == 1 and n['id'] in (280, 282, 284)), key=lambda n: n['time'])
        clearance = post.clearance_violations(trial)
        overspeed = any(post.calc_note_speed([a[k] for k in ('x', 'y', 'hand', 'd')],
                                             [b[k] for k in ('x', 'y', 'hand', 'd')], b['time'] - a['time']) > passage['provenance']['arrival_cap']
                        for a, b in zip(pair, pair[1:]))
        assert overspeed or any(clearance.values())


def test_r7_joint_repair_preserves_records_and_both_boundary_arrows(passage):
    notes = deepcopy(passage['notes'])
    before = deepcopy(notes)
    stats = _repair(notes, lambda n: passage['arrival_caps'][str(n['id'])], total_notes=1480, diagonal_count=210)
    assert _horizontal_count([n for n in notes if n['hand'] == 1]) == 0
    assert [{k: v for k, v in n.items() if k != 'd'} for n in notes] == [{k: v for k, v in n.items() if k != 'd'} for n in before]
    assert {n['id']: n['d'] for n in notes if n['id'] in (272, 298)} == {272: 2, 298: 5}
    assert post.clearance_violations(notes) == {'same_cell_pairs': 0, 'swing_into_pairs': 0, 'flow_repeats': 0}
    assert stats['m1_direction_changed'] >= 11
    assert stats['m1_diagonal_bridges'] >= 1
    again = deepcopy(before)
    assert _repair(again, lambda n: passage['arrival_caps'][str(n['id'])], total_notes=1480, diagonal_count=210) == stats
    assert again == notes


def test_r7_no_diagonal_budget_keeps_incumbent(passage):
    notes = deepcopy(passage['notes'])
    before = deepcopy(notes)
    _repair(notes, lambda n: 26.6666666667, total_notes=20, diagonal_count=3)
    assert notes == before


@pytest.mark.parametrize('directions,gap', [([0, 1, 0, 1, 0], .2), ([2, 3, 2, 3, 2], .251), ([2, 3, 8, 2, 3], .2)])
def test_nontrigger_phrases_stay_unchanged(directions, gap):
    notes = [dict(id=i, time=1 + i * gap, hand=0, x=0, y=1, d=d) for i, d in enumerate(directions)]
    before = deepcopy(notes)
    _repair(notes, lambda n: float('inf'))
    assert notes == before


def test_primary_speed_pass_repairs_before_chain_sync_and_frozen_pass_preserves_it(passage, monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 2.)
    monkeypatch.setattr(config, 'v2_speed_cap_start_notes', 0)
    # Whole-map budget without local assets: far-away settled vertical notes.
    notes = deepcopy(passage['notes'])
    notes.extend(dict(id=10000+i, time=1000.+i, hand=0, x=0, y=1, d=i%2)
                 for i in range(100))
    chains = [dict(id=20000, anchor_id=274, x=2, y=2, d=3)]
    stats = post.apply_speed_cap(notes, chains)
    assert stats['m1_direction_changed'] > 0
    assert _horizontal_count([n for n in notes if n['hand'] == 1]) == 0
    assert chains[0]['d'] == next(n['d'] for n in notes if n['id'] == 274)
    frozen = deepcopy(notes)
    second = post.apply_speed_cap(notes, chains, drop_retained=False)
    assert notes == frozen
    assert 'm1_direction_changed' not in second


def test_disabled_speed_cap_still_allows_explicit_direction_experiment(passage, monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)
    notes = deepcopy(passage['notes'])
    notes.extend(dict(id=10000+i, time=1000.+i, hand=0, x=0, y=1, d=i%2)
                 for i in range(100))
    stats = post.apply_speed_cap(notes)
    assert stats['speed_cap_dropped'] == 0
    assert _horizontal_count([n for n in notes if n['hand'] == 1]) == 0


def test_experiment_off_preserves_rng_and_map(passage, monkeypatch):
    import numpy as np
    monkeypatch.setattr(config, 'v2_playability_enabled', False, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)
    notes = deepcopy(passage['notes'])
    rng = np.random.default_rng(3)
    before_rng = deepcopy(rng.bit_generator.state)
    post.apply_speed_cap(notes, rng=rng)
    assert notes == passage['notes']
    assert rng.bit_generator.state == before_rng


def test_experiment_runtime_rejects_rhythm_or_identity_mutation(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)
    def corrupt(notes, cap, **kwargs):
        notes[0]['time'] += .01
        return {}
    monkeypatch.setattr(post, 'repair_playability_directions', corrupt)
    with pytest.raises(RuntimeError, match='identity|rhythm'):
        post.apply_speed_cap([dict(id=0, time=1., hand=0, x=0, y=0, d=0)])


def test_isolated_recut_prefers_legal_nonhorizontal_and_keeps_infeasible_incumbent(passage):
    from map_creation import v2_playability_directions as directions
    repair = getattr(directions, 'repair_isolated', None)
    assert callable(repair), 'Separate isolated-recut experiment is missing'
    notes = deepcopy(passage['notes'])
    before = deepcopy(notes)
    stats = repair(notes, lambda n: 26.6666666667, eligible_ids={276, 282}, total_notes=1480)
    assert next(n['d'] for n in notes if n['id'] == 276) in (4, 6)
    assert next(n['d'] for n in notes if n['id'] == 282) == 3
    assert stats['isolated_direction_changed'] == 1
    assert stats['isolated_horizontal_after'] < stats['isolated_horizontal_before']
    assert [{k:v for k,v in n.items() if k != 'd'} for n in notes] == [{k:v for k,v in n.items() if k != 'd'} for n in before]
    assert post.clearance_violations(notes) == {'same_cell_pairs': 0, 'swing_into_pairs': 0, 'flow_repeats': 0}


def test_isolated_arm_only_reranks_actual_primary_speed_recuts(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', False, raising=False)
    monkeypatch.setattr(config, 'v2_playability_isolated', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_start_notes', 0)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 2.)
    monkeypatch.setattr(config, 'max_speed', 20.)
    notes = [dict(id=0, time=1., hand=0, x=0, y=1, d=2),
             dict(id=1, time=1.01, hand=0, x=3, y=1, d=3),
             dict(id=2, time=1.2, hand=0, x=0, y=1, d=2),
             dict(id=3, time=1.4, hand=0, x=0, y=1, d=2)]
    notes.extend(dict(id=10000+i, time=1000.+i, hand=0, x=0, y=1, d=i%2) for i in range(100))
    stats = post.apply_speed_cap(notes)
    assert stats['speed_cap_recuts'] >= 1
    assert stats['isolated_direction_changed'] == 1
    assert next(n['d'] for n in notes if n['id'] == 2) in (4, 5, 6, 7)
    assert next(n['d'] for n in notes if n['id'] == 3) == 2


@pytest.mark.parametrize('gap,changed', [(.25, True), (.25001, False)])
def test_exact_trigger_boundary_and_overlapping_windows(gap, changed):
    notes = [dict(id=i, time=1.+i*gap, hand=0, x=0, y=1, d=2+i%2) for i in range(38)]
    before = deepcopy(notes)
    result = _repair(notes, lambda n: float('inf'), total_notes=1000)
    assert (result['m1_direction_changed'] > 0) is changed
    if changed:
        assert result['m1_windows'] >= 3
        assert _horizontal_count(notes) < _horizontal_count(before)
        assert result['m1_diagonal_net'] <= 10
        assert not any(post.clearance_violations(notes).values())
    again = deepcopy(before)
    assert _repair(again, lambda n: float('inf'), total_notes=1000) == result
    assert again == notes


def test_dot_boundary_and_both_stack_arrows_are_frozen():
    directions = [2, 3, 2, 3, 2, 3, 8, 3]
    notes = [dict(id=i, time=1.+i*.25, hand=0, x=0, y=1, d=d) for i,d in enumerate(directions)]
    before = deepcopy(notes)
    _repair(notes, lambda n: float('inf'), total_notes=1000)
    assert [n['d'] for n in notes[5:]] == [3, 8, 3]
    assert not any(post.clearance_violations(notes).values())
    stacked = deepcopy(before)
    stacked[3]['time'] = stacked[2]['time'] + .01
    _repair(stacked, lambda n: float('inf'), total_notes=1000)
    assert [stacked[i]['d'] for i in (2,3)] == [2,3]


def test_primary_original_and_future_frozen_budgets_both_bound_repairs(monkeypatch):
    monkeypatch.setattr(config, 'add_beat_intensity_orig', 100., raising=False)
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_start_notes', 4)
    monkeypatch.setattr(config, 'v2_speed_cap_start_factor', .5)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 2.)
    monkeypatch.setattr(config, 'max_speed', 20.)
    observed = {}
    def inspect_budgets(notes, cap, **kwargs):
        observed.update({n['id']: float(cap(n)) for n in notes})
        return {}
    monkeypatch.setattr(post, 'repair_playability_directions', inspect_budgets)
    notes = [dict(id=0, time=1., hand=0, x=0, y=1, d=2),
             dict(id=1, time=1.001, hand=0, x=3, y=1, d=3),
             dict(id=2, time=2., hand=0, x=0, y=1, d=3),
             dict(id=3, time=3., hand=0, x=0, y=1, d=2)]
    for n in notes:
        n['kind'] = 'retained'
    post.apply_speed_cap(notes)
    assert set(observed) == {0, 2, 3}
    # Each ID retains its original upper bound; the tighter surviving-ramp
    # budget also protects the later frozen pass after the primary deletion.
    assert observed[2] == pytest.approx(20.*(2./3.) * post.get_factor_from_max_speed(20.*(2./3.), .5, 1) * 2.)
    assert observed[3] == pytest.approx(20.*(5./6.) * post.get_factor_from_max_speed(20.*(5./6.), .5, 1) * 2.)


def test_combined_direction_arms_share_one_map_diagonal_growth_budget(passage, monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_m1', True, raising=False)
    monkeypatch.setattr(config, 'v2_playability_isolated', True, raising=False)
    notes = deepcopy(passage['notes'])
    notes.extend(dict(id=5000+i, time=500.+i*.22, hand=1, x=0, y=1, d=2+i%2) for i in range(3))
    notes.extend(dict(id=10000+i, time=1000.+i, hand=0, x=0, y=1, d=i%2) for i in range(100))
    diagonals = sum(n['d'] in (4,5,6,7) for n in notes)
    stats = {}
    post._apply_playability_experiments(notes, lambda n:26.6666666667, stats, isolated_ids={5001})
    assert stats['m1_direction_changed'] > 0
    assert sum(n['d'] in (4,5,6,7) for n in notes) - diagonals <= 1
