"""Combined reach policy: calibrated tails and count-preserving hard repair."""

from copy import deepcopy
import importlib
import importlib.util
import json
import math

import pytest

from map_creation import v2_postprocessing as post
from tools.config import config


def reach():
    name = "map_creation.v2_playability_reach"
    assert importlib.util.find_spec(name) is not None, "combined reach repair is missing"
    return importlib.import_module(name)


def note(identity, time, hand, x, y, direction, **extra):
    return dict(id=identity, time=time, hand=hand, x=x, y=y, d=direction,
                kind="retained", custom={"keep": identity}, **extra)


@pytest.fixture(autouse=True)
def strict_flow(monkeypatch):
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)
    monkeypatch.setattr(config, "min_time_diff", .02)
    monkeypatch.setattr(config, "v2_cleanup_same_dir_gap", .1)
    monkeypatch.setattr(config, "v2_cleanup_swing_window", .15)
    monkeypatch.setattr(config, "v2_fast_counter_direction", "recut")
    monkeypatch.setattr(config, "v2_fast_gap_seconds", .2)


@pytest.mark.parametrize("hand,x,banned", [(0, 3, (2, 4, 6)), (1, 0, (3, 5, 7))])
def test_forbidden_cut_is_exact_and_mirrored(hand, x, banned):
    policy = reach()
    n = note(0, 1., hand, x, 1, 0)
    for direction in range(9):
        assert policy.forbidden_cut(n, direction=direction) == (direction in banned)
        assert not policy.forbidden_cut(n, direction=direction, x=1)
    assert n["d"] == 0


def test_calibrated_recovery_endpoint_and_travel_formulas_keep_dots_honest():
    policy = reach()
    notes = [note(0, 1., 0, 3, 2, 4), note(1, 1.25, 0, 2, 0, 1),
             note(2, 1.5, 0, 3, 2, 8), note(3, 1.75, 0, 3, 2, 2)]
    original = deepcopy(notes)
    result = policy.reach_costs(notes, nps=4.)
    tau = .3571428571428612
    weight = 1 + tau / (.25 + tau)
    metrics = result["metrics"]
    assert metrics["nominal_approach_excursion"]["severities"] == pytest.approx([2., 0., 2.*weight])
    assert metrics["vertical_preparation_excursion"]["severities"] == pytest.approx([0., 0., 0.])
    assert metrics["nominal_center_reach"]["severities"] == pytest.approx([1., 0., weight, weight])
    assert metrics["recovery_adjusted_travel"]["severities"] == pytest.approx([2./(.25+tau)])
    assert metrics["nominal_approach_excursion"]["cutoff"] == 1.4063604240282719
    assert result["hard_bans"] == 2
    assert notes == original
    json.dumps(result)


def test_local_components_expose_independent_tail_guards_without_virtual_dot_endpoints():
    policy = reach()
    previous = note(0,1.,0,2,1,3)
    current = note(1,1.2,0,3,0,2)
    components = policy.local_reach_components(previous,current,4.)
    assert components["nominal_approach_excursion"] > 0.
    assert policy.local_reach_cost(previous,current,4.) == sum(components.values())
    current["d"] = 8
    components = policy.local_reach_components(previous,current,4.)
    assert set(components) == {"nominal_center_reach"}
    assert components["nominal_center_reach"] > 0.


def test_exact_stack_groups_use_best_arrow_pair_and_exposure_carries_through_dots():
    policy = reach()
    notes = [note(0, 1., 0, 0, 1, 2), note(1, 1., 0, 2, 1, 3),
             note(2, 1.25, 0, 1, 1, 3), note(3, 1.5, 0, 1, 1, 8),
             note(4, 1.75, 0, 1, 1, 2)]
    metrics = policy.reach_costs(list(reversed(notes)), nps=4.)["metrics"]
    tau = .3571428571428612
    assert metrics["recovery_adjusted_travel"]["severities"] == pytest.approx([1/(.25+tau)])
    assert len(metrics["repeated_excursion_exposure"]["severities"]) == 3


def test_original_first_hard_only_moves_the_cell_and_preserves_the_arrow():
    policy = reach()
    notes = [note(9, 2., 1, 0, 2, 3), note(3, 1., 0, 1, 0, 0)]
    original = deepcopy(notes)
    result = policy.repair_reach(notes, lambda _: math.inf, nps=4., hard_only=True)
    assert (notes[0]['x'], notes[0]['y'], notes[0]['d']) == (1, 2, 3)
    assert notes[1] == original[1]
    assert result['reach_hard_bans_after'] == 0
    assert result['reach_changed'] == 1
    assert result['reach_candidates'] < 10


def test_hard_only_keeps_all_unbanned_notes_even_with_soft_tail_events():
    policy = reach()
    notes = [note(i, 1.+i*.25, 0, 3, 2, 0 if i % 2 == 0 else 1) for i in range(8)]
    original = deepcopy(notes)
    policy.repair_reach(notes, lambda _: math.inf, nps=4., hard_only=True)
    assert notes == original


def test_hard_repair_accepts_same_hand_simultaneous_stack_at_finite_cap():
    policy = reach()
    notes = [note(0, 1., 1, 0, 0, 3), note(1, 1., 1, 0, 1, 3)]
    result = policy.repair_reach(notes, lambda _: 44.16, nps=4., hard_only=True)
    assert result['reach_hard_after'] == 0
    assert [(n['x'], n['d']) for n in notes] == [(1, 3), (1, 3)]


def test_reverse_dead_end_reopens_only_the_immediately_later_note():
    # Portable original Slow passage, 54.75-56.07 s; settled arrival cap44.16.
    rows = [(350,54.751710990968476,0,0,2,5), (351,54.751710990968476,1,3,1,0),
            (352,54.99646258503402,0,2,0,6), (353,55.06422075689865,1,3,0,6),
            (354,55.27672739773118,0,1,2,0), (355,55.27672739773118,1,0,1,3),
            (356,55.35172974155442,1,2,2,5), (357,55.426732085377665,0,1,0,1),
            (358,55.50173442920091,1,3,0,1), (359,55.5653514739229,0,0,0,0),
            (360,55.6517391168474,1,1,1,0), (361,55.73950113378685,0,2,0,1),
            (362,55.80174380449389,1,3,1,1), (363,55.87674614831713,0,0,1,4),
            (364,56.00175005468921,1,1,2,0), (365,56.06425200787525,0,2,2,3)]
    notes = [note(*row) for row in rows]
    original = deepcopy(notes)
    result = reach().repair_reach(notes, lambda _: 44.16, nps=8., hard_only=True)
    assert result['reach_backtracks'] == [dict(note_id=355, time=rows[5][1], later_groups=1)]
    assert {n['id'] for n, old in zip(notes, original) if n != old} == {355, 356}
    assert result['reach_hard_after'] == 0
    assert not any(result['reach_clearance_after'].values())


def test_raw_endpoint_events_preserve_chronological_identity_order_across_hands():
    policy = reach()
    notes = [note(91,1.,1,0,1,3),note(4,1.1,0,3,1,2),note(78,1.2,1,0,1,8)]
    metrics = policy.reach_costs(list(reversed(notes)),nps=4.)["metrics"]
    assert metrics["nominal_approach_excursion"]["event_note_ids"] == [[91],[4]]
    assert metrics["nominal_center_reach"]["event_note_ids"] == [[91],[4],[78]]


@pytest.mark.parametrize("hand,x,direction", [(0, 3, 2), (0, 3, 4), (0, 3, 6),
                                             (1, 0, 3), (1, 0, 5), (1, 0, 7)])
def test_joint_repair_preserves_fields_and_clears_all_banned_families(hand, x, direction):
    policy = reach()
    notes = [note(42, 2., hand, x, 1, direction)]
    before = deepcopy(notes)
    report = policy.repair_reach(notes, lambda _: math.inf, nps=4.)
    assert not policy.forbidden_cut(notes[0])
    assert notes[0]["d"] != 8
    assert {k:v for k,v in notes[0].items() if k not in ("x", "y", "d")} == {
        k:v for k,v in before[0].items() if k not in ("x", "y", "d")}
    assert report["reach_hard_bans_after"] == 0
    assert report["reach_changed"] == 1
    json.dumps(report)


def test_outgoing_arrival_cap_is_checked_with_unsorted_input_and_both_hands():
    policy = reach()
    notes = [note(8, 2., 0, 3, 1, 2), note(3, 1.5, 0, 2, 1, 3),
             note(10, 2.5, 0, 2, 1, 3), note(9, 2., 1, 0, 0, 0)]
    identities = [id(n) for n in notes]
    # Arrival budget for the fixed successor is deliberately much tighter.
    cap = lambda n: 12. if n["id"] == 10 else 100.
    policy.repair_reach(notes, cap, nps=4.)
    ordered = sorted((n for n in notes if n["hand"] == 0), key=lambda n:n["time"])
    assert [id(n) for n in notes] == identities
    assert not any(policy.forbidden_cut(n) for n in notes)
    for a,b in zip(ordered, ordered[1:]):
        assert post.calc_note_speed([a[k] for k in ("x","y","hand","d")],
                                    [b[k] for k in ("x","y","hand","d")], b["time"]-a["time"]) <= cap(b)+1e-9
    assert not any(post.clearance_violations(notes).values())


def test_dot_history_and_stack_occupancy_remain_legal():
    policy = reach()
    notes = [note(0, 1., 0, 2, 1, 3), note(1, 1.4, 0, 3, 1, 2),
             note(2, 1.4, 1, 0, 1, 0), note(3, 1.8, 0, 1, 1, 8),
             note(4, 2.2, 0, 1, 0, 1)]
    policy.repair_reach(notes, lambda _: math.inf, nps=4.)
    assert not any(policy.forbidden_cut(n) for n in notes)
    assert notes[3]["d"] == 8
    assert not any(post.clearance_violations(notes).values())


def test_frozen_retained_hard_ban_fails_before_any_mutation():
    policy = reach()
    notes = [note(0, 1., 0, 3, 1, 2), note(1, 1.5, 0, 2, 1, 3)]
    original = deepcopy(notes)
    with pytest.raises(ValueError, match="retained"):
        policy.repair_reach(notes, lambda _: math.inf, protect_retained=True)
    assert notes == original


def test_final_repair_changes_mutable_copy_but_preserves_retained_anchor():
    policy = reach()
    anchor = note(0, 1., 0, 2, 1, 3)
    copied = note(1, 1.4, 0, 3, 1, 2)
    copied["kind"] = "copy"
    notes = [anchor, copied]
    frozen = deepcopy(anchor)
    policy.repair_reach(notes, lambda _: math.inf, nps=4., protect_retained=True)
    assert anchor == frozen
    assert not policy.forbidden_cut(copied)


def test_exhausted_hard_search_fails_closed_and_rolls_back_earlier_repairs():
    policy = reach()
    notes = [note(0,1.,0,3,1,2),note(1,4.,1,2,1,2),note(2,5.,1,0,1,3)]
    notes[0]["kind"] = notes[2]["kind"] = "copy"
    before = deepcopy(notes)
    cap = lambda n:0. if n["id"] == 2 else math.inf
    with pytest.raises(ValueError,match="reach repair"):
        policy.repair_reach(notes,cap,nps=4.,protect_retained=True)
    assert notes == before


def test_normal_range_variation_is_preserved_and_solver_consumes_no_randomness():
    import random
    import numpy as np
    policy = reach()
    notes = [note(0, 1., 0, 0, 1, 5), note(1, 1., 1, 3, 1, 4),
             note(2, 2., 0, 1, 1, 6), note(3, 2., 1, 2, 1, 7)]
    before, py_state, np_state = deepcopy(notes), random.getstate(), np.random.get_state()
    policy.repair_reach(notes, lambda _: math.inf, nps=4.)
    assert notes == before
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])


def test_night_raid_both_actual_witnesses_and_neighbors_are_repaired(monkeypatch):
    """Archived played Dense A: raw elapsed seconds, both hands retained.

    Source: locked-fixed .../01/Expert/11/package/Expert.dat, 2026-11-08.
    Include both 75.337 left x3/d6 and 76.873 left x3/d2; these need joint
    geometry/direction choices in the context of ordinary surrounding cuts.
    """
    policy = reach()
    monkeypatch.setattr(config,"v2_playability_enabled",True)
    monkeypatch.setattr(config,"v2_playability_slow90_seconds",.5)
    monkeypatch.setattr(config,"v2_playability_slow45_seconds",.65)
    rows = [
        (74.764164762407,0,0,2,2),(74.88134997363335,1,1,0,1),
        (74.99853518485968,0,2,0,7),(75.07665865901056,1,2,2,0),
        (75.15478213316146,0,1,2,0),(75.22104308390023,1,1,0,6),
        (75.33714285714287,0,3,1,6),(75.45324263038549,1,3,2,0),
        (75.58446124099139,0,2,2,0),(75.68544217687075,1,1,0,1),
        (75.80154195011337,0,3,0,7),(75.91764172335601,1,3,1,5),
        (76.0453514739229,0,1,2,0),(76.1703872971231,1,1,0,6),
        (76.248510771274,0,2,0,1),(76.40475771957578,0,0,1,4),
        (76.48288119372668,1,0,0,3),(76.6391281420285,1,1,2,2),
        (76.71725161617937,0,3,0,7),(76.79537509033027,1,2,0,7),
        (76.87349856448117,0,3,0,2),(76.95162203863205,1,1,0,0),
        (77.02974551278295,0,2,0,7),(77.10786898693385,1,3,2,1),
        (77.21795918367347,0,2,2,0),(77.34223940938654,1,3,0,0)]
    notes = [note(i,*row) for i,row in enumerate(rows)]
    before = deepcopy(notes)
    clearance = post.clearance_violations(notes)
    policy.repair_reach(notes,lambda _:100.,nps=6.)
    assert not any(policy.forbidden_cut(n) for n in notes)
    assert len(notes) == len(before)
    assert [n['time'] for n in notes] == [n['time'] for n in before]
    assert [n['hand'] for n in notes] == [n['hand'] for n in before]
    assert all(post.clearance_violations(notes)[k] <= v for k,v in clearance.items())


def test_soft_reach_changes_stop_at_human_band_arrow_mix_limit():
    policy = reach()
    notes = [note(i,1.+.15*i,0,1,0 if i%2 == 0 else 2,i%2) for i in range(80)]
    report = policy.repair_reach(notes,lambda _:100.,nps=6.)
    assert report["reach_excess_after"] < report["reach_excess_before"]
    assert sum(n["d"] in (4,5,6,7) for n in notes) <= math.floor(len(notes)*policy.diagonal_fraction_limit(6.))


def test_revised_crossing_cost_has_no_reward_inside_human_normal_range():
    policy = reach()
    notes = [note(0,1.,0,2,1,0),note(1,1.1,1,1,1,0)]
    window = policy._Window(notes,4.,post)
    assert window.geometry()[1] == 0.


def test_joint_arrow_rank_penalizes_only_calibrated_m1_tail():
    policy = reach()
    notes = [note(0,1.,0,1,1,7),note(1,1.1,0,1,1,4)]
    window = policy._Window(notes,4.,post)
    # The ordinary fast diagonal is below the supported human cutoff.
    assert window.lateral_excess() == 0.
    notes[1]["d"] = 2
    assert window.lateral_excess() == pytest.approx(.8-policy.calibrated_cutoff("m1",4.))


def test_joint_window_broadens_when_no_single_note_meets_both_arrival_caps():
    policy = reach()
    notes = [note(0,1.,0,0,0,3),note(1,1.3,0,3,0,2),note(2,1.6,0,0,0,5)]
    state = policy._Search(notes,lambda _:10.,4.,False)
    original = {i:(n["x"],n["y"],n["d"]) for i,n in enumerate(notes)}
    for option in state.options(1,broad=True):
        state.install({1:option})
        assert not state.legal({1})
        state.install(original)
    report = policy.repair_reach(notes,lambda _:10.,nps=4.)
    assert report["reach_broadened_windows"] == 1
    assert report["reach_changed"] >= 2
    assert not any(policy.forbidden_cut(n) for n in notes)
    assert not any(post.clearance_violations(notes).values())
    for a,b in zip(notes,notes[1:]):
        assert post.calc_note_speed([a[k] for k in ("x","y","hand","d")],
                                    [b[k] for k in ("x","y","hand","d")], b["time"]-a["time"]) <= 10.+1e-9
