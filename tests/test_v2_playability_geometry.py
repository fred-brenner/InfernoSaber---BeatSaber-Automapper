"""Opt-in position repair preserves production safety and scorer coverage."""

from copy import deepcopy
import json
import math

import pytest

from evaluation.playability_metrics import prepare_notes, score_map
from map_creation import v2_postprocessing as post
from tools.config import config


def note(identity, time, hand, x, y, direction):
    return dict(id=identity, time=time, hand=hand, x=x, y=y, d=direction,
                kind="retained", custom={"keep": identity})


def score(notes):
    return score_map([(n["time"], n["hand"], n["x"], n["y"], n["d"]) for n in notes])


def repair(notes, cap=lambda _: math.inf, **switches):
    from map_creation.v2_playability_geometry import repair_geometry
    return repair_geometry(notes, cap, sightline=switches.get("sightline", False),
                           coordination=switches.get("coordination", False),
                           interference=switches.get("interference", False))


def stable_fields(notes):
    return [{k: v for k, v in n.items() if k not in ("x", "y")} for n in notes]


@pytest.fixture(autouse=True)
def strict_flow(monkeypatch):
    """Keep isolated geometry's legacy flow and scorer contracts explicit."""
    from evaluation.playability_experiments import SWITCHES
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    for field, value in SWITCHES.items():
        monkeypatch.setattr(config, "v2_playability_" + field, value)
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)
    monkeypatch.setattr(config, "min_time_diff", .02)
    monkeypatch.setattr(config, "v2_cleanup_same_dir_gap", .1)
    monkeypatch.setattr(config, "v2_cleanup_swing_window", .15)


def test_disabled_geometry_is_exact_noop():
    notes = [note(9, 1., 0, 0, 0, 1), note(17, 1.2, 1, 0, 2, 0)]
    original = deepcopy(notes)
    report = repair(notes, lambda _: pytest.fail("disabled repair queried cap"))
    assert notes == original
    assert report["geometry_changed"] == 0
    json.dumps(report)


def test_sightline_hidden_rise_improves_without_identity_or_metric_damage():
    notes = [note(9, 1., 0, 0, 0, 1), note(17, 1.2, 1, 0, 2, 0)]
    before, identities = score(notes), stable_fields(notes)
    report = repair(notes, sightline=True)
    after = score(notes)
    assert after["m2"]["value"] < before["m2"]["value"]
    assert after["m4"]["value"] <= before["m4"]["value"]
    assert stable_fields(notes) == identities
    assert report["geometry_sightline_changed"] > 0
    assert post.clearance_violations(notes) == dict(same_cell_pairs=0, swing_into_pairs=0, flow_repeats=0)
    json.dumps(report)


@pytest.mark.parametrize("later", [1., 1.401])
def test_sightline_keeps_simultaneous_pair_and_distant_rise(later):
    notes = [note(0, 1., 0, 0, 0, 1), note(1, later, 1, 0, 2, 0)]
    original = deepcopy(notes)
    repair(notes, sightline=True)
    assert notes == original


def test_paths_improve_legal_travel_corridor_clash():
    # Standing blocks/sweeps are legal; left's earlier exit-to-approach
    # corridor crosses the right-hand volume at 1.2 s.
    notes = [note(0, 1., 0, 0, 0, 0), note(1, 1.2, 1, 2, 0, 0),
             note(2, 1.3, 0, 2, 2, 1)]
    before = score(notes)
    assert sum(post.clearance_violations(notes).values()) == 0
    assert before["m4"]["value"] > 0
    report = repair(notes, interference=True)
    after = score(notes)
    assert after["m4"]["value"] < before["m4"]["value"]
    assert after["m2"]["value"] <= before["m2"]["value"]
    assert report["geometry_interference_changed"] > 0
    assert sum(post.clearance_violations(notes).values()) == 0


def test_paths_preserve_playable_mirrored_double_on_tie():
    notes = [note(0, 1., 0, 1, 1, 5), note(1, 1., 1, 2, 1, 4)]
    original = deepcopy(notes)
    assert score(notes)["m4"]["value"] == 0
    repair(notes, interference=True)
    assert notes == original


def test_coordination_improves_orthogonal_axes_without_scoring_coverage_loss():
    notes = []
    for i in range(5):
        notes += [note(2*i, 1.+.3*i, 0, 0, 2*(i%2), i%2),
                  note(2*i+1, 1.+.3*i, 1, 2+i%2, 1, i%2)]
    before, stable = score(notes), stable_fields(notes)
    assert before["m3"]["value"] == 1
    report = repair(notes, coordination=True)
    after = score(notes)
    assert after["m3"]["value"] < before["m3"]["value"]
    assert after["m3"]["segments"] == before["m3"]["segments"] == 1
    assert stable_fields(notes) == stable
    assert report["geometry_coordination_changed"] > 0
    assert sum(post.clearance_violations(notes).values()) == 0


def test_coordination_keeps_mirrored_diagonal_wave():
    notes = []
    for i in range(5):
        notes += [note(2*i, 1.+.3*i, 0, i%2, 1+i%2, i%2),
                  note(2*i+1, 1.+.3*i, 1, 3-i%2, 1+i%2, i%2)]
    original = deepcopy(notes)
    repair(notes, coordination=True)
    assert notes == original


def test_geometry_never_exceeds_original_per_id_arrival_caps():
    notes = [note(7, 1., 0, 0, 0, 0), note(23, 1.3, 0, 0, 2, 1),
             note(91, 1.31, 1, 2, 1, 0), note(109, 1.6, 0, 0, 0, 0)]
    limits = {7: math.inf, 23: 12., 91: math.inf, 109: 13.}
    # Tight outgoing cap can veto an otherwise attractive sightline move.
    repair(notes, lambda n: limits[n["id"]], sightline=True, interference=True)
    for hand in (0, 1):
        seq = [n for n in notes if n["hand"] == hand]
        for previous, arriving in zip(seq, seq[1:]):
            speed = post.calc_note_speed([previous[k] for k in ("x", "y", "hand", "d")],
                                         [arriving[k] for k in ("x", "y", "hand", "d")],
                                         arriving["time"]-previous["time"])
            assert speed <= limits[arriving["id"]] + 1e-9


def test_geometry_is_repeatable_and_one_cell_per_proposal():
    original = [note(0, 1., 0, 0, 0, 1), note(1, 1.2, 1, 0, 2, 0)]
    first, second = deepcopy(original), deepcopy(original)
    report = repair(first, sightline=True, interference=True, coordination=True)
    assert repair(second, sightline=True, interference=True, coordination=True) == report
    assert first == second
    for old, new in zip(original, first):
        assert abs(old["x"]-new["x"]) + abs(old["y"]-new["y"]) <= 1


def test_sightline_proposal_rechecks_following_nearest_occluder():
    from map_creation.v2_playability_geometry import _Geometry
    notes = [note(0, 1., 0, 0, 0, 0), note(1, 1.2, 1, 0, 1, 2),
             note(2, 1.23, 0, 0, 2, 1)]
    original = deepcopy(notes)
    geometry = _Geometry(notes, lambda _: math.inf)
    # Lowering the middle block clears its own rise but doubles the next
    # rise. The nearer next event dominates the net obstruction cost.
    assert geometry.proposal({1: (0, 0)}) is None
    assert notes == original


@pytest.mark.parametrize("proposal", [{1: (1, 2)}, {0: (1, 0)}, {2: (1, 0)}])
def test_geometry_rejects_original_arrival_budget_violation(proposal):
    from map_creation.v2_playability_geometry import _Geometry
    notes = [note(7, 1., 0, 0, 0, 0), note(23, 1.3, 0, 0, 2, 1),
             note(109, 1.6, 0, 0, 0, 0)]
    original = deepcopy(notes)
    # Original two edges cost 12.67. Either neighboring-cell proposal makes
    # one edge exceed its arriving note's original, ID-keyed budget.
    geometry = _Geometry(notes, lambda n: {7: math.inf, 23: 13., 109: 13.}[n["id"]])
    assert geometry.proposal(proposal, commit=True) is None
    assert notes == original


def test_geometry_rejects_incoming_swing_into_relocated_note():
    from map_creation.v2_playability_geometry import _Geometry
    notes = [note(0, 1., 0, 0, 0, 0), note(1, 1.1, 1, 1, 1, 0)]
    original = deepcopy(notes)
    geometry = _Geometry(notes, lambda _: math.inf)
    # The earlier left-hand up cut sweeps (0, 1) while this right-hand
    # block is still standing. Checking only the moved note's swing misses it.
    assert geometry.proposal({1: (0, 1)}) is None
    assert notes == original


def test_coordination_rejects_loss_of_resolved_axis_or_movement_support():
    from map_creation.v2_playability_geometry import _Geometry
    notes = []
    for i in range(3):
        notes += [note(2*i, 1.+.3*i, 0, 0, i%2, i%2),
                  note(2*i+1, 1.+.3*i, 1, 2+i%2, 2, i%2)]
    original = deepcopy(notes)
    geometry = _Geometry(notes, lambda _: math.inf)
    # Making all right-hand positions identical would remove its axis and
    # turn the scorer's absent segment into a misleading zero.
    assert geometry.proposal({3: (2, 2)}) is None
    assert notes == original


@pytest.mark.parametrize("tuples", [
    [(1., 0, 0, 0, 0), (1.01, 0, 1, 1, 1), (1.2, 1, 1, 2, 2),
     (1.3, 0, 2, 2, 8), (1.5, 1, 3, 0, 3), (1.6, 0, 1, 0, 8)],
    [(1., 0, 0, 0, 0), (1.2, 1, 2, 0, 0), (1.3, 0, 2, 2, 1)],
    [(1., 0, 1, 1, 5), (1., 1, 2, 1, 4)],
    [(1., 0, 0, 0, 8), (1.3, 1, 3, 2, 8), (1.401, 0, 1, 2, 8)],
    [(1.+.3*i, h, (i%2 if h == 0 else 3-i%2), 1+i%2, i%2)
     for i in range(5) for h in (0, 1)],
])
def test_local_costs_match_offline_scorer_geometry(tuples):
    from map_creation.v2_playability_geometry import _Geometry
    notes = [note(i, *t) for i, t in enumerate(tuples)]
    scorer = score(notes)
    local = _Geometry(notes, lambda _: math.inf).costs()
    span = scorer["span_seconds"]
    assert local["m2"] == pytest.approx(scorer["m2"]["value"])
    assert local["m4_severity"] == pytest.approx(scorer["m4"]["value"]*span/60)
    assert local["m3"] == pytest.approx(scorer["m3"]["value"])
    assert local["m3_segments"] == scorer["m3"]["segments"]


def test_incremental_costs_match_exported_result_after_several_edits():
    notes = []
    for i in range(5):
        notes += [note(2*i, 1.+.3*i, 0, 0, 2*(i%2), i%2),
                  note(2*i+1, 1.+.3*i, 1, 2+i%2, 1, i%2)]
    report = repair(notes, coordination=True, sightline=True, interference=True)
    after = score(notes)
    local = report["geometry_after"]
    assert local["m2"] == pytest.approx(after["m2"]["value"])
    assert local["m3"] == pytest.approx(after["m3"]["value"])
    assert local["m4_severity"] == pytest.approx(after["m4"]["value"]*after["span_seconds"]/60)


def test_dot_effective_directions_propagate_to_following_corridors():
    from map_creation.v2_playability_geometry import _Geometry
    notes = [note(0, 1., 0, 0, 0, 0), note(1, 1.3, 0, 0, 0, 8),
             note(2, 1.35, 1, 3, 1, 0), note(3, 1.4, 0, 0, 0, 8),
             note(4, 1.7, 0, 0, 0, 1)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert geometry.effective[1] == 1
    assert geometry.proposal({0: (1, 0)}, commit=True) is not None
    # Geometry-derived pendulum changes must reach the complete dot chain.
    assert geometry.effective[1] == 6
    assert geometry.effective[3] == 5
    prepared = prepare_notes([(n["time"], n["hand"], n["x"], n["y"], n["d"]) for n in notes])
    assert [geometry.metric_effective[i] for i in geometry.kept] == [n["eff"] for n in prepared]
    actual, local = score(notes), geometry.costs()
    assert local["m2"] == pytest.approx(actual["m2"]["value"])
    assert local["m4_severity"] == pytest.approx(actual["m4"]["value"]*actual["span_seconds"]/60)
    assert sum(post.clearance_violations(notes).values()) == 0


def test_report_preserves_rows_lanes_hands_and_vector_support_for_review():
    notes = [note(0, 1., 0, 1, 1, 5), note(1, 1., 1, 2, 1, 4),
             note(2, 1.4, 0, 1, 1, 6), note(3, 1.4, 1, 2, 1, 7)]
    report = repair(notes, interference=True)
    before, after = report["geometry_before"], report["geometry_after"]
    assert before["lane_counts"] == [0, 2, 2, 0]
    assert before["row_counts"] == [0, 4, 0]
    assert before["hand_notes"] == [2, 2]
    assert before["stationary_moves"] == 2
    assert before["mirrored_doubles"] == after["mirrored_doubles"] == 2


def test_pair_window_float_boundary_has_symmetric_cached_cost():
    from map_creation.v2_playability_geometry import _Geometry
    # .251 - .25 is slightly greater than .001 in binary floats. Computing
    # the left/right neighborhoods separately used to miss the pair from
    # the later hand's cache and crash when the earlier participant moved.
    notes = [note(0, .001, 1, 3, 0, 0), note(1, .251, 0, 0, 2, 0)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert geometry.proposal({0: (2, 0)}) == pytest.approx((0., 0., 0.))
    assert geometry.proposal({0: (2, 0)}, commit=True) == pytest.approx((0., 0., 0.))
    assert geometry.costs()["m4_severity"] == 0.


def test_coordination_continues_after_a_window_used_by_other_geometry_arms():
    # Earlier visibility/path repairs consume all four notes in window 12–15.
    # Coordination must still consider the following window in this segment.
    tuples = [
        (1., 1, 2, 2, 8), (1.12, 0, 3, 2, 8), (1.24, 0, 2, 1, 8),
        (1.36, 1, 1, 0, 5), (1.48, 0, 0, 0, 7), (1.6, 0, 0, 2, 8),
        (1.72, 1, 3, 2, 6), (1.84, 1, 3, 1, 5), (1.96, 0, 2, 2, 3),
        (2.08, 1, 1, 1, 6), (2.2, 1, 3, 0, 8), (2.32, 1, 3, 0, 4),
        (2.44, 0, 3, 2, 2), (2.56, 0, 0, 2, 7), (2.68, 1, 2, 2, 3),
        (2.8, 1, 2, 0, 6), (2.92, 1, 2, 2, 3), (3.04, 0, 1, 1, 0),
        (3.16, 0, 2, 1, 7), (3.28, 0, 3, 0, 4), (3.4, 1, 3, 1, 2),
        (3.52, 0, 1, 2, 7), (3.64, 1, 1, 1, 7), (3.76, 0, 3, 1, 0),
        (3.88, 0, 3, 2, 7), (4., 0, 2, 2, 0), (4.12, 1, 1, 0, 4),
        (4.24, 0, 0, 0, 6), (4.36, 1, 1, 2, 7), (4.48, 1, 0, 0, 4),
        (4.6, 1, 0, 0, 1), (4.72, 0, 1, 0, 5),
    ]
    original = [note(i, *n) for i, n in enumerate(tuples)]
    first = deepcopy(original)
    repair(first, sightline=True, interference=True)
    assert all((first[i]["x"], first[i]["y"]) != (original[i]["x"], original[i]["y"])
               for i in range(12, 16))
    combined = deepcopy(original)
    report = repair(combined, sightline=True, interference=True, coordination=True)
    assert report["geometry_coordination_changed"] > 0
    assert any((combined[i]["x"], combined[i]["y"]) != (first[i]["x"], first[i]["y"])
               for i in range(16, 20))
    assert sum(post.clearance_violations(combined).values()) == 0
