"""Combined geometry repairs the played incidents without changing legacy costs."""

from copy import deepcopy
import math

import pytest

from map_creation import v2_postprocessing as post
from map_creation.v2_playability_geometry import _Geometry, repair_geometry
from tools.config import config


def note(identity, time, hand, x, y, direction):
    return dict(id=identity, time=time, hand=hand, x=x, y=y, d=direction)


@pytest.fixture(autouse=True)
def combined(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_reach", True, raising=False)
    monkeypatch.setattr(config, "min_time_diff", .02)
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)
    monkeypatch.setattr(config, "v2_cleanup_same_dir_gap", .1)
    monkeypatch.setattr(config, "v2_cleanup_swing_window", .15)


def test_geometry_v2_sees_same_height_crossed_upper_visibility():
    notes = [note(0, 112.195, 1, 1, 2, 4), note(1, 112.285, 0, 2, 2, 4),
             note(2, 112.374, 1, 1, 2, 1)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert geometry.m2[1] > .5
    assert geometry.m2[2] > 0


def test_geometry_v2_crossed_mirrored_double_has_interference():
    notes = [note(0, 89.629, 0, 2, 0, 0), note(1, 89.647, 1, 1, 0, 0)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert geometry.m4[(0, 1)] == pytest.approx(.928)
    assert geometry.costs()["mirrored_doubles"] == 0


def test_geometry_v2_joint_cell_swap_repairs_crossed_pair_without_extra_reach():
    notes = [note(0, 89.629, 0, 2, 0, 0), note(1, 89.647, 1, 1, 0, 0)]
    report = repair_geometry(notes, lambda _: math.inf, interference=True)
    assert notes[0]["x"] <= notes[1]["x"]
    assert report["geometry_after"]["m4_severity"] == 0.
    assert report["geometry_after"]["reach_local_excess"] == 0.
    assert sum(post.clearance_violations(notes).values()) == 0


def test_geometry_v2_keeps_normal_mirrored_upper_doubles_and_same_hand_stream():
    notes = [note(0, 1., 0, 1, 2, 4), note(1, 1., 1, 2, 2, 5),
             note(2, 1.1, 0, 1, 2, 7), note(3, 1.3, 0, 1, 2, 4)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert list(geometry.m2.values()) == [0., 0., 0., 0.]
    assert geometry.m4[(0, 1)] == 0.


def test_mirrored_pair_uses_the_same_overlap_score_as_every_other_pair():
    from evaluation.playability_metrics import prepare_notes, swing_volume
    from map_creation.v2_playability_costs import interference_severity
    # Ordinary hand order and mirrored arrows must not suppress two shared cells.
    notes = [note(0, 1., 0, 1, 1, 3), note(1, 1., 1, 2, 1, 2)]
    prepared = prepare_notes([tuple(n[k] for k in ('time', 'hand', 'x', 'y', 'd')) for n in notes])
    volumes = [swing_volume(n) for n in prepared]
    assert len(volumes[0] & volumes[1]) == 2
    assert interference_severity(notes[0], notes[1], 0., *volumes) == pytest.approx(2.)
    assert _Geometry(notes, lambda _: math.inf).m4[(0, 1)] == pytest.approx(2.)


def test_mirrored_pair_no_longer_clears_its_existing_travel_overlap():
    from evaluation.playability_metrics_v2 import score_map
    # The existing incoming corridors overlap along the two bottom middle cells.
    # Current arrows are parallel, symmetric up-cuts, but get no exemption.
    notes = [note(0, .8, 0, 2, 1, 1), note(1, .8, 1, 1, 1, 1),
             note(2, 1., 0, 1, 1, 0), note(3, 1., 1, 2, 1, 0)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert geometry.m4[(2, 3)] == pytest.approx(2.)
    scored = score_map([tuple(n[k] for k in ('time', 'hand', 'x', 'y', 'd')) for n in notes])
    incident = next(event for event in scored['m4']['incidents'] if event['t'] == 1. and event['dt'] == 0.)
    assert incident['cells'] == [(1, 0), (2, 0)]
    assert incident['severity'] == pytest.approx(2.)


def test_nonoverlapping_mirrored_pair_still_scores_zero_without_an_exemption():
    notes = [note(0, 1., 0, 0, 1, 0), note(1, 1., 1, 3, 1, 0)]
    assert _Geometry(notes, lambda _: math.inf).m4[(0, 1)] == 0.


def test_geometry_v2_disabled_retains_frozen_crossed_exception(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_reach", False)
    notes = [note(0, 89.629, 0, 2, 0, 0), note(1, 89.647, 1, 1, 0, 0)]
    assert _Geometry(notes, lambda _: math.inf).m4[(0, 1)] == 0.


def test_geometry_v2_repair_improves_played_visibility_without_hard_damage():
    notes = [note(0, 112.195, 1, 1, 2, 4), note(1, 112.285, 0, 2, 2, 4),
             note(2, 112.374, 1, 1, 2, 1), note(3, 113., 0, 0, 0, 1)]
    # Include the surrounding quiet tail: the three-note extract alone would
    # falsely select the 7+ NPS stratum for this otherwise moderate-density map.
    before = deepcopy(notes)
    report = repair_geometry(notes, lambda _: math.inf, sightline=True, interference=True)
    assert report["geometry_sightline_changed"] > 0
    assert report["geometry_after"]["m2"] < report["geometry_before"]["m2"]
    assert post.clearance_violations(notes) == post.clearance_violations(before)
    assert [tuple(n[k] for k in ("id", "time", "hand", "d")) for n in notes] == [
        tuple(n[k] for k in ("id", "time", "hand", "d")) for n in before]


def test_geometry_v2_coordination_stops_inside_human_normal_map_range():
    notes = []
    for segment in range(8):
        for i in range(5):
            time = 1 + 2 * segment + .3 * i
            notes.extend([note(len(notes), time, 0, 0, 2 * (i % 2), i % 2),
                          note(len(notes) + 1, time, 1, 2 + (i % 2) if segment == 0 else 3,
                               1 if segment == 0 else 2 * (i % 2), i % 2)])
    # Eight equally weighted segments: one orthogonal, seven parallel => .125.
    # 80 notes / 15.2 s is 5-7 NPS: the human map Q1-Q3 is .0826-.1374.
    original = deepcopy(notes)
    report = repair_geometry(notes, lambda _: math.inf, coordination=True)
    assert report["geometry_before"]["m3"] == pytest.approx(.125)
    assert report["geometry_coordination_changed"] == 0
    assert notes == original


def test_geometry_v2_runtime_and_explicit_scorer_share_definition():
    from evaluation.playability_metrics import score_map as legacy_score
    from evaluation.playability_metrics_v2 import score_map
    notes = [note(0, 1., 0, 2, 2, 4), note(1, 1.018, 1, 1, 2, 5),
             note(2, 1.13, 0, 2, 2, 7), note(3, 1.4, 0, 2, 0, 1)]
    tuples = [tuple(n[k] for k in ("time", "hand", "x", "y", "d")) for n in notes]
    local, scored = _Geometry(notes, lambda _: math.inf).costs(), score_map(tuples)
    assert local["m2"] == pytest.approx(scored["m2"]["value"])
    assert local["m4_severity"] == pytest.approx(scored["m4"]["value"] * scored["span_seconds"] / 60)
    assert scored["geometry_version"]
    assert legacy_score(tuples)["m2"]["value"] < scored["m2"]["value"]


def test_geometry_v2_rejects_relocation_into_forbidden_cut():
    notes = [note(0, 1., 0, 2, 1, 2)]
    geometry = _Geometry(notes, lambda _: math.inf)
    assert geometry.proposal({0: (3, 1)}) is None
    assert notes[0]["x"] == 2


def test_geometry_v2_rejects_new_calibrated_reach_tail_even_when_cut_is_allowed():
    notes = [note(0, 1., 0, 2, 1, 3)]
    geometry = _Geometry(notes, lambda _: math.inf)
    # Rightward on left x3 is not a hard ban; its center is newly out of
    # the calibrated left-hand range, so this cannot win a geometry tie.
    assert geometry.proposal({0: (3, 1)}) is None
    assert notes[0]["x"] == 2


def test_geometry_v2_preparation_gain_cannot_hide_new_incoming_travel_tail():
    notes = [note(0, 1., 0, 0, 0, 6), note(1, 1.15, 0, 0, 2, 3)]
    geometry = _Geometry(notes, lambda _: math.inf)
    before = deepcopy(notes)
    # Moving the arriving note right makes its approach much easier, but
    # lengthens the incoming saber travel from normal into the excess tail.
    # A summed reach cost rewards that exchange; independent metrics veto it.
    assert geometry.proposal({1: (1, 2)}) is None
    assert notes == before


@pytest.mark.parametrize("combined_mode, expected_change", [(True, True), (False, False)])
def test_geometry_v2_sightline_uses_calibrated_tail_below_legacy_half(monkeypatch, combined_mode, expected_change):
    monkeypatch.setattr(config, "v2_playability_reach", combined_mode)
    notes = [note(0, 1., 0, 0, 0, 1), note(1, 1.3, 1, 0, 1, 0),
             note(2, 5., 0, 0, 1, 0)]
    report = repair_geometry(notes, lambda _: math.inf, sightline=True)
    assert report["geometry_before"]["m2"] == pytest.approx(1 / 12)
    assert bool(report["geometry_sightline_changed"]) is expected_change
    if expected_change:
        assert report["geometry_after"]["m2"] < report["geometry_before"]["m2"]
        assert report["geometry_candidates"] > 0


def test_geometry_v2_final_audit_rolls_back_missed_forbidden_pose(monkeypatch):
    notes = [note(0, 1., 0, 0, 0, 1), note(1, 1.2, 1, 0, 2, 0),
             note(2, 4., 0, 2, 1, 2)]
    before = deepcopy(notes)
    proposal = _Geometry.proposal

    def missed_pose(self, edits, *, commit=False):
        result = proposal(self, edits, commit=commit)
        if commit and result is not None:
            self.notes[-1]["x"] = 3
        return result

    monkeypatch.setattr(_Geometry, "proposal", missed_pose)
    report = repair_geometry(notes, lambda _: math.inf, sightline=True)
    assert report["geometry_rolled_back"]
    assert report["geometry_changed"] == 0
    assert notes == before


def test_geometry_v2_new_resolved_segment_updates_global_coordination_weight():
    notes = []
    for offset, left, right in [(1., [(0, 0), (0, 1), (0, 2)], [(3, 0), (3, 1), (2, 2)]),
                                (3., [(0, 0), (0, 1), (1, 1)], [(2, 0), (2, 1), (3, 2)])]:
        for i in range(3):
            notes.extend([note(len(notes), offset + .4 * i, 0, *left[i], 8),
                          note(len(notes) + 1, offset + .4 * i, 1, *right[i], 8)])
    geometry = _Geometry(notes, lambda _: math.inf)
    # One segment starts unresolved. Moving its final left dot one cell up
    # resolves matching axes without discarding any displacement support.
    deltas = geometry.proposal({10: (1, 2)}, commit=True)
    assert deltas is not None
    assert deltas[2] == pytest.approx(-.2628655560595668)
    assert geometry.costs()["m3"] == pytest.approx(.2628655560595668)
