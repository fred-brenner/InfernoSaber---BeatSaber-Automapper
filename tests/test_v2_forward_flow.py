import pytest

from tools.config import config


def _note(id, time, hand, x, y, d, **extra):
    return {"id": id, "time": time, "hand": hand, "x": x, "y": y, "d": d, **extra}


class _ConstantDraw:
    def __init__(self, value=.9):
        self.value = value
        self.count = 0

    def random(self):
        self.count += 1
        return self.value


@pytest.fixture(autouse=True)
def flow_defaults(monkeypatch):
    for key, value in {"min_time_diff": .02, "v2_cleanup_swing_window": .15,
                       "v2_cleanup_same_dir_gap": .1, "v2_cleanup_same_dir_recut": .5,
                       "v2_fast_gap_seconds": .2,
                       "v2_fast_counter_direction": "recut",
                       "v2_hard_direction_policy": "repair",
                       "v2_flow_min_turn_steps": 3, "v2_diagonal_max_fraction": 1.,
                       "max_speed": 20., "reaction_time": 1.15,
                       "reaction_time_fact": .013, "cdf": 1.2, "cdf_lr": 1.15,
                       "v2_speed_cap_scale": 2., "v2_speed_cap_start_notes": 0,
                       "v2_speed_cap_start_factor": .35,
                       "v2_playability_enabled": True,
                       "v2_playability_priority": True,
                       "v2_playability_reach": True,
                       "v2_playability_slow90_seconds": 0.,
                       "v2_playability_slow45_seconds": 0.,
                       "v2_cleanup_note_heuristics": False,
                       "v2_note_postprocessing_mode": "forward",
                       "v2_pattern_flow": True}.items():
        monkeypatch.setattr(config, key, value)
    yield


def _repair(notes, cap=None):
    from map_creation.v2_forward_flow import repair_patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    return repair_patterns(notes, LocalNoteIndex(notes), cap or (lambda n: 9.))


def test_enabled_gates(monkeypatch):
    from map_creation.v2_forward_flow import enabled
    assert enabled()
    monkeypatch.setattr(config, "v2_pattern_flow", False)
    assert not enabled()
    monkeypatch.setattr(config, "v2_pattern_flow", True)
    monkeypatch.setattr(config, "v2_note_postprocessing_mode", "search")
    assert not enabled()
    monkeypatch.setattr(config, "v2_note_postprocessing_mode", "forward")
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    assert not enabled()
    monkeypatch.setattr(config, "v2_playability_enabled", True)
    monkeypatch.setattr(config, "v2_cleanup_note_heuristics", True)
    assert not enabled()


def test_repair_notes_branches_on_flag(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 1, 1, 0), _note(1, .3, 0, 1, 1, 0)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert "pattern_flow" in out and "direction" not in out
    monkeypatch.setattr(config, "v2_pattern_flow", False)
    notes = [_note(0, 0., 0, 1, 1, 0), _note(1, .3, 0, 1, 1, 0)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert "direction" in out and "pattern_flow" not in out
    assert notes[1]["d"] != 0


def test_prepared_rebound_cases():
    from map_creation.v2_forward_flow import prepared_rebound
    note = _note(1, 1.15, 0, 1, 0, 0)
    assert prepared_rebound(note, _note(0, 1., 0, 1, 0, 1))
    assert prepared_rebound(_note(1, 1.15, 1, 2, 0, 0), _note(0, 1., 1, 2, 0, 1))
    assert not prepared_rebound(note, _note(0, 1., 0, 1, 1, 1))
    assert not prepared_rebound(_note(1, 1.15, 0, 2, 0, 0), _note(0, 1., 0, 0, 0, 1))
    assert not prepared_rebound(_note(1, 1.7, 0, 1, 0, 0), _note(0, 1., 0, 1, 0, 1))
    assert not prepared_rebound(_note(1, 1.15, 0, 1, 0, 8), _note(0, 1., 0, 1, 0, 1))
    assert not prepared_rebound(note, _note(0, 1., 0, 1, 0, 8))
    assert not prepared_rebound(note, None)


def test_prepared_pair_survives_all_layers(monkeypatch):
    from map_creation.v2_forward_directions import (
        DirectionGuard, _detect, audit_directions)
    from map_creation.v2_forward_postprocessing import repair_notes
    pair = lambda: [_note(0, 0., 0, 1, 0, 1), _note(1, .15, 0, 1, 0, 0)]
    notes = pair()
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}
    assert _detect(notes) == {"flow": set(), "counter": set()}
    guard = DirectionGuard([notes[0]])
    assert guard.check([notes[1]]) is None
    notes = pair()
    repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert len(notes) == 2
    assert (notes[0]["x"], notes[0]["y"], notes[0]["d"]) == (1, 0, 1)
    assert (notes[1]["x"], notes[1]["y"], notes[1]["d"]) == (1, 0, 0)
    monkeypatch.setattr(config, "v2_pattern_flow", False)
    notes = pair()
    assert audit_directions(notes)["fast_counter"] == [1]
    guard = DirectionGuard([notes[0]])
    assert guard.check([notes[1]]) == "direction"


def test_center_stream_repaired_both_hands():
    for hand, xs in ((0, 1), (1, 2)):
        notes = [_note(i, .15 * i, hand, xs, 1, i % 2) for i in range(4)]
        out = _repair(notes, cap=lambda n: 99.)
        assert not [n for n in notes if n["x"] in (1, 2) and n["y"] == 1]
        assert [n["id"] for n in notes] == [0, 1, 2, 3]
        assert [n["time"] for n in notes] == pytest.approx([.0, .15, .3, .45])
        assert all(n["hand"] == hand for n in notes)
        assert out["lane_applied"] + out["wave_applied"] + out["changed"] >= 1


def test_terminal_aligned_stack_is_exempt():
    notes = [_note(0, .85, 1, 1, 0, 1),
             _note(1, 1., 1, 2, 1, 0), _note(2, 1., 1, 2, 2, 0),
             _note(3, 1.675, 1, 2, 0, 1)]
    before = [dict(n) for n in notes]
    out = _repair(notes)
    assert notes == before and out["attempted"] == 0
    from map_creation.v2_forward_flow import audit_patterns
    assert audit_patterns(notes)["center_middle"] == 0


def test_non_aligned_stacks_are_not_exempt():
    notes = [_note(0, .85, 1, 1, 0, 1),
             _note(1, 1., 1, 2, 1, 0), _note(2, 1., 1, 3, 2, 0),
             _note(3, 1.675, 1, 2, 0, 1)]
    before = [dict(n) for n in notes]
    out = _repair(notes)
    assert notes == before
    assert out["residual"]["center_middle"] >= 1
    notes = [_note(0, .85, 1, 1, 0, 1),
             _note(1, 1., 1, 2, 1, 0), _note(2, 1., 1, 2, 2, 1),
             _note(3, 1.675, 1, 2, 0, 1)]
    before = [dict(n) for n in notes]
    out = _repair(notes)
    assert notes == before
    assert out["residual"]["center_middle"] >= 1


def test_intervening_note_invalidates_accent_recovery():
    from map_creation.v2_forward_directions import DirectionGuard
    incumbents = [_note(0, .85, 1, 1, 0, 1),
                  _note(1, 1., 1, 2, 1, 0), _note(2, 1., 1, 2, 2, 0),
                  _note(3, 1.675, 1, 2, 0, 1)]
    guard = DirectionGuard(incumbents)
    extra = _note(4, 1., 1, 3, 1, 5, kind="copy", anchor_id=1)
    assert guard.check([extra]) == "pattern_flow"
    intervening = _note(6, 1.3, 1, 3, 1, 8, kind="copy", anchor_id=1)
    assert guard.check([intervening]) is not None


def test_aligned_accent_copy_allowed_and_illegal_center_rejected():
    from map_creation.v2_forward_directions import DirectionGuard
    incumbents = [_note(0, .85, 1, 1, 0, 1),
                  _note(1, 1., 1, 2, 1, 0), _note(2, 1., 1, 2, 2, 0),
                  _note(3, 1.675, 1, 2, 0, 1)]
    guard = DirectionGuard(incumbents)
    aligned = _note(4, 1., 1, 2, 0, 0, kind="copy", anchor_id=1)
    assert guard.check([aligned]) is None
    incumbents = [_note(0, 0., 0, 0, 1, 0), _note(1, .2, 0, 0, 1, 1)]
    guard = DirectionGuard(incumbents)
    center = _note(2, .4, 0, 1, 1, 0, kind="copy", anchor_id=1)
    assert guard.check([center]) == "pattern_flow"


def test_fast_sideways_stream_loses_horizontal():
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .15, 0, 0, 1, 7),
             _note(2, .30, 0, 0, 1, 2), _note(3, .45, 0, 0, 1, 7),
             _note(4, .60, 0, 0, 1, 0)]
    out = _repair(notes)
    assert not [n for n in notes if n["d"] in (2, 3)]
    assert len(notes) == 5
    from map_creation.v2_forward_directions import audit_directions
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}


def test_fully_blocked_index_returns_unchanged_and_bounded():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([])
    calls = {"conflict_ids": 0}

    def spy_ids(self, note, ignore_ids=()):
        calls["conflict_ids"] += 1
        return {1000 + calls["conflict_ids"]}

    index.conflict_ids = spy_ids.__get__(index, LocalNoteIndex)
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .5, 0, 1, 1, 1)]
    from map_creation.v2_forward_flow import repair_patterns
    out = repair_patterns(notes, index, lambda n: 9.)
    assert out["changed"] == 0 and notes[1]["x"] == 1
    assert out["attempted"] == 1 and out["proposals"] <= 12
    assert calls["conflict_ids"] <= out["attempted"] + out["proposals"]


def test_upper_center_up_family_untouched_down_penalized():
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .5, 0, 1, 2, 0)]
    before = [dict(n) for n in notes]
    out = _repair(notes)
    assert notes == before
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .5, 0, 1, 2, 1)]
    out = _repair(notes)
    assert out["changed"] == 1 and (notes[1]["x"], notes[1]["y"]) != (1, 2)


def test_dot_never_recut():
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .15, 0, 1, 1, 8),
             _note(2, .30, 0, 0, 1, 1)]
    from map_creation.v2_forward_directions import audit_directions
    before = audit_directions(notes)
    _repair(notes)
    assert notes[1]["d"] == 8 and len(notes) == 3
    assert audit_directions(notes) == before


def test_wave_applies_canonical_template():
    notes = [_note(i, .15 * i, 0, 1, 1, d)
             for i, d in enumerate((0, 6, 0, 7, 0, 6, 0, 7))]
    out = _repair(notes, cap=lambda n: 99.)
    assert out["wave_applied"] >= 1 and 0 < out["wave_notes"] <= 8
    template = ((1, 2, 0), (0, 0, 6), (0, 2, 0), (1, 0, 7))
    got = [(n["x"], n["y"], n["d"]) for n in notes]
    assert got[:4] == [template[m % 4] for m in range(4)]
    assert not [n for n in notes if n["x"] in (1, 2) and n["y"] == 1]
    from map_creation.v2_forward_directions import audit_directions
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}


def test_wave_skips_vertical_irregular_pause_and_stacks():
    vertical = [_note(i, .15 * i, 0, 0, 1, i % 2) for i in range(8)]
    before = [dict(n) for n in vertical]
    out = _repair(vertical)
    assert out["wave_windows"] == 0 and vertical == before
    times = [0., .15, .3, .45, .75, .9, 1.05, 1.2]
    irregular = [_note(i, times[i], 0, 0, 1, d)
                 for i, d in enumerate((0, 6, 0, 4, 0, 6, 0, 4))]
    before = [dict(n) for n in irregular]
    out = _repair(irregular)
    assert out["wave_windows"] == 0 and irregular == before
    paused = [_note(i, t, 0, 0, 1, d)
              for i, (t, d) in enumerate(zip(
                  (0., .15, .3, 1.2, 1.35, 1.5),
                  (0, 6, 0, 6, 0, 4)))]
    before = [dict(n) for n in paused]
    out = _repair(paused)
    assert out["wave_windows"] == 0 and paused == before
    stacked = [_note(i, .15 * i, 0, 0, 1, d)
               for i, d in enumerate((0, 6, 0))]
    stacked += [_note(10, .45, 0, 0, 1, 0), _note(11, .45, 0, 1, 1, 0)]
    stacked += [_note(20 + i, .6 + .15 * i, 0, 0, 1, d)
                for i, d in enumerate((0, 6, 0))]
    before = [dict(n) for n in stacked]
    out = _repair(stacked)
    assert out["wave_windows"] == 0 and stacked == before


def test_wave_blocker_rejects_template_without_partial_mutation():
    notes = [_note(i, .15 * i, 0, 1, 1, d)
             for i, d in enumerate((0, 6, 0, 7, 0, 6, 0, 7))]
    blockers = [_note(10 + i, .05 + .15 * i, 1, 0, 0, 0) for i in range(4)]
    out = _repair(notes + blockers, cap=lambda n: 99.)
    assert out["wave_applied"] == 0 and out["wave_notes"] == 0


def test_wave_budget_and_non_overlapping_chunks():
    notes = [_note(i, .15 * i, 0, 1, 1, d)
             for i, d in enumerate((0, 6, 0, 7) * 4)]
    out = _repair(notes, cap=lambda n: 99.)
    assert out["wave_windows"] == 2
    assert out["wave_proposals"] <= 12 * out["wave_windows"]
    assert out["wave_notes"] <= 16


def test_reference_wave_fragment_stays_valid():
    cells = ((1, 2, 0), (0, 0, 6), (0, 2, 0), (1, 0, 7), (1, 2, 0))
    notes = [_note(i, .15 * i, 0, x, y, d)
             for i, (x, y, d) in enumerate(cells)]
    before = [dict(n) for n in notes]
    _repair(notes)
    assert notes == before


def test_validate_state_reports_pattern_flow():
    from map_creation.v2_pipeline import _validate_state
    state = {"notes": [_note(0, .5, 0, 0, 1, 2, kind="retained")],
             "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []}
    before = [(n["x"], n["y"], n["d"]) for n in state["notes"]
              if n.get("kind") == "retained"]
    findings = _validate_state(state, 4., playability_mode="forward")
    assert findings["pattern_flow"]["eligible"] == 1
    assert findings["pattern_flow"]["chains_center"] == 0
    assert [(n["x"], n["y"], n["d"]) for n in state["notes"]
            if n.get("kind") == "retained"] == before


def test_expansion_rejects_center_copy_via_shared_reason():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_note(0, 0., 0, 0, 1, 0, kind="retained"),
                       _note(1, .2, 0, 0, 1, 1, kind="retained"),
                       _note(5, .4, 0, 1, 1, 0, kind="copy", anchor_id=1)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 1 and len(state["notes"]) == 2


def test_chain_center_tail_rejected_only_when_enabled(monkeypatch):
    from map_creation.v2_forward_expansion import validate_expansions
    base = lambda: {"notes": [_note(0, 0., 0, 0, 1, 2, kind="retained")],
                    "chains": [{"id": 20, "anchor_id": 0, "time": .02,
                                "end_time": .6, "hand": 0, "x": 1, "y": 1,
                                "tx": 2, "ty": 1, "d": 2, "slices": 8}]}
    state = base()
    assert validate_expansions(state, None)["rejected_chains"] == 1
    monkeypatch.setattr(config, "v2_pattern_flow", False)
    state = base()
    assert validate_expansions(state, None)["rejected_chains"] == 0


def test_empty_and_unsorted_inputs():
    from map_creation.v2_forward_flow import repair_patterns, audit_patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    out = repair_patterns([], LocalNoteIndex([]), lambda n: 9.)
    assert out["attempted"] == 0 and out["wave_windows"] == 0
    assert audit_patterns([])["eligible"] == 0
    notes = [_note(7, .5, 0, 1, 1, 1, kind="retained", source="x", anchor_id=3),
             _note(3, .2, 0, 0, 1, 0)]
    _repair(notes)
    assert [n["id"] for n in notes] == [7, 3]
    assert notes[0]["kind"] == "retained" and notes[0]["anchor_id"] == 3


def test_bounded_blocked_candidate_counters():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([])
    calls = {"conflict_ids": 0}
    real = LocalNoteIndex.conflict_ids

    def spy_ids(self, note, ignore_ids=()):
        calls["conflict_ids"] += 1
        return real(self, note, ignore_ids)

    index.conflict_ids = spy_ids.__get__(index, LocalNoteIndex)
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .5, 0, 1, 1, 1)]
    from map_creation.v2_forward_flow import repair_patterns
    out = repair_patterns(notes, index, lambda n: 9.)
    assert out["attempted"] == 1 and out["proposals"] <= 12
    assert calls["conflict_ids"] <= out["attempted"] + out["proposals"]


def _lane(notes, cap=None):
    from map_creation import v2_forward_flow as flow
    from map_creation import v2_forward_patterns as patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    report = {"lane_runs": 0, "lane_proposals": 0, "lane_applied": 0,
              "lane_notes": 0}
    patterns._lane_runs(flow.FlowContext(notes), LocalNoteIndex(notes),
                        cap or (lambda n: 9.), report)
    return report


def test_lane_translation_blue():
    from map_creation.v2_forward_directions import audit_directions
    notes = [_note(0, .0, 1, 2, 2, 0), _note(1, .15, 1, 2, 1, 1),
             _note(2, .30, 1, 2, 1, 0), _note(3, .45, 1, 2, 1, 1),
             _note(4, 1.2, 1, 3, 0, 0)]
    report = _lane(notes, cap=lambda n: 9.)
    assert report["lane_proposals"] == 1 and report["lane_applied"] == 1
    assert report["lane_notes"] == 4
    assert [n["x"] for n in notes[:4]] == [3] * 4
    assert [(n["y"], n["d"], n["id"]) for n in notes[:4]] == [
        (2, 0, 0), (1, 1, 1), (1, 0, 2), (1, 1, 3)]
    assert [n["time"] for n in notes[:4]] == pytest.approx([0., .15, .3, .45])
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}


def test_lane_translation_red():
    notes = [_note(0, .0, 0, 1, 2, 0), _note(1, .15, 0, 1, 1, 1),
             _note(2, .30, 0, 1, 1, 0), _note(3, .45, 0, 1, 1, 1),
             _note(4, 1.2, 0, 0, 0, 0)]
    report = _lane(notes, cap=lambda n: 9.)
    assert report["lane_applied"] == 1 and report["lane_notes"] == 4
    assert [n["x"] for n in notes[:4]] == [0] * 4


def test_lane_blocked_member_rolls_back_whole_run():
    notes = [_note(0, .0, 1, 2, 2, 0), _note(1, .15, 1, 2, 1, 1),
             _note(2, .30, 1, 2, 1, 0), _note(3, .45, 1, 2, 1, 1),
             _note(4, 1.2, 1, 3, 0, 0), _note(9, .32, 0, 3, 1, 0)]
    report = _lane(notes, cap=lambda n: 9.)
    assert report["lane_applied"] == 0
    assert [(n["x"], n["y"], n["d"]) for n in notes[:4]] == [
        (2, 2, 0), (2, 1, 1), (2, 1, 0), (2, 1, 1)]


def test_lane_long_run_single_proposal():
    notes = [_note(i, .15 * i, 0, 1, 1, i % 2) for i in range(1000)]
    report = _lane(notes, cap=lambda n: 99.)
    assert report["lane_runs"] == 1 and report["lane_proposals"] <= 1
    assert report["lane_applied"] == 1 and report["lane_notes"] == 1000
    assert all(n["x"] == 0 for n in notes)


def test_lane_outer_column_run_unchanged():
    notes = [_note(i, .15 * i, 0, 0, 1, i % 2) for i in range(6)]
    report = _lane(notes)
    assert report["lane_runs"] == 0 and report["lane_proposals"] == 0
    assert all(n["x"] == 0 for n in notes)


def test_lane_rollback_on_cap_exception():
    from map_creation import v2_forward_flow as flow
    from map_creation import v2_forward_patterns as patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(i, .15 * i, 0, 1, 1, i % 2) for i in range(4)]
    report = {"lane_runs": 0, "lane_proposals": 0, "lane_applied": 0,
              "lane_notes": 0}

    def boom(note):
        raise RuntimeError("cap")
    with pytest.raises(RuntimeError):
        patterns._lane_runs(flow.FlowContext(notes), LocalNoteIndex(notes),
                            boom, report)
    assert [(n["x"], n["y"], n["d"]) for n in notes] == [
        (1, 1, i % 2) for i in range(4)]


def test_wave_rollback_on_cap_exception():
    notes = [_note(i, .15 * i, 0, 0, 1, d)
             for i, d in enumerate((0, 6, 0, 7, 0, 6, 0, 7))]

    def boom(note):
        raise RuntimeError("cap")
    with pytest.raises(RuntimeError):
        _repair(notes, cap=boom)
    assert [(n["x"], n["y"], n["d"]) for n in notes] == [
        (0, 1, d) for d in (0, 6, 0, 7, 0, 6, 0, 7)]


def test_local_rollback_on_cap_exception():
    notes = [_note(0, .0, 0, 0, 1, 0), _note(1, .5, 0, 1, 2, 1)]

    def boom(note):
        raise RuntimeError("cap")
    with pytest.raises(RuntimeError):
        _repair(notes, cap=boom)
    assert (notes[1]["x"], notes[1]["y"], notes[1]["d"]) == (1, 2, 1)


def test_wave_equal_cost_template():
    cycle = ((1, 2, 0), (0, 0, 6), (0, 2, 0), (1, 0, 7))
    notes = [_note(i, .15 * i, 0, 0, 2 - 2 * (i % 2), d)
             for i, d in enumerate((4, 6, 4, 6, 4, 6, 4, 6))]
    out = _repair(notes, cap=lambda n: 99.)
    got = [(n["x"], n["y"], n["d"]) for n in notes]
    assert out["wave_applied"] == 1 and out["wave_notes"] > 0
    assert out["wave_notes"] == sum(
        cell != (0, 2 - 2 * (i % 2), (4, 6)[i % 2])
        for i, cell in enumerate(got))
    assert any(got[:4] == [cycle[(p + m) % 4] for m in range(4)]
               for p in range(4))


def test_chain_diagonal_crossing_between_samples():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_note(0, 0., 0, 0, 0, 1, kind="retained")],
             "chains": [{"id": 20, "anchor_id": 0, "time": .02,
                         "end_time": .05, "hand": 0, "x": 1, "y": 0,
                         "tx": 2, "ty": 2, "d": 4, "slices": 2}]}
    assert validate_expansions(state, None)["rejected_chains"] == 1


def test_chain_outer_segment_accepted():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_note(0, 0., 0, 0, 0, 0, kind="retained")],
             "chains": [{"id": 20, "anchor_id": 0, "time": .02,
                         "end_time": .2, "hand": 0, "x": 0, "y": 0,
                         "tx": 0, "ty": 2, "d": 0, "slices": 4}]}
    assert validate_expansions(state, None)["rejected_chains"] == 0


def test_flow_context_shared_group_view():
    from map_creation.v2_forward_flow import FlowContext, audit_patterns
    notes = [_note(i, 1., 0, 0, 1, 0) for i in range(1000)]
    context = FlowContext(notes)
    assert context.group(notes[0]) is context.group(notes[-1])
    assert len(context.group(notes[0])) == 1000
    assert audit_patterns(notes)["eligible"] == 1000


def test_direction_guard_far_incumbent_cost_does_not_block():
    from map_creation.v2_forward_directions import DirectionGuard
    incumbents = [_note(0, 0., 0, 0, 1, 0), _note(1, .2, 0, 0, 1, 1),
                  _note(2, .55, 0, 0, 1, 1)]
    guard = DirectionGuard(incumbents)
    copy = _note(9, .4, 0, 0, 1, 0, kind="copy", anchor_id=1)
    assert guard.check([copy]) is None


def test_end_to_end_forward_pipeline():
    from map_creation.v2_forward_postprocessing import repair_notes
    from map_creation.v2_forward_expansion import (
        apply_note_emphasis, validate_expansions)
    from map_creation.v2_pipeline import _validate_state
    notes = [_note(0, .0, 0, 0, 1, 1, kind="retained"),
             _note(1, .3, 0, 0, 2, 4, kind="retained"),
             _note(2, .5, 0, 1, 0, 1, kind="retained"),
             _note(3, .65, 0, 1, 0, 0, kind="retained"),
             _note(4, 2., 0, 0, 1, 1, kind="retained")]
    state = {"notes": notes, "chains": [], "walls": [], "bombs": [],
             "arcs": [], "lighting": []}
    repair_notes(state["notes"], cap=lambda n: 99., rng=_ConstantDraw())
    retained = {n["id"]: (n["x"], n["y"], n["d"]) for n in state["notes"]
                if n.get("kind") == "retained"}
    assert retained[2] == (1, 0, 1) and retained[3] == (1, 0, 0)
    state["notes"].append(_note(9, 4., 0, 2, 0, 6,
                                kind="copy", anchor_id=4))
    validate_expansions(state, None)
    apply_note_emphasis(state, None, rng=_ConstantDraw())
    findings = _validate_state(state, 4., playability_mode="forward")
    assert "pattern_flow" in findings
    assert "chains_center" in findings["pattern_flow"]
    assert all((n["x"], n["y"], n["d"]) == retained[n["id"]]
               for n in state["notes"] if n["id"] in retained)


def test_phrase_candidate_bank_is_bounded_and_mirrored():
    from map_creation import v2_forward_patterns as patterns
    red = [_note(i, .15 * i, 0, x, 1, i % 2)
           for i, x in enumerate((1, 0, 1, 0))]
    blue = [dict(n, hand=1, x=3 - n["x"]) for n in red]
    red_candidates = list(patterns._phrase_candidates(red, 0, True))
    blue_candidates = list(patterns._phrase_candidates(blue, 1, True))
    assert 0 < len(red_candidates) <= 12
    assert [name for name, _ in red_candidates] == [
        name for name, _ in blue_candidates]
    for (_, red_geometry), (_, blue_geometry) in zip(red_candidates, blue_candidates):
        assert blue_geometry == tuple(
            (3 - x, y, patterns._MIRROR_D.get(d, d)) for x, y, d in red_geometry)
    assert len({geometry for _, geometry in red_candidates}) == len(red_candidates)
    assert all(geometry != tuple((n["x"], n["y"], n["d"]) for n in red)
               for _, geometry in red_candidates)


def test_phrase_report_has_bounded_rejections_and_trials():
    notes = [_note(i, .15 * i, 0, x, 1, i % 2)
             for i, x in enumerate((1, 0, 1, 0))]
    report = _repair(notes, cap=lambda n: 99.)
    assert report["phrase_windows"] >= 1
    assert report["phrase_proposals"] <= 12 * report["phrase_windows"]
    assert report["wave_proposals"] <= 12 * report["wave_windows"]
    assert report["proposals"] <= 12 * report["attempted"]
    assert all(len(examples) <= 3 for examples in report["rejection_examples"].values())


def _phrase_only(notes, cap=lambda n: 99.):
    from map_creation import v2_forward_flow as flow
    from map_creation import v2_forward_patterns as patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    report = {"phrase_windows": 0, "phrase_proposals": 0, "phrase_applied": 0,
              "phrase_notes": 0, "phrase_motifs": {}, "rejections": {},
              "rejection_examples": {}}
    context = flow.FlowContext(notes)
    index = LocalNoteIndex(notes)
    for hand in (0, 1):
        patterns._phrase_hand(context, index, cap, hand, report)
    return report


def test_mixed_phrase_repairs_center_and_mirrors():
    from copy import deepcopy
    from map_creation import v2_forward_flow as flow
    from map_creation import v2_forward_patterns as patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    from map_creation.v2_forward_directions import audit_directions
    from map_creation.v2_forward_flow import audit_patterns
    from map_creation.v2_forward_patterns import _MIRROR_D
    cells = ((0, 0, 1), (0, 0, 0), (1, 1, 1), (1, 1, 0))
    red = [_note(i, .15 * i, 0, x, y, d, kind="retained", source="test")
           for i, (x, y, d) in enumerate(cells)]
    blue = [dict(n, hand=1, x=3 - n["x"], d=_MIRROR_D.get(n["d"], n["d"]))
            for n in red]
    before = [dict(n) for n in red]
    baseline = audit_patterns(red)["center_middle"]
    local = deepcopy(red)
    local_report = {"attempted": 0, "changed": 0, "blocked": 0,
                    "skipped": 0, "proposals": 0}
    patterns._local_repair(flow.FlowContext(local), LocalNoteIndex(local),
                           lambda n: 1., local_report)
    assert local == before and local_report["rejections"]["local:speed"] > 0
    red_report = _phrase_only(red, lambda n: 1.)
    blue_report = _phrase_only(blue, lambda n: 1.)
    assert red_report["phrase_applied"] == 1
    assert audit_patterns(red)["center_middle"] < baseline
    assert len(red) == len(before)
    assert [{k: v for k, v in n.items() if k not in ("x", "y", "d")}
            for n in red] == [
                {k: v for k, v in n.items() if k not in ("x", "y", "d")}
                for n in before]
    assert [(n["x"], n["y"], n["d"]) for n in blue] == [
        (3 - n["x"], n["y"], _MIRROR_D.get(n["d"], n["d"])) for n in red]
    assert red_report == blue_report
    assert audit_directions(red) == {"flow": [], "fast_counter": []}


def test_phrase_opposite_hand_blocker_rejects_atomically():
    notes = [_note(i, .15 * i, 0, x, 1, i % 2)
             for i, x in enumerate((1, 0))]
    blockers = [_note(10 + x * 3 + y, 0., 1, x, y, 0)
                for x in range(3) for y in range(3) if (x, y) != (1, 1)]
    before = [dict(n) for n in notes + blockers]
    report = _phrase_only(notes + blockers)
    assert report["phrase_proposals"] > 0
    assert report["phrase_applied"] == 0
    assert notes + blockers == before


def test_phrase_blocked_successor_preserves_incumbent():
    from map_creation.v2_forward_directions import audit_directions
    cells = ((0, 0, 1), (0, 0, 0), (1, 1, 1), (1, 1, 0))
    notes = [_note(i, .15 * i, 0, *cell) for i, cell in enumerate(cells)]
    notes.append(_note(4, .86, 0, 0, 1, 6))
    before = [dict(n) for n in notes]
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}
    report = _phrase_only(notes, lambda n: 1.)
    assert report["rejections"]["phrase:exit_flow"] > 0
    assert report["phrase_applied"] == 0
    assert notes == before


@pytest.mark.parametrize("extra", [
    [_note(10, .3, 0, 0, 1, 8)],
    [_note(10, .3, 0, 0, 1, 0), _note(11, .3, 0, 1, 1, 0)],
    [_note(10, .015, 0, 0, 1, 0), _note(11, .03, 0, 1, 1, 1)],
])
def test_phrase_skips_stacks_dots_and_ambiguous_onsets(extra):
    notes = [_note(0, 0., 0, 1, 1, 0), _note(1, .15, 0, 0, 1, 1)] + extra
    before = [dict(n) for n in notes]
    report = _phrase_only(notes)
    assert report["phrase_applied"] == 0
    assert notes == before


def test_phrase_zero_cost_vertical_stream_unchanged():
    notes = [_note(i, .15 * i, 0, 0, 1, i % 2) for i in range(8)]
    before = [dict(n) for n in notes]
    report = _phrase_only(notes)
    assert report["phrase_windows"] == 0
    assert notes == before


def test_chunk_upper_down_rejects_center_improvement():
    from map_creation import v2_forward_flow as flow
    from map_creation import v2_forward_patterns as patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(0, 0., 0, 1, 1, 1)]
    index = LocalNoteIndex(notes)
    context = flow.FlowContext(notes)
    pre = patterns._prepare(context, index, notes)
    saved = [(1, 1, 1)]
    notes[0]["y"] = 2
    report = {}
    assert patterns._check_chunk(context, index, lambda n: 99., notes, pre,
                                 saved, True, report, "phrase") is None
    assert report["rejections"] == {"phrase:upper_down": 1}


@pytest.mark.parametrize("length", [2, 3])
def test_short_phrase_chunks_and_upper_down_guard(length):
    from map_creation.v2_forward_patterns import _upper_down
    notes = [_note(i, .15 * i, 0, x, y, d)
             for i, (x, y, d) in enumerate(
                 ((0, 0, 1), (1, 1, 0), (1, 1, 1))[:length])]
    before = sum(_upper_down(n) for n in notes)
    report = _phrase_only(notes)
    assert report["phrase_windows"] == 1
    assert report["phrase_proposals"] <= 12
    assert sum(_upper_down(n) for n in notes) <= before


def test_phrase_trials_restore_after_rejection_and_exception(monkeypatch):
    from map_creation import v2_forward_patterns as patterns
    notes = [_note(i, .15 * i, 0, x, 1, i % 2)
             for i, x in enumerate((1, 0, 1, 0))]
    before = [dict(n) for n in notes]
    real = patterns._check_chunk

    def reject(context, index, cap, changed, pre, saved, strict, report=None,
               family="wave"):
        assert changed != before
        return patterns._reject(report, family, "collision", changed)

    monkeypatch.setattr(patterns, "_check_chunk", reject)
    report = _phrase_only(notes)
    assert notes == before
    assert report["rejections"]["phrase:collision"] == report["phrase_proposals"]
    assert len(report["rejection_examples"]["phrase:collision"]) == 3
    assert set(report["rejection_examples"]["phrase:collision"][0]) == {
        "id", "time", "length"}

    def fail(*args, **kwargs):
        raise RuntimeError("guard failure")

    monkeypatch.setattr(patterns, "_check_chunk", fail)
    with pytest.raises(RuntimeError, match="guard failure"):
        _phrase_only(notes)
    assert notes == before
    monkeypatch.setattr(patterns, "_check_chunk", real)


def test_wave_blocked_full_length_accepts_shorter_prefix():
    from map_creation import v2_forward_flow as flow
    from map_creation import v2_forward_patterns as patterns
    from map_creation.v2_forward_clearance import LocalNoteIndex
    from map_creation.v2_forward_directions import audit_directions
    notes = [_note(i, .15 * i, 0, 1, 1, d)
             for i, d in enumerate((0, 6, 0, 7, 0, 6, 0, 7))]
    successor = _note(20, 1.2, 0, 0, 1, 4)
    blocker = _note(21, 1.05, 1, 1, 0, 1)
    index = LocalNoteIndex(notes + [successor, blocker])
    context = flow.FlowContext(notes + [successor, blocker])
    before = [dict(n) for n in notes]
    assert audit_directions(notes + [successor]) == {"flow": [], "fast_counter": []}
    report = {"wave_proposals": 0, "wave_applied": 0, "wave_notes": 0}
    patterns._wave_attempt(context, index, lambda n: 99., 0,
                           [[note] for note in notes], report)
    assert report["wave_applied"] == 1
    assert notes[6:] == before[6:]
    assert 0 < report["wave_notes"] <= 6
    assert report["wave_proposals"] == 12
    assert sum(example["length"] == 8
               for example in report["rejection_examples"]["wave:exit_flow"]) > 0
    assert audit_directions(notes + [successor]) == {"flow": [], "fast_counter": []}
