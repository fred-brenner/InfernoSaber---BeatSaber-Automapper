"""Forward postprocessing: bounded local passes and the center-view rule."""

import pytest

from tools.config import config


def _note(id, time, hand, x, y, d, **extra):
    return {"id": id, "time": time, "hand": hand, "x": x, "y": y, "d": d, **extra}


class _ScriptedDraws:
    def __init__(self, *values):
        self.values = list(values)
        self.count = 0

    def random(self):
        self.count += 1
        if not self.values:
            raise AssertionError("unscripted RNG draw")
        return self.values.pop(0)


class _ConstantDraw:
    def __init__(self, value=.9):
        self.value = value
        self.count = 0

    def random(self):
        self.count += 1
        return self.value


@pytest.fixture(autouse=True)
def forward_defaults(monkeypatch):
    for key, value in {"min_time_diff": .02, "v2_cleanup_swing_window": .15,
                       "v2_cleanup_same_dir_gap": .1, "v2_fast_gap_seconds": .2,
                       "v2_fast_counter_direction": "recut",
                       "v2_hard_direction_policy": "repair",
                       "v2_flow_min_turn_steps": 3, "v2_diagonal_max_fraction": 1.,
                       "max_speed": 20., "reaction_time": 1.15,
                       "reaction_time_fact": .013, "cdf": 1.2, "cdf_lr": 1.15,
                       "v2_speed_cap_scale": 2., "v2_speed_cap_start_notes": 0,
                       "v2_speed_cap_start_factor": .35,
                       "v2_playability_priority": True}.items():
        monkeypatch.setattr(config, key, value)
    yield


def test_center_view_middle_boundaries():
    from map_creation.v2_forward_postprocessing import clear_center_view
    notes = [_note(0, 1., 0, 1, 1, 2)]
    draws = _ScriptedDraws(.799)
    out = clear_center_view(notes, rng=draws)
    assert notes[0]["x"] == 0 and draws.count == 1 and out["moved"] == 1
    notes = [_note(0, 1., 0, 1, 1, 2)]
    draws = _ScriptedDraws(.8)
    out = clear_center_view(notes, rng=draws)
    assert notes[0]["x"] == 1 and draws.count == 1 and out["moved"] == 0


def test_center_view_right_hand_mirror_and_top():
    from map_creation.v2_forward_postprocessing import clear_center_view
    notes = [_note(0, 1., 1, 2, 1, 3)]
    assert clear_center_view(notes, rng=_ScriptedDraws(.799))["moved"] == 1
    assert notes[0]["x"] == 3
    notes = [_note(0, 1., 0, 1, 2, 0)]
    assert clear_center_view(notes, rng=_ScriptedDraws(.199))["moved"] == 1
    notes = [_note(0, 1., 0, 1, 2, 0)]
    assert clear_center_view(notes, rng=_ScriptedDraws(.2))["moved"] == 0


def test_center_view_bottom_and_wrong_column_draw_nothing():
    from map_creation.v2_forward_postprocessing import clear_center_view
    draws = _ConstantDraw()
    notes = [_note(0, 1., 0, 1, 0, 0), _note(1, 1.5, 0, 2, 1, 3),
             _note(2, 2., 1, 1, 1, 2)]
    out = clear_center_view(notes, rng=draws)
    assert draws.count == 0 and out["moved"] == 0
    assert [(n["x"], n["y"]) for n in notes] == [(1, 0), (2, 1), (1, 1)]


def test_center_view_preserves_dots_metadata_and_blocks_occupied():
    from map_creation.v2_forward_postprocessing import clear_center_view
    notes = [_note(0, 1., 0, 1, 1, 8, kind="retained", source="emphasis",
                  anchor_id=7)]
    out = clear_center_view(notes, rng=_ScriptedDraws(.1))
    assert notes[0]["x"] == 0 and notes[0]["d"] == 8
    assert notes[0]["anchor_id"] == 7 and notes[0]["source"] == "emphasis"
    assert out["moved"] == 1
    notes = [_note(0, 1., 0, 1, 1, 2), _note(1, 1.01, 0, 0, 1, 2)]
    draws = _ScriptedDraws(.1)
    out = clear_center_view(notes, rng=draws)
    assert draws.count == 1 and notes[0]["x"] == 1 and out["moved"] == 0


def test_index_same_cell_and_live_updates():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    existing = _note(0, 1., 0, 1, 1, 2)
    index = LocalNoteIndex([existing])
    assert index.conflicts(_note(1, 1.015, 1, 1, 1, 2))
    assert not index.conflicts(_note(1, 1.05, 0, 0, 1, 3))
    existing["x"] = 0
    assert index.conflicts(_note(1, 1.015, 0, 0, 1, 3))


def test_index_swept_diagonal_and_same_hand_stack():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([_note(0, 1., 0, 0, 1, 2)])
    assert index.conflicts(_note(1, 1.05, 1, 1, 1, 3))
    index = LocalNoteIndex([_note(0, 1., 0, 0, 0, 7)])
    assert index.conflicts(_note(1, 1.1, 1, 0, 1, 5))
    index = LocalNoteIndex([_note(0, 1., 0, 0, 0, 0)])
    assert not index.conflicts(_note(1, 1., 0, 2, 0, 0))


def test_index_accepts_many_and_rejection_leaves_state():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([_note(0, 1., 0, 1, 1, 2)])
    batch = [_note(1, 1., 0, 2, 1, 2), _note(2, 1., 0, 2, 1, 3)]
    assert not index.accepts_many(batch)
    assert index.accepts(_note(2, 1., 0, 2, 1, 3))
    assert index.accepts_many([_note(1, 1., 0, 2, 1, 2)])


def test_index_chain_sample_exempts_anchor_only():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([_note(0, 1., 0, 1, 1, 2), _note(9, 1., 0, 1, 1, 3)])
    sample = _note(5, 1.005, 0, 1, 1, 2, kind="chain_sample", anchor_id=0,
                   chain_id=4)
    assert index.conflicts(sample)
    index2 = LocalNoteIndex([_note(0, 1., 0, 1, 1, 2)])
    assert not index2.conflicts(sample)


def test_repair_direction_fixes_flow_repeat_and_keeps_dots():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 1, 1, 0), _note(1, .3, 0, 1, 1, 0),
             _note(2, .6, 0, 1, 1, 8)]
    out = repair_notes(notes, rng=_ConstantDraw())
    assert notes[0]["d"] == 0
    assert notes[1]["d"] != 0 and notes[2]["d"] == 8
    assert out["direction"]["changed"] == 1


def test_repair_reach_moves_over_cap_note_bounded():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 0, 1, 3), _note(1, 1., 0, 3, 1, 3)]
    out = repair_notes(notes, cap=lambda n: 1., rng=_ScriptedDraws(.9, .9))
    assert notes[1]["x"] in (1, 2) and out["reach"]["changed"] == 2


def test_repair_collision_relocates_blocked_note():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 1., 0, 1, 1, 2), _note(1, 1.05, 1, 1, 1, 3)]
    out = repair_notes(notes, rng=_ConstantDraw())
    assert (notes[0]["x"], notes[0]["y"]) != (notes[1]["x"], notes[1]["y"])
    assert out["collision"]["changed"] >= 1


def test_repair_repeated_position_geometry_and_top_gate():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 1, 1, 1), _note(1, .3, 0, 1, 1, 0)]
    out = repair_notes(notes, rng=_ScriptedDraws(.4, .9, .9))
    assert (notes[1]["x"], notes[1]["y"]) == (1, 2)
    assert out["repeated"]["changed"] == 1
    notes = [_note(0, 0., 0, 1, 1, 1), _note(1, .3, 0, 1, 1, 0)]
    assert repair_notes(notes, rng=_ScriptedDraws(.6, .9, .9))["repeated"]["changed"] == 0
    assert (notes[1]["x"], notes[1]["y"]) == (1, 1)


def test_repair_visibility_moves_endpoint_center_only_when_priority():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(i, .3 * i, 0, 1, 2, 0) for i in range(5)]
    out = repair_notes(notes, cap=lambda n: 99., rng=_ConstantDraw())
    assert out["visibility"]["changed"] >= 1
    notes = [_note(i, .3 * i, 0, 1, 2, 0) for i in range(5)]
    import map_creation.v2_forward_postprocessing as forward
    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "v2_playability_priority", False)
        out = repair_notes(notes, cap=lambda n: 99., rng=_ConstantDraw())
    assert out["visibility"]["changed"] == 0


def test_repair_unsorted_empty_and_dense_no_raise():
    from map_creation.v2_forward_postprocessing import repair_notes
    assert repair_notes([])["seconds"] >= 0.
    notes = [_note(0, 1., 0, 1, 1, 0), _note(1, .5, 0, 1, 1, 0)]
    out = repair_notes(notes, cap=lambda n: .5, rng=_ConstantDraw())
    ids = [n["id"] for n in notes]
    assert sorted(ids) == [0, 1] and len(notes) == 2
    dense = [_note(i, 1., 0 if i % 2 else 1, 1, 1, 2) for i in range(8)]
    out = repair_notes(dense, cap=lambda n: 0., rng=_ConstantDraw())
    assert out["collision"]["unresolved"] > 0
    # The residual same-time pile collapses to its earliest survivors.
    assert out["hard_overlap"]["dropped"] == 5
    assert [n["id"] for n in dense] == [0, 1, 2]
    assert all(n["time"] == 1. and n["d"] == 2 for n in dense)


def test_normalize_notes_skips_legacy_mutators_and_labels():
    from map_creation.v2_forward_postprocessing import normalize_notes
    notes = [_note(7, 1., 0, 1, 1, 0), _note(3, .5, 0, 1, 1, 0)]
    out, report = normalize_notes(notes, [0., .5, 1.])
    assert isinstance(report, dict) and len(out) == 2
    assert [n["d"] for n in out] == [0, 0]


def test_clear_center_view_empty_and_unsorted():
    from map_creation.v2_forward_postprocessing import clear_center_view
    assert clear_center_view([])["moved"] == 0
    notes = [_note(0, 2., 0, 1, 1, 2), _note(1, 1., 0, 1, 1, 2)]
    clear_center_view(notes, rng=_ScriptedDraws(.1, .1))
    assert notes[0]["x"] == 0 and notes[1]["x"] == 0

def test_index_copy_overlapping_own_anchor_conflicts():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([_note(0, 1., 0, 1, 1, 2)])
    copy = _note(3, 1.005, 0, 1, 1, 2, kind="copy", anchor_id=0)
    assert index.conflicts(copy)


def test_index_chain_sample_contract():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([_note(0, 1., 0, 1, 1, 2)])
    s1 = _note(5, 1.005, 0, 1., 1., 2, kind="chain_sample", anchor_id=0,
               chain_id=4)
    assert not index.conflicts(s1)
    index = LocalNoteIndex([s1])
    assert not index.conflicts(_note(0, 1., 0, 1, 1, 2))
    s2 = _note(6, 1.006, 0, 1., 1.4, 2, kind="chain_sample", anchor_id=0,
               chain_id=4)
    assert not index.conflicts(s2)
    other_chain = _note(7, 1.007, 0, 1., 1., 2, kind="chain_sample",
                        anchor_id=9, chain_id=8)
    assert index.conflicts(other_chain)


def test_index_geometry_integer_and_batch_duplicate_ids():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([])
    assert not index.accepts(_note(0, 1., 0, 1.5, 1., 2))
    assert index.accepts(_note(0, 1., 0, 1.5, 1., 2, kind="chain_sample",
                             anchor_id=1, chain_id=2))
    batch = [_note(5, 1., 0, 0, 1, 2), _note(5, 2., 0, 1, 1, 2)]
    assert not index.accepts_many(batch)


def test_repair_preserves_caller_order_and_ids():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(7, 1., 0, 1, 1, 1), _note(3, .5, 0, 1, 1, 0),
             _note(9, .2, 0, 1, 1, 0)]
    repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert [n["id"] for n in notes] == [7, 3, 9]
    assert notes[1]["d"] != 0


def test_repair_reach_fixes_first_note_and_dot_positions():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 3, 1, 2)]
    out = repair_notes(notes, cap=lambda n: 1., rng=_ConstantDraw())
    assert notes[0]["x"] == 2 and out["reach"]["changed"] == 1
    notes = [_note(0, 0., 0, 3, 1, 8)]
    out = repair_notes(notes, cap=lambda n: 1., rng=_ConstantDraw())
    assert notes[0]["x"] == 2 and notes[0]["d"] == 8


def test_repair_reach_single_excursion_not_skipped():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 2, 1, 2)]
    out = repair_notes(notes, cap=lambda n: 1., rng=_ScriptedDraws(.9))
    assert notes[0]["x"] == 1 and out["reach"]["changed"] == 1


def test_repair_reach_disabled_when_flags_off(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_playability_priority", False)
    monkeypatch.setattr(config, "v2_playability_reach", False)
    notes = [_note(0, 0., 0, 3, 1, 2)]
    out = repair_notes(notes, cap=lambda n: 1., rng=_ConstantDraw())
    assert out["reach"]["attempted"] == 0 and notes[0]["x"] == 3


def test_direction_disabled_when_recut_flag_zero(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_cleanup_same_dir_recut", 0.)
    notes = [_note(0, 0., 0, 1, 1, 0), _note(1, .3, 0, 1, 1, 0)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ScriptedDraws(.9, .9, .9))
    assert notes[1]["d"] == 0 and out["direction"]["attempted"] == 0


def test_direction_counter_needs_positive_fast_gap(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_fast_gap_seconds", 0.)
    notes = [_note(0, 0., 0, 0, 1, 2), _note(1, 0., 0, 1, 2, 1)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ScriptedDraws(.9, .9))
    assert notes[1]["d"] == 1


def test_direction_excess_diagonal_triggers_independently(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", .1)
    notes = [_note(0, 0., 0, 1, 1, 4), _note(1, .3, 0, 1, 1, 7),
             _note(2, .6, 0, 1, 1, 4)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ScriptedDraws(.9, .9))
    assert notes[1]["d"] not in (4, 5, 6, 7)


def test_direction_axis_churn_prefers_previous_axis(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", 1.)
    notes = [_note(0, 0., 0, 1, 1, 4), _note(1, .3, 0, 1, 1, 1),
             _note(2, .6, 0, 1, 1, 5)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ScriptedDraws(.9, .9))
    assert notes[2]["d"] not in (4, 5, 6, 7)


def test_repair_dense_saturation_reports_unresolved():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(h * 12 + y * 4 + x, 1., h, x, y, 2)
             for h in (0, 1) for y in range(3) for x in range(4)]
    original = [dict(n) for n in notes]
    calls = {"accepts": 0}
    real = LocalNoteIndex.accepts

    def spy(self, note, ignore_ids=()):
        calls["accepts"] += 1
        return real(self, note, ignore_ids)

    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LocalNoteIndex, "accepts", spy)
        out = repair_notes(notes, cap=lambda n: 0., rng=_ConstantDraw())
    # Nothing relocates inside the saturated grid; the bounded overlap drop
    # removes each cell's second hand, keeping the 12 earlier h0 notes.
    assert notes == original[:12]
    assert out["collision"]["unresolved"] > 0
    assert out["hard_overlap"]["dropped"] == 12
    assert calls["accepts"] <= 8 * 24 * 6

def test_index_fractional_sample_cell_occupancy_and_gap():
    from map_creation.v2_forward_clearance import LocalNoteIndex
    index = LocalNoteIndex([_note(0, 0., 0, 0, 1, 2)])
    near = _note(5, .1, 1, 1.4, 1., 2, kind="chain_sample", anchor_id=9,
                 chain_id=4)
    assert index.conflicts(near)
    far = _note(5, .2, 1, 1.4, 1., 2, kind="chain_sample", anchor_id=9,
                chain_id=4)
    assert not index.conflicts(far)


def test_direction_first_of_hand_honors_diagonal_cap(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", .1)
    notes = [_note(0, 0., 0, 1, 1, 4), _note(1, .3, 0, 1, 1, 7)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert notes[0]["d"] not in (4, 5, 6, 7)


def test_direction_same_axis_only_when_churn_triggered(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", .1)
    monkeypatch.setattr(config, "v2_playability_priority", True)
    notes = [_note(0, 0., 0, 1, 1, 4), _note(1, .3, 0, 1, 1, 4),
             _note(2, .6, 0, 1, 1, 4)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert notes[1]["d"] not in (4, 5, 6, 7)


def test_center_view_seed_reproducible_and_no_flow_lookup():
    import random
    from map_creation.v2_forward_postprocessing import clear_center_view
    from map_creation import v2_postprocessing as post
    notes = [_note(0, 1., 0, 1, 1, 2), _note(1, 2., 0, 1, 2, 0)]
    first = [dict(n) for n in notes]
    clear_center_view(notes, rng=random.Random(7))
    second = [dict(n) for n in notes]
    clear_center_view(first, rng=random.Random(7))
    assert first == second
    calls = {"flow": 0}
    real = post._flow_legal

    def spy(*args):
        calls["flow"] += 1
        return real(*args)

    import pytest
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(post, "_flow_legal", spy)
        clear_center_view([_note(0, 1., 0, 1, 1, 2)], rng=_ScriptedDraws(.1))
    assert calls["flow"] == 0

def test_direction_first_state_uses_final_recut(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", 0.)
    notes = [_note(0, 0., 0, 1, 1, 4), _note(1, .3, 0, 1, 1, 3)]
    repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert notes[0]["d"] == 0 and notes[1]["d"] != 3

def _shared_audit(notes):
    """Independent re-check mirroring the production direction/overlap audits."""
    from map_creation import v2_postprocessing as post
    eps = float(getattr(config, "min_time_diff", .02))
    fast = float(getattr(config, "v2_fast_gap_seconds", .2) or .2)
    ordered = sorted(notes, key=lambda n: (n["time"], n["id"]))
    pred, _ = post._history_context(ordered, eps)
    from map_creation.v2_forward_directions import _fast_ids, _overlap_scan
    fast_ids = _fast_ids(ordered, eps, fast)
    flow = [n["id"] for n in ordered if n["d"] != 8
            and not post._flow_legal(n["d"], pred.get(n["id"]), n["x"], n["y"],
                                     n["time"], eps)]
    counter = [n["id"] for n in ordered
               if n["id"] in fast_ids and post._counter_direction(n)]
    same = max(eps, float(getattr(config, "v2_cleanup_same_dir_gap", .1) or 0))
    cross = max(eps, float(getattr(config, "v2_cleanup_swing_window", .15) or 0))
    return flow, counter, _overlap_scan(ordered, same, cross)


def test_hard_direction_repairs_saved_pair():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(9, 13.885532879818594, 1, 3, 1, 1),
             _note(10, 14.031409390561054, 1, 3, 0, 1)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    second = notes[1]
    assert second["y"] != 0 and second["d"] == 0
    assert _shared_audit(notes) == ([], [], [])
    assert out["hard_direction"]["changed"] >= 1


def test_hard_direction_repairs_mirror_pair():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(9, 0., 0, 0, 1, 1), _note(10, .145876, 0, 0, 0, 1)]
    repair_notes(notes, cap=lambda n: 9., rng=_ConstantDraw())
    assert (notes[1]["x"], notes[1]["y"], notes[1]["d"]) != (0, 0, 1)
    assert _shared_audit(notes) == ([], [], [])


def test_audit_directions_reports_findings():
    from map_creation.v2_forward_directions import audit_directions
    notes = [_note(9, 0., 1, 3, 1, 1), _note(10, .145876, 1, 3, 0, 1)]
    findings = audit_directions(notes)
    assert findings["flow"] == [10]
    notes = [_note(9, 0., 1, 3, 2, 1), _note(10, .15, 1, 3, 0, 0)]
    findings = audit_directions(notes)
    assert findings["fast_counter"] == [9, 10]


def test_enforce_first_endpoint_of_fast_pair():
    from map_creation.v2_forward_directions import enforce_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(9, 0., 1, 3, 2, 1), _note(10, .15, 1, 3, 1, 0)]
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert notes[0]["d"] != 1 and stats["changed"] >= 1
    assert _shared_audit(notes) == ([], [], [])


def test_enforce_saturation_raises_bounded():
    from map_creation.v2_forward_directions import enforce_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    block = [_note(i, .1, 0, x, y, 0)
             for i, (x, y) in enumerate((x, y) for x in (1, 2) for y in range(3))]
    block += [_note(20, .1, 0, 3, 2, 2)]
    notes = block + [_note(30, 0., 1, 3, 1, 1), _note(31, .11, 1, 3, 0, 1)]
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert stats["dropped"] == 1
    assert stats["dropped_notes"][0]["id"] == 31
    assert stats["dropped_notes"][0]["proposals"] <= 72
    assert 31 not in [n["id"] for n in notes]
    assert _shared_audit(notes) == ([], [], [])


def test_enforce_preserves_conflicts_and_rejects_swap():
    from map_creation.v2_forward_directions import enforce_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(9, 0., 0, 1, 1, 2), _note(10, .15, 0, 1, 1, 2),
             _note(11, .01, 1, 1, 1, 0)]
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert notes[1]["d"] != 2
    assert stats["changed"] >= 1


def test_enforce_honors_disabled_and_threshold(monkeypatch):
    from map_creation.v2_forward_directions import enforce_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(9, 0., 1, 3, 1, 0), _note(10, .2, 1, 3, 2, 1)]
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert notes[1]["d"] == 1 and stats["attempted"] == 0
    monkeypatch.setattr(config, "v2_cleanup_same_dir_recut", 0)
    monkeypatch.setattr(config, "v2_fast_gap_seconds", 0.)
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert stats["attempted"] == 0


def test_enforce_dot_run_and_same_onset_stack():
    from map_creation.v2_forward_directions import enforce_directions, audit_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(9, 0., 0, 0, 0, 2), _note(10, .15, 0, 0, 0, 8),
             _note(11, .3, 0, 2, 0, 2)]
    enforce_directions(notes, LocalNoteIndex(notes))
    assert notes[1]["d"] == 8
    stack = [_note(9, 0., 0, 0, 0, 2), _note(10, 1., 0, 0, 2, 3),
             _note(11, 1.01, 0, 2, 2, 5)]
    findings = audit_directions(stack)
    assert findings["flow"] == []


def test_validate_state_forward_drops_residual_direction():
    from map_creation.v2_pipeline import _validate_state
    state = {"notes": [_note(0, 1., 0, 1, 1, 2, kind="retained"),
                       _note(1, 1.15, 0, 1, 1, 2, kind="retained")],
             "chains": [{"id": 7, "anchor_id": 1, "time": 1.15,
                         "end_time": 1.6, "hand": 0, "x": 1, "y": 1,
                         "tx": 2, "ty": 1, "d": 2, "slices": 4}],
             "walls": [], "bombs": [], "arcs": [
                 {"id": 8, "time": 1., "end_time": 1.15, "hand": 0,
                  "endpoint_ids": [0, 1], "gap": .15}],
             "lighting": []}
    findings = _validate_state(state, 4., playability_mode="forward")
    assert [note["id"] for note in state["notes"]] == [0]
    assert state["chains"] == [] and state["arcs"] == []
    assert findings["hard_dropped_final"] == 1
    from map_creation.v2_forward_directions import audit_directions
    assert audit_directions(state["notes"]) == {"flow": [], "fast_counter": []}

def test_enforce_conflict_id_swap_rejected():
    from map_creation.v2_forward_directions import enforce_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(9, 0., 0, 0, 1, 2), _note(10, .15, 0, 0, 1, 2),
             _note(11, .01, 1, 1, 0, 0), _note(12, .13, 1, 0, 0, 0)]
    index = LocalNoteIndex(notes)
    stats = enforce_directions(notes, index)
    assert stats["changed"] + stats["dropped"] >= 1
    moved = notes[1]
    assert index.conflict_ids(moved) <= index.conflict_ids(
        {"id": 10, "time": .15, "hand": 0, "x": 0, "y": 1, "d": 2})


def test_enforce_empty_reports_drop_fields():
    from map_creation.v2_forward_directions import enforce_directions
    from map_creation.v2_forward_clearance import LocalNoteIndex
    stats = enforce_directions([], LocalNoteIndex([]))
    assert stats["dropped"] == 0 and stats["dropped_notes"] == []


def test_hard_direction_repairs_sub_gap_same_direction_pairs():
    """Distinct onsets inside the old 0.1 s same-stroke gap are now hard."""
    from map_creation.v2_forward_postprocessing import repair_notes
    for gap in (.03, .06):
        notes = [_note(0, 0., 0, 1, 1, 0), _note(1, gap, 0, 0, 1, 0)]
        out = repair_notes(notes, cap=lambda n: float("inf"), rng=_ConstantDraw())
        assert _shared_audit(notes) == ([], [], [])
        assert (out["hard_direction"]["changed"]
                + out["hard_direction"]["dropped"]) >= 1


def test_hard_direction_repairs_three_in_a_row():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 1, 1, 0), _note(1, .03, 0, 0, 1, 0),
             _note(2, .06, 0, 2, 1, 0)]
    out = repair_notes(notes, cap=lambda n: float("inf"), rng=_ConstantDraw())
    assert _shared_audit(notes) == ([], [], [])
    assert (out["hard_direction"]["changed"]
            + out["hard_direction"]["dropped"]) >= 1


def test_hard_direction_same_onset_stack_remains_legal():
    """Same-onset (<= min_time_diff) same-direction stacks stay exempt."""
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 1., 0, 0, 1, 0), _note(1, 1.01, 0, 1, 1, 0)]
    out = repair_notes(notes, cap=lambda n: float("inf"), rng=_ConstantDraw())
    assert [n["d"] for n in notes] == [0, 0]
    assert out["hard_direction"]["attempted"] == 0
    assert _shared_audit(notes) == ([], [], [])


def test_audit_overlaps_reports_flagged_ids():
    from map_creation.v2_forward_directions import audit_overlaps
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 0, 0, 1, 0),
             _note(2, .3, 0, 2, 1, 0), _note(3, .5, 1, 2, 1, 0)]
    assert audit_overlaps(notes) == {"overlap": [1]}
    assert audit_overlaps(notes[:1] + notes[2:]) == {"overlap": []}


def test_enforce_overlap_drops_later_same_cell_note():
    from map_creation.v2_forward_directions import enforce_overlaps, audit_overlaps
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 0, 0, 1, 1)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats["checked"] == 2 and stats["dropped"] == 1
    assert stats["dropped_notes"] == [
        {"id": 1, "time": .05, "proposals": 0, "reason": "overlap"}]
    assert [n["id"] for n in notes] == [0]
    assert audit_overlaps(notes) == {"overlap": []}


def test_enforce_overlap_cross_hand_and_equal_time_deterministic():
    from map_creation.v2_forward_directions import enforce_overlaps
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 1, 0, 1, 0)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats["dropped"] == 1 and [n["id"] for n in notes] == [0]
    notes = [_note(0, 1., 0, 0, 1, 0), _note(1, 1., 1, 0, 1, 0)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats["dropped_notes"][0]["id"] == 1
    assert [n["id"] for n in notes] == [0]
    notes = [_note(0, 1., 0, 0, 1, 0), _note(1, 1., 0, 0, 1, 1)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats["dropped_notes"][0]["id"] == 1
    assert [n["id"] for n in notes] == [0]


def test_enforce_overlap_window_and_nonconsecutive_allowed():
    from map_creation.v2_forward_directions import enforce_overlaps
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .2, 1, 0, 1, 0)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats["dropped"] == 0 and len(notes) == 2
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 0, 1, 1, 0),
             _note(2, .06, 0, 0, 1, 0)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats["dropped"] == 0 and len(notes) == 3


def test_repair_hard_overlap_removes_residual_pile():
    from map_creation.v2_forward_postprocessing import repair_notes
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 0, 0, 1, 0)]
    out = repair_notes(notes, cap=lambda n: float("inf"), rng=_ConstantDraw())
    assert out["hard_overlap"]["dropped"] == 1
    assert out["hard_overlap"]["dropped_notes"][0]["id"] == 1
    assert [n["id"] for n in notes] == [0]
    assert _shared_audit(notes) == ([], [], [])


def test_enforce_overlap_disabled_with_direction_rules(monkeypatch):
    from map_creation.v2_forward_directions import enforce_overlaps, audit_overlaps
    from map_creation.v2_forward_clearance import LocalNoteIndex
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_cleanup_same_dir_recut", 0)
    monkeypatch.setattr(config, "v2_fast_counter_direction", "off")
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 0, 0, 1, 0)]
    stats = enforce_overlaps(notes, LocalNoteIndex(notes))
    assert stats == {"checked": 0, "dropped": 0, "dropped_notes": []}
    assert len(notes) == 2 and audit_overlaps(notes) == {"overlap": []}
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .05, 0, 0, 1, 0)]
    out = repair_notes(notes, cap=lambda n: float("inf"), rng=_ConstantDraw())
    assert out["hard_overlap"]["dropped"] == 0 and len(notes) == 2


def test_validate_state_forward_drops_residual_overlap():
    from map_creation.v2_pipeline import _validate_state
    state = {"notes": [_note(0, 1., 0, 0, 1, 0, kind="retained"),
                       _note(1, 1.05, 0, 0, 1, 1, kind="retained")],
             "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []}
    findings = _validate_state(state, 4., playability_mode="forward")
    assert [note["id"] for note in state["notes"]] == [0]
    assert findings["hard_dropped_final"] == 1
    from map_creation.v2_forward_directions import audit_overlaps
    assert audit_overlaps(state["notes"]) == {"overlap": []}


def test_validate_state_forward_still_raises_on_surviving_residual():
    from map_creation.v2_pipeline import _validate_state
    from map_creation import v2_forward_directions as directions
    state = {"notes": [_note(0, 1., 0, 1, 1, 2, kind="retained"),
                       _note(1, 1.15, 0, 1, 1, 2, kind="retained")],
             "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []}
    audits = iter(({"flow": [1], "fast_counter": []},
                   {"flow": [1], "fast_counter": []}))
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(directions, "audit_directions", lambda notes: next(audits))
        with _pytest.raises(ValueError, match="direction"):
            _validate_state(state, 4., playability_mode="forward")


def _random_notes(rng, count=80):
    """Seeded same-hand/cross-hand note lists with dots and onset stacks."""
    notes, nid = [], 0
    for hand in (0, 1):
        time = 0.
        while time < count * .05:
            time += float(rng.choice((.01, .02, .05, .1, .15, .25, .55, .8)))
            stack = 1 + int(rng.random() < .25) + int(rng.random() < .1)
            for _ in range(stack):
                notes.append({"id": nid, "time": time + float(rng.choice((0., .005))),
                              "hand": hand, "x": rng.randrange(0, 4),
                              "y": rng.randrange(0, 3),
                              "d": rng.randrange(0, 9)})
                nid += 1
    rng.shuffle(notes)
    return notes


def test_detect_matches_legacy_findings():
    """``_detect`` mirrors the strict in-phase ``_findings`` exactly."""
    import random
    from map_creation import v2_postprocessing as post
    from map_creation.v2_forward_directions import _detect, _findings, _ordered
    from map_creation.v2_forward_directions import _gates
    eps, flow_on, _, fast, counter_on, _, _, _ = _gates()
    for seed in range(12):
        notes = _random_notes(random.Random(seed))
        with post.playability_generation_phase():
            flow, counter = _findings(_ordered(notes), eps, flow_on, eps,
                                      fast, counter_on)
        flagged = _detect(notes)
        assert flagged["flow"] == set(flow), seed
        assert flagged["counter"] == set(counter), seed


def test_audit_directions_matches_legacy_findings_with_slow_turns():
    """Outside the phase, slow-turn leniency still applies — the audit must
    return the legacy ``_findings`` ids, not the strict superset."""
    import random
    from map_creation.v2_forward_directions import (
        audit_directions, _findings, _gates, _ordered)
    eps, flow_on, _, fast, counter_on, _, _, _ = _gates()
    for seed in range(12):
        notes = _random_notes(random.Random(seed))
        flow, counter = _findings(_ordered(notes), eps, flow_on, eps,
                                  fast, counter_on)
        findings = audit_directions(notes)
        assert findings["flow"] == flow, seed
        assert findings["fast_counter"] == counter, seed


def _reference_enforce(notes, index):
    """Pre-worklist chronological sweep, kept as the equivalence oracle."""
    from collections import deque
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_reach import forbidden_cut
    from map_creation.v2_forward_directions import (
        _MAX_PROPOSALS, _arrow_order, _fast_ids, _findings, _gates,
        _ordered, _overlap_conflicts, _TOL)
    eps, flow_on, _, fast, counter_on, reach_on, same_hand, cross = _gates()
    stats = {"attempted": 0, "proposals": 0, "changed": 0, "relocated": 0,
             "dropped": 0, "dropped_notes": []}
    if not notes or not (flow_on or counter_on):
        return stats
    view = _ordered(notes)
    fast_ids = _fast_ids(view, eps, fast)
    bounds = {0: (0, 2), 1: (1, 3)}
    diagonals = [sum(note["d"] in (4, 5, 6, 7) for note in view)]
    total = len(view)
    fraction = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
    pending = {0: deque(), 1: deque()}
    last_eff = {0: None, 1: None}
    dropped = set()

    def pred_of(note):
        hand = note["hand"]
        while pending[hand] and pending[hand][0]["time"] < note["time"] - eps:
            earlier = pending[hand].popleft()
            before = last_eff[hand]
            if earlier["d"] != 8:
                eff = earlier["d"]
            elif before is not None and before[3] is not None:
                eff = post._dot_virtual_direction(before[3], before[1], before[2],
                                                  earlier["x"], earlier["y"])
            else:
                eff = None
            last_eff[hand] = (earlier["time"], earlier["x"], earlier["y"], eff)
        return last_eff[hand]

    for note in view:
        if note["d"] == 8:
            pending[note["hand"]].append(note)
            continue
        pred = pred_of(note)
        flow_bad = (flow_on
                    and not post._flow_legal(note["d"], pred, note["x"], note["y"],
                                             note["time"], eps))
        counter_bad = (counter_on and note["id"] in fast_ids
                       and post._counter_direction(note))
        if flow_bad or counter_bad:
            stats["attempted"] += 1
            base_cell = (note["x"], note["y"])
            baseline = index.conflict_ids(note)
            low, high = bounds[note["hand"]]
            cells = [base_cell] + sorted(
                [(x, y) for x in range(low, high + 1) for y in range(3)
                 if (x, y) != base_cell],
                key=lambda c: (abs(c[0] - base_cell[0]) + abs(c[1] - base_cell[1]),
                               c[1], c[0]))
            tried = 0
            accepted = False
            for x, y in cells:
                if tried >= _MAX_PROPOSALS:
                    break
                arrows = _arrow_order(note, pred)
                if diagonals[0] > total * fraction + _TOL:
                    arrows = ([d for d in arrows if d not in (4, 5, 6, 7)]
                              + [d for d in arrows if d in (4, 5, 6, 7)])
                for d in arrows:
                    if tried >= _MAX_PROPOSALS:
                        break
                    tried += 1
                    stats["proposals"] += 1
                    trial = {"id": note["id"], "time": note["time"],
                             "hand": note["hand"], "x": x, "y": y, "d": d}
                    if (flow_on
                            and not post._flow_legal(d, pred, x, y,
                                                     note["time"], eps)):
                        continue
                    if (counter_on and note["id"] in fast_ids
                            and post._counter_direction(trial)):
                        continue
                    if reach_on and forbidden_cut(trial):
                        continue
                    if _overlap_conflicts(index, trial, dropped | getattr(index, "_dropped", frozenset()),
                                          same_hand, cross):
                        continue
                    if not index.conflict_ids(trial) <= baseline:
                        continue
                    if note["d"] in (4, 5, 6, 7):
                        diagonals[0] -= 1
                    if d in (4, 5, 6, 7):
                        diagonals[0] += 1
                    note["x"], note["y"], note["d"] = x, y, d
                    stats["changed"] += 1
                    stats["relocated"] += int((x, y) != base_cell)
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                dropped.add(note["id"])
                index.remove_ids({note["id"]})
                stats["dropped"] += 1
                stats["dropped_notes"].append(
                    {"id": note["id"], "time": note["time"],
                     "proposals": tried, "flow": flow_bad,
                     "fast_counter": counter_bad})
                continue
        pending[note["hand"]].append(note)
    if dropped:
        notes[:] = [note for note in notes if note["id"] not in dropped]
        survivors = _ordered(notes)
        flow, counter = _findings(survivors, eps, flow_on, eps, fast, counter_on)
        if flow or counter:
            raise ValueError("reference residuals")
    return stats


def test_enforce_worklist_matches_reference_sweep():
    """The flagged-worklist repair reproduces the old interleaved sweep."""
    import random
    from map_creation.v2_forward_clearance import LocalNoteIndex
    from map_creation.v2_forward_directions import enforce_directions
    for seed in range(8):
        expected = _random_notes(random.Random(100 + seed))
        actual = [dict(note) for note in expected]
        ref_stats = _reference_enforce(expected, LocalNoteIndex(expected))
        new_stats = enforce_directions(actual, LocalNoteIndex(actual))
        assert actual == expected, seed
        assert new_stats == ref_stats, seed


def test_enforce_cascade_repairs_successor_in_same_call():
    """A repair that breaks its same-hand successor is caught and fixed."""
    from map_creation.v2_forward_directions import (
        audit_directions, enforce_directions)
    from map_creation.v2_forward_clearance import LocalNoteIndex
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .15, 0, 0, 1, 0),
             _note(2, .3, 0, 0, 1, 6)]
    assert audit_directions(notes) == {"flow": [1], "fast_counter": []}
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert stats["attempted"] == 2
    assert stats["changed"] + stats["dropped"] == 2
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}


def test_enforce_drop_policy_removes_flagged_without_proposals(monkeypatch):
    from map_creation.v2_forward_directions import (
        audit_directions, enforce_directions)
    from map_creation.v2_forward_clearance import LocalNoteIndex
    monkeypatch.setattr(config, "v2_hard_direction_policy", "drop")
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .15, 0, 0, 1, 0),
             _note(2, .3, 0, 0, 1, 6)]
    stats = enforce_directions(notes, LocalNoteIndex(notes))
    assert stats["proposals"] == 0 and stats["changed"] == 0
    assert stats["dropped"] == 1 and stats["dropped_notes"][0]["id"] == 1
    assert stats["dropped_notes"][0]["proposals"] == 0
    assert [note["id"] for note in notes] == [0, 2]
    assert audit_directions(notes) == {"flow": [], "fast_counter": []}
