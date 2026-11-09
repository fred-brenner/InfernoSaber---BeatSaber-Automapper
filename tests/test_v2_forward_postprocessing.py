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
    before = [dict(n) for n in dense]
    out = repair_notes(dense, cap=lambda n: 0., rng=_ConstantDraw())
    assert out["collision"]["unresolved"] > 0
    assert [n["id"] for n in dense] == [n["id"] for n in before]
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
    assert notes == original
    assert out["collision"]["unresolved"] > 0
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
