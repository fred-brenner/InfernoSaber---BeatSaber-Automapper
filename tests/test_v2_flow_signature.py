"""Signature extraction, specific counters and distance tests.

Pure reporting module: hand-built note lists produce exact counters,
the committed pooled reference constants self-distance to zero, and the
gap-band boundaries at .1/.2/.4/.6 are exact. Definitions mirror the
derivation script (singleton non-dot groups, eps .02, fast gap <= .4);
all features — bands, runs, occupancy, ABAB, specific counts — pool
both hands (the derivation's hand-1-only band overwrite is corrected).
"""
from copy import deepcopy

import pytest


def _note(id, time, hand, x, y, d):
    return {"id": id, "time": time, "hand": hand, "x": x, "y": y, "d": d}


def _band(pairs=0, **counts):
    band = {"pairs": pairs, "outer_pair:0": 0, "outer_pair:1": 0}
    band.update({f"lane:{k}": 0 for k in ("same", "adjacent", "wide")})
    band.update({f"dy:{k}": 0 for k in range(-2, 3)})
    band.update({f"step:{k}": 0 for k in range(5)})
    band.update(counts)
    return band


def _bands(**named):
    return {name: named.get(name, _band())
            for name in ("<.1", ".1-.2", ".2-.4", ".4-.6", ">.6")}


def _occupancy(cells=None):
    cells = cells or {}
    return {f"{x},{y}": cells.get((x, y), 0)
            for x in range(4) for y in range(3)}


def _runs(counts=None):
    return {**{str(k): 0 for k in range(1, 9)},
            **{str(k): v for k, v in (counts or {}).items()}}


def test_band_index_exact_boundaries():
    from map_creation import v2_flow_signature as fs
    gaps = (.0, .05, .1, .15, .2, .3, .4, .45, .6, .61, 9.)
    assert [fs.band_index(g) for g in gaps] == [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]


def test_signature_transition_bands_exact():
    from map_creation import v2_flow_signature as fs
    rows = [(0.0, 2, 0, 0), (.09, 3, 0, 1), (.19, 0, 1, 2),
            (.50, 0, 2, 3), (1.05, 1, 2, 0), (1.70, 1, 0, 1)]
    notes = [_note(i, t, 1, x, y, d) for i, (t, x, y, d) in enumerate(rows)]
    sig = fs.signature(notes)
    assert sig["notes"] == 6
    assert sig["bands"] == _bands(
        **{"<.1": _band(1, **{"lane:adjacent": 1, "dy:0": 1, "step:4": 1,
                              "outer_pair:1": 1}),
           ".1-.2": _band(1, **{"lane:wide": 1, "dy:1": 1, "step:2": 1,
                                "outer_pair:0": 1}),
           ".2-.4": _band(1, **{"lane:same": 1, "dy:1": 1, "step:4": 1,
                                "outer_pair:0": 1}),
           ".4-.6": _band(1, **{"lane:adjacent": 1, "dy:0": 1, "step:2": 1,
                                "outer_pair:0": 1}),
           ">.6": _band(1, **{"lane:same": 1, "dy:-2": 1, "step:4": 1,
                              "outer_pair:0": 1})})
    # runs break on x change or gap > .4: [2] [3] [0,0] [1] [1]
    assert sig["runs"] == _runs({1: 4, 2: 1})
    # only the first 4-group window has all gaps <= .4; xs 2,3,0,0 do not alternate
    assert sig["abab4"] == {"eligible": 1, "alternating": 0}
    assert sig["occupancy"] == {
        "0": _occupancy(),
        "1": _occupancy({(2, 0): 1, (3, 0): 1, (0, 1): 1,
                         (0, 2): 1, (1, 2): 1, (1, 0): 1})}


def test_signature_gap_boundaries_bucketed():
    from map_creation import v2_flow_signature as fs
    # gaps .1/.2/.4/.6 land in ".1-.2", ".2-.4", ".2-.4", ".4-.6"
    rows = [(0.0, 2, 0), (.1, 3, 0), (.3, 2, 0), (.7, 3, 0), (1.3, 2, 0)]
    notes = [_note(i, t, 1, x, y, 0) for i, (t, x, y) in enumerate(rows)]
    sig = fs.signature(notes)
    pairs = {name: band["pairs"] for name, band in sig["bands"].items()}
    assert pairs == {"<.1": 0, ".1-.2": 1, ".2-.4": 2, ".4-.6": 1, ">.6": 0}
    # first window (.1/.2/.4 all <= .4) is an eligible alternating ABAB
    assert sig["abab4"] == {"eligible": 1, "alternating": 1}


def test_signature_abab_alternating_and_outer_pairs():
    from map_creation import v2_flow_signature as fs
    notes = [_note(i, .15 * i, 1, x, 1, d)
             for i, (x, d) in enumerate(((2, 0), (3, 1), (2, 0), (3, 1)))]
    sig = fs.signature(notes)
    assert sig["bands"][".1-.2"] == _band(
        3, **{"lane:adjacent": 3, "dy:0": 3, "step:4": 3, "outer_pair:1": 3})
    assert sig["abab4"] == {"eligible": 1, "alternating": 1}
    assert sig["runs"] == _runs({1: 4})
    assert sig["occupancy"]["1"] == _occupancy({(2, 1): 2, (3, 1): 2})


def test_signature_dot_and_stack_groups_break_pairs():
    from map_creation import v2_flow_signature as fs
    dotted = [_note(0, .0, 1, 2, 0, 0), _note(1, .15, 1, 3, 0, 8),
              _note(2, .30, 1, 2, 0, 0)]
    sig = fs.signature(dotted)
    assert all(band["pairs"] == 0 for band in sig["bands"].values())
    assert sig["runs"] == _runs({1: 2})
    assert sig["abab4"] == {"eligible": 0, "alternating": 0}
    stacked = [_note(0, .0, 1, 2, 0, 0), _note(1, .15, 1, 3, 0, 0),
               _note(2, .15, 1, 2, 0, 1), _note(3, .30, 1, 2, 0, 0)]
    sig = fs.signature(stacked)
    assert all(band["pairs"] == 0 for band in sig["bands"].values())
    assert sig["runs"] == _runs({1: 2})
    # stack members still count toward occupancy
    assert sig["occupancy"]["1"]["3,0"] == 1


def test_signature_slow_gap_counts_band_not_abab_or_run():
    from map_creation import v2_flow_signature as fs
    notes = [_note(i, t, 1, 2, 1, 0) for i, t in enumerate((0., .15, .30, .80))]
    sig = fs.signature(notes)
    assert sig["bands"][".4-.6"]["pairs"] == 1
    assert sig["bands"][".1-.2"]["pairs"] == 2
    assert sig["abab4"] == {"eligible": 0, "alternating": 0}
    assert sig["runs"] == _runs({1: 1, 3: 1})


def test_signature_bands_pool_both_hands():
    """Band transition counters sum hands 0 and 1; a hand-0-only map
    must still produce non-empty band counters (derivation bug fixed:
    the script overwrote hand 0's counters with hand 1's)."""
    from map_creation import v2_flow_signature as fs
    notes = [_note(0, .0, 0, 0, 0, 0), _note(1, .15, 0, 1, 0, 1),
             _note(2, .0, 1, 2, 0, 0), _note(3, .15, 1, 3, 0, 1)]
    sig = fs.signature(notes)
    assert sig["bands"][".1-.2"]["pairs"] == 2
    assert sig["bands"][".1-.2"]["outer_pair:1"] == 2
    assert sig["runs"]["1"] == 4
    assert sig["occupancy"]["0"]["0,0"] == 1
    assert sig["occupancy"]["1"]["2,0"] == 1
    hand0_only = fs.signature(notes[:2])
    assert hand0_only["bands"][".1-.2"] == _band(
        1, **{"lane:adjacent": 1, "dy:0": 1, "step:4": 1, "outer_pair:1": 1})


def test_specific_counts_known_cases():
    from map_creation import v2_flow_signature as fs
    notes = [
        _note(0, 0.0, 0, 1, 1, 0),   # inner y1, fast (next group .3)
        _note(1, .3, 0, 2, 2, 2),    # inner top horizontal, terminal (.5)
        _note(2, .8, 0, 0, 1, 3),    # mid horizontal outer, fast (.2)
        _note(3, 1.0, 0, 1, 2, 3),   # inner top horizontal, fast (.4)
        _note(4, 1.4, 0, 1, 0, 1),   # bottom down, terminal (.5)
        _note(5, 1.9, 0, 2, 1, 1),   # inner y1, fast (.4)
        _note(6, 2.3, 0, 1, 1, 0),   # stack member inner y1, fast (.3)
        _note(7, 2.3, 0, 3, 0, 1),   # stack member bottom down, fast
        _note(8, 2.6, 0, 0, 0, 1),   # bottom down, terminal (last group)
        _note(9, 5.0, 1, 1, 1, 0),   # other hand: inner y1 fast (.2)
        _note(10, 5.2, 1, 3, 0, 1),  # other hand: bottom down terminal
    ]
    assert fs.specific_counts(notes) == {
        "notes": 11, "inner_y1_fast": 4, "inner_y1_terminal": 0,
        "inner_y2_fast": 1, "inner_y2_terminal": 1,
        "top_horizontal": 2, "top_horizontal_fast": 1,
        "top_horizontal_terminal": 1, "mid_horizontal_fast": 1,
        "inner_top_horizontal_fast": 1, "bottom_down_fast": 1}


def test_specific_counts_fast_gap_edge_and_dot_follower():
    from map_creation import v2_flow_signature as fs
    # follower type does not matter: only the next onset group's time
    base = [_note(0, .0, 0, 1, 1, 0)]
    counts = fs.specific_counts(base + [_note(1, .2, 0, 0, 0, 8)])
    assert counts["inner_y1_fast"] == 1 and counts["inner_y1_terminal"] == 0
    counts = fs.specific_counts(base + [_note(1, .4, 0, 0, 0, 1)])
    assert counts["inner_y1_fast"] == 1
    counts = fs.specific_counts(base + [_note(1, .41, 0, 0, 0, 1)])
    assert counts["inner_y1_terminal"] == 1 and counts["inner_y1_fast"] == 0
    counts = fs.specific_counts(base)
    assert counts["inner_y1_terminal"] == 1


def test_distance_self_and_committed_reference_zero():
    from map_creation import v2_flow_signature as fs
    notes = [_note(i, .15 * i, 1, x, 1, i % 2) for i, x in
             enumerate((2, 3, 2, 3))]
    sig = fs.signature(notes)
    assert fs.signature_distance(sig, sig) == 0.
    ref = fs.load_reference()
    assert fs.signature_distance(ref, ref) == 0.
    assert fs.signature_distance(ref) == 0.  # default loads committed JSON
    assert fs.signature_distance(sig) > 0.


def test_distance_component_weights():
    from map_creation import v2_flow_signature as fs
    sig = fs.signature([_note(i, .15 * i, 1, x, 1, i % 2)
                        for i, x in enumerate((2, 3, 2, 3))])
    ref = deepcopy(sig)
    ref["bands"][".1-.2"].update({"lane:adjacent": 0, "lane:same": 3})
    assert fs.signature_distance(sig, ref) == pytest.approx(3.)   # lane x3
    ref = deepcopy(sig)
    ref["bands"][".1-.2"].update({"step:4": 0, "step:0": 3})
    assert fs.signature_distance(sig, ref) == pytest.approx(2.)   # step x2
    ref = deepcopy(sig)
    ref["bands"][".1-.2"].update({"dy:0": 0, "dy:1": 3})
    assert fs.signature_distance(sig, ref) == pytest.approx(1.)   # dy x1
    ref = deepcopy(sig)
    ref["runs"].update({"1": 0, "4": 1})
    assert fs.signature_distance(sig, ref) == pytest.approx(2.)   # runs x2
    ref = deepcopy(sig)
    ref["occupancy"]["1"].update({"2,1": 0, "3,1": 0, "0,0": 4})
    assert fs.signature_distance(sig, ref) == pytest.approx(1.)   # occupancy x1
    ref = deepcopy(sig)
    ref["abab4"]["alternating"] = 0
    assert fs.signature_distance(sig, ref) == pytest.approx(1.)   # abab x1


def test_distance_band_mass_component():
    """Band-mass TV contributes only when pair shares across bands differ."""
    from map_creation import v2_flow_signature as fs
    sig = fs.signature([_note(i, .15 * i, 1, x, 1, i % 2)
                        for i, x in enumerate((2, 3, 2, 3))])
    # moving only the pair mass leaves every per-band conditional share
    # untouched, isolating the band-mass term: {.1-.2: 1} vs {<.1: 1} -> TV 1
    ref = deepcopy(sig)
    ref["bands"][".1-.2"]["pairs"] = 0
    ref["bands"]["<.1"]["pairs"] = 3
    assert fs.signature_distance(sig, ref) == pytest.approx(1.)   # band mass x1
    # identical shares across bands -> no contribution even if totals differ
    ref = deepcopy(sig)
    for name in ref["bands"]:
        ref["bands"][name]["pairs"] *= 2
    assert fs.signature_distance(sig, ref) == pytest.approx(0.)


def test_distance_empty_signature():
    from map_creation import v2_flow_signature as fs
    empty = fs.signature([])
    assert fs.signature_distance(empty, empty) == 0.
    assert fs.signature_distance(empty) > 0.


def test_signature_report_shape():
    from map_creation import v2_flow_signature as fs
    notes = [_note(0, .0, 0, 1, 1, 0), _note(1, .3, 0, 2, 1, 1)]
    report = fs.signature_report(notes)
    assert set(report) == {"signature", "distance", "specific"}
    assert report["signature"] == fs.signature(notes)
    assert report["specific"] == fs.specific_counts(notes)
    assert report["distance"] == fs.signature_distance(report["signature"])
    assert isinstance(report["distance"], float)


def test_committed_reference_constants():
    from map_creation import v2_flow_signature as fs
    ref = fs.load_reference()
    assert ref["notes"] == 15578
    pairs = {name: ref["bands"][name]["pairs"] for name in ref["bands"]}
    assert pairs == {"<.1": 689, ".1-.2": 5505, ".2-.4": 3655,
                     ".4-.6": 626, ">.6": 410}
    assert ref["runs"] == {"1": 7564, "2": 1311, "3": 385, "4": 68,
                           "5": 48, "6": 14, "7": 8, "8": 9}
    assert ref["abab4"] == {"eligible": 7667, "alternating": 926}
    assert (sum(ref["occupancy"]["0"].values())
            + sum(ref["occupancy"]["1"].values()) == 15578)
