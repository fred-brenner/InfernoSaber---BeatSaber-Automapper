"""Opt-in comparison gate: frozen pre-repair excerpts replayed through repair.

Each ``tests/fixtures/flow_signature_<slug>.json`` holds one contiguous 60 s
slice of the exact ``input`` list a paired ``lanes-final`` production run fed
to ``v2_forward_postprocessing.repair_notes``
(``devin-artifacts/lane-flow-20261117``, requested_nps 8 -> internal
difficulty 32). Windows were picked for dense stream content — the most
same-hand gaps < .2 s inside the slice: astronomia t0=17.0 (424 fast gaps,
602 notes), heeartbeeat t0=38.5 (239 fast gaps, 502 notes). Ids and times
are preserved; the excerpt loses pre-t0 context, so boundary effects are
expected and the goldens are measured on the same input.

The test re-runs ``repair_notes`` under the paired-run config, requires the
serialized hard audits (same-hand flow, fast counter-direction, same-cell
overlap) to stay clean, and asserts the signature distance plus the
specific style counters do not regress past the committed goldens.
Excluded from the default suite through the ``comparison`` marker —
the suite's convention is ``pytest -m 'not comparison'``.

Regenerate goldens after an intentional improvement:

    FLOW_SIGNATURE_UPDATE_GOLDENS=1 \
        python -m pytest tests/test_flow_signature_comparison.py -m comparison

then review the goldens diff before committing it.
"""
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import pytest

from tools.config import config

pytestmark = pytest.mark.comparison

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SLUGS = ("astronomia", "heeartbeeat")

# The config the paired lanes-final runs repaired under: production
# defaults at requested_nps 8 — internal difficulty 32 sets max_speed
# (incidentally equal to the 4*8.0 default) and add_beat_intensity stays
# 90 (add_silence +10, emphasize -0), so speed_cap_scale() = .7 * .9 = .63.
_PAIRED_RUN_CONFIG = {
    "min_time_diff": .02,
    "v2_cleanup_swing_window": .15,
    "v2_cleanup_same_dir_gap": .1,
    "v2_cleanup_same_dir_recut": .5,
    "v2_flow_min_turn_steps": 3,
    "v2_fast_gap_seconds": .2,
    "v2_fast_counter_direction": "recut",
    "v2_diagonal_max_fraction": .1,
    "v2_hard_direction_policy": "repair",
    "v2_playability_enabled": True,
    "v2_playability_priority": True,
    "v2_playability_reach": True,
    "v2_playability_late_turns": True,
    "v2_playability_slow90_seconds": .5,
    "v2_playability_slow45_seconds": .65,
    "v2_cleanup_note_heuristics": False,
    "v2_note_postprocessing_mode": "forward",
    "v2_pattern_flow": True,
    "v2_pattern_obstacles": True,
    "max_speed": 32.,
    "add_beat_intensity": 90,
    "add_beat_intensity_orig": 90,
    "v2_speed_cap_scale": .7,
    "v2_speed_cap_start_notes": 30,
    "v2_speed_cap_start_factor": .35,
    "v2_speed_cap_gap_factors": (4.0, 3.5, 1.8, 1.0, 0.7),
    "reaction_time": 1.15,
    "reaction_time_fact": .013,
    "cdf": 1.2,
    "cdf_lr": 1.15,
}


@pytest.fixture(autouse=True)
def paired_run_config(monkeypatch):
    for key, value in _PAIRED_RUN_CONFIG.items():
        monkeypatch.setattr(config, key, value)
    yield


def _load(slug):
    with open(_FIXTURES / f"flow_signature_{slug}.json", encoding="utf-8") as fh:
        return json.load(fh)


def _goldens():
    with open(_FIXTURES / "flow_signature_goldens.json", encoding="utf-8") as fh:
        return json.load(fh)


def _same_x_triples(notes):
    """All touched same-x triples: every triple contains a passed id."""
    from map_creation.v2_forward_flow import FlowContext
    from map_creation.v2_forward_lanes import repetition_cost
    return repetition_cost(FlowContext(notes), notes)


def _measure(notes):
    """Repair one fixture note list and bundle every gated measurement."""
    from map_creation.v2_forward_directions import audit_directions, audit_overlaps
    from map_creation.v2_forward_flow import audit_patterns
    from map_creation.v2_forward_postprocessing import repair_notes
    from map_creation import v2_flow_signature as fs
    report = repair_notes(notes, rng=random.Random(0))
    assert "pattern_flow" in report  # pinned config must keep the flow branch
    return {
        "distance": fs.signature_distance(fs.signature(notes)),
        "counts": {**fs.specific_counts(notes),
                   "same_x_triples": _same_x_triples(notes)},
        "audits": {"directions": audit_directions(notes),
                   "overlaps": audit_overlaps(notes)},
        "pattern_audit": audit_patterns(notes),
        "notes_out": len(notes),
    }


@pytest.mark.parametrize("slug", _SLUGS)
def test_repair_signature_matches_golden(slug):
    fixture = _load(slug)
    goldens = _goldens()
    golden = goldens["fixtures"][slug]
    measured = _measure(deepcopy(fixture["notes"]))

    if os.environ.get("FLOW_SIGNATURE_UPDATE_GOLDENS") == "1":
        goldens["fixtures"][slug] = {
            "distance": measured["distance"],
            "counts": {**measured["counts"], "notes": measured["notes_out"]},
            "audits": measured["audits"],
            "pattern_audit": measured["pattern_audit"],
            "notes_out": measured["notes_out"],
        }
        if slug == _SLUGS[-1]:
            path = _FIXTURES / "flow_signature_goldens.json"
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(goldens, fh, indent=2)
                fh.write("\n")
        return

    assert measured["audits"]["directions"] == {"flow": [], "fast_counter": []}
    assert measured["audits"]["overlaps"] == {"overlap": []}

    epsilon = goldens["tolerances"]["distance_epsilon"]
    assert measured["distance"] <= golden["distance"] + epsilon, (
        f"{slug}: signature distance {measured['distance']:.4f} exceeds "
        f"golden {golden['distance']:.4f} + epsilon {epsilon}")

    tolerance = goldens["tolerances"]["count_tolerance"]
    for key, limit in golden["counts"].items():
        assert measured["counts"][key] <= limit + tolerance, (
            f"{slug}: {key} = {measured['counts'][key]} exceeds golden "
            f"{limit} + tolerance {tolerance}")
    assert measured["notes_out"] == golden["counts"]["notes"]

    # Allowed patterns get a floor, not a ceiling: a change that crushes
    # bottom-row downcuts or inner-top fast notes is a regression even
    # though nothing bans them.
    floors = goldens["tolerances"]["protected_counts"]
    fraction = goldens["tolerances"]["protected_floor_fraction"]
    for key in floors:
        floor = golden["counts"][key] * fraction
        assert measured["counts"][key] >= floor, (
            f"{slug}: protected {key} = {measured['counts'][key]} fell "
            f"below {fraction:.0%} of golden {golden['counts'][key]}")

    for key, limit in golden["pattern_audit"].items():
        if key == "eligible":  # denominator, not a violation count
            continue
        assert measured["pattern_audit"][key] <= limit, (
            f"{slug}: pattern_audit.{key} = {measured['pattern_audit'][key]} "
            f"exceeds golden {limit}")
