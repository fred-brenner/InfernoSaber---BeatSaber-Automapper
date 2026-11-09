"""Sustained upper-center exposure keeps fixed, zero-inclusive endpoints."""
import importlib
import importlib.util
import json
import subprocess
import sys

import pytest


def api():
    name = 'map_creation.v2_playability_visibility'
    assert importlib.util.find_spec(name) is not None, 'shared visibility events are missing'
    return importlib.import_module(name)


def note(i, time, *, hand=0, x=1, y=1, d=0):
    return dict(id=i, time=time, hand=hand, x=x, y=y, d=d)


def reference(cutoff=.5, *, supported=True):
    m = api()
    profile = dict(cutoff=cutoff if supported else None, supported=supported, fallback=False)
    return dict(reference_version='test-reference', event_version=m.VERSION, budget=.03,
                bands={band: dict(profile) for band in ('<3', '3-5', '5-7', '7+')})


def test_same_height_central_upper_stream_has_exposure_and_fixed_endpoint():
    events = api().visibility_events([note(i, .2*i, y=2) for i in range(4)])
    assert len(events) == 1
    event = events[0]
    assert event['id'] == 3 and event['hand'] == 0
    assert event['time'] == pytest.approx(.6) and event['start_time'] == 0
    assert event['onsets'] == 4 and event['note_ids'] == [0, 1, 2, 3]
    assert event['group_ids'] == [3]
    assert event['weights'] == [.25] * 4 and event['severity'] == 1.


def test_clear_and_outer_upper_phrases_retain_zero_eligible_events():
    m = api()
    for x, y in ((1, 0), (1, 1), (0, 2), (3, 2)):
        events = m.visibility_events([note(i, .1*i, x=x, y=y) for i in range(6)])
        assert len(events) == 3 and all(e['severity'] == 0. for e in events)
    assert m.visibility_events([note(i, float(i), y=2) for i in range(6)]) == []


def test_rolling_last_eight_onsets_evict_an_old_upper_note():
    events = api().visibility_events([note(i, .1*i, y=2 if i == 0 else 1) for i in range(9)])
    assert len(events) == 6
    assert events[0]['severity'] == .25
    assert events[-2]['severity'] == .125
    assert events[-1]['severity'] == 0.
    assert events[-1]['note_ids'] == list(range(1, 9)) and events[-1]['onsets'] == 8


def test_exact_window_boundary_is_eligible_and_a_pause_clears_context():
    events = api().visibility_events([note(i, t, y=2) for i, t in enumerate((0., .5, 1., 1.5, 3.0001))])
    assert len(events) == 1 and events[0]['time'] == 1.5
    assert events[0]['severity'] == 1.
    rounded = api().visibility_events([note(i, t, y=2) for i, t in enumerate((0., .5, 1., 1.5+1e-12))])
    assert len(rounded) == 1


def test_same_hand_group_is_anchored_and_cross_hand_interleaving_cannot_split_it():
    notes = [note(0, 0., y=2), note(10, .01, hand=1), note(1, .02),
             note(2, .038), note(3, .2), note(4, .4)]
    events = api().visibility_events(notes)
    assert len(events) == 1
    event = events[0]
    assert event['onsets'] == 4 and event['id'] == 4
    assert event['note_ids'] == [0, 1, 2, 3, 4]
    assert event['weights'] == [.125, .125, .25, .25, .25]
    assert event['severity'] == .125


def test_stacks_share_onset_weight_and_do_not_inflate_exposure():
    m = api()
    notes = [n for i in range(4) for n in (note(2*i, .2*i, y=2), note(2*i+1, .2*i, y=0))]
    event = m.visibility_events(notes)[0]
    assert event['onsets'] == 4 and event['severity'] == .5
    assert event['group_ids'] == [6, 7]
    assert event['weights'] == [.125] * 8 and sum(event['weights']) == 1.
    assert m.visibility_events(list(reversed(notes))) == [event]


def test_arrow_edits_and_input_order_leave_event_identity_weights_and_denominator_fixed():
    m = api()
    notes = [note(i, .2*i, hand=i % 2, y=2 if i % 3 else 0, d=8) for i in range(10)]
    original = json.loads(json.dumps(notes))
    events = m.visibility_events(notes)
    assert m.visibility_events(list(reversed(notes))) == events
    assert notes == original
    assert m.visibility_events([dict(n, d=3) for n in notes]) == events
    moved = m.visibility_events([dict(n, y=0) for n in notes])
    assert len(moved) == len(events)
    for before, after in zip(events, moved):
        assert {k: v for k, v in before.items() if k != 'severity'} == {k: v for k, v in after.items() if k != 'severity'}
        assert after['severity'] == 0.


def test_costs_use_strict_reference_tail_and_expose_flagged_event_details():
    m = api()
    notes = [note(i, .2*i, y=2 if i < 2 else 1) for i in range(4)]
    tied = m.visibility_costs(notes, nps=6., reference=reference(.5))
    assert tied['eligible_events'] == 1 and tied['flagged_events'] == 0
    assert tied['excess'] == 0. and tied['max_severity'] == .5
    assert tied['nps'] == 6. and tied['nps_band'] == '5-7'
    assert tied['supported'] and not tied['fallback'] and tied['budget'] == .03
    notes[-1]['y'] = 2
    flagged = m.visibility_costs(notes, nps=6., reference=reference(.5))
    assert flagged['flagged_events'] == 1 and flagged['excess'] == .25
    assert flagged['events'][0]['flagged'] and flagged['events'][0]['excess'] == .25
    assert flagged['version'] == m.VERSION and flagged['reference_version'] == 'test-reference'
    json.dumps(flagged, allow_nan=False)


def test_unsupported_reference_reports_measurements_without_fabricated_flags():
    result = api().visibility_costs([note(i, .2*i, y=2) for i in range(4)], reference=reference(supported=False))
    assert result['eligible_events'] == 1 and result['max_severity'] == 1.
    assert result['cutoff'] is None and result['flagged_events'] is None and result['excess'] is None
    assert result['events'][0]['flagged'] is None and result['events'][0]['excess'] is None


def test_costs_measure_nps_from_anchored_groups_and_handle_empty_maps():
    m = api()
    notes = [note(0, 0.), note(1, .01), note(2, .2), note(3, .4), note(4, .6)]
    result = m.visibility_costs(notes, reference=reference())
    assert result['nps'] == pytest.approx(4/.6) and result['nps_band'] == '5-7'
    empty = m.visibility_costs([], reference=reference())
    assert empty['nps'] == 0. and empty['eligible_events'] == 0 and empty['flagged_events'] == 0
    assert empty['excess'] == 0. and empty['max_severity'] == 0.


def test_reference_definition_mismatch_is_rejected():
    ref = reference()
    ref['event_version'] = 'different-events'
    with pytest.raises(ValueError, match='version'):
        api().visibility_costs([], reference=ref)


def test_runtime_visibility_import_has_no_tensorflow_or_evaluation_dependency():
    api()
    run = subprocess.run([sys.executable, '-s', '-c',
        'import sys; from map_creation.v2_playability_visibility import visibility_events; '
        'assert visibility_events([]) == []; '
        'assert not any(n == "tensorflow" or n.startswith("evaluation.") for n in sys.modules)'],
        text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
