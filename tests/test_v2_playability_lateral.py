"""Repeated fast M1 horizontal stress has a human event-tail contract."""

from copy import deepcopy
import importlib
import importlib.util
import json

import pytest


def lateral():
    name = "map_creation.v2_playability_lateral"
    assert importlib.util.find_spec(name) is not None, "repeated M1 stress component is missing"
    return importlib.import_module(name)


def notes(directions, gap=.221, hand=0):
    return [dict(id=i,time=1.+gap*i,hand=hand,x=1,y=1,d=d) for i,d in enumerate(directions)]


def test_m1_run_denominator_keeps_zero_vertical_and_diagonal_arrivals():
    report = lateral().consecutive_horizontal_load(notes([0,1,2,3,2,4,5,0]),4.)
    assert report['eligible_events'] == 7
    assert report['severities'] == [0,0,1,2,0,0,0]
    assert report['event_note_ids'] == [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]


def test_ordinary_horizontal_incidents_are_allowed_but_twelve_note_run_is_flagged():
    policy = lateral()
    for nps in (2.,4.,6.,8.):
        ordinary = policy.consecutive_horizontal_load(notes([2,3,2]),nps)
        assert ordinary['total_excess'] == 0.
        repeated = policy.consecutive_horizontal_load(notes([2,3]*6),nps)
        assert repeated['max_reversals'] == 11
        assert repeated['flagged_events'] > 0
        assert repeated['total_excess'] > 0.


def test_slow_arrival_same_arrow_or_dot_breaks_horizontal_run():
    policy = lateral()
    report = policy.consecutive_horizontal_load(notes([2,3,3,2,8,3,2]),4.)
    assert report['severities'] == [1,0,1,0,1]
    assert policy.consecutive_horizontal_load(notes([2,3]*6,gap=.251),4.)['max_reversals'] == 0
    assert policy.consecutive_horizontal_load(notes([2,3,2],gap=.25),4.)['severities'] == [1,2]


def test_stacks_collapse_at_anchored_twenty_milliseconds_without_chaining():
    policy = lateral()
    seq = notes([2,3,3,2],gap=.1)
    for note,time in zip(seq,(1.,1.015,1.03,1.28)):
        note['time'] = time
    report = policy.consecutive_horizontal_load(seq,4.)
    assert report['severities'] == [1,2]
    assert report['event_note_ids'] == [[0,2],[2,3]]


def test_hand_mirroring_input_order_and_annotations_are_preserved():
    policy = lateral()
    seq = notes([2,3]*6)
    seq[0]['custom'] = {'keep':True}
    original = deepcopy(seq)
    mirrored = [{**n,'hand':1,'d':5-n['d']} for n in seq]
    assert policy.consecutive_horizontal_load(seq,4.) == policy.consecutive_horizontal_load(mirrored,4.)
    assert policy.consecutive_horizontal_load(list(reversed(seq)),4.) == policy.consecutive_horizontal_load(seq,4.)
    assert seq == original
    json.dumps(policy.consecutive_horizontal_load(seq),allow_nan=False)
    assert policy.consecutive_horizontal_load([],4.)['total_excess'] == 0.


def test_missing_note_ids_use_chronological_ordinals_for_each_transition():
    seq = notes([2,3,2,3])
    for note in seq:
        del note['id']
    assert lateral().consecutive_horizontal_load(seq,4.)['event_note_ids'] == [[0,1],[1,2],[2,3]]

