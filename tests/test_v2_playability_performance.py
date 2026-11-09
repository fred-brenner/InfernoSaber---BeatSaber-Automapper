"""Performance memoization contracts: cached backends keep exact numerics."""

import math

import pytest

from tools.config import config
from map_creation import v2_playability_policy as policy
from map_creation import v2_postprocessing as post
from map_creation import v2_playability_priority_metrics as metrics
from map_creation.v2_playability_priority import _PrioritySearch


def _notes(*rows):
    return [{"id": i, "time": t, "hand": h, "x": x, "y": y, "d": d}
            for i, (t, h, x, y, d) in enumerate(rows)]


def _hand_stream(directions=(0, 1, 0)):
    return _notes(*[(t, 0, 1, 1, d) for t, d in zip((0., 1., 2.), directions)])


@pytest.fixture(autouse=True)
def perf_defaults(monkeypatch):
    for key, value in {"max_speed": 20., "reaction_time": 1.15,
                       "reaction_time_fact": .013, "cdf": 1.2, "cdf_lr": 1.15}.items():
        monkeypatch.setattr(config, key, value)
    for module, name in ((policy, "_cached_speed"),
                         (metrics, "_cached_segment_distance")):
        cache = getattr(module, name, None)
        if cache is not None and callable(getattr(cache, "cache_clear", None)):
            cache.cache_clear()
    yield
    for module, name in ((policy, "_cached_speed"),
                         (metrics, "_cached_segment_distance")):
        cache = getattr(module, name, None)
        if cache is not None and callable(getattr(cache, "cache_clear", None)):
            cache.cache_clear()


def test_speed_groups_reuse_cached_backend(monkeypatch):
    calls = {"count": 0}
    real = post.calc_note_speed

    def spy(last, new, gap):
        calls["count"] += 1
        return real(last, new, gap)

    monkeypatch.setattr(post, "calc_note_speed", spy)
    notes = _hand_stream()
    groups = policy.SpeedGroups(notes, lambda n: 3., .02)
    assert groups.violations(range(3)) == set()
    first = calls["count"]
    groups = policy.SpeedGroups(notes, lambda n: 3., .02)
    assert groups.violations(range(3)) == set()
    assert calls["count"] == first


def test_speed_groups_live_note_edits_recompute():
    notes = _hand_stream()
    groups = policy.SpeedGroups(notes, lambda n: 3., .02)
    assert groups.violations(range(3)) == set()
    notes[1]["x"] = 3
    assert groups.violations(range(3)) == {1, 2}
    notes[1]["x"] = 1
    assert groups.violations(range(3)) == set()
    notes[1]["d"] = 0
    assert groups.violations(range(3)) == {1, 2}
    notes[1]["d"] = 1
    assert groups.violations(range(3)) == set()
    notes[1]["time"] = .2
    assert groups.violations(range(3)) == {1}
    notes[1]["time"] = 1.
    assert groups.violations(range(3)) == set()


def test_speed_groups_config_edit_recomputes(monkeypatch):
    notes = _hand_stream()
    groups = policy.SpeedGroups(notes, lambda n: 3., .02)
    assert groups.violations(range(3)) == set()
    monkeypatch.setattr(config, "reaction_time", 3.)
    assert groups.violations(range(3)) == {1, 2}
    monkeypatch.setattr(config, "reaction_time", 1.15)
    assert groups.violations(range(3)) == set()


def test_speed_groups_backend_identity_invalidates(monkeypatch):
    counts = {"a": 0, "b": 0}
    real = post.calc_note_speed

    def spy_a(last, new, gap):
        counts["a"] += 1
        return real(last, new, gap)

    def spy_b(last, new, gap):
        counts["b"] += 1
        return real(last, new, gap)

    notes = _hand_stream()
    groups = policy.SpeedGroups(notes, lambda n: 3., .02)
    monkeypatch.setattr(post, "calc_note_speed", spy_a)
    assert groups.violations(range(3)) == set()
    assert counts["a"] == 2
    monkeypatch.setattr(post, "calc_note_speed", spy_b)
    assert groups.violations(range(3)) == set()
    assert counts["b"] == 2
    assert groups.violations(range(3)) == set()
    assert counts["b"] == 2


def test_calc_note_speed_literal_values(monkeypatch):
    assert post.calc_note_speed([1, 1, 0, 0], [1, 1, 0, 1], 1.) == pytest.approx(1.6215, rel=1e-4)
    assert post.calc_note_speed([1, 1, 0, 0], [2, 1, 0, 1], 1.) == pytest.approx(2.7715, rel=1e-4)
    assert post.calc_note_speed([1, 1, 0, 0], [1, 1, 0, 0], 1.) == pytest.approx(4.0215, rel=1e-4)
    assert post.calc_note_speed([1, 1, 0, 0], [1, 1, 0, 1], .5) == pytest.approx(3.243, rel=1e-4)
    monkeypatch.setattr(config, "max_speed", 40.)
    assert post.calc_note_speed([1, 1, 0, 0], [1, 1, 0, 1], 1.) == pytest.approx(1.9205, rel=1e-4)


def test_segment_distance_memoizes_and_invalidates(monkeypatch):
    calls = {"count": 0}
    real = metrics._point_segment_distance

    def spy(p, a, b):
        calls["count"] += 1
        return real(p, a, b)

    monkeypatch.setattr(metrics, "_point_segment_distance", spy)
    first, second = [[0., 0.], [1., 0.]], [[0., 2.], [1., 2.]]
    assert metrics._segment_distance(first, second) == 2.
    cached_calls = calls["count"]
    assert metrics._segment_distance(first, second) == 2.
    assert calls["count"] == cached_calls
    first[0][1] = 1.
    assert metrics._segment_distance(first, second) == 1.
    assert calls["count"] > cached_calls


def test_segment_distance_tolerance_change_invalidates(monkeypatch):
    first, second = [[0., 0.], [1., 0.]], [[0., 5e-7], [1., 5e-7]]
    monkeypatch.setattr(metrics, "GEOMETRY_TOLERANCE", 1e-9)
    assert metrics._segment_distance(first, second) == pytest.approx(5e-7)
    monkeypatch.setattr(metrics, "GEOMETRY_TOLERANCE", 1e-6)
    assert metrics._segment_distance(first, second) == 0.


@pytest.mark.parametrize("first,second,distance", [
    ([[0., 0.], [1., 0.]], [[2., 0.], [3., 0.]], 1.),
    ([[0., 0.], [1., 1.]], [[0., 1.], [1., 0.]], 0.),
    ([[0., 0.], [0., 0.]], [[3., 4.], [3., 4.]], 5.),
    ([[0., 0.], [2., 0.]], [[.5, -1.], [.5, 1.]], 0.),
    ([[0., 0.], [0., 0.]], [[1., 2.], [1., 2.]], math.sqrt(5.)),
    ([[1., 0.], [0., 0.]], [[1., 2.], [0., 2.]], 2.),
])
def test_segment_distance_literal_geometry(first, second, distance):
    assert metrics._segment_distance(first, second) == pytest.approx(distance)
    assert metrics._segment_distance(first, second) == pytest.approx(distance)


def test_priority_metrics_cold_and_warm_identical():
    notes = _notes((0., 0, 0, 1, 0), (.3, 1, 3, 1, 0),
                   (.6, 0, 0, 1, 1), (.9, 1, 3, 1, 1),
                   (1.2, 0, 1, 1, 4), (1.4, 1, 2, 0, 5))
    cold = metrics.priority_metrics(notes)
    warm = metrics.priority_metrics(notes)
    assert cold == warm


def test_soft_score_default_and_skipped_primary(monkeypatch):
    notes = _notes((0., 0, 0, 1, 0), (.3, 1, 3, 1, 0),
                   (.6, 0, 0, 1, 1), (.9, 1, 3, 1, 1))
    search = _PrioritySearch(notes, lambda n: float("inf"), False)
    fragment = list(notes)
    pair_keys = set(search.pairs)
    assert search._soft_score(fragment, pair_keys) == (
        6.009769039345553, 0., 0., 0., 0.)
    calls = {"prep": 0, "axis": 0}
    real_prep = metrics.preparation_events
    real_axis = metrics.axis_change_events

    def spy_prep(fragment):
        calls["prep"] += 1
        return real_prep(fragment)

    def spy_axis(fragment, *args):
        calls["axis"] += 1
        return real_axis(fragment, *args)

    monkeypatch.setattr(metrics, "preparation_events", spy_prep)
    monkeypatch.setattr(metrics, "axis_change_events", spy_axis)
    skipped = search._soft_score(fragment, pair_keys, include_primary=False)
    assert skipped == (0., 0., 0., 0., 0.)
    assert calls == {"prep": 0, "axis": 0}


def test_reach_legal_rejects_flow_before_speed(monkeypatch):
    from map_creation.v2_playability_reach import _Search
    notes = _notes((0., 0, 1, 1, 0), (.3, 0, 1, 1, 1), (.6, 0, 1, 1, 0))
    state = _Search(notes, lambda n: float("inf"), 4., False)
    calls = {"count": 0}
    real_legal = state.speed_groups.legal

    def spy(changed):
        calls["count"] += 1
        return real_legal(changed)

    monkeypatch.setattr(state.speed_groups, "legal", spy)
    state.install({1: (1, 1, 0)})
    assert state.legal({1}) is False
    assert calls["count"] == 0


def test_reach_legal_runs_speed_for_flow_passing_move(monkeypatch):
    from map_creation.v2_playability_reach import _Search
    notes = _notes((0., 0, 1, 1, 0), (.3, 0, 1, 1, 1), (.6, 0, 1, 1, 0))
    state = _Search(notes, lambda n: 1., 4., False)
    calls = {"count": 0}
    real_legal = state.speed_groups.legal

    def spy(changed):
        calls["count"] += 1
        return real_legal(changed)

    monkeypatch.setattr(state.speed_groups, "legal", spy)
    state.install({1: (3, 1, 1)})
    assert state.legal({1}) is False
    assert calls["count"] == 1
