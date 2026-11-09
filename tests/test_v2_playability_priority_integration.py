"""The priority arm bypasses historical objectives and audits final notes."""
from copy import deepcopy
import math

import pytest

from tools.config import config


def state(notes):
    return dict(notes=notes, bombs=[], walls=[], chains=[], arcs=[], lighting=[])


def enable(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_priority', True, raising=False)


def test_priority_arm_scopes_the_new_policy_and_restores_settings():
    from evaluation.playability_experiments import experiment_policy
    before = getattr(config, 'v2_playability_priority', False)
    with experiment_policy('priority'):
        assert config.v2_playability_enabled and config.v2_playability_priority
        assert config.v2_playability_reach
        assert not any(getattr(config, 'v2_playability_'+key) for key in
                       ('m1', 'm3', 'dwell', 'isolated', 'recovery_balanced'))
        assert config.v2_playability_recovery_ratio == 0
    assert config.v2_playability_priority == before


def test_priority_final_validation_rejects_intersecting_paths(monkeypatch):
    from map_creation.v2_pipeline import _validate_state
    enable(monkeypatch)
    monkeypatch.setattr(config, 'v2_playability_reach', False)
    notes = [dict(id=0, time=2., hand=0, x=1, y=1, d=3),
             dict(id=1, time=2., hand=1, x=2, y=1, d=2)]
    with pytest.raises(ValueError, match='priority|crossing|interference'):
        _validate_state(state(notes), 10.)


def test_priority_final_validation_rejects_hard_preparation(monkeypatch):
    from map_creation.v2_pipeline import _validate_state
    enable(monkeypatch)
    monkeypatch.setattr(config, 'v2_playability_reach', False)
    with pytest.raises(ValueError, match='priority|preparation|reach'):
        _validate_state(state([dict(id=0, time=2., hand=0, x=3, y=1, d=2)]), 10.)


def test_disabled_priority_flag_preserves_legacy_validation(monkeypatch):
    from map_creation.v2_pipeline import _validate_state
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    monkeypatch.setattr(config, 'v2_playability_priority', True, raising=False)
    _validate_state(state([dict(id=0, time=2., hand=0, x=3, y=1, d=2)]), 10.)


def test_priority_primary_dispatch_bypasses_historical_passes(monkeypatch):
    from map_creation import v2_postprocessing as post
    from map_creation import v2_playability_priority as priority
    enable(monkeypatch)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    monkeypatch.setattr(config, 'v2_playability_m1', True)
    calls = []
    def repair(notes, cap, *, protect_retained=False):
        calls.append(protect_retained)
        return dict(priority_preserved=True, priority_hard_after=0)
    def historical(*args, **kwargs):
        raise AssertionError('historical optimization ran in the priority policy')
    monkeypatch.setattr(priority, 'repair_priority', repair)
    monkeypatch.setattr(post, 'repair_playability_directions', historical)
    notes = [dict(id=0, time=2., hand=0, x=1, y=1, d=0)]
    before = deepcopy(notes)
    stats = {}
    post._apply_playability_experiments(notes, lambda n: math.inf, stats)
    assert calls == [False] and notes == before
    assert stats['priority_preserved'] and stats['playability_preserved']


def test_priority_finalizer_uses_frozen_mode_and_syncs_chain_anchor(monkeypatch):
    from map_creation import v2_postprocessing as post
    from map_creation import v2_playability_priority as priority
    enable(monkeypatch)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    calls = []
    def repair(notes, cap, *, protect_retained=False):
        calls.append(protect_retained)
        notes[0].update(x=1, y=1, d=0)
        return dict(priority_hard_after=0)
    monkeypatch.setattr(priority, 'repair_priority', repair)
    notes = [dict(id=0, time=2., hand=0, x=0, y=2, d=4, kind='copy')]
    chains = [dict(anchor_id=0, x=0, y=2, d=4)]
    post.finalize_playability_reach(notes, chains, cap=lambda n: math.inf)
    assert calls == [True]
    assert (chains[0]['x'], chains[0]['y'], chains[0]['d']) == (1, 1, 0)
