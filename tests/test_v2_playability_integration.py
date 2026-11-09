"""Combined candidate enters through ordinary generation and validates the final stream."""
from copy import deepcopy

import pytest

from tools.config import config


@pytest.fixture(autouse=True)
def legacy_profile(monkeypatch):
    """Exercise historical arms independently of the promoted production policy."""
    from evaluation.playability_experiments import SWITCHES
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    for field, value in SWITCHES.items():
        monkeypatch.setattr(config, 'v2_playability_' + field, value)


def state_with(notes):
    return {**{family: [] for family in ('notes', 'bombs', 'walls', 'chains', 'arcs', 'lighting')}, 'notes': notes}


def test_combined_policy_scopes_all_requested_repairs(monkeypatch):
    from evaluation.playability_experiments import experiment_policy

    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    with experiment_policy('combined'):
        assert config.v2_playability_enabled
        assert config.v2_playability_reach
        assert all(getattr(config, 'v2_playability_'+name) for name in ('m1','m2','m3','m4','dwell','late_turns'))
        assert config.v2_playability_isolated
        assert config.v2_playability_slow90_seconds == .5
        assert config.v2_playability_slow45_seconds == .65
    assert config.v2_playability_enabled is False


def test_combined_final_validation_rejects_forbidden_cut(monkeypatch):
    from map_creation.v2_pipeline import _validate_state

    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True, raising=False)
    note = dict(id=1, time=2., hand=0, x=3, y=0, d=2, kind='retained')
    with pytest.raises(ValueError, match='forbidden|reach'):
        _validate_state(state_with([note]), 10.)


def test_legacy_final_validation_keeps_existing_contract(monkeypatch):
    from map_creation.v2_pipeline import _validate_state

    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    monkeypatch.setattr(config, 'v2_playability_reach', True, raising=False)
    note = dict(id=1, time=2., hand=0, x=3, y=0, d=2, kind='retained')
    _validate_state(state_with([note]), 10.)


def test_final_reach_repairs_mutable_copy_and_preserves_retained(monkeypatch):
    from map_creation.v2_postprocessing import finalize_playability_reach

    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True, raising=False)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)
    retained = dict(id=0, time=2., hand=0, x=2, y=1, d=1, kind='retained')
    notes = [retained, dict(id=1, time=3., hand=0, x=3, y=1, d=4, kind='copy', anchor_id=0)]
    before = deepcopy(retained)
    report = finalize_playability_reach(notes, protect_retained=True)
    assert retained == before
    assert [(n['id'],n['time'],n['hand']) for n in notes] == [(0,2.,0),(1,3.,0)]
    assert report['reach_hard_after'] == 0


def test_combined_diagonal_pass_preserves_original_survivor_selection(monkeypatch):
    from map_creation.v2_postprocessing import apply_diagonal_cap

    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True, raising=False)
    monkeypatch.setattr(config, 'v2_diagonal_max_fraction', .1)
    notes = [dict(id=i,time=2.+i*.3,hand=0,x=1,y=1,
                  d=(4 if i%2==0 else 7) if i<8 else (0 if i%2==0 else 1),kind='retained')
             for i in range(20)]
    import numpy as np
    control = deepcopy(notes)
    stats = apply_diagonal_cap(notes, rng=np.random.default_rng(3))
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    control_stats = apply_diagonal_cap(control, rng=np.random.default_rng(3))
    assert stats == control_stats
    assert notes == control


def test_combined_refines_geometry_after_joint_arrow_and_reach_changes(monkeypatch):
    from map_creation import v2_postprocessing as post
    from map_creation import v2_playability_geometry as geometry
    from map_creation import v2_playability_reach as reach

    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True, raising=False)
    for name in ('m1','m2','m4','dwell','late_turns','isolated'):
        monkeypatch.setattr(config, 'v2_playability_'+name, False)
    monkeypatch.setattr(config, 'v2_playability_m3', True)
    calls = []
    def geometric(notes, cap, **switches):
        calls.append('geometry')
        return {'geometry_changed': 0}
    def joint(notes, cap, **options):
        calls.append('reach')
        return {'reach_hard_after': 0}
    monkeypatch.setattr(geometry, 'repair_geometry', geometric)
    monkeypatch.setattr(reach, 'repair_reach', joint)
    notes = [dict(id=0,time=1.,hand=0,x=1,y=1,d=0,kind='retained')]
    stats = {}
    post._apply_playability_experiments(notes, lambda note: float('inf'), stats)
    assert calls == ['reach','geometry','reach','geometry']
    assert stats['settled_geometry_changed'] == 0


def test_geometry_respects_shared_optional_guard(monkeypatch):
    from map_creation.v2_playability_geometry import repair_geometry
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    notes = [dict(id=0,time=1.,hand=0,x=2,y=0,d=0),
             dict(id=1,time=1.018,hand=1,x=1,y=0,d=0)]
    original = deepcopy(notes)
    class Reject:
        def allows(self, changed_ids):
            return False
        def commit(self, changed_ids):
            raise AssertionError('vetoed proposal committed')
    stats = repair_geometry(notes, lambda _: float('inf'), interference=True, guard=Reject())
    assert notes == original
    assert stats['geometry_changed'] == 0


def test_full_geometry_rollback_restores_shared_accepted_snapshot(monkeypatch):
    from map_creation import v2_postprocessing as post
    from map_creation.v2_playability_geometry import repair_geometry
    from map_creation.v2_playability_policy import PatternGuard
    from map_creation import v2_playability_costs as costs
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    notes = [dict(id=0,time=1.,hand=0,x=2,y=0,d=0),
             dict(id=1,time=1.018,hand=1,x=1,y=0,d=0)]
    original = deepcopy(notes)
    calibrated = costs.calibrated_limits
    monkeypatch.setattr(costs, 'calibrated_limits', lambda nps: {**calibrated(nps), 'm4': 0.})
    policy = PatternGuard(notes, nps=4.)
    calls = []
    def forced_residual(notes):
        calls.append(1)
        return {'same_cell_pairs': int(len(calls) > 1), 'swing_into_pairs': 0, 'flow_repeats': 0}
    monkeypatch.setattr(post, 'clearance_violations', forced_residual)
    result = repair_geometry(notes, lambda _: float('inf'), interference=True, guard=policy)
    assert result['geometry_rolled_back']
    assert notes == original
    assert policy.accepted == notes


def test_deferred_decoder_selection_commits_before_postprocessing_hand_snapshot(monkeypatch):
    import sys
    from types import SimpleNamespace
    from map_creation import v2_postprocessing as post
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    monkeypatch.setattr(config, 'v2_playability_recovery_balanced', True)
    monkeypatch.setattr(config, 'v2_speed_cap_scale', 0.)
    calls = []
    def recover(notes, cap, *, guard):
        assert notes[0]['hand'] == 0
        assert guard is not None
        calls.append('recovery')
        notes[0]['hand'] = 1
        return {'recovery_hand_swaps': 1}
    monkeypatch.setitem(sys.modules, 'map_creation.v2_playability_recovery', SimpleNamespace(repair_recovery=recover))
    def metrics(notes, cap, stats, isolated_ids=()):
        assert notes[0]['hand'] == 1
        calls.append('metrics')
    monkeypatch.setattr(post, '_apply_playability_experiments', metrics)
    notes = [dict(id=0,time=2.,hand=0,x=1,y=1,d=0)]
    result = post.apply_speed_cap(notes)
    assert calls == ['recovery', 'metrics']
    assert result['recovery_hand_swaps'] == 1
