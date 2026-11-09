"""The final v2 policy is the ordinary worker default, with explicit legacy arms."""
from map_creation.v2_generation import settings_snapshot
from tools.config import config


def test_default_worker_snapshot_selects_final_priority_and_late_flow():
    snapshot = settings_snapshot()['config']
    assert snapshot['v2_playability_enabled'] is True
    assert snapshot['v2_playability_priority'] is True
    assert snapshot['v2_playability_reach'] is True
    assert snapshot['v2_playability_late_turns'] is True
    assert snapshot['v2_playability_slow90_seconds'] == .5
    assert snapshot['v2_playability_slow45_seconds'] == .65
    assert snapshot['v2_playability_m1'] is False
    assert snapshot['v2_playability_m3'] is False
    assert snapshot['v2_playability_dwell'] is False


def test_default_policy_matches_the_final_generation_profile():
    from evaluation.playability_experiments import experiment_policy, _policy_settings
    ordinary = _policy_settings()
    with experiment_policy('priority'):
        assert _policy_settings() == ordinary


def test_app1_and_combined_are_explicit_historical_overrides_of_final_default():
    from evaluation.playability_experiments import experiment_policy
    before = {k: v for k, v in settings_snapshot()['config'].items()
              if k.startswith('v2_playability_')}
    with experiment_policy('app1_control'):
        assert config.v2_playability_enabled is False
        assert config.v2_playability_priority is False
        assert config.v2_playability_late_turns is False
    with experiment_policy('combined'):
        assert config.v2_playability_enabled is True
        assert config.v2_playability_priority is False
        assert config.v2_playability_m3 is True
    assert {k: v for k, v in settings_snapshot()['config'].items()
            if k.startswith('v2_playability_')} == before
