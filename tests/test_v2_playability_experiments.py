import numpy as np


def test_recovery_swaps_only_when_both_gap_and_model_support_allow():
    from map_creation.v2_pipeline import _recover_single_hand

    red = np.zeros(13)
    blue = np.zeros(13)
    red[[0, 1]] = [.5, .5]
    blue[[0, 3]] = [.52, .48]
    outputs = {"red": red[None], "blue": blue[None]}
    assert _recover_single_hand((1, 0), outputs, (1., .5), 1.1, .75) == (0, 3)
    assert _recover_single_hand((1, 0), outputs, (1., 1.), 1.1, .75) == (1, 0)
    assert _recover_single_hand((1, 0), outputs, (1., .5), 1.1, .95) == (1, 0)
    assert _recover_single_hand((1, 3), outputs, (1., .5), 1.1, .75) == (1, 3)
    assert _recover_single_hand((1, 0), outputs, (1., .5), 1.25, .75) == (1, 0)


def test_seed_is_stable_across_order_and_labels():
    from evaluation.playability_experiments import case_seed

    assert case_seed('abc', 5.0, 1) == case_seed('abc', 5, 1)
    assert case_seed('abc', 5, 1) != case_seed('abc', 5, 2)
    assert case_seed('abc', 5, 1) != case_seed('abc', 6, 1)


def test_scoped_policies_and_legacy_arms(monkeypatch):
    from evaluation.evaluate_v2_postprocessing import _arm_processors, ARMS
    from evaluation.playability_experiments import experiment_policy
    from tools.config import config

    monkeypatch.setattr(config, 'v2_playability_m2', True)
    with experiment_policy('app1_control'):
        assert config.v2_playability_enabled is False
        assert not config.v2_playability_m2
        assert _arm_processors('app1_control', ARMS['app1_control']) is None
    with experiment_policy('m1,slow90_350'):
        assert config.v2_playability_enabled and config.v2_playability_m1
        assert not config.v2_playability_m2
        assert config.v2_playability_slow90_seconds == .35
        assert config.v2_playability_slow45_seconds == 0
    assert config.v2_playability_m2
    assert _arm_processors('full', ARMS['full']) is not None


def test_resume_rejects_changed_package_and_settings(tmp_path):
    import pytest
    from evaluation.playability_experiments import _completed_case, _package_hashes

    package = tmp_path / 'package'
    package.mkdir()
    chart = package / 'Easy.dat'
    chart.write_bytes(b'chart bytes')
    expected = dict(song='song', audio_hash='audio', seed=11, arm='m1', difficulty='Easy',
                    replicate=11, actual_settings={'nps': 5.75})
    previous = {**expected, 'package': 'package', 'package_hashes': _package_hashes(package)}
    _completed_case(tmp_path, previous, expected)
    with pytest.raises(ValueError, match='settings'):
        _completed_case(tmp_path, previous, {**expected, 'actual_settings': {'nps': 4}})
    chart.write_bytes(b'changed')
    with pytest.raises(ValueError, match='package'):
        _completed_case(tmp_path, previous, expected)


def _balanced_recovery_fixture():
    red = np.zeros(13)
    blue = np.zeros(13)
    red[[0, 5]] = [.5, .5]
    blue[[0, 8]] = [.52, .48]
    notes = [dict(time=t, hand=0, x=0, y=1) for t in (.5, .7, .85, 1.)]
    notes.extend(dict(time=t, hand=1, x=3, y=1) for t in (.4, .6))
    return {'red': red[None], 'blue': blue[None]}, sorted(notes, key=lambda n:n['time'])


def test_balanced_recovery_reduces_local_hand_excess_at_original_single_onset():
    from map_creation.v2_pipeline import _recover_single_hand
    outputs, notes = _balanced_recovery_fixture()
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes) == (0, 8)
    assert _recover_single_hand((5, 8), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes) == (5, 8)


def test_balanced_recovery_keeps_balanced_phrase_and_nearby_double_unchanged():
    from map_creation.v2_pipeline import _recover_single_hand
    outputs, notes = _balanced_recovery_fixture()
    balanced = notes + [dict(time=t, hand=1, x=3, y=1) for t in (.3, .2)]
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=balanced) == (5, 0)
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes, following_event={'time':1.2, 'count':2}) == (5, 0)
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes, last_recovery_time=.8) == (5, 0)


def test_balanced_recovery_uses_supported_nearby_own_side_cell():
    from map_creation.v2_pipeline import _recover_single_hand
    outputs, notes = _balanced_recovery_fixture()
    outputs['blue'][0] = 0.
    outputs['blue'][0, [0, 5, 8]] = [.1, .5, .4]
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes) == (0, 8)
    outputs['blue'][0, [0, 5, 8]] = [.5, .48, .02]
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes) == (5, 0)
    for n in notes:
        if n['hand'] == 1:
            n.update(x=2, y=0)
    outputs['blue'][0] = 0.
    outputs['blue'][0, [0, 12]] = [.52, .48]
    assert _recover_single_hand((5, 0), outputs, (1., .6), 1.1, .9,
                                recent_notes=notes) == (5, 0)


def test_combined_uses_bounded_recovery_while_archived_hand_arms_keep_legacy_scope():
    from evaluation.playability_experiments import experiment_policy
    from tools.config import config
    with experiment_policy('combined'):
        assert config.v2_playability_recovery_balanced
        assert config.v2_playability_recovery_ratio == .9
    with experiment_policy('hand_75'):
        assert not config.v2_playability_recovery_balanced
        assert config.v2_playability_recovery_ratio == .75
    with experiment_policy('app1_control'):
        assert not config.v2_playability_recovery_balanced
        assert config.v2_playability_recovery_ratio == 0.
