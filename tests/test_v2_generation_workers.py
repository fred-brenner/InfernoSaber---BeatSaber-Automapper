"""Entry-point and spawn contracts without loading or training any networks."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import main_multi
from map_creation import v2_generation
from tools.config import config, paths


def _observed_worker_settings(_):
    from tools.config import config, paths
    import os, sys
    return (config.v2_inference_threads, config.slider_time_gap.tolist(), paths.songs_pred,
            config.v1_lighting_model_path, config.v1_lighting_vocabulary_path,
            (config.v2_intensity_low_boost, config.v2_intensity_high_reduction,
             config.v2_intensity_scale, config.v2_intensity_adjustment_strength),
            os.environ['CUDA_VISIBLE_DEVICES'], 'tensorflow' in sys.modules)


def test_spawn_transfers_settings_before_tensorflow():
    snapshot = v2_generation.settings_snapshot()
    snapshot['config']['v2_inference_threads'] = 1
    snapshot['config']['slider_time_gap'] = np.array([.17, .39])
    snapshot['paths']['songs_pred'] = '/disposable/v2-input/'
    snapshot['config']['v1_lighting_model_path'] = '/retained/v1.h5'
    snapshot['config']['v1_lighting_vocabulary_path'] = '/retained/v1-events.pkl'
    snapshot['config'].update(v2_intensity_low_boost=.2, v2_intensity_high_reduction=.3,
                              v2_intensity_scale=1.25, v2_intensity_adjustment_strength=.6)
    context = mp.get_context('spawn')
    with context.Pool(1, initializer=main_multi._worker_init, initargs=(snapshot,)) as pool:
        result = pool.map(_observed_worker_settings, [0, 1])
    assert result == [(1, [.17, .39], '/disposable/v2-input/', '/retained/v1.h5',
                       '/retained/v1-events.pkl', (.2, .3, 1.25, .6), '-1', False)] * 2


def test_groups_prefer_song_cache_but_use_idle_workers():
    groups = main_multi._groups(['a.egg', 'b.egg'], [12., 20.], 2)
    assert groups == [[('a.egg', 12.), ('a.egg', 20.)], [('b.egg', 12.), ('b.egg', 20.)]]
    split = main_multi._groups(['a.egg'], [4., 12., 20.], 2)
    assert len(split) == 2
    assert [item for group in split for item in group] == [('a.egg', 4.), ('a.egg', 12.), ('a.egg', 20.)]


@pytest.mark.parametrize('values', [[], [1] * 6, [3, 3], [float('nan')], [-1], [True], [3, 3.001]])
def test_invalid_difficulties(values):
    with pytest.raises(ValueError):
        main_multi._difficulties(values)


def test_difficulty_is_requested_nps():
    assert main_multi._difficulties([5, 3]) == [12., 20.]


def test_worker_failure_is_explicit_and_does_not_abort_other_variants(monkeypatch):
    def generate(song, difficulty, logger):
        if difficulty == 12:
            raise ValueError('incompatible model')
        return {'status': 'ok', 'song': song, 'internal_difficulty': difficulty}
    monkeypatch.setattr(main_multi, 'generate_song', generate)
    results = main_multi.process_song([('one.egg', 12), ('two.egg', 20)], 2)
    assert results[0]['status'] == 'failed'
    assert 'incompatible model' in results[0]['error']
    assert results[1]['status'] == 'ok'


def test_combination_requires_this_runs_successes_not_old_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'new_map_path', str(tmp_path) + '/')
    monkeypatch.setattr(config, 'auto_move_song_afterwards', False)
    monkeypatch.setattr(config, 'general_diff', 'ExpertPlus')
    old = tmp_path / '1234_20.0_failed'
    old.mkdir()
    (old / 'ExpertPlus.dat').write_text('stale')
    records = []
    for name, difficulty in [('ok', 12.), ('ok', 20.), ('failed', 12.)]:
        folder = tmp_path / f'1234_{difficulty:.1f}_{name}'
        folder.mkdir()
        records.append({'song': name, 'internal_difficulty': difficulty, 'status': 'ok', 'path': str(folder)})
    records.append({'song': 'failed', 'internal_difficulty': 20., 'status': 'failed'})
    combined = []
    def combine(name, variants, folder, zip_path):
        combined.append((name, variants))
        return {'path': str(folder), 'zip_path': str(zip_path)}
    monkeypatch.setattr(main_multi, 'combine_packages', combine)
    result = main_multi.combine_maps(['ok.egg', 'failed.egg'], records, [12., 20.], False)
    assert len(result) == 1
    assert combined[0][0] == 'ok'
    assert [v['difficulty'] for v in combined[0][1]] == ['Easy', 'Normal']
    assert (old / 'ExpertPlus.dat').read_text() == 'stale'


def test_snapshot_scope_does_not_accumulate_difficulty_changes(monkeypatch):
    monkeypatch.setattr(config, 'jump_speed_offset', .7)
    original = v2_generation.settings_snapshot()
    with v2_generation.scoped_settings():
        v2_generation._difficulty_settings(20)
        config.slider_time_gap[0] = 9
    assert config.jump_speed_offset == .7
    np.testing.assert_array_equal(config.slider_time_gap, original['config']['slider_time_gap'])


def test_entrypoint_imports_have_no_ui_network_or_model_side_effects():
    code = '''
import requests
requests.get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network on import"))
import gradio
original = gradio.Blocks
class Forbidden:
    def __init__(self, *a, **k):
        raise AssertionError("UI built on import")
gradio.Blocks = Forbidden
import main, main_multi, main_app, main_training, sys
assert 'tensorflow' not in sys.modules
assert not hasattr(main_app, 'demo')
assert callable(main_app.build_app)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_main_repeated_calls_restore_offsets_and_accept_legacy_model_argument(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(paths, 'songs_pred', str(tmp_path) + '/')
    monkeypatch.setattr(main, 'validate_production_bundle', lambda: {})
    monkeypatch.setattr(main, 'check_music_files', lambda files, directory: ['song.egg'])
    calls = []
    def generate(name, logger):
        calls.append((config.use_mapper_selection, config.jump_speed_offset_orig, config.max_speed_orig))
        return {'path': str(tmp_path)}
    monkeypatch.setattr(main, 'generate_song', generate)
    monkeypatch.setattr(main, 'write_zip', lambda *args: str(tmp_path / 'song.zip'))
    initial = config.jump_speed_offset_orig
    for _ in range(2):
        result = main.main('fav_15', 20, False, js_offset=.5)
        assert result['status'] == 'ok'
    assert calls == [('v2_expert', initial + .5, 20)] * 2
    assert config.jump_speed_offset_orig == initial


def test_app_surfaces_worker_failures(monkeypatch):
    import main_app
    monkeypatch.setattr(main_app, 'get_summary', lambda *args: 'Ready')
    monkeypatch.setattr(main_app, 'validate_production_bundle', lambda: {})
    monkeypatch.setattr(main_app, 'main_multi_par', lambda *args: {'failures': [{'error': 'broken'}]})
    result = list(main_app.run_process(2, 'fav_15', 3, 5, 0, 0, 0))
    assert result[-1].endswith('Process failed: 1 failed variants/packages')
    assert 'successfully' not in result[-1]


@pytest.mark.parametrize('threads', [0, True, 1.5])
def test_bad_worker_bootstrap_fails_before_spawning(monkeypatch, threads):
    monkeypatch.setattr(config, 'v2_inference_threads', threads)
    monkeypatch.setattr(main_multi.mp, 'get_context', lambda *args: pytest.fail('spawned invalid worker'))
    with pytest.raises(ValueError, match='positive integer'):
        main_multi.main_multi_par(2, [3, 5], False)


def test_single_multi_difficulty_keeps_automatic_expert(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'new_map_path', str(tmp_path) + '/')
    monkeypatch.setattr(config, 'auto_move_song_afterwards', False)
    monkeypatch.setattr(config, 'general_diff', 'ExpertPlus')
    source = tmp_path / 'variant'
    source.mkdir()
    record = {'song': 'song', 'status': 'ok', 'internal_difficulty': 20., 'path': str(source),
              'package': {'variants': [{'difficulty': 'ExpertPlus', 'filename': 'ExpertPlus.dat'},
                                       {'difficulty': 'Expert', 'filename': 'Expert.dat'}]}}
    seen = []
    def combine(name, variants, folder, zip_path):
        seen.extend(variants)
        return {'path': str(folder)}
    monkeypatch.setattr(main_multi, 'combine_packages', combine)
    main_multi.combine_maps(['song.egg'], [record], [20.], False)
    assert [(v['difficulty'], v['filename']) for v in seen] == [('Easy', 'ExpertPlus.dat'), ('Expert', 'Expert.dat')]


def test_prediction_evidence_is_numeric_and_outside_package(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'temp_path', str(tmp_path))
    path = v2_generation.save_predictions({'raw_intensity': np.array([.25, .75], np.float32),
                                          'intensity': np.array([.27, .73], np.float32), 'conditioning_nps': 5.6,
                                          'raw_model_notes': np.zeros((0, 6)), 'notes': []})
    assert Path(path).parent == tmp_path / 'v2-generation'
    with np.load(path, allow_pickle=False) as archive:
        assert archive['final_notes'].shape == (0, 6)
        np.testing.assert_array_equal(archive['raw_intensity'], [.25, .75])
        np.testing.assert_allclose(archive['intensity'], [.27, .73])
        assert archive['conditioning_nps'] == 5.6
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_generation_requests_separate_automatic_expert_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from map_creation import v2_runtime, v2_export
    monkeypatch.setattr(paths, 'songs_pred', str(tmp_path) + '/')
    monkeypatch.setattr(paths, 'temp_path', str(tmp_path / 'temp'))
    monkeypatch.setattr(paths, 'new_map_path', str(tmp_path / 'output'))
    monkeypatch.setattr(config, 'create_expert_flag', True)
    monkeypatch.setattr(config, 'general_diff', 'ExpertPlus')
    monkeypatch.setattr(config, 'expert_fact', .63)
    (tmp_path / 'song.egg').write_bytes(b'dummy-decoder-input')
    prepared = SimpleNamespace(song=SimpleNamespace(metadata={'duration': 1., 'bpm': 120.}))
    requests, packages = [], []
    def generate(audio, density, logger):
        assert audio is prepared
        requests.append(density)
        return SimpleNamespace(state={'notes': []}, diagnostics={}, stages={})
    monkeypatch.setattr(v2_runtime, 'get_runtime', lambda: SimpleNamespace(
        prepare_audio=lambda path: prepared, generate=generate))
    def package(folder, name, bpm, variants, audio, version):
        packages.extend(variants)
        return {'path': str(folder)}
    monkeypatch.setattr(v2_export, 'write_package', package)
    report = v2_generation.generate_song('song', 20.)
    assert requests == [5., 3.15]
    assert [(v['difficulty'], v['automatic_expert']) for v in packages] == [('ExpertPlus', False), ('Expert', True)]
    assert report['status'] == 'ok'
