"""Runtime settings persist to untracked override files, never tracked configs."""
import contextlib
import importlib
import sys
import types

import numpy as np

from app_helper.update_dir_path import update_dir_path
from tools.config import config, paths


@contextlib.contextmanager
def _swapped_module(name, module):
    sentinel = object()
    previous = sys.modules.get(name, sentinel)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def test_config_writes_go_to_untracked_user_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_dir_path('tools/config/config.py', 'num_workers', 4)
    target = tmp_path / 'tools/config/user_settings.py'
    assert target.read_text().endswith('num_workers = 4\n')
    assert not (tmp_path / 'tools/config/config.py').exists()


def test_paths_writes_go_to_untracked_user_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_dir_path('tools/config/paths.py', 'dir_path', 'C:/data/')
    assert 'dir_path = "C:/data/"' in (tmp_path / 'tools/config/user_paths.py').read_text()


def test_override_file_updates_in_place_and_appends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_dir_path('tools/config/config.py', 'num_workers', 4)
    update_dir_path('tools/config/config.py', 'num_workers', 8)
    update_dir_path('tools/config/config.py', 'verbose_level', 2)
    text = (tmp_path / 'tools/config/user_settings.py').read_text()
    assert text.count('num_workers =') == 1
    assert 'num_workers = 8' in text and 'verbose_level = 2' in text


def test_numpy_values_serialize_as_valid_python(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_dir_path('tools/config/config.py', 'slider_time_gap', np.array([.17, .39]))
    namespace = {}
    exec((tmp_path / 'tools/config/user_settings.py').read_text(), namespace)
    np.testing.assert_allclose(namespace['slider_time_gap'], [.17, .39])


def test_other_files_keep_replace_only_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path / 'custom.py'
    other.write_text('value = 1\n')
    update_dir_path(str(other), 'value', 2)
    assert other.read_text() == 'value = 2\n'
    update_dir_path(str(other), 'missing', 1)   # no append outside overrides
    assert 'missing' not in other.read_text()


def test_user_settings_override_config_and_rederive():
    user = types.ModuleType('tools.config.user_settings')
    user.jump_speed_offset = 0.9
    user.verbose_level = 5
    with _swapped_module('tools.config.user_settings', user):
        importlib.reload(config)
        try:
            assert config.verbose_level == 5
            assert config.jump_speed_offset == 0.9
            assert config.jump_speed_offset_orig == 0.9  # derived follows override
        finally:
            importlib.reload(config)


def test_explicit_derived_override_wins():
    user = types.ModuleType('tools.config.user_settings')
    user.silence_threshold = .7
    user.silence_threshold_orig = .9
    with _swapped_module('tools.config.user_settings', user):
        importlib.reload(config)
        try:
            assert config.silence_threshold == .7
            assert config.silence_threshold_orig == .9
        finally:
            importlib.reload(config)


def test_user_paths_override_drives_derived_paths():
    user = types.ModuleType('tools.config.user_paths')
    user.dir_path = '/custom/data/'
    user.bs_song_path = '/custom/beatsaber/'
    with _swapped_module('tools.config.user_paths', user):
        importlib.reload(paths)
        try:
            assert paths.dir_path == '/custom/data/'
            assert paths.bs_song_path == '/custom/beatsaber/'
            assert paths.songs_pred == '/custom/data/prediction/songs_predict/'
            assert paths.new_map_path == '/custom/data/prediction/new_map/'
        finally:
            importlib.reload(paths)
