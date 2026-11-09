"""Settings created during one generation must not leak into the next."""
import numpy as np
import pytest

from map_creation.v2_generation import scoped_settings, settings_snapshot
from tools.config import config, paths


def test_new_obstacle_settings_do_not_change_next_scope_snapshot(monkeypatch):
    for key in ('obstacle_positions', 'obstacle_allowed_types'):
        monkeypatch.delattr(config, key, raising=False)
    before = settings_snapshot()
    with scoped_settings():
        config.obstacle_positions = [[0], [3]]
        config.obstacle_allowed_types = [0, 1, 2]
    with scoped_settings():
        np.testing.assert_equal(settings_snapshot(), before)
    assert not hasattr(config, 'obstacle_positions')
    assert not hasattr(config, 'obstacle_allowed_types')


def test_nested_scopes_restore_existing_values_and_new_settings_on_exception(monkeypatch):
    monkeypatch.setattr(config, 'scope_test_array', np.array([1, 2]), raising=False)
    monkeypatch.setattr(config, 'scope_test_list', [[1]], raising=False)
    monkeypatch.setattr(paths, 'scope_test_path', 'original/', raising=False)
    with scoped_settings():
        config.scope_test_array[0] = 7
        config.scope_test_list[0].append(2)
        paths.scope_test_path = 'outer/'
        config.scope_test_outer = [3]
        with pytest.raises(RuntimeError, match='failure'):
            with scoped_settings():
                config.scope_test_array[:] = 9
                config.scope_test_list[0].append(4)
                config.scope_test_outer.append(5)
                config.scope_test_inner = np.array([8])
                paths.scope_test_path = 'inner/'
                paths.scope_test_new_path = 'new/'
                raise RuntimeError('failure')
        np.testing.assert_array_equal(config.scope_test_array, [7, 2])
        assert config.scope_test_list == [[1, 2]]
        assert config.scope_test_outer == [3]
        assert paths.scope_test_path == 'outer/'
        assert not hasattr(config, 'scope_test_inner')
        assert not hasattr(paths, 'scope_test_new_path')
    np.testing.assert_array_equal(config.scope_test_array, [1, 2])
    assert config.scope_test_list == [[1]]
    assert paths.scope_test_path == 'original/'
    assert not hasattr(config, 'scope_test_outer')


def test_private_and_unsupported_caches_and_preexisting_names_survive(monkeypatch):
    existing = object()
    monkeypatch.setattr(config, 'scope_test_existing_object', existing, raising=False)
    monkeypatch.setattr(paths, 'scope_test_existing_object', existing, raising=False)
    new_cache = object()
    try:
        with scoped_settings():
            config.scope_test_existing_object = [1]
            paths.scope_test_existing_object = 'promoted/'
            config._scope_test_private = [1]
            config.scope_test_new_cache = new_cache
            paths.scope_test_new_cache = new_cache
        assert config.scope_test_existing_object == [1]
        assert paths.scope_test_existing_object == 'promoted/'
        assert config._scope_test_private == [1]
        assert config.scope_test_new_cache is new_cache
        assert paths.scope_test_new_cache is new_cache
    finally:
        for module, key in ((config, '_scope_test_private'), (config, 'scope_test_new_cache'),
                            (paths, 'scope_test_new_cache')):
            if hasattr(module, key):
                delattr(module, key)


def test_settings_introduced_by_applied_snapshot_are_removed_on_exit():
    snapshot = {'config': {'scope_test_applied': [1]},
                'paths': {'scope_test_applied_path': 'temporary/'}}
    with scoped_settings(snapshot):
        assert config.scope_test_applied == [1]
        assert paths.scope_test_applied_path == 'temporary/'
    assert not hasattr(config, 'scope_test_applied')
    assert not hasattr(paths, 'scope_test_applied_path')
