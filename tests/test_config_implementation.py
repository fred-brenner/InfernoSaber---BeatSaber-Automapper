"""Tests validating consistent usage of the shared configuration object."""

from __future__ import annotations

import ast
from importlib import reload
from pathlib import Path

import pytest

from tools.config import config as shared_config
from tools.config import get_config
import tools.config.paths as paths_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_config_scripts():
    """Yield Python files that access the shared config instance."""
    for path in REPO_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            # Non-text python file – skip safely.
            continue
        if "config = get_config(" in source:
            yield path, source


CONFIG_SCRIPTS = list(_iter_config_scripts())
CONFIG_SCRIPT_IDS = [str(path) for path, _ in CONFIG_SCRIPTS]


@pytest.mark.parametrize(
    "script_path, source",
    CONFIG_SCRIPTS,
    ids=CONFIG_SCRIPT_IDS,
)
def test_scripts_import_and_assign_shared_config(script_path: Path, source: str) -> None:
    """Every script should import and assign the shared config singleton."""
    tree = ast.parse(source, filename=str(script_path))

    imported_get_config = False
    assigned_shared_config = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "get_config" for alias in node.names):
                imported_get_config = True
        elif isinstance(node, ast.Assign):
            targets_config = any(
                isinstance(target, ast.Name) and target.id == "config" for target in node.targets
            )
            if not targets_config:
                continue
            value = node.value
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name) and func.id == "get_config":
                    assigned_shared_config = True
                elif isinstance(func, ast.Attribute) and func.attr == "get_config":
                    assigned_shared_config = True

    assert imported_get_config, f"{script_path} must import get_config from the config package"
    assert assigned_shared_config, f"{script_path} must assign config = get_config()"


def test_get_config_returns_singleton() -> None:
    """The accessor must always return the same configuration instance."""
    first = get_config()
    second = get_config()

    assert first is second


def test_only_config_module_instantiates_config_class() -> None:
    """Ensure the Config class is instantiated only inside its defining module."""
    config_instantiations = []

    for path in REPO_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "Config":
                    config_instantiations.append(path)
                elif isinstance(func, ast.Attribute) and func.attr == "Config":
                    config_instantiations.append(path)

    assert config_instantiations == [REPO_ROOT / "tools" / "config" / "config.py"]


def test_paths_module_uses_current_mapper_selection(monkeypatch):
    """Reloading the paths module should respect the active mapper selection."""
    config = get_config()
    original_selection = config.use_mapper_selection

    try:
        monkeypatch.setattr(config, "use_mapper_selection", "CustomMapper", raising=False)
        reloaded_paths = reload(paths_module)
        assert reloaded_paths.model_path.endswith("custommapper/"), reloaded_paths.model_path

        monkeypatch.setattr(config, "use_mapper_selection", "", raising=False)
        reloaded_paths = reload(paths_module)
        assert reloaded_paths.model_path.endswith("general_new/"), reloaded_paths.model_path
    finally:
        monkeypatch.setattr(config, "use_mapper_selection", original_selection, raising=False)
        reload(paths_module)

    assert shared_config is get_config()
