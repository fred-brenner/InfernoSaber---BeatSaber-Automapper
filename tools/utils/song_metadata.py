"""Utilities for extracting and storing song metadata."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional
import re
from pathlib import Path

from tools.config import paths, config

try:
    from mutagen import File as MutagenFile
except ImportError:  # pragma: no cover - gracefully handle optional dependency
    MutagenFile = None


_METADATA_KEYS = ("title", "artist", "album", "genre")


def _sanitize_metadata(metadata: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Return a copy of *metadata* with falsy entries removed and values cast to strings."""

    if not metadata:
        return {}

    sanitized: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        text = str(value).strip()
        if text:
            sanitized[key] = text
    return sanitized


def _metadata_from_filename(file_path: str) -> Dict[str, str]:
    """Attempt to derive song metadata from *file_path* using configured conventions."""

    if not getattr(config, "enable_auto_metadata", False):
        return {}

    convention = getattr(config, "metadata_naming_convention", "")
    if not convention or "{artist}" not in convention or "{title}" not in convention:
        return {}

    filename = Path(file_path).stem

    pattern = re.escape(convention)
    pattern = pattern.replace(r"\{artist\}", r"(?P<artist>.+?)")
    pattern = pattern.replace(r"\{title\}", r"(?P<title>.+?)")

    match = re.fullmatch(pattern, filename)
    if not match:
        return {}

    metadata: Dict[str, Optional[str]] = {
        key: (value.strip() if value is not None else value)
        for key, value in match.groupdict().items()
    }
    return _sanitize_metadata(metadata)


def extract_metadata(file_path: str) -> Dict[str, str]:
    """Extract metadata from *file_path* if possible.

    Returns an empty dictionary if the metadata cannot be read and the filename does not
    match the configured metadata naming convention.
    """

    metadata: Dict[str, Optional[str]] = {}

    if MutagenFile is not None:
        try:
            audio = MutagenFile(file_path, easy=True)
        except Exception:
            audio = None

        if audio and getattr(audio, "tags", None):
            for key in _METADATA_KEYS:
                value = audio.tags.get(key)
                if value:
                    metadata[key] = value

    sanitized = _sanitize_metadata(metadata)
    if sanitized:
        return sanitized

    return _metadata_from_filename(file_path)


def save_metadata(name: str, metadata: Dict[str, Optional[str]]) -> None:
    """Persist *metadata* for the song *name* (without extension)."""

    sanitized = _sanitize_metadata(metadata)
    metadata_path = os.path.join(paths.songs_pred, f"{name}.json")

    if sanitized:
        os.makedirs(paths.songs_pred, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(sanitized, file, ensure_ascii=False, indent=2)
    elif os.path.exists(metadata_path):
        os.remove(metadata_path)


def load_metadata(name: str) -> Dict[str, str]:
    """Load persisted metadata for the song *name* (without extension)."""

    metadata_path = os.path.join(paths.songs_pred, f"{name}.json")
    if not os.path.isfile(metadata_path):
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    return _sanitize_metadata(data)


def metadata_to_tags(metadata: Dict[str, Optional[str]] | None) -> Optional[Dict[str, str]]:
    """Prepare *metadata* for embedding into an audio file."""

    return _sanitize_metadata(metadata or {}) or None

