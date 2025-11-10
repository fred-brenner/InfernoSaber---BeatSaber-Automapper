"""Utilities for extracting and storing song metadata."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from tools.config import paths

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


def extract_metadata(file_path: str) -> Dict[str, str]:
    """Extract metadata from *file_path* if possible.

    Returns an empty dictionary if the metadata cannot be read or the dependency is missing.
    """

    if MutagenFile is None:
        return {}

    try:
        audio = MutagenFile(file_path, easy=True)
    except Exception:
        return {}

    if not audio or not getattr(audio, "tags", None):
        return {}

    metadata: Dict[str, Optional[str]] = {}
    for key in _METADATA_KEYS:
        value = audio.tags.get(key)
        if value:
            metadata[key] = value

    return _sanitize_metadata(metadata)


def save_metadata(name: str, metadata: Dict[str, Optional[str]]) -> None:
    """Persist *metadata* for the song *name* (without extension)."""

    sanitized = _sanitize_metadata(metadata)
    metadata_path = os.path.join(paths.song_data, f"{name}.json")

    if sanitized:
        os.makedirs(paths.song_data, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(sanitized, file, ensure_ascii=False, indent=2)
    elif os.path.exists(metadata_path):
        os.remove(metadata_path)


def load_metadata(name: str) -> Dict[str, str]:
    """Load persisted metadata for the song *name* (without extension)."""

    metadata_path = os.path.join(paths.song_data, f"{name}.json")
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

