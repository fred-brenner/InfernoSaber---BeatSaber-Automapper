import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import importlib
import pytest

from tools.config import config


def reload_module():
    module = importlib.import_module("tools.utils.song_metadata")
    return importlib.reload(module)


def test_extract_metadata_prefers_audio_tags(monkeypatch):
    module = reload_module()

    class DummyAudio:
        tags = {
            "title": ["Some Title"],
            "artist": ["Some Artist"],
            "genre": ["Electronic"],
        }

    monkeypatch.setattr(module, "MutagenFile", lambda *_args, **_kwargs: DummyAudio())

    metadata = module.extract_metadata("ignored.mp3")
    assert metadata == {
        "title": "Some Title",
        "artist": "Some Artist",
        "genre": "Electronic",
    }


def test_extract_metadata_fallbacks_to_filename(monkeypatch):
    module = reload_module()

    class DummyAudio:
        tags = {}

    monkeypatch.setattr(module, "MutagenFile", lambda *_args, **_kwargs: DummyAudio())

    metadata = module.extract_metadata("Artist Name - Song Title.ogg")
    assert metadata == {"artist": "Artist Name", "title": "Song Title"}


def test_extract_metadata_without_mutagen(monkeypatch):
    module = reload_module()
    monkeypatch.setattr(module, "MutagenFile", None)

    metadata = module.extract_metadata("Other Artist - Other Song.flac")
    assert metadata == {"artist": "Other Artist", "title": "Other Song"}


def test_extract_metadata_returns_empty_when_no_match(monkeypatch):
    module = reload_module()

    class DummyAudio:
        tags = {}

    monkeypatch.setattr(module, "MutagenFile", lambda *_args, **_kwargs: DummyAudio())

    with monkeypatch.context() as m:
        m.setattr(config, "metadata_naming_convention", "{artist} - {title}")
        metadata = module.extract_metadata("NotMatchingFilename.mp3")

    assert metadata == {}
