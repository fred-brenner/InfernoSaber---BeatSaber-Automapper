"""Tempo accuracy against generated audio, without model or song assets."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from map_creation import find_bpm


@pytest.fixture
def click_file(tmp_path):
    def write(bpm=128, duration=60, offset=0.371, dropouts=False,
              subdivisions=False):
        sr = 44100
        audio = np.zeros(round(duration * sr), dtype=np.float32)
        t = np.arange(round(0.03 * sr)) / sr
        pulse = np.exp(-t / 0.005) * np.sin(2 * np.pi * 1800 * t)
        if bpm:
            for index, when in enumerate(np.arange(offset, duration - 0.05, 60 / bpm)):
                if dropouts and (index % 9 == 0 or 25 < when < 30):
                    continue
                start = round(when * sr)
                audio[start:start + len(pulse)] += pulse * (1 if index % 4 == 0 else 0.7)
                if subdivisions:
                    start = round((when + 30 / bpm) * sr)
                    if start + len(pulse) <= len(audio):
                        audio[start:start + len(pulse)] += pulse * 0.3
        path = tmp_path / "fractional rhythm.wav"
        wavfile.write(path, sr, audio)
        return path
    return write


def assert_equivalent_tempo(measured, expected):
    assert measured > 0
    # Integer multiples and their reciprocals describe acceptable pulse rates.
    factor = (round(measured / expected) if measured >= expected
              else 1 / round(expected / measured))
    assert abs(measured / factor - expected) <= 0.1


@pytest.mark.parametrize("bpm", [60, 90, 105, 120, 121.8, 122, 127.3, 128, 160, 180, 240])
@pytest.mark.parametrize("subdivisions", [False, True])
def test_tempo_accuracy_with_missing_beats(click_file, bpm, subdivisions):
    path = click_file(bpm=bpm, dropouts=True, subdivisions=subdivisions)
    result = find_bpm.get_file_bpm(str(path))
    assert isinstance(result, list) and len(result) == 2
    assert_equivalent_tempo(result[0], bpm)
    assert result[1] == pytest.approx(60, abs=1 / 44100)


@pytest.mark.parametrize("offset", [0.173, 0.647])
def test_fractional_tempo_survives_time_offset(click_file, offset):
    bpm, duration = find_bpm.get_file_bpm(str(click_file(bpm=121.8, offset=offset)))
    assert_equivalent_tempo(bpm, 121.8)
    assert duration == pytest.approx(60, abs=1 / 44100)


@pytest.mark.parametrize("params", [
    None, {}, {"mode": "default"}, argparse.Namespace(mode="default"),
    {"mode": "fast"}, argparse.Namespace(mode="super-fast"),
    {"mode": "fast", "samplerate": 16000, "win_s": 1024, "hop_s": 256},
    argparse.Namespace(samplerate=22050, win_s=2048, hop_s=256),
])
def test_modes_and_manual_overrides(click_file, params):
    bpm, duration = find_bpm.get_file_bpm(str(click_file(bpm=127.3)), params)
    assert_equivalent_tempo(bpm, 127.3)
    # Preserve the decoder's frame-count duration, including resampler rounding.
    assert duration == pytest.approx(60, abs=0.005)


@pytest.mark.parametrize("bpm,duration", [(0, 5), (0, 0.01), (1, 5), (60, 2), (60, 3)])
def test_no_tempo_without_four_onsets(click_file, bpm, duration):
    measured, length = find_bpm.get_file_bpm(str(click_file(bpm=bpm, duration=duration)))
    assert measured == 0
    assert length == pytest.approx(duration, abs=1 / 44100)


@pytest.mark.parametrize("params", [{"mode": "unknown"}, argparse.Namespace(mode="unknown")])
def test_unknown_mode_is_rejected(params):
    with pytest.raises(ValueError, match="unknown mode"):
        find_bpm.get_file_bpm("unused.wav", params)


def test_missing_audio_propagates_decoder_error(tmp_path):
    with pytest.raises(RuntimeError):
        find_bpm.get_file_bpm(str(tmp_path / "missing.wav"))


def test_stream_is_closed_after_read_error(monkeypatch):
    class BrokenStream:
        samplerate = 44100
        closed = False

        def __call__(self):
            raise OSError("decoder read failed")

        def close(self):
            self.closed = True

    stream = BrokenStream()
    monkeypatch.setattr(find_bpm, "source", lambda *args: stream)
    with pytest.raises(OSError, match="decoder read failed"):
        find_bpm.get_file_bpm("broken.wav")
    assert stream.closed


def test_cli_prints_scalar_bpm_and_filename(click_file):
    path = click_file(bpm=127.3)
    result = subprocess.run(
        [sys.executable, str(ROOT_DIR / "map_creation/find_bpm.py"),
         "--mode", "fast", str(path)],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True,
    )
    measured, filename = result.stdout.strip().split(maxsplit=1)
    assert_equivalent_tempo(float(measured), 127.3)
    assert filename == str(path)
