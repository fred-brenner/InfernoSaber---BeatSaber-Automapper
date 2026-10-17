"""Configured audio/context contracts stay aligned before TensorFlow is loaded."""

import subprocess
import sys

import pytest

from tools.config import config
from training.beat_models.settings import architecture_settings


def test_configured_audio_and_contexts_match_in_a_fresh_process():
    # Dimension settings are read once per process, just like the existing config.
    code = '''
import sys
import numpy as np
from tools.config import config
legacy = (config.samplerate_music, config.hop_size, config.window, config.lstm_len)
config.beat_models_samplerate_music = 8000
config.beat_models_hop_size = 80
config.beat_models_fft_size = 256
config.beat_models_mel_bins = 20
config.timing_kernel_size = 5
config.timing_convolutions_per_block = 1
config.timing_dilations = (1, 3)
config.timing_chunk_frames = 64
config.notes_patch_frames = 40
config.notes_history_length = 5
config.random_seed = 19
from training.beat_models.contracts import (
    CORE, HALO, TIMING_WINDOW, PATCH, HISTORY, FEATURES, SEED,
    REPRESENTATION, Song, note_inputs, timing_inputs)
from training.beat_models.features import extract_features
assert (CORE, HALO, TIMING_WINDOW, PATCH, HISTORY, FEATURES, SEED) == (64, 8, 80, 40, 5, 22, 19)
assert REPRESENTATION['sample_rate'] == 8000
assert REPRESENTATION['halo'] == 8 and REPRESENTATION['history'] == 5
arrays, metadata = extract_features(np.zeros(12513, dtype=np.float32))
assert metadata['duration'] == 12513 / 8000
assert arrays['features'].shape == (157, 22)
metadata['bpm'] = 120
arrays['timing'] = np.zeros(157, dtype=np.int32)
song = Song('configured', metadata, arrays)
conditions = np.zeros((157, 4), dtype=np.float32)
inputs, labels, mask = timing_inputs(song, conditions, 0)
assert inputs['audio'].shape == (1, 80, 22)
assert labels.shape == mask.shape == (1, 64)
assert not inputs['mask'][0, :8].any()
notes = note_inputs(song, 50, 1, conditions, [], [])
assert notes['audio'].shape == (1, 40, 20, 1)
assert notes['history'].shape == (1, 5, 2)
assert (config.samplerate_music, config.hop_size, config.window, config.lstm_len) == legacy
assert not any(name == 'tensorflow' or name.startswith('tensorflow.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage,field,value", [
    ("timing", "kernel_size", 4),
    ("timing", "dilations", (1, 1, 2)),
    ("notes", "patch_frames", 201),
    ("notes", "pool_sizes", ((4, 4),)),
])
def test_ambiguous_context_and_cnn_settings_are_rejected(stage, field, value, monkeypatch):
    monkeypatch.setattr(config, f"{stage}_{field}", value)
    with pytest.raises(ValueError):
        architecture_settings(stage)


def test_configured_group_splits_preserve_one_training_group_and_report_ratios(monkeypatch):
    from training.beat_models.data import assign_splits, prepare_songs

    monkeypatch.setattr(config, "beat_models_validation_fraction", .25)
    monkeypatch.setattr(config, "beat_models_test_fraction", .25)
    songs, report = prepare_songs(dummy=True)
    assert report["split_ratio"] == [.5, .25, .25]
    assert {split: sum(song.metadata["split"] == split for song in songs)
            for split in ("train", "validation", "test")} == {"train": 6, "validation": 3, "test": 3}
    monkeypatch.setattr(config, "beat_models_validation_fraction", .49)
    monkeypatch.setattr(config, "beat_models_test_fraction", .49)
    small = assign_splits(songs[:4])
    assert sum(song.metadata["split"] == "train" for song in small) == 1
