"""Beat-pair training input checks without starting model training."""

import numpy as np

from training.beat_models import __main__ as command
from training.beat_models import runner
from training.beat_models.artifacts import write_dataset


def test_intensity_training_does_not_load_fine_audio(tmp_path, monkeypatch):
    songs, manifest = command._dummy_data()
    dataset = tmp_path / "dataset"
    write_dataset(dataset, songs, manifest["normalization"])
    original = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def tracked(archive, key):
        reads.append(key)
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    monkeypatch.setattr(runner, "train_stage", lambda *args, **kwargs: None)
    command.main(["train", "--stage", "intensity", "--dataset", str(dataset),
                  "--run-dir", str(tmp_path / "run"), "--dry-run"])
    assert reads and "coarse_features" in reads
    assert "features" not in reads
