"""Enabled-song supervision masking and history-corruption augmentation contracts."""

from copy import deepcopy

import numpy as np
import pytest

from training.pipeline_v2.contracts import StageExamples


def _arc_examples(attach_targets, history_attach=None, supervised=None):
    """Arc stage examples: [attach, mu, tmu, anchor] history rows."""
    count = len(attach_targets)
    history = np.zeros((count, 4, 4), np.float32)
    mask = np.zeros((count, 4), bool)
    if history_attach is not None:
        for row, values in enumerate(history_attach):
            for column, attach in enumerate(values):
                history[row, column] = [attach, .5, .5, 0]
                mask[row, column] = True
    inputs = {"history": history, "history_mask": mask,
              "history_age": np.zeros((count, 4, 1), np.float32)}
    weights = {"attach": np.ones(count, np.float32) if supervised is None else np.asarray(supervised, np.float32),
               "shape": np.asarray(attach_targets, np.float32), "anchor": np.asarray(attach_targets, np.float32)}
    targets = {"attach": np.asarray(attach_targets, np.int32),
               "shape": np.ones((count, 2), np.float32), "anchor": np.zeros(count, np.int32)}
    return StageExamples(inputs, targets, weights, np.arange(count, dtype=np.float64))


def _hazard_examples(wall_targets, bomb_targets, wall_history=None, bomb_history=None):
    count = len(wall_targets)
    history = np.zeros((count, 4, 13), np.float32)
    mask = np.zeros((count, 4), bool)
    for row, value in enumerate(wall_history or []):
        history[row, :, 0] = value
        mask[row] = True
    for row, value in enumerate(bomb_history or []):
        history[row, :, 1:] = value
        mask[row] = True
    inputs = {"history": history, "history_mask": mask,
              "history_age": np.zeros((count, 4, 1), np.float32)}
    targets = {"wall": np.asarray(wall_targets, np.int32),
               "bombs": np.asarray(bomb_targets, np.float32).reshape(count, 12)}
    weights = {"wall": np.ones(count, np.float32),
               "bombs": np.ones((count, 12), np.float32)}
    return StageExamples(inputs, targets, weights, np.arange(count, dtype=np.float64))


def _note_examples(targets, history=None, mask=None):
    """Categorical (red, blue) note-stage examples for positions/arrows/multiplier."""
    count = len(targets)
    if history is None:
        history = np.zeros((count, 4, 2), np.int32)
    if mask is None:
        mask = np.ones((count, history.shape[1]), bool)
    inputs = {"history": history, "history_mask": mask,
              "history_age": np.zeros((count, history.shape[1], 1), np.float32)}
    targets = {"red": np.asarray([pair[0] for pair in targets], np.int32),
               "blue": np.asarray([pair[1] for pair in targets], np.int32)}
    weights = {"red": np.ones(count, np.float32), "blue": np.ones(count, np.float32)}
    return StageExamples(inputs, targets, weights, np.arange(count, dtype=np.float64))


class TestEnabledSongMask:
    def test_positive_song_is_unchanged(self):
        from training.pipeline_v2.supervision import enabled_song_mask

        examples = _arc_examples([0, 1, 0], [[0, 0, 0, 0]] * 3)
        result, report = enabled_song_mask(examples, "arcs")
        assert result is examples
        assert report == {}

    def test_disabled_song_masks_only_its_head(self):
        from training.pipeline_v2.supervision import enabled_song_mask

        examples = _arc_examples([0, 0, 0])
        result, report = enabled_song_mask(examples, "arcs")
        assert result is not examples
        assert report["masked_supervised_values"] == {"attach": 3}
        assert np.count_nonzero(result.weights["attach"]) == 0
        # Other heads and every other family are untouched.
        np.testing.assert_array_equal(result.weights["shape"], examples.weights["shape"])
        assert result.inputs is examples.inputs and result.targets is examples.targets
        np.testing.assert_array_equal(result.times, examples.times)

    def test_masked_positives_do_not_enable(self):
        from training.pipeline_v2.supervision import enabled_song_mask

        # attach=1 but weight=0 everywhere: unsupported, counts as disabled.
        examples = _arc_examples([1, 0, 0], supervised=[0, 1, 1])
        result, report = enabled_song_mask(examples, "arcs")
        assert report["masked_supervised_values"] == {"attach": 2}
        assert np.count_nonzero(result.weights["attach"]) == 0

    def test_hazard_heads_mask_independently(self):
        from training.pipeline_v2.supervision import enabled_song_mask

        bombs = np.zeros((3, 12), np.float32)
        bombs[1, 5] = 1
        examples = _hazard_examples([0, 0, 0], bombs)
        result, report = enabled_song_mask(examples, "hazards")
        assert report["masked_supervised_values"] == {"wall": 3}
        assert np.count_nonzero(result.weights["wall"]) == 0
        np.testing.assert_array_equal(result.weights["bombs"], examples.weights["bombs"])

    def test_non_sparse_stage_is_a_noop(self):
        from training.pipeline_v2.supervision import enabled_song_mask

        examples = _arc_examples([0, 0])
        result, report = enabled_song_mask(examples, "positions")
        assert result is examples and report == {}


def _reader_dataset(tmp_path, dataset):
    from training.pipeline_v2.artifacts import load_dataset, write_dataset

    root = tmp_path / "dataset"
    write_dataset(root, dataset)
    return load_dataset(root)


@pytest.fixture(scope="module")
def dataset():
    from training.pipeline_v2.data import make_dummy_dataset

    return make_dummy_dataset()


class TestEnabledSongTrainingData:
    def test_flag_off_preserves_original_arrays(self, tmp_path, dataset):
        from training.pipeline_v2.training_data import TrainingData

        reader = TrainingData(_reader_dataset(tmp_path, dataset), "arcs",
                              enabled_song_supervision=False)
        assert reader.report["enabled_song_supervision"]["active"] is False
        expected = next(song.examples["arcs"] for song in dataset.songs
                        if song.base.metadata["split"] == "train")
        actual = next(iter(reader.examples("train")))
        np.testing.assert_array_equal(actual.weights["attach"], expected.weights["attach"])

    def test_flag_on_reports_masked_counts(self, tmp_path, dataset):
        from training.pipeline_v2.training_data import TrainingData

        reader = TrainingData(_reader_dataset(tmp_path, dataset), "hazards",
                              enabled_song_supervision=True)
        summary = reader.report["enabled_song_supervision"]
        assert summary["active"] is True
        assert set(summary["masked_supervised_values"]) == {"train", "validation", "test"}
        for split, counts in summary["masked_supervised_values"].items():
            assert summary["feature_disabled_songs"][split] >= len(counts) >= 0
        # Every masked head lost its whole song supervision for that head.
        for examples in reader.examples():
            for head in ("wall", "bombs"):
                positive = np.asarray(examples.targets[head]) > 0
                weight = np.broadcast_to(examples.weights[head], positive.shape)
                if not (positive & (weight > 0)).any():
                    # No effective supervised positive -> feature-disabled song
                    # must contribute zero supervision for this head.
                    assert np.count_nonzero(examples.weights[head]) == 0

    def test_streamed_reads_apply_the_same_mask(self, tmp_path, dataset):
        from training.pipeline_v2.training_data import TrainingData

        cached = TrainingData(_reader_dataset(tmp_path / "a", dataset), "hazards",
                              ram_cache_mib=64, enabled_song_supervision=True)
        streamed = TrainingData(_reader_dataset(tmp_path / "b", dataset), "hazards",
                                ram_cache_mib=0, enabled_song_supervision=True)
        for cached_song, streamed_song in zip(cached.examples(), streamed.examples()):
            for head in ("wall", "bombs"):
                np.testing.assert_array_equal(cached_song.weights[head], streamed_song.weights[head])

    def test_invalid_flag_rejected(self, tmp_path, dataset):
        from training.pipeline_v2.training_data import TrainingData

        with pytest.raises(ValueError, match="enabled_song_supervision"):
            TrainingData(_reader_dataset(tmp_path, dataset), "arcs",
                         enabled_song_supervision="yes")


class TestHistoryCorruption:
    def test_full_corruption_keeps_only_the_most_recent_positive_record(self, dataset):
        from training.pipeline_v2.runner import batches

        clean = list(batches(dataset, "arcs", "train", batch_size=64))
        assert any(np.count_nonzero(batch.inputs["history"]) for batch in clean)
        corrupted = list(batches(dataset, "arcs", "train", batch_size=64,
                                 corruption={"rate": 1., "drop": 1.},
                                 corruption_rng=np.random.default_rng(0)))
        for old, new in zip(clean, corrupted):
            had = (old.inputs["history"] > 0).any(axis=-1)
            kept = (new.inputs["history"] > 0).any(axis=-1)
            for row in range(len(kept)):
                if had[row].any():
                    expected = np.zeros(len(kept[row]), bool)
                    expected[np.flatnonzero(had[row])[-1]] = True
                    np.testing.assert_array_equal(kept[row], expected)
                else:
                    assert not kept[row].any()
            np.testing.assert_array_equal(new.inputs["history"][kept], old.inputs["history"][kept])
            for key, value in old.inputs.items():
                if key != "history":
                    np.testing.assert_array_equal(new.inputs[key], value)
            for name in ("targets", "weights"):
                for key, value in getattr(old, name).items():
                    np.testing.assert_array_equal(getattr(new, name)[key], value)
        # Source dataset examples remain uncorrupted.
        for old, new in zip(clean, batches(dataset, "arcs", "train", batch_size=64)):
            np.testing.assert_array_equal(new.inputs["history"], old.inputs["history"])

    def test_full_corruption_without_keep_guard_may_empty_windows(self, dataset):
        from training.pipeline_v2.runner import batches

        clean = list(batches(dataset, "arcs", "train", batch_size=64))
        corrupted = list(batches(dataset, "arcs", "train", batch_size=64,
                                 corruption={"rate": 1., "drop": 1., "keep_guard": False},
                                 corruption_rng=np.random.default_rng(0)))
        emptied = 0
        for old, new in zip(clean, corrupted):
            had = (old.inputs["history"] > 0).any(axis=-1)
            kept = (new.inputs["history"] > 0).any(axis=-1)
            emptied += int((had.any(axis=1) & ~kept.any(axis=1)).sum())
            for key, value in old.inputs.items():
                if key != "history":
                    np.testing.assert_array_equal(new.inputs[key], value)
            for name in ("targets", "weights"):
                for key, value in getattr(old, name).items():
                    np.testing.assert_array_equal(getattr(new, name)[key], value)
        assert emptied > 0  # rate=1/drop=1 always empties; no record restored

    def test_zero_rate_is_a_noop(self, dataset):
        from training.pipeline_v2.runner import batches

        clean = list(batches(dataset, "hazards", "train", batch_size=64))
        corrupted = list(batches(dataset, "hazards", "train", batch_size=64,
                                 corruption={"rate": 0., "drop": 1.},
                                 corruption_rng=np.random.default_rng(0)))
        for old, new in zip(clean, corrupted):
            np.testing.assert_array_equal(old.inputs["history"], new.inputs["history"])

    def test_partial_drop_and_determinism(self, dataset):
        from training.pipeline_v2.runner import batches

        kwargs = {"corruption": {"rate": 1., "drop": .5}}
        first = list(batches(dataset, "hazards", "train", batch_size=64,
                             corruption_rng=np.random.default_rng(7), **kwargs))
        second = list(batches(dataset, "hazards", "train", batch_size=64,
                              corruption_rng=np.random.default_rng(7), **kwargs))
        for one, two in zip(first, second):
            np.testing.assert_array_equal(one.inputs["history"], two.inputs["history"])
        clean = list(batches(dataset, "hazards", "train", batch_size=64))
        changed = sum(not np.array_equal(old.inputs["history"], new.inputs["history"])
                      for old, new in zip(clean, first))
        assert changed > 0

    def test_unknown_stage_rejected(self):
        from training.pipeline_v2.runner import _corrupt_history

        examples = _note_examples([[1, 0], [0, 5]])
        with pytest.raises(ValueError, match="History corruption is undefined"):
            _corrupt_history(examples, "hands", np.random.default_rng(0), 1., 1., classes=13)

    def test_categorical_corruption_substitutes_only_valid_slots(self):
        from training.pipeline_v2.runner import _corrupt_history

        rng = np.random.default_rng(0)
        for stage, classes in (("positions", 13), ("arrows", 10)):
            history = np.array([[[13, 13], [1, 0], [4, 8], [14, 14]],
                                [[9, 2], [7, 7], [14, 14], [14, 14]]], np.int32)
            mask = np.array([[True, True, True, False],
                             [True, True, False, False]])
            examples = _note_examples([[1, 0]] * 2, history, mask)
            result = _corrupt_history(examples, stage, rng, 1., 1., classes=classes)
            out = result.inputs["history"]
            # Valid classes stay in range; sentinels and masked slots survive.
            valid = mask[..., None] & (history < classes)
            assert (out[valid] < classes).all()
            np.testing.assert_array_equal(out[~valid], history[~valid])
            assert out[0, 0, 0] == 13
            # Valid classes were substituted; the (0,0)-guard may restore one cell.
            equal = valid & (out == history)
            for row in np.argwhere(equal.any(axis=-1)):
                assert equal[row].sum() == 1 and (out[row] != 0).any()
            # No impossible all-absent row.
            assert not (out == 0).all(axis=-1)[mask].any()
            # Other inputs untouched.
            for key, value in examples.inputs.items():
                if key != "history":
                    np.testing.assert_array_equal(result.inputs[key], value)

    def test_categorical_corruption_is_deterministic_and_rate_gated(self):
        from training.pipeline_v2.runner import _corrupt_history

        history = np.tile(np.arange(12, dtype=np.int32).reshape(4, 3)[:, :2], (2, 1, 1))
        mask = np.ones((2, 4), bool)
        examples = _note_examples([[1, 0]] * 2, history, mask)
        kwargs = dict(rate=.5, drop=.5, classes=13)
        one = _corrupt_history(examples, "positions", np.random.default_rng(7), **kwargs)
        two = _corrupt_history(examples, "positions", np.random.default_rng(7), **kwargs)
        np.testing.assert_array_equal(one.inputs["history"], two.inputs["history"])
        clean = _corrupt_history(examples, "positions", np.random.default_rng(7), rate=0., drop=1., classes=13)
        assert clean is examples

    def test_settings_read_corruption_keys(self, monkeypatch):
        from tools.config import config
        from training.pipeline_v2.settings import training_settings

        monkeypatch.setattr(config, "hazards_history_corrupt_rate", .25, raising=False)
        monkeypatch.setattr(config, "hazards_history_corrupt_drop", .5, raising=False)
        settings = training_settings("hazards")
        assert settings["history_corrupt_rate"] == .25
        assert settings["history_corrupt_drop"] == .5
        assert settings["enabled_song_supervision"] is True
        monkeypatch.setattr(config, "arcs_history_corrupt_rate", 1.5, raising=False)
        with pytest.raises(ValueError, match="must not exceed one"):
            training_settings("arcs")

    def test_settings_read_negative_supervision_weight(self, monkeypatch):
        from tools.config import config
        from training.pipeline_v2.settings import training_settings

        monkeypatch.setattr(config, "hazards_negative_supervision_weight", .1, raising=False)
        monkeypatch.setattr(config, "arcs_negative_supervision_weight", .3, raising=False)
        assert training_settings("hazards")["negative_supervision_weight"] == .1
        assert training_settings("arcs")["negative_supervision_weight"] == .3
        monkeypatch.setattr(config, "arcs_negative_supervision_weight", 1.5, raising=False)
        with pytest.raises(ValueError, match="must not exceed one"):
            training_settings("arcs")


class TestRegimeReporting:
    def test_regime_split_and_enabled_flags(self):
        from training.pipeline_v2.calibration import _regime_report

        # Query 0 attaches with no positive history (start); query 1 attaches
        # with positive history (continuation); query 2 is a supervised NONE.
        examples = _arc_examples([1, 1, 0],
                                 history_attach=[[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        predicted = {"attach": np.asarray([0, 1, 0])}
        report = _regime_report("arcs", examples, predicted)
        assert report["enabled"] == {"attach": True}
        start, continuation = report["regimes"]["start"]["attach"], report["regimes"]["continuation"]["attach"]
        assert start["false_negative"] == 1. and start["reference_positives"] == 1.
        assert continuation["true_positive"] == 1. and continuation["f1"] == 1.

    def test_disabled_song_flag(self):
        from training.pipeline_v2.calibration import _regime_report

        examples = _arc_examples([0, 0])
        report = _regime_report("arcs", examples, {"attach": np.asarray([0, 0])})
        assert report["enabled"] == {"attach": False}

    def test_aggregate_accumulates_regimes_and_enabled(self):
        from training.pipeline_v2.calibration import _aggregate

        from training.pipeline_v2.calibration import _regime_report

        enabled_song = _arc_examples([1, 0], history_attach=[[0, 0, 0, 0]] * 2)
        disabled_song = _arc_examples([0, 0])
        reports = []
        for song_examples, decisions, counts in ((enabled_song, [1, 0], {"true_positive": 1.}),
                                                 (disabled_song, [0, 1], {"false_positive": 1.})):
            regime = _regime_report("arcs", song_examples, {"attach": np.asarray(decisions)})
            heads = {"true_positive": 0., "false_positive": 0., "false_negative": 0., **counts}
            reports.append({"heads": {"attach": heads}, "regimes": regime["regimes"],
                            "enabled": regime["enabled"]})
        aggregate = _aggregate(reports, "arcs")
        assert aggregate["enabled_songs"] == {"attach": 1}
        assert aggregate["enabled_only"]["attach"]["true_positive"] == 1.
        assert aggregate["enabled_only"]["attach"]["false_positive"] == 0.
        assert aggregate["heads"]["attach"]["false_positive"] == 1.
        assert aggregate["regimes"]["start"]["attach"]["reference_positives"] == 1.
        assert aggregate["regimes"]["continuation"]["attach"]["reference_positives"] == 0.
