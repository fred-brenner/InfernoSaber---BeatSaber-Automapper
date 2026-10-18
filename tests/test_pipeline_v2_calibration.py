"""Synthetic calibration checks: decoded feedback, held-out reads, no training."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from training.pipeline_v2 import calibration
from training.pipeline_v2.arc_policy import ARC_POLICY_V2
from training.pipeline_v2.basic_lighting import POLICY_DEFAULTS
from training.pipeline_v2.contracts import StageExamples
from training.pipeline_v2.hazard_sections import candidates, policy_defaults
from training.pipeline_v2.policies import split_identity


def _options():
    return {"history_length": 4, "lighting_block_length": 2, "arc_shape_bounds": [0., 2.],
            "stage_policies": {"arcs": deepcopy(ARC_POLICY_V2), "hazards": policy_defaults(),
                               "lighting": {**POLICY_DEFAULTS, "schedule": {"unit": "beats", "step": .5}}}}


def _song(stage, examples, duration=2.5):
    base = SimpleNamespace(song_id="validation-song", metadata={"split": "validation", "group_id": "validation-recording",
                                                               "duration": duration, "bpm": 120.})
    return SimpleNamespace(base=base, examples={stage: examples}, objects={})


def _lighting_song():
    timeline = np.asarray([[0., .5], [1., 1.5], [2., -1.]])
    mask = timeline >= 0
    inputs = {"audio": np.zeros((3, 2, 16, 1), np.float32), "gaps": np.zeros((3, 2, 1), np.float32),
              "history": np.zeros((3, 2, 2), np.float32), "history_mask": np.zeros((3, 2), bool), "mask": mask}
    target = np.zeros((3, 2, 2), np.float32)
    target[..., 0] = mask
    weights = np.broadcast_to(mask[..., None], target.shape).astype(np.float32).copy()
    return _song("lighting", StageExamples(inputs, {"events": target}, {"events": weights}, timeline))


VOCAB = {"lighting": [{"type": 0, "value": 1}, {"type": 0, "value": 2}],
         "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3}],
         "multiplier": [{"kind": "none"}]}


def test_lighting_replay_feeds_projected_own_history_and_ignores_padding():
    song, inputs_seen = _lighting_song(), []

    def predictor(inputs):
        inputs_seen.append(deepcopy(inputs))
        return {"events": np.broadcast_to([.8, .7], (1, 2, 2)).copy()}

    before = deepcopy(song.examples["lighting"].inputs)
    report = calibration.replay_stage(song, "lighting", predictor, VOCAB, _options(), .5)
    assert not inputs_seen[0]["history_mask"].any()
    np.testing.assert_array_equal(inputs_seen[1]["history"][0], [[1, 0], [1, 0]])
    assert inputs_seen[1]["history_mask"].all()
    assert report["heads"]["events"]["f1"] == 1.
    assert len(report["objects"]["lighting"]) == 5
    assert all(event["value"] == 1 for event in report["objects"]["lighting"])
    for key in before:
        np.testing.assert_array_equal(song.examples["lighting"].inputs[key], before[key])


def test_teacher_forced_projection_scores_colliding_type_values_after_actual_decoder():
    song = _lighting_song()
    outputs = {"events": np.broadcast_to([.8, .7], (3, 2, 2)).copy()}
    report = calibration.replay_stage(song, "lighting", None, VOCAB, _options(), .5,
                                      conditioning="teacher_forced", outputs=outputs)
    assert report["objective"]["true_positive"] == 5
    assert report["objective"]["false_positive"] == 0
    assert report["objective"]["f1"] == 1


def test_class_masks_exclude_unknown_values_from_decoded_positive_metrics():
    song = _lighting_song()
    examples = song.examples["lighting"]
    # Only the supported second type contributes; the first type remains a
    # causal decoded decision despite its zero supervision weight.
    vocabulary = {**VOCAB, "lighting": [{"type": 0, "value": 1}, {"type": 1, "value": 2}]}
    examples.targets["events"].fill(0)
    examples.targets["events"][..., 1] = examples.inputs["mask"]
    examples.weights["events"][..., 0] = 0
    outputs = {"events": np.broadcast_to([.9, .8], (3, 2, 2)).copy()}
    report = calibration.replay_stage(song, "lighting", None, vocabulary, _options(), .5,
                                      conditioning="teacher_forced", outputs=outputs, retain_objects=False)
    assert report["objective"]["reference_positives"] == 5
    assert report["objective"]["true_positive"] == 5
    assert report["objective"]["false_positive"] == 0
    assert report["diagnostics"]["events"] == 10


def test_arc_replay_none_history_is_zero_and_equal_time_queries_are_causal(monkeypatch):
    notes = [{"id": 0, "time": 0., "hand": 0}, {"id": 1, "time": .3, "hand": 0},
             {"id": 2, "time": .6, "hand": 0}, {"id": 3, "time": 0., "hand": 1},
             {"id": 4, "time": .3, "hand": 1}]
    timeline = np.asarray([0., 0., .3])
    inputs = {"audio": np.zeros((3, 16), np.float32), "history": np.zeros((3, 4, 4), np.float32),
              "history_mask": np.zeros((3, 4), bool), "history_age": np.zeros((3, 4, 1), np.float32),
              "conditions": np.zeros((3, 6), np.float32)}
    examples = StageExamples(inputs, {"attach": np.zeros(3, np.int32), "shape": np.zeros((3, 2), np.float32),
                                      "anchor": np.zeros(3, np.int32)},
                             {"attach": np.ones(3), "shape": np.zeros(3), "anchor": np.zeros(3)}, timeline)
    song, seen = _song("arcs", examples), []
    monkeypatch.setattr(calibration, "_reference_state", lambda *args, **kwargs: {"notes": deepcopy(notes), "arcs": []})

    def predictor(values):
        seen.append(deepcopy(values))
        return {"attach": np.asarray([[.8, .2]]), "shape": np.asarray([[1.7, 1.4]]), "anchor": np.asarray([[0., 0., 1.]])}

    report = calibration.replay_stage(song, "arcs", predictor, VOCAB, _options(), .5)
    assert not seen[0]["history_mask"].any()
    assert not seen[1]["history_mask"].any()
    assert seen[2]["history_mask"].sum() == 2
    assert np.all(seen[2]["history"] == 0)
    assert report["objects"]["arcs"] == []


def test_arc_rejected_proposals_do_not_feed_unused_geometry_and_probability_is_disabled(monkeypatch):
    notes = [{"id": 0, "time": 0., "hand": 0}, {"id": 1, "time": .3, "hand": 0}, {"id": 2, "time": .6, "hand": 0}]
    timeline = np.asarray([0., .3])
    inputs = {"audio": np.zeros((2, 16), np.float32), "history": np.zeros((2, 4, 4), np.float32),
              "history_mask": np.zeros((2, 4), bool), "history_age": np.zeros((2, 4, 1), np.float32),
              "conditions": np.zeros((2, 6), np.float32)}
    examples = StageExamples(inputs, {"attach": np.ones(2, np.int32), "shape": np.ones((2, 2), np.float32),
                                      "anchor": np.zeros(2, np.int32)}, {"attach": np.ones(2), "shape": np.ones(2), "anchor": np.ones(2)}, timeline)
    song, seen = _song("arcs", examples), []
    monkeypatch.setattr(calibration, "_reference_state", lambda *args, **kwargs: {"notes": deepcopy(notes), "arcs": []})
    monkeypatch.setattr(calibration, "_arc_validator", lambda state, *args: state.update(arcs=[]))

    def predictor(values):
        seen.append(deepcopy(values))
        return {"attach": np.asarray([[.1, .9]]), "shape": np.asarray([[1.7, 1.4]]), "anchor": np.asarray([[0., 0., 1.]])}

    report = calibration.replay_stage(song, "arcs", predictor, VOCAB, _options(), .5)
    assert seen[1]["history_mask"].sum() == 1
    assert np.all(seen[1]["history"] == 0)
    assert report["diagnostics"]["clearance_rejections"] == 2
    assert report["diagnostics"]["slider_probability"] == 1.


def test_hazard_calibration_leaves_wall_argmax_and_scores_final_cleared_objects(monkeypatch):
    options = _options()
    queries = candidates(.25, 120., options)
    count = len(queries)
    inputs = {"audio": np.zeros((count, 16), np.float32), "history": np.zeros((count, 4, 13), np.float32),
              "history_mask": np.zeros((count, 4), bool), "history_age": np.zeros((count, 4, 1), np.float32),
              "conditions": np.zeros((count, 6), np.float32)}
    target = np.zeros(count, np.int32)
    target[::2] = 1
    bomb_target = np.zeros((count, 12), np.float32)
    bomb_target[1::2, 3] = 1
    weights = {"wall": np.asarray([query["kind"] == "wall" for query in queries], np.float32),
               "bombs": np.broadcast_to(np.asarray([query["kind"] == "bomb" for query in queries])[:, None], bomb_target.shape).astype(np.float32)}
    examples = StageExamples(inputs, {"wall": target, "bombs": bomb_target}, weights, np.asarray([query["time"] for query in queries]))
    song = _song("hazards", examples, duration=.25)
    monkeypatch.setattr(calibration, "_reference_state", lambda *args, **kwargs: {"notes": [], "chains": []})
    output = {"wall": np.broadcast_to([.4, .6], (count, 2)).copy(), "bombs": np.zeros((count, 12))}
    output["bombs"][:, 3] = .8
    low = calibration.replay_stage(song, "hazards", None, VOCAB, options, .2, conditioning="teacher_forced", outputs=output)
    high = calibration.replay_stage(song, "hazards", None, VOCAB, options, .9, conditioning="teacher_forced", outputs=output)
    assert low["heads"]["wall"] == high["heads"]["wall"]
    assert low["heads"]["wall"]["f1"] == 1.
    assert low["objective"]["f1"] == 1.
    assert high["objective"]["f1"] == 0.


def test_calibration_selects_validation_before_loading_arrays_and_creates_bound_sidecar(tmp_path, monkeypatch):
    song = _lighting_song()
    read = []

    class Songs:
        records = [{"song_id": song.base.song_id, **song.base.metadata},
                   {"song_id": "never-open-test", "group_id": "test", "split": "test"},
                   {"song_id": "never-open-train", "group_id": "train", "split": "train"}]

        def __getitem__(self, index):
            assert index == 0, "Calibration opened a non-validation song"
            read.append(index)
            return song

    dataset = SimpleNamespace(songs=Songs(), vocabularies=VOCAB,
                              manifest={"status": "prepared", "data_settings": _options(), "songs": Songs.records})

    def model(inputs, training=False):
        assert training is False
        return {"events": np.broadcast_to([.6, .2], (len(inputs["audio"]), 2, 2)).copy()}

    loaded = []
    metadata = {"status": "trained", "checkpoint_sha256": "a" * 64,
                "training": {"split_identity": split_identity(dataset.manifest, "lighting")}}

    def load(path, stage, selected):
        loaded.append((path, stage, selected))
        return model, metadata

    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: (None, metadata))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact", load)
    report = calibration.calibrate_stage(dataset, "lighting", "selected-artifact", tmp_path / "calibration")
    assert loaded == [("selected-artifact", "lighting", dataset)]
    assert len(read) == 4  # One teacher-forced scan plus three own-stage replays.
    assert report["threshold"] == .6
    assert len(report["teacher_forced_ranking"]) == 99
    assert [row["threshold"] for row in report["autoregressive_shortlist"]] == [.6, .59, .58]
    assert report["validation_song_ids"] == [song.base.song_id]
    assert report["validation_group_ids"] == ["validation-recording"]
    assert report["checkpoint_sha256"] == "a" * 64
    assert report["optimizer_steps"] == 0
    assert report["positive_cases_detected"]
    assert not report["gameplay_validated"]
    assert (tmp_path / "calibration" / "calibration.json").is_file()


def test_autoregressive_reranking_can_change_teacher_forced_winner(tmp_path, monkeypatch):
    song = _lighting_song()
    dataset = SimpleNamespace(songs=[song], vocabularies=VOCAB,
                              manifest={"status": "prepared", "data_settings": _options(),
                                        "songs": [{"song_id": song.base.song_id, **song.base.metadata}]})
    metadata = {"status": "trained", "checkpoint_sha256": "b" * 64,
                "training": {"split_identity": split_identity(dataset.manifest, "lighting")}}
    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: (None, metadata))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact",
                        lambda *args: (lambda *args, **kwargs: {}, metadata))
    monkeypatch.setattr(calibration, "teacher_forced_outputs", lambda *args: {})

    def replay(*args, conditioning="autoregressive", **kwargs):
        threshold = args[5]
        perfect = conditioning == "teacher_forced" or threshold < .99
        scores = calibration._statistics(1. if perfect else 0., 0., 0. if perfect else 1.)
        return {"heads": {"events": scores}, "objects": {}}

    monkeypatch.setattr(calibration, "replay_stage", replay)
    report = calibration.calibrate_stage(dataset, "lighting", "artifact", tmp_path / "reranked")
    assert report["teacher_forced_ranking"][0]["threshold"] == .99
    assert report["threshold"] == .98
    assert {row["threshold"] for row in report["autoregressive_shortlist"]} == {.97, .98, .99}


def test_real_model_calibration_reuses_compiled_inference_for_autoregressive_passes(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    song = _lighting_song()
    dataset = SimpleNamespace(songs=[song], vocabularies=VOCAB,
                              manifest={"status": "prepared", "data_settings": _options(),
                                        "songs": [{"song_id": song.base.song_id, **song.base.metadata}]})
    metadata = {"status": "trained", "checkpoint_sha256": "c" * 64,
                "training": {"split_identity": split_identity(dataset.manifest, "lighting")}}

    class InferenceModel(tf.keras.Model):
        traces = 0

        def call(self, inputs, training=None):
            assert training is False
            self.traces += 1
            batch = tf.shape(inputs["audio"])[0]
            return {"events": tf.broadcast_to(tf.constant([.6, .2]), [batch, 2, 2])}

    model = InferenceModel()
    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: (None, metadata))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact", lambda *args: (model, metadata))
    report = calibration.calibrate_stage(dataset, "lighting", "synthetic-artifact", tmp_path / "compiled")
    assert report["runtime"]["compiled_calls"] is True
    assert report["threshold"] == .6
    assert model.traces <= 4  # Initial/generalized traces, reused for all three AR passes.


def test_calibration_rejects_untrained_artifacts_before_validation_reads(tmp_path, monkeypatch):
    dataset = SimpleNamespace(songs=[], vocabularies=VOCAB,
                              manifest={"status": "prepared", "data_settings": _options()})
    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: (None, {"status": "untrained"}))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact", lambda *args: pytest.fail("Untrained model loaded"))
    with pytest.raises(ValueError, match="trained stage artifact"):
        calibration.calibrate_stage(dataset, "lighting", "untrained", tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_disabled_arc_generation_rejects_calibration_before_loading_model(tmp_path, monkeypatch):
    from tools.config import config

    dataset = SimpleNamespace(songs=[], vocabularies=VOCAB,
                              manifest={"status": "prepared", "data_settings": _options()})
    monkeypatch.setattr(config, "add_slider_flag", False)
    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: pytest.fail("Disabled arcs opened artifact"))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact", lambda *args: pytest.fail("Disabled arcs loaded model"))
    with pytest.raises(ValueError, match="add_slider_flag enabled"):
        calibration.calibrate_stage(dataset, "arcs", "artifact", tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_arc_clearance_controls_are_recorded_without_restricting_later_user_controls(tmp_path, monkeypatch):
    from tools.config import config
    from training.pipeline_v2.artifacts import compatibility

    song = _song("arcs", SimpleNamespace(times=np.asarray([.2])))
    dataset = SimpleNamespace(songs=[song], vocabularies=VOCAB,
                              manifest={"status": "prepared", "data_settings": _options(),
                                        "songs": [{"song_id": song.base.song_id, **song.base.metadata}]})
    metadata = {"status": "trained", "checkpoint_sha256": "d" * 64, "compatibility": compatibility(dataset, "arcs"),
                "training": {"split_identity": split_identity(dataset.manifest, "arcs")}}
    monkeypatch.setattr(config, "add_slider_flag", True)
    monkeypatch.setattr(config, "slider_movement_minimum", 4.)
    monkeypatch.setattr(config, "min_time_diff", .03)
    monkeypatch.setattr(config, "slider_probability", .25)
    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: (None, metadata))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact", lambda *args: (lambda *args, **kwargs: {}, metadata))
    monkeypatch.setattr(calibration, "teacher_forced_outputs", lambda *args: {})
    monkeypatch.setattr(calibration, "_reference_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(calibration, "replay_stage", lambda *args, **kwargs: {
        "heads": {"attach": calibration._statistics(1., 0., 0.)}, "objects": {}})
    output = tmp_path / "arc-calibration"
    report = calibration.calibrate_stage(dataset, "arcs", "synthetic-artifact", output)
    controls = report["reproducibility"]["arc_clearance_controls"]
    assert controls == {"add_slider_flag": True, "slider_movement_minimum": 4., "min_time_diff": .03,
                        "slider_probability": 1., "configured_slider_probability": .25,
                        "probability_filter": "disabled during calibration"}
    assert config.slider_probability == .25
    monkeypatch.setattr(config, "add_slider_flag", False)
    monkeypatch.setattr(config, "slider_movement_minimum", 1.)
    monkeypatch.setattr(config, "min_time_diff", .1)
    monkeypatch.setattr(config, "slider_probability", .9)
    assert calibration.load_calibration(output, "arcs", metadata) == report


@pytest.mark.parametrize("change", ["split", "group", "audio", "examples", "missing_provenance"])
def test_calibration_rejects_compatible_artifact_with_different_training_split_before_model_load(tmp_path, monkeypatch, change):
    song = _lighting_song()
    records = [{"song_id": song.base.song_id, **song.base.metadata, "audio_sha256": "recorded-audio", "sha256": "prepared-array-hash"}]
    manifest = {"status": "prepared", "data_settings": _options(), "songs": records}
    recorded = split_identity(manifest, "lighting")
    metadata = {"status": "trained", "checkpoint_sha256": "a" * 64, "training": {"split_identity": recorded}}
    dataset = SimpleNamespace(songs=[], vocabularies=VOCAB, manifest=deepcopy(manifest))
    if change == "split":
        dataset.manifest["songs"][0]["split"] = "train"
    elif change == "group":
        dataset.manifest["songs"][0]["group_id"] = "another-recording"
    elif change == "audio":
        dataset.manifest["songs"][0]["audio_sha256"] = "another-audio"
    elif change == "examples":
        dataset.manifest["songs"][0]["sha256"] = "changed-prepared-examples"
    else:
        metadata["training"] = {}
    monkeypatch.setattr(calibration, "load_checkpoint_manifest", lambda *args: (None, metadata))
    monkeypatch.setattr("training.pipeline_v2.runner.load_stage_artifact", lambda *args: pytest.fail("Mismatched model loaded"))
    with pytest.raises(ValueError, match="training split"):
        calibration.calibrate_stage(dataset, "lighting", "artifact", tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_nonfinite_or_invalid_probabilities_are_rejected():
    song = _lighting_song()
    for probability in (float("nan"), -1., 2.):
        outputs = {"events": np.full((3, 2, 2), probability)}
        with pytest.raises(ValueError, match="finite|probabilities"):
            calibration.replay_stage(song, "lighting", None, VOCAB, _options(), .5,
                                      conditioning="teacher_forced", outputs=outputs)


def test_replay_history_keeps_full_preceding_context_when_a_sibling_has_equal_time():
    inputs = {"history": np.zeros((1, 2, 13), np.float32), "history_mask": np.zeros((1, 2), bool),
              "history_age": np.zeros((1, 2, 1), np.float32), "conditions": np.zeros((1, 6), np.float32)}
    examples = StageExamples(inputs, {}, {}, np.asarray([1.]))
    history = [{"time": .25, "value": [1, *([0] * 12)]},
               {"time": .5, "value": [2, *([0] * 12)]},
               {"time": 1., "value": [3, *([0] * 12)]}]
    selected = calibration._history_inputs(examples, 0, "hazards", history, _options())
    assert selected["history_mask"].all()
    np.testing.assert_array_equal(selected["history"][0, :, 0], [1, 2])
    np.testing.assert_allclose(selected["history_age"][0, :, 0], [.75, .5])


def test_arc_calibration_uses_effective_group_overlay_without_opening_test_or_retained_train():
    from training.pipeline_v2.arc_policy import build_split_overlay

    summaries = [{"song_id": f"song-{group}-{variant}", "group_id": f"group-{group}", "split": "train",
                  "positive_count": int(variant == 0), "audio_sha256": f"audio-{group}"}
                 for group in range(4) for variant in range(2)]
    summaries.extend([{"song_id": "original-validation", "group_id": "validation", "split": "validation", "positive_count": 0},
                      {"song_id": "test", "group_id": "test", "split": "test", "positive_count": 1}])
    overlay = build_split_overlay(summaries)
    expected = [item["song_id"] for item in summaries if overlay["records"][item["song_id"]] == "validation"]
    opened = []

    class Songs:
        records = summaries

        def __getitem__(self, index):
            record = summaries[index]
            assert record["song_id"] in expected
            opened.append(record["song_id"])
            return SimpleNamespace(base=SimpleNamespace(song_id=record["song_id"], metadata=record))

    dataset = SimpleNamespace(songs=Songs(), manifest={"stage_splits": {"arcs": overlay}})
    selected = list(calibration._validation_songs(dataset, "arcs"))
    assert opened == expected
    assert len(selected) == 5
    assert sum(song.base.metadata["split"] == "train" for song in selected) == 4


def test_calibration_rejects_indexed_audio_alias_drift_and_cross_split_leakage():
    song = _lighting_song()

    class Songs:
        records = [{"song_id": song.base.song_id, **song.base.metadata, "audio_sha256": "recorded-audio"}]

        def __getitem__(self, index):
            return song

    dataset = SimpleNamespace(songs=Songs(), manifest={})
    with pytest.raises(ValueError, match="metadata disagrees"):
        list(calibration._validation_songs(dataset, "lighting"))
    Songs.records.append({"song_id": "test", "group_id": "test", "split": "test", "audio_sha256": "recorded-audio"})
    with pytest.raises(ValueError, match="identity leaks"):
        list(calibration._validation_songs(dataset, "lighting"))


def _sidecar(tmp_path):
    from training.beat_models.artifacts import statistics_hash, write_json
    from training.pipeline_v2.artifacts import compatibility

    dataset = SimpleNamespace(vocabularies=VOCAB, manifest={"data_settings": _options(), "songs": [
        {"song_id": "train", "group_id": "train", "split": "train"},
        {"song_id": "validation", "group_id": "validation", "split": "validation"}]})
    identity = compatibility(dataset, "lighting")
    indexed_split = split_identity(dataset.manifest, "lighting")
    artifact = {"status": "trained", "checkpoint_sha256": "a" * 64, "compatibility": identity,
                "training": {"split_identity": indexed_split}}
    report = {"kind": "pipeline_v2_stage_calibration", "version": 1, "stage": "lighting", "status": "calibrated",
              "fit_split": "validation", "optimizer_steps": 0, "threshold": .6, "thresholds": {"events": .6},
              "checkpoint_sha256": artifact["checkpoint_sha256"], "compatibility": identity, "split_identity": indexed_split,
              "validation_song_ids": ["validation"], "validation_group_ids": ["validation"]}
    report["fingerprint"] = statistics_hash(report)
    path = tmp_path / "calibration.json"
    write_json(path, report)
    return path, report, artifact


def test_calibration_sidecar_is_bound_to_exact_checkpoint_and_semantic_contract(tmp_path):
    path, report, artifact = _sidecar(tmp_path)
    assert calibration.load_calibration(path, "lighting", artifact) == report
    assert calibration.load_calibration(tmp_path, "lighting", artifact) == report
    with pytest.raises(ValueError, match="another trained checkpoint"):
        calibration.load_calibration(path, "lighting", {**artifact, "checkpoint_sha256": "b" * 64})
    changed = deepcopy(artifact)
    changed["compatibility"]["stage_data"]["policy"]["schedule"]["step"] = .25
    with pytest.raises(ValueError, match="compatibility mismatch"):
        calibration.load_calibration(path, "lighting", changed)


def test_sidecar_tampering_is_rejected_before_threshold_use(tmp_path):
    import json

    path, report, artifact = _sidecar(tmp_path)
    report["threshold"] = .2
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        calibration.load_calibration(path, "lighting", artifact)


def test_sidecar_training_split_and_arc_overlay_provenance_must_match_checkpoint(tmp_path):
    path, report, artifact = _sidecar(tmp_path)
    changed = deepcopy(artifact)
    changed["training"]["split_identity"]["records"][0]["split"] = "test"
    with pytest.raises(ValueError, match="training split identity"):
        calibration.load_calibration(path, "lighting", changed)
    changed = deepcopy(artifact)
    changed["training"]["stage_split"] = {"fingerprint": "different-overlay"}
    with pytest.raises(ValueError, match="arc overlay"):
        calibration.load_calibration(path, "lighting", changed)


def test_sidecar_declared_validation_records_cannot_name_training_records(tmp_path):
    import json

    from training.beat_models.artifacts import statistics_hash

    path, report, artifact = _sidecar(tmp_path)
    report.pop("fingerprint")
    report["validation_song_ids"] = ["train"]
    report["validation_group_ids"] = ["train"]
    report["fingerprint"] = statistics_hash(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its validation split"):
        calibration.load_calibration(path, "lighting", artifact)


@pytest.mark.parametrize("field,value", [("kind", "other"), ("version", 2), ("version", True),
                                         ("fit_split", "test"), ("status", "trained"),
                                         ("optimizer_steps", 1), ("threshold", 0), ("threshold", 1)])
def test_intact_but_invalid_sidecar_contract_is_rejected(tmp_path, field, value):
    import json

    from training.beat_models.artifacts import statistics_hash

    path, report, artifact = _sidecar(tmp_path)
    report.pop("fingerprint")
    report[field] = value
    report["fingerprint"] = statistics_hash(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar|threshold"):
        calibration.load_calibration(path, "lighting", artifact)
