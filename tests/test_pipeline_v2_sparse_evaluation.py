"""Synthetic reporting checks; no corpus comparison, real model, or training."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from training.pipeline_v2 import calibration
from training.pipeline_v2.arc_policy import ARC_POLICY_V2, build_split_overlay
from training.pipeline_v2.basic_lighting import POLICY_DEFAULTS
from training.pipeline_v2.contracts import StageExamples, V2Song
from training.pipeline_v2.evaluation import ReferenceBaseline, evaluate_stage, fit_reference_baselines
from training.pipeline_v2.hazard_sections import candidates, policy_defaults


VOCAB = {"multiplier": [{"kind": "none"}],
         "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 5}],
         "lighting": [{"type": 0, "value": 1}, {"type": 0, "value": 3}]}


def _options():
    return {"history_length": 4, "lighting_block_length": 2, "arc_shape_bounds": [0., 2.],
            "stage_policies": {"arcs": deepcopy(ARC_POLICY_V2), "hazards": policy_defaults(),
                               "lighting": {**POLICY_DEFAULTS, "schedule": {"unit": "beats", "step": .5}}}}


def _song(stage, split, examples, *, identity=None, bpm=120., duration=1.25):
    identity = identity or split
    metadata = {"split": split, "group_id": f"recording-{identity}", "duration": duration, "bpm": bpm,
                "provenance": {"adapter": "v2_synthetic_fixture"}, "sources": []}
    base = SimpleNamespace(song_id=f"song-{identity}", metadata=metadata, arrays={})
    objects = {family: [] for family in ("notes", "walls", "bombs", "arcs", "chains", "lighting", "ignored")}
    objects[{"hazards": "walls", "arcs": "arcs", "lighting": "lighting"}[stage]] = [{"id": "source-positive"}]
    return V2Song(base, objects, {stage: deepcopy(examples)}, diagnostics={"targets": {}})


def _dataset(stage, examples, *, bpm=120., duration=1.25):
    return SimpleNamespace(songs=[_song(stage, split, examples, bpm=bpm, duration=duration) for split in ("train", "validation", "test")],
                           vocabularies=deepcopy(VOCAB), manifest={"status": "structural_dummy", "data_settings": _options()})


def _flat_inputs(count, width):
    return {"audio": np.zeros((count, 16), np.float32), "history": np.zeros((count, 4, width), np.float32),
            "history_mask": np.zeros((count, 4), bool), "history_age": np.zeros((count, 4, 1), np.float32),
            "conditions": np.zeros((count, 6), np.float32)}


def _hazards(*, duration=1.25, bpm=120.):
    options = _options()
    queries = candidates(duration, bpm, options)
    count = len(queries)
    inputs = _flat_inputs(count, 13)
    inputs["interval"] = np.asarray([[query["time"], query["end_time"]] for query in queries], np.float32)
    target = np.zeros(count, np.int32)
    target[2] = 1
    bombs = np.zeros((count, 12), np.float32)
    bombs[1, 3] = 1
    weights = {"wall": np.asarray([query["kind"] == "wall" for query in queries], np.float32),
               "bombs": np.broadcast_to(np.asarray([query["kind"] == "bomb" for query in queries])[:, None], bombs.shape).astype(np.float32)}
    return StageExamples(inputs, {"wall": target, "bombs": bombs}, weights,
                         np.asarray([query["time"] for query in queries], np.float64))


def _lighting():
    timeline = np.asarray([[0., .25], [.5, -1.]])
    mask = timeline >= 0
    inputs = {"audio": np.zeros((2, 2, 16, 1), np.float32), "gaps": np.zeros((2, 2, 1), np.float32),
              "history": np.zeros((2, 2, 2), np.float32), "history_mask": np.asarray([[False, False], [True, True]]), "mask": mask}
    inputs["history"][1, :, 0] = 1
    target = np.zeros((2, 2, 2), np.float32)
    target[..., 0] = mask
    weights = np.broadcast_to(mask[..., None], target.shape).astype(np.float32).copy()
    return StageExamples(inputs, {"events": target}, {"events": weights}, timeline)


def _arcs():
    return StageExamples(_flat_inputs(2, 4),
                         {"attach": np.asarray([1, 0], np.int32), "shape": np.ones((2, 2), np.float32),
                          "anchor": np.zeros(2, np.int32)},
                         {"attach": np.ones(2, np.float32), "shape": np.asarray([1., 0.], np.float32),
                          "anchor": np.asarray([1., 0.], np.float32)}, np.asarray([0., .25]))


@pytest.fixture(autouse=True)
def fixed_reference_context(monkeypatch):
    # Production decoders still execute; only unrelated source-note expansion is
    # replaced by explicit upstream reference notes for these reporting fixtures.
    notes = [{"id": index, "time": .25 * index, "hand": 0, "x": 1, "y": 1, "d": 8}
             for index in range(3)]
    monkeypatch.setattr(calibration, "_reference_state", lambda *args, **kwargs: {"notes": deepcopy(notes), "chains": [], "walls": [], "bombs": [], "arcs": []})


def _empty(stage):
    def predict(inputs):
        batch = len(inputs["audio"])
        if stage == "lighting":
            return {"events": np.zeros((batch, inputs["audio"].shape[1], 2), np.float32)}
        if stage == "hazards":
            return {"wall": np.broadcast_to([1., 0.], (batch, 2)).copy(), "bombs": np.zeros((batch, 12), np.float32)}
        return {"attach": np.broadcast_to([1., 0.], (batch, 2)).copy(), "shape": np.ones((batch, 2), np.float32),
                "anchor": np.broadcast_to([1., 0., 0.], (batch, 3)).copy()}
    return predict


@pytest.mark.parametrize("stage,examples", [("hazards", _hazards()), ("lighting", _lighting()), ("arcs", _arcs())])
def test_empty_sparse_predictions_explicitly_fail_supported_positive_cases(stage, examples):
    dataset = _dataset(stage, examples)
    report = evaluate_stage(dataset, stage, _empty(stage), allow_synthetic=True, batch_size=2)
    assert report["optimizer_steps"] == 0
    assert report["all_empty_failure"]
    assert report["positive_case_status"] == "failed_all_empty"
    for conditioning in ("teacher_forced", "autoregressive"):
        model = report["decoded_scores"][conditioning]["model"]
        assert model["all_empty_failure"]
        assert model["objective"]["reference_positives"] > 0
        assert model["objective"]["predicted_positives"] == 0
        assert model["objective"]["f1"] == 0
    if stage == "hazards":
        assert report["scores"]["model"]["wall"]["accuracy"] == .9
        assert report["positive_sensitive_scores"]["model"]["wall"]["recall"] == 0


def test_lighting_raw_threshold_scores_are_separate_from_actual_type_projection():
    dataset = _dataset("lighting", _lighting())

    def predictor(inputs):
        return {"events": np.broadcast_to([.8, .9], (len(inputs["audio"]), 2, 2)).copy()}

    report = evaluate_stage(dataset, "lighting", predictor, allow_synthetic=True, batch_size=1)
    assert report["positive_sensitive_scores"]["model"]["events"]["true_positive"] == 3
    for mode in report["decoded_scores"].values():
        decoded = mode["model"]["heads"]["events"]
        assert decoded["true_positive"] == 0
        assert decoded["false_positive"] == decoded["false_negative"] == 3
        assert decoded["f1"] == 0


def test_autoregressive_scores_use_own_accepted_history_and_preserve_source_inputs():
    dataset = _dataset("lighting", _lighting())
    before = deepcopy(dataset.songs[-1].examples["lighting"])
    observed_batches = []

    def predictor(inputs):
        observed_batches.append(len(inputs["audio"]))
        result = np.zeros((len(inputs["audio"]), 2, 2), np.float32)
        result[:, :, 0] = np.any(inputs["history"] > 0, axis=(1, 2))[:, None] * .9
        return {"events": result}

    report = evaluate_stage(dataset, "lighting", predictor, allow_synthetic=True, batch_size=1)
    assert max(observed_batches) == 1
    teacher = report["decoded_scores"]["teacher_forced"]["model"]["objective"]
    autoregressive = report["decoded_scores"]["autoregressive"]["model"]["objective"]
    assert teacher["true_positive"] == 1
    assert autoregressive["true_positive"] == 0
    assert autoregressive["all_empty_failure"]
    for key, array in before.inputs.items():
        np.testing.assert_array_equal(array, dataset.songs[-1].examples["lighting"].inputs[key])


def test_hazard_decoded_metrics_score_merged_duration_rejection_separately_from_raw():
    dataset = _dataset("hazards", _hazards(duration=.25, bpm=240.), duration=.25, bpm=240.)

    def predictor(inputs):
        wall = np.isclose(inputs["interval"][:, 0], .0625) & (inputs["interval"][:, 1] > inputs["interval"][:, 0])
        return {"wall": np.stack([~wall, wall], axis=-1).astype(np.float32), "bombs": np.zeros((len(wall), 12), np.float32)}

    report = evaluate_stage(dataset, "hazards", predictor, allow_synthetic=True)
    assert report["positive_sensitive_scores"]["model"]["wall"]["f1"] == 1.
    assert report["decoded_scores"]["teacher_forced"]["model"]["heads"]["wall"]["f1"] == 0.
    assert report["songs"][0]["decoded_scores"]["model"]["autoregressive"]["diagnostics"]["final_rejections"] == {"short_merged_wall": 1}


def test_arc_overlay_selects_effective_holdout_before_lazy_arrays_and_exposes_upstream_training():
    songs = [_song("arcs", split, _arcs(), identity=identity)
             for identity, split in (("a", "train"), ("b", "train"), ("c", "train"), ("v", "validation"), ("t", "test"))]
    records = [{"song_id": song.base.song_id, **song.base.metadata} for song in songs]
    overlay = build_split_overlay([{**record, "positive_count": 1} for record in records])
    opened = []

    class LazySongs:
        def __init__(self):
            self.records = records

        def __getitem__(self, index):
            assert index != 4, "Validation evaluation opened TEST arrays"
            opened.append(index)
            return songs[index]

    dataset = SimpleNamespace(songs=LazySongs(), vocabularies=deepcopy(VOCAB),
                              manifest={"status": "structural_dummy", "data_settings": _options(), "stage_splits": {"arcs": overlay}})
    report = evaluate_stage(dataset, "arcs", _empty("arcs"), split="validation", allow_synthetic=True)
    assert set(report["baseline"]["training_song_ids"]) == set(overlay["splits"]["train"])
    assert {song["song_id"] for song in report["songs"]} == set(overlay["splits"]["validation"])
    assert len(report["songs"]) == 3
    assert report["stage_split_fingerprint"] == overlay["fingerprint"]
    assert sum(song["upstream_training_exposure"] for song in report["songs"]) == 2
    assert set(report["upstream_exposure"]["group_ids"]) == set(overlay["moved_group_ids"])
    assert 4 not in opened


def test_lazy_selected_source_must_match_saved_record_identity():
    dataset = _dataset("lighting", _lighting())
    original = dataset.songs

    class LazySongs:
        records = [{"song_id": song.base.song_id, **song.base.metadata, "decoded_sha256": "saved-audio"} for song in original]

        def __getitem__(self, index):
            return original[index]

    dataset.songs = LazySongs()
    with pytest.raises(ValueError, match="identity disagrees"):
        evaluate_stage(dataset, "lighting", _empty("lighting"), allow_synthetic=True)


def test_missing_supported_positives_are_reported_without_claiming_an_empty_success():
    examples = _lighting()
    examples.targets["events"][:] = 0
    dataset = _dataset("lighting", examples)
    for song in dataset.songs:
        song.diagnostics["targets"]["unknown_lighting"] = ["source-positive"]
        song.objects["ignored"] = [{"id": "unsupported-source", "family": "lighting", "reason": "custom geometry"}]
    report = evaluate_stage(dataset, "lighting", _empty("lighting"), allow_synthetic=True)
    assert report["positive_case_status"] == "no_supported_positive_reference"
    assert report["source_coverage"]["represented_fraction"] == 0.
    assert report["source_coverage"]["reference_objects"] == report["source_coverage"]["unrepresented_objects"] == 2
    assert not report["decoded_scores"]["autoregressive"]["model"]["objective"]["positive_cases_detected"]


def test_same_type_lighting_collisions_remain_raw_source_misses_after_masking():
    examples = _lighting()
    examples.targets["events"][0, 0] = 0
    examples.weights["events"][0, 0] = 0
    dataset = _dataset("lighting", examples)
    for song in dataset.songs:
        song.objects["lighting"] = [{"id": identity} for identity in ("first", "second", "supported")]
        song.diagnostics["targets"]["lighting_collisions"] = [
            {"query": 0, "time": 0., "type": 0, "source_ids": ["first", "second"]}]
    report = evaluate_stage(dataset, "lighting", _empty("lighting"), allow_synthetic=True)
    assert report["source_coverage"]["reference_objects"] == 3
    assert report["source_coverage"]["unrepresented_objects"] == 2
    assert report["source_coverage"]["represented_fraction"] == pytest.approx(1 / 3)
    missing = report["songs"][0]["source_coverage"]["unrepresented_records"]
    assert {(record["id"], record["reason"]) for record in missing} == {
        ("first", "same_type_collision"), ("second", "same_type_collision")}


def test_hazard_raw_reporting_includes_rejected_legacy_bombs_and_partial_wall_coverage():
    dataset = _dataset("hazards", _hazards())
    for song in dataset.songs:
        song.objects["walls"] = [{"id": "complete"}, {"id": "partial"}, {"id": "unrepresented"}]
        song.objects["ignored"] = [{"id": "_notes:3", "family": "notes", "raw": {"_type": 3}, "reason": "invalid geometry"}]
        song.diagnostics["targets"].update({"unscheduled_walls": ["unrepresented"],
            "hazard_raw_source_counts": {"walls": 3, "bombs": 1},
            "hazard_supported_reference": {"wall_source_objects": 2, "fully_supported_wall_source_objects": 1}})
    report = evaluate_stage(dataset, "hazards", _empty("hazards"), allow_synthetic=True)
    assert report["source_coverage"]["reference_objects"] == 4
    assert report["source_coverage"]["unrepresented_objects"] == 2
    assert report["source_coverage"]["wall_source_coverage"] == {
        "raw_objects": 3, "fully_represented_objects": 1, "partially_represented_objects": 1, "unrepresented_objects": 1}
    missing = report["songs"][0]["source_coverage"]["unrepresented_records"]
    assert any(record["id"] == "_notes:3" and record["family"] == "bombs" for record in missing)


def test_legacy_hazard_evaluation_keeps_augmented_query_api_and_baselines():
    from training.pipeline_v2.data import make_dummy_dataset

    dataset = make_dummy_dataset(settings={"stage_policies": {}})
    fitted = fit_reference_baselines(dataset, "hazards")
    predictor = ReferenceBaseline("hazards", dataset.vocabularies, fitted)
    report = evaluate_stage(dataset, "hazards", predictor, batch_size=2, allow_synthetic=True)
    assert report["scores"]["model"] == report["scores"]["frequency"]
    assert set(report["scheduler_only_scores"]) == {"model", "frequency", "none"}
    assert sum(report["hazard_query_counts"].values()) == report["examples"]
    assert "human" in report["hazard_query_conditioning"]
    assert "decoded_scores" not in report
