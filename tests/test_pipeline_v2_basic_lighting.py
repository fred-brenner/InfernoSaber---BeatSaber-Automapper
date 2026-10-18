"""Synthetic basic lighting eligibility, scheduling and causal supervision."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from training.pipeline_v2 import basic_lighting as lighting
from training.pipeline_v2.contracts import V2Song
from training.pipeline_v2.data import read_objects


def _base(events, *, duration=1.1, split="train", identity="train", bpm=120.):
    frames = np.arange(0., duration, .01)
    return SimpleNamespace(song_id=identity, metadata={"duration": duration, "bpm": bpm, "split": split,
                           "raw_beatmap": {"version": "3.2.0", "basicBeatmapEvents": deepcopy(events)}},
                           arrays={"features": np.zeros((len(frames), 8), np.float32), "frame_times": frames,
                                   "original_notes": np.empty((0, 8)), "retained_indices": np.empty(0, np.int64),
                                   "original_event_indices": np.empty(0, np.int64)})


def _song(events, **kwargs):
    base = _base(events, **kwargs)
    objects, diagnostics = read_objects(base)
    return V2Song(base, objects, diagnostics=diagnostics)


def _event(time, kind=0, value=1, brightness=1, **extra):
    return {"b": time * 2, "et": kind, "i": value, "f": brightness, **extra}


def _options(schedule=None):
    policy = {**lighting.POLICY_DEFAULTS, "schedule": schedule or {"unit": "beats", "step": .5}}
    return {"lighting_block_length": 3, "stage_policies": {"lighting": policy}}


@pytest.mark.parametrize("kind,value,expected", [(0, 0, 0), (4, 12, 12), (8, -17, 0), (9, 77, 0), (12, 0, 0), (13, 15, 15)])
def test_classifier_canonical_basic_profile(kind, value, expected):
    template, reason = lighting.classify(_event(0, kind, value))
    assert reason is None
    assert template == {"type": kind, "value": expected}


@pytest.mark.parametrize("event,reason", [(_event(0, 0, 13), "unsupported_event_value"),
                                         (_event(0, 12, 16), "unsupported_event_value"),
                                         (_event(0, 5), "unsupported_event_type"),
                                         (_event(0, 0, 2000000001), "chroma_encoded_value"),
                                         (_event(0, 0, 1, .5), "unsupported_float_value"),
                                         (_event(0, customData={"lightID": [0]}), "custom_data_dependent"),
                                         (_event(0, _customData={"_color": [1, 0, 0]}), "custom_data_dependent")])
def test_classifier_preserves_unsupported_data_without_clamping(event, reason):
    before = deepcopy(event)
    assert lighting.classify(event) == (None, reason)
    assert event == before


def test_empty_custom_data_and_v2_aliases_are_supported():
    assert lighting.classify({"_type": 2, "_value": 4, "_customData": {}}) == ({"type": 2, "value": 4}, None)
    assert lighting.classify(_event(0, customData={"unused": [], "nested": {}}))[1] is None


def test_decoder_projects_same_type_but_keeps_simultaneous_types_and_off():
    vocabulary = [{"type": 0, "value": 0}, {"type": 0, "value": 1}, {"type": 1, "value": 2}]
    scores = np.asarray([[[.9, .8, .7], [.1, .2, .3]]], np.float32)
    np.testing.assert_array_equal(lighting.decode(scores, vocabulary, .5), [[[1, 0, 1], [0, 0, 0]]])
    assert lighting.decode_events(scores[0, 0], vocabulary) == [vocabulary[0], vocabulary[2]]
    np.testing.assert_array_equal(lighting.decode([.8, .8, .1], vocabulary), [1, 0, 0])


def test_scheduler_retains_onsets_and_partial_tail_independent_of_notes():
    base = _base([], duration=.263)
    base.arrays["features"][11, -1] = 2
    queries = lighting.candidates(base, _options())
    times = np.asarray([query["time"] for query in queries])
    np.testing.assert_allclose(times, [0., .11, .25, np.nextafter(.263, 0)])
    assert times[-1] < .263
    changed = deepcopy(base)
    changed.arrays["original_notes"] = np.ones((100, 8))
    assert lighting.candidates(changed, _options()) == queries
    options = _options()
    options["stage_policies"]["lighting"]["max_candidates"] = 3
    with pytest.raises(ValueError, match="max_candidates"):
        lighting.candidates(base, options)


def test_fit_reads_each_train_once_and_uses_minimum_queries_not_candidate_order(monkeypatch):
    bases = [_base([_event(.04)], duration=1., identity="train"),
             _base([_event(.1, 4, 12)], duration=1., split="validation", identity="validation")]
    import training.pipeline_v2.data as data

    calls = []
    original = data.read_objects

    def read(base):
        calls.append(base.song_id)
        return original(base)

    monkeypatch.setattr(data, "read_objects", read)
    policy, vocabulary, report = lighting.fit_policy(bases, {})
    assert calls == ["train"]
    assert report["training_song_ids"] == ["train"]
    assert vocabulary == [{"type": 0, "value": 1}]
    assert policy["schedule"] == {"unit": "seconds", "step": .04}
    assert report["selected"]["coverage"] == 1
    assert len(report["candidate_schedules"]) == 7
    assert report["selected"]["queries"] == min(item["queries"] for item in report["candidate_schedules"] if item["coverage"] >= .95)


def test_preflight_reports_source_and_schedule_progress():
    progress = []
    report = lighting.preflight([_base([_event(0)])], {}, progress=progress.append)
    assert report["status"] == "ready"
    assert {event["phase"] for event in progress} == {"lighting_sources", "lighting_schedule"}
    assert all(event["elapsed_seconds"] >= 0 for event in progress)
    assert progress[0]["song_id"] == "train"


def test_collision_coverage_requires_finer_grid():
    base = _base([_event(.023, 0, 1), _event(.043, 0, 2)], duration=1.)
    policy, vocabulary, report = lighting.fit_policy([base], {})
    assert policy["schedule"] == {"unit": "seconds", "step": .02}
    assert len(vocabulary) == 2
    assert report["selected"]["effective_events"] == 2
    coarser = next(item for item in report["candidate_schedules"] if item["schedule"] == {"unit": "seconds", "step": .04})
    assert coarser["coverage"] == 0
    assert coarser["collision_queries"] == 1
    assert coarser["collision_source_events"] == 2


def test_fitting_drops_classes_with_no_effective_positive_weight():
    events = [_event(index * .25) for index in range(40)]
    events += [_event(0, 1, 1), _event(0, 1, 9)]
    _, vocabulary, report = lighting.fit_policy([_base(events, duration=10.)], {})
    assert vocabulary == [{"type": 0, "value": 1}]
    assert report["dropped_zero_positive_classes"] == [{"type": 1, "value": 1}, {"type": 1, "value": 9}]
    assert report["selected"]["coverage"] == pytest.approx(40 / 42)


def test_unrepresentable_conflicts_fail_coverage_instead_of_inventing_class():
    base = _base([_event(0, 0, 1), _event(0, 0, 2)])
    report = lighting.preflight([base], {})
    assert report["status"] == "coverage_unmet"
    with pytest.raises(ValueError, match="effective TRAIN coverage"):
        lighting.fit_policy([base], {})


def test_class_type_masks_preserve_supported_history_and_every_none_query():
    events = [_event(0), _event(0, 1, 2, .5),
              _event(.25, 0, 1), _event(.25, 0, 2), _event(.25, 1, 2),
              _event(.5), _event(.5, "unknown", 2)]
    song = _song(events)
    vocabulary = [{"type": 0, "value": 1}, {"type": 0, "value": 2}, {"type": 1, "value": 2}]
    examples, report = lighting.build_examples(song, {}, vocabulary, _options())
    assert examples.weights["events"].shape == examples.targets["events"].shape == (2, 3, 3)
    np.testing.assert_array_equal(examples.weights["events"][0], [[1, 1, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(examples.targets["events"][0], [[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(examples.inputs["history"][1], examples.targets["events"][0])
    assert examples.inputs["history_mask"][1].all()
    assert examples.inputs["mask"].all()
    assert report["lighting_history_queries"] == 6
    assert len(report["lighting_collisions"]) == 1


def test_unknown_known_type_masks_only_its_own_classes_and_incomplete_sources_have_history():
    song = _song([_event(0), _event(0, 1, 12)])
    song.diagnostics["source_objects_complete"] = False
    vocabulary = [{"type": 0, "value": 1}, {"type": 1, "value": 2}]
    examples, report = lighting.build_examples(song, {}, vocabulary, _options())
    assert not examples.weights["events"].any()
    assert examples.inputs["history_mask"][1].all()
    np.testing.assert_array_equal(examples.inputs["history"][1, 0], [1, 0])
    assert report["unknown_lighting"] == ["basicBeatmapEvents:1"]


def test_padding_has_no_loss_weight_or_history_record():
    song = _song([_event(0)], duration=1.01)
    options = _options()
    options["lighting_block_length"] = 4
    examples, report = lighting.build_examples(song, {}, [{"type": 0, "value": 1}], options)
    assert examples.inputs["mask"].sum() == report["lighting_history_queries"] == 6
    assert examples.times[-1, -1] == -1
    assert not examples.weights["events"][-1, 2:].any()
    assert examples.inputs["history_mask"][-1].sum() == 4


def test_preparation_freezes_policy_and_audits_effective_support_without_changing_sources():
    from training.pipeline_v2.data import prepare_dataset
    from training.pipeline_v2.fixtures import make_dummy_songs

    bases = make_dummy_songs()
    before = deepcopy([base.metadata for base in bases])
    options = _options()
    original_options = deepcopy(options)
    result = prepare_dataset(bases, settings=options)
    assert options == original_options
    assert [base.metadata for base in bases] == before
    audit = result.manifest["preparation_audit"]
    assert audit["lighting"]["status"] == "ready"
    assert result.manifest["data_settings"]["stage_policies"]["lighting"] == audit["lighting"]["policy"]
    assert set(audit["support"]) == set(result.songs[0].examples)
    for split, heads in audit["support"]["lighting"].items():
        songs = [song for song in result.songs if song.base.metadata["split"] == split]
        expected = sum((song.examples["lighting"].targets["events"] * song.examples["lighting"].weights["events"]).sum(axis=(0, 1))
                       for song in songs)
        np.testing.assert_array_equal(heads["events"]["positive"], expected)
    assert all(record["object_counts"]["lighting"] == 3 for record in audit["songs"])


def test_preparation_failure_carries_reviewable_lighting_audit():
    from training.pipeline_v2.data import prepare_dataset
    from training.pipeline_v2.fixtures import make_dummy_songs

    bases = make_dummy_songs()
    for base in bases:
        base.metadata["raw_beatmap"]["basicBeatmapEvents"] = [_event(0, 0, 1), _event(0, 0, 2)]
    emitted = []
    with pytest.raises(ValueError, match="effective TRAIN coverage") as caught:
        prepare_dataset(bases, settings=_options(), emit=emitted.append)
    assert caught.value.preparation_audit["lighting"]["status"] == "coverage_unmet"
    assert len(caught.value.preparation_audit["lighting"]["candidate_schedules"]) == 7
    assert emitted == []


def test_frozen_evaluation_reuses_policy_without_refitting_or_arc_overlay(monkeypatch):
    from training.pipeline_v2.data import make_dummy_dataset, prepare_dataset
    import training.pipeline_v2.arc_policy as arcs

    training = make_dummy_dataset()
    base = deepcopy(training.songs[0].base)
    base.metadata["split"] = "test"

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation attempted to refit a training contract")

    monkeypatch.setattr(lighting, "preflight", forbidden)
    monkeypatch.setattr(arcs, "build_split_overlay", forbidden)
    result = prepare_dataset([base], settings=training.manifest["data_settings"], vocabularies=training.vocabularies)
    assert result.manifest["data_settings"] == training.manifest["data_settings"]
    assert result.manifest["stage_splits"] == {}
    assert result.manifest["preparation_audit"]["lighting"] is None


def test_prewarm_batches_all_missing_encoder_queries_before_stage_builders():
    from training.pipeline_v2.data import prepare_dataset
    from training.pipeline_v2.fixtures import make_dummy_songs

    class Encoder:
        metadata = {"encoder_fingerprint": "synthetic-cache"}

        def __init__(self):
            self.cached, self.missing_batches, self.calls = set(), [], []

        def __call__(self, times):
            times = np.asarray(times)
            missing = [float(time) for time in times if float(time) not in self.cached]
            self.calls.append(times.copy())
            if missing:
                self.missing_batches.append(missing)
                self.cached.update(missing)
            return np.zeros((len(times), 16), np.float32)

    bases = make_dummy_songs()
    providers = {base.song_id: Encoder() for base in bases}
    options = _options()
    options["encoder_batch_size"] = 8
    progress = []
    result = prepare_dataset(bases, settings=options, encoder=providers, progress=progress.append)
    completed = [event for event in progress if event.get("phase") == "examples"]
    assert len(completed) == len(result.songs) == len(bases)
    assert {event["phase"] for event in progress} >= {"lighting_sources", "lighting_schedule", "vocabulary", "examples"}
    for song in result.songs:
        provider = providers[song.base.song_id]
        count = song.diagnostics["encoder_prewarm"]["query_times"]
        assert sum(map(len, provider.missing_batches)) == count
        assert max(map(len, provider.missing_batches)) <= 8
        assert len(provider.calls[0]) == 8
    assert progress[-1]["index"] == len(bases)
    assert progress[-1]["elapsed_seconds"] >= 0


def test_explicit_settings_without_stage_policies_preserve_legacy_preparation():
    from training.pipeline_v2.data import prepare_dataset
    from training.pipeline_v2.fixtures import make_dummy_songs

    result = prepare_dataset(make_dummy_songs(), settings={})
    assert "stage_policies" not in result.manifest["data_settings"]
    assert result.manifest["stage_splits"] == {}
    assert result.manifest["preparation_audit"]["lighting"] is None
    assert result.songs[0].examples["lighting"].weights["events"].ndim == 2
    assert result.songs[0].diagnostics["targets"]["hazard_query_policy"] == "source_positive_queries_plus_inference_scheduler_queries"


def test_preparation_audit_keeps_hazard_exclusions_arc_ids_and_nested_rejections():
    from training.pipeline_v2.contracts import StageExamples
    from training.pipeline_v2.data import _preparation_record

    song = _song([])
    song.examples["arcs"] = StageExamples({}, {"attach": np.asarray([0], np.int32)},
                                        {"attach": np.ones(1, np.float32)}, np.asarray([0.], np.float64))
    song.diagnostics["targets"] = {
        "hazard_source_exclusions": [{"id": "wall:0", "family": "walls", "reason": "section_clearance"}],
        "hazard_reference_rollout": {"section_rejections": {"note_or_chain_collision": 3},
                                     "final_rejections": {"short_merged_wall": 2}},
        "unscheduled_arcs": ["sliders:0"]}
    record = _preparation_record(song)
    assert record["unsupported_reasons"] == {"section_clearance": 1, "unscheduled_arcs": 1}
    assert {item["id"] for item in record["source_exclusions"]} == {"wall:0", "sliders:0"}
    assert record["rejection_counts"]["targets.hazard_reference_rollout.section_rejections"] == {"note_or_chain_collision": 3}
    assert record["rejection_counts"]["targets.hazard_reference_rollout.final_rejections"] == {"short_merged_wall": 2}
