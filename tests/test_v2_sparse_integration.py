"""Repaired sparse stages compose/export without changing upstream contracts."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_export import serialize_beatmap
from map_creation.v2_pipeline import generate
from map_creation.v2_runtime import validate_bundle
from tools.config import config
from training.beat_models.artifacts import file_hash, statistics_hash
from training.beat_models.contracts import CORE, REPRESENTATION, Song
from training.beat_models.settings import architecture_settings as beat_architecture
from training.pipeline_v2.artifacts import compatibility, load_checkpoint_manifest
from training.pipeline_v2.encoder import ENCODER_REPRESENTATION
from training.pipeline_v2.pipeline import dummy_predictors
from training.pipeline_v2.policies import semantic_identity
from training.pipeline_v2.settings import STAGES, preparation_settings


def _song(duration=1.6):
    frames, bins = int(round(duration * 100)), int(np.ceil(duration))
    return Song("sparse-integration", {"duration": duration, "bpm": 120.,
                "raw_beatmap": {"untouched": ["source labels are irrelevant to production"]}},
                {"features": np.zeros((frames, 82), np.float32),
                 "coarse_features": np.zeros((bins, 82), np.float32),
                 "frame_times": np.arange(frames, dtype=np.float64) / 100,
                 "bin_durations": np.minimum(1., duration - np.arange(bins)).astype(np.float32)})


def _vocab():
    return {"multiplier": [{"kind": "none"}],
            "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 5}],
            "lighting": [{"type": 0, "value": 1}, {"type": 0, "value": 3}, {"type": 1, "value": 2}]}


class _Predictor:
    status = "dummy"

    def __init__(self, call):
        self.call = call

    def __call__(self, inputs):
        return self.call(inputs)


def _predictors(vocab):
    predictors = dummy_predictors(vocab)

    def timing(inputs):
        values = np.zeros((len(inputs["audio"]), CORE, 3), np.float32)
        values[..., 0] = 1.
        for frame in (50, 75, 100):
            values[:, frame, 0] = 0.
            values[:, frame, 2] = 1.
        return {"counts": values}

    predictors["timing"] = _Predictor(timing)
    predictors["positions"] = _Predictor(lambda inputs: {
        "red": np.eye(13, dtype=np.float32)[[6]], "blue": np.eye(13, dtype=np.float32)[[11]]})
    predictors["arrows"] = _Predictor(lambda inputs: {
        "red": np.eye(9, dtype=np.float32)[[8]], "blue": np.eye(9, dtype=np.float32)[[8]]})
    predictors["lighting"] = _Predictor(lambda inputs: {
        "events": np.broadcast_to(np.array([.8, .9, .7], np.float32),
                                   (len(inputs["audio"]), inputs["audio"].shape[1], 3)).copy()})
    return predictors


def _processors():
    from map_creation.v2_postprocessing import validate_arcs, validate_hazards

    def timing(song, beats, audio_path, logger=None):
        return [{"time": float(time), "count": int(beats["counts"][frame])}
                for frame, time in zip(beats["event_frames"], beats["event_times"])], {}

    return SimpleNamespace(postprocess_timing=timing,
                           postprocess_notes=lambda notes, times, logger=None: (notes, {}),
                           validate_expansions=lambda state, song, logger=None: {},
                           validate_hazards=validate_hazards, validate_arcs=validate_arcs)


def _settings():
    options = preparation_settings()
    assert set(options["stage_policies"]) == {"hazards", "lighting", "arcs"}
    return options


def _generate(song, options):
    vocab = _vocab()
    return generate(song, 4., _predictors(vocab), lambda times: np.zeros((len(times), 16), np.float32),
                    options, vocab, postprocessors=_processors(), random_factor=0)


def test_repaired_production_all_sparse_families_serialize_and_preserve_audio_source(monkeypatch):
    monkeypatch.setattr(config, "slider_probability", 1.)
    monkeypatch.setattr(config, "slider_movement_minimum", 0.)
    monkeypatch.setattr(config, "add_slider_flag", True)
    source = _song()
    before = deepcopy(source)
    result = _generate(source, _settings())
    assert result.diagnostics["production_validated"] and result.diagnostics["export_ready"]
    assert result.diagnostics["gameplay_validated"] is False
    assert all(result.state[family] for family in ("notes", "walls", "bombs", "arcs", "lighting"))
    assert [(item["time"], item["end_time"]) for item in result.state["walls"]] == [(0., 1.6)]
    assert len(result.state["arcs"]) == 4  # .25s gaps, below the former .5s gate
    assert result.stages["hazards"]["wall_section_proposals"] > result.stages["hazards"]["walls"]
    lighting = result.state["lighting"]
    assert {(event["type"], event["value"]) for event in lighting} == {(0, 3), (1, 2)}
    assert len({(event["time"], event["type"]) for event in lighting}) == len(lighting)
    assert max(event["time"] for event in lighting) < 1.6
    state_before = deepcopy(result.state)
    modern, modern_omissions = serialize_beatmap(result.state, 120., "v3")
    legacy, legacy_omissions = serialize_beatmap(result.state, 120., "v2")
    assert len(modern["colorNotes"]) == len(result.state["notes"])
    assert len(modern["obstacles"]) == len(legacy["_obstacles"]) == 1
    assert modern["obstacles"][0]["d"] == legacy["_obstacles"][0]["_duration"] == 3.2
    assert len(modern["bombNotes"]) == sum(note["_type"] == 3 for note in legacy["_notes"])
    assert len(modern["sliders"]) == legacy_omissions["arcs"] == 4
    assert modern_omissions == {"arcs": 0, "chain_bodies": 0, "walls": 0}
    assert legacy_omissions["walls"] == 0
    assert len(modern["basicBeatmapEvents"]) == len(legacy["_events"]) == len(lighting)
    for family in ("notes", "walls", "bombs", "arcs", "lighting", "chains"):
        assert result.state[family] == state_before[family]
    assert source.metadata == before.metadata
    for key, array in source.arrays.items():
        np.testing.assert_array_equal(array, before.arrays[key])


def test_repaired_production_keeps_native_chains_with_all_other_object_families(monkeypatch):
    from map_creation.v2_postprocessing import validate_expansions

    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(config, "slider_probability", 1.)
    monkeypatch.setattr(config, "slider_movement_minimum", 0.)
    monkeypatch.setattr(config, "add_slider_flag", True)
    vocab = _vocab()
    vocab["multiplier"].append({"kind": "chain", "dx": 1, "dy": 0,
                                "span": .1, "slices": 3, "squish": .75})
    processors = _processors()
    processors.validate_expansions = validate_expansions
    result = generate(_song(), 4., _predictors(vocab),
                      lambda times: np.zeros((len(times), 16), np.float32),
                      _settings(), vocab, postprocessors=processors, random_factor=0)
    assert all(result.state[family] for family in ("notes", "chains", "walls", "bombs", "arcs", "lighting"))
    assert len(result.state["chains"]) == 3
    # A chain owns each red swing; only the clear blue attachments survive.
    assert len(result.state["arcs"]) == 2
    assert all(arc["hand"] == 1 for arc in result.state["arcs"])
    modern, omissions = serialize_beatmap(result.state, 120., "v3")
    legacy, legacy_omissions = serialize_beatmap(result.state, 120., "v2")
    assert len(modern["burstSliders"]) == 3 and not any(omissions.values())
    assert legacy_omissions["chain_bodies"] == 3
    assert legacy_omissions["arcs"] == 2
    assert sum(note["_type"] in (0, 1) for note in legacy["_notes"]) == len(modern["colorNotes"]) == 6
    for chain in modern["burstSliders"]:
        assert chain["tb"] - chain["b"] == pytest.approx(.2)
        assert any((note["b"], note["c"], note["x"], note["y"]) ==
                   (chain["b"], chain["c"], chain["x"], chain["y"]) for note in modern["colorNotes"])


def test_repaired_stage_selection_does_not_change_upstream_notes_or_contract_identity(monkeypatch):
    monkeypatch.setattr(config, "slider_probability", 1.)
    monkeypatch.setattr(config, "slider_movement_minimum", 0.)
    frozen, legacy = _settings(), _settings()
    legacy.pop("stage_policies")
    repaired = _generate(_song(), frozen)
    old = _generate(_song(), legacy)
    assert repaired.state["notes"] == old.state["notes"]
    np.testing.assert_array_equal(repaired.state["counts"], old.state["counts"])
    np.testing.assert_array_equal(repaired.state["event_times"], old.state["event_times"])
    context = {"beat_representation": REPRESENTATION, "encoder_identity": {"retained": "unchanged"},
               "normalization_sha256": "same", "vocabularies": _vocab()}
    for stage in ("positions", "arrows", "multiplier"):
        first = compatibility({**context, "data_settings": frozen}, stage)
        second = compatibility({**context, "data_settings": legacy}, stage)
        assert semantic_identity(first) == semantic_identity(second)
    for stage in ("hazards", "arcs", "lighting"):
        assert semantic_identity(compatibility({**context, "data_settings": frozen}, stage)) != semantic_identity(
            compatibility({**context, "data_settings": legacy}, stage))


def test_repaired_decoder_stays_frozen_and_application_controls_report_removals(monkeypatch):
    frozen = _settings()
    assert frozen["stage_policies"]["hazards"]["clearance"]["add_obstacle_flag"]
    monkeypatch.setattr(config, "add_obstacle_flag", False)
    result = _generate(_song(), frozen)
    assert result.diagnostics["hazard_clearance"]["final_walls"] == 1
    assert result.diagnostics["hazard_clearance"]["application_filters"]["rejected_walls"] == 1
    assert result.state["walls"] == []
    assert frozen["stage_policies"]["hazards"]["clearance"]["add_obstacle_flag"] is True


@pytest.mark.parametrize("duration", [.01, .06, .13])
def test_short_audio_partial_sections_and_lighting_tails_export(duration):
    source = _song(duration)
    before = deepcopy(source)
    result = _generate(source, _settings())
    assert result.state["notes"] == []
    assert result.state["arcs"] == []
    assert result.stages["hazards"]["queries"] == 2 * int(np.ceil(duration / .125))
    assert all(wall["end_time"] <= duration for wall in result.state["walls"])
    assert all(0 <= event["time"] < duration for event in result.state["lighting"])
    assert len(result.state["walls"]) == int(duration >= .1)
    for version in ("v2", "v3"):
        document, _ = serialize_beatmap(result.state, 120., version)
        assert document
    assert source.metadata == before.metadata
    for key, array in source.arrays.items():
        np.testing.assert_array_equal(array, before.arrays[key])


def _legacy_bundle(root, policies=None):
    """Non-executable checkpoint fixture using the pre-repair identity format."""
    root.mkdir()

    def write(name, document):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")

    stats = {"version": 1, "fit_split": "train", "song_ids": ["train"],
             **{key: {"mean": [0.] * 82, "scale": [1.] * 82, "count": 1}
                for key in ("features", "coarse_features")}}
    settings, vocab = preparation_settings(), _vocab()
    settings.pop("stage_policies", None)
    if policies:
        settings["stage_policies"] = policies
    write("normalization.json", stats)
    write("vocabularies.json", vocab)
    (root / "encoder").mkdir()
    (root / "encoder/encoder.h5").write_bytes(b"non-executable retained encoder")
    encoder_hash = file_hash(root / "encoder/encoder.h5")
    identity = {"status": "retained", "checkpoint_fingerprint": encoder_hash, "grid_seconds": .5,
                "representation": deepcopy(ENCODER_REPRESENTATION)}
    write("encoder/manifest.json", {"kind": "retained_v2_audio_encoder", "checkpoint_sha256": encoder_hash,
                                    "encoder_identity": identity})
    context = {"beat_representation": REPRESENTATION, "normalization_sha256": statistics_hash(stats),
               "encoder_identity": identity, "data_settings": settings, "vocabularies": vocab}
    stages = {stage: stage for stage in ("intensity", "timing", *STAGES)}
    for stage in stages:
        directory = root / stage
        directory.mkdir()
        (directory / "best.keras").write_bytes(f"non-executable {stage}".encode())
        manifest = {"schema_version": 1, "stage": stage, "status": "trained", "checkpoint": "best.keras",
                    "checkpoint_sha256": file_hash(directory / "best.keras"), "training": {"status": "trained"}}
        if stage in ("intensity", "timing"):
            manifest.update(kind="model_checkpoint", representation=REPRESENTATION, normalization=stats,
                            normalization_sha256=statistics_hash(stats))
            manifest["training"]["architecture"] = beat_architecture(stage)
        else:
            saved = compatibility(context, stage)
            # Old upstream checkpoints recorded the entire dataset settings.
            # Repaired sparse checkpoints retain their new stage-local identity.
            if stage not in (policies or {}):
                saved.pop("stage_data")
                saved.pop("contract_version")
                saved["data_settings"] = deepcopy(settings)
                saved["data_settings"].pop("stage_policies", None)
                saved.pop("sha256")
                saved["sha256"] = statistics_hash(saved)
            manifest.update(kind="pipeline_v2_checkpoint", compatibility=saved, vocabularies=vocab)
        write(f"{stage}/manifest.json", manifest)
    write("bundle.json", {"kind": "infernosaber_v2_inference_bundle", "schema_version": 1,
                          "model_stages": stages, "encoder": "encoder/encoder.h5", "data_settings": settings,
                          "normalization_sha256": statistics_hash(stats), "vocabularies_sha256": statistics_hash(vocab),
                          "files": {str(path.relative_to(root)): {"sha256": file_hash(path)}
                                    for path in root.rglob("*") if path.is_file()}})
    return root, context


def test_legacy_bundle_identity_still_loads_and_checksum_is_enforced(tmp_path):
    root, _ = _legacy_bundle(tmp_path / "legacy")
    result = validate_bundle(root)
    assert "stage_policies" not in result["bundle"]["data_settings"]
    assert all("stage_data" not in result["manifests"][stage]["compatibility"] for stage in STAGES)
    (root / "positions/best.keras").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(root)


def test_mixed_bundle_reuses_legacy_upstreams_and_rejects_wrong_sparse_policy(tmp_path):
    root, context = _legacy_bundle(tmp_path / "mixed", _settings()["stage_policies"])
    result = validate_bundle(root)
    assert result["manifests"]["positions"]["compatibility"].get("contract_version") is None
    assert result["manifests"]["hazards"]["compatibility"]["stage_data"]["policy_version"] == 2
    changed = deepcopy(context)
    changed["data_settings"]["stage_policies"]["hazards"]["grid_beats"] = .5
    load_checkpoint_manifest(root / "positions", "positions", changed)
    with pytest.raises(ValueError, match="dataset contract"):
        load_checkpoint_manifest(root / "hazards", "hazards", changed)
