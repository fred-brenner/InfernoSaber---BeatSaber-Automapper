"""Standalone immutable bundle assembly using non-executable synthetic assets."""

from copy import deepcopy
import json
import pytest

from map_creation.v2_runtime import validate_bundle
from training.beat_models.artifacts import file_hash, statistics_hash, write_json
from training.beat_models.contracts import FEATURES, REPRESENTATION
from training.beat_models.settings import architecture_settings as beat_architecture
from training.pipeline_v2.arc_policy import ARC_POLICY_V2
from training.pipeline_v2.artifacts import compatibility
from training.pipeline_v2.basic_lighting import POLICY_DEFAULTS
from training.pipeline_v2.bundle import assemble_bundle
from training.pipeline_v2.encoder import ENCODER_REPRESENTATION
from training.pipeline_v2.hazard_sections import policy_defaults
from training.pipeline_v2.policies import semantic_identity, split_identity
from training.pipeline_v2.settings import STAGES, preparation_settings


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _fixture(root):
    root.mkdir()
    stats = {"version": 1, "fit_split": "train", "song_ids": ["synthetic-training-song"],
             **{key: {"mean": [0.] * FEATURES, "scale": [1.] * FEATURES, "count": 1}
                for key in ("features", "coarse_features")}}
    vocab = {"multiplier": [{"kind": "none"}], "walls": [{"kind": "none"}],
             "lighting": [{"type": 0, "value": 1}]}
    _write(root / "normalization.json", stats)
    _write(root / "vocabularies.json", vocab)
    (root / "encoder").mkdir()
    (root / "encoder/encoder.h5").write_bytes(b"synthetic retained encoder; no executable model")
    encoder_hash = file_hash(root / "encoder/encoder.h5")
    encoder_identity = {"status": "retained", "checkpoint_fingerprint": encoder_hash,
                        "grid_seconds": .5, "representation": deepcopy(ENCODER_REPRESENTATION)}
    _write(root / "encoder/manifest.json", {"kind": "retained_v2_audio_encoder", "checkpoint_sha256": encoder_hash,
                                           "encoder_identity": encoder_identity})
    options = preparation_settings()
    options.pop("stage_policies", None)
    context = {"beat_representation": REPRESENTATION, "normalization_sha256": statistics_hash(stats),
               "encoder_identity": encoder_identity, "data_settings": options, "vocabularies": vocab}
    stages = {stage: stage for stage in ("intensity", "timing", *STAGES)}
    for stage in stages:
        directory = root / stage
        directory.mkdir()
        (directory / "best.keras").write_bytes(f"synthetic original {stage}".encode())
        manifest = {"schema_version": 1, "stage": stage, "status": "trained", "checkpoint": "best.keras",
                    "checkpoint_sha256": file_hash(directory / "best.keras"), "training": {"status": "trained"}}
        if stage in ("intensity", "timing"):
            manifest.update(kind="model_checkpoint", representation=REPRESENTATION, normalization=stats,
                            normalization_sha256=statistics_hash(stats))
            manifest["training"]["architecture"] = beat_architecture(stage)
        else:
            manifest.update(kind="pipeline_v2_checkpoint", compatibility=compatibility(context, stage), vocabularies=vocab)
        _write(root / stage / "manifest.json", manifest)
    document = {"kind": "infernosaber_v2_inference_bundle", "schema_version": 1, "model_stages": stages,
                "encoder": "encoder/encoder.h5", "data_settings": options,
                "normalization_sha256": statistics_hash(stats), "vocabularies_sha256": statistics_hash(vocab),
                "files": {path.relative_to(root).as_posix(): {"sha256": file_hash(path)}
                          for path in root.rglob("*") if path.is_file()}}
    _write(root / "bundle.json", document)
    return context


def _replacement(root, context, stage, *, mutate=None, calibration_mutate=None):
    root.mkdir()
    (root / "best.keras").write_bytes(f"synthetic repaired {stage}".encode())
    context = deepcopy(context)
    policies = {"hazards": policy_defaults(), "arcs": deepcopy(ARC_POLICY_V2),
                "lighting": {**POLICY_DEFAULTS, "schedule": {"unit": "beats", "step": .25}}}
    context["data_settings"].setdefault("stage_policies", {})[stage] = policies[stage]
    if stage == "hazards":
        context["vocabularies"]["walls"].append({"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3})
    elif stage == "lighting":
        context["vocabularies"]["lighting"] = [{"type": 1, "value": 2}]
    if mutate:
        mutate(context)
    indexed_split = split_identity({"songs": [
        {"song_id": "synthetic-training-song", "group_id": "synthetic-training-recording", "split": "train"},
        {"song_id": "synthetic-validation-song", "group_id": "synthetic-validation-recording", "split": "validation"}]}, stage)
    manifest = {"kind": "pipeline_v2_checkpoint", "schema_version": 1, "stage": stage, "status": "trained",
                "checkpoint": "best.keras", "checkpoint_sha256": file_hash(root / "best.keras"),
                "compatibility": compatibility(context, stage), "vocabularies": context["vocabularies"],
                "training": {"status": "trained", "split_identity": indexed_split}}
    _write(root / "manifest.json", manifest)
    head = {"hazards": "bombs", "arcs": "attach", "lighting": "events"}[stage]
    sidecar = {"kind": "pipeline_v2_stage_calibration", "version": 1, "stage": stage, "status": "calibrated",
               "fit_split": "validation", "optimizer_steps": 0, "threshold": .72, "thresholds": {head: .72},
               "checkpoint_sha256": manifest["checkpoint_sha256"], "compatibility": manifest["compatibility"],
               "validation_song_ids": ["synthetic-validation-song"], "validation_group_ids": ["synthetic-validation-recording"],
               "split_identity": indexed_split}
    if calibration_mutate:
        calibration_mutate(sidecar)
    sidecar["fingerprint"] = statistics_hash(sidecar)
    _write(root / "calibration.json", sidecar)
    return root, root / "calibration.json"


def _hashes(root):
    return {path.relative_to(root).as_posix(): file_hash(path) for path in root.rglob("*") if path.is_file()}


def test_assembly_merges_repaired_contracts_and_preserves_all_upstream_assets(tmp_path):
    base = tmp_path / "base"
    context = _fixture(base)
    before = _hashes(base)
    replacements, calibrations = {}, {}
    for stage in ("hazards", "arcs", "lighting"):
        replacements[stage], calibrations[stage] = _replacement(tmp_path / f"repaired-{stage}", context, stage)
    output = tmp_path / "new-bundle"
    document = assemble_bundle(base, replacements, calibrations, output)
    assert _hashes(base) == before
    assert document["calibrations"] == {stage: f"calibrations/{stage}.json" for stage in replacements}
    assert set(document["assembly"]["replacements"]) == set(replacements)
    for stage in ("intensity", "timing", "positions", "arrows", "multiplier"):
        assert file_hash(output / stage / "best.keras") == file_hash(base / stage / "best.keras")
        assert file_hash(output / stage / "manifest.json") == file_hash(base / stage / "manifest.json")
    assert file_hash(output / "encoder/encoder.h5") == file_hash(base / "encoder/encoder.h5")
    assert file_hash(output / "encoder/manifest.json") == file_hash(base / "encoder/manifest.json")
    assert file_hash(output / "normalization.json") == file_hash(base / "normalization.json")
    for stage in replacements:
        assert file_hash(output / stage / "best.keras") == file_hash(replacements[stage] / "best.keras")
    validated = validate_bundle(output)
    assert len(validated["vocabularies"]["walls"]) == 2
    assert validated["vocabularies"]["lighting"] == [{"type": 1, "value": 2}]
    assert validated["vocabularies"]["multiplier"] == context["vocabularies"]["multiplier"]
    assert all(validated["bundle"]["data_settings"]["stage_policies"][stage]["version"] == 2 for stage in replacements)
    assert set(document["files"]) == set(_hashes(output)) - {"bundle.json"}
    for filename, record in document["files"].items():
        assert file_hash(output / filename) == record["sha256"]
    relocated = tmp_path / "relocated"
    output.rename(relocated)
    validate_bundle(relocated)


def test_incremental_assembly_preserves_previously_calibrated_stages(tmp_path):
    base = tmp_path / "base"
    context = _fixture(base)
    hazard, hazard_calibration = _replacement(tmp_path / "hazards", context, "hazards")
    first = tmp_path / "first"
    assemble_bundle(base, {"hazards": hazard}, {"hazards": hazard_calibration}, first)
    lighting, lighting_calibration = _replacement(tmp_path / "lighting", context, "lighting")
    final = tmp_path / "second"
    assembled = assemble_bundle(first, {"lighting": lighting}, {"lighting": lighting_calibration}, final)
    assert set(assembled["calibrations"]) == {"hazards", "lighting"}
    assert file_hash(first / "calibrations/hazards.json") == file_hash(final / "calibrations/hazards.json")
    assert file_hash(first / "hazards/best.keras") == file_hash(final / "hazards/best.keras")
    validate_bundle(final)


def test_hazard_occupancy_radius_is_merged_into_scheduler_settings(tmp_path):
    base = tmp_path / "base"
    context = _fixture(base)
    artifact, sidecar = _replacement(
        tmp_path / "hazards", context, "hazards",
        mutate=lambda value: value["data_settings"]["schedule"].update(clearance_seconds=.25))
    output = tmp_path / "new-bundle"
    document = assemble_bundle(base, {"hazards": artifact}, {"hazards": sidecar}, output)
    assert document["data_settings"]["schedule"]["clearance_seconds"] == .25
    assert "occupancy_clearance_seconds" not in document["data_settings"]
    validate_bundle(output)


def test_lighting_onset_threshold_is_merged_into_scheduler_settings(tmp_path):
    base = tmp_path / "base"
    context = _fixture(base)
    artifact, sidecar = _replacement(
        tmp_path / "lighting", context, "lighting",
        mutate=lambda value: value["data_settings"]["schedule"].update(lighting_onset_threshold=.7))
    output = tmp_path / "new-bundle"
    document = assemble_bundle(base, {"lighting": artifact}, {"lighting": sidecar}, output)
    assert document["data_settings"]["schedule"]["lighting_onset_threshold"] == .7
    assert "lighting_onset_threshold" not in document["data_settings"]
    validate_bundle(output)


@pytest.mark.parametrize("stage,field", [("hazards", "occupancy"), ("arcs", "endpoint_matching")])
def test_semantic_compatibility_covers_feature_radius_and_arc_endpoint_matching(stage, field):
    options = preparation_settings()
    vocab = {"walls": [{"kind": "none"}], "lighting": [{"type": 0, "value": 1}], "multiplier": [{"kind": "none"}]}
    context = {"data_settings": options, "vocabularies": vocab}
    changed = deepcopy(context)
    if field == "occupancy":
        changed["data_settings"]["schedule"]["clearance_seconds"] += .05
    else:
        changed["data_settings"]["lighting_match_tolerance"] += .01
    assert semantic_identity(compatibility(context, stage)) != semantic_identity(compatibility(changed, stage))


@pytest.mark.parametrize("changed", ["normalization", "encoder", "expansion"])
def test_incompatible_replacement_is_rejected_before_creating_output(tmp_path, changed):
    base = tmp_path / "base"
    context = _fixture(base)

    def mutate(value):
        if changed == "normalization":
            value["normalization_sha256"] = "different-normalization"
        elif changed == "encoder":
            value["encoder_identity"]["checkpoint_fingerprint"] = "different-encoder"
        else:
            value["data_settings"]["max_copies"] += 1

    artifact, sidecar = _replacement(tmp_path / "artifact", context, "hazards", mutate=mutate)
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="encoder, vocabulary or dataset contract"):
        assemble_bundle(base, {"hazards": artifact}, {"hazards": sidecar}, output)
    assert not output.exists()


@pytest.mark.parametrize("field,value,reason", [("checkpoint_sha256", "wrong-checkpoint", "another trained checkpoint"),
                                                ("fit_split", "test", "validation calibration sidecar")])
def test_wrong_checkpoint_or_test_calibration_is_rejected(tmp_path, field, value, reason):
    base = tmp_path / "base"
    context = _fixture(base)
    artifact, sidecar = _replacement(tmp_path / "artifact", context, "arcs",
                                     calibration_mutate=lambda report: report.update({field: value}))
    with pytest.raises(ValueError, match=reason):
        assemble_bundle(base, {"arcs": artifact}, {"arcs": sidecar}, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_new_directory_and_explicit_calibration_are_required(tmp_path):
    base = tmp_path / "base"
    context = _fixture(base)
    artifact, sidecar = _replacement(tmp_path / "artifact", context, "lighting")
    before = _hashes(base)
    for output in (base, base / "nested-output"):
        with pytest.raises(ValueError, match="new output directory"):
            assemble_bundle(base, {"lighting": artifact}, {"lighting": sidecar}, output)
    with pytest.raises(ValueError, match="requires its validation calibration"):
        assemble_bundle(base, {"lighting": artifact}, {}, tmp_path / "no-calibration")
    with pytest.raises(ValueError, match="explicit hazards, arcs or lighting"):
        assemble_bundle(base, {"positions": artifact}, {"positions": sidecar}, tmp_path / "upstream-change")
    assert _hashes(base) == before


def test_untrained_or_legacy_replacements_are_rejected(tmp_path):
    base = tmp_path / "base"
    context = _fixture(base)
    artifact, sidecar = _replacement(tmp_path / "artifact", context, "lighting")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "untrained"
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly trained"):
        assemble_bundle(base, {"lighting": artifact}, {"lighting": sidecar}, tmp_path / "untrained")
    with pytest.raises(ValueError, match="repaired sparse-stage policy"):
        assemble_bundle(base, {"lighting": base / "lighting"}, {"lighting": sidecar}, tmp_path / "legacy")
