"""Explicit immutable assembly of calibrated sparse-stage inference bundles."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from training.beat_models.artifacts import (file_hash, json_value, new_run_directory,
                                           statistics_hash, write_json)

from .artifacts import load_checkpoint_manifest
from .calibration import load_calibration
from .policies import SPARSE_STAGES, semantic_identity
from .settings import STAGES


def _policy_settings(settings, identity):
    """Merge one frozen stage contract; final checks protect shared settings."""
    stage = identity["stage"]
    data = semantic_identity(identity)["stage_data"]
    if data.get("policy_version") != 2 or data.get("policy", {}).get("version") != 2:
        raise ValueError("Bundle replacements require a repaired sparse-stage policy")
    settings.setdefault("stage_policies", {})[stage] = deepcopy(data["policy"])
    for key, value in data.items():
        schedule_key = {"occupancy_clearance_seconds": "clearance_seconds", "lighting_onset_threshold": "lighting_onset_threshold"}.get(key)
        if schedule_key is not None:
            settings.setdefault("schedule", {})[schedule_key] = deepcopy(value)
        elif key not in ("policy", "policy_version", "schedule"):
            settings[key] = deepcopy(value)


def _merge_vocabulary(vocabularies, manifest):
    family = {"hazards": "walls", "lighting": "lighting"}.get(manifest["stage"])
    if family is None:
        return
    source = manifest["vocabularies"]
    vocabularies[family] = deepcopy(source[family])
    if "counts" in vocabularies and family in source.get("counts", {}):
        vocabularies["counts"][family] = deepcopy(source["counts"][family])
    if "training_song_ids" in vocabularies:
        vocabularies["training_song_ids"] = sorted(set(vocabularies["training_song_ids"]) | set(source.get("training_song_ids", [])))
    if "fingerprint" in vocabularies:
        payload = {key: value for key, value in vocabularies.items() if key != "fingerprint"}
        vocabularies["fingerprint"] = statistics_hash(payload)


def _copy(source, root, relative):
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def assemble_bundle(base_bundle, replacements, calibrations, output):
    """Copy a validated base into a fresh bundle with explicit stage replacements.

    Only hazards, arcs and lighting may be replaced. Each replacement requires
    its matching validation calibration. The five upstream models, retained
    encoder and normalization remain byte-for-byte copies of the base assets.
    No TensorFlow model is loaded, and no configured/current bundle is changed.
    """
    from map_creation.v2_runtime import _local, validate_bundle

    if not isinstance(replacements, dict) or not replacements or set(replacements) - set(SPARSE_STAGES):
        raise ValueError("Select explicit hazards, arcs or lighting bundle replacements")
    if not isinstance(calibrations, dict) or set(calibrations) != set(replacements):
        raise ValueError("Each repaired bundle replacement requires its validation calibration")
    supplied = Path(base_bundle)
    base = validate_bundle(supplied.parent if supplied.is_file() and supplied.name == "bundle.json" else supplied)
    base_root = base["root"]
    destination = Path(output).resolve()
    if destination.exists() or destination.is_relative_to(base_root):
        raise ValueError("Bundle assembly requires a new output directory outside the immutable base bundle")
    options = deepcopy(base["bundle"]["data_settings"])
    vocabularies = deepcopy(base["vocabularies"])
    selected, sidecars = {}, {}
    for stage, path in replacements.items():
        root, manifest = load_checkpoint_manifest(path, stage)
        if manifest.get("status") != "trained" or manifest.get("training", {}).get("status", "trained") != "trained":
            raise ValueError("Bundle replacements must be explicitly trained artifacts")
        _policy_settings(options, manifest["compatibility"])
        _merge_vocabulary(vocabularies, manifest)
        sidecars[stage] = load_calibration(calibrations[stage], stage, manifest)
        selected[stage] = root, manifest
    context = {"beat_representation": base["manifests"]["timing"]["representation"],
               "normalization_sha256": base["bundle"]["normalization_sha256"],
               "encoder_identity": base["encoder_identity"], "data_settings": options,
               "vocabularies": vocabularies}
    # Check every downstream artifact against the combined context before any
    # files are written. This also catches expansion/encoder changes that would
    # invalidate an unchanged upstream stage or another replacement.
    for stage in STAGES:
        root = selected[stage][0] if stage in selected else _local(base_root, f"{base['bundle']['model_stages'][stage]}/manifest.json").parent
        load_checkpoint_manifest(root, stage, context)
    root = new_run_directory(destination)
    stages = {stage: stage for stage in ("intensity", "timing", *STAGES)}
    for stage in stages:
        source_root, manifest = selected[stage] if stage in selected else (
            _local(base_root, f"{base['bundle']['model_stages'][stage]}/manifest.json").parent, base["manifests"][stage])
        _copy(source_root / "manifest.json", root, f"{stage}/manifest.json")
        _copy(source_root / manifest["checkpoint"], root, f"{stage}/{manifest['checkpoint']}")
    encoder = _local(base_root, base["bundle"]["encoder"])
    encoder_relative = f"encoder/{encoder.name}"
    _copy(encoder, root, encoder_relative)
    _copy(encoder.parent / "manifest.json", root, "encoder/manifest.json")
    _copy(_local(base_root, "normalization.json"), root, "normalization.json")
    write_json(root / "vocabularies.json", vocabularies)
    calibration_files = {}
    for stage in sorted(set(base["bundle"].get("calibrations", {})) | set(calibrations)):
        supplied_sidecar = Path(calibrations[stage]) if stage in calibrations else _local(base_root, base["bundle"]["calibrations"][stage])
        source = supplied_sidecar / "calibration.json" if supplied_sidecar.is_dir() else supplied_sidecar
        relative = f"calibrations/{stage}.json"
        _copy(source, root, relative)
        calibration_files[stage] = relative
    document = deepcopy(base["bundle"])
    document.update(model_stages=stages, encoder=encoder_relative, data_settings=options,
                    vocabularies_sha256=statistics_hash(vocabularies), calibrations=calibration_files,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    assembly={"base_bundle_sha256": file_hash(base_root / "bundle.json"),
                              "replacements": {stage: {"checkpoint_sha256": manifest["checkpoint_sha256"],
                                                       "calibration_fingerprint": sidecars[stage]["fingerprint"]}
                                               for stage, (_, manifest) in selected.items()},
                              "optimizer_steps": 0, "promotion": "explicit_new_directory_only"},
                    files={path.relative_to(root).as_posix(): {"sha256": file_hash(path)}
                           for path in sorted(root.rglob("*")) if path.is_file()})
    write_json(root / "bundle.json", document)
    validate_bundle(root)
    return json_value(document)
