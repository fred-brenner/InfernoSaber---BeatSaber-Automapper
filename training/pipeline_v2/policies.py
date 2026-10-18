"""Versioned sparse-stage contracts, separate from network architecture."""

from copy import deepcopy

from training.beat_models.artifacts import json_value


SPARSE_STAGES = ("hazards", "lighting", "arcs")


def repaired(options, stage):
    policy = options.get("stage_policies", {}).get(stage, {})
    version = policy.get("version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported {stage} policy version: {version}")
    return version == 2


def preparation_policies():
    from tools.config import config
    from .hazard_sections import policy_defaults

    if not config.pipeline_v2_repair_sparse_stages:
        return {}
    hazards = policy_defaults()
    hazards.update(version=2, grid_beats=config.pipeline_v2_hazard_section_beats,
                   max_candidates=config.pipeline_v2_sparse_max_candidates)
    # Freeze user controls with the data, so supervision and inference agree.
    for key in hazards.get("clearance", {}):
        if hasattr(config, key):
            hazards["clearance"][key] = json_value(getattr(config, key))
    return {"hazards": hazards,
            "lighting": {"version": 2, "profile": "default_basic_v1",
                         "max_candidates": config.pipeline_v2_sparse_max_candidates,
                         "match_tolerance": config.pipeline_v2_lighting_match_tolerance,
                         "coverage_target": config.pipeline_v2_lighting_coverage_target,
                         "schedule": {"unit": "beats", "step": .5}},
            "arcs": {"version": 2, "min_gap_seconds": config.pipeline_v2_arc_min_gap_seconds,
                     "max_gap_seconds": config.pipeline_v2_arc_max_gap_seconds,
                     "use_legacy_gap": config.pipeline_v2_arc_use_legacy_gap,
                     "legacy_gap": json_value(config.slider_time_gap), "threshold": .5}}


def stage_data(settings, stage):
    """Only fields that change this stage's examples or inference semantics."""
    keys = ["encoder_grid_seconds", "history_length"]
    if stage == "lighting":
        keys = ["encoder_grid_seconds", "lighting_block_length", "lighting_match_tolerance"]
    if stage in ("multiplier", "hazards", "arcs"):
        keys += ["max_copies", "max_chain_slices", "max_chain_span_seconds",
                 "expansion_time_quantum", "chain_span_quantum"]
    if stage == "arcs":
        keys += ["arc_shape_bounds", "lighting_match_tolerance"]
    result = {key: deepcopy(settings.get(key)) for key in keys}
    if stage == "lighting":
        result["lighting_onset_threshold"] = settings.get("schedule", {}).get("lighting_onset_threshold", 1.)
    if stage == "hazards":
        result["occupancy_clearance_seconds"] = settings.get("schedule", {}).get("clearance_seconds", .1)
    if stage in SPARSE_STAGES:
        result["policy_version"] = 2 if repaired(settings, stage) else 1
        if repaired(settings, stage):
            result["policy"] = deepcopy(settings["stage_policies"][stage])
        else:
            prefixes = {"hazards": ("wall_", "bomb_", "clearance_"),
                        "lighting": ("lighting_",), "arcs": ("arc_",)}[stage]
            result["schedule"] = {key: deepcopy(value) for key, value in settings.get("schedule", {}).items()
                                  if key.startswith(prefixes) or key == "max_candidates"}
        if stage in ("hazards", "lighting"):
            key = "bomb_threshold" if stage == "hazards" else "lighting_threshold"
            result[key] = settings.get(key, .5)
    return json_value(result)


def semantic_identity(identity):
    """Compare old manifests without rewriting or weakening their hash checks."""
    result = {key: deepcopy(value) for key, value in identity.items()
              if key not in ("sha256", "contract_version", "data_settings", "stage_data")}
    result["stage_data"] = deepcopy(identity.get("stage_data")) if "stage_data" in identity else stage_data(
        identity.get("data_settings") or {}, identity["stage"])
    return result


def split_identity(manifest, stage):
    """Bind calibration to the exact indexed learning split, independently of inference."""
    from .arc_policy import stage_split
    from training.beat_models.artifacts import statistics_hash

    rows = []
    for record in manifest.get("songs", []):
        row = {key: deepcopy(record[key]) for key in
               ("song_id", "group_id", "audio_sha256", "decoded_sha256", "sha256", "metadata_sha256")
               if record.get(key) is not None}
        row["split"] = stage_split(manifest, stage, record)
        rows.append(row)
    if not rows or len({row["song_id"] for row in rows}) != len(rows):
        raise ValueError("Training split identity requires unique indexed song records")
    result = {"version": 1, "stage": stage, "records": sorted(rows, key=lambda row: row["song_id"])}
    result["sha256"] = statistics_hash(result)
    return result
