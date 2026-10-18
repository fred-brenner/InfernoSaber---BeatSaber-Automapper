"""Recorded arc eligibility, canonical history, and arc-only recording splits."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

import numpy as np


ARC_POLICY_V2 = {
    "version": 2,
    "min_gap_seconds": .15,
    "max_gap_seconds": .8,
    "use_legacy_gap": False,
    "legacy_gap": [.5, 12.],
    "threshold": .5,
}
SPLITS = ("train", "validation", "test")


def history_value(record):
    """Describe the accepted decision; NONE never carries unused geometry."""
    rejected = (record.get("rejected", False) or record.get("accepted") is False
                or record.get("status") == "rejected")
    if "value" in record:
        value = np.asarray(record["value"]).reshape(4)
        attach = record.get("attach", value[0])
        if rejected or not attach or not value[0]:
            return [0., 0., 0., 0.]
        return [1., float(value[1]), float(value[2]), float(value[3])]
    if rejected or not record.get("attach", 0):
        return [0., 0., 0., 0.]
    shape = record.get("shape", [record.get("mu", 0.), record.get("tmu", 0.)])
    return [1., float(shape[0]), float(shape[1]), float(record.get("anchor", 0))]


def arc_policy(options):
    """Return the explicit recorded v2 policy; absence retains legacy routing."""
    policy = (options or {}).get("stage_policies", {}).get("arcs")
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy.get("version") != 2:
        raise ValueError("Unsupported recorded arc policy version")
    required = set(ARC_POLICY_V2)
    if set(policy) != required or not isinstance(policy["use_legacy_gap"], bool):
        raise ValueError("Incomplete or invalid recorded arc policy")
    bounds = [policy["min_gap_seconds"], policy["max_gap_seconds"], policy["threshold"]]
    legacy = policy["legacy_gap"]
    if not isinstance(legacy, (tuple, list)) or len(legacy) != 2:
        raise ValueError("Arc legacy gap requires two ordered bounds")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in [*bounds, *legacy]):
        raise ValueError("Arc policy parameters must be finite numbers")
    if not (0 <= bounds[0] <= bounds[1] and 0 <= bounds[2] <= 1
            and 0 <= legacy[0] < legacy[1]):
        raise ValueError("Arc policy bounds are invalid")
    return policy


def eligible_gap(gap, options):
    """Use inclusive v2 seconds; the optional legacy gate has strict bounds."""
    policy = arc_policy(options)
    if policy is None:
        from .scheduling import schedule_settings

        schedule = schedule_settings((options or {}).get("schedule"))
        lower, upper = schedule["arc_min_gap_seconds"], schedule["arc_max_gap_seconds"]
        return bool(0 < gap and lower <= gap <= upper)
    lower, upper = policy["min_gap_seconds"], policy["max_gap_seconds"]
    allowed = 0 < gap and lower <= gap <= upper
    if policy["use_legacy_gap"]:
        allowed = allowed and policy["legacy_gap"][0] < gap < policy["legacy_gap"][1]
    return bool(allowed)


def arc_candidates(notes, options):
    """Share adjacent same-hand endpoint selection across v2 callers."""
    from .scheduling import arc_candidates as scheduled_candidates
    from .scheduling import schedule_settings

    policy = arc_policy(options)
    schedule = schedule_settings((options or {}).get("schedule"))
    if policy is not None:
        schedule.update(arc_min_gap_seconds=policy["min_gap_seconds"],
                        arc_max_gap_seconds=policy["max_gap_seconds"])
    maximum = schedule["max_candidates"]
    if policy is not None and policy["use_legacy_gap"]:
        notes = list(notes)
        # Eligibility includes both recorded gates; count only eligible pairs.
        schedule["max_candidates"] = max(maximum, len(notes))
    queries = scheduled_candidates(notes, schedule)
    if policy is not None and policy["use_legacy_gap"]:
        queries = [query for query in queries if eligible_gap(query["gap"], options)]
        if len(queries) > maximum:
            raise ValueError(f"Candidate schedule has {len(queries)} queries; exceeds configured "
                             f"max_candidates={maximum}; no queries were discarded")
    return queries


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _summaries(records):
    result, seen = [], set()
    for record in records:
        item = {key: record[key] for key in ("song_id", "group_id", "split", "positive_count")}
        if any(not isinstance(item[key], str) or not item[key] for key in ("song_id", "group_id")):
            raise ValueError("Arc split records need song and recording group IDs")
        if item["song_id"] in seen:
            raise ValueError("Arc split contains duplicate song IDs")
        seen.add(item["song_id"])
        count = item["positive_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Arc positive_count must be a nonnegative integer")
        if item["split"] not in SPLITS:
            raise ValueError("Arc split records need an explicit source split")
        for key in ("audio_sha256", "decoded_sha256"):
            if record.get(key):
                if not isinstance(record[key], str):
                    raise ValueError("Arc audio identities must be strings")
                item[key] = record[key]
        result.append(item)
    if not result:
        raise ValueError("Arc split requires recording summaries")
    return sorted(result, key=lambda item: item["song_id"])


def _validate_identities(records, assignments):
    identities = {}
    for item in records:
        split = assignments[item["song_id"]]
        for key in ("group_id", "audio_sha256", "decoded_sha256"):
            if item.get(key):
                token = ("group" if key == "group_id" else "audio", item[key])
                if identities.get(token, split) != split:
                    raise ValueError(f"Arc recording identity leaks between splits: {key}")
                identities[token] = split


def build_split_overlay(records, *, structural_dummy=False):
    """Move two seeded positive TRAIN groups, preserving every song variant.

    Summaries count only attach=1 targets with positive supervision weight.
    Explicit structural dummy mode keeps its original splits for smoke checks.
    """
    if not isinstance(structural_dummy, bool):
        raise ValueError("structural_dummy must be explicit boolean mode")
    records = _summaries(records)
    original = {item["song_id"]: item["split"] for item in records}
    _validate_identities(records, original)
    positives = Counter()
    for item in records:
        if item["split"] == "train":
            positives[item["group_id"]] += item["positive_count"]
    positive_groups = sorted(group for group, count in positives.items() if count > 0)
    if not structural_dummy and len(positive_groups) < 3:
        raise ValueError("Arc split repair requires at least three positive TRAIN recording groups")
    moved = [] if structural_dummy else sorted(np.random.default_rng(3).permutation(positive_groups)[:2].tolist())
    groups = {item["group_id"]: "validation" if item["group_id"] in moved else item["split"]
              for item in records}
    assignments = {item["song_id"]: groups[item["group_id"]] for item in records}
    _validate_identities(records, assignments)
    splits = {split: [item["song_id"] for item in records if assignments[item["song_id"]] == split]
              for split in SPLITS}
    counts = {split: {"songs": len(splits[split]),
                      "groups": len({item["group_id"] for item in records if assignments[item["song_id"]] == split}),
                      "positive_count": sum(item["positive_count"] for item in records if assignments[item["song_id"]] == split)}
              for split in SPLITS}
    result = {"version": 1, "stage": "arcs", "seed": 3,
              "mode": "structural_dummy" if structural_dummy else "positive_train_group_holdout",
              "positive_train_group_ids": positive_groups, "moved_group_ids": moved,
              "groups": groups, "records": assignments, "splits": splits, "counts": counts,
              "source_records": records,
              "provenance": {"selection": "sorted positive TRAIN group IDs; default_rng(3) permutation; first two",
                             "scope": "arcs_only", "test_unchanged": True,
                             "previous_upstream_exposure": {
                                 "group_ids": moved, "split": "train",
                                 "stages": ["intensity", "timing", "positions", "arrows", "multiplier", "hazards"],
                                 "interpretation": "Arc-stage holdout only; these groups were in upstream TRAIN."}}}
    result["fingerprint"] = _fingerprint(result)
    return result


def validate_split_overlay(manifest):
    """Check the saved policy, fingerprint, source records and whole groups."""
    overlay = manifest.get("stage_splits", {}).get("arcs")
    if overlay is None:
        return True
    if not isinstance(overlay, dict):
        raise ValueError("Invalid arc stage split overlay")
    payload = deepcopy(overlay)
    fingerprint = payload.pop("fingerprint", None)
    if fingerprint != _fingerprint(payload):
        raise ValueError("Arc split overlay fingerprint mismatch")
    if overlay.get("mode") not in ("structural_dummy", "positive_train_group_holdout"):
        raise ValueError("Invalid arc split overlay mode")
    expected = build_split_overlay(overlay["source_records"], structural_dummy=overlay["mode"] == "structural_dummy")
    if expected != overlay:
        raise ValueError("Arc split overlay disagrees with its deterministic source policy")
    songs = manifest.get("songs")
    if songs is not None:
        indexed = {item["song_id"]: item for item in songs}
        if len(indexed) != len(songs) or set(indexed) != set(overlay["records"]):
            raise ValueError("Arc split overlay does not cover the dataset song records")
        for source in overlay["source_records"]:
            item = indexed[source["song_id"]]
            if any(item.get(key) != source[key] for key in ("group_id", "split")):
                raise ValueError("Arc split overlay source recording disagrees with dataset")
            for key in ("audio_sha256", "decoded_sha256"):
                if item.get(key) and item[key] != source.get(key):
                    raise ValueError("Arc split overlay source audio disagrees with dataset")
    return True


def stage_split(manifest, stage, record):
    """Resolve a validated overlay without changing shared song metadata."""
    if stage != "arcs" or not manifest.get("stage_splits", {}).get("arcs"):
        return record["split"]
    overlay = manifest["stage_splits"]["arcs"]
    group = record.get("group_id")
    effective = overlay["groups"].get(group)
    if effective is None:
        raise ValueError("Recording group is missing from arc split overlay")
    song_id = record.get("song_id")
    if song_id is not None and overlay["records"].get(song_id) != effective:
        raise ValueError("Song identity disagrees with arc split overlay")
    return effective


def canonical_history_inputs(inputs):
    """Normalize legacy cached NONE histories without editing caller arrays."""
    values = inputs["history"]
    none = values[..., 0] == 0
    if not np.any(values[none] != 0):
        return inputs
    canonical = values.copy()
    canonical[none] = history_value({"attach": 0})
    return {**inputs, "history": canonical}
