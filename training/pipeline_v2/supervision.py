"""Streaming supervision accounting and TRAIN-only sparse-object balancing."""

import numpy as np

from .policies import SPARSE_STAGES


def support(examples, stage):
    result = {}
    for head, values in examples.targets.items():
        if stage == "arcs" and head != "attach":
            continue
        values = np.asarray(values)
        weights = np.asarray(examples.weights[head], np.float64)
        while weights.ndim < values.ndim:
            weights = weights[..., None]
        weights = np.broadcast_to(weights, values.shape)
        positive = values > 0
        axes = tuple(range(values.ndim - 1)) if head in ("bombs", "events") else None
        result[head] = {"positive": np.asarray((weights * positive).sum(axis=axes)).tolist(),
                        "negative": np.asarray((weights * ~positive).sum(axis=axes)).tolist()}
    return result


def add_support(total, current, group=None):
    for head, counts in current.items():
        target = total.setdefault(head, {"positive": np.zeros_like(counts["positive"], dtype=np.float64),
                                        "negative": np.zeros_like(counts["negative"], dtype=np.float64),
                                        "positive_groups": set(), "groups": set()})
        for key in ("positive", "negative"):
            target[key] += np.asarray(counts[key])
        if group is not None:
            target["groups"].add(group)
            if np.sum(counts["positive"]) > 0:
                target["positive_groups"].add(group)


def describe_support(total):
    return {head: {"positive": counts["positive"].tolist(), "negative": counts["negative"].tolist(),
                   "positive_groups": sorted(counts["positive_groups"]), "groups": sorted(counts["groups"])}
            for head, counts in total.items()}


def validate_support(splits, stage):
    if stage not in SPARSE_STAGES:
        return
    required = {"hazards": ("wall", "bombs"), "lighting": ("events",), "arcs": ("attach",)}[stage]
    for split in ("train", "validation"):
        for head in required:
            counts = splits.get(split, {}).get(head, {})
            for label in ("positive", "negative"):
                if np.sum(counts.get(label, 0)) <= 0:
                    raise ValueError(f"{stage}.{head} {split} has no supervised {label} examples; repair the dataset before training")
            if stage == "lighting" and split == "train" and np.any(np.asarray(counts.get("positive", [])) <= 0):
                raise ValueError("Lighting vocabulary contains classes without effective TRAIN positives; refit preparation after exclusions")


def class_balance(splits, stage, cap=100.):
    """Factors multiply loss values; supervision masks remain unchanged."""
    if isinstance(cap, bool) or not np.isfinite(cap) or cap < 1:
        raise ValueError("Positive class weight cap must be finite and at least one")
    result = {}
    for head, counts in splits["train"].items():
        positive, negative = (np.asarray(counts[key], np.float64) for key in ("positive", "negative"))
        if head in ("wall", "attach"):
            total = positive + negative
            result[head] = {"positive": float(total / (2 * positive)),
                            "negative": float(total / (2 * negative))}
        elif head in ("bombs", "events"):
            ratio = np.divide(negative, positive, out=np.ones_like(positive), where=positive > 0)
            result[head] = {"positive": np.clip(ratio, 1., cap).tolist(), "negative": 1.}
    return result
