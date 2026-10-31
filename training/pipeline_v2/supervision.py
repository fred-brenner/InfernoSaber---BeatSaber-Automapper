"""Streaming supervision accounting and TRAIN-only sparse-object balancing."""

import numpy as np

from .contracts import StageExamples
from .policies import SPARSE_STAGES

ENABLED_SUPERVISION_HEADS = {"hazards": ("wall", "bombs"), "arcs": ("attach",), "lighting": ("events",)}


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


def enabled_song_mask(examples, stage):
    """Zero one head's weights when its song never uses that feature.

    A song without effective supervised positives is feature-disabled for that
    head only; other heads keep supervision, and the queries remain inputs.
    Generation always activates the stage, so disabled maps must not teach
    "never fire". Returns the examples (original or a masked copy) and a
    per-head report of newly masked supervised values.
    """
    heads = ENABLED_SUPERVISION_HEADS.get(stage)
    if not heads:
        return examples, {}
    positives = support(examples, stage)
    weights = dict(examples.weights)
    masked = {}
    for head in heads:
        weight = weights.get(head)
        if weight is None:
            continue
        effective = np.asarray(positives.get(head, {}).get("positive", 0.))
        if np.sum(effective) <= 0 and np.count_nonzero(weight):
            masked[head] = int(np.count_nonzero(weight))
            weights[head] = np.zeros_like(weight)
    if not masked:
        return examples, {}
    return StageExamples(examples.inputs, examples.targets, weights, examples.times), {"masked_supervised_values": masked}


NEGATIVE_WEIGHT_HEADS = {"wall", "bombs", "attach", "anchor"}


def class_balance(splits, stage, cap=100., negative_weight=1.):
    """Factors multiply loss values; supervision masks remain unchanged.

    ``negative_weight`` below one is PU-style supervision for classification
    heads: a reference's non-placement is stylistic, not a confidently wrong
    negative, so it loses proportionally less.
    """
    if isinstance(cap, bool) or not np.isfinite(cap) or cap < 1:
        raise ValueError("Positive class weight cap must be finite and at least one")
    if isinstance(negative_weight, bool) or not np.isfinite(negative_weight) or not 0 <= negative_weight <= 1:
        raise ValueError("Negative supervision weight must be between zero and one")
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
        if head in NEGATIVE_WEIGHT_HEADS and head in result:
            result[head]["negative"] = result[head]["negative"] * negative_weight
    return result
