"""Elapsed-time onset and retained-note diagnostics for the new framework."""

import numpy as np

from .contracts import decode_note


def match_onsets(reference, predicted, tolerance=0.020):
    """Maximum one-to-one matching for ordered onsets, inclusive 20-ms boundary."""
    reference, predicted = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Timing tolerance must be finite and nonnegative")
    for values in (reference, predicted):
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError("Onsets must be finite, distinct and ordered")
    pairs, i, j = [], 0, 0
    while i < len(reference) and j < len(predicted):
        gap = predicted[j] - reference[i]
        if abs(gap) <= tolerance + 1e-12:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif gap < -tolerance:
            j += 1
        else:
            i += 1
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def timing_scores(reference, predicted, tolerance=0.020):
    pairs = match_onsets(reference, predicted, tolerance)
    matched = len(pairs)
    precision = matched / len(predicted) if len(predicted) else float(not len(reference))
    recall = matched / len(reference) if len(reference) else float(not len(predicted))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.abs(np.asarray(reference)[pairs[:, 0]] - np.asarray(predicted)[pairs[:, 1]])
    return {"tolerance_seconds": tolerance, "matched_onsets": matched,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_absolute_error_seconds": float(errors.mean()) if matched else None}


def _fraction(correct, total):
    return correct / total if total else None


def generation_metrics(song, generated, requested_nps):
    arrays, duration = song.arrays, float(song.metadata["duration"])
    pairs = match_onsets(arrays["event_times"], generated.event_times)
    truth, predicted = arrays["note_classes"], generated.note_classes
    reference_count, predicted_count = int(np.count_nonzero(truth)), int(np.count_nonzero(predicted))
    count_error = predicted_count - reference_count
    hands = {}
    for hand, name in enumerate(("red", "blue")):
        correct, present, position, direction, compared = 0, 0, 0, 0, 0
        for ref, pred in pairs:
            target, output = int(truth[ref, hand]), int(predicted[pred, hand])
            correct += target == output
            present += bool(target) == bool(output)
            if target and output:
                expected, actual = decode_note(target), decode_note(output)
                position += expected[:2] == actual[:2]
                direction += expected[2] == actual[2]
                compared += 1
        hands[name] = {"class_accuracy": _fraction(correct, len(pairs)),
                       "presence_accuracy": _fraction(present, len(pairs)),
                       "position_accuracy": _fraction(position, compared),
                       "direction_accuracy": _fraction(direction, compared),
                       "compared_colored_notes": compared}
    bins = len(arrays["bin_durations"])
    bin_counts = np.zeros(bins, dtype=np.int64)
    for time, classes in zip(generated.event_times, predicted):
        bin_counts[min(int(time), bins - 1)] += np.count_nonzero(classes)
    target = arrays["intensity"]
    emitted = bin_counts / predicted_count if predicted_count else np.zeros(bins)
    target_valid = bool(song.metadata.get("intensity_target_valid", reference_count > 0))
    return {"song_id": song.song_id, "split": song.metadata.get("split"),
            "group_id": song.metadata.get("group_id"), "duration_seconds": duration,
            "requested_nps": float(requested_nps), "generated_nps": predicted_count / duration,
            "original_colored_notes": len(arrays["original_notes"]),
            "reduced_colored_notes": reference_count, "reference_unique_onsets": len(truth),
            "generated_colored_notes": predicted_count, "generated_unique_onsets": len(predicted),
            "colored_note_count_error": count_error, "absolute_colored_note_count_error": abs(count_error),
            "requested_density_error_nps": predicted_count / duration - requested_nps,
            "reference_density_error_nps": count_error / duration,
            "timing": timing_scores(arrays["event_times"], generated.event_times),
            "intensity": {"target_valid": target_valid,
                          "predicted_share_mae": float(np.abs(generated.intensity - target).mean()) if target_valid else None,
                          "predicted_share_total_variation": float(np.abs(generated.intensity - target).sum() / 2) if target_valid else None,
                          "emitted_share_total_variation": float(np.abs(emitted - target).sum() / 2) if target_valid else None,
                          "emitted_bin_counts": bin_counts.tolist()},
            "hands": hands,
            "event_count_mismatches": int(np.count_nonzero(np.count_nonzero(predicted, axis=1) != generated.counts[generated.event_frames]))}


def summarize(reports):
    reports = list(reports)
    totals = {key: sum(report[key] for report in reports) for key in
              ("original_colored_notes", "reduced_colored_notes", "reference_unique_onsets",
               "generated_colored_notes", "generated_unique_onsets", "absolute_colored_note_count_error")}
    matches = sum(report["timing"]["matched_onsets"] for report in reports)
    reference, predicted = totals["reference_unique_onsets"], totals["generated_unique_onsets"]
    precision = matches / predicted if predicted else float(not reference)
    recall = matches / reference if reference else float(not predicted)
    totals.update({"songs": len(reports), "timing_precision": precision, "timing_recall": recall,
                   "timing_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return totals
