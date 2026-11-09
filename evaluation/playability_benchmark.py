"""Evaluation-only, frozen-human scorecards and paired song-level decisions.

Inputs use the scorer's elapsed-second ``(t, hand, x, y, direction)`` tuples.
Nothing here is imported by generation or consults a human corpus at runtime.
The official M1–M5 verdict is reused verbatim; engineering distances use the
paired control's density band. A decision only advances a policy for playtest.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from pathlib import Path
import json

import numpy as np

from evaluation.playability_directions import _DIRECTION_POSITION
from evaluation.playability_metrics import (
    M1_GAP_BANDS, M1_STRESS, REVERSE, T_FAST, _cross_hand_partners, _dominant_axis,
    _event_segments, _hand_vectors, _is_mirrored_double, prepare_notes, score_map,
)
from evaluation.playability_report import (
    BAND_LABELS, BAND_MIN_N, METRICS, _incident_rate, _jsonable,
    _m5_eligible, _metric_units, band_label, decode_package_notes, verdict,
)

DISTANCE_METRICS = ("m1", "m2", "m3", "m4", "m5_non_reversal", "m5_fast_reversal")
_DWELL_STATISTICS = ("notes_q1", "notes_median", "notes_q3", "milliseconds_q1",
                     "milliseconds_median", "milliseconds_q3", "one_note_share", "at_least_five_share")
DWELL_DISTANCE_METRICS = tuple(f"dwell_h{hand}_a{angle}_{statistic}"
    for hand in (0, 1) for angle in (45, 90) for statistic in _DWELL_STATISTICS)
_BOOLEAN_CHECKS = ("timestamps_hands_count_preserved", "chains_intact",
                   "arc_endpoints_intact", "roundtrip_v2_valid", "roundtrip_v3_valid")
_COUNT_CHECKS = ("new_hard_clearance_violations", "frozen_speed_violations_increase",
                 "frozen_flow_violations_increase")


def _distribution(values):
    values = [float(value) for value in values]
    quartiles = np.percentile(values, [25, 50, 75]) if values else [None] * 3
    return {"count": len(values), **dict(zip(("q1", "median", "q3"), quartiles))}


def _run_summary(observations):
    lengths = [run[0] for run in observations]
    return {"notes": _distribution(lengths),
            "milliseconds": _distribution([run[1] for run in observations]),
            "one_note_share": sum(n == 1 for n in lengths) / len(lengths) if lengths else None,
            "at_least_five_share": sum(n >= 5 for n in lengths) / len(lengths) if lengths else None,
            "observations": observations}


def _dwell_statistic(summary, name):
    if name.startswith(("notes_", "milliseconds_")):
        unit, statistic = name.split("_", 1)
        return summary[unit][statistic]
    return summary[name]


def _axis_dwell(prepared):
    result = {}
    for hand in (0, 1):
        observed = {"45": [], "90": []}
        notes = [note for note in prepared if note["hand"] == hand]
        for directed, sequence in groupby(notes, lambda note: note["d"] != 8):
            if not directed:
                continue
            arrows = list(sequence)
            start = 0
            for index in range(1, len(arrows)):
                before = _DIRECTION_POSITION[arrows[index - 1]["d"]] % 4
                after = _DIRECTION_POSITION[arrows[index]["d"]] % 4
                steps = abs(before - after)
                angle = 45 * min(steps, 4 - steps)
                if angle:
                    observed[str(angle)].append([index - start,
                        1000 * (arrows[index]["t"] - arrows[start]["t"])])
                    start = index
        result[str(hand)] = {angle: _run_summary(runs) for angle, runs in observed.items()}
        result[str(hand)]["pooled"] = _run_summary(observed["45"] + observed["90"])
    return result


def _coverage(prepared):
    segments = _event_segments(prepared)
    eligible, vectors = 0, {"0": 0, "1": 0}
    for segment in segments:
        per_hand = [_hand_vectors(segment, hand) for hand in (0, 1)]
        if all(_dominant_axis(hand_vectors) is not None for hand_vectors in per_hand):
            eligible += 1
            for hand, hand_vectors in enumerate(per_hand):
                vectors[str(hand)] += len(hand_vectors)
    partners = mirrored = 0
    for index, note in enumerate(prepared):
        if note["hand"] == 0:
            for partner in _cross_hand_partners(prepared, index):
                partners += 1
                mirrored += int(_is_mirrored_double(note, prepared[partner]))
    transitions = [note for note in prepared if note["pred"] is not None]
    return {"m1_transitions": len(transitions),
            "m1_speed_weight": sum(max(0.0, 1 - (note["t"] - note["pred"]["t"]) / T_FAST)
                                   for note in transitions),
            "m2_notes": len(prepared), "m3_activity_segments": len(segments),
            "m3_eligible_segments": eligible, "m3_eligible_vectors": vectors,
            "m3_eligible_fraction": eligible / len(segments) if segments else None,
            "m4_cross_hand_pairs": partners, "mirrored_doubles": mirrored}


def _horizontal_runs(prepared):
    result = {}
    for hand in (0, 1):
        runs, starts, length, start = [], [], 0, None
        notes = [note for note in prepared if note["hand"] == hand]
        for before, after in zip(notes, notes[1:]):
            legal = (before["d"] in (2, 3) and after["d"] == REVERSE[before["d"]]
                     and after["t"] - before["t"] <= 0.25 + 1e-9)
            if legal:
                if not length:
                    start = before["t"]
                length += 1
            elif length:
                runs.append(length)
                starts.append(start)
                length = 0
        if length:
            runs.append(length)
            starts.append(start)
        result[str(hand)] = {"arrivals": runs, "start_seconds": starts,
                             "longest": max(runs, default=0)}
    return result


def _reference_group(rows, thresholds):
    minutes = sum(row["scores"]["span_seconds"] for row in rows) / 60
    result = {"maps": len(rows), "minutes": minutes}
    for metric in METRICS:
        scalars = [row["scores"][metric]["value"] for row in rows]
        units = _metric_units(rows, metric)
        count = int(np.sum(units >= thresholds[metric])) if thresholds[metric] is not None else 0
        values = ([ _incident_rate(row["scores"][metric]["units"], thresholds[metric],
                                    row["scores"]["span_seconds"] / 60) for row in rows]
                  if metric == "m1" else scalars)
        result[metric] = {**_distribution(values), "values": values,
                          "scalar_values": scalars, "positive_events": len(units),
                          "incident_count": count, "minutes": minutes,
                          "human_incident_rate": count / minutes if minutes > 0 else None}
        if metric == "m1":
            result[metric]["median"] = count / minutes if minutes > 0 else None
    eligible = [row for row in rows if _m5_eligible(row["scores"])]
    shares = [row["scores"]["m5"]["non_reversal_share"] for row in eligible]
    rates = [row["scores"]["m5"]["fast_reversals_per_minute"] for row in eligible]
    minutes = sum(row["scores"]["span_seconds"] for row in eligible) / 60
    events = sum(row["scores"]["m5"]["fast_reversals"] for row in eligible)
    result["m5_non_reversal"] = {**_distribution(shares), "values": shares}
    result["m5_fast_reversal"] = {**_distribution(rates), "values": rates,
                                  "median": events / minutes if minutes > 0 else None,
                                  "incident_count": events, "minutes": minutes}
    return result


def prepare_human_reference(rows, *, skipped=()):
    """Freeze scored human rows; optional ``notes`` enable ending-angle dwell.

    ``rows`` use ``playability_report.score_rows`` records. The small retained
    scorer projection permits exact official verdicts after a JSON round-trip.
    Sparse official bands use the scorer's all-band fallback. Dwell expands to
    adjacent density bands until five source maps have observed runs.
    """
    rows = list(rows)
    thresholds = {}
    official = {}
    minutes = sum(row["scores"]["span_seconds"] for row in rows) / 60
    for metric in METRICS:
        units = _metric_units(rows, metric)
        threshold = float(np.percentile(units, 95)) if len(units) else None
        thresholds[metric] = threshold
        incidents = int(np.sum(units >= threshold)) if threshold is not None else 0
        official[metric] = {"incident_threshold": threshold, "pooled_events": len(units),
                            "incident_count": incidents, "minutes": minutes,
                            "incident_rate": incidents / minutes if minutes > 0 else None}
    by_band = {label: [row for row in rows if band_label(row["scores"]["nps"]) == label]
               for label in BAND_LABELS}
    bands = {label: _reference_group(subset, thresholds) for label, subset in by_band.items()}
    all_rows = _reference_group(rows, thresholds)
    observed = {id(row): _axis_dwell(prepare_notes(row["notes"])) for row in rows if "notes" in row}
    for index, label in enumerate(BAND_LABELS):
        bands[label]["axis_dwell"] = {}
        for hand in ("0", "1"):
            bands[label]["axis_dwell"][hand] = {}
            for angle in ("45", "90"):
                selected = [index]
                def eligible_runs():
                    return [observed[id(row)][hand][angle]["observations"]
                            for selected_index in selected for row in by_band[BAND_LABELS[selected_index]]
                            if id(row) in observed and observed[id(row)][hand][angle]["observations"]]
                source = eligible_runs()
                # Add the nearest adjacent band, lower first on a distance tie.
                for distance in range(1, len(BAND_LABELS)):
                    if len(source) >= BAND_MIN_N:
                        break
                    for neighbor in (index - distance, index + distance):
                        if 0 <= neighbor < len(BAND_LABELS):
                            selected.append(neighbor)
                            source = eligible_runs()
                            if len(source) >= BAND_MIN_N:
                                break
                bands[label]["axis_dwell"][hand][angle] = {
                    **_run_summary([run for runs in source for run in runs]),
                    "reference_maps": len(source), "supported": len(source) >= BAND_MIN_N,
                    "fallback": len(selected) > 1,
                    "source_bands": [BAND_LABELS[i] for i in sorted(selected)]}
                bands[label]["axis_dwell"][hand][angle]["per_map_distributions"] = {
                    statistic: _distribution([_dwell_statistic(_run_summary(runs), statistic) for runs in source])
                    for statistic in _DWELL_STATISTICS}
    projection = []
    for row in rows:
        scores = row["scores"]
        projection.append({"scores": {"nps": scores["nps"], "span_seconds": scores["span_seconds"],
            **{metric: {"value": scores[metric]["value"], "units": scores[metric]["units"]}
               for metric in METRICS},
            "m5": ({key: scores.get("m5", {}).get(key) for key in (
                "transitions", "non_reversal_share", "fast_reversals", "fast_reversals_per_minute")} if scores.get("m5") else {})}})
    return _jsonable({"schema_version": 1, "maps": len(rows), "skipped": list(skipped),
                      "official": official, "all": all_rows, "bands": bands,
                      "official_rows": projection,
                      "sources": [{"package": row.get("package"), "difficulty": row.get("difficulty")}
                                  for row in rows]})


def load_human_reference(root, *, cache_path=None):
    """Read only curated ``bs_map_input`` packages, recording each skipped map.

    A supplied cache path writes the newly frozen reference; it never silently
    replaces a run's reference with an old cache. To reuse one, load its JSON
    explicitly and pass it to ``compact_scorecard``.
    """
    root = Path(root)
    nested = root / "bs_map_input"
    if nested.is_dir():
        root = nested
    if root.name == "favorites_bs_input":
        raise ValueError("Favorites are training input, not a human comparison corpus")
    rows, skipped = [], []
    for package in sorted(root.iterdir()):
        if not package.is_dir():
            continue
        try:
            decoded = decode_package_notes(package)
            if len(decoded["notes"]) < 2:
                raise ValueError("Expected at least two playable notes")
            rows.append({"package": package.name, "difficulty": decoded["difficulty"],
                         "notes": decoded["notes"], "scores": score_map(decoded["notes"])})
        except Exception as error:
            skipped.append({"package": package.name, "error": str(error)})
    result = prepare_human_reference(rows, skipped=skipped)
    if cache_path is not None:
        Path(cache_path).write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")
    return result


def _matched_reference(reference, label, metric):
    if metric.startswith("dwell_"):
        _, hand, angle, statistic = metric.split("_", 3)
        summary = reference["bands"][label]["axis_dwell"][hand[1:]][angle[1:]]
        maps = summary["reference_maps"]
        distribution = summary.get("per_map_distributions", {}).get(statistic, {})
        return {"median": _dwell_statistic(summary, statistic) if summary["supported"] else None,
                "q1": distribution.get("q1"), "q3": distribution.get("q3"),
                "reference_maps": maps, "band_maps": maps, "fallback": summary["fallback"]}
    group = reference["bands"][label]
    if metric.startswith("m5"):
        count = group[metric]["count"]
    else:
        count = group["maps"]
    fallback = count < BAND_MIN_N
    selected = reference["all"] if fallback else group
    return {**selected[metric], "band_maps": count, "fallback": fallback,
            "reference_maps": selected[metric]["count"]}


def _dwell_distances(dwell, reference):
    output = {}
    for hand in ("0", "1"):
        output[hand] = {}
        for angle in ("45", "90"):
            observed, human = dwell[hand][angle], reference[hand][angle]
            distances = {}
            for unit in ("notes", "milliseconds"):
                for statistic in ("q1", "median", "q3"):
                    left, right = observed[unit][statistic], human[unit][statistic]
                    distances[f"{unit}_{statistic}"] = (abs(left - right)
                        if human["supported"] and left is not None and right is not None else None)
            for statistic in ("one_note_share", "at_least_five_share"):
                left, right = observed[statistic], human[statistic]
                distances[statistic] = (abs(left - right)
                    if human["supported"] and left is not None and right is not None else None)
            output[hand][angle] = {"distances": distances, "reference_maps": human["reference_maps"],
                                  "source_bands": human["source_bands"], "fallback": human["fallback"],
                                  "reference": {key: human[key] for key in (
                                      "notes", "milliseconds", "one_note_share", "at_least_five_share",
                                      "reference_maps", "source_bands", "supported", "fallback")}}
    return output


def _m1_events(prepared, threshold, minutes):
    bands = {label: {"transitions": 0, "positive_events": 0, "incident_count": 0,
                     "speed_weight": 0.0} for label in M1_GAP_BANDS}
    incidents = []
    for note in prepared:
        if note["pred"] is None:
            continue
        gap = note["t"] - note["pred"]["t"]
        label = "<=0.2" if gap <= 0.2 else "0.2-0.35" if gap <= 0.35 else ">0.35"
        speed = max(0.0, 1 - gap / T_FAST)
        stress = M1_STRESS[note["d"]] * speed
        band = bands[label]
        band["transitions"] += 1
        band["speed_weight"] += speed
        band["positive_events"] += int(stress > 0)
        if threshold is not None and stress > 0 and stress >= threshold:
            band["incident_count"] += 1
        if stress > 0:
            incidents.append({"t": note["t"], "hand": note["hand"],
                              "gap_seconds": gap, "severity": stress})
    for band in bands.values():
        band["incident_rate"] = band["incident_count"] / minutes if minutes > 0 and threshold is not None else None
        if threshold is None:
            band["incident_count"] = None
    return bands, sorted(incidents, key=lambda item: -item["severity"])[:10]


def compact_scorecard(notes, reference=None, *, control_nps=None):
    """Score normalized notes, preserving official verdicts and uncapped counts.

    Human-free calls still provide complete observations. ``control_nps`` fixes
    engineering comparisons to the paired control band; official verdicts keep
    their original realized-NPS matching rule. Missing eligibility stays null.
    """
    notes = list(notes)
    scores = score_map(notes)
    prepared = prepare_notes(notes)
    minutes = scores["span_seconds"] / 60
    coverage = _coverage(prepared)
    dwell, runs = _axis_dwell(prepared), _horizontal_runs(prepared)
    label = band_label(scores["nps"])
    control_label = band_label(scores["nps"] if control_nps is None else control_nps)
    officials = verdict(scores, reference["official_rows"]) if reference is not None else {}
    metrics, distances = {}, {key: None for key in DISTANCE_METRICS}
    for metric in METRICS:
        score = scores[metric]
        threshold = reference["official"][metric]["incident_threshold"] if reference else None
        incident_count = int(np.sum(np.asarray(score["units"]) >= threshold)) if threshold is not None else None
        metrics[metric] = {"value": score["value"], "positive_events": len(score["units"]),
                           "threshold": threshold, "incident_count": incident_count,
                           "minutes": minutes, "incident_rate": (incident_count / minutes
                               if incident_count is not None and minutes > 0 else None),
                           "official": officials.get(metric),
                           "top_incidents": score.get("incidents", [])}
        if metric == "m1":
            metrics[metric]["gap_band_mean_stress"] = score["bands"]
            metrics[metric]["transitions"] = score["transitions"]
            metrics[metric]["gap_bands"], metrics[metric]["top_incidents"] = _m1_events(prepared, threshold, minutes)
        if reference is not None:
            matched = _matched_reference(reference, control_label, metric)
            value = metrics[metric]["incident_rate"] if metric == "m1" else score["value"]
            eligible = (minutes > 0 and (coverage["m1_speed_weight"] > 0 if metric == "m1"
                         else coverage["m3_eligible_segments"] > 0 if metric == "m3"
                         else coverage["m4_cross_hand_pairs"] > 0 if metric == "m4" else bool(prepared)))
            if eligible and matched["median"] is not None and value is not None:
                distances[metric] = max(0.0, value - matched["median"])
            metrics[metric]["engineering_reference"] = {key: matched[key] for key in
                ("median", "q1", "q3", "band_maps", "reference_maps", "fallback")}
    m5 = scores["m5"]
    metrics["m5"] = {key: m5[key] for key in ("non_reversal_share", "transitions", "fast_reversals", "fast_reversals_per_minute")}
    metrics["m5"].update(minutes=minutes, official=officials.get("m5"))
    if reference is not None:
        metrics["m5"]["engineering_reference"] = {}
        for metric, field in (("m5_non_reversal", "non_reversal_share"),
                              ("m5_fast_reversal", "fast_reversals_per_minute")):
            matched = _matched_reference(reference, control_label, metric)
            metrics["m5"]["engineering_reference"][metric] = {key: matched[key] for key in
                ("median", "q1", "q3", "band_maps", "reference_maps", "fallback")}
            if _m5_eligible(scores) and matched["median"] is not None:
                distances[metric] = abs(m5[field] - matched["median"])
    total = len(notes)
    summary = {"exported_notes": total,
               "exported_nps": total / scores["span_seconds"] if scores["span_seconds"] > 0 else 0.0,
               "dot_share": sum(note[4] == 8 for note in notes) / total if total else None,
               "vertical_share": sum(note[4] in (0, 1) for note in notes) / total if total else None,
               "directional_vertical_share": (sum(note[4] in (0, 1) for note in notes)
                    / sum(note[4] != 8 for note in notes) if any(note[4] != 8 for note in notes) else None),
               "horizontal_share": sum(note[4] in (2, 3) for note in notes) / total if total else None,
               "left_share": sum(note[1] == 0 for note in notes) / total if total else None,
               "lane_counts": [sum(note[2] == x for note in notes) for x in range(4)],
               "row_counts": [sum(note[3] == y for note in notes) for y in range(3)]}
    dwell_comparison = _dwell_distances(dwell, reference["bands"][control_label]["axis_dwell"]) if reference else None
    for metric in DWELL_DISTANCE_METRICS:
        _, hand, angle, statistic = metric.split("_", 3)
        distances[metric] = dwell_comparison[hand[1:]][angle[1:]]["distances"][statistic] if dwell_comparison else None
    return _jsonable({"schema_version": 1, "notes": scores["notes"],
        "span_seconds": scores["span_seconds"], "nps": scores["nps"],
        "band": label, "control_band": control_label, "band_migration": label != control_label,
        "metrics": metrics, "distances": distances, "axis_dwell": dwell,
        "axis_dwell_comparison": dwell_comparison,
        "diagnostics": m5["diagnostics"], "movement_axis": m5["movement_axis"],
        "coverage": coverage, "note_summary": summary, "horizontal_runs": runs,
        "longest_fast_horizontal_run": max(runs["0"]["longest"], runs["1"]["longest"])})


def _song_means(observations):
    songs = defaultdict(list)
    for song, value in observations:
        songs[song].append(value)
    return {song: float(np.mean(values)) for song, values in songs.items()}


def freeze_material_harm_margins(control_rows, reference, *, aa_rows=()):
    """Freeze numerical band margins from controls, human IQR and A/A noise.

    Controls are records containing ``song`` and either ``control`` or
    ``scorecard``. A/A rows use paired-report records and are never candidates.
    The caller persists this return value before candidate generation.
    """
    controls = list(control_rows)
    aa_rows = list(aa_rows)
    result = {}
    for label in BAND_LABELS:
        result[label] = {}
        for metric in (*DISTANCE_METRICS, *DWELL_DISTANCE_METRICS):
            observations = []
            for row in controls:
                card = row.get("control", row.get("scorecard", {}))
                value = card.get("distances", {}).get(metric)
                if card.get("control_band", card.get("band")) == label and value is not None:
                    observations.append((row["song"], value))
            values = list(_song_means(observations).values())
            matched = _matched_reference(reference, label, metric)
            iqr = (matched["q3"] - matched["q1"] if matched["q1"] is not None else 0.0)
            noise = []
            for row in aa_rows:
                control, candidate = row.get("control", {}), row.get("candidate", {})
                a = control.get("distances", {}).get(metric)
                b = candidate.get("distances", {}).get(metric)
                if control.get("control_band", control.get("band")) == label and a is not None and b is not None:
                    noise.append(abs(b - a))
            result[label][metric] = max(0.05 * float(np.mean(values)) if values else 0.0,
                                        0.02 * iqr, max(noise, default=0.0))
    return result


def _paired_metric(rows, metric, margins, rng, bootstrap_samples):
    observations = []
    missing = []
    for row in rows:
        control, candidate = row["control"], row["candidate"]
        a, b = control["distances"].get(metric), candidate["distances"].get(metric)
        band = control.get("control_band", control["band"])
        margin = (margins or {}).get(band, {}).get(metric)
        if a is None or b is None:
            missing.append({**{key: row.get(key) for key in ("song", "difficulty", "seed")},
                            "control_missing": a is None, "candidate_missing": b is None})
            continue
        observations.append((row["song"], a, b, margin))
    control = _song_means([(song, a) for song, a, _, _ in observations])
    candidate = _song_means([(song, b) for song, _, b, _ in observations])
    harm = _song_means([(song, margin) for song, _, _, margin in observations if margin is not None])
    songs = sorted(control)
    deltas = np.asarray([candidate[song] - control[song] for song in songs])
    if len(deltas):
        boot = np.mean(rng.choice(deltas, size=(bootstrap_samples, len(deltas))), axis=1)
        ci = [float(value) for value in np.percentile(boot, [2.5, 97.5])]
    else:
        ci = [None, None]
    a = float(np.mean(list(control.values()))) if songs else None
    b = float(np.mean(list(candidate.values()))) if songs else None
    margin = float(np.mean(list(harm.values()))) if len(harm) == len(songs) and songs and all(item[3] is not None for item in observations) else None
    return {"pairs": len(observations), "songs": len(songs), "missing_eligibility": missing,
            "control_mean_distance": a, "candidate_mean_distance": b,
            "delta": b - a if a is not None else None, "delta_ci95": ci,
            "closure": (a - b) / a if a is not None and a > 0 else None,
            "harm_margin": margin,
            "harm_passed": ci[1] <= margin if ci[1] is not None and margin is not None else None,
            "song_deltas": dict(zip(songs, deltas.tolist()))}


def package_guardrails(rows):
    """Evaluate observed package invariants and the predeclared numeric limits."""
    rows = list(rows)
    density, dot, coverage, runtimes, generation_runtimes = defaultdict(list), [], [], [], []
    missing = defaultdict(list)
    hard_failures = []
    all_vertical = []
    runtime_scopes = defaultdict(int)
    for row in rows:
        control, candidate = row["control"], row["candidate"]
        name = {key: row.get(key) for key in ("song", "difficulty", "seed")}
        a = control["note_summary"].get("exported_nps", control.get("nps"))
        b = candidate["note_summary"].get("exported_nps", candidate.get("nps"))
        if a is not None and b is not None and a > 0:
            density[row.get("difficulty", "unknown")].append((b - a) / a)
        else:
            missing["density"].append(name)
        a, b = control["note_summary"].get("dot_share"), candidate["note_summary"].get("dot_share")
        if a is not None and b is not None:
            dot.append(b - a)
        else:
            missing["dots"].append(name)
        a, b = control["coverage"].get("m3_eligible_segments"), candidate["coverage"].get("m3_eligible_segments")
        if a is not None and b is not None:
            coverage.append((a, b))
        else:
            missing["m3_coverage"].append(name)
        a, b = row.get("control_runtime_seconds"), row.get("candidate_runtime_seconds")
        if a is not None and b is not None and a > 0:
            scope = row.get("runtime_scope", "end_to_end")
            runtime_scopes[scope] += 1
            if scope == "generation_only":
                generation_runtimes.append((b - a) / a)
                missing["runtime"].append({**name, "reason": "end_to_end_duration_unrecorded"})
            else:
                runtimes.append((b - a) / a)
        else:
            missing["runtime"].append(name)
        if (candidate["note_summary"].get("directional_vertical_share", candidate["note_summary"].get("vertical_share")) == 1
                and control["note_summary"].get("directional_vertical_share", control["note_summary"].get("vertical_share")) != 1):
            all_vertical.append(name)
        checks = row.get("checks", {})
        for key in (*_BOOLEAN_CHECKS, *_COUNT_CHECKS):
            value = checks.get(key)
            if value is None:
                missing["hard_checks"].append({**name, "check": key})
            elif (value is not True if key in _BOOLEAN_CHECKS else value > 0):
                hard_failures.append({**name, "check": key, "value": value})
    density_result = {difficulty: {"median_relative_change": float(np.median(values)),
                                   "passed": abs(float(np.median(values))) <= 0.03 + 1e-12,
                                   "pairs": len(values)} for difficulty, values in density.items()}
    loss = (1 - sum(b for _, b in coverage) / sum(a for a, _ in coverage)
            if coverage and sum(a for a, _ in coverage) > 0 else 0.0 if coverage else None)
    runtime = float(np.median(runtimes)) if runtimes else None
    dots = float(np.median(dot)) if dot else None
    def status(available, fail, key):
        return False if fail else None if not available or missing[key] else True
    return {"density": {"passed": status(bool(density), any(not value["passed"] for value in density_result.values()), "density"),
                         "by_difficulty": density_result, "limit": 0.03, "missing": missing["density"]},
            "dots": {"passed": status(bool(dot), dots is not None and dots > 0.01 + 1e-12, "dots"),
                     "median_share_increase": dots, "limit": 0.01, "missing": missing["dots"]},
            "m3_coverage": {"passed": status(bool(coverage), loss is not None and loss > 0.05 + 1e-12, "m3_coverage"),
                            "relative_loss": loss, "limit": 0.05, "missing": missing["m3_coverage"]},
            "runtime": {"passed": status(bool(runtimes), runtime is not None and runtime > 0.1 + 1e-12, "runtime"),
                        "median_relative_change": runtime, "scope_counts": dict(runtime_scopes),
                        "generation_only_median_relative_change": (float(np.median(generation_runtimes))
                                                                   if generation_runtimes else None),
                        "limit": 0.1, "missing": missing["runtime"]},
            "hard_checks": {"passed": status(bool(rows), bool(hard_failures), "hard_checks"),
                            "failures": hard_failures, "missing": missing["hard_checks"]},
            "axis_use": {"passed": not bool(all_vertical), "new_all_vertical_maps": all_vertical}}


def _pooled_events(rows):
    output = {}
    for arm in ("control", "candidate"):
        output[arm] = {}
        for metric in (*METRICS, "m5"):
            count, minutes, positive, maps = 0, 0.0, 0, 0
            for row in rows:
                observed = row[arm].get("metrics", {}).get(metric, {})
                events = observed.get("fast_reversals" if metric == "m5" else "incident_count")
                denominator = observed.get("minutes")
                # Unscorable M5 maps have no eligible map-minute denominator.
                if events is not None and denominator is not None and denominator > 0 and (metric != "m5" or observed.get("transitions", 0) > 0):
                    count += events
                    minutes += denominator
                    positive += observed.get("positive_events", events)
                    maps += 1
            output[arm][metric] = {"count": count, "positive_events": positive,
                "minutes": minutes, "rate": count / minutes if minutes > 0 else None, "maps": maps}
    return output


def _pooled_dwell(rows):
    output = {}
    for label in BAND_LABELS:
        selected = [row for row in rows if row["control"].get("control_band", row["control"]["band"]) == label]
        output[label] = {}
        for hand in ("0", "1"):
            output[label][hand] = {}
            for angle in ("45", "90"):
                output[label][hand][angle] = {}
                for arm in ("control", "candidate"):
                    observations = [run for row in selected
                        for run in row[arm].get("axis_dwell", {}).get(hand, {}).get(angle, {}).get("observations", [])]
                    output[label][hand][angle][arm] = _run_summary(observations)
                output[label][hand][angle]["reference"] = next((
                    row["control"].get("axis_dwell_comparison", {}).get(hand, {}).get(angle, {}).get("reference")
                    for row in selected if row["control"].get("axis_dwell_comparison")), None)
    return output


def paired_decision_report(rows, *, target="m1", frozen_margins=None,
                           bootstrap_seed=20261107, bootstrap_samples=2000):
    """Pair difficulties/seeds first, then give every song equal weight.

    Bootstrap units are song-mean paired differences. Incomplete pairs and
    absent eligibility are disclosed and prevent advancement. Margins must
    have been frozen independently before candidate runs.
    """
    if target not in (*DISTANCE_METRICS, *DWELL_DISTANCE_METRICS):
        raise ValueError(f"Unknown primary metric: {target}")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rows = list(rows)
    complete = [row for row in rows if row.get("control") is not None and row.get("candidate") is not None]
    incomplete = [{key: row.get(key) for key in ("song", "difficulty", "seed")}
                  for row in rows if row not in complete]
    rng = np.random.default_rng(bootstrap_seed)
    metrics = {metric: _paired_metric(complete, metric, frozen_margins, rng, bootstrap_samples)
               for metric in DISTANCE_METRICS}
    dwell_metrics = {metric: _paired_metric(complete, metric, frozen_margins, rng, bootstrap_samples)
                     for metric in DWELL_DISTANCE_METRICS}
    bands = {label: {metric: _paired_metric([row for row in complete
                         if row["control"].get("control_band", row["control"]["band"]) == label],
                         metric, frozen_margins, rng, bootstrap_samples)
                    for metric in DISTANCE_METRICS}
             for label in BAND_LABELS}
    difficulties = {difficulty: {metric: _paired_metric([row for row in complete if row.get("difficulty") == difficulty],
                             metric, frozen_margins, rng, bootstrap_samples)
                        for metric in DISTANCE_METRICS}
                    for difficulty in sorted({row.get("difficulty", "unknown") for row in complete})}
    longest_a = _song_means([(row["song"], row["control"].get("longest_fast_horizontal_run", 0)) for row in complete])
    longest_b = _song_means([(row["song"], row["candidate"].get("longest_fast_horizontal_run", 0)) for row in complete])
    maximum_a, maximum_b = defaultdict(int), defaultdict(int)
    for row in complete:
        maximum_a[row["song"]] = max(maximum_a[row["song"]], row["control"].get("longest_fast_horizontal_run", 0))
        maximum_b[row["song"]] = max(maximum_b[row["song"]], row["candidate"].get("longest_fast_horizontal_run", 0))
    p90_a = float(np.percentile(list(maximum_a.values()), 90)) if maximum_a else None
    p90_b = float(np.percentile(list(maximum_b.values()), 90)) if maximum_b else None
    primary = {**metrics, **dwell_metrics}[target]
    threshold = 0.2 if target == "m1" else 0.15
    if primary["control_mean_distance"] == 0:
        target_passed = primary["harm_passed"] is True
    else:
        target_passed = primary["closure"] is not None and primary["closure"] >= threshold
        if target == "m1":
            target_passed = target_passed and p90_b is not None and p90_b < p90_a
    guards = package_guardrails(complete)
    dwell_failures, dwell_unknown, dwell_lost_support = [], [], []
    for name, measured in dwell_metrics.items():
        lost = [item for item in measured["missing_eligibility"]
                if item["candidate_missing"] and not item["control_missing"]]
        if measured["harm_passed"] is False or lost:
            dwell_failures.append(name)
            dwell_lost_support.extend({**item, "metric": name} for item in lost)
        elif measured["harm_passed"] is None:
            dwell_unknown.append(name)
    guards["axis_dwell"] = {"passed": False if dwell_failures else None if dwell_unknown else True,
        "failed_metrics": dwell_failures, "unverified_metrics": dwell_unknown,
        "lost_eligibility": dwell_lost_support}
    available_metrics = [value for value in metrics.values() if value["pairs"]]
    other_passed = all(value["harm_passed"] is True for name, value in metrics.items() if name != target and value["pairs"])
    eligibility_passed = not any(item["candidate_missing"] and not item["control_missing"]
                                for value in metrics.values() for item in value["missing_eligibility"])
    frozen = frozen_margins is not None and all(value["harm_margin"] is not None for value in available_metrics)
    advance = (bool(complete) and not incomplete and target_passed and other_passed and eligibility_passed
               and frozen and all(value["passed"] is True for value in guards.values()))
    return _jsonable({"schema_version": 1, "decision": "advance_for_playtest" if advance else "experimental",
        "target": target, "required_closure": threshold, "target_passed": bool(target_passed),
        "margins_frozen": frozen, "bootstrap": {"unit": "song", "seed": bootstrap_seed,
            "samples": bootstrap_samples, "confidence": 0.95},
        "pairs": len(complete), "songs": len({row["song"] for row in complete}),
        "incomplete_pairs": incomplete, "metrics": metrics, "dwell_metrics": dwell_metrics,
        "bands": bands, "difficulties": difficulties,
        "longest_fast_horizontal_run": {"control_song_p90": p90_a, "candidate_song_p90": p90_b,
            "control_song_maxima": dict(maximum_a), "candidate_song_maxima": dict(maximum_b),
            "control_song_means": longest_a, "candidate_song_means": longest_b,
            "exploratory_control_song_mean_p90": float(np.percentile(list(longest_a.values()), 90)) if longest_a else None,
            "exploratory_candidate_song_mean_p90": float(np.percentile(list(longest_b.values()), 90)) if longest_b else None},
        "band_migrations": [{"song": row["song"], "difficulty": row.get("difficulty"), "seed": row.get("seed"),
                              "control": row["control"]["band"], "candidate": row["candidate"]["band"]}
                             for row in complete if row["control"]["band"] != row["candidate"]["band"]],
        "guardrails": guards, "pooled_events": _pooled_events(complete),
        "axis_dwell": _pooled_dwell(complete)})


def freeze_run_reference(*args, **kwargs):
    from evaluation.playability_benchmark_run import freeze_run_reference as freeze
    return freeze(*args, **kwargs)


def report_run(*args, **kwargs):
    from evaluation.playability_benchmark_run import report_run as report
    return report(*args, **kwargs)


def main(argv=None):
    from evaluation.playability_benchmark_run import main as run_main
    return run_main(argv)


if __name__ == '__main__':
    main()
