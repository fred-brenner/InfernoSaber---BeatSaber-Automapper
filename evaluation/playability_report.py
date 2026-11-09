"""Playability corpus report — score generated maps against a human baseline.

Reads a v2 ``Info.dat`` package, selects Standard-characteristic
difficulties, and converts v2 ``_notes`` or v3 ``colorNotes`` entries into
``(t_seconds, hand, x, y, d)`` tuples for ``playability_metrics``.
Validation reuses ``evaluation.human_maps`` internals to mirror the
reference loader's Info/timing rules (v2 Info.dat, positive constant BPM,
no offsets or shuffle, in-package paths), then adds its own checks: grid
coordinates, canonical difficulty names, and per-difficulty decode.
Unlike the reference loader it does not require the audio file, and
skipped v2 bombs never reach the seconds check. Any failure raises and
``score_rows`` records the package as skipped — a corpus run never
crashes on one bad package.

Each generated map gets two corpus-relative verdicts per metric M1–M4
(``verdict``): the scalar percentile against human per-map values in its
matched NPS band (all-band fallback below five band maps), and the
incident rate — generated ``units`` at/above the pooled human p95 per
map-minute versus the human rate. A map flags at percentile >= 95 or
rate >= 3x human (a zero human rate flags any positive generated rate).
M5 separately reports low non-reversal share (lower-tail percentile <= 5)
and high fast-reversal events per minute (>= 3x the pooled human rate).
Only maps with eligible M5 transitions enter its NPS-matched baseline;
unavailable M5 comparisons are n/a and tied shares alone never flag.

Run from the repository root:
    python -m evaluation.playability_report \
        --human-root ../Data/training/evaluate_beats \
        --generated ../Data/prediction/new_map [--json report.json]
``--human-root`` resolves ``root/bs_map_input`` when present, else the
root itself; immediate subdirectories are the candidate packages.
``--dataset`` optionally widens the human baseline with a prepared
pipeline_v2 dataset split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluation.human_maps import (
    _check_timing,
    _number,
    _objects,
    _package_file,
    _read_json,
    _version,
)
from evaluation.playability_metrics import INCIDENT_TOP_K, NPS_BANDS, score_map

DIFFICULTY_ORDER = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
_CHARACTERISTIC = "Standard"

# Note field names per beatmap version; v2 ``_type`` doubles as the color
# (0 red/left, 1 blue/right, 3 bomb — bombs are skipped, not notes).
_V2_FIELDS = {"beat": "_time", "x": "_lineIndex", "y": "_lineLayer",
              "color": "_type", "direction": "_cutDirection"}
_V3_FIELDS = {"beat": "b", "x": "x", "y": "y", "color": "c", "direction": "d"}


def _load_info(package):
    """Resolve and validate a package's v2 Info.dat; return its data and BPM.

    Same acceptance rules as ``human_maps.load_reference``: one
    case-insensitive Info.dat, v2 version, positive constant BPM."""
    package = Path(package).expanduser().resolve()
    info_files = [path for path in package.iterdir()
                  if path.name.lower() == "info.dat" and path.is_file()]
    if not info_files:
        raise FileNotFoundError(f"Missing Info.dat in {package}")
    if len(info_files) != 1:
        raise ValueError(f"Ambiguous Info.dat files in {package}")
    info = _read_json(info_files[0])
    _version(info, ("2",), "Info.dat")
    bpm = _number(info.get("_beatsPerMinute"), "BPM")
    if bpm <= 0:
        raise ValueError("BPM must be positive")
    _check_timing(info, bpm)
    return package, info, bpm


def _standard_beatmaps(info):
    """``{difficulty: filename}`` for beatmaps in the Standard group.

    Unknown or duplicated difficulty names are validation failures — a
    Standard beatmap outside the canonical five cannot be ranked."""
    entries = {}
    for group in _objects(info, "_difficultyBeatmapSets"):
        if group.get("_beatmapCharacteristicName") != _CHARACTERISTIC:
            continue
        for beatmap in _objects(group, "_difficultyBeatmaps"):
            difficulty = beatmap.get("_difficulty")
            if difficulty not in DIFFICULTY_ORDER:
                raise ValueError(
                    f"Unsupported {_CHARACTERISTIC} difficulty: {difficulty!r}")
            if difficulty in entries:
                raise ValueError(f"Duplicate {_CHARACTERISTIC}/{difficulty} beatmap")
            filename = beatmap.get("_beatmapFilename")
            if not isinstance(filename, str) or not filename:
                raise ValueError("Expected a nonempty filename in Info.dat")
            entries[difficulty] = filename
    return entries


def list_standard_difficulties(package):
    """Difficulties present in the package's Standard group, canonical order."""
    _, info, _ = _load_info(package)
    present = _standard_beatmaps(info)
    return [name for name in DIFFICULTY_ORDER if name in present]


def _cell(value, label, limit):
    """Grid coordinate: a finite integral number in ``0..limit-1``."""
    number = _number(value, label)
    if not number.is_integer() or not 0 <= number < limit:
        raise ValueError(f"{label} must be an integer in 0..{limit - 1}")
    return int(number)


def _decode_notes(beatmap, version, bpm):
    """Sorted ``(t, hand, x, y, d)`` tuples from the beatmap's note array.

    v2 ``_type`` 3 bombs drop out; every kept note emits one tuple —
    same-hand stacks are collapsed inside ``prepare_notes``, not here."""
    key, fields = ("_notes", _V2_FIELDS) if version == "2" else ("colorNotes", _V3_FIELDS)
    valid_colors = (0, 1, 3) if version == "2" else (0, 1)
    notes = []
    for entry in _objects(beatmap, key):
        beat = _number(entry.get(fields["beat"]), "Note beat time")
        if beat < 0:
            raise ValueError("Note beat time must not be negative")
        color = entry.get(fields["color"])
        if isinstance(color, bool) or color not in valid_colors:
            raise ValueError(f"Unsupported note color/type: {color!r}")
        if color == 3:  # v2 bomb — not a playable note
            continue
        seconds = beat * 60 / bpm
        if not math.isfinite(seconds):
            raise ValueError("Note elapsed time must be finite")
        notes.append((seconds, int(color),
                      _cell(entry.get(fields["x"]), "Note column", 4),
                      _cell(entry.get(fields["y"]), "Note row", 3),
                      _cell(entry.get(fields["direction"]), "Note cut direction", 9)))
    notes.sort(key=lambda note: note[0])
    return notes


def decode_package_notes(package, difficulty=None):
    """Decode one Standard difficulty of a map package into note tuples.

    ``difficulty=None`` selects the highest present in canonical order.
    Returns ``{"notes": [(t_seconds, hand, x, y, d), ...], "bpm": float,
    "difficulty": str}`` sorted by time; raises on any validation failure."""
    package, info, bpm = _load_info(package)
    entries = _standard_beatmaps(info)
    if difficulty is None:
        if not entries:
            raise ValueError(f"Expected one {_CHARACTERISTIC} beatmap; found 0")
        difficulty = next(name for name in reversed(DIFFICULTY_ORDER) if name in entries)
    filename = entries.get(difficulty)
    if filename is None:
        raise ValueError(f"Expected one {_CHARACTERISTIC}/{difficulty} beatmap; found 0")
    beatmap = _read_json(_package_file(package, filename))
    version = _version(beatmap, ("2", "3"), "beatmap")
    _check_timing(beatmap, bpm)
    return {"notes": _decode_notes(beatmap, version, bpm),
            "bpm": bpm, "difficulty": difficulty}


# ---------------------------------------------------------------------------
# Corpus comparison: NPS bands, scored rows, per-metric verdicts.
# ---------------------------------------------------------------------------

METRICS = ("m1", "m2", "m3", "m4")
VERDICT_METRICS = (*METRICS, "m5")
BAND_LABELS = ("<3", "3-5", "5-7", "7+")
BAND_MIN_N = 5          # fewer human maps in-band → all-band scalar fallback
PERCENTILE_FLAG = 95.0  # scalar percentile flag threshold
RATE_FLAG_RATIO = 3.0   # incident-rate flag: generated ≥ 3× the human rate
MIN_NOTES = 2           # fewer playable notes → no meaningful span/metrics


def band_label(nps):
    """NPS band name for a map's measured notes-per-second value."""
    for (low, high), label in zip(NPS_BANDS, BAND_LABELS):
        if low <= nps < high:
            return label
    return BAND_LABELS[-1]


def _scores_from_notes(notes):
    """``score_map`` wrapper rejecting maps too small to score."""
    if len(notes) < MIN_NOTES:
        raise ValueError(
            f"Expected at least {MIN_NOTES} playable notes; found {len(notes)}")
    return score_map(notes)


def score_rows(packages, per_difficulty):
    """Score every package; failures become skipped entries, never raise.

    ``per_difficulty`` scores every listed Standard difficulty (generated
    side); otherwise only the highest present difficulty is decoded (human
    side). Rows are ``{"package", "difficulty", "scores"}`` where scores is
    the ``score_map`` result; skipped entries are ``{"package", "error"}``."""
    rows, skipped = [], []
    for package in packages:
        package = Path(package)
        try:
            difficulties = (list_standard_difficulties(package)
                            if per_difficulty else [None])
            if not difficulties:
                raise ValueError(f"Expected one {_CHARACTERISTIC} beatmap; found 0")
        except Exception as error:
            skipped.append({"package": package.name, "error": str(error)})
            continue
        for difficulty in difficulties:
            try:
                decoded = decode_package_notes(package, difficulty)
                rows.append({"package": package.name,
                             "difficulty": decoded["difficulty"],
                             "scores": _scores_from_notes(decoded["notes"])})
            except Exception as error:
                label = f"{difficulty}: " if per_difficulty else ""
                skipped.append({"package": package.name,
                                "error": f"{label}{error}"})
    return rows, skipped


def _metric_units(rows, metric):
    """Concatenated ``units`` pool across scored rows.

    Units are pooled verbatim: M1/M2/M4 already store positive
    contributions only, while M3 legitimately emits zero-divergence
    segments — dropping those zeros would push the pooled p95 threshold
    up and under-report incident rates."""
    pools = [np.asarray(row["scores"][metric]["units"], dtype=float)
             for row in rows]
    pools = [pool for pool in pools if pool.size]
    return np.concatenate(pools) if pools else np.empty(0)


def _incident_rate(units, threshold, minutes):
    """Units at/above the pooled human p95, expressed per map-minute."""
    if threshold is None or minutes <= 0:
        return 0.0
    return float((np.asarray(units, dtype=float) >= threshold).sum() / minutes)


def _m5_eligible(scores):
    """Usable direction summary; old cached rows and no-transition maps omit it."""
    metric = scores.get("m5")
    return bool(metric and metric.get("transitions", 0) > 0
                and metric.get("non_reversal_share") is not None
                and metric.get("fast_reversals_per_minute") is not None
                and metric.get("fast_reversals") is not None
                and scores["span_seconds"] > 0)


def _m5_human_rate(rows):
    """Fast-reversal events per minute from eligible counts and full map spans."""
    minutes = sum(row["scores"]["span_seconds"] for row in rows) / 60
    return (float(sum(row["scores"]["m5"]["fast_reversals"] for row in rows)
                  / minutes) if minutes > 0 else None)


def _m5_verdict(gen_scores, human_rows, gen_band):
    """Low share and high event rate use the same eligible reference maps."""
    eligible = [row for row in human_rows if _m5_eligible(row["scores"])]
    band_rows = [row for row in eligible
                 if band_label(row["scores"]["nps"]) == gen_band]
    reference = band_rows if len(band_rows) >= BAND_MIN_N else eligible
    metric = gen_scores.get("m5") or {}
    result = {"percentile": None, "band": gen_band, "band_n": len(band_rows),
              "reference_n": len(reference), "incident_rate": None,
              "human_incident_rate": None, "flag": False,
              "percentile_direction": "lower", "incident_rate_direction": "higher",
              "non_reversal_share": metric.get("non_reversal_share"),
              "fast_reversals_per_minute": metric.get("fast_reversals_per_minute"),
              "non_reversal_share_flag": False, "fast_reversal_rate_flag": False}
    if not reference or not _m5_eligible(gen_scores):
        return result
    scalars = np.asarray([row["scores"]["m5"]["non_reversal_share"]
                          for row in reference], dtype=float)
    # Count ties as baseline observations: a shared distribution (including
    # all zeros) gets percentile 100 rather than a false low-tail flag.
    percentile = float(100 * (scalars <= metric["non_reversal_share"]).sum()
                       / scalars.size)
    human_rate = _m5_human_rate(reference)
    incident_rate = float(metric["fast_reversals_per_minute"])
    share_flag = percentile <= 100 - PERCENTILE_FLAG
    rate_flag = (incident_rate > 0 if human_rate == 0
                 else incident_rate >= RATE_FLAG_RATIO * human_rate)
    result.update(percentile=percentile, incident_rate=incident_rate,
                  human_incident_rate=human_rate,
                  non_reversal_share_flag=bool(share_flag),
                  fast_reversal_rate_flag=bool(rate_flag),
                  flag=bool(share_flag or rate_flag))
    return result


def verdict(gen_scores, human_rows):
    """Corpus-relative verdicts for one generated map versus human rows.

    Returns ``{metric: {"percentile", "band", "band_n", "incident_rate",
    "human_incident_rate", "flag"}}`` for m1..m4. The scalar percentile is
    the share of human per-map values ``<=`` the generated value within
    the matched NPS band, falling back to all-band when the band holds
    fewer than ``BAND_MIN_N`` human maps. The incident rate counts
    generated units at/above the pooled human p95 per map-minute; the
    human rate applies the same formula over all human units and spans.
    Flag: percentile >= 95 or rate >= 3x human — a zero human rate flags
    any positive generated rate. M5 retains the same verdict keys, with
    a lower-tail share percentile and fast-event rate from eligible NPS
    reference rows; unavailable values are None and never flag."""
    gen_band = band_label(gen_scores["nps"])
    band_rows = [row for row in human_rows
                 if band_label(row["scores"]["nps"]) == gen_band]
    band_n = len(band_rows)
    reference = band_rows if band_n >= BAND_MIN_N else human_rows
    human_minutes = sum(row["scores"]["span_seconds"]
                        for row in human_rows) / 60
    gen_minutes = gen_scores["span_seconds"] / 60
    result = {}
    for metric in METRICS:
        scalars = np.asarray([row["scores"][metric]["value"]
                              for row in reference])
        value = float(gen_scores[metric]["value"])
        percentile = (float(100 * (scalars <= value).sum() / scalars.size)
                      if scalars.size else 0.0)
        pooled = _metric_units(human_rows, metric)
        threshold = float(np.percentile(pooled, 95)) if pooled.size else None
        incident_rate = _incident_rate(
            gen_scores[metric]["units"], threshold, gen_minutes)
        human_rate = _incident_rate(pooled, threshold, human_minutes)
        flag = (percentile >= PERCENTILE_FLAG
                or (incident_rate > 0 if human_rate == 0
                    else incident_rate >= RATE_FLAG_RATIO * human_rate))
        result[metric] = {"percentile": percentile, "band": gen_band,
                          "band_n": band_n, "incident_rate": incident_rate,
                          "human_incident_rate": human_rate,
                          "flag": bool(flag)}
    result["m5"] = _m5_verdict(gen_scores, human_rows, gen_band)
    return result


# ---------------------------------------------------------------------------
# CLI: package discovery, optional dataset baseline, console + JSON report.
# ---------------------------------------------------------------------------

def _package_dirs(root):
    """Immediate subdirectories of a corpus root — each a candidate package."""
    return [path for path in sorted(Path(root).iterdir()) if path.is_dir()]


def _human_root(root):
    """``root/bs_map_input`` when present, else ``root`` itself."""
    nested = Path(root) / "bs_map_input"
    return nested if nested.is_dir() else Path(root)


def _dataset_rows(root, split):
    """Extra human rows from a prepared pipeline_v2 dataset.

    Manifest songs carry retained BEAT-timed notes; convert to seconds via
    the song BPM before scoring. ``pattern_analysis`` is imported lazily —
    it is the heavier sibling module."""
    from evaluation import pattern_analysis

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows, skipped = [], []
    for entry in manifest.get("songs", []):
        if split != "all" and entry.get("split") != split:
            continue
        try:
            meta = json.loads(
                (root / entry["metadata_file"]).read_text(encoding="utf-8"))
            metadata = meta["metadata"]
            bpm = float(metadata["bpm"])
            beats = pattern_analysis.decode_human_song(root / entry["file"], bpm)
            notes = [(beat * 60 / bpm, hand, x, y, d)
                     for beat, hand, x, y, d in beats]
            difficulty = (metadata.get("provenance") or {}).get("difficulty")
            rows.append({"package": metadata.get("song_name")
                         or entry.get("song_id", "?"),
                         "difficulty": difficulty or "dataset",
                         "scores": _scores_from_notes(notes)})
        except Exception as error:
            skipped.append({"package": entry.get("song_id", "?"),
                            "error": str(error)})
    return rows, skipped


def _jsonable(value):
    """Recursively convert numpy arrays/scalars and tuples for JSON output."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _human_quantiles(human_rows):
    """Per-band legacy p50/p95 and both named M5 score quantiles."""
    quantiles = {}
    for label in BAND_LABELS:
        subset = [row for row in human_rows
                  if band_label(row["scores"]["nps"]) == label]
        if not subset:
            continue
        bands = {"maps": len(subset)}
        for metric in METRICS:
            values = np.asarray([row["scores"][metric]["value"]
                                 for row in subset])
            bands[metric] = {"p50": float(np.percentile(values, 50)),
                             "p95": float(np.percentile(values, 95))}
        eligible = [row for row in subset if _m5_eligible(row["scores"])]
        bands["m5"] = {"maps": len(eligible)}
        for name in ("non_reversal_share", "fast_reversals_per_minute"):
            values = [row["scores"]["m5"][name] for row in eligible]
            bands["m5"][name] = {
                f"p{percentile}": float(np.percentile(values, percentile)) if values else None
                for percentile in (5, 50, 95)}
        quantiles[label] = bands
    return quantiles


def _human_rate_summary(human_rows):
    """Legacy pooled p95/rates plus directly counted M5 fast events/min."""
    minutes = sum(row["scores"]["span_seconds"] for row in human_rows) / 60
    thresholds, rates = {}, {}
    for metric in METRICS:
        pooled = _metric_units(human_rows, metric)
        threshold = float(np.percentile(pooled, 95)) if pooled.size else None
        thresholds[metric] = threshold
        rates[metric] = _incident_rate(pooled, threshold, minutes)
    # M5 counts named fast events directly; binary units have no p95 threshold.
    thresholds["m5"] = None
    rates["m5"] = _m5_human_rate(
        [row for row in human_rows if _m5_eligible(row["scores"])])
    return thresholds, rates


def _display_number(value, fmt=".2f"):
    return "n/a" if value is None else format(value, fmt)


def _print_human(human_root, human_rows, quantiles, thresholds, rates):
    print(f"=== HUMAN BASELINE: {len(human_rows)} maps from {human_root} ===")
    for label in BAND_LABELS:
        bands = quantiles.get(label)
        if bands is None:
            continue
        cells = "  ".join(f"{metric.upper()} {bands[metric]['p50']:5.2f}/"
                          f"{bands[metric]['p95']:5.2f}" for metric in METRICS)
        print(f"  {label:>3} n={bands['maps']:<3} {cells}  (scalar p50/p95)")
        m5 = bands["m5"]
        share, fast = m5["non_reversal_share"], m5["fast_reversals_per_minute"]
        print(f"      M5 n={m5['maps']} non-reversal share (low) "
              f"{_display_number(share['p5'], '.2%')}/{_display_number(share['p50'], '.2%')} "
              f"(p5/p50) | fast reversals/min (high) "
              f"{_display_number(fast['p50'])}/{_display_number(fast['p95'])} (p50/p95)")
    cells = []
    for metric in METRICS:
        threshold = thresholds[metric]
        text = f"{threshold:.2f}" if threshold is not None else "n/a"
        cells.append(f"{metric.upper()} {text:>5}/{rates[metric]:.2f}")
    print(f"  pooled p95 / human rate per min: {'  '.join(cells)}")
    print(f"  M5 pooled fast reversals/min (high): {_display_number(rates['m5'])} "
          "(events / eligible map-minutes)")


def _print_generated(gen_rows, verdicts):
    for row, result in zip(gen_rows, verdicts):
        scores = row["scores"]
        cells = []
        for metric in METRICS:
            entry = result[metric]
            mark = "*" if entry["band_n"] < BAND_MIN_N else " "
            cells.append(f"{metric.upper()} {scores[metric]['value']:5.2f} "
                         f"p{entry['percentile']:3.0f}{mark} "
                         f"{entry['incident_rate']:.1f}/"
                         f"{entry['human_incident_rate']:.1f}")
        m5 = result["m5"]
        mark = "*" if m5["band_n"] < BAND_MIN_N else ""
        percentile = ("n/a" if m5["percentile"] is None
                      else f"p{m5['percentile']:.0f}{mark}")
        cells.append(f"M5 non-reversal share (low) "
                     f"{_display_number(m5['non_reversal_share'], '.2%')} {percentile} | "
                     f"fast reversals/min (high) "
                     f"{_display_number(m5['fast_reversals_per_minute'])}/"
                     f"{_display_number(m5['human_incident_rate'])} "
                     f"(reference n={m5['reference_n']})")
        flagged = [metric.upper() for metric in VERDICT_METRICS
                   if result[metric]["flag"]]
        tag = f"  FLAG {','.join(flagged)}" if flagged else ""
        print(f"  {row['package'][:28]:<28} {row['difficulty']:<10} "
              f"nps {scores['nps']:4.1f} | " + " | ".join(cells) + tag)
        diagnostics = (scores.get("m5") or {}).get("diagnostics")
        if diagnostics:
            overall = diagnostics["overall"]
            counts = "/".join(str(overall["angles"][angle]["count"])
                              for angle in ("0", "45", "90", ">=135"))
            dwell = overall["non_reversal_intervals"]
            print(f"      Cut turns (0/45/90/>=135 deg): {counts}  "
                  f"median between non-reversals: "
                  f"{_display_number(dwell['transitions']['median'], '.1f')} transitions / "
                  f"{_display_number(dwell['seconds']['median'])} s")
    if gen_rows:
        print(f"  (* = all-band reference: <{BAND_MIN_N} eligible human maps in "
              f"band; M1-M4 columns are value percentile rate/human-rate)")
        print("  (M5 flags lower-tail share p<=5 or fast rate >=3x human; "
              "n/a = missing M5, no eligible transitions or no eligible human baseline)")


def _print_incidents(flagged_rows, top_k):
    print(f"\n=== TOP INCIDENTS (k={top_k} of {INCIDENT_TOP_K} max) ===")
    if not any(row["scores"][metric]["incidents"]
               for row in flagged_rows for metric in ("m4", "m2")):
        print("  (no recorded M4/M2 incidents)")
        return
    for row in flagged_rows:
        scores = row["scores"]
        label = f"{row['package']} {row['difficulty']}"
        for event in scores["m4"]["incidents"][:top_k]:
            print(f"  {label} M4 t={event['t']:.2f}s sev={event['severity']:.2f} "
                  f"cells={event['cells']} a={event['a']} b={event['b']} "
                  f"crossover={event['crossover']:.2f}")
        for event in scores["m2"]["incidents"][:top_k]:
            print(f"  {label} M2 t={event['t']:.2f}s occl={event['occl']:.2f} "
                  f"note={event['note']} occluder={event['occluder']} "
                  f"dt={event['dt']:.2f}s")


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--human-root", type=Path, required=True,
                         help="evaluate_beats root (resolves bs_map_input) or "
                              "the human package folder itself")
    command.add_argument("--generated", type=Path,
                         help="Folder of generated map packages; every "
                              "Standard difficulty is scored")
    command.add_argument("--dataset", type=Path,
                         help="Optional prepared pipeline_v2 dataset added "
                              "to the human baseline")
    command.add_argument("--split", default="train",
                         help="Dataset split to add (or 'all')")
    command.add_argument("--difficulty", choices=DIFFICULTY_ORDER,
                         help="Score only this generated difficulty")
    command.add_argument("--json", type=Path,
                         help="Write the machine-readable report here")
    command.add_argument("--top-k", type=int, default=5,
                         help="Incidents listed per flagged map "
                              f"(metric lists cap at {INCIDENT_TOP_K})")
    args = command.parse_args(argv)
    if args.top_k < 0:
        command.error("--top-k must not be negative")

    human_root = _human_root(args.human_root)
    if not human_root.is_dir():
        command.error(f"Human package root not found: {human_root}")
    if args.generated is not None and not args.generated.is_dir():
        command.error(f"Generated package folder not found: {args.generated}")

    human_rows, human_skipped = score_rows(
        _package_dirs(human_root), per_difficulty=False)
    dataset_skipped = []
    if args.dataset is not None:
        manifest = args.dataset / "manifest.json"
        if not manifest.is_file():
            command.error(f"Dataset manifest not found: {manifest}")
        try:
            extra_rows, dataset_skipped = _dataset_rows(args.dataset, args.split)
        except Exception as error:
            command.error(f"Could not read dataset manifest: {error}")
        human_rows += extra_rows

    gen_rows, gen_skipped = [], []
    if args.generated is not None:
        gen_rows, gen_skipped = score_rows(
            _package_dirs(args.generated), per_difficulty=True)
        if args.difficulty is not None:
            gen_rows = [row for row in gen_rows
                        if row["difficulty"] == args.difficulty]

    quantiles = _human_quantiles(human_rows)
    thresholds, rates = _human_rate_summary(human_rows)
    _print_human(human_root, human_rows, quantiles, thresholds, rates)

    print(f"\n=== GENERATED: {len(gen_rows)} maps ===")
    verdicts = [verdict(row["scores"], human_rows) for row in gen_rows]
    _print_generated(gen_rows, verdicts)

    generated_report = [
        {"package": row["package"], "difficulty": row["difficulty"],
         "nps": row["scores"]["nps"],
         "flagged": [metric.upper() for metric in VERDICT_METRICS
                     if result[metric]["flag"]],
         "verdict": result, "scores": row["scores"]}
        for row, result in zip(gen_rows, verdicts)]

    skipped_all = ([("human", item) for item in human_skipped]
                   + [("dataset", item) for item in dataset_skipped]
                   + [("generated", item) for item in gen_skipped])
    print(f"\n=== SKIPPED: {len(skipped_all)} ===")
    for source, item in skipped_all:
        print(f"  [{source}] {item['package']}: {item['error']}")

    flagged_rows = [row for row, result in zip(gen_rows, verdicts)
                    if any(result[metric]["flag"] for metric in VERDICT_METRICS)]
    if flagged_rows:
        _print_incidents(flagged_rows, args.top_k)

    report = {"human": {"root": str(human_root), "maps": len(human_rows),
                        "quantiles": quantiles, "incident_p95": thresholds,
                        "incident_rates": rates, "rows": human_rows},
              "generated": generated_report,
              "skipped": {"human": human_skipped, "dataset": dataset_skipped,
                          "generated": gen_skipped}}
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_jsonable(report), indent=2),
                             encoding="utf-8")
        print(f"\nReport written to {args.json}")
    return report


if __name__ == "__main__":
    main()
