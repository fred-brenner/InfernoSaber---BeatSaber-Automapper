"""Pattern mining for note sequences: human training data vs generated maps.

Reads the prepared pipeline_v2 dataset (retained note contract, seconds) and
generated map packages (v3 colorNotes, beats), then reports transition and
movement-pattern statistics on a comparable footing. All measurements are
expressed in beats so maps with different BPM stay comparable.

Run from the repository root:
    python evaluation/pattern_analysis.py \
        --dataset ../Data/training/pipeline_v2/full/dataset \
        --generated ../Data/prediction/new_map
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

# Beat Saber cut directions: 0 up, 1 down, 2 left, 3 right,
# 4 up-left, 5 up-right, 6 down-left, 7 down-right, 8 dot.
DIR_VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
           4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}
DIR_MIRROR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}
DIR_NAMES = {0: "up", 1: "down", 2: "left", 3: "right",
             4: "up-left", 5: "up-right", 6: "down-left", 7: "down-right", 8: "dot"}

# Consecutive same-hand notes within this gap are treated as one phrase;
# longer pauses break movement patterns.
FLOW_GAP_BEATS = 1.0
STREAM_GAP_BEATS = 0.5


def decode_human_song(npz_path, bpm):
    """Retained notes from a prepared song: (beat, hand, x, y, direction)."""
    arrays = np.load(npz_path)
    times = arrays["base::event_times"]
    classes = arrays["base::note_classes"]
    notes = []
    for event, when in enumerate(times):
        for hand in (0, 1):
            category = int(classes[event, hand])
            if not category:
                continue
            position, direction = divmod(category - 1, 9)
            notes.append((float(when) * bpm / 60., hand, position % 4, position // 4, direction))
    return notes


def decode_generated_map(beatmap_path):
    """colorNotes from a v3 beatmap, deduplicated to one note per (beat, hand)."""
    with open(beatmap_path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    raw = data.get("colorNotes")
    if not isinstance(raw, list):
        return []
    notes, seen = [], set()
    for note in sorted(raw, key=lambda n: (n.get("b", 0), n.get("c", 0))):
        beat, hand = note.get("b"), note.get("c")
        x, y, direction = note.get("x"), note.get("y"), note.get("d")
        if hand not in (0, 1) or None in (beat, x, y, direction):
            continue
        key = (round(float(beat), 4), hand)
        if key in seen:
            continue
        seen.add(key)
        notes.append((float(beat), hand, int(x), int(y), int(direction)))
    return notes


def _sign(value):
    return (value > 0) - (value < 0)


def _dir_angle_change(previous, current):
    """Cut-direction rotation between two directed notes, in degrees."""
    if previous == 8 or current == 8:
        return "dot"
    px, py = DIR_VEC[previous]
    cx, cy = DIR_VEC[current]
    cosine = (px * cx + py * cy) / math.hypot(px, py) / math.hypot(cx, cy)
    return int(round(math.degrees(math.acos(max(-1., min(1., cosine)))) / 45)) * 45


def _sign_runs(values):
    """Run lengths (in transitions) of consecutive nonzero same-sign steps."""
    runs, current, sign = [], 0, 0
    for value in values:
        step = _sign(value)
        if step and step == sign:
            current += 1
        else:
            if current:
                runs.append(current)
            current, sign = (1, step) if step else (0, 0)
    if current:
        runs.append(current)
    return runs


def _alternating_runs(values):
    """Run lengths (in transitions) of nonzero sign-alternating steps."""
    runs, current, previous = [], 0, 0
    for value in values:
        step = _sign(value)
        if step and previous and step == -previous:
            current += 1
        else:
            if current >= 2:
                runs.append(current)
            current = 1 if step else 0
        previous = step
    if current >= 2:
        runs.append(current)
    return runs


def _flag_runs(flags):
    """Run lengths (in transitions) of consecutive True flags."""
    runs, current = [], 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _segments(seq):
    """Split one hand's (beat, x, y, d) sequence into gap-connected blocks."""
    segments, current = [], []
    for index, note in enumerate(seq):
        if index and note[0] - seq[index - 1][0] <= FLOW_GAP_BEATS:
            current.append(note)
        else:
            if len(current) > 1:
                segments.append(current)
            current = [note]
    if len(current) > 1:
        segments.append(current)
    return segments


def analyze_map(notes):
    """Transition and pattern statistics for one map's note list."""
    notes = sorted(notes)
    events = {}
    for beat, hand, x, y, direction in notes:
        events.setdefault(beat, {})[hand] = (x, y, direction)
    times = sorted(events)

    stats = Counter()
    dir_matrix = Counter()
    hand_bigrams = Counter()
    transitions = []

    # Event level: hand signatures L/R/D, doubles, mirrored doubles.
    signatures = []
    for beat in times:
        hands = events[beat]
        signatures.append("D" if len(hands) == 2 else ("L" if 0 in hands else "R"))
        if len(hands) == 2:
            stats["doubles"] += 1
            lx, ly, ld = hands[0]
            rx, ry, rd = hands[1]
            if lx + rx == 3 and ly == ry and DIR_MIRROR[ld] == rd:
                stats["mirrored_doubles"] += 1
    stats["events"], stats["notes"] = len(times), len(notes)
    for prev, cur in zip(signatures, signatures[1:]):
        hand_bigrams[(prev, cur)] += 1
    for index in range(len(times) - 1):
        if signatures[index] == "D" or signatures[index + 1] == "D":
            continue
        if times[index + 1] - times[index] <= FLOW_GAP_BEATS:
            stats["single_pairs"] += 1
            stats["alternating_pairs"] += int(signatures[index] != signatures[index + 1])

    # Cross-hand drift: consecutive single-note events (any hand) marching in x.
    singles = [(beat, next(iter(events[beat].values()))) for beat in times
               if len(events[beat]) == 1]
    for segment in _segments([(beat, pos[0], pos[1], pos[2]) for beat, pos in singles]):
        dx = [b[1] - a[1] for a, b in zip(segment, segment[1:])]
        for run in _sign_runs(dx):
            stats["xhand_sweep_runs2+"] += int(run >= 2)
            stats["xhand_sweep_runs3+"] += int(run >= 3)
            stats["xhand_sweep_notes3+"] += (run + 1) * int(run >= 3)

    # Hand level: connected blocks, transition descriptors, pattern runs.
    run_kinds = ("h_sweep", "v_sweep", "x_wave", "y_wave", "same_dir", "dir_step")
    for hand in (0, 1):
        seq = [(beat, *events[beat][hand]) for beat in times if hand in events[beat]]
        for segment in _segments(seq):
            dx = [b[1] - a[1] for a, b in zip(segment, segment[1:])]
            dy = [b[2] - a[2] for a, b in zip(segment, segment[1:])]
            gaps = [b[0] - a[0] for a, b in zip(segment, segment[1:])]
            changes = [_dir_angle_change(a[3], b[3]) for a, b in zip(segment, segment[1:])]
            runs = {"h_sweep": _sign_runs(dx), "v_sweep": _sign_runs(dy),
                    "x_wave": _alternating_runs(dx), "y_wave": _alternating_runs(dy),
                    "same_dir": _flag_runs([a[3] == b[3] != 8 for a, b in zip(segment, segment[1:])]),
                    "dir_step": _flag_runs([change == 45 for change in changes])}
            for kind in run_kinds:
                stats[f"{kind}_runs2+"] += sum(1 for run in runs[kind] if run >= 2)
                stats[f"{kind}_runs3+"] += sum(1 for run in runs[kind] if run >= 3)
                stats[f"{kind}_notes3+"] += sum(run + 1 for run in runs[kind] if run >= 3)
            for index in range(len(dx)):
                a, b = segment[index], segment[index + 1]
                transitions.append({"hand": hand, "gap": gaps[index], "dx": dx[index], "dy": dy[index],
                                    "prev_d": a[3], "d": b[3], "change": changes[index]})
                dir_matrix[(a[3], b[3])] += 1
                stats["transitions"] += 1
                stats["fast_transitions"] += int(gaps[index] <= STREAM_GAP_BEATS)

    for item in transitions:
        stats["dx_stay"] += int(item["dx"] == 0)
        stats[f"dx_abs{min(abs(item['dx']), 3)}"] += 1
        stats["dy_stay"] += int(item["dy"] == 0)
        stats[f"dy_abs{min(abs(item['dy']), 2)}"] += 1
        stats[f"manhattan{min(abs(item['dx']) + abs(item['dy']), 3)}"] += 1
        stats[f"dirchange_{item['change']}"] += 1
        pvx, pvy = DIR_VEC[item["prev_d"]]
        nvx, nvy = DIR_VEC[item["d"]]
        stats["follow_prev"] += int(item["dx"] * pvx + item["dy"] * pvy > 0)
        stats["against_prev"] += int(item["dx"] * pvx + item["dy"] * pvy < 0)
        stats["into_next"] += int(item["dx"] * nvx + item["dy"] * nvy > 0)
        stats["against_next"] += int(item["dx"] * nvx + item["dy"] * nvy < 0)

    # Marginals: cell and direction usage per hand.
    cells, directions = Counter(), Counter()
    for beat, hand, x, y, direction in notes:
        cells[(hand, x, y)] += 1
        directions[(hand, direction)] += 1

    return {"stats": stats, "dir_matrix": dir_matrix, "hand_bigrams": hand_bigrams,
            "cells": cells, "directions": directions}


def merge_results(results):
    total, matrix, bigrams = Counter(), Counter(), Counter()
    cells, directions = Counter(), Counter()
    for result in results:
        total.update(result["stats"])
        matrix.update(result["dir_matrix"])
        bigrams.update(result["hand_bigrams"])
        cells.update(result["cells"])
        directions.update(result["directions"])
    return total, matrix, bigrams, cells, directions


def pct(part, whole):
    return 100. * part / whole if whole else 0.


def report_corpus(name, results):
    total, matrix, bigrams, cells, directions = merge_results(results)
    events, notes, transitions = total["events"], total["notes"], total["transitions"]
    print(f"\n=== {name}: {len(results)} maps, {events} events, {notes} notes, "
          f"{transitions} connected transitions ({pct(total['fast_transitions'], transitions):.0f}% fast) ===")
    print(f"doubles/event {pct(total['doubles'], events):5.1f}%  mirrored {pct(total['mirrored_doubles'], total['doubles']):5.1f}%  "
          f"single-note alternation {pct(total['alternating_pairs'], total['single_pairs']):5.1f}%")
    print("|dx| stay {:4.1f}% 1 {:4.1f}% 2 {:4.1f}% 3 {:4.1f}% | |dy| stay {:4.1f}% 1 {:4.1f}% 2+ {:4.1f}% | "
          "manhattan 0 {:4.1f}% 1 {:4.1f}% 2 {:4.1f}% 3+ {:4.1f}%".format(
              pct(total["dx_stay"], transitions), pct(total["dx_abs1"], transitions),
              pct(total["dx_abs2"], transitions), pct(total["dx_abs3"], transitions),
              pct(total["dy_stay"], transitions), pct(total["dy_abs1"], transitions),
              pct(total["dy_abs2"], transitions), pct(total["manhattan0"], transitions),
              pct(total["manhattan1"], transitions), pct(total["manhattan2"], transitions),
              pct(total["manhattan3"], transitions)))
    dsum = sum(total[f"dirchange_{key}"] for key in (0, 45, 90, 135, 180, "dot"))
    print("dir-change same {:4.1f}% 45 {:4.1f}% 90 {:4.1f}% 135 {:4.1f}% 180 {:4.1f}% dot {:4.1f}%".format(
        pct(total["dirchange_0"], dsum), pct(total["dirchange_45"], dsum), pct(total["dirchange_90"], dsum),
        pct(total["dirchange_135"], dsum), pct(total["dirchange_180"], dsum), pct(total["dirchange_dot"], dsum)))
    print("movement vs cuts: follow-prev {:4.1f}% against-prev {:4.1f}% into-next {:4.1f}% against-next {:4.1f}%".format(
        pct(total["follow_prev"], transitions), pct(total["against_prev"], transitions),
        pct(total["into_next"], transitions), pct(total["against_next"], transitions)))
    for key, label in (("h_sweep", "horizontal sweep"), ("v_sweep", "vertical sweep"),
                       ("x_wave", "x zigzag wave"), ("y_wave", "y zigzag wave"),
                       ("same_dir", "same-dir repeat"), ("dir_step", "45-degree step"),
                       ("xhand_sweep", "cross-hand x drift")):
        print(f"{label:>17}: runs>=2 /100ev {total[f'{key}_runs2+'] / max(events, 1) * 100:5.2f}  "
              f"runs>=3 /100ev {total[f'{key}_runs3+'] / max(events, 1) * 100:5.2f}  "
              f"notes in >=3: {pct(total[f'{key}_notes3+'], notes):5.2f}%")
    dir_sum = sum(directions.values())
    top = ", ".join(f"{DIR_NAMES[d]} {pct(c, dir_sum):4.1f}" for (_, d), c in directions.most_common(9))
    print(f"directions: {top}")
    grid = [[sum(cells.get((hand, x, y), 0) for hand in (0, 1)) for x in range(4)] for y in range(2, -1, -1)]
    total_cells = sum(cells.values())
    for row in grid:
        print("   cells: " + "  ".join(f"{pct(value, total_cells):4.1f}%" for value in row))
    return total, matrix, bigrams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--split", default="train", help="dataset split to analyze (or 'all')")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    human, difficulties = [], []
    for entry in manifest["songs"]:
        if args.split != "all" and entry.get("split") != args.split:
            continue
        meta = json.loads((args.dataset / entry["metadata_file"]).read_text())
        notes = decode_human_song(args.dataset / entry["file"], float(meta["metadata"]["bpm"]))
        if notes:
            human.append((analyze_map(notes), float(meta["metadata"].get("difficulty_nps", 0))))
    report_corpus(f"HUMAN ({args.split})", [result for result, _ in human])
    for low, high in ((0, 3), (3, 5), (5, 7), (7, 100)):
        subset = [result for result, nps in human if low <= nps < high]
        if subset:
            report_corpus(f"HUMAN nps {low}-{high}", subset)

    generated = {}
    for package in sorted(args.generated.iterdir()):
        if not package.is_dir():
            continue
        for beatmap in package.glob("*.dat"):
            if beatmap.stem.lower() in ("info", "metadata"):
                continue
            notes = decode_generated_map(beatmap)
            if notes:
                generated.setdefault(beatmap.stem, []).append(analyze_map(notes))
    for difficulty in sorted(generated):
        report_corpus(f"GENERATED {difficulty}", generated[difficulty])
    all_generated = [result for results in generated.values() for result in results]
    if len(generated) > 1:
        report_corpus("GENERATED (all)", all_generated)

    # Direction bigrams: where does generated diverge from human?
    _, human_matrix, _, _, _ = merge_results([result for result, _ in human])
    _, gen_matrix, _, _, _ = merge_results(all_generated)
    human_sum, gen_sum = sum(human_matrix.values()), sum(gen_matrix.values())
    print("\n=== direction bigrams most under-used by generation (human share >= 0.5%) ===")
    rows = sorted(((count / human_sum - gen_matrix.get(pair, 0) / gen_sum, pair,
                    count / human_sum, gen_matrix.get(pair, 0) / gen_sum)
                   for pair, count in human_matrix.items() if count / human_sum >= 0.005),
                  reverse=True)
    for delta, (prev, cur), hs, gs in rows[:15]:
        print(f"{DIR_NAMES[prev]:>9} -> {DIR_NAMES[cur]:<9} human {hs * 100:5.2f}%  gen {gs * 100:5.2f}%  "
              f"ratio {hs / max(gs, 1e-9):5.2f}x")


if __name__ == "__main__":
    main()
