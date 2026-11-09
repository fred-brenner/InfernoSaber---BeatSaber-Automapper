"""Direction DEVIATION from the expected 180deg reversal, per same-hand pair.

divergence = 180 - circ_delta(prev_dir, dir), in [0,180]
  0deg   = clean reversal (default alternation)
  45deg  = raw delta 135
  90deg  = raw delta 90
  >=135deg = raw delta <=45 (includes same-direction repeats = 180deg dev)

Per bucket: share of transitions, rate/min, median gap, and dwell =
transitions/seconds between consecutive deviation events (dev >= 22.5).
"""
import sys
from pathlib import Path

import numpy as np

from evaluation.playability_metrics import prepare_notes
from evaluation.playability_report import (
    decode_package_notes, list_standard_difficulties)

DIR_ANGLE = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}


def circ(a, b):
    return min(abs(a - b), 360 - abs(a - b))


def events(prepared):
    """Yield (dev_deg, gap, t) per same-hand transition."""
    out = []
    for hand in (0, 1):
        ns = [n for n in prepared if n["hand"] == hand]
        for a, b in zip(ns, ns[1:]):
            if a["eff"] is None or b["eff"] is None:
                continue
            d = circ(DIR_ANGLE[a["eff"]], DIR_ANGLE[b["eff"]])
            out.append((180 - d, b["t"] - a["t"], b["t"]))
    return out


def collect(packages, all_diffs):
    evs, span = [], 0.0
    for pkg in packages:
        for d in (list_standard_difficulties(pkg) if all_diffs else [None]):
            try:
                r = decode_package_notes(pkg, d if all_diffs else None)
            except Exception:
                continue
            p = prepare_notes(r["notes"])
            if not p:
                continue
            span += p[-1]["t"] - p[0]["t"]
            evs += events(p)
    return evs, span


def report(evs, span, label):
    n = max(len(evs), 1)
    print(f"\n===== {label} ({n} transitions, {span/60:.0f} min) =====")
    for lo, hi, name in ((0, 22.5, "0 (clean rev)"), (22.5, 67.5, "45"),
                         (67.5, 112.5, "90"), (112.5, 181, ">=135")):
        v = [(dev, gap) for dev, gap, _ in evs if lo <= dev < hi]
        if v:
            gaps = [g for _, g in v]
            print(f"  dev {name:14s} {100*len(v)/n:5.1f}%"
                  f"  {len(v)/(span/60):6.1f}/min"
                  f"  gap p25/p50/p75 {np.percentile(gaps,[25,50,75]).round(2)}")
    # dwell: between consecutive DEVIATION events (dev>=22.5), per corpus
    devs = sorted((t, dev) for dev, _, t in evs if dev >= 22.5)
    if len(devs) > 1:
        dt = np.diff([t for t, _ in devs])
        print(f"deviation events (dev>=22.5): {len(devs)}"
              f"  median {np.median(dt):.2f}s apart"
              f"  mean {np.mean(dt):.2f}s")
    # also: dwell between >=135 deviations (true breaks)
    big = sorted(t for dev, _, t in evs if dev >= 112.5)
    if len(big) > 1:
        dt = np.diff(big)
        print(f"big deviations (dev>=135): {len(big)}"
              f"  median {np.median(dt):.2f}s apart")


human_root = Path(sys.argv[1]); gen_root = Path(sys.argv[2])
n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 60
human_pkgs = sorted(p for p in human_root.iterdir() if p.is_dir())
gen_pkgs = sorted(p for p in gen_root.iterdir() if p.is_dir())[:n_gen]

he, hs = collect(human_pkgs, all_diffs=False)
ge, gs = collect(gen_pkgs, all_diffs=True)
report(he, hs, "HUMAN")
report(ge, gs, f"GENERATED ({n_gen} pkgs)")
