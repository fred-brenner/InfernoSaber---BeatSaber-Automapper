# Flow signature and specific style rules — design

## Context

Three commits (`c13df79`, `4d1ceab`, `2b524f2`) improved generated Expert+
flow toward seven human references, but each round fixed its own metric
while regressing others, and the remaining center-visibility problem is the
largest measured gap. This round replaces per-metric whack-a-mole with a
versioned flow signature plus a small set of *specific* rules, all derived
from the reference corpus (`devin-artifacts/flow-signature-20261118/
reference-metrics.json`).

## Derived reference metrics (15,578 notes, 7 maps)

Specific rules must hit near-0% or near-100% on references:

| Rule candidate | References | Generated (lanes-final) | Verdict |
|---|---|---|---|
| Inner-middle (x∈{1,2}, y=1) with next same-hand onset ≤ .4s | 57 (0.37%; KB 44, others ≤6) | 54–165 per map | **ban in streams** |
| Inner-middle with no fast follower (terminal) | 24 | — | exempt |
| Inner-top (x∈{1,2}, y=2) fast follower | 1,519 (9.7%; up-family dominant) | 99–191 per map | **allowed** — signature fan/up pattern |
| Inner-top horizontal (y=2, d∈{2,3}) fast follower | 18 (0.12%) | — | ban in streams |
| Top-row horizontal anywhere (y=2, d∈{2,3}) fast follower | 35 (0.22%) | 12–51 per map | ban in streams |
| Middle-row horizontal fast | 106 (0.68%) | 19–53 per map | comparable — keep existing soft cost |
| Bottom-row down fast | 3,734 (24%) | — | allowed — never penalize |

Ruling on the user's center definition: "middle and top row, lanes 2–3" is
only half supported. Inner-top cells are a core reference pattern
(Arche 126 fast inner-top notes, 125 up-family; Goodbye 418 all up).
Banning them would move output away from the references, so the center
penalty covers the middle row only; the top row gets the specific
horizontal-cut ban instead. The exemption is exactly the user's wording: a
note is exempt when no fast-paced same-hand note follows (next onset > .4s).

## Speed-cap calibration

Flat cap `max_speed·factor·0.63 ≈ 15.5` vs reference same-hand transition
speeds (calc_note_speed nominal model): p50 26–74 in the <.2s bands, p90
≈ 51–74; nearly all fast-band reference transitions exceed the cap, while
>.6s p90 ≈ 8–16 sits below it. The cap therefore blocks exactly the fast,
human-style moves the repairs try to add, while staying loose in slow
sections. Replace the flat bound with a per-gap-band multiplier, calibrated
to ~p90 of each reference band and rounded conservatively:

- gap < .1s → ×4.0 (≈62, ref p90 61–75)
- gap < .2s → ×3.5 (≈54, ref p90 40–52)
- gap ≤ .4s → ×1.8 (≈28, ref p90 18–26)
- gap ≤ .6s → ×1.0 (≈15.5)
- gap > .6s → ×0.7 (≈10.8)

Applied only inside `_speed_ok` as `max(cap(note)·factor(dt), incumbent)`;
the per-note base cap and ramp are unchanged. This widens fast-section
repairs to human range without touching slow-section safety.

## Signature (improvement 1)

`map_creation/v2_flow_signature.py` computes, per map: same-hand
consecutive-singleton transition histograms keyed by gap band — lane type
(same/adjacent/wide), Δy, cut-direction step distance, outer-pair flag —
plus same-x run-length distribution (capped at 8), 12-cell occupancy per
hand, ABAB-4 window rate, and the specific-rule counters above.
`signature_distance` compares against a pooled reference signature
committed as `map_creation/reference_signature.json` (constants derived
from the seven references); distance = weighted total-variation per
distribution, documented in the module. It is the reporting/gate metric,
not a per-candidate hot-path cost.

## Unified objective (improvement 4)

Extend `style_cost` to a 5-component tuple `(center, horizontal, boundary,
upper, top_horizontal)` shared by every pass (wave, phrase, local, lateral,
reach tie-breaks). New components:

- `center`: x∈{1,2} and y=1 and next same-hand gap ≤ .4 → 12. (Drops the
  aligned-stack/terminal-accent machinery for this component; the
  following-gap exemption covers the observed references cases.)
- `top_horizontal`: y=2 and d∈{2,3} and next same-hand gap ≤ .4 → 6.

`_check_chunk` returns `(*agg5, repetition, displacement, deviation)`;
callers that indexed key[4]/key[:4] shift accordingly.

## Bounded second sweep (improvement 5)

`repair_lanes` runs a second sweep after the first, with windows offset by
two groups. Same acceptance rules (strict repetition reduction, no style
component increase, all hard guards), at most two sweeps total, at most 10
trials per window per sweep. No convergence loop.

## Upstream and additions (improvements 6, 7)

- `_reach_pass`: among positions whose reach cost is within tolerance of
  the best, prefer one that does not create a same-x triple touching the
  note and does not place x∈{1,2},y=1 with a fast follower.
- Expansion, note emphasis, and gap fills: added notes are rejected when
  they would create a touched same-x triple or an inner-middle-fast note.
  Shared helper `flow.accepts_added(context, note)`; counts reported.

## Opt-in gate (improvement 8)

A `comparison`-marked pytest replays `repair_notes` on two committed frozen
pre-repair fixtures (`tests/fixtures/flow_signature_*.json`, ~600 lines
each) and asserts the specific-rule counters and signature distance do not
regress vs committed goldens. Excluded from the default suite via the
existing `-m 'not comparison'` selection.

## Out of scope

- No model, decoding, or sampler changes.
- No inner-top cell bans (contradicted by references).
- No changes to `v2_pattern_flow=False` or legacy/search paths.
- No unbounded loops; all budgets stay ≤12 trials and ≤2 sweeps.
