# Reference phrase flow

The user requested another implementation and comparison round using Standard
Expert+ from Night Raid with a Dragon, Goodbye, Lost It, KillerBeast and
Astronomia. Arche Expert+ / new gud remains a regression reference.

## Evidence and intent

Frozen references, six fresh CPU baselines and first-rejection traces are in
the sibling `devin-artifacts/flow-round2-20261117` directory. Baselines use
requested NPS 8 and seed 20261117; reference maps have different densities.
Compare each map separately and stratify by same-hand timing.

Four new references have zero center-middle notes in continuing single-arrow
sequences with both adjacent same-hand gaps in the inclusive 100–200 ms band.
KillerBeast has five. Horizontal cuts are uncommon but present. Common motifs
include bottom rebounds, outer-column vertical bounces, diagonal rockers and
fan-like changes between outer-middle and upper-inner upward cuts.

Existing whole-run translations most frequently fail their speed guard.
Existing wave trials commonly fail entry/exit direction checks. Keep these
guards; improve the available bounded candidates. Center repair must not create
additional upper-center downward-family cuts as a side effect.

Most reference obstacles occupy outer columns, including upper-edge slabs.
Some are much shorter than the generator's minimum wall duration. Retain the
configured minimum and clearance; use regular note onsets for modest edge
accents during active passages. Existing calm-gap decoration remains separate.

Legacy v2 editor BPM markers in three references are metadata; these maps
encode elapsed time with their Info.dat base BPM and contain no runtime BPM
events. This analysis handles them explicitly without changing the repository's
corpus decoder or preprocessing policy.

## Selected approach

Extend the current final mutable pattern pass, after hard repair and before the
retained snapshot. Keep whole-run translation, add bounded shorter wave trials,
then repair non-overlapping groups of two to four notes using two positional
transforms and a small static motif bank. Finish with the existing local pass.
Use the same physical, direction, overlap, speed and downstream addition guards.

A stronger scalar penalty alone cannot overcome blocked transitions. An
unbounded optimizer or decoder retraining is unnecessary for this iteration.

## Contracts

- Preserve note IDs, times, hands, count, dots and all fields except x/y/d.
- Do not relax speed, physical conflict, forbidden-cut or overlap checks.
- Keep candidate rollback exception-safe and use live index coordinates.
- At most 12 wave trials per existing window: its full length (at most eight),
  two fewer notes (at least four), and four notes, each with four phases.
  Equal candidate costs prefer longer prefixes, then the lower phase index.
- At most 12 phrase trials per non-overlapping window, at most four notes.
- Keep the existing 12 costly singleton trials and one chronological pass.
- Preserve comfortable patterns with zero style cost; no forced variety.
- Skip dots, stacks and ambiguous staggered groups for phrase overlays.
- Do not change same-onset semantics in unrelated detectors in this iteration.
- Store bounded first-rejection counters and at most three examples per reason.
- Never increase raw upper-center downward-family counts within a style edit.
- `v2_pattern_flow=False`, search and legacy retain their previous behavior.
- New `v2_pattern_obstacles=True` applies only with the existing flow policy
  enabled, heuristic walls enabled and the user's obstacle toggle enabled.
- Obstacle accents use x=0 or 3, y=2, width=height=1. No forced lean/crouch.
- At most two accents per eligible eight-onset regular window, windows at least
  four seconds apart; respect configured wall minimum and all final validators.
- No new dependencies, model changes, training or installation in Beat Saber.

## Acceptance

Run focused note-flow, expansion/emphasis, hazard and export coverage, followed
by one full CPU suite excluding opt-in comparisons. Re-generate the six frozen
audio inputs with identical settings and verify identical pre-repair notes.
Independently audit serialized direction, fast-counter and same-cell overlap,
verify retained snapshot equality and new-wall clearance, and report per-map
note retention, placement, motif counts and elapsed repair time.

Changes in reference-style statistics are evidence, not player approval.
