# Adjacent-lane flow

The user's playthrough found long repetitions in one lane and requested
verification against the six existing Expert+ references, followed by bounded
countermeasures if supported. The request authorizes implementation in the
existing postprocessing architecture. The subsequently requested
Heeartbeeat Oveerheeat!!!! Standard Expert+ is the seventh reference.

## Evidence

Frozen measurements are in the sibling `devin-artifacts/lane-flow-20261117`
directory. Among consecutive same-hand singleton arrow groups at gaps above
20 ms and at most 400 ms, references repeat a lane on 22–33% of transitions;
the current six samples repeat it on 65–75%. Adjacent-lane movement is 32–56%
versus 15–22%. The 100–200 ms stratum confirms the discrepancy.
Heeartbeeat contains 1,933 notes at 280 BPM; its corresponding shares are
27.2% same-lane and 62.7% adjacent-lane, with a longest measured run of seven.

The raw model output repeats lanes on 22–32%. Replaying every current repair
exactly reproduces its frozen output: reach repair raises repetition to
40–56%, and later lane/phrase/local style repairs raise it further. Reach
repair currently processes positions before repairing invalid directions.
An isolated arrow-first experiment did not materially reduce repetition and
worsened it in five of six pre-pattern comparisons. Preserve the existing
repair order; the direction-first hypothesis was not supported.

The user's follow-up specifically identifies the last commit as a regression.
Frozen paired output confirms this in all six maps: Astronomia's repetition
rose 59.1% to 64.5%, Lost It's 61.7% to 65.7%. The latter's longest run rose
41 to 47 notes. Connected phrase edits must no longer create additional
same-lane triples, even if another style cost improves.

## Approach

Use three bounded changes within the existing forward policy:

1. Prevent connected phrase edits from increasing repeated-lane triples.
   Replace the generic all-to-outer-column proposal with a connected
   center-to-bottom proposal that preserves x and d.
2. Prefer fewer same-lane triples when otherwise-equivalent style candidates
   compete; preserve existing cut/placement priorities and hard guards.
3. As the last mutable pattern step, try adjacent-lane patterns on single-lane
   runs of at least four notes. Use observed reference cycles, not arbitrary
   random position changes.

Re-enabling the historical repeated-cell pass would primarily change height,
not lane. Forcing every consecutive note to alternate would remove legitimate
short repeats and accents. Merely adding a final random shift would continue
fighting earlier repairs.

## Contracts

- Feature gates remain those of `v2_pattern_flow`; search/legacy/off paths retain
  their previous sequence and defaults.
- Preserve the existing reach, collision and hard direction/overlap sequence.
- Repetition is a soft preference, never a new dropping condition.
- Connected phrase repair rejects candidates that increase local repeated-lane
  triples. Its previous mixed-column-to-outer-column collapse is no longer an
  acceptable repair when it creates such triples.
- Count a repetition cost for each consecutive singleton-arrow triple whose
  x values match and whose two same-hand gaps are in `(eps, .4]`.
  Count only triples intersecting the edited note IDs; skip ambiguous groups.
- Last-step lane overlays consider maximal same-lane runs with 100–400 ms
  gaps; skip dots, stacks, ambiguous groups and adjacent ambiguous context.
  Use non-overlapping chunks of four to eight notes, at most 10 candidates.
  Start with up to eight; shorten an irregular chunk longer than four to four,
  then consume it once. This avoids requiring a long uneven run to change as
  one unit.
- Templates require regular timing under the existing `_regular` predicate.
  Two x-only alternatives and two direction-preserving lane/height alternatives
  also serve irregular timing. On the inner home lane, move middle-row up-family
  arrows to row2 and middle/top down-family arrows to row0. Leave other heights
  and all arrows unchanged; existing guards decide whether this move is valid.
  No forced changes to shorter repeats or already-varied streams.
- Two additional bottom-row alternatives preserve each chunk's first and last
  notes and every cut direction, changing only interior x/y. This keeps the
  phrase entry and exit fixed while attempting a compact lateral movement.
  The lateral static bank retains rebound/rocker cycles; the separate existing
  wave pass remains unchanged.
- Overlays must strictly reduce local repetition cost and cannot increase any
  of the four existing style-cost components or raw upper-center downcuts.
  Keep all speed, forbidden-cut, collision, direction and entry/exit guards.
- Preserve IDs, times, hands, count, dots and non-geometry fields in style
  processing; preserve downstream retained-note immutability.
- Keep diagnostic samples bounded at three per family/reason.
- No model, dataset, audio decoder, hazard-policy or serialization changes.
- CPU only; no new dependency, commit, push or library installation in this task.

## Acceptance

Seven paired generations use the existing frozen audio, seed and NPS, with
identical pre-repair notes. Report lane/cell repetition, adjacent transitions,
four-note ABAB windows and long-run occupancy by hand and timing stratum.
Require improvements across the maps without new hard audit failures;
inspect retention changes and any regressions in prior style targets.
Check runtime and candidate limits, focused tests and one full CPU suite.
Automated results do not establish in-game comfort.
