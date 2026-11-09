# Forward, best-effort note postprocessing

Status: approved, including the clarifications below.

## User Clarifications

- Remaining reach/path findings are diagnostics, not reasons to stop.
- Loaded production settings determine which checks are active.
- Keep both the existing local upper-center visibility rule and the independent
  new center move; they have different origins.
- Complete all base-note movement before adding copies or chains. Do not repair
  expansion groups by moving or recutting their members.
- Below ten seconds is a useful postprocessing goal, not a strict requirement.
- A final change summary is useful but has no required report schema.

## Intent

Replace expensive production note-repair search with a fixed sequence of
small chronological passes. The user prefers useful approximations and quick
generation over exhaustive satisfaction of comfort rules. Add an independent
outward move to reduce notes blocking the middle of the view.

The prior November 8 optimization policy remains available for explicit
experiments; this proposal changes ordinary generation, not historical results.
Existing memoization and cheap-first validation changes in b65d2b4 stay intact.

## Approaches

1. Cap the current optimizer: smaller change, but retains reverse search,
   backtracking, joint option products, and interacting repair passes.
2. Replace production repair with ordered local rules: predictable work,
   straightforward diagnostics, but not globally optimal rule satisfaction.

The approved approach is 2. The user delegated the implementation approach.
A rewrite of timing, model inference, hazards, lighting, or export is unnecessary.

## Scope And Selection

- Add `v2_note_postprocessing_mode = "forward"` as the production default;
  `"search"` selects the existing behavior.
- Select forward adapters only for the ordinary default v2 note-postprocessor
  namespace when legacy cleanup is disabled, playability is enabled, and either
  the priority or reach feature is enabled.
- Explicit custom postprocessors, historical experiment wrappers, disabled
  playability, and legacy v1 retain their existing behavior.
- Keep the existing Emphasize Beats and Single Mode controls.
- Do not force existing flags off: priority enables the local priority
  preferences; reach enables local reach checks; existing late-turn settings
  retain their local meaning. Either flag can select forward adapters. If both
  are disabled, keep the existing non-priority adapter path.
- The selected adapter, not just the global mode setting, determines final
  validation policy. Custom historical postprocessors stay strict even when
  the global production default is forward.
- No architecture, model weights, training, supported-platform, timing-generation,
  package-format, or library-export changes.

## Ordered Production Steps

Each pass visits time-ordered notes once. Iterating notes and nearby notes is
necessary; repeated convergence sweeps, reverse repair, backtracking, beam search,
and Cartesian products of note assignments are not permitted.

1. Structural normalization: preserve stable IDs and valid elapsed timestamps,
   enforce Single Mode and structural duplicate handling, and retain the existing
   onset-snap and intro-floor behavior. Do not invoke the old clearance fixpoint.
2. Direction rules: apply fast same-hand flow, counter-direction, and diagonal
   preferences using the immediate previous same-hand swing and a small fixed
   list of direction alternatives. Dots remain dots. No phrase-level search.
3. Reach and speed rules: consider only the current note and fixed local
   alternatives. Try at most eight joint cell/direction proposals per note,
   preferring the existing assignment and small in-hand changes. Use the
   existing speed formula and cut convention; do not fit new scoring weights.
4. Local interference: use a temporal neighborhood index, same-cell occupancy,
   swept-cut cells, and exact cut-segment intersection where needed. Make one
   local correction attempt sequence, not a fixpoint. An optional edit must not
   introduce a new concrete cell/cut collision.
5. Repeated-position separation: retain the useful one-cell cut-direction move
   for exact consecutive same-hand position repeats, with local clearance
   checks rather than full-map metric rescoring.
6. Existing upper-center visibility: keep the 1.5-second, four-to-eight-onset
   exposure definition and calibrated severity cutoff. Inspect only the current
   endpoint group, without reopening earlier decisions. Then apply the
   independent center-view rule below.
7. Freeze retained anchors, run learned expansion, and inspect only mutable
   additions with the same bounded local checks. Reject unsafe new copies or
   chains instead of moving frozen anchors or rerunning global repair.
8. Run v1-style emphasis with unchanged gap/probability/geometry semantics, but
   use the indexed local guard in forward mode. Keep the strict emphasis guard
   for search mode. Then proceed to hazards, arcs, walls, lighting, and export.

All movement and recutting finishes before expansion. Reject a conflicting
learned expansion atomically using the existing rollback interface. Accepted
copies/chains are not subsequently moved or recut. Emphasis is checked
sequentially on insertion after learned expansion. No old post-expansion speed,
counter-sweep, reach, or clearance repair runs in forward mode.

The calibrated M2/global event-budget optimizer is not run in forward mode.
Retain the meaningful local visibility check and add the independent center rule;
report residual visibility and other event findings without claiming the old
3%/10% calibrated budgets were enforced. Search mode retains those budgets.

The implementation plan must map these steps onto the existing pipeline hooks.
There must be no hidden call back into `repair_priority`, its soft/visibility
searches, the old speed-cap convergence loop, or the clearance fixpoint in the
forward default path.

## Acceptance And Remaining Violations

- Never return invalid grid coordinates, invalid times, duplicate IDs, broken
  anchor relationships, malformed chains, or invalid package syntax.
- Preserve note IDs, onsets, hands, and count during optimization passes.
  Existing structural deduplication and intro-floor omissions remain distinct.
- Never delete retained notes solely to achieve an optimization score or speed
  target. A bounded rule may leave its incumbent unchanged.
- Comfort preferences, speed, reach, axis cadence, and travel/crossing heuristics
  are best effort in forward mode. If local attempts cannot resolve an existing
  violation, record it rather than exhausting a search budget or failing the song.
- Optional relocation and expansion must not introduce a concrete same-cell
  or cut-path collision. Skip a blocked optional edit.
- Final validation must be policy-aware: retain structural validation, but do not
  secretly reject a forward-mode map using the old strict search-policy audit.
  Search mode keeps its current hard audits and rollback behavior.
- Specifically, the flag-gated `forbidden_cut` raise and the priority
  preparation/crossing hard-audit raise in `v2_pipeline._validate_state` become
  reported findings for the selected forward adapter. The user explicitly
  approved this change to the old hard-ban contract.
- Forward diagnostics must not claim `priority_preserved=True` or zero violations
  without evidence. Report the selected policy, attempted/applied/skipped changes,
  unresolved findings, work counts, and per-step elapsed time.
- Keep existing general diagnostic envelopes where consumers need them; add a
  dedicated `forward_postprocessing` record. Mark strict priority enforcement
  as not run, not as a fabricated zero-violation success. Structural production
  validation and best-effort playability must be clearly distinguished.

This deliberately relaxes the old requirement that every priority hard finding
be solved before exporting. It does not permit invalid beatmap objects or
unchecked new optional collisions.

## Independent Center-View Rule

Columns are numbered 1 through 4 left to right. The left-hand 3x3 matrix uses
columns 1 through 3; the right-hand matrix uses columns 2 through 4.

- Eligible column: left hand column 2 (`x=1`), right hand column 3 (`x=2`).
- Middle-row weight 1.0; top-row weight 0.25; bottom-row weight 0.0.
- Use one RNG draw per eligible non-bottom-row note. Move when
  `draw < 0.8 * row_weight`: middle 80%, top 20%, bottom never.
- Move one column outward: left hand column 2 to 1, right hand column 3 to 4.
- Keep time, row, hand, arrow, ID, and metadata unchanged. Dots are eligible too.
- Keep the proposal inside that hand's matrix and reject it if the destination
  or the note's resulting cut path interferes with another nearby note.
- Do not condition eligibility on a preceding/following note, repeat position,
  flow score, travel score, or full-map metric. The only other-note inspection
  in this rule is the required local collision/cut-path guard.
- Run once before the retained snapshot and expansion, so copies/chains are not
  independently pulled apart. A blocked move is skipped, not rerouted or recut.

The approved column interpretation corrects the apparent "second/third row"
typo in the user's final sentence.

## Work Bounds And Modularity

- Put small, named passes and a reusable local index in a dedicated v2 module;
  keep orchestration in the existing pipeline rather than a second generator.
- At most eight proposals per mutable note per relevant pass. No joint
  multi-note assignment search. Use chronological local windows and reusable
  cached geometry rather than full-stream rescoring for each proposal.
- Preserve fixed earlier decisions; later passes may not reopen whole phrases.
- Reuse the existing scalar speed calculation and geometry helpers. The lead
  owns any new rule constants or acceptance formulas in the implementation plan.
- No new dependency or framework.

## Verification

- Deterministic tests for exact center eligibility, hand mirroring, row weights,
  RNG boundaries and consumption, dots, metadata preservation, and blocked moves.
- Test same-cell conflict, swept-cut conflict, and crossing segments without a
  shared integer cell. Preserve in-hand bounds and frozen expansion anchors.
- Test bounded work on dense/adversarial streams; unresolved comfort violations
  must become diagnostics rather than a budget-exhaustion exception.
- Test the default pipeline cannot enter any of the superseded expensive
  repair/fixpoint functions, while explicit search mode remains available.
- Preserve old search tests rather than rewriting their expectations to match
  the approximation. Run focused CPU tests, then one final full CPU suite.
- Replay the frozen 1,335-note input already captured for this task. Record
  runtime and rule outcomes, not a claim of identical output to exhaustive search.
  Aim for less than ten seconds of forward postprocessing on the available CPU
  (versus 63.5 seconds for optimized search); use measured evidence rather than
  a flaky wall-time unit-test assertion.
- Run one disposable real-song CPU generation and validate v2/v3 packages.
  Do not export to Beat Saber or start a corpus comparison.

## Approval

The user approved the staged best-effort default, diagnostic-only remaining
reach/path findings, and the center rule's column/probability interpretation.
