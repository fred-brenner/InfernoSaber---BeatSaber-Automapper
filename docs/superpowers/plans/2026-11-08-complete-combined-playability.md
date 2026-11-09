# Complete combined playability implementation plan

> **For agentic workers:** Use test-driven development, parallel independent implementation tasks and a final independent review. The user authorized implementation, CPU generation validation and a new complete playtest pack on 2026-11-08.

**Goal:** Deliver one complete experimental candidate addressing M1–M5, calibrated reach/placement and pattern preservation, then validate fresh generations before publishing a blind playtest pack.

**Architecture:** Retain the seven-stage CPU pipeline and app1 timing/survivor rules. Use original-first bounded reverse hard repair and a shared local acceptance policy for optional geometry, direction and reach proposals. Frozen human calibration governs soft tails; runtime never reads the human corpus. Keep existing experiment APIs and disabled behavior.

**Tech Stack:** Python, NumPy, existing TensorFlow CPU inference and v2/v3 package tooling.

**Spec:** User's approved complete-scope table in this chat, original `2026-11-06-playability-metrics-design.md`, and the binding decisions in `2026-11-08-combined-reach-playtest.md`.

## Global constraints

- Zero far-opposite inward cuts, including diagonal variants. No new dots or note deletion as repair shortcuts.
- Preserve note identity/time/hand during postprocessing; model-supported hand recovery originates at position decoding, records alternatives without changing model history, and commits locally after primary selection while retaining onset counts/doubles.
- Human trigger budgets are 3% of eligible events for specific metrics and up to 10% for coarse proxies, not target percentages of top-row notes. Keep allowed variation.
- Preserve original early diagonal cap and survivor selection. Allow later legal bridges without treating a human upper percentile as a target.
- Check incoming/outgoing speed/flow, simultaneous occupancy, both-hand clearance, virtual dots, retained anchors and final expanded notes.
- Preserve pre-existing changes, CRLF files, model/data assets and historical reports. Work in this explicitly authorized development checkout. No commits, release, push, GPU or game-library export.
- Perform automated verification during development. Produce one complete new player pack only after end-to-end CPU validation.

## Review focus

- A later optional pass must not undo a comfortable phrase or an earlier movement/visibility gain.
- Actual 135-degree arrow turns must remain visible as 45-degree cut-axis changes.
- Soft placement limits must allow ordinary human top-row/outer-lane use and handle sparse strata honestly.
- Reverse traversal must handle simultaneous groups, dot descendants and bounded backtracking, with explicit rollback on infeasibility.
- Decoder hand recovery must preserve doubles, timing, history consistency and balanced supported assignments.

## Tasks

### 1. Calibrated placement exposure

Files: new `map_creation/v2_playability_placement.py` and frozen reference JSON, calibration/evaluation support, focused tests.
Interface: `placement_costs(notes, nps=None)` returns per-component excess and eligible event diagnostics; `local_placement_cost(notes, nps)` returns a comparable finite scalar.

- [x] Test ordinary allowed top/outer positions, repeated exposure tails, mirrored hands, dots/stacks and NPS strata.
- [x] Calibrate phrase-level row/lane exposure on all curated human packages, recording every skip and the <=10% coarse budget.
- [x] Implement portable deterministic costs; expose them to the shared repair policy and scorecards.

### 2. M5 complete direction/recovery behavior

Files: `v2_playability_directions.py`, `v2_pipeline.py`, experiment policy wiring and focused tests.
Interface: direction repair accepts the shared guard via optional keyword `guard=None`; hand recovery remains at decoder time.

- [x] Pin short axis churn, long runs, 45/90 ending angles and incoming/next-change timing fixtures.
- [x] Preserve normal-range dwell; repair both short detours and overlong runs only with feasible supported alternatives.
- [x] Resolve model-supported fast-reversal/recovery in the complete policy without changing onset counts or doubles; record any irreducible excess.
- [x] Verify local speed/flow, reach and geometric acceptance on every direction proposal.

### 3. Shared minimal repair and integration

Files: new `map_creation/v2_playability_policy.py`, reach/geometry/postprocessing/pipeline hooks and focused tests.
Interface: `PatternGuard(notes, nps=None)`, `allows(changed_ids)`, `commit(changed_ids)` operate on a shared chronological note stream, comparing optional candidates to the current accepted snapshot. Hard repairs bypass soft tradeoff vetoes but retain all physical legality checks and explicit diagnostics.

- [x] Write failing tests for unchanged comfortable notes, local moderate movement/churn regressions, cross-pass undo, reverse dead-end rollback and frozen expansions.
- [x] Implement original-first reverse hard search with bounded joint repair/backtracking.
- [x] Make all optional passes use shared acceptance and calibrated placement; require actual target improvement and retain original on ties.
- [x] Restore original early diagonal cap/survivor behavior and preserve the disabled path.
- [x] Final full-stream physical/ban checks and truthful residual diagnostics.

### 4. Complete validation and player pack

Files: external timestamped results, production-equivalent runner/audit scripts, documentation and project guidance.

- [x] Replay all five frozen preferred charts and complaint windows; compare all metrics, per-hand dwell, exposure, moderate movement, edit footprint and residuals.
- [x] Run focused tests, complete CPU non-comparison suite and independent review; fix material findings before completion.
- [x] Generate fresh paired complete candidates with original recording/settings/seed and game-library export disabled. Audit actual decoded packages and anchors, not just receipts.
- [x] Build one blind extractable pack after generation validation, with private key, blank feedback sheet, hashes and passage guide. Include comfortable controls and report remaining limitations.
- [x] Update docs/AGENTS and completion ledger; final answer links the new pack and reports actual verification.

## Implementation decisions and validation status

The mandatory hard repair has its own physical audit and explicit rollback; optional passes share committed snapshots. M3 uses a strict 3% human segment cutoff rather than map quartiles as an event threshold. Placement and joint M5 duration/count/incoming-gap tails are coarse <=10% proxies. Bounded recovery is active in the combined policy at decoder time; residual fast-reversal excess remains explicitly measured.

Initial validation passed 237 focused tests and 1,613 CPU non-comparison tests. Five saved-chart replay audits passed. A final-review M5 late-tail budget issue is being fixed before the final source freeze, rerun and fresh generations; preliminary evidence remains superseded, not final acceptance.

Validation rejected the first fresh candidate despite physical correctness: immediate decoder hand swaps changed autoregressive history and substantially worsened a comfortable passage. The complete policy now preserves the original model/primary stream, defers only supported local hand choices, and guards full-map M3 and full per-hand M5 context. Superseded run artifacts remain archived; new inference and package acceptance must use the next source freeze.

The complete M1 implementation includes a human-calibrated consecutive-horizontal event tail and bounded joint direction/cell bridges. The exact R7 regression improves 12 to 5 consecutive reversals and 27.5 to 3.0 repeated-load excess without new reach, placement, coordination or dwell tails. Partial feasible relief is retained when the old all-vertical witness would require uniform rows and violate other constraints. Adjacent supported hand pairs provide a real interior recovery path while preserving per-hand cut order. The final source freeze includes these components; all older validation runs remain historical.

## Completed validation and delivery

CPU final suite: **1,649 passed, 10 deselected**, 187.05 seconds. Integrated focused suite: 273 passed. Independent final delta review: 141 passed, no outstanding major findings. Five exact preferred-chart replays and eight fresh CPU generations passed native package/endpoint/anchor checks. All four fresh candidate primary streams match their app1 controls exactly, retain final counts (1,386/1,596/1,088/708), and export zero banned cuts and zero clearance violations. One pre-existing opening-ramp exception remains unchanged; it is explicit in receipts.

The actual comfortable 74–84-second control is unchanged. Dense commits one supported joint hand pair (two notes), fast reversals 1,240 to 1,238; structural M5 excess remains. Fresh Horizon's original-app1 99.5–109.5-second phrase improves M1 .312 to .283 and repeated-run peak 15 to 9, with residual stress. The exact earlier preferred Horizon version still scores better on some measures; the report keeps both comparisons and does not claim uniform improvement. Observed candidate CPU time is 18.7–63.3 seconds (1.80–3.38 times control on this shared runtime).

Delivery contains five blind A/B pairs, full songs, ten map folders, passage guide and blank feedback. Aggregate ZIP SHA-256 `f5ceecec5e33b7f8b525ebc6201f5b815ed0740bbea3e392bec680fce034f913`. All map members remain byte-identical to audited packages; archive CRC and unchanged 118-file source freeze were verified after adding the guide/feedback. No game-library export, release or commits. App1 defaults remain unchanged pending player acceptance.

[Playtest guide](../../../../Data/training/evaluate_beats/results/complete-combined-playability-20261108/playtest-final/PLAYTEST.md) · [Engineering results](../../../../Data/training/evaluate_beats/results/complete-combined-playability-20261108/results.md)
