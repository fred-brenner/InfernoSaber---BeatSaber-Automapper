# Priority Playability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax. The user authorized execution and requested GPT-6.1 Terra High workers.

**Goal:** Implement every decision in the player priority review, validate the final serialized maps from fresh CPU generations, and produce a new blind playtest pack focused on difficult passages.

**Architecture:** Introduce a versioned `priority` experimental policy instead of changing historical arm definitions. Share pure event measurements between its bounded joint cell/arrow search, calibration and independent package reports. Run repairs before retained-note freeze, enforce the same hard conditions on expansions, and audit exported notes independently.

**Tech Stack:** Existing CPU Python/NumPy runtime and pytest; existing seven-stage inference, package exporter and experiment runner. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-08-playability-priority-review.md`.

## Global Constraints

- Highest priority: M4; forbid every detected crossing at <=250 ms, penalize slower pairs through 1.5 s, use 1.5 s immediate same-hand predecessor lookback, no mirror exemption or extra pattern lookahead.
- Very high: preparation in each hand's 3x3; max per-edge excursion of two cells is forbidden; a diagonal one cell outside two edges is one strong soft excursion. Preserve all explicit edge/row examples and compound locally with speed/repetition.
- Very high: M5 90-degree base cost is twice 45; first isolated change small; frequent changes strong, frequent irregular changes strongest. Opposite arrows share an axis. No forced variety or long-run/reversal optimization.
- Placement stays active medium; retain helpful M2 behavior. M3/M1 and lowest-priority counters are monitored, and cannot veto primary repairs. Follow-through outside alone has no cost. Retain existing travel semantics; 0.18 s remains optional.
- Calibrate human flags with eligible-event denominators, <=3% specific and up to 10% coarse; use only curated humans and record individual skips. Hard bans are independent of calibration.
- Preserve app1 disabled behavior and historical arms, note IDs/onsets/hands/counts during repair, physical flow/speed/clearance, retained snapshots, supported package formats, Windows compatibility and CPU-only execution.
- Preserve dirty checkout and user assets. No release, push, commits, training or library export. Keep historical evidence untouched and private blind assignments outside public packs.
- Use GPT-6.1 Terra High subagents; parallel exploration is read-only, implementation tasks sequential with independent review. Use before-snapshots for review because HEAD omits existing experimental files.

## Review Focus

1. Tiny overlap, mirror pairs, exact 250 ms and 1.5 s boundaries, and a predecessor outside the old 400 ms window must be handled consistently in runtime and exported-map audits (Task 1).
2. Dense awkward local passages must not be diluted by easy context; edge preparation is distinct from permitted follow-through (Tasks 1–2).
3. Hard repair or medium placement must not reintroduce churn; long steady patterns must remain comfortable without forced variety (Task 2).
4. Copies, chains, dots, same-time stacks, start ramps and incoming/outgoing changed edges must survive final audits or fail explicitly (Tasks 2–3).
5. A public blind pack must contain actual completed fresh generations with no private key, stale source receipts or misleading replay claims (Task 4).

## Task 1: Shared versioned event definitions

**Files:** Create `map_creation/v2_playability_priority_metrics.py` and `tests/test_v2_playability_priority_metrics.py`.

**Interfaces:** Raw note dictionaries use `id/time/x/y/hand/d`; all time values are elapsed seconds. Produce pure `preparation_events(notes)`, `crossing_events(notes)`, `axis_change_events(notes)`, and `priority_metrics(notes)` APIs. Return JSON-compatible events with note identities/times, eligibility, hard findings, separate raw/component costs and totals. Historical helpers stay unchanged.

- [x] Write failing tests for all explicit reach examples/mirrors, two-edge diagonal counting, no standalone follow-through, dots/stacks, local speed/repetition, M4 boundaries/lookback/no exemption, axis magnitude/cadence/irregularity and unchanged long tails.
- [x] Implement shared deterministic measurements using bounded sorted neighborhoods; choose transparent provisional soft constants and record them, separating human-calibrated flags from raw active penalties.
- [x] Run focused tests and record red/green evidence. Independent reviewer verifies spec and quality against the scoped snapshot diff.

## Task 2: Joint priority repair

**Files:** Create `map_creation/v2_playability_priority.py` and `tests/test_v2_playability_priority.py`.

**Interfaces:** Consume Task 1 events. Produce `repair_priority(notes, cap, *, protect_retained=False)` returning JSON-compatible before/after audit and search statistics; mutate only x/y/d. Produce `audit_priority(notes)` with explicit hard findings. Existing production speed/flow/clearance functions remain the physical authority.

- [x] Test a dense crossing/awkward/churn passage, legal comfortable stable patterns, primary-versus-monitor tradeoffs, incoming/outgoing speed/flow, immutable retained notes, stacks/dots, bounded deterministic work and explicit infeasibility without silent invalid success.
- [x] Implement bounded reverse joint cell/arrow choices with necessary local backtracking or sequence alternatives. Prioritize hard crossings/reach; then strong reach/axis and slow M4; then existing placement/M2 and retained travel with minimal edits. Keep a legal incumbent, inspect affected dependencies, and never let a soft pass recreate hard faults. Avoid a stack of competing historical repairs.
- [x] Run focused tests and realistic saved passages for search feasibility/runtime. Resolve failures before independent review. If the existing crossing proxy proves incompatible with valid generation, present concrete evidence and wait for the user's decision instead of weakening it silently.

## Task 3: Runtime integration, calibration and reports

**Files:** Modify `tools/config/config.py`, `evaluation/playability_experiments.py`, `map_creation/v2_postprocessing.py`, `map_creation/v2_pipeline.py`; add focused integration tests and `evaluation/playability_priority.py`, a versioned human reference JSON if needed. Update relevant generation/development docs and AGENTS pointer.

**Interfaces:** `v2_playability_priority=False` selects historical behavior; experiment arm `priority` selects the new policy and bypasses old active M1/M3/dwell/recovery passes. Run primary repair before retained freeze and final hard validation after mutable expansion, keeping chain anchors synchronized. Offline diagnostics consume the same event definitions and expose event denominators/flags plus monitoring metrics.

- [x] Add failing integration and final-serialization rejection tests, disabled equivalence, no forced-variety/M3 repairs and scoped-settings restoration.
- [x] Wire the policy and final audits; calibrate only against curated humans before candidate-result inspection, recording individual skips and limitations. Verify all agreed requirements have runtime or monitoring coverage.
- [x] Run focused checks and full CPU `python -s -m pytest -q -m 'not comparison'`; review combined changes independently. Resolve important findings before source freeze.

## Task 4: Fresh generations and playtest deliverable

**Files:** New run artifacts under `../Data/training/evaluate_beats/results/priority-playability-20261108`; harness and ledger under `.superpowers/sdd/2026-11-08-priority-playability/`.

- [x] Freeze final source/reference/harness/model/settings hashes. Adapt previous tooling into the new workspace with explicit new output roots; never execute historical output-writing scripts directly.
- [x] Run saved-chart reproductions separately from fresh CPU paired generations, emphasizing Break the Horizon and Night raid complaint windows and additional dense coverage. Preserve comfortable windows as regression controls, rather than playtest filler.
- [x] Independently decode and audit actual serialized outputs for hard rules, physical checks, preserved density/onsets/hands, local preparation/cadence/placement and monitoring metrics; report actual timing, skips, limitations and unresolved findings. Reject invalid candidate packages and revise/revalidate before packaging.
- [x] Build a small blind A/B pack with passage guide and blank feedback; private key outside all public files. Verify CRC, extractable map folders, completion receipts, source freeze and leakage. Record final ZIP hash and link it with concise validation evidence; player quality remains unverified.
