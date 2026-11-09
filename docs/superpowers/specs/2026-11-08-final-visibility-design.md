# Final upper-center visibility design — 2026-11-08

User request: fix local upper-center visibility and M2, keep the reach/crossing
improvements, validate fresh generations, then finish A/B testing and deliver
five ordinary song packages at difficulty settings 5 through 8.

## Accepted scope and engineering choices

The current Night raid candidate clusters seven same-hand top notes in
76.045–77.218 s; central top notes rise5→10/38 in74–78s even though legacy M2
improves. Existing placement tails detect some excess, but primary repairs may
trade it away. This task fixes measurement and acceptance together.

Use the existing CPU v2 framework. Preserve the explicit ≤250ms intersecting
path ban,1.5s path context, opposite-edge preparation ban, all existing specific
reach/axis tail guards, physical flow/caps/clearance and note identity/onsets/
hands/counts/dots. Follow-through and monitoring priorities remain unchanged.
Historical scorer/reference versions and R4 source proofs/packages remain intact.

The chosen new visibility component is central-top occupancy(y2,x1or2) averaged
over the last8 same-hand20ms first-note-anchored onset groups in1.5s, with at
least4groups. Stacks contribute fractional occupancy, zero endpoints remain
eligible, and other-hand interleaving does not reset context. This is a proxy for
sustained local exposure, not a headset trajectory model or a top-row ban.
The1.5s limit avoids conflating slow isolated upper notes with continuous exposure.

Alternatives considered: a global top-row quota misses the observed local
concentration; increasing old M2 weights cannot detect an equal-height stream
that currently scores zero; a strict consecutive-run cap is evaded by one outer
note and prescribes patterns. Shared bounded rolling exposure directly catches
the reported case and permits ordinary human-range placement.

Calibrate the new component independently against the same curated human set,
strict3% pooled AND equal-map event tails, support>=5maps perband/globalfallback.
Record individual9skips. Exploration found51accepted with band cutoffs
.4/.5/(3/7)/.375; final tests/fit must reproduce those without tuning to generated
results. No old calibration or existing corpus preparation changes.

## Runtime and reporting

New pure module provides visibility_events(notes) and visibility_costs(notes,
nps=None,reference=None). Eligible event IDs/group memberships/weights depend
only on identity/time/hand, so cell/arrow edits change severity without changing
denominators. Each event exposes endpoint group_ids and context note_ids/weights.
A separate versioned reference and offline calibrator preserve historical M2.

Current priority reports expose the new M2 component, per-event flags and local
maxima; retain legacy rise-only M2 explicitly. A current scalar is mean calibrated
upper-center excess with its definition/version, not a silent sum of unlike
scores. Pairwise M2 remains an existing subordinate repair cost.

Cache note-to-visibility-event dependencies. All soft proposals must avoid new
or worse individual visibility tails. Dedicated bounded joint cell/arrow repair
after hard repair reduces existing exposure without dropping/moving onsets or
relaxing reach/path/physical constraints. Freeze final no-regression limits from
the original pre-hard stream: max(original event severity,human cutoff). For
retained-protected expansions, derive original visibility from retained-only
notes; new endpoints get human cutoff. Match endpoint groups via retained member
IDs to handle near20ms regrouping. Audit final serialized notes against these
limits. New hard crossing/reach rules remain absolute, and native pre-existing
visibility tails are soft exceptions, not a success claim of zero exposure.

## Bounded residual visibility exceptions

The full high-density Horizon case exposed two mild new visibility tails after
bounded repair: .5 at4onsets and .4 at5onsets, original .25/.2 and human cutoff
.375. Pair/triple/four-frontier probes found no guarded repair; a near repair
creates higher-priority near-fast crossing tails. M2 was specified as a soft
measure, so an absolute final ban on every new human-tail flag was too strict.

Keep the metric/reference/cutoffs and candidate visibility guard unchanged. The
repair still attempts to remove all excess. At final acceptance only, a residual
new tail may be explicit if original severity<=human cutoff, final severity is
within cutoff+1/onsets(one onset group's occupancy), and accepted residuals are
at most floor(3% of eligible events) per hand. Any hand over the rate budget
fails entirely; worsening an already above-cutoff original tail remains denied.
Known seven-of-eight exposure .875 still fails the .375+.125=.5 ceiling.

Original visibility_limits remain max(original,human cutoff). Store residuals
separately with policy version, baseline/base limit/severity/onsets/ceiling and
per-hand counts. Recompute the classification from final live events, and audit
it independently from serialized maps. No hard reach/crossing, speed, clearance,
primary-tail or measurement-threshold relaxation follows from this exception.

## Final behavior and delivery

Enable corrected priority policy as v2 default, including existing late-phase
slow-turn settings(.5s90/.65s45). Historical arms explicitly reset switches;
v1 and public CLI/package formats remain unchanged. No release/tag/push/training
or library export is authorized by this local task.

Five selected local songs: Night raid, Break the Horizon, No Sleep, Angry Too,
Bunker Machen Laut. Each gets app settings5/6/7/8(internal20/24/28/32), existing
ascending Easy/Normal/Hard/Expert labels. The20fresh charts serve final validation
and delivery; difficult8settings run first as pilots with exact completed-proof
reuse. Output is five multi-difficulty packages plus a convenient archive, with
no A/B labels, randomized key, control generations or feedback questionnaire.
Freeze source/model/reference/settings/harness and retain per-chart completion
and serialized audits. CPU only; use bounded independent worker processes.
