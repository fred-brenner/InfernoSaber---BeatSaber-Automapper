# Playability priorities after player review — 2026-11-08

These are the user's decisions from the metric-by-metric review following the
rejected combined playtest. They govern the next generator revision and
supersede conflicting optimization goals in earlier experimental plans.
This is a requirements record, not a claim that these changes are implemented.

## Priority and treatment

| Priority | Measure | Treatment |
| --- | --- | --- |
| Highest | M4 cross-hand path interference | Active: forbid fast crossings; penalize slower crossings. |
| Very high | Cut preparation and reach | Active: hard limits plus strong local soft penalties. |
| Very high | M5 axis-change frequency and regularity | Active: discourage rapid, irregular changes. |
| Medium | Placement frequency | Keep active; defer refinement until the primary problems improve. |
| Medium | M3 bimanual coordination | Monitor for now; preserve independent and asymmetric patterns. |
| Low | Same-hand travel and recovery | Retain working behavior; limited refinement if needed. |
| Low | M1 lateral load | Monitor as an outcome of path separation and axis stability. |
| Low | M2 visibility and occlusion | Monitor; retain the improvements that helped in the last round. |
| Lowest | M5 fast same-hand reversal count | Monitor only. |
| Lowest | M5 long unchanged-axis runs / lack of variety | Monitor only; no active variety penalty. |
| No standalone penalty | Follow-through outside the grid | Allowed by itself. |

## Cut preparation and reach

Columns below are numbered **1–4 from left to right**. Each hand has a normal
3×3 area: columns **1–3 for the left hand**, **2–4 for the right hand**, and all
three rows. Evaluate where the hand must start the cut, not just the note cell.

- A preparation position one cell outside the hand's area receives a strong
  soft penalty. Two cells beyond any one edge are forbidden. Far-opposite
  inward cuts are explicitly forbidden.
- A diagonal preparation one cell outside each of two edges counts as **one**
  excursion and receives the strong soft penalty. It does not count as two
  cells beyond an edge by adding the two offsets.
- A left-hand note in column 1 already receives a placement penalty and should
  be rare; mirror this for right-hand notes in column 4.
- A left-hand note in column 2 cutting inward to the right already receives a
  penalty; mirror this for right-hand notes in column 3 cutting left.
- Top-row notes cutting down and bottom-row notes cutting up receive soft
  preparation penalties.
- Penalties compound strongly with local speed and repeated exposure, making
  repeated awkward preparation practically unavailable in fast passages.
  A good map-wide average cannot excuse a bad local passage.

Follow-through outside the grid is acceptable by itself. Its endpoint can
still affect the next movement and interference with the other hand.

## M4 cross-hand path interference

- **Detector refinement approved after implementation exploration:** use actual
  intersecting straight cut/travel segments for crossings, and penalize nearby
  non-intersecting paths softly. Coarse shared grid-cell regions alone are not
  a hard violation. The previous proxy flagged 20,170 of 47,647 fast human
  pairs, including a verified non-intersecting example; the user explicitly
  chose this refinement before generator integration.
- Forbid crossings at near times / high frequency, with **250 ms** as the
  requested fast boundary. A severity threshold must not excuse a detected
  fast crossing merely because its overlap is small.
- At slower speeds, use a soft penalty that decreases with available recovery
  time, extending the scoring window through **1.5 seconds**.
- Increase the previous same-hand note lookback used to construct the travel
  path from **400 ms to 1.5 seconds**.
- Remove the blanket mirrored-double exemption. Symmetric arrows do not
  exempt a pair from the existing crossing check.
- Continue the sequential note/pair checks. The user specifically declined
  additional previous/following-pattern lookahead; the immediate previous
  same-hand note remains necessary for constructing its travel path.

## M5 axis-change frequency and regularity

Evaluate each hand's **undirected cut axis**. Left/right cuts share an axis,
as do up/down cuts; an ordinary 180° reversal is no axis change.

- The base penalty for a **90° axis change is twice that for 45°**.
- The first isolated change receives only a small penalty.
- Another change after very few notes, for example two notes, receives a
  strongly increasing penalty.
- Frequent irregular changes receive the strongest penalty and should become
  practically unavailable. Prefer recognizable change patterns while still
  allowing changes.
- Regularity can reduce the extra irregularity penalty; frequent changes
  still incur their frequency penalty.
- In the user's example
  `0-45-0-90-0-0-45-0-0-0-45-0-45-0-0`, **0 means no axis change**.
  Consider both change magnitude and cadence; there is no fixed requirement
  to change axis after a particular number of notes.

Comfortable long runs on one axis are allowed. Keep their duration and fast
reversal counts in reports, without forcing changes or variety to lower them.

## Placement frequency — medium priority, active

Keep the existing placement measures active: top-row frequency, outer columns,
opposite-half placement per hand, and row/lane concentration in short stretches.
Defer further refinement until the primary problems have improved; return to
this if playtesting still identifies it. These measures supplement the
preparation/reach rules above.

## Lower-priority and monitoring measures

- **Travel/recovery:** retain the working distance-and-gap approach. Reducing
  its time allowance from 0.36 s to **0.18 s is a candidate**, not a firm
  requirement. One continuous larger jump should require less time than two
  separate smaller jumps covering the same total distance.
- **M1 lateral load:** watch whether path separation and fewer axis changes
  improve it. Avoid independently forcing vertical cuts to lower this score.
- **M2 visibility:** the gameplay view is between the middle and top rows, so
  upper-band obstruction matters more. Retain helpful existing improvements;
  focus on unexpected patterns rather than blanket penalties for ordinary
  patterns. Do not make this a major direct optimization target now.
- **M3 coordination:** monitor at medium priority, allowing exceptions and
  independent hand motion. Revisit if it remains a problem after the primary
  changes; purely coordinated notes are not the goal.
- **Fast reversals and long axis runs:** lowest-priority monitoring. Ordinary
  alternating cuts can be comfortable; a high counter alone is not a repair
  instruction. The user generally does not want forced variety implemented.

## Calibration and implementation status

Calibrate soft metrics against the curated human baseline using eligible
notes/pattern events as the denominator: target at most **3%** triggering per
specific metric, allowing up to about **10%** for coarse or ambiguous proxies.
These budgets concern flagged events, not raw shares such as top-row occupancy.
Explicit hard bans remain hard bans even if a few human examples violate them.

The opt-in `priority` policy now implements these decisions with shared
`priority-events-v2-segments-20261108` measurements, human-reference thresholds,
bounded joint position/arrow repair and final serialized-map hard audits.
Historical `combined` and other experiment arms retain their recorded definitions;
their playtest reports remain evidence of those definitions. Both priority and
master playability switches remain disabled by default pending player feedback.

The implementation also permits weighted soft tradeoffs: reducing severe reach
can add slower intersections or rare larger movements. Such outcomes must be
reported alongside local flags and raw severity, rather than presented as an
improvement to every metric. Full CPU checks, fresh generation and pack receipts
are recorded in `.superpowers/sdd/2026-11-08-priority-playability/progress.md`.
Automated checks do not establish player acceptance.
