# Hazards, lighting and arcs: repaired training contracts

The repaired contracts are preparation/inference policies, not new network
designs. Lighting's output width follows its new effective vocabulary. Existing
datasets and training runs remain unchanged. On 2026-10-19, the repaired arc
checkpoint was installed in `Data/model/v2_expert` at the user's explicit request,
with validation-calibrated threshold **0.18**. It failed autoregressive positive
evaluation: **0/358 validation arcs** and **0/66 curated arcs** recovered. Calibration
completed, but did not establish a useful model; see the [dated baseline review](baseline.md#arc-only-deployment-and-failed-positive-case-evaluation-2026-10-19).
The previous bundle is preserved as `Data/model/v2_expert_before_arcs-20261019-191457`.
On 2026-10-20, the repaired hazard checkpoint from
`runs/sparse-repair-20261019/hazards` was also installed at the user's request,
with validation-calibrated bomb threshold **0.97** and categorical wall selection.
Predicted-history validation F1 is only **0.02952 for walls / 0.002513 for bombs**;
the seven curated songs recover **0/209 supported wall sections / 0/52 bomb cells**.
The failed quality screen and backup
`Data/model/v2_expert_before_hazards-20261020-161736` are recorded in the
[hazard deployment review](baseline.md#hazard-only-deployment-and-cpu-validation-2026-10-20).
Production lighting uses only the original v1 `fav_15` checkpoint and vocabulary.
Independent v2 lighting is **not yet implemented for production — optional/deferred**;
its contracts and tools below remain experimental. Useful autoregressive
hazard and arc recovery remain unresolved. Experimental
v2 lighting training or bundle replacement does not change the production path.
Wall vocabulary fitting and ordering remain unchanged: this corpus retains the
17-category wall head and 19-row history embedding. Unsupported geometries stay
masked or rejected; eligibility filtering does not resize the hazard network.

## Contracts

**Hazards:** preparation, concept composition and production use the same
quarter-beat wall sections and bomb queries. Notes elsewhere on the grid do not
remove a query. Wall endpoints round to the nearest boundary, choosing the
earlier boundary on ties. Original records and displacement remain in the audit.
Identical overlapping geometries are united; incompatible geometries mask wall
supervision. Bombs map once to their nearest query, with separate cell masks.
Frozen spatial checks include notes, chain occupancy, swing paths and obstacle
controls. Consecutive cleared identical wall sections merge; NONE closes a run.
The 0.1-second minimum applies after merging. Short source walls are reported,
not lengthened. History contains cleared section decisions, including pending
runs that may subsequently be discarded. Final clearance never moves notes.
Unsupported records retain their raw JSON. Recoverable positive wall durations
mask the full source span; explicit nonpositive durations mask the onset.
Unrecoverable geometry/intervals use conservative supervision masks and remain
visible in the audit. Source-wall reporting distinguishes fully and partially
represented objects.

**Feasibility-masked decode (experimental):** `options["hazard_feasibility_decode"]`
(config `v2_hazard_feasibility_decode`) evaluates the same frozen clearance rules
*before* each decision — a wall argmax on an infeasible class is vetoed to NONE
and infeasible bomb cells are zeroed. A veto is a no-op versus the decoder's
built-in post-decision rejections; the flag exists for the masked-sampling path
and diagnostics (`wall_mask_redirects`, `bomb_cells_masked`).
`options["hazard_sample_temperature"]` (config `v2_hazard_sample_temperature`,
0 disables) replaces argmax/threshold with masked sampling: walls draw from the
temperature-scaled feasible-class softmax including NONE, bombs draw
temperature-scaled Bernoulli per eligible cell. Replay seeds via
`hazard_sample_seed`; live generation samples unseeded per map.

**Feasibility inputs (gated training-time variant):** `hazard_feasibility_input`
(config `pipeline_v2_hazard_feasibility_input`, default off) adds two per-query
inputs to hazards: `feasible_wall` (one bool per wall class, NONE always legal)
and `feasible_bombs` (12 bools), evaluated through the same frozen clearance
rules under the evolving decoder state. Preparation attaches them under
reference decisions; rollout attaches them under the model's own decisions, so
train and decode semantics match. The hazard model consumes both as features and
applies them inside the heads — a `MaskLogits` layer before the wall softmax and
a multiply after the bomb sigmoid — so the heads are probability distributions
over *legal* options only. Ordinary categorical/binary losses need no change:
infeasible classes carry no probability mass and produce no gradient, so
"impossible" options are neither negatives nor wasted model capacity. The flag
is recorded in `stage_data` only when enabled, rides bundle assembly like other
contract fields, and each checkpoint derives its input signature from its own
`input_shapes` — legacy datasets and bundles stay byte-identical. The decode-side
`hazard_feasibility_decode` flag remains useful for bundles without these inputs.

**Factorized bomb head (Arm F, gated):** `hazards_bomb_presence` (default off)
adds a scalar sigmoid `bomb_presence` output — "does any bomb belong at this
query" — alongside the masked 12-cell `bombs` scores that answer "where". The
masked-sigmoid cells alone learn legality but not firing rate: their rankings
separated in aggregate yet carried no per-query calibration, which is why
threshold decode stayed silent and rate-matching flooded. Supervision is
*derived* inside `loss_components` from the existing cell targets/weights —
presence target = any supervised positive cell, presence weight = any
supervised cell — so no re-preparation or batch schema change is needed, and
multi-bomb queries (≈43% of positives) keep all their cells. The presence
component joins loss aggregation and per-head epoch metrics with default
weight 1.0; configured `hazards_loss_weights` stay `wall`/`bombs` only. At
decode (rollout and calibration replay alike) the firing score is
`p_presence × p_cell` per eligible cell, so the swept threshold calibrates the
product — the query-level "when" gate suppresses per-map floods and silences
maps where presence never rises. `build_model`/`load_stage_artifact` read the
flag from `manifest["training"]["bomb_presence"]` (recorded via training
settings), so artifacts rebuild byte-exact and legacy checkpoints load
unchanged; `bomb_presence=True` on a non-hazards stage raises.

**History-corruption keep-guard:** corruption drops positive history records
per-record at `rate`/`drop`, then restores the most recent positive record in
any sample that would otherwise lose *all* of them. Fully-emptied windows taught
"fire under all-NONE history" — the poisoned cold-start conditional behind
start-regime spam — whereas partial amnesia is the realistic decode failure.

**Experimental v2 lighting:** the `default_basic_v1` profile supports types 0–4/values 0–12,
ring types 8/9 canonicalized to value 0, and speed types 12/13/values 0–15.
Brightness remains fixed at 1. Custom dependencies, Chroma colors, unsupported
brightness and other families remain preserved and excluded with reasons.
Semantics depend on the type; there is no universal maximum event value.
See the [lightshow format](https://bsmg.wiki/mapping/map-format/lightshow.html)
and [official editor terminology](https://beatsaber.com/documentation/terminology/index.html).

Preparation tests beat grids `.5, .25, .125, .0625, .03125` and second grids
`.04, .02`, each with onset candidates and final-tail coverage. It freezes the
fewest-query schedule covering at least 95% of eligible TRAIN commands within
20 ms after collision handling; ties use timing error, then that listed order.
A failed gate produces an audit and stops preparation. Candidate caps are
explicit; no truncation or tolerance widening is permitted. The vocabulary
contains only effective supervised TRAIN positives. Masks are per class/type;
unsupported commands cannot erase unrelated supported commands. Same-type
conflicts mask that type. OFF is a command; zero targets mean no command.
Every real query enters history, including empty projections. Decoding keeps
the highest-scoring value of each type, then applies its calibrated threshold.

**Arcs:** preparation, scheduling and validation share inclusive `.15–.8 s`
eligibility. `pipeline_v2_arc_use_legacy_gap` can additionally restrict it using
the legacy slider bounds. Endpoint, movement, chain, hazard and probability
filters remain. Accepted history is `[1, mu, tmu, anchor]`; NONE/rejected history
is `[0, 0, 0, 0]`, including legacy composition. Equal-time decisions never enter
one another's preceding history. Legacy cached NONE rows are normalized on read
without modifying the stored arrays or checkpoint graph.

An arc-only, fingerprinted split overlay sorts positive TRAIN recording groups,
permutes them with NumPy seed 3, and moves the first two complete groups to
validation. Difficulties and duplicates stay together. Original validation and
TEST assignments remain; other stages retain their global splits. The selected
groups were previously exposed to upstream models. This is an arc-stage holdout,
not a clean end-to-end holdout; the existing arc-positive TEST has only one
recording group.

## Learning and reporting

Network widths, learning rates, batch sizes and accelerated stage-only loading
remain. The hazards/arcs trainers and optional experimental v2 lighting trainer
shuffle recordings and examples. Wall occupancy and arc attachment use inverse
binary-group frequency; lighting/bombs use
per-class positive BCE weights capped at 100. Statistics come only from effective
TRAIN. Masks remain supervision, not balancing factors. Shape/anchor losses are
positive-only. Missing positive or negative TRAIN/validation supervision fails
before optimizer creation.

Compiled steps aggregate head numerators/denominators and bounded streaming
metrics. Reports contain the weighted objective, unweighted losses, PR-AUC,
precision/recall/F1 and stage-specific diagnostics. Checkpoint selection maximizes
validation mean wall-presence/bomb PR-AUC, lighting command micro PR-AUC, or arc
attachment PR-AUC. An empty prediction fails positive-case evaluation even if
its loss is low. Approximate PR-AUC uses 512 logit-spaced thresholds.

Two optional levers address the sparse-stage history/coverage failures without
re-preparation or architecture changes, configured per stage:

- `*_enabled_song_supervision` (default `True` for hazards/arcs) masks a head's
  weights to zero for any song with no effective supervised positive for that
  head. Wall and bomb supervision mask independently; unsupported positives
  count as disabled. Masked queries remain inputs; supervision counts, class
  balance and `supervised_values` reflect effective supervision. Training
  reports record masked values and feature-disabled songs under
  `training_data.enabled_song_supervision`.
- `*_history_corrupt_rate` / `*_history_corrupt_drop` (defaults `0.0`/`1.0`,
  disabled) corrupt the own-stage `history` input on TRAIN batches only:
  each selected example's positive history records are emptied to NONE rows
  with probability `drop`, simulating missed autoregressive decisions. Masks,
  ages, targets and weights are untouched; validation/test stay
  teacher-forced. Corruption uses a dedicated seed-derived stream, so runs
  stay reproducible. Reports record `history_corruption`.
  For the categorical note stages (`positions`, `arrows`, `multiplier`)
  corruption instead substitutes each occupied history slot's `(red, blue)`
  class with a uniform random different class at probability `drop`,
  simulating misdecoded own-decisions (wrong cell/direction or an
  absent/extra hand). Sentinels and padding are never touched and an
  impossible all-absent `(0, 0)` row restores one channel. The multiplier
  vocabulary supplies its class count, so its corruption needs a
  `TrainingData` source (the training path always provides one).
- `*_negative_supervision_weight` (`hazards_negative_supervision_weight`,
  `arcs_negative_supervision_weight`; default `1.0`, range `0–1`) scales the
  negative-class factor inside `class_balance` for the stage's classification
  heads (`wall`, `bombs`, `attach`, `anchor`). Downweighting implements
  positive-unlabeled semantics: a reference map's non-placement of a stylistic
  object is treated as weak negative evidence, while positive cells keep full
  weight. Regression heads (`shape`) and `events` are unaffected. `1.0`
  reproduces prior behavior; reports record `negative_supervision_weight`.
- `pipeline_v2_hazard_context_*` (`_seconds` `2.0`, `_back_seconds` `0.5`,
  `_bins` `10`, schedule-recorded) adds a `context` input to hazards: a
  `(bins, 2 hands, 12 cells)` binary occupancy grid of notes and chain slices
  spanning `[now-back, now+forward)`. It answers "is this space free of notes"
  — the structural gap behind random wall/bomb placements. Emission is gated
  on the recorded schedule spec: datasets and bundles prepared before the
  keys existed emit byte-identical legacy inputs, and each hazards model
  declares the input iff its own checkpoint/input_shapes contain it
  (`load_stage_artifact` derives the signature from the artifact, not current
  config). The spec propagates to assembled bundles via `stage_data`; a
  context-trained hazards model therefore requires re-prepared data and a
  bundle carrying the spec.

Calibration ranks `.01–.99` by validation F1 through the real decoder, then
autoregressively replays the best three. F1 ties select the higher threshold.
Wall geometry stays categorical. The sidecar binds thresholds to checkpoint
hashes, semantic contracts and the exact original training split identity.
Calibration rejects compatible-looking data with different indexed splits or
recording identities before loading the model. Calibration uses
reference upstream context and predicted own-stage history, with no optimizer.
Arc calibration requires sliders enabled and records movement/clearance controls;
it disables the random acceptance filter for repeatability. Production controls
remain adjustable. Neither calibration nor bundle assembly establishes gameplay quality.

Per-song and aggregate calibration/evaluation reports split head scores by
history regime (`start`: no positive decision in prepared history;
`continuation`: a positive exists) — the cold-start regime is where the old
models failed autoregressively. `enabled_only` restricts scores to songs where
the feature was actually used, matching always-on generation semantics; the
song counts are recorded under `enabled_songs`. These fields are additive;
threshold selection still uses the existing objective.

Replay reports additionally carry `tolerant` scores alongside strict F1:
decoded predictions are matched one-to-one against reference positives within
`tolerance_seconds` (default `.25`, a schedule option). Wall matches require
overlapping onset/end within the window, bombs require matching grid cell,
arcs require matching head/tail times and endpoint geometry. Strict metrics
remain the calibration objective; tolerant metrics answer "right idea, nearby
placement" separately from exact cell equality.

## Commands

Run from the repository root. Every output directory must be new. Preparation
and dry runs use CPU; the GPU commands below are for later manual execution.
The existing intensity, timing, positions, arrows and multiplier models are
retained.

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python -m training.pipeline_v2 prepare \
  --beat-dataset ../Data/training/beat_models_v2/full-2026-10-17/dataset \
  --encoder-checkpoint ../Data/model/v2_expert/encoder/encoder.h5 \
  --output ../Data/training/pipeline_v2/sparse-repair-next/dataset
python -m training.pipeline_v2 validate \
  --dataset ../Data/training/pipeline_v2/sparse-repair-next/dataset
python main_training.py --models hazards arcs --dry-run \
  --dataset ../Data/training/pipeline_v2/sparse-repair-next/dataset \
  --run-dir ../Data/training/pipeline_v2/sparse-repair-next/dry-run
```

Review `preparation-audit.json` before training: scheduler coverage/query counts,
effective supervision and positive groups, every source exclusion, quantization,
and source versus retained/final objects are distinct measurements. The dataset
manifest records policies, hashes, normalization, vocabulary and split overlay.
Detailed exclusions live in the separate, SHA-256-referenced audit file; the
manifest keeps aggregate summaries. Normal dataset loading does not parse that
large audit. Diagnostic tools can explicitly call
`training.pipeline_v2.artifacts.load_preparation_audit` to verify and read it.

The dataset prepared for this repair is
`../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset`.
The commands below use that dataset and new run/calibration directories.
Use your chosen replacement path instead if you repeat preparation.

```bash
source /home/frede/.local/share/infernosaber-baseline/gpu/env.sh
python main_training.py --models hazards arcs \
  --dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset \
  --run-dir ../Data/training/pipeline_v2/runs/sparse-repair-20261018
```

After those runs, return to CPU for explicit calibration:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
for stage in hazards arcs; do
  python -m training.pipeline_v2 calibrate \
    --dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset \
    --stage "$stage" \
    --artifact "../Data/training/pipeline_v2/runs/sparse-repair-20261018/$stage" \
    --output "../Data/training/pipeline_v2/sparse-repair-20261018-r2/calibration/$stage"
done
```

Prepare the curated seven packages using the frozen repaired training contracts,
then explicitly compare each stage against TRAIN-frequency and empty baselines:

```bash
python -m training.pipeline_v2 prepare --evaluation-only \
  --source-root ../Data/training/evaluate_beats/bs_map_input \
  --training-dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset \
  --encoder-checkpoint ../Data/model/v2_expert/encoder/encoder.h5 \
  --output ../Data/training/pipeline_v2/sparse-repair-20261018-r2/curated
for stage in hazards arcs; do
  threshold=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' \
    "../Data/training/pipeline_v2/sparse-repair-20261018-r2/calibration/$stage/calibration.json")
  python -m evaluation.evaluate_pipeline_v2 --execute --stage "$stage" \
    --dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset \
    --evaluation-dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/curated \
    --artifact "../Data/training/pipeline_v2/runs/sparse-repair-20261018/$stage" \
    --threshold "$threshold" \
    --output "../Data/training/pipeline_v2/sparse-repair-20261018-r2/comparison/$stage"
done
```

Reports include reference-conditioned and own-history autoregressive decoding.
Never tune thresholds on the curated TEST examples. Compare positive cases and
exported objects before promotion; more generated objects alone is not success.

Assemble a separate candidate bundle after reviewing calibration/evaluation:

```bash
python - <<'PY'
from pathlib import Path
from training.pipeline_v2.bundle import assemble_bundle
stages = ('hazards', 'arcs')
run = Path('../Data/training/pipeline_v2/runs/sparse-repair-20261018')
cal = Path('../Data/training/pipeline_v2/sparse-repair-20261018-r2/calibration')
assemble_bundle(
    '../Data/model/v2_expert',
    {stage: run / stage for stage in stages},
    {stage: cal / stage / 'calibration.json' for stage in stages},
    '../Data/model/v2_expert_sparse_candidate')
PY
```

This does not replace `v2_expert` or change configuration. The strict experimental
eight-stage assembly tool retains the original v2 lighting artifact; production
ignores that artifact and continues to use the configured v1 lighting assets.
The existing generation commands can use the hazards/arcs candidate by explicitly
setting `v2_model_bundle_path`.
Use disposable input/output directories and disable Beat Saber library export
for package checks. Keep custom lighting and clean unseen arc-positive evaluation
recordings as separate follow-ups. These commands are later opt-in work; no
training or corpus comparison was performed for the v1-lighting integration.

## Optional experimental v2 lighting

Independent v2 lighting is not yet implemented for production — optional/deferred.
The normal `main_training.py` menu and `--models all` omit it; selecting
`lighting` or `7` reports the explicit experimental trainer command. To pursue
this separate experiment, use a new output directory and the frozen repaired
dataset above:

```bash
# CPU contract check without optimizer updates.
source /home/frede/.local/share/infernosaber-baseline/env.sh
python -m training.pipeline_v2 train --stage lighting --dry-run \
  --dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset \
  --run-dir ../Data/training/pipeline_v2/runs/lighting-experimental-dry-run

# Optional later GPU training; explicitly experimental.
source /home/frede/.local/share/infernosaber-baseline/gpu/env.sh
python -m training.pipeline_v2 train --stage lighting \
  --dataset ../Data/training/pipeline_v2/sparse-repair-20261018-r2/dataset \
  --run-dir ../Data/training/pipeline_v2/runs/lighting-experimental
```

Experimental calibration and independent comparisons retain `--stage lighting`
and require the resulting artifact. Run comparisons only on explicit request.
Those results cannot establish production lighting quality because production
uses the separate v1 model and legacy16 semantics.
