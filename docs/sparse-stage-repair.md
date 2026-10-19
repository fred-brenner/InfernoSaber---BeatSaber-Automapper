# Hazards, lighting and arcs: repaired training contracts

The repaired contracts are preparation/inference policies, not new network
designs. Lighting's output width follows its new effective vocabulary. Existing
datasets and training runs remain unchanged. On 2026-10-19, the repaired arc
checkpoint was installed in `Data/model/v2_expert` at the user's explicit request,
with validation-calibrated threshold **0.18**. It failed autoregressive positive
evaluation: **0/358 validation arcs** and **0/66 curated arcs** recovered. Calibration
completed, but did not establish a useful model; see the [dated baseline review](baseline.md#arc-only-deployment-and-failed-positive-case-evaluation-2026-10-19).
The previous bundle is preserved as `Data/model/v2_expert_before_arcs-20261019-191457`.
Hazards still use their recorded legacy schedule and thresholds. Production
lighting now uses only the original v1 `fav_15` checkpoint and vocabulary.
Independent v2 lighting is **not yet implemented for production — optional/deferred**;
its contracts and tools below remain experimental. Hazard replacement training
and useful autoregressive arc recovery remain pending. Experimental
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
