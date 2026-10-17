# V2 concept pipeline and training

The concept pipeline implements the stage contracts in
[model-architecture.md](model-architecture.md). It connects the retained
intensity/timing pair and audio encoder to **positions → arrows → multiplier →
hazards → arcs**, with lighting on its own timeline. The former joint `notes`
checkpoint is not used by this composition.

This milestone prepares training and intermediate predictions. The default
concept command stops at a named review boundary and saves its partial result.
Movement, swing, clearance, final BPM alignment and package export still need
review. A diagnostic bypass records those unresolved boundaries in the output;
it does not produce a playable map or approve its safety.

## Data and model contracts

All times are elapsed seconds. Requested NPS counts retained colored notes before
expansion; a red/blue pair counts as two. BPM stays a separate input. Existing
intensity/timing features, synchronization, targets, splits and checkpoints are
preserved. The new package is `training.pipeline_v2`; its versioned artifacts are
separate from `training.beat_models` artifacts.

The immutable reader reuses the beat framework's v2 Info and ordinary v2/v3 note
support. It adds bombs, ordinary walls, basic lighting, arcs and burst chains,
preserving source records and identifiers. Existing constant-BPM, zero-offset,
Standard-characteristic and ordinary-geometry restrictions still apply. Missing
or changed source files and decoding failures are reported per song. Unsupported
objects remain in diagnostics and receive masked targets; they are not silently
relabeled as negative examples. Original prepared triples retain their existing
unverified provenance flag; intact packages are preferable for object families
that older preparation may have removed.

Model tuning and future training parameters live in `tools/config/config.py`:
`positions_*`, `arrows_*`, `multiplier_*`, `hazards_*`, `lighting_*`, `arcs_*`, plus
`pipeline_v2_*` preparation and scheduler settings. Shared history length is
currently changed together across the five event models. The default is 32;
lighting defaults to blocks of 16. Every effective configuration is recorded.

| Stage | Target and loss | Context |
| --- | --- | --- |
| Positions | Red/blue 13-way categories; masked mean categorical loss | Retained encoder latent, preceding positions and ages, gap/count/intensity/NPS/BPM |
| Arrows | Red/blue nine-way directions; absent hands masked | Fixed positions, neighboring positions/gaps, preceding arrows, latent and conditions |
| Multiplier | Red/blue bounded template categories, including NONE | Directed retained notes, neighboring retained notes, preceding expansions, latent and conditions |
| Hazards | Wall-template softmax plus 12 independent bomb probabilities; categorical/BCE losses | Expanded-note/chain occupancy, candidate interval, preceding hazards, latent and conditions |
| Arcs | Attachment softmax, two bounded handle controls, three anchor modes | Existing same-hand endpoints, preceding arcs, latent and conditions; control losses masked unless attached |
| Lighting | Independent type/value probabilities; masked BCE | Current latent/time block and a separate strictly preceding history block |

Five new event models follow Dense64/Embedding16/GRU64/Dense128 designs. Lighting
retains the documented Conv2D/LSTM/Dense1024 topology; its class count comes from
the frozen training vocabulary, so its parameter count depends on the dataset.
The old lighting checkpoint is not compatible with the new schedule and
multi-label targets. Padding is masked before embeddings/recurrent layers;
lighting's current-block and history masks are independent.

Measured CPU construction with the synthetic frozen vocabulary (3 multiplier,
2 wall and 2 lighting classes): positions **40,970**, arrows **53,362**,
multiplier **64,054**, hazards **50,062**, arcs **51,559**, lighting **6,584,928**
parameters. Template-dependent heads and the retained lighting network grow
with the real training vocabulary. These are untrained model sizes.

### Decisions agreed during implementation

- **Encoder:** unchanged retained weights and 14,800 Hz, two-second, 24×20
  spectrogram construction. A fixed half-second audio grid establishes one
  min/max scale per recording. Later query sets reuse that scale. This approved
  normalization change makes latents independent of which stage requests them.
  Edge patches are zero padded and silence has an explicit finite fallback.
  Queries use the beat framework's canonical decoded duration. Any rate-specific
  decoder tail shortfall is zero padded in the patch and recorded; native samples
  and event timestamps remain unchanged.
- **Multiplier:** unchanged retained timing targets. Only notes already grouped
  by the existing 20 ms synchronization rule and explicit chains become expansion
  targets. Later normal notes remain independent timing events. Copies inherit
  the anchor's hand and direction; conflicting/dot groups and unsupported
  templates are preserved and masked. Timing offsets are bounded template data,
  quantized separately from the synchronization threshold. Anchors never move.
- **Lighting:** simultaneous type/value commands form multi-label targets;
  an all-zero vector means no command. Current audio/time blocks predict their
  own targets, including partial tails. Unsupported brightness/custom events are
  retained in diagnostics rather than squeezed into type/value labels.
- **Hazard positives:** human wall intervals and bomb times supplement scheduler
  proposals during supervised preparation. Their origin is recorded. Inference
  uses scheduler proposals only; reports separate source-conditioned accuracy,
  scheduler-conditioned accuracy and scheduler coverage.

Multiplier, wall and lighting vocabularies are fitted on training recordings
only, capped by configuration, frozen in JSON, and fingerprinted. Validation,
test and curated comparison preparation reuse them. Unknown classes are masked
and reported. Arc handles use `mu`/`tmu`, initially bounded to `[0,2]`; out-of-range
controls are reported. Native source arcs need two identifiable existing
same-hand endpoints. Synthetic start arcs and invented endpoints are excluded.

Schedulers provide explicit candidate times: wall-gap and note-aware strobe
proposals, an independent bomb grid, audio-onset/BPM lighting candidates and
adjacent same-hand arc endpoint pairs. They do not establish collision clearance
or gameplay quality. Exceeding the configured candidate cap raises a reported
error instead of silently trimming the song.

## Preparation and later training

Run commands from the repository root. These examples use existing local assets;
choose a **new output directory** on every run. No command below was used to train
models during this implementation.

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh

# Reuse the existing training split and beat-feature normalization.
python -m training.pipeline_v2 prepare \
  --beat-dataset ../Data/training/beat_models_v2/full-2026-10-17/dataset \
  --encoder-checkpoint ../Data/model/fav_15/tf_model_enc_16bneck_5_5__22_3.h5 \
  --output ../Data/training/pipeline_v2/full/dataset

python -m training.pipeline_v2 validate \
  --dataset ../Data/training/pipeline_v2/full/dataset
```

Preparation runs only the frozen encoder to cache inputs. For intact raw
packages, replace `--beat-dataset` with `--source-root /path/to/packages` and
provide `--reserved-root ../Data/training/evaluate_beats/bs_map_input`.
Use `--normalization-from` with the existing beat training dataset if retaining
the trained intensity/timing pair. `--prepared-root` explicitly selects the
existing beatmap/Info/audio triple adapter. It is never selected implicitly.

Datasets use JSON metadata and non-executable NPZ arrays. The writer emits one
song at a time; the training loader streams songs and batches. Source hashes,
split groups, normalization, encoder identity, vocabularies, target coverage and
skips are recorded. `--structure-only` substitutes marked zero latents for
structural inspection and is rejected by actual training.

For future GPU training, activate the existing GPU profile and train each stage
separately. Existing intensity/timing commands remain documented in
[beat-model-training.md](beat-model-training.md); they do not need retraining for
this milestone. The retained audio encoder is loaded explicitly and frozen.

```bash
source /home/frede/.local/share/infernosaber-baseline/gpu/env.sh

# First validate one real training batch without optimizer updates.
python -m training.pipeline_v2 train --stage positions --dry-run \
  --dataset ../Data/training/pipeline_v2/full/dataset \
  --run-dir ../Data/training/pipeline_v2/runs/positions-dry-run

# Run these independently when ready; each destination must be new.
python -m training.pipeline_v2 train --stage positions --dataset ../Data/training/pipeline_v2/full/dataset --run-dir ../Data/training/pipeline_v2/runs/positions
python -m training.pipeline_v2 train --stage arrows --dataset ../Data/training/pipeline_v2/full/dataset --run-dir ../Data/training/pipeline_v2/runs/arrows
python -m training.pipeline_v2 train --stage multiplier --dataset ../Data/training/pipeline_v2/full/dataset --run-dir ../Data/training/pipeline_v2/runs/multiplier
python -m training.pipeline_v2 train --stage hazards --dataset ../Data/training/pipeline_v2/full/dataset --run-dir ../Data/training/pipeline_v2/runs/hazards
python -m training.pipeline_v2 train --stage lighting --dataset ../Data/training/pipeline_v2/full/dataset --run-dir ../Data/training/pipeline_v2/runs/lighting
python -m training.pipeline_v2 train --stage arcs --dataset ../Data/training/pipeline_v2/full/dataset --run-dir ../Data/training/pipeline_v2/runs/arcs
```

The new stages use fixed upstream reference inputs and strictly preceding
teacher-forced history during training. Defaults are Adam `5e-4`, clipping at 1,
80 epochs, validation early stopping after 10 epochs, batch 128 (lighting 8).
`--epochs` and `--batch-size` override config. Each run saves the best `.keras`
checkpoint with its compatible encoder, vocabulary, data and architecture
identity. Training itself and GPU execution remain unverified in this milestone.

### Framework and concept commands

```bash
python -m training.pipeline_v2 prepare --dummy --output /tmp/v2-dummy-dataset
python -m training.pipeline_v2 smoke --backend dummy --output /tmp/v2-dummy-smoke
python -m training.pipeline_v2 smoke --backend keras --output /tmp/v2-untrained-smoke
python -m training.pipeline_v2 concept --dataset /tmp/v2-dummy-dataset \
  --requested-nps 5 --output /tmp/v2-concept
```

The default concept command saves `concept.json` and returns exit code 3 at
`bpm_alignment`. Other named review stops are `position_movement`, `arrow_swing`,
`hazard_clearance`, `arc_clearance` and `export`. `--diagnostic` explicitly permits
traversal with all unresolved reviews recorded. Dummy smoke uses a dummy-only
contract mode to exercise the entire sequence. Keras smoke builds untrained
networks, differentiates losses and checks serialization; it makes no optimizer.

After later training, `concept --backend keras` requires `--beat-run` selecting
the retained intensity/timing run, `--encoder-checkpoint`, and each of
`--positions-artifact`, `--arrows-artifact`, `--multiplier-artifact`,
`--hazards-artifact`, `--lighting-artifact`, `--arcs-artifact`. It checks
compatibility before composition. There is no implicit newest-model selection.
`--max-events` is an explicitly marked partial diagnostic.

## Independent comparisons — prepared, not executed

The existing intensity/timing comparison remains `evaluation.evaluate_beat_algorithms`.
The new `evaluation.evaluate_pipeline_v2` command compares one downstream stage
at a time, holding upstream reference objects and preceding reference history
fixed. It compares against human targets and training-only frequency/NONE
baselines, with source coverage and unsupported objects reported separately.
The encoder comparison supports latent/preprocessing checks and optional retained
autoencoder reconstruction. These are not end-to-end gameplay scores or claims
that a new stage beats the old complete generator.

Prepare the curated seven songs using the **frozen training** vocabulary,
normalization and encoder. This is a later command, not an executed comparison:

```bash
python -m training.pipeline_v2 prepare --evaluation-only \
  --training-dataset ../Data/training/pipeline_v2/full/dataset \
  --source-root ../Data/training/evaluate_beats/bs_map_input \
  --encoder-checkpoint ../Data/model/fav_15/tf_model_enc_16bneck_5_5__22_3.h5 \
  --output ../Data/training/pipeline_v2/curated/dataset

# Default invocation only describes the comparison; it loads no data/models.
python -m evaluation.evaluate_pipeline_v2 --stage positions \
  --dataset ../Data/training/pipeline_v2/full/dataset \
  --evaluation-dataset ../Data/training/pipeline_v2/curated/dataset \
  --artifact ../Data/training/pipeline_v2/runs/positions \
  --output ../Data/training/evaluate_beats/results/v2-positions
```

Select `arrows`, `multiplier`, `hazards`, `lighting`, `arcs` or `encoder` for the
other independent comparisons. An encoder comparison additionally needs an
explicit curated `--audio` and its retained checkpoint; `--autoencoder-artifact`
is optional. Only explicit `--execute` performs a comparison. Favorites and
unrelated recordings are rejected. Hazards expose augmented-reference,
scheduler-only and human-positive-only scores. Arbitrary re-encoding overlap
cannot be inferred from hashes and remains a stated limitation.

Comparison tests carry the `comparison` marker and are opt-in. Framework
acceptance uses `python -m pytest -q -m 'not comparison'`; neither the comparison
tests nor the seven-song model comparisons are run in this milestone.
