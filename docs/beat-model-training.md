# Intensity, timing, and note learning

The independent `training.beat_models` framework prepares data, defines three
small models, and writes intermediate predictions. It does not call the legacy
map generator, BPM alignment, export, lighting, obstacle, or multiplier stages.
Existing pretrained `fav_15` checkpoints use different inputs and outputs and
cannot be loaded as these new models.

Run commands from the repository root using the selected project Python
environment, with system site packages disabled. CPU verification uses the
[retained CPU profile](development.md). Future GPU training uses the existing
[WSL GPU profile](gpu-execution.md) and `tensorflow.keras`, including
`TF_USE_LEGACY_KERAS=1` where specified by that profile. The framework does not
change installed packages or application paths.

## Targets and timing

The new difficulty value is **retained colored notes per second over the full
decoded audio duration**. A simultaneous red/blue pair counts as two notes and
one onset. Bombs count as neither. This differs from the legacy app's approximate
BPS difficulty setting and from its cached density, which could include bombs.

Reduction first synchronizes nearby source onsets using the existing
`tools.config.config.min_time_diff` value (currently 0.02 seconds). Each event is
anchored at its earliest timestamp and includes onsets up to and including that
timestamp plus the configured interval, across both colors. Grouping does not
chain through the last merged note. Merging is silent; source/group membership
and counts remain recorded in the dataset. For each event/color group with
one common non-dot direction, select the first position along the swing; a down
cut selects the highest note. Conflicting directions, dots, and projection ties
retain original source order and are counted in diagnostics. Original colored
note JSON records, numeric records, and retained source indices are saved for
later expansion work. The `original_event_indices` array maps every original
colored note, including discarded notes, to its synchronized event. Original
timestamps are preserved even when a retained note belongs to an earlier event
anchor. Intensity bins use the synchronized event timestamp consistently with
timing and note targets.

Supported inputs initially use v2 Info and ordinary v2/v3 notes, constant positive
BPM, zero offset/shuffle, integer 4-by-3 positions, and directions 0 through 8.
Unsupported timing, direction-angle offsets, geometry, versions, missing files,
and decode errors are reported individually. A prepared-input adapter uses exact
beatmap/Info/audio basename triples; it records original difficulty and
characteristic provenance as unverified. Raw Standard maps first use the exact
Info filename. If missing, the reader accepts an unambiguous case-only match or
the conventional `<difficulty>.dat` / `<difficulty>Standard.dat` pair, in either
direction. This restores compatibility with the old cleanup's filename renames
without modifying source files. Arbitrary replacement names, ambiguous matches,
and aliases referenced by another beatmap remain unsupported. Declared/resolved
names and resolution policy are recorded in provenance and
`preparation.filename_fallbacks`. Audio references and prepared triples retain
their exact-reference policy.

During audio decoding, only the complete native FFmpeg line
`[vorbis @ 0x...] Could not update timestamps for discarded samples.` is filtered.
Other decoder warnings/errors are forwarded unchanged, and decode exceptions
still skip the affected map with a reason. The filter is scoped to decoding;
waveform samples and timestamps are unchanged. Deferred import work, including
optional BPM-change support, is tracked in [TODO.md](../TODO.md).

Every stage uses one audio timeline, defaulting to mono 16 kHz, a 512-sample centered Hann FFT,
hop 160 (100 frames/second), 80 log-mel channels, volume, and onset strength.
Padding is explicit; the last decoder block is trimmed to its real sample count.
Normalization is fitted only on training groups, with separate fine/coarse
statistics. The old training-frame deletion, frequency-major reshape, and
`config.beat_spacing` conversion are not used.

Intensity is a full-song probability distribution over one-second bins. For
bin duration `w`, song duration `T`, share `p`, and requested NPS `D`, the local
requested rate is `p * D * T / w`. The final bin retains its actual duration.
Shares sum to one for nonempty targets; empty-note examples carry a false target
mask rather than uniform labels. BPM is a separate tempo feature and does not
multiply the requested NPS. Zero requested density produces no note events.

Timing labels are counts 0/1/2 at the nearest valid 100 Hz frame after source
synchronization. The default 20 ms interval resolves near-time frame collisions.
If a smaller configured interval leaves a collision between separate events,
that map remains unsupported. The original exact seconds remain in the dataset.
There is no musical-grid snap or forced global
note budget; requested NPS is conditioning, not a guaranteed output count.

Note categories are `0 = no note`, otherwise
`1 + (y * 4 + x) * 9 + direction`, independently for each hand. History uses
separate START=109 and PAD=110 tokens and an explicit mask; a zero-valued absent
hand is valid history. History contains only preceding events from the same
song. Constrained decoding emits one occupied hand for timing count one and
both hands for count two, conserving the timing stage's note count.

## Model contracts

All features and model probabilities are float32; category IDs are integers,
masks are boolean, and stored exact timestamps use float64 seconds.

| Stage | Inputs | Outputs |
| --- | --- | --- |
| Intensity | Whole-song coarse audio `[B,S,82]`, bin durations and masks `[B,S]` | Shares `[B,S]` |
| Timing | Audio `[B,1020,82]`, four conditioning features and validity mask | Count probabilities `[B,512,3]` |
| Notes | Audio patch `[B,200,80,1]` and mask `[B,200]`, both-hand history `[B,32,2]`, ages, history mask, six current-event features | Red and blue probabilities `[B,109]` each |

The four timing features are relative intensity, requested NPS/10, local NPS/10,
and BPM/120. The six note features prepend previous-event gap in seconds and
required count. A timing window contains 512 core frames and 254 context frames
on each side; only core predictions are emitted. Partial tails retain masks.

Intensity uses 48-channel residual temporal convolutions with two kernel-3
convolutions per block and dilations 1/2/4/8 (61-second coarse context). Its
log-density output receives `log(bin_duration)` before whole-song masked
softmax. Timing uses 64 channels and dilations 1 through 64 (509 fine frames).
Temporal blocks use layer normalization, ReLU, and dropout 0.1.
This is a compact temporal-convolution starting point, informed by
[Bai, Kolter, and Koltun's sequence-model study](https://arxiv.org/abs/1803.01271).
Its beat-mapping quality remains untested until training and held-out evaluation.

The note model uses CNN16/32 and Dense64 audio features, shared 16-dimensional
hand embeddings with history dropout, GRU64 over 32 preceding events, a shared
Dense128 trunk, and two 109-class heads. Unlike the old mapper, the audio patch
and target belong to the same current event. Each model must remain below one
million parameters. The initial models contain 60,481 intensity parameters,
180,483 timing parameters, and 173,450 note parameters.

Losses are distribution cross-entropy for intensity, masked categorical
cross-entropy plus 0.1-weight normalized count Huber loss for timing, and mean
red/blue categorical loss for notes. The count auxiliary compares the valid
core's expected count with its labels, not with a whole-song average.

## Data isolation and artifacts

Split recording groups before windows are extracted, using seed 3 and an
80/10/10 train/validation/test division. All difficulties and identifiable duplicate
recordings stay together; aliases support re-encoded recordings. Identifiable
curated benchmark audio is reserved from training. Unknown overlap must be
reported rather than described as held-out generalization.

Datasets use non-executable NumPy arrays plus JSON manifests recording source
hashes, representation versions, split assignments, normalization statistics,
preparation diagnostics, and configuration. Checkpoints carry their stage,
representation and normalization identity, and dummy/untrained/trained status.
Use explicit checkpoint paths and a new output directory for each run; old model
discovery prefixes and implicit latest-checkpoint selection are not used.
Synchronized targets use target version 2 (feature/schema version 1) and record
the configured synchronization interval in the representation. Re-prepare older
datasets and regenerate untrained checkpoints; exact-time target version 1
artifacts are intentionally incompatible. Changing the interval also changes
checkpoint/data compatibility. Normalization continues to use training songs
only.

Training supports reference, noisy-reference, and predicted intensity
conditioning. Noisy reference is the default: mild multiplicative noise followed
by normalization preserves total mass. Predicted mode requires a compatible
intensity checkpoint. Teacher-forced evaluation and composed autoregressive
evaluation are distinct; only the latter exercises predicted note history.
The checkpoint must have `trained` status for predicted downstream training;
explicit untrained checkpoints are accepted for diagnostic evaluation.

Future training defaults are Adam at 0.001, gradient clipping at 1, a maximum of 100
epochs, early stopping after 10 epochs without validation-loss improvement, and
best-checkpoint saving. Default batches are one song, eight timing chunks, and
128 note events. These defaults are configurable and are not measured quality
results.

### Training and model configuration

All new-model defaults live in [tools/config/config.py](../tools/config/config.py),
under `Do not change unless you change the DNN models`. The existing legacy
autoencoder, mapper, beat-prediction and lighting settings keep their original
names and values. The new models have separate sections:

| Model | Prefix | Examples |
| --- | --- | --- |
| Intensity | `intensity_` | `intensity_learning_rate`, `intensity_n_epochs`, `intensity_batch_size`, `intensity_channels` |
| Timing | `timing_` | `timing_learning_rate`, `timing_dilations`, `timing_chunk_frames`, `timing_count_loss_weight` |
| Notes | `notes_` | `notes_learning_rate`, `notes_history_length`, `notes_embedding_dim`, `notes_gru_units` |

Each section includes optimizer name, learning rate, epochs, batch size,
gradient clipping, seed, shuffling, console verbosity, validation monitor and
its min/max direction, plus early-stopping patience and minimum improvement.
Network widths, kernels, dropout, activation and relevant loss weights are
configured alongside the training defaults. Timing and notes separately define
their intensity-conditioning mode and noise standard deviation.

Shared new-model audio settings use `beat_models_` names, keeping them separate
from the legacy `samplerate_music` and `hop_size`. Data splitting uses the
existing `random_seed`, `beat_models_validation_fraction`, and
`beat_models_test_fraction`; the remaining fraction is training data. Timing
context margins are derived from the configured
kernel, convolution count and dilation schedule, so changing the receptive field
cannot silently leave a shorter context window. Class definitions and timing/
density units remain fixed contracts, not tuning parameters.

Explicit `--epochs`, `--batch-size`, and `--conditioning` arguments override the
selected model's config values. Omit these flags to control the run entirely
from config. The configured epoch count is the maximum for that run; it is not
subject to a separate hardcoded 100-epoch ceiling. For example:

```python
intensity_learning_rate = 3e-4
intensity_n_epochs = 150
intensity_batch_size = 2
```

Start a new Python process after changing input/context dimensions. A changed
representation requires a newly prepared dataset and compatible checkpoints;
changes to learning rate, batch size or channel widths alone do not change the
prepared-data contract. With the original defaults, target-v2 dataset and model
compatibility is preserved. Dry-run reports and training manifests record the
effective stage settings, architecture and CLI overrides for reproducibility.

### Intensity training display

Each intensity epoch reports these diagnostics for training and, with the
`val_` prefix, validation:

- `kl_divergence`: cross-entropy loss minus the mean target entropy, in nats.
  Lower is better; an exact prediction has zero divergence.
- `uniform_rate_loss`: reference cross entropy for a constant notes-per-second
  rate. Its shares are proportional to bin duration, including partial tails.
- `uniform_rate_improvement_pct`: percentage reduction in KL divergence relative
  to that reference: `100 * (reference_KL - model_KL) / reference_KL`.
  Zero matches the reference, positive values improve on it, and negative values
  are worse. If reference KL is at most `1e-7`, the undefined percentage is omitted.

Reference values average nonempty songs equally within each split. Empty
intensity targets are excluded. The diagnostics reuse the epoch's existing loss
and add no inference passes or optimizer updates. Early stopping continues to
monitor `val_loss` with the same patience.

The additional curves are saved in `report.json` under `history`; target entropy
and reference values are recorded under `intensity_references`, with explanations
under `metric_definitions`. Intensity dry runs also record `preview_metrics` for
their preview batch. Existing completed runs and checkpoints are not rewritten.

## Commands

The following shell examples run from the repository root. Replace the source
path with the intact training packages. Each destination must be new; use a
different run name on reruns. Keep these artifacts outside the source checkout.
On Windows, use the equivalent Python environment and paths.

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
beat_data="../Data/training/beat_models_v2"
python -m training.beat_models prepare --dummy --output "$beat_data/dummy"
python -m training.beat_models validate --dataset "$beat_data/dummy"
python -m training.beat_models smoke --backend dummy \
  --dataset "$beat_data/dummy" --output "$beat_data/dummy-smoke"
python -m training.beat_models smoke --backend keras \
  --dataset "$beat_data/dummy" --output "$beat_data/keras-smoke" --max-events 8
```

Prepare real training packages using the existing source path setting by
omitting `--source-root`, or select a path explicitly:

```bash
python -m training.beat_models prepare --source-root /path/to/training-packages \
  --reserved-root ../Data/training/evaluate_beats/bs_map_input \
  --output "$beat_data/training"
python -m training.beat_models validate --dataset "$beat_data/training"
```

Use `--difficulty ExpertPlus` to select that Standard difficulty; otherwise all
Standard difficulties are considered. The separate `--prepared-root` adapter
expects `maps/name.dat`, `maps/name_info.dat`, and `songs_egg/name.egg` (or those
exact triples in a single directory). Its Info difficulty listing is historical
provenance and is not used to guess a replacement file. Use `--aliases aliases.json`
for lists of recording-equivalent audio paths/hashes/song IDs, for example
`[["/audio/original.ogg", "/audio/reencoded.ogg"]]`. An alias-to-canonical JSON
object is also accepted. Reservation works by source/decoded hashes and these
explicit aliases; it cannot identify arbitrary re-encodes automatically.

The curated evaluation command keeps accepted maps in the test split and reuses
existing normalization. For an execution-only check, the synthetic training
statistics below are intentional; quality evaluation must use the real training
dataset/checkpoint's statistics instead.

```bash
python -m training.beat_models prepare \
  --source-root ../Data/training/evaluate_beats/bs_map_input \
  --difficulty ExpertPlus --evaluation-only \
  --normalization-from "$beat_data/dummy" --output "$beat_data/curated"
python -m training.beat_models smoke --backend keras \
  --dataset "$beat_data/curated" --output "$beat_data/curated-smoke"
```

Validate future training configurations without creating an optimizer:

```bash
for stage in intensity timing notes; do
  python -m training.beat_models train --stage "$stage" --dry-run \
    --dataset "$beat_data/training" --run-dir "$beat_data/dry-$stage"
done
```

Only when ready to train, activate the selected CPU or GPU environment, then
explicitly omit `--dry-run`. These training commands were **not executed** during
framework implementation:

```bash
python -m training.beat_models train --stage intensity \
  --dataset "$beat_data/training" --run-dir "$beat_data/intensity-run"
python -m training.beat_models train --stage timing --conditioning noisy-reference \
  --dataset "$beat_data/training" --run-dir "$beat_data/timing-run"
python -m training.beat_models train --stage notes --conditioning noisy-reference \
  --dataset "$beat_data/training" --run-dir "$beat_data/notes-run"
```

For predicted conditioning, select `--conditioning predicted` and
`--intensity-artifact "$beat_data/intensity-run"`. Learning requires nonempty,
separate train and validation splits and valid targets for the selected stage.
The loader allocates event patches by batch instead of storing every patch.
Validation uses fixed reference conditioning, or the same explicit intensity
checkpoint in predicted mode. `--epochs` and `--batch-size` override defaults.

Evaluate explicit compatible checkpoints on the test split:

```bash
python -m training.beat_models evaluate --backend keras \
  --dataset "$beat_data/training" --split test \
  --intensity-artifact "$beat_data/intensity-run" \
  --timing-artifact "$beat_data/timing-run" \
  --notes-artifact "$beat_data/notes-run" \
  --output "$beat_data/test-evaluation"
```

For the curated **beat-only** comparison of a trained intensity/timing pair
against the old beat model, use [the beat benchmark](development.md#trained-intensitytiming-versus-legacy-beats).
Its `--backend compare --trained-run <run>` command does not load a notes
checkpoint or execute later generation stages. It reports raw and processed
legacy beats separately, common-reference scores and actual output density.

Evaluation saves per-stage teacher-forced metrics, composed autoregressive
metrics, and compressed prediction arrays. `--requested-nps` overrides reference
density. Optional `--max-events 8` bounds note decoding for execution diagnostics;
reports mark truncated outputs as partial, and their count/quality metrics are
not full-song quality measurements. Omit the cap for a complete evaluation.
Smoke checks use at most eight reference note events per song and the first/last
timing chunks. Runtime versions, source/code hashes, normalization identity, split
membership and artifact status accompany reports. NPZ readers disable pickle.

## Verification limits

Smoke checks run forward passes, losses, gradients, and serialization round
trips on CPU without applying gradients or changing any model weights. The
curated seven-song check exercises preparation and bounded model batches; random
initialized outputs establish execution and contracts, not mapping quality.

No new model has been trained as part of this milestone. New GPU training,
native Windows execution, playability, and complete map generation require
separate validation. See the dated [baseline evidence](baseline.md) for exactly
which checks were completed.
