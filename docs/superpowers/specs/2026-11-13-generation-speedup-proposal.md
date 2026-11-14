# Generation speedup proposal

Status: Task P2, Task P3 and Task P4 implemented (2026-11-13, commits
4e36119, fe3de57, 318457d, faaad7b on main_app_dev); Task P1 approved for
implementation (2026-11-14); P5, P6, P7 remain proposed.

## Intent

Reduce wall-clock and CPU time for new-map generation without changing
generation semantics, output contracts, or supported platforms. All proposals
preserve the seven-stage projection, the retained encoder, v1 lighting, the
forward postprocessing policy, and per-difficulty `scoped_settings` behavior.
Numerical parity is required only where noted; temperature-sampled outputs are
statistical anyway.

## Measured baseline

From `temp/v2-generation/*.json` (2026-11-11, 149 s song, ~932 events, ~1115
notes, NPS 7, forward mode, heuristic hazards/arcs):

- Per variant (`generation_seconds` ≈ 12.5 s): positions 2.7 s, arrows 3.0 s,
  multiplier 2.5 s — ~930 sequential batch-1 `tf.function` calls each,
  ~2.7-3.2 ms/event. Timing 2.2 s (mostly legacy audio postprocessing, not the
  model). Lighting 0.9 s (57 block calls, 459 MB model). Onset snap 0.5 s.
  Forward repair + emphasis + hazards + arcs + intensity ≈ 0.7 s.
- ~66% of every variant is sequential batch-1 model dispatch.
- Per song/worker overhead: `bundle_load_seconds` ≈ 10.8 s once per worker
  (TF import, bundle sha256, 7 stage models, encoder, 459 MB lighting);
  `preparation_seconds` ≈ 2.5 s per song.
- 9 packages × 5 difficulties on 8 workers: 127.8 s wall vs ~51 s ideal —
  group quantization plus startup.

Per-song total ≈ 2.5 s prep + 5 × ~12.5 s generation ≈ 65 s single worker.

## Ordered proposals

### Task P1 — Cross-variant batched decode (~3x per song; structural)

positions/arrows/multiplier are autoregressive within a variant but fully
independent across the difficulty variants of one song that already run
sequentially on the same worker. Run variants in lockstep: at each event
index, `build_inputs` for every live variant, concatenate into one batch, one
`_predict`, then per-variant decode/history/state update. ~2800 calls become
~930 calls of batch-5.

- Add a `generate_variants` driver in `map_creation/v2_pipeline.py` holding
  per-variant state (candidates, state, history, rng, kinematics, recovery
  counters); `generate_song` calls it once instead of looping
  `runtime.generate` per difficulty.
- Only the model call batches. `build_inputs` reads `settings`/state, not
  `config`; config-dependent steps (`postprocess_timing`, hazard/arc
  validators, repair, emphasis) stay per-variant under `scoped_settings`.
- `v2_arrows_variety` splits rows by chosen model per step (≤2 calls).
- Hazards/arcs model stages get the same treatment when re-enabled; production
  heuristics are unaffected.
- Caveat: batched matmul may shift float reduction order; verify parity at
  temperature 0 in a regression check, statistical equality otherwise.
- Expected: the ~8.2 s of model stages → ~2.5-3 s for all five variants;
  per-song generation ~65 s → ~25-30 s on one worker.

#### Task P1a — Per-variant context refactor (preparatory, behavior-identical)

`generate()` is one long function whose decode loops keep per-variant state in
locals (`history`, `kinematics`, `recovery_*`, `last_hand_time`, candidate
lists, `state`/`result`). Restructure it so each variant run is a context
object the stage loops iterate — the prerequisite for lockstep batching.

- Extract the per-variant mutable state into a dedicated structure and give
  each autoregressive stage (positions, arrows, multiplier) a step-level or
  loop-level function operating on one context; `generate()` keeps its
  signature and composes the same sequence for one variant.
- Behavior must be identical: same call order, same decoded values, same
  diagnostics/stage records, same `scoped_settings` interplay. Existing tests
  pass unchanged; no batching yet in this task.
- Keep hazards/arcs heuristic paths and lighting untouched.
- Success gate: the existing v2 pipeline tests (postprocessing/generation/
  sparse-integration) all pass without modification.

#### Task P1b — Lockstep `generate_variants` driver + wiring

Run K variants of one song in lockstep, batching `_predict` across variants
per event index (≤ ~930 calls of batch-K instead of ~2800 batch-1 calls).

- New `generate_variants(song, requested_nps_list, predictors, encoder,
  settings_list, vocabularies, *, lighting_generator, audio_path, logger,
  postprocessors, rngs) -> list[PipelineResult]` in `v2_pipeline.py` using the
  P1a contexts; per-variant phases that read `config` stay sequential per
  variant; only `_predict` batches (concat inputs axis 0, split outputs).
- `v2_arrows_variety`: group the step's rows by chosen model → ≤2 calls/step.
- Candidate lists differ per variant: iterate to the longest, skip finished
  variants; union all candidate times into one `encoder(...)` prefetch per
  stage (shared latent cache is a bonus win).
- `v2_runtime.py`/`v2_generation.py`: `generate_song` uses
  `generate_variants` when >1 difficulty; `generate`/`runtime.generate` stay
  the single-variant path (eval tools, tests). Per-variant `scoped_settings`
  boundaries must match the old sequential loop exactly.
- Parity gate: at `v2_note_sample_temperature=0` and `random_note_map_factor=0`
  a K-variant batch produces outputs within a documented tolerance of the
  same variants run one-by-one (float-reduction order may differ — justify
  rtol/atol; identical argmax decisions is the real bar).
- Success gate: parity test + existing suite pass; stage `elapsed_seconds`
  for positions/arrows/multiplier drops ~K-fold on a multi-difficulty run.

#### Task P1c — Process-level orchestration: batch the song's difficulty run

P1b ships the driver, but `process_song` calls `generate_song` once per
(song, difficulty), so the standard multi-difficulty app run issues K
single-variant calls and lockstep never engages (only the single-difficulty
auto-Expert twin reaches K=2). This task wires the orchestration that
realizes the headline win.

- Add a group generation entry (e.g. `generate_song_group(name,
  difficulties, logger)` in `v2_generation.py`): one `prepare_audio`, one
  `runtime.generate_batch` over all requested difficulties — then per
  difficulty its own `write_package` into `1234_{d}_{name}` and its own
  completion record (`internal_difficulty`, `path`, `status`, elapsed),
  identical record shape to `generate_song`'s so `_combine_maps`/
  `combine_packages` and package semantics are untouched.
- `process_song` in `main_multi.py`: coalesce consecutive same-song
  (song, difficulty) entries in its work list into one group call (groups
  already keep a song's difficulties adjacent); v1 selection path and
  single-difficulty runs keep today's calls.
- Failure granularity: on a batched-call exception, fall back to the
  existing per-difficulty sequential loop for that song (log the fallback);
  preserves today's per-difficulty failure semantics — one bad variant must
  not silently kill its siblings.
- Preserve the `create_expert_flag` twin semantics for single-difficulty
  runs; group runs (K>1) create no twins, matching `main_multi`'s
  `create_expert_flag=False` for multi-difficulty.
- No `_groups` changes in this task (splitting reduces batch width — the P5
  tension stays documented).
- Success gate: a 3-difficulty app-style run engages lockstep (batch
  predict call counts verifiable in tests), records/records flow identical,
  per-difficulty failure fallback covered by test, existing suite green.

### Task P2 — Per-song audio-analysis cache (~1.5-2.5 s/variant)

Each variant currently re-decodes audio ~3x and recomputes the onset envelope
~3x: `_legacy_timing_features` (aubio + `log_specgram` + PIL resize),
`sanity_check_timing2` (`librosa.load` + `onset_strength` + `onset_detect`),
`_detect_onset_peaks` (same librosa recipe + RMS, also used by onset coverage
and the max-gap "onsets" pool). All are audio-only; only detection thresholds
vary per difficulty.

- Memoize on `(audio_path, size, mtime)` or hang on `PreparedAudio`: decoded
  44.1 kHz mono waveform, onset envelope, RMS, legacy volume/onset windows and
  spectrum. Recompute per-variant only the thresholded outputs
  (`get_silent_times`, `onset_detect` parameters).
- One decode + envelope per song instead of ≥6 per variant. Removes most of
  the ~2.2 s "timing" stage audio cost and the ~0.5 s onset-snap cost.

### Task P3 — Vectorize `spectrogram_patches` (~1-3 s/song)

`training/pipeline_v2/encoder.py` runs a Python loop of per-patch
`scipy.signal.spectrogram` + PIL resize for the ~300 grid patches plus ~1000
unique event centers.

- Stack patches into one `(K, PATCH_SAMPLES)` `signal.spectrogram(..., axis=1)`
  call; vectorize log/min-subtract; keep the PIL resize loop for parity.
- Speeds both `FixedEncoder.__init__` grid normalization and per-stage query
  prefetch inside positions/arrows/multiplier.
- Parity-checkable against the retained encoder representation contract.

Measured: ~1.25-1.3x on the function (~tens of ms/song), well below the
1-3 s estimate — scipy already batched the FFT internally, so only per-call
overhead dropped. Output parity verified bitwise.

### Task P4 — Auto-scale inference threads vs workers (~10-20% under load)

`v2_inference_threads = 2` × 16 workers → 64 contending TF threads plus
OpenMP/MKL. Batch-1 ~1 MB models gain nothing from intraop parallelism.

- `v2_inference_threads = max(1, cores // workers)` at `_run` scope, or a
  config auto mode; set `OMP_NUM_THREADS=1` (and MKL equivalents) in
  `_worker_init` before first TF/numpy-heavy import.
- Most relevant exactly at the 16-worker app ceiling; keep ≥2 only for 1-2
  worker runs where a single worker has idle cores.

### Task P5 — Balance `_groups` when songs ≥ workers (~20-30% on awkward ratios)

`_groups` only splits when `len(groups) < workers`; with 9 songs on 8 workers
no split occurs and one worker takes two songs (~100 s vs ~55 s).

- Split groups until `len(groups)` approaches a small multiple of `workers`
  (LPT-style by difficulty count), accepting duplicated `prepare_audio` on
  otherwise-idle workers.
- Tension with P1: splitting a song reduces batch width. Apply after P1 only
  when it improves makespan; with lockstep, same-song-single-worker is near
  optimal and splitting mainly fixes today's quantization.

### Task P6 — Cut per-call dispatch overhead (~1.3-1.6x residual; more if ONNX)

- Concrete `tf.function`s with fixed `input_signature` per stage.
- Hoist `preparation_defaults(settings)` out of `build_inputs` (re-validated
  per candidate, ~2800x/variant) — pass the precomputed options through.
- Optional later discussion: ONNX Runtime sessions for the hot stages
  (~5-10x on dispatch) — requires conversion tooling, parity validation, and
  bundle packaging changes; needs its own design approval per project rules.

### Task P7 — Pre-work and startup amortization

- `check_music_files` normalization loop (pydub decode + RMS + re-encode,
  ~1-4 s/song) is serial parent work on the critical path: parallelize with a
  thread pool (pydub/ffmpeg releases the GIL) and/or skip already-normalized
  files.
- Worker startup ~11-15 s each is amortized across a batch but dominates
  single-song runs; a resident worker pool across consecutive app runs would
  remove it — architectural change, separate discussion.

## High-CPU relevance

- Per-task work is a serial ~12.5 s Python/TF chain; worker scaling is
  near-linear across songs today, but makespan is quantized per song and hurt
  by TF/OMP oversubscription (P4, P5).
- P1+P2 cut per-song CPU ~3x: a 16-core box then finishes ~16 songs in the
  time ~5-6 take now. RAM (~2 GB/worker: TF runtime + 609 MB bundle incl.
  459 MB lighting) remains the binding constraint at 16 workers.
- Single-song latency: `_groups` already splits difficulties across idle
  workers (~25 s wall vs ~65 s serial); P1 reaches comparable latency on one
  worker with ~5x less CPU — better for queue throughput.
- Clamp `num_workers = min(songs, physical cores)` in the app or document it;
  workers beyond #songs only pay startup.

## Suggested order

1. P2 + P3 + P4 — small, independent, ~30-40% combined.
2. P1 — the structural ~3x win.
3. P5, P6 — follow-ups once P1 lands.
4. P7 — separate UX/architecture discussion.

## Validation

- Reuse the existing per-stage `elapsed_seconds` report fields; add a
  before/after record on the same songs/settings for the task ledger.
- `python -m pytest -q` in the CPU environment; forward/gradient checks cover
  input contracts. Parity check for P1/P3 at `v2_note_sample_temperature = 0`.
- Exercise one local sample song end-to-end with a disposable data/output
  folder; package export to the Beat Saber library disabled.
- High-CPU claims validated by wall-clock comparisons at 1, 4, 8, 16 workers
  on the same batch, not by model.
