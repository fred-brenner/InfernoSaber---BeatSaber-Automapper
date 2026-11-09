# InfernoSaber agent instructions

## Direction

Revive the existing Beat Saber automapper incrementally. App startup and legacy model generation were verified on CPU and GPU.
V2 application generation uses CPU, with seven v2 stages (arcs and walls
currently decoded by the pre-model heuristics — `v2_arcs_heuristic` /
`v2_walls_heuristic = True`, with sparse gameplay bombs placed outside the
note safe area (`v2_bomb_note_cell_gap`/`v2_bomb_note_time_gap`) by
`_heuristic_bomb_candidates`, gated by `add_bombs_flag`/`v2_bombs_prob`), a
bounded onset-peak snap post-cleanup
(`v2_notes_onset_snap`), a configurable start-of-map safety zone
(`v2_map_intro_no_notes_seconds`/`v2_map_intro_no_hazards_seconds`), strict
≥135° same-hand flow at fast gaps, with eligible slow turns in the final
priority policy (`v2_flow_min_turn_steps` / `v2_playability_late_turns`), a diagonal-direction cap
(`v2_diagonal_max_fraction`), post-pause entry-ramped, silence-aware timing
fills plus a pluggable max-gap safety net
(`v2_fill_entry_delay_seconds`/`v2_fill_respect_silence`/
`v2_onset_snap_unmatched_policy`/`v2_max_note_gap_seconds`/
`v2_gap_fill_source`), a decorative calm-area pulse pass that adds edge wall
flashes, corner bombs and laser flashes inside intentionally note-free gaps
without touching notes (`v2_calm_effects`), and a standalone append-only
v1-style note emphasis pass restoring slow-swing double/triple stacks after
learned expansion (`map_creation/v2_note_emphasis.py`, gated by
`emphasize_beats_flag`/`single_notes_only_flag`), the retained
encoder, and exclusive v1 `fav_15` lighting; GPU remains the training runtime.
The five v1 presets remain selectable legacy options in the app and CLI
(`fav_15`, `pp3_15`, `easy_15`, `hard_15`, `expert_15`); a v1 selection
downloads its model folder on demand and runs the unchanged legacy pipeline
to per-difficulty `1234_*` folders, which combine into `12345_*` packages
(folder + ZIP) like the v2 path.
WSL/Linux is the primary development environment; preserve Windows app
compatibility. Complete the requested task, including small supporting 
refactors when they simplify the change. Keep unrelated cleanup separate. 
Discuss changes to model architecture, generation semantics, or supported 
platforms before implementing them unless the task already authorizes them.

## Context pointers

- For domain terms, difficulty semantics, or timing units, read [CONTEXT.md](CONTEXT.md).
- For environment setup or running the WSL app, read
  [docs/development.md](docs/development.md).
- For RTX 5080 GPU execution, read [docs/gpu-execution.md](docs/gpu-execution.md).
- For the v2 target stages and their output responsibilities, read
  [docs/model-architecture.md](docs/model-architecture.md).
- For v2 dataset preparation, individual stage training, concept composition,
  review stops and opt-in model comparisons, read [docs/pipeline-v2.md](docs/pipeline-v2.md).
  It also covers training throughput, the `main_training.py` selection script,
  and the evaluated `v2_expert` inference bundle — the history-repair arm B
  install was evaluated live and rolled back (kept at
  `v2_expert_armB-history-repair`); the October bundle is active again.
  Completed training does not establish positive-case or gameplay quality;
  read the recorded limitations.
  Corpus comparisons require an explicit request.
- For repaired hazards/arcs and experimental v2 lighting datasets, masks, arc split overlays,
  calibration, or mixed legacy/replacement bundles, read
  [docs/sparse-stage-repair.md](docs/sparse-stage-repair.md). It also covers the
  optional `*_enabled_song_supervision`, `*_history_corrupt_*` and
  `*_negative_supervision_weight` training levers for sparse-stage history
  dependence and positive-unlabeled semantics, plus window-tolerant replay
  metrics (`tolerance_seconds`, default `.25`), the gated hazards `context`
  note-space input (`pipeline_v2_hazard_context_*`), and the gated hazards
  feasibility-mask inputs (`pipeline_v2_hazard_feasibility_input`). New preparation uses
  versioned sparse-stage policies. Preserve existing datasets and inference
  bundles; repair implementation alone does not establish replacement quality.
- For implemented intensity/timing/note contracts and training status, read
  [docs/model-beat-architecture.md](docs/model-beat-architecture.md).
- For production v2 generation, intensity postprocessing, note-decoding
  temperature sampling (`v2_note_sample_temperature`), model caching, worker
  settings, package formats, and unchanged external commands, read
  [docs/generation-v2.md](docs/generation-v2.md).
- For legacy generation calls, checkpoints, and preprocessing, read
  [docs/model-architecture_old.md](docs/model-architecture_old.md).
- For feature priorities or planning model and pipeline improvements, read
  [docs/roadmap.md](docs/roadmap.md).
- For optional import fixes and small follow-ups, read [TODO.md](TODO.md).
- For recorded baseline results or comparisons, read
  [docs/baseline.md](docs/baseline.md). It distinguishes verified checks from pending work.
- For human-map beat benchmarks, trained intensity/timing comparisons, reference
  timing rules, and reruns, read
  [docs/development.md](docs/development.md#human-map-beat-benchmark).
  Use only the curated `evaluate_beats/bs_map_input` packages for comparisons;
  exclude `favorites_bs_input` and report skipped packages individually.
  Favorites remain valid *training* input: `--source-root` can be repeated to
  ingest several package folders, with automatic beatmap+audio deduplication.
- Before changing playability rules or optimization priorities, read the latest
  [player priority review](docs/superpowers/specs/2026-11-08-playability-priority-review.md).
  It supersedes conflicting goals in earlier experiments; distinguish accepted
  requirements from implemented behavior.
  The revised `priority` experiment is separate from historical `combined`; see
  [priority policy](docs/generation-v2.md#priority-policy-after-the-november-8-player-review)
  for its shared segment detector, joint repairs and final hard audits.
  Current CPU validation and the blind playtest are recorded in
  [baseline.md](docs/baseline.md#2026-11-08--priority-playability-validation-and-playtest).
  The final v2 default extends this policy with separately versioned local M2
  upper-center exposure, calibrated against human events and guarded during
  joint repair. Historical rise-only M2 remains explicit in reports; read
  [final visibility design](docs/superpowers/specs/2026-11-08-final-visibility-design.md).
  M2 stays soft: final-only residuals may exceed their unchanged base allowance
  by at most one onset group's occupancy, at <=3% eligible events per hand.
  Record exceptions separately; never inflate original limits or relax the
  reach/path hard bans or primary guards to pass them.
  Final delivery is five ordinary song packages at settings 5/6/7/8, without
  further A/B rounds; current validation evidence is recorded in the task ledger.
  The November 9 repeated-position pass makes guarded one-cell cut-direction
  moves after calibrated repair; see [repeat separation](docs/generation-v2.md#exact-repeated-position-separation).
  Repair-time evaluation memoizes arrival-speed and segment-geometry results in
  bounded caches and orders cheap conjunction guards before expensive checks;
  all safety semantics are unchanged.
- For the five note-pattern playability metrics, including M5 same-hand direction
  variety, fast reversals and time between changes, read
  [docs/development.md](docs/development.md#playability-metrics).
  The scorer runs offline. The opt-in CPU generation experiments and paired
  comparison instructions are in [generation-v2.md](docs/generation-v2.md#playability-experiments)
  and [development.md](docs/development.md#paired-playability-comparisons).
  `v2_playability_enabled = False` selects the app1 behavior. For the complete
  combined policy, shared acceptance guards, calibrated placement/M5 tails and
  bounded decoder recovery, read [generation-v2.md](docs/generation-v2.md#playability-experiments).
  Calibration budgets apply to eligible flagged events, not raw top-row share;
  keep historical scorecards and distinguish runtime validation from player approval.
  The November 8 offline event-calibration profiles use human trigger budgets
  of 3% for specific measures and up to 10% for coarse proxies; see
  [event calibration](docs/development.md#event-calibration-from-playtest-feedback).
  The combined candidate implements the far-opposite inward-cut ban and
  calibrated reach repair through `v2_playability_reach`. The `combined`
  experiment enables M1–M4, reach, slow turns and dwell together. It repairs
  before the retained snapshot, audits mutable expansions and rejects any
  forbidden final colored cut. Read the combined playtest record in
  [baseline.md](docs/baseline.md#2026-11-08--combined-reach-generator-and-second-playtest).
- For the existing user workflow, read [Readme.md](Readme.md).
- For the new intensity/timing/note framework, training commands, data contracts,
  and verification limits, read [docs/beat-model-training.md](docs/beat-model-training.md).
  It also documents intensity/timing input caching and grouped Keras execution;
  these have separate settings from the downstream trainers.
  Its training-framework acceptance stops at intermediate predictions; production
  composition and package export are documented separately in generation-v2.md.
- For legacy training work, consult
  [the existing installation and training guide](How%20to%20-%20Linux%20install%20for%20training%20and%20app.md)
  and verify its assumptions against the current code and environment.

Historical concept notes are reference material. Their priorities and "DONE"
labels require verification before becoming current requirements.

## Environment and execution

- Run project commands from the repository root; path discovery depends on the
  working directory and the presence of `main.py`.
- Use the selected project environment's Python interpreter with system site
  packages disabled.
- Use the recorded CPU or GPU package lock for the selected environment. Validate
  pretrained model loading when changing TensorFlow, Keras, or related packages.
- Use `tensorflow.keras` consistently. The GPU activation profile selects legacy
  Keras through `TF_USE_LEGACY_KERAS=1`; standalone Keras imports select a different
  API in that environment and break the existing TCN loading path.
- Main-script imports are inert: `main_app.build_app()` explicitly builds the UI
  and checks for updates; its guarded launch is spawn-safe. Generation loads the
  validated seven-stage projection of local `v2_expert`, retained encoder and
  configured v1 lighting assets once per CPU worker, and caches one recording.
  Reapply intensity postprocessing from cached raw shares for each generation;
  keep user-requested NPS separate from effective model conditioning NPS.
  V2 lighting is experimental: not yet implemented for production — optional/deferred.
  Preserve the bundle and experimental strict eight-stage validation/tools.
  Preserve explicit current-run completion records when combining packages; never
  infer success from output-folder existence. `main_training.py` is inert on import; running
  it selects the seven production v2 stages, sequentially by default or with
  bounded parallelism via `--parallel N`, routing intensity/timing and downstream
  models to their respective datasets/trainers. Menu numbers remain 1–6 and 8
  (arcs); lighting/7 directs callers to the explicit experimental pipeline trainer.

## Validation

- For small code changes, run focused tests covering the affected behavior.
- For major or framework changes, run the full `python -m pytest -q` suite in the
  relevant environment(s). The suite covers song metadata and legacy TCN model loading.
- For v2 framework-only acceptance, exclude opt-in model comparisons with
  `python -m pytest -q -m 'not comparison'`. Keep training and corpus comparisons
  separate from synthetic forward/gradient/serialization checks.
- For documentation-only changes, limit validation to reviewing the text,
  relevant links, and the final diff; runtime tests are not required.
- For substantial changes to audio processing, map generation, or export behavior,
  also exercise a local sample song when the runtime is available. Use a disposable
  data/output folder and disable export to the user's Beat Saber library for checks.
- For the current beatmap syntax, successful map-package generation is sufficient
  baseline acceptance. Use viewer or in-game checks when explicitly requested or
  when changing beatmap syntax or compatibility.
- State which runtime and automated checks ran and which remain unverified.
- Only evaluate the CPU runtime, unless the task explicitly requires GPU execution. GPU runtime is only used for training.

## Files and compatibility

- Preserve pre-existing working-tree changes and untracked assets. The nested
  `mnt/` folder contains local model assets; treat it as user data.
- Keep model weights, training datasets, song audio, generated maps, environments,
  and caches out of source commits.
- Preserve the line endings of existing files when editing them. Keep any
  repository-wide normalization separate from functional work.
- Current path helpers expect forward slashes and trailing `/` on directory paths.
  Runtime/app settings writes (`update_dir_path` for `config.py`/`paths.py`)
  redirect to the untracked `tools/config/user_settings.py`/`user_paths.py`,
  which both modules import so tracked files stay clean for `git pull --ff-only`;
  expect these files after a manual app run rather than diffs to the tracked
  config files.
- When changing timing, distinguish elapsed seconds from serialized beat times.
  Preserve the currently implemented v3 output and v2 legacy option unless the
  requested task changes that compatibility.
- Releases: bump `InfernoSaber_version` in `tools/config/config.py`, commit
  `Release X.Y.Z` on `main_app_dev`, fast-forward `main_app`, tag `vX.Y.Z`, push
  branch+tag — then publish the GitHub release on the `fred-brenner/InfernoSaber-App`
  launcher repo (target `main`), which is what the in-app update checker reads.
  A `-reinstall` tag suffix forces the reinstall warning path for older installs.
  Pinokio Update pulls `main_app` of this repo ff-only; tracked files must stay
  clean on user machines (runtime settings persist to untracked
  `tools/config/user_settings.py`/`user_paths.py`).

## Coding Rules

- Preserve the original framework of settings, file paths, etc if possible.
- The project is referenced in my secondary project which builds a Pinokio App out of it. 
  If architectural changes to the framework are required, let me know.
- Only use simple, modular, and efficient code. Avoid complex structures, prefer combining tiny atomic functions!
- You are editing the development branch, and are free to (hard) adjust settings if needed.
- Update the AGENTS.md file according to my usage and the current state of the project. If you find any outdated information, please correct it.
- For large tasks, use subagents with GPT-6.1 Sol XHigh for specific tasks.
- Carefull when changing beatsaber map preprocessing tasks, I've put a lot of work into the current preprocessing pipeline, because beatsaber allows multiple different formats. 
