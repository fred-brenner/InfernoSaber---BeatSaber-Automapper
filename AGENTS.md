# InfernoSaber agent instructions

## Direction

Revive the existing Beat Saber automapper incrementally. App startup, pretrained
model loading, and map generation have been verified on CPU and GPU.
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
- For implemented intensity/timing/note contracts and training status, read
  [docs/model-beat-architecture.md](docs/model-beat-architecture.md).
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
- For the existing user workflow, read [Readme.md](Readme.md).
- For the new intensity/timing/note framework, training commands, data contracts,
  and verification limits, read [docs/beat-model-training.md](docs/beat-model-training.md).
  Its acceptance stops at intermediate predictions; full map generation is separate.
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
- Inspect scripts before importing or running them: `main_app.py` performs a
  GitHub update check at import, and `main_training.py` has top-level GPU checks
  and interactive training orchestration.

## Validation

- For small code changes, run focused tests covering the affected behavior.
- For major or framework changes, run the full `python -m pytest -q` suite in the
  relevant environment(s). The suite covers song metadata and legacy TCN model loading.
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
  Setup callbacks can persist paths into `tools/config/paths.py`; account for these
  writes when checking the diff after a manual app run.
- When changing timing, distinguish elapsed seconds from serialized beat times.
  Preserve the currently implemented v3 output and v2 legacy option unless the
  requested task changes that compatibility.

## Coding Rules

- Preserve the original framework of settings, file paths, etc if possible.
- The project is referenced in my secondary project which builds a Pinokio App out of it. 
  If architectural changes to the framework are required, let me know.
- Only use simple, modular, and efficient code. Avoid complex structures, prefer combining tiny atomic functions!
- You are editing the development branch, and are free to (hard) adjust settings if needed.
- Update the AGENTS.md file according to my usage and the current state of the project. If you find any outdated information, please correct it.
- For large tasks, use subagents with GPT-6.1 Sol XHigh for specific tasks.
- Carefull when changing beatsaber map preprocessing tasks, I've put a lot of work into the current preprocessing pipeline, because beatsaber allows multiple different formats. 
