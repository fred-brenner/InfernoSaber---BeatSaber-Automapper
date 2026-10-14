# InfernoSaber agent instructions

## Direction

Revive the existing Beat Saber automapper incrementally. The first milestone is
to run the existing app and generate a usable map with pretrained models.
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
- For baseline comparisons or revival planning, read
  [docs/baseline.md](docs/baseline.md). It distinguishes verified checks from pending work.
- For the existing user workflow, read [Readme.md](Readme.md).
- For training work, consult
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

- In an environment with the test dependencies, run `python -m pytest -q` for
  relevant code changes. The suite covers song metadata and legacy TCN model loading.
- For audio processing, map generation, or export changes, also exercise a local
  sample song when the runtime is available. Use a disposable data/output folder
  and disable export to the user's Beat Saber library for development checks.
- For the current beatmap syntax, successful map-package generation is sufficient
  baseline acceptance. Use viewer or in-game checks when explicitly requested or
  when changing beatmap syntax or compatibility.
- State which runtime and automated checks ran and which remain unverified.
- For documentation-only changes, check links, consistency, and the final diff.

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
- If relevant, use Subagents for specific tasks, and keep the main agent for overall coordination.
