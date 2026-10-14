# InfernoSaber revival baseline

Recorded on 2026-10-14, before the initial documentation changes. This is a dated
snapshot, not a claim about the latest upstream state or a completed runtime setup.

## Agreed revival direction

- First milestone: run the existing app and generate a usable map with pretrained models.
- Primary development environment: WSL/Linux, preserving Windows app compatibility.
- Immediate scope: record the baseline and draft agent documentation. Runtime
  setup and generation verification are follow-up work.

## Repository state

| Item | Observed baseline |
| --- | --- |
| Branch | `main_app_dev` |
| HEAD | `60780a5acda4da67d0c67d01a4705719a77a91e2` |
| Latest commit | `macos fix folder selection`, dated 2026-04-28 |
| Tracking branch | `origin/main_app_dev` |
| Cached tracking comparison | 0 commits ahead, 0 behind; no remote fetch performed |
| Tracked files | 85, including 77 Python files |
| Staged changes | None |
| Modified tracked files | 64; all differ only in LF/CRLF line endings |

The line-ending finding was checked by comparing each modified file's bytes with
the Git index after normalizing CRLF to LF. `git ls-files --eol` confirms LF in
the index and CRLF in the working copy for those 64 files. Some other tracked
files already use CRLF in both places. No line-ending normalization was applied.

Pre-existing untracked files include Python caches, `bs_shift/output.csv`, and
the nested `mnt/` directory. The latter contains 24 files totaling 1,703,654,415
bytes, including model weights and encoders under `fav_15` folders. Their
presence was verified; their completeness and loadability were not.

## Current implementation, from source inspection

| Area | Existing entry point or component |
| --- | --- |
| Browser-based app | `main_app.py`, using Gradio |
| Single-run inference | `main.py` |
| Multiple-difficulty inference | `main_multi.py` |
| Interactive training orchestration | `main_training.py` |
| Audio preprocessing | `preprocessing/` |
| Beat timing prediction | `beat_prediction/` |
| Note mapping, postprocessing, and map serialization | `map_creation/` |
| Lighting prediction | `lighting_prediction/` |
| Training-map preparation and export helpers | `bs_shift/` |
| Shared configuration and path setup | `tools/config/`, `app_helper/` |

The implemented generation path combines audio preprocessing, beat timing
prediction, encoded audio for note-pattern prediction, lighting prediction,
and postprocessing for notes, obstacles, and arcs. The source currently selects
the `fav_15` model style and v3 beatmap output, with a v2 legacy option.

The downloader references `BierHerr/InfernoSaber` on Hugging Face and uses the
selected model style as a revision. Download availability and model loading were
not checked. The old README's branch/install advice has not been revalidated.

## Development environment

These observations describe this chat's WSL shell, not a native Windows or
Pinokio installation.

| Item | Observed baseline |
| --- | --- |
| Execution environment | WSL2/Linux; repository on the Windows filesystem under `/mnt/c/` |
| Default `python3` | `/usr/bin/python3`, Python 3.12.3 |
| Additional interpreter | `/usr/bin/python3.10`, Python 3.10.19 |
| `python` command | Not available in the current shell |
| Active environment | Neither `VIRTUAL_ENV` nor `CONDA_PREFIX` is set |
| Available tools | `ffmpeg`, `nvidia-smi`; actual GPU usability not verified |
| Runtime module discovery | TensorFlow, Keras, Gradio, ffmpy, librosa, mutagen, tkinter, and huggingface-hub unavailable to both inspected interpreters |
| Test module discovery | NumPy available; pytest unavailable to both inspected interpreters |

The existing installation guide targets Python 3.10. `requirements.txt` constrains
TensorFlow to `~=2.15.1` and Keras to `~=2.15.0`, among other packages. It does not
declare pytest, and its aubio requirement is commented out even though map
generation imports aubio. A working application environment has not yet been
established; Python 3.10 is the initial candidate to validate against this stack.

`tools/config/paths.py` currently leaves `dir_path`, `bs_song_path`, and
`bs_input_path` empty. Selecting an app data folder persists `dir_path` into that
source file and creates the expected directories. Existing weights inside the
nested `mnt/` tree are not automatically configured by their mere presence.

## Verification

- **Syntax:** all 77 tracked Python files parsed successfully with Python 3.10
  and Python 3.12. The Python 3.12 check emitted an invalid-escape warning in
  `countlines.py`; syntax parsing does not check dependencies or runtime behavior.
- **Current-shell test command:** `python3 -m pytest -q` could not start because
  pytest is not installed.
- **Isolated metadata tests:** all 4 passed in 1.38 seconds with Python 3.10.19,
  pytest 9.1.1, and NumPy 1.26.4 in a clean temporary environment.
- **System-package compatibility:** an earlier temporary environment inheriting
  system site packages failed during collection because Python 3.10 could not
  load the system NumPy C extension. Package discovery alone did not establish
  import compatibility; installing NumPy inside the clean environment resolved
  the test setup without changing project code.
- **App startup, model loading, song conversion, generated-map validation,
  Windows execution, and training:** not exercised.

For the successful isolated check, a temporary Python 3.10 environment outside
the repository installed its own pytest and NumPy, with no system site packages.
Tests ran from the repository root using:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$TEST_PYTHON" -B -m pytest -q -p no:cacheprovider
```

`TEST_PYTHON` denotes the temporary environment's interpreter. The environment
was removed after the run; no test dependencies were installed into the project
or system Python. This verifies the four mocked metadata tests, not real-file
audio-tag extraction or the inference pipeline. A future project environment
can use `python -m pytest -q` from the repository root once its test dependencies
are installed.

## Historical concept notes

The owner supplied `concept infernosaber.pdf` from their Downloads folder. It
contains 2019 mapping/encoding experiments, 2023 v3.2.0 format notes, and a
2024-05-06 preliminary model overview. The overview explicitly says its
`dev_v2` branch is outdated.

Its proposed separation into mapping, arrow, multiplier, intensity, obstacle,
event, and arc generators is historical design input. The proposal's "DONE"
and priority labels are not verified implementation status. The historical
InfernoSaber v2 branch and the Beat Saber v2 beatmap format are separate concepts.
The PDF has not been copied into the repository; this summary captures its
provenance and status for future agents.

## Proposed acceptance criteria for the first milestone

1. A documented Python environment can start the existing app from the repository root.
2. The selected pretrained model set loads successfully.
3. One local sample song produces a map package in a disposable output location,
   with export to the user's Beat Saber library disabled for the development run.
4. The package's metadata, beatmap files, audio, and supporting assets load in a
   viewer or Beat Saber. The owner assesses timing and playability.
5. Relevant automated checks pass; Windows compatibility is checked where an
   execution environment is available, with remaining checks explicitly recorded.

Before the generation check, choose the local sample song, intended difficulty,
and canonical data/model folder. Agree on how note density should count
simultaneous notes and bombs before using BPS as a quality metric.
