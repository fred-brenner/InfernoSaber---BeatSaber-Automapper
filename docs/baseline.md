# InfernoSaber revival baseline

Recorded on 2026-10-14, before the initial documentation changes. This is a dated
snapshot, not a claim about the latest upstream state or a completed runtime setup.
The [runtime check below](#runtime-check-2026-10-14) records the subsequent
environment setup, app checks, and real-song generation.

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
4. For the current beatmap syntax, successful package generation establishes the
   working baseline. Viewer or in-game validation is reserved for explicit
   requests or changes to beatmap syntax/compatibility, per the owner's clarification.
5. Relevant automated checks pass; Windows compatibility is checked where an
   execution environment is available, with remaining checks explicitly recorded.

Before the generation check, choose the local sample song, intended difficulty,
and canonical data/model folder. Agree on how note density should count
simultaneous notes and bombs before using BPS as a quality metric.

## Runtime check: 2026-10-14

The runtime work started at commit `91a03f9e04185df05a391b43e071cfa9332a7644`.
The original snapshot above remains historical. The owner authorized isolated
Python 3.10 setup, `fav_15` at BPS difficulty 5, and generation with Beat Saber
export disabled; the owner then supplied the sample song below.

### Working environment

- Python 3.10.19 at `/home/frede/.venvs/infernosaber`, with system site packages disabled.
- TensorFlow 2.15.1, Keras 2.15.0, keras-tcn 3.5.4, NumPy 1.26.4,
  scikit-learn 1.3.2, Gradio 5.29.1, and pinned upstream aubio 0.5.0a0.
- CPU inference with two TensorFlow threads per pool and one generation worker.
- Dedicated data root: `/home/frede/.local/share/infernosaber-baseline/Data/`.
- Existing configuration files retain the framework and CRLF line endings;
  `bs_song_path` remains empty, BPS 5 maps to `max_speed = 20`, and other difficulty
  slots are disabled.

The complete local `fav_15` set was copied from the nested
`mnt/c/Users/frede/Desktop/bs_test/Data/model/fav_15` folder in the repository.
The copy contains 9 files totaling 1,223,812,581 bytes. The other two local
`fav_15` folders were incomplete. No model download was needed.

See [development.md](development.md) for activation and setup details, and
[requirements-baseline.txt](../requirements-baseline.txt) for the resolved package
versions. The full lock was also reinstalled in a second clean temporary
environment: dependency checks, runtime imports, and all four metadata tests
passed (0.87 seconds for the tests). That environment was removed afterward.

### Setup findings

- The host's extra NVIDIA pip index repeatedly failed DNS resolution. Installation
  commands used PyPI alone without changing global pip configuration.
- Aubio 0.4.9 failed to compile against the host's FFmpeg headers with
  `AVStream has no member named codec`. Upstream commit
  `ad5cf975aed08cc4562dd008cf9f83b12b82ffb8` built successfully with the environment's
  NumPy and `--no-build-isolation`.
- Missing Tkinter/Tcl/Tk support was supplied through locally extracted Ubuntu
  packages and environment-local paths. No system packages were installed.
- TensorFlow emitted CUDA initialization messages; the selected CPU execution
  path still passed all reported checks. GPU execution was not evaluated.

### App and model checks

| Check | Result |
| --- | --- |
| `python -m pip check` | No broken requirements |
| Clean reinstall from the lock | Passed; runtime imports and 4 metadata tests also passed |
| Existing metadata tests | 4 passed in 0.72 seconds |
| App `/`, `/config`, `/gradio_api/info` | HTTP 200 for all three |
| Live app configuration | `fav_15`, BPS 5, one worker, other slots zero, Beat Saber folder empty |
| Gradio summary API | Successfully returned configured difficulty and disabled-library-export status |
| Five model files | Loaded with compilation enabled; all produced finite forward-pass outputs |
| Three stored pickle files | Loaded; both one-hot encoders passed transform/inverse-transform checks |

| Model | Parameters | Input shape(s), excluding batch | Output shape, excluding batch |
| --- | ---: | --- | --- |
| Beat predictor | 531,777 | `(24,264)`, `(24,1)`, `(24,1)` | `(1,)` |
| Audio encoder | 68,576 | `(24,20,1)` | `(16,)` |
| Note mapper | 62,848,896 | `(16,16,1)`, `(16,1)`, `(16,3380)` | `(16,3380)` |
| Lighting generator | 38,264,640 | `(16,16,1)`, `(16,1)`, `(16,1904)` | `(16,1904)` |
| Training autoencoder | 159,873 | `(24,20,1)` | `(24,20,1)` |

The notes dictionary contains 3,428 entries; the note and event encoders contain
3,380 and 1,904 classes respectively. Loading the autoencoder and making a
forward pass verifies those operations only, not the training workflow.

### Real-song generation

- Original input: `C:\Users\frede\Desktop\bs_test\Data\prediction\songs_predict\BWB - No Sleep.egg`.
- Source SHA-256: `b213b729a4cffad84421e6f5b6a8c7ce3dfae6db070e2ec3771e2edbdf10a0f5`.
- Input format: Ogg/Vorbis, stereo, 48 kHz; duration 134.506667 seconds.
- Source audio was copied into the isolated input folder and remained unchanged.
- Actual app call: `Client("http://127.0.0.1:7860").predict(1, "fav_15", 5, 0, 0, 0, 0, api_name="/run_process")`.
- The app API completed in 22.129 seconds, including preprocessing and model
  generation. This is a single-run measurement, not a performance benchmark.
- Beat Saber export remained disabled; all generated assets stayed in the
  isolated data root.

Output: `Data/prediction/new_map/12345_BWB - No Sleep.zip`, under the data root
above, alongside the unpacked map package.

| Generated content | Result |
| --- | --- |
| Beatmap format | v3.2.0; level metadata v2.0.0 |
| Beatmap file / label | `Easy.dat` / `Easy`, reflecting the app's first-slot labeling behavior |
| Numeric generation setting | BPS difficulty 5 |
| Colored notes | 517 |
| Bombs | 0 |
| Obstacles | 45 |
| Arcs | 104 |
| Lighting events | 480 |
| Detected BPM | 131.0905712462031 |

Package checks passed: JSON parsing; referenced beatmap/audio/cover files;
colored-note position, color, direction, and finite nonnegative times; ZIP CRC
and member matching; and complete FFmpeg audio decoding. These are basic
structural checks, not a comprehensive Beat Saber format validator.

The audio tags identify the song as `Vor dem Bass` by `bitcoinworldbeats`; the
app correctly preferred those tags over the filename for level metadata.

Reports and logs are stored under
`/home/frede/.local/share/infernosaber-baseline/`, including `model-check.json`,
`app-check.json`, `generation-check.json`, and `logs/app.log`.
The clean-reinstall result is recorded in `rebuild-check.json` and `logs/rebuild.log`.

### Remaining verification

- Exercise native Windows/Pinokio installation and folder-dialog behavior.
- Exercise the full training workflow if it becomes the next priority.

## GPU execution: 2026-10-14

The owner clarified that successful generation is sufficient acceptance for the
current beatmap syntax. Viewer or in-game checks are reserved for explicit
requests or format compatibility changes. The existing CPU generation therefore
establishes a working baseline.

GPU execution was then verified on the RTX 5080 (16 GB, compute capability 12.0)
under WSL2 with Windows driver 617.14. Direct WSL CUDA initialization succeeded;
the original TensorFlow 2.15 environment still enumerated no GPUs and lacked its
required cuDNN 8 library.

A separate environment at `/home/frede/.venvs/infernosaber-gpu` uses TensorFlow
2.21.0, tf-keras 2.21.0, NumPy 1.26.4, h5py 3.14.0, CUDA runtime 12.9.79,
and cuDNN 9.27.0.42. Its 134 packages are pinned in
[requirements-gpu.txt](../requirements-gpu.txt). See
[gpu-execution.md](gpu-execution.md) for activation and details.

Nine modules' standalone Keras imports were changed to `tensorflow.keras` so
the GPU profile can select the legacy API consistently. A new TCN H5 roundtrip
test failed before this correction and passed afterward. The CPU TensorFlow 2.15
environment still passes the complete five-test suite; the GPU environment also
passes all five tests. The model architectures and source weight files are unchanged.

| GPU check | Result |
| --- | --- |
| Driver and device | RTX 5080 visible; WSL CUDA initialization successful |
| Strict device placement | Matrix, convolution, and gradients executed on `/GPU:0` |
| Original model set | All five H5 models loaded and produced finite GPU outputs |
| Stored encoders | Both one-hot conversion roundtrips passed |
| Autoencoder training step | Finite optimizer step with GPU-resident trainable variables |
| Dependency consistency | `pip check` passed |
| App startup | HTTP/configuration/API checks passed |
| Real-song generation | Successful package creation at `fav_15` / BPS 5 |

The GPU app generated the same source song in 23.338 seconds and produced
`/home/frede/.local/share/infernosaber-baseline/gpu/Data/prediction/new_map/12345_BWB - No Sleep.zip`.
The package contains 509 colored notes, 43 obstacles, 104 arcs, and 480 lighting
events. Monitoring observed GPU compute worker PID 50633, with peak sampled
device utilization of 17% and total device memory use of 7,149 MiB. The memory
sample is device-wide, not a per-process allocation measurement.

The source song remained unchanged and Beat Saber export stayed disabled.
The GPU data/output root is separate from the retained CPU baseline; the active
`paths.py` now selects that GPU workspace. The GPU app now uses localhost port
7860 after the temporary CPU and GPU test instances were stopped.

The first strict GPU check took 22.844 seconds and exercised TensorFlow's PTX
compilation route for this hardware. The CPU and GPU timings above are individual
runs and do not establish a speedup. Full training-data orchestration remains
untested; the autoencoder optimizer step is verified.

Evidence is under `/home/frede/.local/share/infernosaber-baseline/gpu/`, including
`device-check.json`, `model-check.json`, `generation-call.json`,
`generation-check.json`, and logs. The earlier CPU reports remain intact.
