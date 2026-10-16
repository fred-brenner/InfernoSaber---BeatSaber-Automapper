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

## Human-map benchmark, 2026-10-14

Historical comparison: this set included two packages from `favorites_bs_input`.
The owner excluded that source on 2026-10-16 and designated the curated
`evaluate_beats/bs_map_input` folder as the comparison source. Use the curated-set baseline
below for future comparisons; the earlier results remain a record of that run.

The restored command was exercised against three local human-authored packages,
using their Standard / ExpertPlus beatmaps. References use unique colored-note
onsets, constant BPM, and zero song offset. Source paths, audio/map/info hashes,
and full timestamps are retained in the JSON reports.

| Song | Human mapper | Package | Audio duration (s) | Reference onsets |
| --- | --- | --- | ---: | ---: |
| Lost In The Echo | Heisenberg | `evaluate_beats/bs_map_input/4ac1 (Lost In The Echo - heisenbergirl)` | 205.726 | 733 |
| Avalanche | Zorowo | `favorites_bs_input/1460 (Avalanche - Zorowo)` | 120.764 | 840 |
| Stamp On The Ground - Nightcore | Nuketime | `favorites_bs_input/14e7 (Stamp On The Ground - Nightcore - Nuketime)` | 184.137 | 956 |

Package paths above are relative to
`/mnt/c/Users/frede/Desktop/BS_Automapper/Data/training/`. The original evaluation
set contained only Lost In The Echo; the other two packages came from local
favorites/training assets. Their relationship to the pretrained model's actual
training set is unknown. This is a local execution/comparison baseline, not a
held-out evaluation set.

The model was the existing `fav_15` beat checkpoint, SHA-256
`43803cea040631b99db72f7f1a123ca002e09b0e31c7dbf38f95e7c94741e448`.
Both profiles used seed 3, 20 ms tolerance, and the current configured timing
settings. No model weights, generation semantics, or persistent settings changed.

All nine CPU song/BPS combinations succeeded; a separate GPU check also completed
all nine combinations. Repeating CPU BPS 5 produced identical proposal timestamps
for all three songs. Routine validation now defaults to CPU, as described in
[development guidance](development.md#human-map-beat-benchmark).
Results below pool counts across the three songs. Measured density counts
processed proposals over the full 510.628 seconds of audio; reference density is
4.953 unique colored-note onsets/second. Timing MAE covers matched pairs only.

| Runtime | BPS setting | Proposals | Matches | Precision | Recall | F1 | Timing MAE (ms) | Measured proposal BPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU / TensorFlow 2.15.1 | 3 | 2,489 | 567 | 0.228 | 0.224 | 0.226 | 10.055 | 4.874 |
| CPU / TensorFlow 2.15.1 | 5 | 2,100 | 454 | 0.216 | 0.180 | 0.196 | 9.947 | 4.113 |
| CPU / TensorFlow 2.15.1 | 7 | 3,033 | 665 | 0.219 | 0.263 | 0.239 | 10.022 | 5.940 |
| GPU / TensorFlow 2.21.0 + legacy Keras | 3 | 2,490 | 568 | 0.228 | 0.225 | 0.226 | 10.040 | 4.876 |
| GPU / TensorFlow 2.21.0 + legacy Keras | 5 | 2,101 | 455 | 0.217 | 0.180 | 0.197 | 9.968 | 4.115 |
| GPU / TensorFlow 2.21.0 + legacy Keras | 7 | 3,033 | 665 | 0.219 | 0.263 | 0.239 | 10.022 | 5.940 |

One proposal/match differs between the CPU and GPU BPS-5 runs, on Avalanche.
Compare subsequent changes within the same runtime profile before attributing
small cross-profile differences to model quality. The BPS setting is an
approximate difficulty control; these results do not establish monotonic density
or an optimal setting for a human difficulty label.

Automated validation: `python -m pytest -q` passed **77 tests in each profile**,
including reference loading, scoring, failure reporting, temporary audio/config
isolation, song metadata, and legacy TCN loading. CLI help was checked in module
and direct-script forms. Source audio/info/beatmap hashes and model hashes matched
after the runtime runs, and recorded benchmark source hashes matched the files
that were validated. App configuration files were not written.

Local evidence, kept outside source control:

- CPU: `/home/frede/.local/share/infernosaber-baseline/beat-benchmark-2026-10-14-cpu.json`
  and its sibling `.log` file.
- GPU: `/home/frede/.local/share/infernosaber-baseline/gpu/beat-benchmark-2026-10-14-verified.json`
  and `gpu/logs/beat-benchmark-2026-10-14-verified.log` under the baseline root.

The reports retain the exact invocation and input paths for reruns; use a new
output filename. See [benchmark usage](development.md#human-map-beat-benchmark).
Scores describe processed timing proposals and agreement with these references.
Raw model output, final beatmap timing, musical quality, held-out generalization,
tempo-change/offset alignment, and native Windows execution remain unverified.

## Curated human-map benchmark, 2026-10-16

The active comparison source is exclusively the seven packages currently in
`Data/training/evaluate_beats/bs_map_input/`, beside the repository. The owner
confirmed this seven-package set after removing three of the initially added
packages. No `favorites_bs_input` content is used.

Selected Standard / ExpertPlus maps: Bad Hatter, Better When You're Gone, Insane,
Lost In The Echo, Shiawase No Monosashi, Shut Up and Dance, and Sweet Nothing.
All seven preprocess successfully; the final preprocessing report has zero skips.
An initially overstrict wall-position check was corrected to preserve signed
obstacle lanes and wide walls, consistent with the legacy dictionary format.
The source maps were not edited.

The evaluation-root command rebuilt 14 flat map/Info copies, seven audio copies,
21 legacy dictionary arrays, and the two aligned difficulty/name index arrays.
Prior derived contents and the preliminary diagnostic report were archived within
`evaluate_beats/archive/`. The 57 files in the current raw input and existing
`temp` package retained their exact hashes. Runtime configuration files and
pretrained model weights were not changed.

CPU validation used the retained TensorFlow 2.15.1 environment, the same `fav_15`
checkpoint, seed 3, 20 ms tolerance, and BPS settings 3, 5, and 7. All 21
song/setting combinations succeeded. The full automated suite passed **111 tests**;
prepare-only operation was also checked without importing TensorFlow. GPU and
native Windows execution were not evaluated for this change.

Current evidence under `Data/training/evaluate_beats/`:

- `preprocessing_report.json`: accepted source paths/hashes, preparation settings,
  skip list, and backup location.
- `results/beat-benchmark-cpu-2026-10-16.json`: complete CPU benchmark results with
  the preprocessing report embedded.
- `results/beat-benchmark-cpu-2026-10-16.log`: runtime log.

These results supersede the earlier mixed-source comparison for future work.
They still measure processed timing proposals; training overlap and final-map
quality remain unverified. See [the repeatable command](development.md#human-map-beat-benchmark).

## BPM estimation correction, 2026-10-16

The seven curated songs exposed a systematic 1.4–1.9% tempo overestimate in
`map_creation/find_bpm.py`. Directly feeding a perfectly periodic 128 BPM onset
sequence into the pinned aubio C beat tracker returns **129.740418 BPM**. This
isolates a bias inside the tracker independently of decoding and the Python
median-of-intervals wrapper. The comb filter's shifted harmonic-lag indexing is
consistent with the bias; see the [pinned tracker source](https://github.com/aubio/aubio/blob/ad5cf975aed08cc4562dd008cf9f83b12b82ffb8/src/tempo/beattracking.c#L149-L164)
and [zero-based autocorrelation implementation](https://github.com/aubio/aubio/blob/ad5cf975aed08cc4562dd008cf9f83b12b82ffb8/src/mathutils.c).
Decoded durations from aubio and independent audio readers agree within
milliseconds, ruling out a 1.4–1.9% audio-clock discrepancy.

### Detector choice and behavior

Use the existing librosa 0.11.0 onset extractor and SciPy 1.13.1 for whole-song
periodicity. Librosa's direct tempo estimate selects a discrete autocorrelation
lag, which is too coarse for this task's 0.1 BPM acceptance target; see its
[tempo source](https://librosa.org/doc/0.11.0/_modules/librosa/feature/rhythm.html).
[Essentia RhythmExtractor2013](https://essentia.upf.edu/reference/std_RhythmExtractor2013.html)
was considered, but its [Python bindings lack native Windows support](https://essentia.upf.edu/installing.html#building-essentia-on-windows).
The selected method adds no packages, model downloads, or changes to the CPU lock.

The implementation retains `aubio.source`, the existing default/fast/super-fast
sampling presets and manual overrides, and `[bpm, decoded_duration_seconds]`.
It extracts a librosa onset-strength envelope with `n_fft=win_s`,
`hop_length=hop_s`, and `n_mels=min(64, win_s // 4)`. A whole-song
[SciPy periodogram](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.signal.periodogram.html)
uses a Hann window, constant detrending, spectrum scaling, and FFT length
`16 * next_power_of_two(onset_length)`. Candidate pulse rates span 40–320 BPM
at 0.005 BPM spacing; four harmonic powers are summed with weights `1/harmonic²`.
FFT padding samples the spectral peak more finely; it does not add observations.

The strongest fractional pulse rate is returned directly. It can represent a
subdivision of the musical beat. Reference BPM values and map metadata never
enter detection; no integer rounding or fixed correction factor is applied.
Silence, audio shorter than the FFT window, and fewer than four detected onsets
return zero BPM with decoded duration. Decoder failures propagate, and the
decoder closes after successful reads and read failures. Dictionary and argparse
namespace overrides both work; the CLI prints scalar BPM and the filename.

### Seven-song CPU verification

The owner supplied Tunebat labels and selected **0.1 BPM maximum error** after
normalizing integer multiples or reciprocals. These labels are comparison
references; their rounded values do not establish exact physical tempo.

| Song | Tunebat reference | Previous BPM | New raw BPM | New BPM / 2 |
| --- | ---: | ---: | ---: | ---: |
| Bad Hatter | 122 | 124.004273 | 244.000 | 122.0000 |
| Better When You're Gone | 128 | 129.835855 | 255.940 | 127.9700 |
| Insane | 105 | 106.457298 | 209.995 | 104.9975 |
| Lost In The Echo | 120 | 122.254876 | 240.000 | 120.0000 |
| Shiawase No Monosashi | 128 | 130.178110 | 255.990 | 127.9950 |
| Shut Up and Dance | 128 | 130.199662 | 256.005 | 128.0025 |
| Sweet Nothing | 128 | 130.203974 | 256.000 | 128.0000 |

All seven pass in all three sampling modes: **21/21 successful comparisons,
zero skips/failures**. Maximum normalized error is **0.030 BPM**, on Better When
You're Gone in default mode, corresponding to approximately 25 ms of accumulated
tempo-only drift over that recording relative to the supplied 128 BPM label.
Maximum error is 0.0125 BPM in each faster mode. Default-mode decoded durations
match the previous implementation exactly. Faster modes retain the decoder's
existing few-millisecond resampling differences. Insane's Info BPM is 210,
equivalent to twice its supplied 105 BPM reference; all other Info values match
the supplied labels.

The comparison, previous detector snapshot, executable comparison harness,
runtime log, full test log, and generation smoke report/log/harness are under
`../Data/training/evaluate_beats/results/bpm-comparison-cpu-2026-10-16T194526Z/`.
The measured report is `comparison-2026-10-16T194726Z.json`. It records input
paths/hashes, map and supplied BPM values, old/new estimates, integer factors,
errors, durations, runtimes, detector/source hashes, runtime versions, and
per-input failure handling. It evaluates BPM only. The earlier human-note timing
benchmark report remains the baseline for processed proposals.

Rerun the comparison from the repository root; each invocation creates a new
timestamped report alongside the retained pre-change detector:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python ../Data/training/evaluate_beats/results/bpm-comparison-cpu-2026-10-16T194526Z/compare_bpm.py
```

### Automated and generation checks

The regression was verified before implementation: generated 90 BPM audio with
missing beats measured 89.174863 BPM and failed the 0.1 BPM assertion. After the
change, **42 focused tests passed**, including 22 cases across 60–240 BPM with
fractional tempos, missing beats, and subdivisions, plus offsets, presets,
manual overrides, duration, silent/sparse inputs, decoder errors, cleanup, and
CLI output. The full retained CPU suite passed **153 tests** with one existing
Keras HDF5-format warning.

A disposable CPU run of `map_creation.gen_beats.main(..., debug_beats=False)`
used a copied Better When You're Gone recording, BPS setting 5, and the existing
`fav_15` model set. It completed generation in 9.519 seconds with raw BPM 255.94,
producing v3 Expert (313 colored notes) and ExpertPlus (378 colored notes)
beatmaps. JSON references, note bounds/times, full FFmpeg audio decode, and ZIP
CRC passed. Beat Saber export was disabled. Source-package, model, and on-disk
configuration hashes remained unchanged.

Runtime: Python 3.10.19, TensorFlow 2.15.1, NumPy 1.26.4, librosa 0.11.0,
SciPy 1.13.1, and pinned aubio 0.5.0a0; CPU devices only. The beat model, final
alignment/grid snapping code, and model weights were preserved. Native Windows,
viewer/in-game behavior, variable-tempo tracking, and phase-alignment quality
were not evaluated by this change.
