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

## CPU beat benchmark with corrected BPM, 2026-10-16

Reran all seven curated songs at BPS settings 3, 5, and 7, Standard/ExpertPlus,
seed 3, and the original 20 ms matching tolerance. All **21 runs succeeded**,
with zero skipped inputs. Python, TensorFlow, devices, settings, input hashes,
and the `fav_15` checkpoint match the earlier CPU benchmark. Only
`map_creation/find_bpm.py` differs among the earlier pipeline source hashes.
Captured timings before BPM processing reproduce the earlier run exactly in
all 21 cases. The rerun preserved pipeline sources, model weights, raw packages,
prepared evaluation caches, configuration, and previous reports.

| BPS setting | Previous F1 | Corrected-BPM F1 | Corrected BPM, bypass final alignment |
| ---: | ---: | ---: | ---: |
| 3 | 17.85% | 56.72% | 34.09% |
| 5 | 17.27% | 63.86% | 39.93% |
| 7 | 18.89% | 61.43% | 38.08% |

Both the detector change and final alignment improve 20 ms F1 in **21/21**
comparisons. Across settings, final alignment raises pooled F1 from 37.42% to
60.73%. It also improves pooled F1 at every evaluated tolerance (10, 20, 35, 50,
and 100 ms). Final alignment is no longer the main observed quality loss at
20 ms. Bypass variants are offline rescoring of captured arrays; the benchmark
itself runs the existing generator and alignment implementation.

Per-song results at BPS setting 5:

| Song | Previous F1 | Corrected-BPM F1 | Bypass final alignment F1 |
| --- | ---: | ---: | ---: |
| Bad Hatter | 19.91% | 60.96% | 45.67% |
| Better When You're Gone | 16.83% | 48.31% | 40.05% |
| Insane | 14.59% | 53.60% | 38.54% |
| Lost In The Echo | 16.50% | 68.56% | 44.32% |
| Shiawase No Monosashi | 15.80% | 75.68% | 60.29% |
| Shut Up and Dance | 15.56% | 67.22% | 15.85% |
| Sweet Nothing | 19.69% | 64.47% | 36.38% |

The improvement primarily repairs timing precision: pooled previous/new F1 is
9.22%/58.59% at 10 ms, 18.06%/60.73% at 20 ms, and 66.32%/66.67% at 100 ms.
Broad-window onset agreement changes little. Matched-pair timing error at 20 ms
falls from approximately 9.95 ms to 2.51 ms, with different matched populations.

The largest intermediate F1 loss is now onset rematching
(`sanity_check_timing2`) in 15 runs and density filling in six. Pooled onset
rematching reduces F1 from 63.11% to 42.39%, while final snapping subsequently
recovers much of the loss. These local measurements identify candidates for
complete-pipeline ablations; they do not establish that removing either stage
would improve final output. No learned-model change is justified by this
comparison alone.

Two remaining considerations:

- Integer-equivalent tempo values change the current aligner's grid resolution.
  A 2x pulse halves the interval used for whole-beat rounding and changes
  collisions. On saved **old** timings immediately before final alignment,
  using the corrected BPM at the original pulse level yields BPS 5 F1 52.92%;
  using its 2x pulse yields 63.57%, close to the full new run's 63.86%.
  The gains therefore include both fractional-tempo accuracy and finer grid
  resolution. These diagnostic variants keep the old upstream timings fixed.
- Better When You're Gone remains sensitive to its small residual BPM
  difference. Replacing 255.94 with the reference-equivalent 256 in the final
  step alone increases its BPS 5 F1 from 48.31% to 62.83%. Across all seven songs,
  the same-pulse reference variant only raises BPS 5 F1 from 63.86% to 65.01%.
  This measures agreement with the human map grid; it does not independently
  establish the recording's exact tempo or justify rounding all detected BPMs.

The unchanged collision handling still produces outliers: maximum final-stage
movement is 4.185 seconds (Insane, BPS 3), and Better When You're Gone has
nonmonotonic final arrays at BPS 5 and 7. There are six duplicate predictions
across the 21 final arrays. These remain alignment correctness concerns despite
its positive aggregate contribution. The next quality investigation should
measure onset rematching and filling through the full downstream pipeline while
retaining final alignment as the current baseline.

Evidence is saved outside the repository under
`../Data/training/evaluate_beats/results/beat-benchmark-new-bpm-cpu-2026-10-16T195723Z/`:
`benchmark.json` contains the new standard CPU results; `alignment-analysis.json`
and `.csv` contain stages and offline variants; `comparison-summary.json` and
`per-song-bps5.csv` contain comparisons; `tempo-level-diagnostic.json` isolates
the final-step pulse-level effect on old timings. Executable harnesses and logs
are retained alongside the reports. Inference took 86.52 seconds in total.

This evaluates processed timing proposals against the same human maps. Training
overlap and general musical quality remain unverified. No model or alignment
code was changed, no Beat Saber export occurred, and no GPU runtime was used.

## New intensity, timing, and note framework: CPU acceptance (2026-10-16)

Implemented the independent `training.beat_models` framework and its version-1
contracts. The dummy pipeline passed before neural builders were introduced.
Preparation, validation, deterministic dummy inference, untrained Keras smoke,
three stage-specific training dry runs, and explicit-checkpoint evaluation now
complete on CPU. See [the training guide](beat-model-training.md) for commands,
units, unsupported data cases, and checkpoint compatibility.

The retained CPU environment used Python 3.10.19, TensorFlow/Keras 2.15.1/2.15.0,
NumPy 1.26.4, and disabled GPU visibility. Full `python -m pytest -q` acceptance:
**213 passed**, including 60 new framework tests. Warnings are upstream Python
`randrange` deprecations and the existing legacy HDF5 save warning. No installed
packages, existing model weights, legacy generator, alignment, or export code
were changed during this implementation.

| Model | Parameters | Real-data examples checked | Save/reload maximum prediction difference |
| --- | ---: | ---: | ---: |
| Intensity | 60,481 | 6 whole-song examples | 0 |
| Timing | 180,483 | 12 first/last context windows | 0 |
| Notes | 173,450 | 48 reference note events | 0 |

Finite outputs, losses, connected finite gradients, validity masks, and exact
weight preservation passed. Separate guards rejected any attempted `fit`, Adam
construction, or optimizer application during executed Keras diagnostics. All
three future-training entry points passed `--dry-run`; the training branches
were not executed. Saved model artifacts are explicitly **untrained**.

The seven curated raw packages were selected at Standard/ExpertPlus. No
favorites inputs were used. Six maps passed preparation and remained in the
evaluation-only test split, with synthetic-training normalization used solely
for execution checks. The real evaluation data did not fit normalization.

| Song | Decoded seconds | Original colored notes | Retained colored notes | Unique retained onsets |
| --- | ---: | ---: | ---: | ---: |
| Bad Hatter | 176.027188 | 924 | 908 | 744 |
| Insane | 161.476375 | 718 | 709 | 616 |
| Lost In The Echo | 205.725500 | 896 | 894 | 733 |
| Shiawase No Monosashi | 171.064938 | 778 | 766 | 640 |
| Shut Up and Dance | 199.899813 | 894 | 887 | 627 |
| Sweet Nothing | 208.823563 | 1163 | 1158 | 1022 |

**Skipped:** `8215 (Better When You're Gone - Halcyon12)/ExpertPlus.dat`.
Source beats `219.83332824707031` and `219.83334350585938` become seconds
`103.04687261581421` and `103.04687976837158`. They differ by about 7.15
microseconds but both map to frame 10305. Exact-source grouping followed by
nearest-frame labeling therefore rejects this map rather than silently merging
its distinct onsets. This is a source/representation compatibility limit, not a
failure of audio decoding. The original package was left unchanged.

Explicit untrained checkpoints also completed teacher-forced stage evaluation
and composed autoregressive inference on the held-out dummy split and all six
accepted real songs. Note evaluation/decoding was capped at eight events per
song; timing covered each song. Reports identify the bounded results as partial.
They include intensity share/bin-count errors, timing precision/recall/F1 and
note-rate error, per-hand class/presence/position/direction accuracy, and composed
prediction arrays. These numbers verify diagnostic plumbing; random initialized
outputs and synthetic normalization support **no model-quality comparison** with
the previous trained benchmark.

Evidence resides outside the source checkout at
`../Data/training/evaluate_beats/results/beat-model-framework-cpu-2026-10-16T212414Z/`.
`acceptance-summary.json` indexes the final outcome; `commands.json` and logs
record executed diagnostics, `pytest-cpu.log` records the full suite,
`curated-dataset/manifest.json` records accepted/skipped sources and splits,
`curated-smoke/report.json` records no-update/serialization checks, and
`curated-evaluation/report.json` records stage and composed diagnostics.
`verify_commands.py` and `guarded_cli.py` preserve the executable checks. An
initial standalone harness import-path error was corrected before Keras checks;
its failed-attempt log is retained separately. Source hashes before and after
match for all **52 files** in the seven packages. A start-of-task tracked-file
hash comparison also confirms that pre-existing runtime changes were preserved.

No new model was trained and no optimizer step occurred. Actual CPU/GPU
training, native Windows execution, mapping quality, and full map generation
remain unverified. The existing pretrained generation pipeline remains separate;
this milestone ends at intermediate predictions.

## Near-time source synchronization: CPU acceptance (2026-10-17)

The new framework now reuses `config.min_time_diff` (0.02 seconds) to synchronize
nearby source events silently before per-hand first-cut reduction. Each group
uses its earliest timestamp and includes notes up to that timestamp plus the
configured interval, matching the existing helper's inclusive comparison.
Grouping is anchored rather than chained through the last merged note.
A group produces one timing entry with at most one colored note per hand.
Original numeric/JSON records and exact timestamps remain unchanged, and the
new `original_event_indices` array maps all source notes to synchronized events
for the future multiplier. Intensity is counted at each synchronized event's
anchor, including when the source notes straddle an intensity-bin boundary.

Target version is now **2**; schema and audio-feature versions remain 1. The
representation records the synchronization interval. Older exact-time datasets
and untrained checkpoints must be regenerated. See [the current training
guide](beat-model-training.md) for fresh versioned output directories.

CPU verification: **215 tests passed**, including near-time merging, inclusive
boundaries, first-onset anchoring, configurable/disabled synchronization,
source-order fallbacks, preserved source records, mapping validation, and
intensity-bin consistency. All seven Standard/ExpertPlus curated packages now
prepare, validate and complete bounded dummy/Keras smoke checks, with **zero
skips**. This supersedes the previous Better When You're Gone compatibility
skip; the map is accepted without changing its source files.

| Song affected | Original distinct onsets | Synchronized onsets | Original colored notes | Retained colored notes |
| --- | ---: | ---: | ---: | ---: |
| Better When You're Gone | 359 | 357 | 483 | 483 |
| Sweet Nothing | 1022 | 992 | 1163 | 1126 |

Other five songs retain their previous onset and retained-note counts. The
configured interval intentionally includes all nearby cuts, rather than only
microsecond numerical differences. Seven-song outputs are evaluation-only, with
synthetic-training normalization used solely for execution diagnostics. Seven
whole-song intensity examples, 14 first/last timing chunks, and 56 reference
note events passed finite loss/gradient checks with exact save/reload parity and
unchanged weights. No model training or optimizer update occurred.

Evidence: `../Data/training/evaluate_beats/results/beat-model-sync-cpu-2026-10-17-73e270/`.
`acceptance-summary.json`, `commands.json`, and `pytest-cpu.log` record acceptance;
`curated-v2/manifest.json` records grouping/provenance and all seven source maps;
`keras-smoke/report.json` records bounded CPU checks. Before/after hashes match
for all 52 input-package files. The existing setting, neural model definitions,
legacy generator and alignment files were left unchanged. GPU execution and
trained-model quality were not evaluated.

## Import filename compatibility and native Vorbis logging (2026-10-17)

The new raw-package reader now accepts the old cleanup's known Standard filename
pair in either direction: `<difficulty>.dat` and `<difficulty>Standard.dat`.
Exact Info references take precedence; unique case-only matches are also
accepted. Arbitrary replacements, ambiguous case matches, external references,
and alias files explicitly claimed by another beatmap remain rejected. Declared
and resolved names are recorded in source provenance and preparation reports.
Original packages are read without renaming maps or rewriting Info files.
Malformed sibling references are reported individually and do not invalidate a
valid fallback. The legacy cleanup itself was left unchanged.

Only the complete native FFmpeg stderr line
`[vorbis @ 0x...] Could not update timestamps for discarded samples.` is filtered
while aubio constructs, reads and closes the audio source. Other bytes and
warnings/errors are forwarded unchanged, including other codecs, different
Vorbis messages, and incomplete lines; decoder exceptions propagate. A scoped
fd2 pipe filter is used because native writes bypass Python stderr redirection.
Decode scopes are serialized; stderr is restored on exceptions, and failed
output sinks cannot deadlock decoding. Descriptor redirection applies within the
process during that decode scope. Sample extraction and normalization are
unchanged. Optional BPM-change support and further import follow-ups are tracked
in [TODO.md](../TODO.md).

CPU acceptance: **242 tests passed** (27 new logging/import regressions), with the
same upstream deprecation and legacy-save warnings. The real package
`162e4 (Up & Down - cat_using_a_toaster)` now fully prepares its Standard/ExpertPlus
map through `ExpertPlusStandard.dat` → `ExpertPlus.dat`: 1,121 original colored
notes, 1,036 retained notes and 742 synchronized onsets. Its 2,643,687 decoded
samples (165.2304375 seconds) are bit-identical with and without the filter; the
matching native warning falls from one occurrence to zero. All files in that
source package retain their hashes.

A read-only audit of the existing full-dataset preparation report found **111**
missing filename references, all `ExpertPlusStandard.dat`. All 111 now resolve;
89 pass the note reader and 22 remain unsupported for BPM changes. This audit
checks file resolution and note parsing; full decoding/target preparation was
verified for the real sample above. Full dataset preparation must be rerun into
a fresh output directory to include recovered maps. The existing full dataset,
checkpoints and training runs were not modified. No model training or optimizer
step was executed; new native Windows/GPU execution was not exercised.

Evidence: `../Data/training/beat_models_v2/import-compat-cpu-2026-10-17/` contains
`acceptance-summary.json`, `filename-audit.json`, `real-package-check.json`,
`pytest-cpu.log`, native raw/filtered stderr logs and reproducible decode/import
probes. See [the framework guide](beat-model-training.md) for current policies.

## Intensity epoch diagnostics (2026-10-17)

Intensity training now reports KL divergence, duration-aware uniform-note-rate
reference loss, and percentage reduction in KL relative to that reference, for
both training and validation. Reference entropy/loss are computed from nonempty
targets in each respective split, averaging songs equally; no new model passes
are required. A zero-reference-KL percentage is omitted. The metrics are added
before Keras History/progress callbacks, so console output and `report.json`
history contain the same values. Baselines and definitions are recorded in the
run configuration, and dry runs report preview-batch diagnostics. See the
[training guide](beat-model-training.md#intensity-training-display).

CPU acceptance: **246 tests passed** in TensorFlow 2.15.1; **4 targeted metric
checks also passed** on CPU using TensorFlow/tf-keras 2.21. The tests simulate
callback events without `fit` or optimizer construction and verify console
output, history, partial-bin weighting, empty-target exclusion, negative gains,
undefined percentages, NaN preservation, and unchanged model weights. The
existing completed intensity checkpoint still matches its saved manifest hash.
No training, optimizer update, dataset/preprocessing change, or model-architecture
change was performed. GPU execution of the new display was not exercised.

## Centralized new-model training configuration: CPU acceptance (2026-10-17)

The existing DNN section in `tools/config/config.py` now holds separate
`intensity_*`, `timing_*`, and `notes_*` blocks. They configure optimizer name,
learning rate, epoch and batch limits, clipping, seeds, shuffling, verbosity,
monitor/direction, early stopping, network widths/kernels/dropout/activation,
context lengths, conditioning noise, and loss weights. Shared new-model audio
and recording split fractions use `beat_models_*`, separate from legacy audio
settings. Existing configuration entries and CRLF line endings were preserved
byte-for-byte; the legacy import/format handling was not rewritten.

Builders and training/preview/evaluation paths use the configured stage values.
CLI epochs, batch size and conditioning override config only when explicitly
provided. Reports/manifests retain effective training and architecture settings.
The configured epoch count replaces the old hardcoded 100-epoch ceiling. Timing
context margins derive from the receptive field; incompatible context changes
require restarting the process and preparing compatible artifacts.

CPU verification: **271 tests passed**. New tests cover model-specific settings,
CLI priority, seeds/noise, non-default architectures and losses, grouped split
fractions, and fresh-process audio/context dimension consistency. All three
CLI training dry runs completed with optimizer constructors and `fit` forbidden.
The default representation is exactly unchanged, and seven existing target-v2
curated songs plus their previous untrained checkpoints still load. Bounded
forward/loss checks kept weights unchanged: seven intensity examples, 14 timing
windows and seven reference note events. Default parameter counts remain
60,481 / 180,483 / 173,450. No training, optimizer updates, or GPU execution ran.

Evidence is stored outside the checkout under
`../Data/training/evaluate_beats/results/beat-model-config-cpu-2026-10-17-3s3neh2s/`:
`acceptance-summary.json`, `pytest-cpu.log`, `no-training-check.json`, and the three
`dry-*/report.json` files. [The training guide](beat-model-training.md) explains
parameter locations, CLI overrides and artifact compatibility.

## Trained intensity/timing versus legacy beats: CPU comparison (2026-10-17)

The curated `evaluate_beats` benchmark now supports `--backend compare` and
`--backend intensity-timing`, with explicit `--trained-run` or separate stage
artifacts. Run `20261017-151437` supplies only its trained intensity and timing
checkpoints. The new adapter reuses the shared decoder/features and saved
training normalization, predicts whole-song intensity, supplies Info BPM
separately from requested colored NPS, and emits raw 100 Hz count argmax.
It does not load the trained notes checkpoint or call the later pipeline.
No threshold, spacing filter, rematching, snapping or forced count was added.

For the old model, an evaluation-only observer captures thresholded neural
outputs before existing spacing/filling/alignment and also retains its unchanged
processed debug result. The wrapper is restored after each call, including on
failure. The source generator, alignment, legacy preprocessing, training code,
model definitions/settings, and checkpoint files were not changed.

All seven Standard/ExpertPlus packages completed, with zero preprocessing skips
or failed inference cases. Sources are the same seven packages as the corrected
BPM baseline; no favorites are included. Primary scoring uses identical 20 ms
synchronized human references for every stage: **4,709 onsets**. Exact-reference
scores against **4,741 onsets** are retained for historical comparison. Models
are evaluated with the configured decoder, without benchmark threshold tuning.

Pooled onset F1 at an inclusive 20 ms tolerance:

| Numeric setting | Old raw thresholded | Old processed beat path | New intensity + timing |
| ---: | ---: | ---: | ---: |
| 3 | 21.88% | 56.90% | 40.30% |
| 5 | 21.87% | 64.06% | 51.41% |
| 7 | 22.03% | 61.59% | 55.93% |

The new pair beats the raw legacy output on all **21/21 song/setting cases**.
Against the processed legacy path it wins 1/7 songs at settings 3 and 5, and 3/7
at setting 7. At setting 5, Shut Up and Dance improves from 67.22% to 70.52%;
the other six songs remain below the processed baseline. This supports improved
raw proposals under each stage's existing decoding/time conventions, not an
improved replacement for the complete current beat path.

At setting 5, new precision/recall is **79.77% / 37.93%**, versus processed legacy
**59.18% / 69.82%**. The new output has 2,239 onsets (1.821/second), versus 5,556
processed old onsets (4.518/second) and 4,709 reference onsets (3.829/second).
Including predicted doubles, the new output contains 2,649 retained colored
notes, or 2.154 NPS. Its matched timing error averages 3.69 ms; old processed
matches average 2.53 ms. These means use different matched populations.

Equal numeric controls are not density-equivalent: legacy BPS is an approximate
application control, while new NPS counts colored notes and counts doubles twice.
The additional new run at each map's actual retained NPS still produces pooled
**50.18% F1**, precision 81.26%, recall 36.29%. It emits **2,376 colored notes**
against 5,773 retained reference notes. Requested density remains conditioning,
not a forced budget. The dominant observed weakness is missing events/low output
density, rather than poor placement of the emitted matches. This comparison does
not isolate whether predicted intensity, conditioning, or timing decisions cause
the omissions; a reference-intensity ablation on validation data remains a
separate diagnostic. Notes, alignment and training were left untouched.

Under the previous exact-reference convention, old processed pooled F1 remains
**56.72% / 63.86% / 61.43%**, reproducing the corrected-BPM baseline. New exact
F1 is **40.10% / 51.17% / 55.69%**. The small shift in the primary old numbers
therefore comes from synchronized references, not a legacy behavior change.

Overlap audit against the actual training dataset manifest found no matches for
any curated song's audio bytes or mono-16k decoded PCM across train, validation
or test, nor in either checkpoint's source list. Dataset preparation reserved
22 matching difficulty variants. This supports identifiable-recording exclusion
for the new run; arbitrary re-encodes without aliases and old-model training
overlap remain limitations. The seven-map result is not a gameplay-quality or
broad generalization claim.

The trained tf-keras 2.21 archives cannot deserialize directly into the retained
CPU Keras 2.15 package namespace. The adapter checks each manifest, architecture,
normalization and checksum, compares the full serialized execution graph to the
current model, then strictly restores the unchanged weights on CPU. Both trained
models loaded and produced finite probabilities through every valid tail frame.
No packages were installed or checkpoints converted.

Validation used Python 3.10.19, TensorFlow 2.15.1/Keras 2.15.0, NumPy 1.26.4,
and disabled GPU visibility. Full CPU suite: **324 passed**. There were **70**
successful stage cases: 21 old raw, 21 old processed and 28 new runs, including
seven reference-density runs. Guards rejected training/optimizer updates and
non-beat model loading/construction during the benchmark. All 52 input files
and nine checkpoint weight/metadata files, including the unused notes stage,
retain their original hashes. A separate real pair-only CLI smoke confirms no
legacy generator or new generation runner import. GPU evaluation and note
inference were not performed.

Evidence directory:
`../Data/training/evaluate_beats/results/beat-pair-comparison-cpu-2026-10-17-91fa7c/`.
`benchmark.json` contains the full comparison and adjacent `benchmark-arrays/`
contains probabilities/intensity/counts. `comparison-summary.json` and
`per-song-setting-5.csv` collect conclusions; `overlap-audit.py/.json` reproduce
the source/split audit. `guarded_benchmark.py`, `benchmark.log`,
`guarded_pair_only.py`, `pair-only-smoke.json`, and `pytest-cpu.log` retain the
execution checks. Post-inference reporting fixes for failed-run totals and
source-hash coverage passed focused tests; all measured inference cases had
succeeded, so those fixes do not change their predictions or scores.


## V2 concept framework: CPU acceptance without training (2026-10-17)

Implemented `training.pipeline_v2` following the agreed
[architecture](model-architecture.md). The connected concept uses the existing
intensity/timing pair and retained audio encoder, followed by separate positions,
arrows, multiplier, hazards and arcs models; lighting uses its own candidate
timeline. New model and preparation parameters live in `tools/config/config.py`.
Existing intensity/timing targets and checkpoints, the former joint-notes model,
legacy generation and source preprocessing remain unchanged.

The approved decisions are implemented: synchronized copies and explicit chains
for the multiplier, fixed per-song encoder normalization, simultaneous multi-label
lighting, and native human hazard positives in addition to scheduler candidates.
Hazard query provenance and scheduler coverage remain separate. Training-only
vocabularies and normalization are frozen for later curated comparisons.
Unsupported objects are preserved with masked targets and diagnostics; base-reader,
reservation and downstream preparation skips all reach the comparison report.

The framework provides immutable JSON/NPZ datasets, streamed stage batches,
explicit checkpoint compatibility, future training commands and guarded concept
composition. Independent encoder and six downstream comparisons are prepared,
require explicit execution, and were **not run**. No real-song accuracy result
is claimed. No full corpus preparation was run for these new stages.

Synthetic CLI acceptance completed under guards that reject `fit`, optimizer
construction and optimizer steps. It exercised four prepared dummy songs,
dataset validation, the complete dummy composition, all six untrained neural
models and the default named review stop. Every new network produced finite
outputs, losses and gradients; model weights remained unchanged. All six strict
`.keras` reloads reproduced predictions exactly (maximum difference **0.0**).
Additional tests exercise active audio/history/output gradient branches and
rejection of structural/synthetic fixtures by actual training entry points.

Synthetic-vocabulary parameter counts are positions 40,970; arrows 53,362;
multiplier 64,054; hazards 50,062; arcs 51,559; lighting 6,584,928. Lighting retains
the agreed legacy network topology; its multi-label class count, and template
heads in other stages, depend on the frozen real training vocabulary.

A separate generated 3.25-second waveform passed the actual retained encoder on
CPU with unchanged checkpoint/model weights. It exposed a 16-sample shortfall in
the native 14,800 Hz resampled tail. Encoder representation v2 now validates query
times against the beat framework's canonical duration and zero-pads unavailable
samples within the centered patch. Native samples, source hashes and event times
remain unchanged; both durations and the tail difference are recorded. A query
at 3.249 seconds succeeds, while the exact 3.25-second end remains excluded.

Final CPU suite: **405 passed**, with **10 opt-in comparison tests excluded**.
All 52 curated input files and nine files in trained run `20261017-151437` retain
their original hashes. Pre-existing working-tree edits and configuration values
were preserved; only a new v2 config section was inserted.

Runtime: Python 3.10.19, TensorFlow 2.15.1/Keras 2.15.0, NumPy 1.26.4, GPU
visibility disabled. Training, optimizer updates, GPU execution, independent
model comparisons and gameplay quality remain unverified. The default concept
stops at `bpm_alignment`; movement, swing, hazard/arc clearance and export also
have explicit review stops. Diagnostic traversal records bypasses and never
marks the result playable or export-ready.

Evidence is outside the checkout in
`../Data/training/pipeline_v2/runs/framework-cpu-2026-10-17-ed24f6/`.
`final/acceptance-status.json`, `final/keras-smoke/report.json`,
`final/review-stop/concept.json`, and `retained-encoder-synthetic.json` contain the
final synthetic checks. `final/pytest-cpu.log` and `final/pytest-cpu.xml` contain
the CPU suite; `acceptance-summary.json` and `files-before.json`/`files-after.json`
record the final result and source preservation. Exact preparation, stage-training
and later comparison commands are in [the v2 handoff](pipeline-v2.md).


## V2 training throughput and development launcher (2026-10-17)

The first user-started positions run exposed two training bottlenecks. A batch
needed 903,408 bytes of positions arrays but the full-song loader also decoded
104,104,462 unrelated bytes, including lighting. The process repeatedly waited
on `/mnt/c` filesystem access. The eager batch loop additionally dispatched
model/loss/gradient operations and copied metrics to Python for every batch.

The new training reader selects only the active stage, performs one combined
integrity/provenance/validation/count scan, and uses a configurable 1,024 MiB
RAM budget. Larger stages stream; a single producer prefetches two batches.
Startup reports checked songs and cache usage. Reusable TensorFlow functions
execute training and validation, accumulate per-head float64 sums/counts on the
model's device, and synchronize at progress/epoch boundaries. The existing
architecture, masks, shuffle, partial batches, objective, Adam/clipnorm settings,
early stopping and best-checkpoint behavior are preserved. Existing datasets
remain compatible and need no repeated preparation.

The obsolete `main_training.py` now presents a small selection menu for the six
new downstream stages and runs selected stages sequentially in separate child
processes. It uses the activated Python and a fresh timestamped output root.
`--models` skips the menu; `--dry-run` forwards validation-only requests.
Import/help are inert, failure stops the queue and Ctrl+C returns 130. The old
five-letter legacy selection has been retired. See [the launcher and performance
settings](pipeline-v2.md#small-development-launcher) for exact commands.

Read-only CPU measurements on the existing 875-map dataset:

| Check | Measured result |
| --- | --- |
| Unrelated decoded arrays per positions batch | 104,104,462 bytes before; **0** after |
| First selected batch, including source checks/metadata | median **0.304 s → 0.123 s** (2.46×) |
| One-time complete positions input scan | **126.54 s**; **325,616,016 bytes** cached (310.53 MiB) |
| Warm train + validation input passes | **555,127 examples / 4,338 batches** in **0.656 s** and **0.550 s** |
| Archive/JSON opens during those warm passes | **0**, guarded against accidental access |
| CPU forward + gradients, batch 128 | median **81.95 ms → 9.65 ms** (8.49×) |
| Eager/compiled numeric comparison | identical loss; maximum gradient difference **1.49e-8**; unchanged weights |

These are data-input and inference/differentiation measurements, not an
end-to-end GPU training speedup. Model calls in the gradient timing used
`training=False`; no optimizer updates ran. The real corpus input test performed
no model inference or accuracy comparison. The source dataset manifest and all
prepared artifact bytes were left unchanged.

Final retained CPU suite: **494 passed**, **10 opt-in comparisons excluded**.
All six models have eager/compiled output/loss/gradient and masked-target parity,
dynamic final-batch graph reuse, weighted metric/precision and unchanged-weight
checks. One real Adam graph was traced without execution: optimizer iterations
remained **0**, weights were unchanged, and finite checks precede weight updates.
The same 20 execution tests also passed using the TensorFlow **2.21.0** / legacy
Keras training environment with `CUDA_VISIBLE_DEVICES=-1`; no GPU computation
ran. Primary CPU checks used Python 3.10.19, TensorFlow 2.15.1 and NumPy 1.26.4.
All 52 curated input files and nine files in trained run `20261017-151437` retained
their hashes. The ongoing user-owned positions process was neither stopped nor
restarted. Actual optimizer execution and GPU throughput of the optimized runner
remain for the user's next explicitly started run.

Evidence:
`/home/frede/.local/share/infernosaber-baseline/diagnostics/v2-training-speed-20261017-221651/`.
`reader-benchmark.json` and `execution-benchmark.json` contain individual timings;
`benchmark_reader.py` and `benchmark_execution.py` reproduce them without training.
`pytest-cpu.log`/`.xml` and `pytest-training-env-cpu.log` retain test results;
`acceptance-summary.json` and `files-before.json`/`files-after.json` record the
scope and source preservation. The user elected to start fresh, so the launcher
creates a new run rather than restoring the previous positions weights.

## Intensity/timing training input and dispatch (2026-10-17)

Intensity/timing already used Keras-compiled training/validation steps and held
prepared arrays in RAM. They did not have the six downstream models' eager-step
or repeated-archive-loading bottleneck. Their remaining overhead is now reduced
with stage-specific array loading, explicit two-batch prefetch and 16 sequential
batch updates per graph call. The grouping preserves individual optimizer
updates and batch sizes; it does not accumulate gradients. The launcher now
selects all eight v2 stages and routes the beat pair to this trainer.

Intensity retains only coarse audio, bin durations and normalized target shares,
and no longer builds unused frame-level timing conditions. Timing retains its
needed fine/coarse audio and count/event arrays, excluding original-note
provenance arrays. Fixed reference/predicted conditions are cached; noisy
reference conditions are regenerated once per epoch, preserving shuffle/noise
RNG order. Checksums, selected target validation and split/normalization
provenance are checked once. The full dataset reader and former joint notes
trainer retain their existing behavior. Existing datasets remain compatible.

Unlike the six downstream stages, the beat pair retains its existing complete
RAM input cache: timing globally shuffles chunk references, so a small song cache
would repeatedly reread archives. Only batch lookahead is capped at two here.
Config uses `beat_models_training_` settings, separate from the downstream
`pipeline_v2_training_` settings. Keras loss aggregation, masks, intensity
diagnostics, optimizer settings, early stopping and best-checkpoint selection
are unchanged. Batch progress rounds to graph-call boundaries; reports include
effective execution settings, retained bytes and epoch throughput.

Read-only CPU input checks on the existing 875-map beat dataset:

| Check | Measured result |
| --- | --- |
| Intensity selected arrays, all 875 maps | **58,910,208 bytes (56.2 MiB)**; one-time checked load **58.52 s** |
| Timing selected arrays, all 875 maps | **6,020,085,856 bytes (5,741.2 MiB)**; one-time checked load **93.38 s** |
| Train + validation intensity input pass | **783 songs / 783 batches**, **0.027 s** then **0.011 s** |
| Train + validation timing input pass | **30,869 chunks / 3,859 batches**, **2.593 s** then **1.617 s** |
| File opens in either warm input pass | **0**, guarded against accidental archive/JSON access |
| Previously unused intensity frame conditions, all maps | **279,829,408 bytes**, **0.297 s** to construct; now omitted |
| Twelve-map sample retained arrays | full reader **106,799,052 bytes**; intensity **1,030,512 bytes**; timing **105,436,900 bytes** |

These are input-only measurements, without model inference, optimizer updates
or accuracy comparisons. The retained-byte figures exclude Python manifest
objects and TensorFlow memory. The evidence also records sample array-loading
times; the full-reader sample includes full provenance validation while selected
sample timings cover decompression only, so they are not an end-to-end speedup
comparison. All 875 prepared NPZ checksums were verified again after the checks.

Retained CPU suite: **612 passed**, **10 comparisons excluded**. Seven additional
execution-setting validation cases passed separately after that suite had
already collected its tests. The 37 focused input, execution and configuration
checks also passed using TensorFlow **2.21.0** / legacy Keras with
`CUDA_VISIBLE_DEVICES=-1`. Primary CPU runtime was Python **3.10.19**, TensorFlow
**2.15.1**, NumPy **1.26.4**. New checks cover eager/compiled predictions, losses,
masked gradients, variable song lengths/final batches, profile caching, epoch
noise, callback ordering, mocked training integration and serialization.
Grouped optimizer graphs were traced without execution: iterations and train
counters remained zero and weights stayed unchanged. No training was started;
actual optimizer execution and GPU throughput remain unverified. The existing
user-owned positions process was left running.

Evidence:
`/home/frede/.local/share/infernosaber-baseline/diagnostics/beat-pair-training-speed-20261017-223230/`.
`benchmark_reader.py`, `reader-benchmark.json` and its log retain the reproducible
input measurements; `pytest-cpu`, `pytest-settings` and
`pytest-training-env-cpu` logs/XML retain the checks. Source snapshots and the
acceptance summary record preservation of prior working-tree changes, curated
inputs and trained checkpoints. Training and comparisons were not run.

## V2 completed runs and inference bundle (2026-10-18)

All eight selected user-run trainings completed with finite histories, matching
checkpoint hashes and report/manifest consistency. Each stopped after ten
epochs without validation improvement. Official artifact loaders verify their
current architecture, encoder, normalization and frozen-vocabulary identities.
The outdated joint `notes` model is excluded throughout this evaluation and
from the copied inference bundle.

| Stage | Best epoch / completed epochs | Best validation loss | Selected run |
| --- | --- | --- | --- |
| Intensity | 3 / 13 | 5.194017 | beat `20261017-151437` |
| Timing | 2 / 12 | 0.109131 | beat `20261017-151437` |
| Positions | 23 / 33 | 1.079350 | downstream `20261018-114213-618262` |
| Arrows | 15 / 25 | 0.509317 | downstream `20261017-225050-484720` |
| Multiplier | 6 / 16 | 0.059555 | downstream `20261017-225050-484720` |
| Hazards | 10 / 20 | 0.452233 | downstream `20261018-114213-618262` |
| Lighting | 2 / 12 | 0.002638 | downstream `20261018-112404-720174` |
| Arcs | 13 / 23 | 2.13e-8 | downstream `20261018-112404-720174` |

Independent CPU comparisons used all seven curated `bs_map_input` packages,
with no skips or exact source/decoded-hash overlap with learning groups. Frozen
normalization, the retained encoder and vocabularies were reused. Frequency
baselines use only the 695 training-map records. Arbitrary re-encoded recording
overlap remains unknown. All downstream comparisons retain human upstream
objects and preceding human history, except the explicitly isolated Positions
own-history rollout.

| Stage | Curated result | Interpretation |
| --- | --- | --- |
| Intensity | Mean KL 0.06423 vs uniform-rate 0.11091: **42.08% improvement** | Useful intensity distribution learning; CE 5.10093 vs 5.14760. |
| Timing | Composed onset F1 **50.18%** at 20 ms; **2,376 / 5,773** retained colored notes | Density is **58.84% below** the supplied reference request. Reference intensity still produces only 2,468 notes. |
| Positions | Occupied-cell agreement **37.66%** with human history, **16.04%** with own history; frequency **12.82%** | Substantial sensitivity to preceding prediction errors. Entire-pair own-history agreement is 10.72%. |
| Arrows | Occupied-hand accuracy **78.66%**, frequency **24.32%** | Useful conditional direction learning; generated positions/directions remain separate checks. |
| Multiplier | **22 TP / 10 FP / 24 FN**, positive recall **47.83%**, F1 **56.41%** | Only 46 supported positive expansions; NONE dominates headline accuracy. |
| Hazards | Bomb precision **95.45%**, recall **52.50%**, F1 **67.74%**; positive wall exact geometry **48/97** | Conditional on scheduler/native-positive queries. The scheduler matches 0/156 source walls and 20/80 bombs; 59 walls are outside supported templates. |
| Lighting | **72 TP / 71 FP / 10,583 FN**; represented-label recall **0.676%**, F1 **1.334%** at threshold 0.5 | Very weak positive recovery despite low BCE. Source coverage is 56.87%; labels can coalesce source commands. |
| Arcs | **0/66** positive attachments recovered; shape MAE **0.4862** | All 66 positives are rejected. Validation has zero positive arcs, so its near-zero loss is misleading for positive quality. |

The complete arc preparation audit found 1,318 positive attachments across 31
training maps, zero validation positives, and 18 positives in one non-curated
test map. No non-curated model comparison was run. Anchor labels are zero in
the audited supervision. Multiplier validation is 96.81% NONE. Lighting has
2,048 frozen classes and 40,663,104 parameters; only 0.056% of supervised
training binary labels are positive. These findings describe data/selection
and positive-output limitations, not corrupt checkpoints. No thresholds,
architectures, weights, alignment or generation semantics were changed.

The requested inference copy is `../Data/model/v2_expert/`. It contains one
directory per stage with the original `best.keras`, `manifest.json` and
`report.json`, plus `encoder/encoder.h5`, its identity, normalization statistics,
vocabularies, `evaluation.json`, `bundle.json` and a usage note. All eight models
and the unchanged retained encoder load from the copied locations and pass
finite CPU forward checks. Checkpoints are byte-for-byte copies; normalization
and encoder identities match. The bundle was verified in a sibling staging
directory before publication into the originally empty destination. No source
checkpoint or training dataset was changed.

Runtime: Python **3.10.19**, TensorFlow **2.15.1**, NumPy **1.26.4**, CPU only.
The ten opt-in comparison contract tests passed with
`INFERNOSABER_RUN_V2_COMPARISONS=1 python -m pytest -q tests/test_pipeline_v2_evaluation.py`.
There were no training or optimizer updates. GPU inference, full composition,
collision clearance, final alignment/export and in-game quality were not tested.

Evidence:
`../Data/training/evaluate_beats/results/v2-completed-runs-20261018-140958/`.
`evaluation-summary.json` combines the audit and model quality findings;
`run-audit.json`, `beat-summary.json`, `downstream-summary.json` and their
per-song reports preserve the metrics and supervision counts. The beat outputs
are bit-identical to the earlier comparison of this same pair.
`copied-inference-bundle.json`, `package.log` and file snapshots record the
verified inference copy and source preservation.


## V2 production generator acceptance — 2026-10-18

The application now uses the local evaluated `v2_expert` bundle for all generation.
`main.py`, `main_multi.py`, `main_app.py`, their external arguments and environment
variables remain available. Legacy model names select v2; there is no downloader
or active v1 generation fallback. The historical runner is isolated in
`evaluation/legacy_generator.py` for explicitly requested comparisons. Training
entry points remain unchanged. See [generation commands and lifecycle](generation-v2.md).

Production composition applies retained timing cleanup before position/arrow
inference, and legacy note movement/flow before multiplier, hazards and arcs.
Heuristic extra notes remain separate from retained anchors. V2 owns hazards,
arcs, chains and independent lighting; no procedural fallback fills absent
predictions. Structured export writes v3 3.2.0 or legacy v2 2.0.0. Automatic Expert
uses a separate reduced-density model pass and preserves its legacy jump settings.

Validation used the recorded WSL CPU environment (Python 3.10, TensorFlow/Keras
2.15). The full `python -m pytest -q -m 'not comparison'` suite passed **751 tests**,
with **10 opt-in comparisons deselected**, in **121.41 seconds**. After the final
automatic-Expert metadata correction, **70 focused entry/export/metadata tests**
also passed. Existing dependency deprecation/legacy-format warnings remain. Checks cover dummy composition with all
object families, both serializers, audio-only inference, masks/tails, propagation
of counts and timestamps, zero/short maps, collision rejection, independent
lighting, checkpoint compatibility, resident caches, spawn configuration,
failure reporting, metadata assembly and retained training-launcher behavior.
The app built its 115 Gradio components without an update check; importing the
main scripts built no UI, loaded no TensorFlow, and made no update request.

The final full-song acceptance used **two CPU workers**, disposable copies of
**Bad Hatter** and **Shut Up and Dance**, and requested retained NPS **3 and 5**.
Library export was disabled. Both complete combined ZIPs were published with
**zero failures or format omissions**. Ascending requested difficulties retain
the existing **Easy** and **Normal** labels in the combined metadata.

Final run elapsed: **37.26 seconds** including worker loading and packaging,
after input preparation. Each worker loaded the bundle once, prepared its recording
once, and reused audio/BPM/intensity/encoder state for the second difficulty.

| Song | Requested NPS | Raw timing events | Cleaned timing events | Final colored notes | Unique onsets | Variant seconds |
|---|---:|---:|---:|---:|---:|---:|
| Bad Hatter | 3 | 197 | 475 | 326 | 273 | 25.93 |
| Bad Hatter | 5 | 346 | 803 | 637 | 527 | 9.71 |
| Shut Up and Dance | 3 | 335 | 403 | 340 | 249 | 25.74 |
| Shut Up and Dance | 5 | 471 | 805 | 683 | 537 | 10.10 |

Per-stage seconds in the same run (timing/arrow stages include their associated
cleanup; first variants include graph warm-up):

| Stage | Bad Hatter 3 | Bad Hatter 5 | Shut Up and Dance 3 | Shut Up and Dance 5 |
|---|---:|---:|---:|---:|
| intensity | 0.241 | 0.000 | 0.244 | 0.000 |
| timing | 2.272 | 1.845 | 1.643 | 1.554 |
| positions | 1.660 | 1.582 | 1.322 | 1.660 |
| arrows | 1.378 | 1.786 | 1.223 | 1.716 |
| multiplier | 0.928 | 1.009 | 0.943 | 1.056 |
| hazards | 2.296 | 0.634 | 3.094 | 0.966 |
| arcs | 0.704 | 1.020 | 0.599 | 1.034 |
| lighting | 3.517 | 1.675 | 3.907 | 1.972 |

Both workers reported one bundle load (~10.4–10.5 s), one audio preparation
(~2.0–2.2 s), and a cache hit for the second difficulty. Predicted intensity reuse
was explicit. The requested NPS is conditioning, not an exact final-count promise;
retained legacy cleanup still changes the final density substantially.

The final Bad Hatter variants contain 20 heuristic extra notes each, no accepted
learned copies, and one lighting command each. Shut Up and Dance contains 29/44
heuristic notes, 0/2 learned copies, and no lighting commands. All four predictions
contain no bombs, walls, arcs or chains. Raw hazard and arc decisions were already
NONE; those absences were not introduced by the clearance filters. Non-NONE
multiplier choices and rejected additions are recorded separately. Sparse lighting
and absent objects remain model limitations; no accuracy improvement is claimed.

Inspection parsed every Info/beatmap JSON, checked version/difficulty metadata,
audio/cover references, finite numbers, coordinates, durations, timing bounds,
chronological notes and ZIP contents against the published folders. Synthetic
checks additionally exercise chain/arc endpoints and legacy omissions. Companion
NPZ files were reopened with `allow_pickle=False`, validating normalized intensity
and timing probabilities, pre-cleanup notes and final note counts. SHA-256 hashes
of all **83 bundle and curated source files** match the pre-run snapshot. No
training, optimizer update, checkpoint change, library export, or package change
was performed on those source assets.

Artifacts (outside source control):

- [Bad Hatter — finished ZIP](../../Data/prediction/v2_acceptance/20261018-170043/new_map/12345_Bad%20Hatter.zip)
- [Shut Up and Dance — finished ZIP](../../Data/prediction/v2_acceptance/20261018-170043/new_map/12345_Shut%20Up%20and%20Dance.zip)
- [Complete generation report](../../Data/prediction/v2_acceptance/20261018-170043/acceptance.json)
- [Package inspection](../../Data/prediction/v2_acceptance/20261018-170043/package-inspection.json)

CPU logs and before/after asset evidence: `/home/frede/.local/share/infernosaber-baseline/diagnostics/v2-generation-20261018-164614`.

Gameplay and native Windows execution remain unverified. Arc clearance currently
samples the endpoint segment rather than reconstructing the native curved-handle
trajectory; reports explicitly distinguish production validation from gameplay
validation. Optional library export was tested with disposable targets, not the
user's installed game. Legacy v2 and all object families were exercised by dummy
integration; the two full-song packages use v3. Further reduction of legacy
postprocessing, curved-arc clearance review and model/preprocessing fusion remain
follow-up work in `TODO.md`.
