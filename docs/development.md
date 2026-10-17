# WSL development environment

The CPU baseline was verified on 2026-10-14 with Python 3.10.19, the `fav_15` model
set, and one real-song generation run at BPS difficulty 5. RTX 5080 GPU execution
is now verified in a separate environment; use [gpu-execution.md](gpu-execution.md)
for its activation profile and package lock. Native Windows execution and the
full training workflow have not been verified.
Successful generation is the acceptance criterion for the current beatmap syntax;
viewer or in-game checks apply when requested or when format compatibility changes.

## Use the retained CPU environment

Run commands from the repository root:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python --version
python -m pip check
python -m pytest -q
python main_app.py
```

The activation file selects `/home/frede/.venvs/infernosaber`, adds the local
Tcl/Tk runtime paths, disables Gradio analytics, and selects CPU execution with
two TensorFlow threads per thread pool. It also disables Python bytecode writes.
The current app on `http://127.0.0.1:7860` uses the GPU profile. Stop it before
starting a CPU instance on the same port.

Original CPU baseline locations (retained):

| Purpose | Location |
| --- | --- |
| Python environment | `/home/frede/.venvs/infernosaber` |
| Data root | `/home/frede/.local/share/infernosaber-baseline/Data/` |
| Model copy | `Data/model/fav_15/` under that root |
| Input copies | `Data/prediction/songs_predict/` |
| Generated packages | `Data/prediction/new_map/` |
| Logs and check reports | `/home/frede/.local/share/infernosaber-baseline/` |

The source model set and sample song were copied into this workspace. Generation
may normalize, rename, or remove input files in its configured input directory;
use copies of source songs.

## Python packages

[requirements-baseline.txt](../requirements-baseline.txt) records the resolved
package versions from the successful WSL run. It supplements the original
`requirements.txt` without changing its declared constraints. The key versions
are TensorFlow 2.15.1, Keras 2.15.0, keras-tcn 3.5.4, NumPy 1.26.4,
scikit-learn 1.3.2, and Gradio 5.29.1.

The inference import path also needs aubio, pydub, and matplotlib; these are
included in the lock. Pytest is included for the existing test suite.

To recreate the Python package set in a fresh environment, first provide the
host prerequisites below, then run from the repository root:

```bash
python3.10 -m venv /home/frede/.venvs/infernosaber-rebuild
source /home/frede/.venvs/infernosaber-rebuild/bin/activate

# Limit these installation commands to PyPI; retain global pip configuration.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=

python -m pip install pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 numpy==1.26.4
python -m pip install --no-build-isolation -r requirements-baseline.txt
python -m pip check
```

Use a new environment directory when reproducing the setup. System site packages
must remain disabled: the system NumPy discovered by Python 3.10 failed to import
its compiled extension during the original baseline.

The package set passed dependency checks, app HTTP/API checks, model loading and
forward passes, encoder conversion checks, metadata tests, and one-song
generation. A second clean temporary environment successfully reinstalled the
lock, passed `pip check`, imported the runtime stack including Tkinter, and passed
all four metadata tests. That temporary environment was removed after verification.

## Host prerequisites and local Tk support

This WSL host already provided FFmpeg 6.1.1, GCC, Python 3.10 development headers,
pkg-config, libavcodec/libavformat/libavutil/libswresample development packages,
and libsamplerate development files. These are host prerequisites rather than
packages captured by pip. Use `ffmpeg -version` and `pkg-config` when diagnosing
codec/build differences on another host.

Python 3.10's Tkinter module and Tcl/Tk libraries were missing. To keep this setup
local, Ubuntu packages were downloaded and extracted without installing them
system-wide:

- `python3.10-tk` 3.10.22-1+noble1, from the configured deadsnakes PPA;
- `libtcl8.6` 8.6.14+dfsg-1build1;
- `libtk8.6` 8.6.14-1build1.

The extracted files live under
`/home/frede/.local/share/infernosaber-baseline/tk-runtime`. A `.pth` file inside
the virtual environment adds the Python module and extension directories;
`env.sh` sets `LD_LIBRARY_PATH`, `TCL_LIBRARY`, and `TK_LIBRARY`. Tkinter import
and Tcl initialization passed. Interactive folder-dialog behavior has not been
tested.

For a fresh environment on this host, copy the local module-path file:

```bash
cp /home/frede/.venvs/infernosaber/lib/python3.10/site-packages/infernosaber-tk.pth \
   /home/frede/.venvs/infernosaber-rebuild/lib/python3.10/site-packages/
```

The existing `env.sh` activates the original environment. Activate the rebuild
environment after sourcing it if testing a replacement.

## Aubio build compatibility

The PyPI aubio 0.4.9 release failed to build against this host's FFmpeg headers:
`AVStream` no longer exposes the `codec` field referenced by that release.
The working environment instead uses aubio 0.5.0a0 from upstream commit
`ad5cf975aed08cc4562dd008cf9f83b12b82ffb8`.

The lock pins that commit. Build it with NumPy 1.26.4 already installed and
`--no-build-isolation`, as in the commands above. This follows aubio's
[documented source-install method](https://aubio.org/manual/latest/installing.html#cheat-sheet)
using its official GitHub mirror. Its successful use here includes real Ogg/Vorbis
input decoding and resampling during map generation.

## Baseline settings and checks

The existing configuration framework is retained. The settings below describe
the initial CPU run; the active data root was subsequently moved to the separate
GPU workspace described in [gpu-execution.md](gpu-execution.md):

- `tools/config/paths.py`: the data root above; `bs_song_path` remains empty.
- `tools/config/config.py`: `fav_15`, `max_speed = 4 * 5.0`, one worker,
  `difficulty_1 = 5`, and all other difficulty slots set to zero.
- The app's generation callback passed `export_results_to_bs=False` because
  no Beat Saber library was configured. Keep the Beat Saber folder field empty
  for development runs.

The local model check can be rerun from the repository root:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python /home/frede/.local/share/infernosaber-baseline/check_models.py
```

That baseline harness loads all five H5 files using the project's Keras loader
and TCN custom layer, runs forward passes, and checks the stored encoders.
It is a local check artifact outside the source repository.

The app's settings callbacks rewrite configuration files with LF line endings.
After the baseline generation callback, `config.py` was restored to its original
CRLF line endings while retaining the intended settings. Account for this
behavior when reviewing changes after manual app use.

For exact sample details, generated assets, and remaining verification, see the
[runtime baseline report](baseline.md#runtime-check-2026-10-14).

## Human-map beat benchmark

Run the restored comparison command from the repository root in the retained
CPU profile:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python -m evaluation.evaluate_beat_algorithms \
  --evaluation-root "../Data/training/evaluate_beats" \
  --bps 3 5 7 --difficulty ExpertPlus --characteristic Standard \
  --tolerance 0.02
```

Routine benchmark validation uses CPU. GPU evaluation requires an explicit task
request under the current agent instructions; GPU is otherwise reserved for training.

The comparison set is curated exclusively in `evaluate_beats/bs_map_input`.
Use these packages for project benchmarks; `favorites_bs_input` and the legacy
`evaluate_beats/temp` contents are excluded. The evaluation-root command inspects
every immediate package folder, validates its selected human beatmap and fully
decodes its Ogg audio, then rebuilds these sibling folders from accepted packages:

| Folder/file | Contents |
| --- | --- |
| `maps/` | Selected raw beatmap and Info copies with flat song filenames |
| `songs_egg/` | Unmodified Ogg audio copies |
| `maps_dict_all/` | Legacy NumPy notes, events, and obstacles arrays |
| `songs_diff/` | Matching sorted `name_ar.npy` / `diff_ar.npy` indices |
| `preprocessing_report.json` | Input provenance, accepted packages, and per-package skip reasons |
| `results/` | New timestamped benchmark JSON reports |
| `archive/` | Previous derived folders and preprocessing report, retained before replacement |

Raw input packages and unrelated folders remain intact. Preparation failures
leave no partial song in the active derived folders. The benchmark reads only
the accepted source packages; the derived arrays provide an updated snapshot for
existing evaluation tools. Existing runtime/application paths are not rewritten.
`--prepare-only` refreshes this snapshot without loading TensorFlow or a model.
`--output` optionally selects a new benchmark report destination. Skipped inputs
are excluded from scoring, listed in the preprocessing and benchmark reports,
and cause exit status 1 even when all accepted songs complete successfully.

The legacy arrays retain individual notes and bombs, with timestamps in seconds.
Obstacle duration retains its legacy beat units. `diff_ar.npy` uses the legacy
individual-note count divided by the first-to-last-note interval, rounded to two
decimals. These cache values are separate from the benchmark's unique colored
onset density over full audio duration, described below.

For explicit external fixtures, `--maps-dir` accepts one or more unpacked map packages or folders whose immediate
children are packages. Discovery is sorted and duplicate paths are removed.
Each package must contain `Info.dat`, its referenced audio, and the exact human
difficulty/characteristic requested. There is no automatic difficulty fallback;
run separate commands to compare different human difficulty labels.
The current local evaluation packages are under
`/mnt/c/Users/frede/Desktop/BS_Automapper/Data/training/evaluate_beats/bs_map_input/`.
Use native paths when running on Windows.

The configured beat model is used by default. `--model-dir` selects an existing
model folder for this process without editing configuration files or downloading
weights. Every song/BPS pair starts with the same configured settings, sets both
`max_speed` and `max_speed_orig` to four times the BPS setting, and seeds Python
and NumPy (`--seed`, default 3). Audio is copied afresh into a disposable directory
for every pair. Configuration, paths, and random state are restored afterward.
The command calls the existing `gen_beats.main(..., debug_beats=True)` path,
which loads only the beat model and returns **processed timing proposals** before
note generation, map creation, or Beat Saber export. It does not normalize the
source audio or use the legacy training-map importer.

Reference loading currently supports v2 `Info.dat` with v2 or v3 beatmaps,
constant positive BPM, zero song offset, and unshuffled timing. Beat times are
converted to elapsed seconds using `beat * 60 / BPM`. Nonempty tempo-change
metadata, nonzero offsets/shuffle, and unknown format versions fail explicitly.
Reference onsets are unique timestamps of colored notes: simultaneous notes count
once, bombs are excluded, and nearby distinct timestamps remain distinct. Arcs,
obstacles, and lighting do not add reference onsets.

Scoring uses sorted one-to-one matching within an inclusive tolerance in seconds
(default 0.02, or 20 ms). A reference can match only one prediction; duplicate
predictions remain separate and can therefore become false positives. Earliest
eligible matches maximize the match count; their timing error is deterministic,
not a minimum-error assignment. Empty predictions or references yield zero
precision, recall, and F1, with null timing error when no pairs match. Mean absolute
timing error is reported in milliseconds over matched pairs only.

The JSON report contains per-song/per-BPS counts, precision/recall/F1, matched
timing error, reference and prediction timestamps, and measured onset density.
Density counts timestamps over the **full audio duration**, including silence;
it is distinct from the BPS difficulty setting and from individual-note density.
Aggregates pool counts and matched timing errors across successful songs at each
BPS setting. Failed inputs/runs are listed and excluded from scoring, and cause
exit status 1. The destination must be a new file outside the input packages.

Reports also retain input paths and SHA-256 hashes, human author/difficulty,
model identity, base/effective timing settings, seed, runtime versions/devices,
Git revision/dirty status, and benchmark source hashes. Keep reports and song
assets outside the repository. Seeded runs help comparisons within one runtime;
they do not guarantee identical TensorFlow results across devices or versions.

Training overlap is recorded as unknown; these local packages do not establish
a held-out set. Agreement with one human map is a timing reference, not a measure
of musical quality. Raw model probabilities and final serialized beatmap timings
are not evaluated by this command. See the [dated benchmark results](baseline.md#human-map-benchmark-2026-10-14)
and [remaining benchmark work](roadmap.md#human-map-timing-benchmark).
The older three-song comparison included favorites and is retained as historical
evidence only; use the curated-set results for subsequent decisions.


### Trained intensity/timing versus legacy beats

The comparison backend evaluates only beats. It does not load the trained notes
checkpoint or call the later generation stages. Run it in the retained CPU
profile with an explicitly selected trained run:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python -m evaluation.evaluate_beat_algorithms \
  --evaluation-root "../Data/training/evaluate_beats" \
  --backend compare \
  --trained-run "../Data/training/beat_models_v2/runs/20261017-151437" \
  --bps 3 5 7 --difficulty ExpertPlus --characteristic Standard \
  --tolerance 0.02
```

Use `--backend intensity-timing` for the new pair alone. Alternatively supply
`--intensity-artifact` and `--timing-artifact` instead of `--trained-run`.
`--backend legacy` preserves the original command and report behavior, and
remains the default for existing callers. `--prepare-only` refreshes the existing
evaluation snapshot without loading any model, including with a new backend.

The new pair uses the shared audio extractor, saved training normalization,
predicted whole-song intensity, Info BPM as a separate tempo input, and requested
retained colored notes/second. Timing emits each valid core frame once and
preserves tails; count argmax selects zero/one/two notes. There is no subsequent
threshold tuning, minimum-spacing pass, onset rematching, snapping or density
forcing. A simultaneous pair is one onset and two retained notes. The extra
reference-density scenario supplies only the human map's global retained NPS;
its local intensity still comes from the trained model.

The legacy observer records thresholded model onsets before spacing/filling,
onset rematching and BPM alignment, and also retains the unchanged processed
legacy result. Observation changes only an in-memory wrapper during a debug
call and restores it on failure; generation source files and algorithms stay
unchanged. The original legacy index-to-time convention is retained.

All three stages are scored against the **same synchronized reference** using
`config.min_time_diff` (currently 20 ms). Exact source-reference scores are also
reported for continuity with older benchmark reports. The primary metric is
pooled one-to-one onset F1 at the requested tolerance; saved metrics include
10/20/35/50/100 ms tolerances, precision, recall, matched error and actual onset
rate. Equal numeric legacy BPS and new colored NPS controls are not guaranteed
to produce equal densities. The report records those units and a separate
new-model run at each human map's retained NPS. It does not tune the model's
predictions to a reference count or select a better decoder on benchmark songs.

Schema-2 comparison reports contain per-stage summaries, matched-song paired
comparisons, source/checkpoint hashes, normalization identity, overlap evidence,
intensity errors, colored-note counts and per-frame class diagnostics. Adjacent
`*-arrays/` directories contain non-executable NPZ probabilities/shares/counts;
they require a new destination on each run. Per-map failures are recorded and
produce a nonzero exit status. Shared preparation failures are excluded from
paired comparisons, and summaries must be read together with their run counts.

GPU-trained tf-keras 2.21 archives use a module name unavailable in the CPU Keras
2.15 runtime. The adapter validates the recorded architecture and complete
serialized execution graph against the current stage, then strictly loads its
weights into that matching CPU model. Unknown graph/configuration changes or
mismatched stages, statistics, statuses and checksums are rejected. This neither
converts nor overwrites checkpoint files.

Identifiable recording overlap must be checked separately from timing quality.
The old model's training overlap remains unknown. Agreement on these seven human
maps is not a measure of playability or general musical quality. Trained
beat-only results are recorded in [the baseline](baseline.md).

## New beat-learning framework

Use [the new training guide](beat-model-training.md) for dataset preparation,
dummy/CPU forward checks, and explicit later training commands. It uses the
existing Python environments and path configuration with isolated dataset/run
folders. Importing its CLI does not launch the legacy training orchestrator.
