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
