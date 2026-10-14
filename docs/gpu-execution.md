# RTX 5080 GPU execution on WSL

Verified on 2026-10-14 with the NVIDIA GeForce RTX 5080 (16 GB, compute capability
12.0), Windows driver 617.14, and Python 3.10.19. The original CPU environment
and its generated output are retained separately.

## Run the GPU app

From the repository root:

```bash
source /home/frede/.local/share/infernosaber-baseline/gpu/env.sh
python main_app.py
```

The GPU app is available at `http://127.0.0.1:7860` while the current task's server
is running. Stop an existing instance before starting another on that port.

The activation profile selects `/home/frede/.venvs/infernosaber-gpu` and sets:

- `TF_USE_LEGACY_KERAS=1`, selecting the legacy Keras API used by existing models;
- `CUDA_VISIBLE_DEVICES=0` and GPU memory growth;
- WSL's CUDA driver directory and environment-local NVIDIA library directories
  ahead of the host's other CUDA installations;
- the environment's CUDA compiler/libdevice locations and local Tk runtime;
- two TensorFlow threads per pool, disabled Gradio analytics, and disabled bytecode writes.

Activating the virtual environment alone does not configure all these paths;
use the activation profile above.

## Verified package set

| Component | GPU environment | Retained CPU baseline |
| --- | --- | --- |
| TensorFlow | 2.21.0 | 2.15.1 |
| Legacy Keras API | tf-keras 2.21.0 | Keras 2.15.0 |
| keras-tcn | 3.5.4 | 3.5.4 |
| NumPy | 1.26.4 | 1.26.4 |
| h5py | 3.14.0 | 3.16.0 |
| CUDA runtime | 12.9.79 | CPU execution selected |
| cuDNN | 9.27.0.42 | CPU execution selected |

[requirements-gpu.txt](../requirements-gpu.txt) locks all 134 installed packages,
including the NVIDIA runtime libraries. The original `requirements.txt` and
[CPU lock](../requirements-baseline.txt) retain their original baseline constraints.
The GPU lock's package dependencies passed `pip check` and the complete tested
generation path. A second clean installation of the GPU lock has not been run.

For a fresh GPU environment, first install NumPy 1.26.4 and the build tools
(`pip==26.2.1`, `setuptools==84.0.0`, `wheel==0.48.0`), then install the lock with:

```bash
python -m pip install --no-build-isolation -r requirements-gpu.txt
python -m pip check
```

The host prerequisites and environment-local Tk support are described in
[development.md](development.md). The CPU baseline activation explicitly clears
`TF_USE_LEGACY_KERAS` and `XLA_FLAGS` when switching back to TensorFlow 2.15.

## Legacy model compatibility

Project imports now consistently use `tensorflow.keras`. In the GPU profile,
TensorFlow routes this API to tf-keras 2.21. In the CPU environment, it continues
to use the original Keras 2.15 API. No pretrained model files were converted.

Standalone Keras 3 remains installed as a TensorFlow dependency, but the project
does not select it. Mixing its loader with the legacy TCN layer reproduced
`Unexpected object from deserialization, expected a layer or operation`.
`tests/test_keras_models.py` reproduces that failure by saving and reloading a
small TCN model through the project loader; it passes after the import correction.

keras-tcn 3.5.4 preserves the saved `use_weight_norm` argument. The inspected
`fav_15` TCN sets it to false, so its TensorFlow Addons branch is unused.
Addons 0.23 remains a declared TCN dependency, but its runtime is not supported
by TensorFlow 2.21. Other models requiring that branch need separate compatibility
work before being considered verified. See the source-backed
[compatibility investigation](gpu-compatibility.md) for the package rationale.

## Data, generation, and validation

The current `tools/config/paths.py` selects:

```text
/home/frede/.local/share/infernosaber-baseline/gpu/Data/
```

Its model directory links to the retained baseline model copy. Input songs and
generated output are separate from the original CPU data root. Configuration
selects `fav_15`, BPS difficulty 5, one worker, and no other difficulty slots.
`bs_song_path` remains empty and the app passes `export_results_to_bs=False`.

The CPU and GPU activation profiles select execution libraries; they both use
the data path currently configured in `paths.py`. The original CPU results remain
under `/home/frede/.local/share/infernosaber-baseline/Data/`.

Verified operations and workflows:

- Matrix multiplication, convolution, and gradients ran explicitly on `/GPU:0`
  with soft device placement disabled.
- All five original H5 models loaded with compilation enabled and produced finite
  forward-pass outputs on `/GPU:0`; stored encoders passed conversion checks.
- The autoencoder completed an optimizer step with GPU-resident trainable variables.
  Its input dataset preparation uses the required CPU operation normally.
- The regression test and four metadata tests passed in both environments.
- The GPU app generated `BWB - No Sleep.egg` at `fav_15` / BPS 5 in 23.338 seconds.
  Monitoring captured its GPU compute worker. The generated package contains
  509 colored notes, 43 obstacles, 104 arcs, and 480 lighting events.
- The original source song is unchanged and Beat Saber export stayed disabled.

Successful package generation establishes the baseline for the current beatmap
syntax. Viewer or in-game confirmation is reserved for explicit requests or
format compatibility changes.

The TensorFlow wheel uses PTX compilation for this GPU rather than native sm_120
kernel binaries. The first strict device check took 22.844 seconds; later model
checks reused the driver cache. Cold compilation can affect startup time. These
single CPU/GPU runs are not a speed comparison benchmark.

Local evidence is under `/home/frede/.local/share/infernosaber-baseline/gpu/`:
`device-check.json`, `model-check.json`, `generation-check.json`, and `logs/`.
The corresponding check commands are:

```bash
python /home/frede/.local/share/infernosaber-baseline/gpu/check_gpu.py
python /home/frede/.local/share/infernosaber-baseline/gpu/check_models.py
```

The full training-data orchestration has not been exercised; the GPU optimizer
check establishes a working training step for the existing autoencoder only.
