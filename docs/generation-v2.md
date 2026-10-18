# V2 map generation

The application uses the evaluated local `../Data/model/v2_expert` bundle,
including its retained encoder. `tools/config/config.py:v2_model_bundle_path`
can select another compatible bundle; a relative value is relative to the
repository. Prediction and optional Beat Saber library paths stay separately
configured in `tools/config/paths.py`. No training datasets or historical source
paths are required. Missing files, changed hashes, incompatible architecture,
normalization, encoder identity, or vocabulary fail explicitly. Generation does
not download models or train them.

## Existing commands

Run from the repository root in the selected CPU environment:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python main_app.py
```

The existing single-song command and environment variables remain available:

```bash
max_speed=5 inferno_model=fav_15 python main.py
```

`inferno_model` and the Python `main(use_model=...)` argument remain accepted for
calling compatibility; both select v2. `max_speed=5` on the command line requests
5 retained colored notes per second. The Python function's `diff` argument and
internal `config.max_speed` retain the historical factor of four: `diff=20`
means 5 NPS. This internal value still drives legacy cleanup, NJS and jump offset.
`main.py` continues to disable library export in its CLI defaults.

Existing multi-difficulty CLI:

```bash
diff_list='[3,5]' python main_multi.py
```

This retains the CLI's optional library-export behavior and existing worker
count defaults. Programmatic callers can select the count and disable export:

```python
from main_multi import main_multi_par
result = main_multi_par(2, [3, 5], export_results_to_bs=False)
```

Call multiprocessing from a guarded `if __name__ == '__main__':` entry point.
The app, `main_multi`, and `main_multi_par` signatures remain available;
`main_training.py` continues to select/train all eight v2 stages independently.

The app displays only `v2_expert`. Importing any main script does not build the
UI, check for updates, or load TensorFlow. Explicit `main_app.build_app()` builds
the UI; `build_app(check_updates=False)` also skips the startup update check.

## Runtime and workers

Each spawned CPU worker receives a serializable settings/path snapshot before
TensorFlow loads. It loads each of the eight networks and encoder once, retains
one recording's decoded features, detected BPM, intensity and encoder latents,
and replaces that song cache when the recording changes. Timing windows and
encoder queries are batched; chronological stages retain bounded preceding
history and indexed note lookups. Model calls use reusable TensorFlow graphs.

A song's difficulties share a worker when there are enough songs. With fewer
songs than workers, difficulty groups split to use the requested parallelism.
Different workers intentionally have separate model and song caches. Memory use
therefore grows with worker count, especially for the retained large lighting
network. Settings in `config.py` are `v2_inference_threads` (2 per worker),
`v2_timing_batch_size` (8), and `v2_compile_inference` (True).

`main_multi_par` returns explicit `results`, `packages`, and `failures`. Only
successful variants from that call may be combined. A failed requested variant
prevents publication of a new combined package for that song. Previously complete
outputs remain intact, but never count as a successful current result. The app
reports failures instead of displaying unconditional success.

## Production stages and controls

The order is audio/BPM → intensity/timing → timing cleanup → positions/arrows →
legacy note cleanup → multiplier → hazards → arcs → package. Lighting predicts
its own timeline, including simultaneous commands and partial final blocks.

Beat features use the trained 16 kHz contract. The encoder independently uses
its trained 14,800 Hz preprocessing and fixed song normalization. Neither input
builder needs invented note or intensity targets. Fixed BPM remains optional;
otherwise the existing periodicity detector supplies BPM separately from NPS.

Timing cleanup keeps onset matching, filling, silence removal and alignment,
with actual timestamps in seconds. Moving events retains their count; nearby
merges use `min_time_diff`, keep the largest count, and cap it at two. Added events
use the nearest timing frame's stronger one/two-note probability. Zero density
remains empty. Mirroring, single-note mode, flow/movement, breaks, emphasis,
waveform changes and dot controls run before learned expansion. `random_factor`
adds bounded normalized variation only to position/arrow probabilities.

After cleanup, training reduction selects one representative per hand; heuristic
extra notes remain separate. Stable note IDs anchor expansions and arc endpoints.
Invalid/duplicate expansions and collisions are rejected without moving anchors.
Single-note mode suppresses emphasis/mirroring conflicts and multiplier expansion.
V2 supplies all bombs, walls and arcs, filtered by applicable legacy options and
clearance against final notes and chain occupancy. Arcs require existing same-hand
endpoints. The old synthetic slider turbo start is disabled. Current learned
thresholds remain unchanged; sparse lighting or absent arcs are reported, with
no procedural fallback. These conservative checks do not prove gameplay safety.

## Packages and diagnostics

Default output remains Beat Saber **3.2.0**: notes, bombs, walls, basic lighting,
arcs and native burst chains. The legacy **2.0.0** option keeps ordinary notes and
chain heads, omitting arcs, chain bodies and walls that cannot use legacy geometry.
Omissions are counted. Seconds and durations convert to beats once at export.

Metadata, cover/audio names, difficulty labels, NJS, jump offsets and ZIP layout
remain compatible. Multi-difficulty metadata is assembled as JSON. Per-variant
folders keep `1234_<internal difficulty>_<song>`; combined folders/ZIPs keep
`12345_<song>`. Ascending requested difficulties retain the Easy/Normal/Hard/
Expert/ExpertPlus labels. Optional automatic Expert uses a separate reduced-density
pass. Packages and ZIPs publish through temporary destinations only after success.

Reports live outside packages under `paths.temp_path/v2-generation/`. They include
raw/cleaned counts, heuristic changes, learned expansions, rejected objects,
legacy omissions, per-stage durations and cache metrics. Companion numeric NPZ
files contain intensity shares, raw timing probabilities/counts, processed events,
pre-cleanup model notes and final notes; they load with `allow_pickle=False`. The historical v1 runner
now lives in `evaluation/legacy_generator.py` for explicitly requested benchmarks;
it is not an application model-selection option.

See [baseline acceptance](baseline.md) for the executed checks and finished sample
packages. GPU inference, native Windows execution and in-game playability require
separate checks; this integration uses the recorded WSL CPU runtime.
