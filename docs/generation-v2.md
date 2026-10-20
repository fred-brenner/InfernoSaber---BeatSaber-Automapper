# V2 map generation

The application uses seven v2 stages from the evaluated local
`../Data/model/v2_expert` bundle: intensity, timing, positions, arrows, multiplier,
hazards and arcs, plus its retained encoder. Lighting comes exclusively from
the original v1 `fav_15` model and event vocabulary. Independent v2 lighting is
**not yet implemented for production — optional/deferred**; its experimental
code and artifacts remain available. Existing bundle contents are unchanged.

Settings in `tools/config/config.py` accept repository-relative or absolute paths:

- `v2_model_bundle_path`: `../Data/model/v2_expert`
- `v1_lighting_model_path`: `../Data/model/fav_15/tf_event_gen_2_12_5_5__22_18.h5`
- `v1_lighting_vocabulary_path`: `../Data/model/fav_15/onehot_encoder_events.pkl`

Application preflights validate the seven-stage production projection and require
the v1 lighting assets; they do not require the experimental v2 lighting stage.
Strict eight-stage bundle validation and experimental tools remain available.
Prediction and optional Beat Saber library paths stay separately configured in
`tools/config/paths.py`. No training datasets or historical source paths are
required. Missing assets or incompatible selected production contracts fail
explicitly; v2 bundle hashes, normalization, encoder identity and vocabularies
remain validated. Generation does not download models or train them.

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
`main_training.py` selects the seven production v2 stages. Explicit experimental
lighting training uses `python -m training.pipeline_v2 train --stage lighting`;
see [training selection](pipeline-v2.md#small-development-launcher).

The app displays only `v2_expert`. Importing any main script does not build the
UI, check for updates, or load TensorFlow. Explicit `main_app.build_app()` builds
the UI; `build_app(check_updates=False)` also skips the startup update check.

## Runtime and workers

Each spawned CPU worker receives a serializable settings/path snapshot before
TensorFlow loads. It loads the seven v2 networks, retained encoder and v1 lighting
model/vocabulary once per worker, retains one recording's decoded features,
detected BPM, intensity and encoder latents, and replaces that song cache when
the recording changes. Timing windows and
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
legacy note cleanup → multiplier → hazards → arcs → package. V1 lighting uses
the cleaned gameplay timing sequence, with audio latents from the current
`FixedEncoder`. It preserves the legacy 16-step block convention and pairs
decoded events with the timestamp prefix. For `N` cleaned gameplay timestamps,
there are `max(floor(N/16) - 1, 0)` calls and 16 times that many events; incomplete
tails and the final complete block are omitted. Songs with fewer than 32 timings
therefore have no lighting events. Each block uses argmax type/value classes and
one-hot history, with the existing random score weighting. This is separate from
the experimental v2 independent schedule and multi-label decoder.

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
adds bounded normalized variation to v2 position/arrow probabilities. The same
control also feeds the separate original v1 lighting score weighting through
`config.random_note_map_factor`.

After cleanup, training reduction selects one representative per hand; heuristic
extra notes remain separate. Stable note IDs anchor expansions and arc endpoints.
Invalid/duplicate expansions and collisions are rejected without moving anchors.
Single-note mode suppresses emphasis/mirroring conflicts and multiplier expansion.
V2 supplies all bombs, walls and arcs, filtered by applicable legacy options and
clearance against final notes and chain occupancy. Arcs require existing same-hand
endpoints. The old synthetic slider turbo start is disabled. Current learned
thresholds remain unchanged for legacy bundles; absent arcs are reported, with no
procedural fallback. Repaired production replacement artifacts select versioned
hazard and arc policies. Experimental v2 lighting policies do not affect
production. A candidate bundle may include checkpoint-bound validation
calibration sidecars; runtime verifies their hashes
and contracts before applying thresholds. Stage-specific compatibility keeps
unchanged upstream checkpoints usable. See [repair workflow](sparse-stage-repair.md). These conservative checks do not prove gameplay safety.

The 2026-10-19 arc-only update uses the repaired contract and a calibrated **0.18**
attachment threshold. It was installed at the user's request despite failing
autoregressive positive-case evaluation: **0/358 validation** and **0/66 curated**
arcs recovered. The disposable Insane packages also contain no arcs. Other stage
assets are unchanged, and the previous bundle remains available for rollback.
See the [evaluation and backup paths](baseline.md#arc-only-deployment-and-failed-positive-case-evaluation-2026-10-19).

The 2026-10-20 hazard-only update also selects the repaired quarter-beat policy.
Its checkpoint-bound bomb threshold is **0.97**; wall geometry remains categorical.
It was installed as requested despite failing the seven-song positive-case screen:
**0/209 supported wall sections** and **0/52 bomb cells** matched. Full CPU sample
packages do contain valid cleared walls, but reference agreement and own-history
robustness remain poor. Other stage assets, including arcs and v1 lighting, are
unchanged. See the [hazard report and rollback backup](baseline.md#hazard-only-deployment-and-cpu-validation-2026-10-20).

## Intensity postprocessing

Generation adjusts the intensity profile before timing inference. The four
`config.py` controls apply only during production generation:

| Setting | Default | Meaning |
| --- | --- | --- |
| `v2_intensity_low_boost` | `0.1` | Fraction of the deficit below uniform density to lift. |
| `v2_intensity_high_reduction` | `0.1` | Fraction of the excess above uniform density to trim. |
| `v2_intensity_scale` | `1.0` | Overall target gain; one preserves average conditioning NPS. |
| `v2_intensity_adjustment_strength` | `0.8` | Fraction of the change toward the configured target to apply. |

Boost, reduction and strength must be finite numeric values in `[0, 1]`; scale
must be finite and nonnegative. Invalid settings raise an explicit error.
Uniform shares are each bin's actual duration divided by decoded song duration,
so a partial final bin receives proportionately less mass. For raw shares `s`
and uniform shares `u`, generation computes:

```text
target = s + low_boost * max(u - s, 0) - high_reduction * max(s - u, 0)
target = target / sum(target)
mass = (1 - strength) * s + strength * scale * target
conditioning_nps = requested_nps * sum(mass)
adjusted_shares = mass / sum(mass)
```

Timing and the downstream stages receive the adjusted normalized shares and
effective conditioning NPS. The original requested NPS still controls package
identity and legacy gameplay settings. Zero mass retains the raw normalized
profile with zero conditioning NPS and skips timing prediction. Zero requested
NPS stays empty. Strength zero preserves the original conditioning exactly.

The defaults gently flatten the profile while preserving its average density:
relative intensities `0.5 / 1.5` become `0.54 / 1.46`. With both shaping controls
set to one, the same inputs become `0.9 / 1.1` at strength `0.8`. Scale `1.25`
with strength `0.8` raises average conditioning NPS by 20%. These changes affect
conditioning; the timing decoder, gameplay cleanup and note expansion still
determine the actual emitted and exported NPS.

Workers cache the raw model shares and reapply the current controls for every
generation call, including cached difficulties. Adjustments never accumulate
between calls. Reports record the controls and requested/effective NPS under
`intensity_postprocessing`; prediction NPZ files save `raw_intensity`, adjusted
`intensity`, and scalar `conditioning_nps`. Training, prepared targets and model
artifacts retain their original contracts.

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
files contain raw/adjusted intensity shares, effective conditioning NPS, raw timing
probabilities/counts, processed events, pre-cleanup model notes and final notes;
they load with `allow_pickle=False`. The historical v1 runner
now lives in `evaluation/legacy_generator.py` for explicitly requested benchmarks;
it is not an application model-selection option.

See [baseline acceptance](baseline.md) for the executed checks and finished sample
packages. GPU inference, native Windows execution and in-game playability require
separate checks; this integration uses the recorded WSL CPU runtime.
