# InfernoSaber project goals and roadmap

The owner set these feature priorities on 2026-10-14, in the order below.
They guide future work; proposed approaches and benchmark details still need
design work before implementation.

## Starting point

App startup, pretrained-model loading, and map-package generation have been
verified in isolated Python 3.10 CPU and GPU environments. The generation check
used `fav_15` at BPS difficulty setting 5 with Beat Saber library export disabled.
See [the baseline](baseline.md) for evidence and remaining verification limits.
This establishes execution, not a map-quality benchmark.

For the current pipeline, see [model architecture](model-beat-architecture.md).
For BPS terminology, see [CONTEXT.md](../CONTEXT.md): the BPS difficulty setting
is an approximate control, distinct from measured output density.

## 1. Improve the beat timing model

Improve the first model's selection of gameplay timings. This has already
received substantial development effort and remains the highest priority.
Timing should be accurate and fitting to the music while avoiding mechanically
predictable results. Musical accuracy and expressive rhythmic choices both
matter; variation should follow the music.

Evaluate the model together with its surrounding processing. Existing
postprocessing improves its proposals and refits timings after changes from
other pipeline stages. Improvements should be visible in the final generated
beatmap as well as in the model's intermediate output.

The intended evaluation direction is comparison with human-authored beatmaps,
expanded to multiple songs and multiple BPS difficulty settings. A human map
provides a useful reference, while allowing other musically valid choices.
Matching one reference alone does not establish musical quality.

The existing comparison script and proposed benchmark work are described
[below](#human-map-timing-benchmark).

## 2. Improve and expand mapping options

Improve the quality and variety of generated gameplay, including obstacles,
arcs, and other mapping options. Replace or extend the existing simple
algorithms with AI models where learned behavior produces better results.

The selected [v2 architecture](model-architecture.md) now has a
[train-ready concept framework](pipeline-v2.md) for positions, arrows,
multiplier, hazards, independent lighting and arcs. Adapting playability checks
to each stage's ownership remains explicit review work before package export.

## 3. Fix the causes of postprocessing corrections

Keep the checks that ensure playable, correctly positioned notes. Progressively
fix the upstream causes of the problems they detect so fewer corrective edits
are needed.

This includes changes to model representations, training, or pipeline order
when justified; it goes beyond cleaning up postprocessing code. A useful
proposed measure is the frequency and type of corrections before and after
each change, alongside final map quality and density.

The simplified representation in priority 4 may be explored alongside this
work when it directly addresses those causes. The priority order remains as
listed here.

## 4. Simplify note-pattern generation, then expand it

The selected new architecture separates normalized intensity, note timing, and
per-hand note patterns. It excludes bombs and retains at most one colored note
per saber in a synchronized event. A red/blue pair counts as two retained notes;
its timing remains one onset. See [the new framework](beat-model-training.md)
for the data contract and later training workflow.

The current milestone ends at tested intermediate predictions and train-ready
model definitions. Training, a full generation rebuild, and quality evaluation
follow separately. Model size or successful untrained inference does not
establish improved gameplay.

The v2 multiplier now has train-ready targets from original/reduced relationships
and explicit chains; the combined hazards model owns bombs and walls. These new
models remain untrained. The existing seven-song CPU benchmark remains a
regression reference rather than a sufficient training or generalization set.

## Human-map timing benchmark

The [beat comparison command](../evaluation/evaluate_beat_algorithms.py) was
restored on 2026-10-14. It evaluates all selected human map packages across
requested BPS settings, uses one-to-one matching with an explicit tolerance,
defines empty-input behavior, and saves per-song and aggregate results with
timing error, measured onset density, model/settings, and code/input identity.
It reads colored-note reference onsets directly from package JSON and keeps
source audio and application configuration intact. See
[usage and scoring rules](development.md#human-map-beat-benchmark).

The default legacy stage remains processed timing proposals from `debug_beats=True`.
The trained-pair comparison additionally captures legacy thresholded raw outputs
and the new intensity/timing frame probabilities, stopping before notes. Final
serialized beatmap timings remain outside this comparison. References require
constant BPM and zero song offset; unsupported timing metadata fails explicitly.

Remaining benchmark work:

1. Curate a stable evaluation set with documented training overlap and verified
   audio alignment. The new trained-pair check excludes identifiable curated
   recordings from all learning splits; old-model overlap remains unknown.
2. Choose BPS settings, tolerances, and acceptable density tradeoffs for each
   human difficulty. Reporting measured density alone does not normalize it.
3. Extend the implemented raw/processed beat comparison to final beatmap timings
   after the later generation stages are rebuilt.
4. Add musical-fit and repetitive-pattern criteria alongside timestamp agreement.
5. Extend reference alignment to tempo changes/offsets when needed and tested.

The [dated local results](baseline.md#human-map-benchmark-2026-10-14) establish a
working timing benchmark. They do not establish musical quality or held-out
generalization.
