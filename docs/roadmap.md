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

For the current pipeline, see [model architecture](model-architecture.md).
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

Which options receive their own models, their training targets, and how they
interact with note generation remain design questions. Existing playability
checks should continue to guard generated output as these options evolve.

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

The current note-pattern model predicts complete simultaneous note/bomb
snapshots. Explore a simpler initial representation with at most one note or
bomb per side in each time slot. Whether obstacles belong in this representation
is still open.

A second stage would reconstruct richer snapshots, including multiple notes
per side, using the simplified output and the current song snippet. This stage
could be AI or non-AI; an autoencoder-like reconstruction process is a proposed
analogy, not a selected architecture.

Before implementation, resolve what "side" means for bombs and obstacles, how
the reduced representation preserves musical intent, and how expansion works
with timing, model feedback, and playability checks. Compare the complete
pipeline against the current representation rather than judging simplification
by model size alone.

## Human-map timing benchmark

The [beat comparison command](../evaluation/evaluate_beat_algorithms.py) was
restored on 2026-10-14. It evaluates all selected human map packages across
requested BPS settings, uses one-to-one matching with an explicit tolerance,
defines empty-input behavior, and saves per-song and aggregate results with
timing error, measured onset density, model/settings, and code/input identity.
It reads colored-note reference onsets directly from package JSON and keeps
source audio and application configuration intact. See
[usage and scoring rules](development.md#human-map-beat-benchmark).

The evaluated stage remains processed timing proposals from `debug_beats=True`.
Raw timing-model probabilities and final beatmap timings require separate
instrumentation. References currently require constant BPM and zero song offset;
unsupported timing metadata fails explicitly.

Remaining benchmark work:

1. Curate a stable evaluation set with documented training overlap and verified
   audio alignment. The restored local set records overlap as unknown.
2. Choose BPS settings, tolerances, and acceptable density tradeoffs for each
   human difficulty. Reporting measured density alone does not normalize it.
3. Compare raw model output, processed proposals, and final beatmap timings so
   regressions can be attributed to their stage.
4. Add musical-fit and repetitive-pattern criteria alongside timestamp agreement.
5. Extend reference alignment to tempo changes/offsets when needed and tested.

The [dated local results](baseline.md#human-map-benchmark-2026-10-14) establish a
working timing benchmark. They do not establish musical quality or held-out
generalization.
