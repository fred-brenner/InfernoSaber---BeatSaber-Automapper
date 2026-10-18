"""Observe legacy learned timings without changing the generation pipeline."""

import numpy as np


class LegacyBeatObserver:
    """Capture thresholded outputs while forwarding the normal debug result.

    The wrapper exists only during one sequential evaluation call. Captured
    timings use the generator's established frame indexing and initial window
    cutoff, before spacing, filling, onset rematching, or BPM alignment.
    """

    def __init__(self, generator_module=None):
        self.generator_module = generator_module
        self.records = []

    def __call__(self, names, debug_beats=True):
        if debug_beats is not True:
            raise ValueError("Legacy timing observation requires debug_beats=True")

        from tools.config import config

        generator = self.generator_module
        if generator is None:
            from evaluation import legacy_generator as generator

        record = {"name": names[0] if names else None,
                  "bps_setting": float(config.max_speed_orig) / 4,
                  "times_seconds": [], "captured": False, "status": "failed"}
        original = generator.apply_first_beat_thresholding

        def observe(probabilities):
            # The original threshold function or later spacing pass may mutate
            # their arrays. Store independent values before either can do so.
            record.update({
                "probabilities": np.asarray(probabilities).reshape(-1).copy().tolist(),
                "effective_threshold": float(config.thresh_beat),
                "threshold_start": float(config.threshold_start),
                "threshold_end": float(config.threshold_end),
                "beat_spacing": float(config.beat_spacing),
                "window": float(config.window),
            })
            binary = original(probabilities)
            captured = np.asarray(binary).reshape(-1).copy()
            times = np.flatnonzero(captured) / record["beat_spacing"]
            record.update({"thresholded_binary": captured.tolist(),
                           "times_seconds": times[times > record["window"]].tolist(),
                           "captured": True})
            return binary

        generator.apply_first_beat_thresholding = observe
        try:
            result = generator.main(names, debug_beats=debug_beats)
            if result is None or np.isscalar(result):
                record["error"] = f"Beat generation returned failure status {result!r}"
            elif not record["captured"]:
                record["error"] = "Generator returned without calling beat thresholding"
            else:
                record["status"] = "ok"
            return result
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            generator.apply_first_beat_thresholding = original
            self.records.append(record)
