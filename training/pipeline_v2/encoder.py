"""Retained encoder preprocessing with a fixed, query-independent song scale.

No checkpoint is discovered or loaded here. Callers inject the retained encoder
and its fingerprint, making real inference an explicit operation.
"""

import hashlib
import json

import numpy as np

SAMPLE_RATE = 14800
PATCH_SAMPLES = 29600
ENCODER_REPRESENTATION = {"version": 2, "sample_rate": SAMPLE_RATE, "window_seconds": 2.,
                          "shape": [24, 20, 1], "spectrum": "absolute_x100_hann_log_power",
                          "nperseg": 1480, "noverlap": 15, "eps": 1e-10,
                          "resize": "Pillow_default_float_bicubic", "padding": "constant_zero",
                          "duration_policy": "canonical_beat_duration_with_native_tail_padding",
                          "normalization": "fixed_song_grid_patch_min_then_global_minmax"}


def decode_encoder_audio(path):
    """Decode a separate mono 14,800 Hz waveform, trimming native tail padding."""
    import aubio
    from training.beat_models.audio_logging import filter_vorbis_timestamp_warning

    blocks = []
    with filter_vorbis_timestamp_warning():
        source = aubio.source(str(path), samplerate=SAMPLE_RATE, hop_size=4096, channels=1)
        try:
            while True:
                samples, read = source()
                blocks.append(np.asarray(samples[:read], np.float32).copy())
                if read < source.hop_size:
                    break
        finally:
            del source
    samples = np.concatenate(blocks)
    _waveform(samples)
    return samples


def _waveform(samples):
    samples = np.asarray(samples, np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Encoder requires a nonempty finite mono waveform")
    return samples


def _duration(samples, duration):
    if duration is None:
        return len(samples) / SAMPLE_RATE
    if (isinstance(duration, (bool, np.bool_)) or not isinstance(duration, (int, float, np.integer, np.floating))
            or not np.isfinite(duration) or duration <= 0):
        raise ValueError("Encoder duration must be finite and positive")
    return float(duration)


def spectrogram_patches(samples, times, duration=None):
    """Produce unnormalized legacy-compatible patches, padding both boundaries."""
    from PIL import Image
    from scipy import signal

    samples = _waveform(samples)
    duration = _duration(samples, duration)
    times = np.asarray(times, np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= duration):
        raise ValueError("Encoder queries must lie inside decoded audio")
    result = np.empty((len(times), 24, 20, 1), np.float32)
    half = PATCH_SAMPLES // 2
    for index, center in enumerate(np.around(times * SAMPLE_RATE).astype(np.int64)):
        patch = np.zeros(PATCH_SAMPLES, np.float32)
        start, end = int(center - half), int(center + half)
        lo, hi = max(0, start), min(len(samples), end)
        if hi > lo:
            patch[lo - start:hi - start] = samples[lo:hi]
        patch = np.abs(patch) * 100
        _, _, power = signal.spectrogram(patch, fs=SAMPLE_RATE, window="hann", nperseg=1480,
                                         noverlap=15, detrend=False)
        spectrum = np.log(power.T.astype(np.float32) + 1e-10)
        spectrum -= spectrum.min()
        resized = np.asarray(Image.fromarray(spectrum).resize((24, spectrum.shape[0]))).T
        if resized.shape != (24, 20):
            raise ValueError("Retained encoder spectrogram shape changed")
        result[index, :, :, 0] = resized
    return result


class FixedEncoder:
    """Cache query latents using one scale fitted on a deterministic audio grid.

    ``predict`` accepts float32 [N,24,20,1] and returns float32 [N,16]. The
    fingerprint must identify the exact retained encoder weights.
    """

    def __init__(self, samples, predict, fingerprint, grid_seconds=.5, batch_size=128, duration=None):
        self.samples = _waveform(samples).copy()
        self.duration = _duration(self.samples, duration)
        if not callable(predict) or not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("An explicit encoder callable and checkpoint fingerprint are required")
        if not np.isfinite(grid_seconds) or grid_seconds <= 0:
            raise ValueError("Encoder normalization grid must be finite and positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Encoder batch size must be a positive integer")
        self.predict = predict
        self.batch_size = batch_size
        self.grid = np.arange(0, self.duration, grid_seconds, dtype=np.float64)
        minimum, maximum = np.inf, -np.inf
        for start in range(0, len(self.grid), batch_size):
            patches = spectrogram_patches(self.samples, self.grid[start:start + batch_size], self.duration)
            minimum, maximum = min(minimum, float(patches.min())), max(maximum, float(patches.max()))
        self.minimum = minimum
        self.scale = maximum - minimum if maximum > minimum else 1.
        self.metadata = {**ENCODER_REPRESENTATION, "grid_seconds": float(grid_seconds),
                         "grid_count": len(self.grid), "minimum": minimum, "maximum": maximum, "scale": self.scale,
                         "decoded_sample_count": len(self.samples), "decoded_duration_seconds": len(self.samples) / SAMPLE_RATE,
                         "canonical_duration_seconds": self.duration,
                         "query_duration_source": "native_decode" if duration is None else "canonical_audio",
                         "tail_shortfall_samples": max(0, int(np.ceil(self.duration * SAMPLE_RATE)) - len(self.samples)),
                         "tail_shortfall_seconds": max(0., self.duration - len(self.samples) / SAMPLE_RATE),
                         "silence_scale_fallback": maximum == minimum,
                         "encoder_fingerprint": fingerprint,
                         "decoded_sha256": hashlib.sha256(self.samples.astype("<f4").tobytes()).hexdigest()}
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.cache = {}

    def patches(self, times):
        return ((spectrogram_patches(self.samples, times, self.duration) - self.minimum) / self.scale).astype(np.float32)

    def __call__(self, times):
        times = np.asarray(times, np.float64)
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= self.duration):
            raise ValueError("Encoder queries must lie inside decoded audio")
        centers = np.around(times * SAMPLE_RATE).astype(np.int64)
        missing = {}
        for center, time in zip(centers, times):
            if int(center) not in self.cache:
                missing.setdefault(int(center), float(time))
        keys = list(missing)
        for start in range(0, len(keys), self.batch_size):
            batch = keys[start:start + self.batch_size]
            values = np.asarray(self.predict(self.patches([missing[key] for key in batch])), np.float32)
            if values.shape != (len(batch), 16) or not np.isfinite(values).all():
                raise ValueError("Retained encoder must return finite [N,16] latents")
            for key, value in zip(batch, values):
                self.cache[key] = value.copy()
        return np.asarray([self.cache[int(center)] for center in centers], np.float32).reshape(-1, 16)
