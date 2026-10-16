"""Shared CPU audio features and training-song-only normalization.

Frames are centered at sample-derived 100 Hz timestamps. Zero padding supplies
context at both ends; no timestamp at or after the decoded duration is retained.
Coarse features average linear mel power in each second before log compression.
"""

from copy import deepcopy
import hashlib

import numpy as np

from .contracts import FFT, FEATURES, HOP, MELS, SAMPLE_RATE, Song


def decode_audio(path):
    """Decode mono with aubio, discarding padding in the last decoder block."""
    import aubio

    source = aubio.source(str(path), samplerate=SAMPLE_RATE, hop_size=4096, channels=1)
    blocks = []
    try:
        while True:
            samples, read = source()
            block = np.asarray(samples[:read], dtype=np.float32).copy()
            if not np.isfinite(block).all():
                raise ValueError("Decoded audio contains nonfinite samples")
            blocks.append(block)
            if read < source.hop_size:
                break
    finally:
        del source
    audio = np.concatenate(blocks)
    if not len(audio):
        raise ValueError("Decoded audio is empty")
    return audio


def decoded_hash(samples):
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def extract_features(samples):
    """Return unnormalized arrays and feature metadata for one waveform."""
    import librosa

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be a nonempty finite mono waveform")
    duration = len(audio) / SAMPLE_RATE
    frame_count = (len(audio) + HOP - 1) // HOP
    frame_times = np.arange(frame_count, dtype=np.float64) * (HOP / SAMPLE_RATE)
    # Explicit padding avoids librosa's short-input warning while preserving
    # centered, constant-zero STFT semantics for even one-sample recordings.
    padded = np.pad(audio, (FFT // 2, FFT // 2))
    power = librosa.feature.melspectrogram(
        y=padded, sr=SAMPLE_RATE, n_fft=FFT, hop_length=HOP,
        n_mels=MELS, center=False, power=2.0, window="hann",
        htk=False, norm="slaney", fmin=0.0, fmax=SAMPLE_RATE / 2,
    ).T[:frame_count].astype(np.float32)
    volume = power.sum(axis=1, dtype=np.float64)
    onset = np.maximum(np.diff(power, axis=0, prepend=power[:1]), 0).sum(axis=1, dtype=np.float64)
    linear = np.column_stack((power, volume, onset))
    features = np.log1p(linear).astype(np.float32)
    bins = np.floor(frame_times).astype(np.int64)
    bin_count = int(np.ceil(duration))
    durations = np.minimum(1.0, duration - np.arange(bin_count)).astype(np.float32)
    coarse = np.vstack([np.log1p(linear[bins == index].mean(axis=0))
                        for index in range(bin_count)]).astype(np.float32)
    arrays = {"features": features, "coarse_features": coarse,
              "frame_times": frame_times, "bin_durations": durations}
    metadata = {"duration": duration, "decoded_sample_count": len(audio),
                "decoded_sha256": decoded_hash(audio),
                "feature_policy": {"sample_rate": SAMPLE_RATE, "hop": HOP, "fft": FFT,
                                   "mels": MELS, "center": True, "padding": "constant_zero",
                                   "log": "log1p_linear_power", "frame_timestamp": "center_sample",
                                   "coarse": "mean_linear_power_then_log1p",
                                   "valid_frames": frame_count}}
    return arrays, metadata


def fit_normalization(songs):
    """Fit independent fine/coarse channel statistics from train songs only."""
    train = [song for song in songs if song.metadata.get("split") == "train"]
    if not train:
        raise ValueError("Normalization requires at least one training song")
    if any("normalization_version" in song.metadata for song in train):
        raise ValueError("Fit normalization from raw training features only")
    stats = {"version": 1, "fit_split": "train", "song_ids": sorted(song.song_id for song in train)}
    for key in ("features", "coarse_features"):
        count = 0
        total = np.zeros(FEATURES, dtype=np.float64)
        squares = np.zeros(FEATURES, dtype=np.float64)
        for song in train:
            array = np.asarray(song.arrays[key], dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != FEATURES or not len(array) or not np.isfinite(array).all():
                raise ValueError(f"Invalid normalization input: {key}")
            count += len(array)
            total += array.sum(axis=0)
            squares += np.square(array).sum(axis=0)
        mean = total / count
        scale = np.sqrt(np.maximum(squares / count - np.square(mean), 0))
        scale[scale < 1e-8] = 1.0
        stats[key] = {"mean": mean.tolist(), "scale": scale.tolist(), "count": count}
    return stats


def normalize_songs(songs, stats):
    """Apply saved train statistics without mutating raw songs or their targets."""
    if stats.get("version") != 1 or stats.get("fit_split") != "train":
        raise ValueError("Unsupported normalization statistics")
    from .artifacts import statistics_hash
    normalization_hash = statistics_hash(stats)
    transformed = []
    for song in songs:
        if "normalization_version" in song.metadata:
            raise ValueError("Song features are already normalized")
        arrays = {key: value.copy() for key, value in song.arrays.items()}
        for key in ("features", "coarse_features"):
            mean = np.asarray(stats[key]["mean"], dtype=np.float64)
            scale = np.asarray(stats[key]["scale"], dtype=np.float64)
            if mean.shape != (FEATURES,) or scale.shape != (FEATURES,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Invalid normalization channel statistics")
            arrays[key] = ((arrays[key] - mean) / scale).astype(np.float32)
        metadata = deepcopy(song.metadata)
        metadata["normalization_version"] = stats["version"]
        metadata["normalization_sha256"] = normalization_hash
        transformed.append(Song(song.song_id, metadata, arrays))
    return transformed
