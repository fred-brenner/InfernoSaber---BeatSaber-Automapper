"""Estimate a constant pulse rate from whole-song onset periodicity."""

from collections.abc import Mapping

from aubio import source
import librosa
import numpy as np
from scipy.signal import periodogram


def _estimate_bpm(audio, samplerate, win_s, hop_s):
    if len(audio) < win_s or not np.any(audio):
        return 0.0

    onset = librosa.onset.onset_strength(
        y=audio, sr=samplerate, hop_length=hop_s, n_fft=win_s,
        n_mels=min(64, win_s // 4),
    )
    if not np.any(onset) or len(librosa.onset.onset_detect(
            onset_envelope=onset, sr=samplerate, hop_length=hop_s)) < 4:
        return 0.0

    # Whole-song frequency evidence avoids the short-period bias of the old
    # streaming tracker. Padding samples the spectral peak more finely.
    nfft = 16 * (1 << (len(onset) - 1).bit_length())
    frequencies, power = periodogram(
        onset, fs=samplerate / hop_s, window="hann", nfft=nfft,
        detrend="constant", scaling="spectrum",
    )
    candidates = np.linspace(40.0, 320.0, 56001)  # 0.005 BPM spacing.
    scores = sum(
        np.interp(candidates * harmonic / 60, frequencies, power,
                  left=0, right=0) / harmonic**2
        for harmonic in range(1, 5)
    )
    # Return the strongest pulse, including subdivisions, without rounding to
    # integer BPM or using song metadata to choose a musical tempo level.
    return float(candidates[np.argmax(scores)])


def get_file_bpm(path, params=None):
    """Return [pulse BPM, decoded duration in seconds].

    params accepts a mapping or argparse namespace with mode, samplerate,
    win_s and hop_s overrides. Silence or fewer than four onsets yields 0 BPM;
    decoder errors propagate. The estimate describes a constant song tempo.
    """
    if params is None:
        params = {}
    elif not isinstance(params, Mapping):
        params = vars(params)

    presets = {"default": (44100, 1024, 512), "fast": (8000, 512, 128),
               "super-fast": (4000, 128, 64)}
    mode = params.get("mode", "default")
    if mode not in presets:
        raise ValueError(f"unknown mode {mode}")
    samplerate, win_s, hop_s = presets[mode]
    samplerate = params.get("samplerate", samplerate)
    win_s = params.get("win_s", win_s)
    hop_s = params.get("hop_s", hop_s)

    stream = source(str(path), samplerate, hop_s)
    samplerate = stream.samplerate
    blocks = []
    total_frames = 0
    try:
        while True:
            samples, read = stream()
            # Aubio reuses its buffer; ignore the padding in the final block.
            blocks.append(samples[:read].copy())
            total_frames += read
            if read < hop_s:
                break
    finally:
        stream.close()

    bpm = _estimate_bpm(np.concatenate(blocks), samplerate, win_s, hop_s)
    return [bpm, total_frames / samplerate]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',
                        help="mode [default|fast|super-fast]",
                        dest="mode", default='default')
    parser.add_argument('sources', nargs='+', help="input_files")
    args = parser.parse_args()
    for filename in args.sources:
        bpm, _ = get_file_bpm(filename, params=args)
        print(f"{bpm:.6f} {filename}")
