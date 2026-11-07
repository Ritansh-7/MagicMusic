from __future__ import annotations

import numpy as np

try:
    import librosa
except Exception:
    librosa = None


def analyze(audio_path: str) -> dict:
    if librosa is None:
        return {}
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception:
        return {}
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    except Exception:
        return {}
    return {
        "tempo": float(tempo) if tempo is not None else 0.0,
        "rms_energy": float(np.mean(rms)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "zero_crossing_rate": float(np.mean(zcr)),
        "chroma_mean": [float(x) for x in np.mean(chroma, axis=1)],
        "mfcc_mean": [float(x) for x in np.mean(mfcc, axis=1)],
    }
