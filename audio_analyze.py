import requests, hashlib, json, os
import librosa

def analyze_url(url, out_json="audio_features.json"):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    data = r.content
    h = hashlib.sha256(data).hexdigest()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        path = f.name
    y, sr = librosa.load(path, sr=22050, mono=True, duration=120)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float((y**2).mean()**0.5)
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
    feats = {"audio_hash": h, "tempo_bpm": float(tempo), "rms_energy": rms, "zcr": zcr, "spectral_centroid": spectral_centroid, "mfcc": mfcc}
    with open(out_json, "w") as f:
        json.dump(feats, f)
    os.remove(path)
    return feats

if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    print(analyze_url(url))