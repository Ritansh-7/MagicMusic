from __future__ import annotations


def analyze(audio_path: str) -> dict:
    try:
        import essentia
        import essentia.standard as es
    except Exception:
        return {}
    try:
        audio = es.MonoLoader(filename=audio_path)()
        rhythm = es.RhythmExtractor2013()(audio)
        key = es.KeyExtractor()(audio)
    except Exception:
        return {}
    return {
        "bpm": float(rhythm[0]) if rhythm else 0.0,
        "key": str(key[0]) if key else "",
        "scale": str(key[1]) if len(key) > 1 else "",
    }
