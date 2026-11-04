import os
import json
import re

MODEL_NAME = os.getenv("HF_MODEL_NAME", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
HF_TOKEN = os.getenv("HF_TOKEN")

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-generation",
            model=MODEL_NAME,
            token=HF_TOKEN,
            device_map="auto",
        )
    except Exception:
        _pipeline = False
    return _pipeline

def classify_text(text: str) -> dict:
    pipe = _get_pipeline()
    if not pipe:
        return {}
    prompt = (
        "Classify this song title/artist text into JSON with keys: "
        "intent, mood, genre, theme, religion, era. Text: " + text
    )
    try:
        output = pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        match = re.search(r"\{.*\}", output, re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}