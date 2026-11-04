import re

def generate_search_concepts(profile: dict) -> list[str]:
    concepts = []
    intent = profile.get("primary_intent", "unknown")
    theme = profile.get("theme", "unknown")
    language = profile.get("language", "")
    era = profile.get("release_decade") or profile.get("era", "")
    style = profile.get("style", [])
    style_text = " ".join(style) if isinstance(style, list) else str(style)
    if intent != "unknown":
        concepts.append(" ".join(x for x in [language, theme, intent, style_text] if x and x != "unknown").strip())
    if theme != "unknown":
        concepts.append(" ".join(x for x in [language, theme, "bhajan" if intent == "devotional" else "song", era] if x and x != "unknown").strip())
    if era and era != "unknown":
        concepts.append(" ".join(x for x in [language, theme, "from", era] if x and x != "unknown").strip())
    return [c for c in dict.fromkeys(concepts) if c]
EOF
python -m py_compile candidate_generator.py; if ($?) { echo "compile_ok" }