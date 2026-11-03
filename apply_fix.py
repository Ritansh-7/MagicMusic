import re

with open('recommender.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update _infer_candidate_intents to normalize punctuation
old_infer = '''    text = " ".join(
        str(item.get(field, ""))
        for field in (
            "title", "album", "artists", "track_genre", "genre", "language", "mood",
            "description", "tags", "intent_labels", "candidate_source",
        )
    ).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))'''

new_infer = '''    text = " ".join(
        str(item.get(field, ""))
        for field in (
            "title", "album", "artists", "track_genre", "genre", "language", "mood",
            "description", "tags", "intent_labels", "candidate_source",
        )
    ).casefold()
    text = re.sub(r"[-_/]+", " ", text)
    tokens = set(re.findall(r"[a-z0-9]+", text))'''

text = text.replace(old_infer, new_infer)

# 2. Update INTENT_KEYWORDS rap_hiphop and energetic
old_rap_kw = '''    "energetic": {"fight", "fighting", "anthem", "rock", "dance", "party", "bhangra", "dhamaka", "dhamaal", "energetic", "power", "action", "high energy", "kala chashma", "mashup", "dj chetas", "club", "house party", "party hits", "club mix", "tony kakkar"},
    "acoustic_indie": {"acoustic", "unplugged", "indie", "folk", "coke studio", "singer songwriter", "nilotpal bora", "prateek kuhad", "anuv jain", "jasleen royal", "lucky ali", "ballad", "guitar"},
    "rap_hiphop": {"rap", "hip hop", "hiphop", "freestyle", "gully", "mixtape", "hustle", "mtv hustle", "aditya pushkarna"},'''

new_rap_kw = '''    "energetic": {"fight", "fighting", "anthem", "rock", "dance", "party", "bhangra", "dhamaka", "dhamaal", "energetic", "power", "action", "high energy", "kala chashma", "mashup", "dj chetas", "club", "house party", "party hits", "club mix", "tony kakkar", "dhaakad", "jugnu", "banger", "josh high", "energy"},
    "acoustic_indie": {"acoustic", "unplugged", "indie", "folk", "coke studio", "singer songwriter", "nilotpal bora", "prateek kuhad", "anuv jain", "jasleen royal", "lucky ali", "ballad", "guitar", "hussain haidry", "tripling"},
    "rap_hiphop": {"rap", "hip hop", "hiphop", "freestyle", "gully", "mixtape", "hustle", "mtv hustle", "aditya pushkarna", "divine", "raftaar", "badshah", "naezy", "mc stan", "gully boy", "apna time aayega", "azadi", "mere gully mein", "emiway", "krsna", "seedhe maut", "ikka"},'''

text = text.replace(old_rap_kw, new_rap_kw)

# 3. Update double_verify_candidate to block rap/energetic for mellow indie/soul seeds
old_party_guard = '''    # 3. Party / Aggressive Rap vs Mellow Indie/Acoustic/Romantic Guard
    party_tokens = [
        "kala chashma", "dj chetas", "house party", "party hits", "club mix",
        "hustle", "mtv hustle", "tony kakkar"
    ]
    is_cand_party_or_rap = (
        "rap_hiphop" in candidate_intents
        or any(w in cand_text for w in party_tokens)
        or "mashup" in cand_title
    )
    is_seed_mellow_soul = bool(
        {"romantic", "calm", "acoustic_indie", "motivational"}.intersection(seed_intents)
        or any(w in seed_text for w in ["pyaar", "ishq", "love", "romantic", "soothing", "lofi", "zindagi", "mauka", "aspirants"])
    )
    if is_cand_party_or_rap and is_seed_mellow_soul and not {"energetic", "rap_hiphop"}.intersection(seed_intents):
        return False, "Party/aggressive rap track conflicts with mellow acoustic/inspirational vibe"'''

new_party_guard = '''    # 3. Party / Aggressive Rap vs Mellow Indie/Acoustic/Romantic Guard
    party_or_rap_artists = ["divine", "raftaar", "badshah", "naezy", "mc stan", "emiway", "krsna", "seedhe maut", "tony kakkar"]
    party_or_rap_tokens = [
        "kala chashma", "dj chetas", "house party", "party hits", "club mix",
        "hustle", "mtv hustle", "tony kakkar", "dhaakad", "jugnu", "banger",
        "josh high", "gully boy", "apna time aayega", "mere gully mein", "azadi"
    ]
    is_cand_party_or_rap = (
        "rap_hiphop" in candidate_intents
        or "energetic" in candidate_intents
        or any(w in cand_text for w in party_or_rap_tokens)
        or any(w in cand_artists for w in party_or_rap_artists)
        or "mashup" in cand_title
    )
    is_seed_mellow_soul = bool(
        {"romantic", "calm", "acoustic_indie"}.intersection(seed_intents)
        or ("motivational" in seed_intents and "acoustic_indie" in seed_intents)
        or any(w in seed_text for w in ["pyaar", "ishq", "love", "romantic", "soothing", "lofi", "zindagi", "mauka", "aspirants", "dhaaga"])
    )
    if is_cand_party_or_rap and is_seed_mellow_soul and not {"energetic", "rap_hiphop"}.intersection(seed_intents):
        return False, "Party/aggressive rap track conflicts with mellow acoustic/inspirational vibe"'''

text = text.replace(old_party_guard, new_party_guard)

with open('recommender.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated recommender.py successfully!')
