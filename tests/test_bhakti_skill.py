import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bhakti_skill import is_strong_bhakti_seed


def test_film_song_with_explicit_spiritual_phrase_is_devotional():
    assert is_strong_bhakti_seed({
        "title": "Ye Toh Sach Hai Ke Bhagwan",
        "album": "Hum Saath Saath Hain",
    })


def test_single_deity_word_in_secular_title_is_not_enough():
    assert not is_strong_bhakti_seed({
        "title": "Bhagwan Mujhe Maaf Karna",
        "album": "A Bollywood Film",
    })


def test_explicit_bhajan_form_remains_devotional():
    assert is_strong_bhakti_seed({"title": "Shree Hanuman Chalisa"})
