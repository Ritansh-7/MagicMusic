from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_ranking import (
    include_all_results,
    prioritize_title_matches,
    title_query_match_score,
)


def test_title_match_accepts_small_vowel_transliteration_difference():
    assert title_query_match_score(
        "Saraswati Yeh Tera (Yeh Teraa Ghar Yeh Meraa Ghar / Soundtrack Version)",
        "yeh tera ghar",
    ) > 0.9


def test_title_match_rejects_partial_unrelated_token_hits():
    query = "yeh tera ghar"
    assert title_query_match_score("Mujhe Kya Pata Tera Ghar", query) < 0.82
    assert title_query_match_score("Ghar", query) < 0.82
    assert title_query_match_score("Ye Mera Jaha", query) < 0.82


def test_title_match_accepts_exact_song_titles_and_releases():
    assert title_query_match_score("Yeh Tera Ghar Yeh Mera Ghar", "yeh tera ghar") == 1.0
    assert title_query_match_score("Ghar (From Jab Harry Met Sejal)", "ghar") == 1.0


def test_song_title_is_prioritized_over_related_title_token_matches():
    results = [
        {"title": "Ye Mera Jaha"},
        {"title": "Ghar"},
        {"title": "Mujhe Kya Pata Tera Ghar"},
        {
            "title": (
                "Saraswati Yeh Tera "
                "(Yeh Teraa Ghar Yeh Meraa Ghar / Soundtrack Version)"
            )
        },
    ]

    ordered = prioritize_title_matches(results, "yeh tera ghar")

    assert ordered[0] is results[3]
    assert ordered[1:] == results[:3]


def test_include_all_results_preserves_ranked_order_and_appends_unranked_items():
    ranked = [{"title": "Exact match"}, {"title": "Related result"}]
    all_results = [ranked[0], ranked[1], {"title": "Other result"}]

    ordered = include_all_results(ranked, all_results, lambda item: item["title"])

    assert ordered == [*ranked, all_results[2]]
