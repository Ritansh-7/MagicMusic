from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from user_store import UserStore


def test_login_accepts_legacy_and_new_password_hashes(tmp_path):
    store = UserStore(tmp_path / "events.sqlite3", database_url="")
    ok, user_id = store.register("new@example.com", "new-password")
    assert ok
    assert store.authenticate("new@example.com", "new-password")[0]

    salt = bytes.fromhex("11" * 16)
    digest = UserStore._digest_password("old-password", salt, UserStore._LEGACY_PASSWORD_ROUNDS)
    encoded = f"{salt.hex()}${digest}"
    assert UserStore._verify_password("old-password", encoded)
    assert not UserStore._verify_password("wrong-password", encoded)


def test_admin_allowlist_reads_streamlit_secrets(monkeypatch, tmp_path):
    import streamlit as st

    store = UserStore(tmp_path / "events.sqlite3", database_url="")
    ok, user_id = store.register("admin@example.com", "admin-password")
    assert ok

    monkeypatch.delenv("TUNEMATCH_ADMIN_EMAIL", raising=False)
    monkeypatch.setattr(
        st,
        "secrets",
        {"TUNEMATCH_ADMIN_EMAIL": "ADMIN@example.com"},
    )

    assert store.is_admin(user_id)


def test_user_events_are_scoped_to_the_requested_account(tmp_path):
    store = UserStore(tmp_path / "events.sqlite3", database_url="")
    first_ok, first_id = store.register("first@example.com", "first-password")
    second_ok, second_id = store.register("second@example.com", "second-password")
    assert first_ok and second_ok

    store.log_event(
        first_id,
        "track-one",
        "jio_play",
        "session-one",
        metadata_json='{"title": "First song"}',
    )
    store.log_event(
        second_id,
        "track-two",
        "jio_play",
        "session-two",
        metadata_json='{"title": "Second song"}',
    )

    events = store.user_events(first_id)
    assert len(events) == 1
    assert events[0]["track_id"] == "track-one"
    assert events[0]["metadata_json"] == '{"title": "First song"}'


def test_recommendation_context_returns_recent_plays_and_likes(tmp_path):
    store = UserStore(tmp_path / "events.sqlite3", database_url="")
    ok, user_id = store.register("listener@example.com", "listener-password")
    assert ok

    store.log_event(user_id, "one", "play", "session", metadata_json='{"title":"First"}')
    store.log_event(user_id, "two", "play", "session", metadata_json='{"title":"Second"}')
    store.log_event(user_id, "liked-track", "like", "session")

    context = store.recommendation_context(user_id, limit=1)
    assert len(context["recent_events"]) == 1
    assert context["recent_events"][0]["track_id"] == "two"
    assert context["preferred_track_ids"] == ["liked-track"]


def test_listening_time_summary_counts_only_listen_progress(tmp_path):
    store = UserStore(tmp_path / "events.sqlite3", database_url="")
    ok, user_id = store.register("listener@example.com", "listener-password")
    assert ok

    store.log_event(
        user_id, "track-one", "listen_progress", "session-one", duration_ms=120_000
    )
    store.log_event(
        user_id, None, "session_end", "session-one", duration_ms=300_000
    )

    assert store.summary(user_id)["listening_ms"] == 120_000
