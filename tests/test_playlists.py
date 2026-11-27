"""The dashboard's playlist tabs: each tab reads and syncs its own playlist.

No database is needed (db.py's calls are replaced by small fakes, YouTube by a fake),
so these run in plain `pytest tests`. Two playlists sharing the real database are
covered by test_youtube_dashboard.py (needs TEST_DATABASE_URL).
"""

import re
import sys

import pytest

PW = {"X-App-Password": "pw"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for key, value in {"APP_PASSWORD": "pw", "DATABASE_URL": "", "YOUTUBE_API_KEY": "key",
                       "JOBS_DIR": str(tmp_path / "jobs"), "YOUTUBE_AUTO_SYNC_HOURS": "0"}.items():
        monkeypatch.setenv(key, value)  # DATABASE_URL="" keeps .env's real database out of reach
    for name in ("app", "db"):
        sys.modules.pop(name, None)
    import app

    calls = {"fetched": [], "listed": [], "started": []}

    def fetch(playlist_id, api_key):
        calls["fetched"].append(playlist_id)
        return {"playlist_title": f"Title of {playlist_id}", "videos": [], "skipped": 0}

    monkeypatch.setattr(app.youtube, "fetch_playlist", fetch)
    for name, fn in {"ready": lambda: True,
                     "current_subtitles": lambda pid: [],
                     "last_syncs": lambda pid: {"latest": None, "ok": None},
                     "list_videos": lambda pid: calls["listed"].append(pid) or [],
                     "start_sync": lambda pid, actor: calls["started"].append(pid) or 1,
                     "finish_sync": lambda *a, **k: None,
                     "upsert_videos": lambda pid, videos: [],
                     "mark_removed": lambda pid, ids: 0,
                     "log_events": lambda *a, **k: None}.items():
        monkeypatch.setattr(app.db, name, fn)

    from fastapi.testclient import TestClient

    with TestClient(app.app) as client:
        yield client, app, calls
    for name in ("app", "db"):
        sys.modules.pop(name, None)


def test_configured_playlists_are_valid(env):
    _, app, _ = env
    keys = [p["key"] for p in app.PLAYLISTS]
    ids = [p["playlist_id"] for p in app.PLAYLISTS]
    assert len(set(keys)) == len(keys) and len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[\w-]+", k) for k in keys)  # the frontend routes #youtube/<key>
    assert not any(i.startswith("VL") for i in ids)  # a /show/VL... link's ID without its "VL"


def test_login_lists_the_tabs(env):
    client, app, _ = env
    data = client.post("/api/login", headers=PW).json()
    assert [(p["key"], p["label"]) for p in data["playlists"]] == [("ch3", "Chapter 3"), ("ch4", "Chapter 4")]
    assert data["playlists"][1]["url"] == f"https://www.youtube.com/playlist?list={app.PLAYLISTS[1]['playlist_id']}"


def test_each_tab_reads_and_syncs_its_own_playlist(env):
    client, app, calls = env
    ch3, ch4 = (p["playlist_id"] for p in app.PLAYLISTS)
    res = client.get("/api/youtube/videos?playlist=ch4", headers=PW)
    assert res.status_code == 200, res.text
    assert (res.json()["playlist"], res.json()["playlist_id"], res.json()["playlist_label"]) == ("ch4", ch4, "Chapter 4")
    assert client.get("/api/youtube/videos", headers=PW).json()["playlist"] == "ch3"  # no key: the first tab
    assert calls["listed"] == [ch4, ch3]
    res = client.post("/api/youtube/sync?playlist=ch4", headers=PW)
    assert res.status_code == 200 and res.json()["playlist_title"] == f"Title of {ch4}"
    assert calls["fetched"] == calls["started"] == [ch4]


def test_unknown_playlist_is_404(env):
    client, _, calls = env
    for res in (client.get("/api/youtube/videos?playlist=ch9", headers=PW),
                client.post("/api/youtube/sync?playlist=ch9", headers=PW)):
        assert res.status_code == 404 and "ch9" in res.json()["detail"]
    assert calls["fetched"] == calls["listed"] == []


def test_a_running_sync_blocks_only_its_own_playlist(env):
    client, app, calls = env
    ch3, ch4 = (p["playlist_id"] for p in app.PLAYLISTS)
    with app._sync_lock(ch3):  # a Chapter 3 sync is running
        res = client.post("/api/youtube/sync?playlist=ch3", headers=PW)
        assert res.status_code == 409 and "Chapter 3" in res.json()["detail"]
        assert client.post("/api/youtube/sync?playlist=ch4", headers=PW).status_code == 200
    assert calls["fetched"] == [ch4]
