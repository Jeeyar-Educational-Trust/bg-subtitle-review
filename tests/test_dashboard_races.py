"""Regression tests for the playlist dashboard's double-click races and step ordering.

No database is needed: db.py's calls are replaced by a small in-memory fake, and
YouTube / Azure by fakes, so these run in plain `pytest tests`. The database itself is
covered by test_youtube_dashboard.py (needs TEST_DATABASE_URL).
"""

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

CUES = [{"start": 0.5, "end": 2.5, "text": "నమస్కారం", "confidence": 0.9, "review": None},
        {"start": 3.0, "end": 5.0, "text": "భగవద్గీత", "confidence": 0.8, "review": None}]


@pytest.fixture
def env(tmp_path, monkeypatch):
    for key, value in {"APP_PASSWORD": "pw", "DATABASE_URL": "", "YOUTUBE_API_KEY": "key",
                       "JOBS_DIR": str(tmp_path / "jobs"),
                       "YOUTUBE_AUTO_SYNC_HOURS": "0", "YOUTUBE_MAX_CONCURRENT": "2"}.items():
        monkeypatch.setenv(key, value)  # DATABASE_URL="" keeps .env's real database out of reach
    for name in ("app", "db"):
        sys.modules.pop(name, None)
    import app

    lock = threading.Lock()
    state = {
        "videos": {"v1": {"video_id": "v1", "title": "Episode 1", "current_job_id": None,
                          "workflow_status": "not_started", "original_status": "unknown",
                          "telugu_approved_at": None, "telugu_approved_by": None,
                          "english_approved_at": None, "english_approved_by": None}},
        "artifacts": {}, "downloads": [], "delays": [], "cleared": [], "translate_error": None,
        "english": "English",
    }

    def get_video(vid):
        with lock:
            v = state["videos"].get(vid)
            return dict(v) if v else None

    def update_video(vid, conn=None, *, if_job=None, **fields):
        with lock:
            v = state["videos"][vid]
            if if_job is not None and v["current_job_id"] != if_job:
                return 0
            v.update(fields)
            return 1

    def link_job(job, video, status, audio_track, actor):
        time.sleep(0.3)  # a round trip to Neon: the window a double-click lands in
        with lock:
            state["videos"][video["video_id"]].update(
                current_job_id=job.id, workflow_status=status, telugu_approved_at=None,
                telugu_approved_by=None, english_approved_at=None, english_approved_by=None)

    def save_artifacts(job_id, video_id, **kw):
        with lock:
            state["artifacts"][job_id] = json.loads(json.dumps(kw))

    def get_artifacts(job_id):
        time.sleep(0.2)  # a round trip to Neon (the saved job text is the largest row)
        with lock:
            art = state["artifacts"].get(job_id)
            return dict(art, job_id=job_id) if art else None

    def clear_approvals(video_id, *, telugu, english, actor, reason):
        job_id = state["videos"][video_id]["current_job_id"]
        review = app.JOBS[job_id].path("review.json")
        state["cleared"].append({"english": english, "reason": reason,
                                 "review_then": review.read_text(encoding="utf-8") if review.exists() else None})
        update_video(video_id, english_approved_at=None, english_approved_by=None,
                     **({"telugu_approved_at": None, "telugu_approved_by": None} if telugu else {}))
        return True

    def fake_download(video_id, dest_dir, **kw):
        with lock:
            n = len(state["downloads"]) + 1
            state["downloads"].append(video_id)
            delay = state["delays"].pop(0) if state["delays"] else 0.05
        time.sleep(delay)
        path = Path(dest_dir) / "upload.m4a"
        path.write_bytes(f"audio-{n}".encode())
        return path, "140 · te · Telugu original", {}

    def fake_speech(src, dest_dir):
        out = Path(dest_dir) / "speech.flac"
        out.write_bytes(b"flac")
        return out

    def fake_translate(cues, **kw):
        if state["translate_error"]:
            raise RuntimeError(state["translate_error"])
        rows = [{"cue": c.number, "start": c.start, "end": c.end, "telugu": c.telugu,
                 "english": f"{state['english']} {c.number}", "correction": "", "review": None} for c in cues]
        return {"brief": "Brief.", "rows": rows, "all_items": rows}

    monkeypatch.setattr(app, "_require_youtube", lambda: None)
    monkeypatch.setattr(app, "_link_job", link_job)
    monkeypatch.setattr(app, "_ffmpeg_or_none", lambda: None)
    for name, fn in {"ready": lambda: True, "get_video": get_video, "update_video": update_video,
                     "save_artifacts": save_artifacts, "get_artifacts": get_artifacts,
                     "clear_approvals": clear_approvals, "save_draft": lambda *a, **k: None,
                     "log_event": lambda *a, **k: None}.items():
        monkeypatch.setattr(app.db, name, fn)
    monkeypatch.setattr(app.youtube, "download_original_audio", fake_download)
    monkeypatch.setattr(app.transcribe, "make_compact_audio",
                        lambda src, dest: Path(dest).write_bytes(Path(src).read_bytes()))
    monkeypatch.setattr(app.transcribe, "make_speech_audio", fake_speech)
    monkeypatch.setattr(app.transcribe, "transcribe_audio", lambda *a, **k: [dict(c) for c in CUES])
    monkeypatch.setattr(app.pipeline, "translate_cues", fake_translate)
    monkeypatch.setattr(app.pipeline, "load_skill_text", lambda _p: "skill")
    yield app, state
    for name in ("app", "db"):
        sys.modules.pop(name, None)


def wait_for(cond, timeout=10):
    deadline = time.time() + timeout
    while not cond():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def wait_step(app, state, job_id):
    """Until the step has ended and its result has been copied to the (fake) database."""
    job = app.JOBS[job_id]
    wait_for(lambda: job.state not in app.BUSY_STATES
             and (state["artifacts"].get(job_id) or {}).get("meta", {}).get("progress") == job.progress)
    return job


def generate(app, state):
    job_id = app.youtube_generate("v1", x_reviewer="", _=None)["id"]
    return wait_step(app, state, job_id)


def translate(app, job, model=""):
    return app.start_translation(job.id, chunk_size=40, overlap=3, model=model, x_reviewer="", _=None)


def in_threads(fn, n=2):
    results = []

    def run():
        try:
            results.append(("ok", fn()))
        except Exception as err:  # HTTPException from the route
            results.append((getattr(err, "status_code", repr(err)), getattr(err, "detail", "")))

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_double_click_generate_starts_one_job(env):
    app, state = env
    results = in_threads(lambda: app.youtube_generate("v1", x_reviewer="", _=None)["id"])
    started = [r[1] for r in results if r[0] == "ok"]
    assert len(started) == 1, results
    assert [r[0] for r in results if r[0] != "ok"] == [409]
    wait_step(app, state, started[0])
    assert state["downloads"] == ["v1"]  # one download, one transcription
    assert state["videos"]["v1"]["current_job_id"] == started[0]
    assert state["videos"]["v1"]["workflow_status"] == "telugu_review"


def test_double_click_open_rebuilds_the_job_once(env):
    app, state = env
    job_id = generate(app, state).id
    with app._LOCK:  # what the 7-day cleanup does
        shutil.rmtree(app.JOBS.pop(job_id).dir)
    state["downloads"].clear()

    results = in_threads(lambda: app.youtube_open("v1", _=None))
    assert sorted(r[1]["restored"] for r in results) == [False, True], results
    job = app.JOBS[job_id]
    wait_for(lambda: not job.audio_pending)
    assert job.has_audio and state["downloads"] == ["v1"]
    on_disk = json.loads(job.path("job.json").read_text(encoding="utf-8"))
    assert on_disk["has_audio"] is True and on_disk["progress"] == job.progress  # memory and disk agree
    assert not list(app.JOBS_DIR.glob(".refetch-*"))


def test_orphaned_audio_download_never_touches_the_rebuilt_job(env):
    app, state = env
    job_id = generate(app, state).id
    with app._LOCK:
        shutil.rmtree(app.JOBS.pop(job_id).dir)
    state["downloads"].clear()
    state["delays"] = [0.8, 0.1]  # the first rebuild's download is the slow one

    app.youtube_open("v1", _=None)
    first = app.JOBS[job_id]
    assert app.delete_job(job_id, _=None) == {"ok": True}  # deleted while its audio still downloads
    app.youtube_open("v1", _=None)  # ...and rebuilt again
    second = app.JOBS[job_id]
    assert second is not first
    wait_for(lambda: not first.audio_pending and not second.audio_pending)

    assert second.has_audio
    assert second.path("audio.mp3").read_bytes() == b"audio-2"  # not overwritten by the orphaned download
    on_disk = json.loads(second.path("job.json").read_text(encoding="utf-8"))
    assert on_disk["progress"] == second.progress
    assert not list(app.JOBS_DIR.glob(".refetch-*"))


def test_job_deleted_during_audio_download_is_not_recreated(env):
    app, state = env
    job_id = generate(app, state).id
    with app._LOCK:
        shutil.rmtree(app.JOBS.pop(job_id).dir)
    state["delays"] = [0.4]
    app.youtube_open("v1", _=None)
    job = app.JOBS[job_id]
    app.delete_job(job_id, _=None)
    wait_for(lambda: not job.audio_pending)
    assert not job.dir.exists()
    assert not list(app.JOBS_DIR.glob(".refetch-*"))


def test_rebuilt_manual_upload_says_the_player_audio_came_from_youtube(env):
    app, state = env
    job = generate(app, state)
    with app._LOCK:  # make it look like "Upload audio instead": no YouTube track recorded
        job.options.pop("audio_track")
        job.save()
    app._persist(job)
    with app._LOCK:
        shutil.rmtree(app.JOBS.pop(job.id).dir)
    app.youtube_open("v1", _=None)
    rebuilt = app.JOBS[job.id]
    wait_for(lambda: not rebuilt.audio_pending)
    assert rebuilt.has_audio
    assert rebuilt.options["player_audio"] == "140 · te · Telugu original"
    assert "transcript was made from the uploaded file" in rebuilt.progress[-1]
    assert app._youtube_public(rebuilt)["player_audio"] == "140 · te · Telugu original"


def test_failed_retranslation_keeps_the_english_approval(env):
    app, state = env
    job = generate(app, state)
    state["videos"]["v1"]["telugu_approved_at"] = "2026-11-25T00:00:00Z"
    translate(app, job)
    wait_step(app, state, job.id)
    state["videos"]["v1"].update(english_approved_at="2026-11-25T01:00:00Z", english_approved_by="Ravi")
    state["cleared"].clear()
    before = job.path("review.json").read_text(encoding="utf-8")

    state["translate_error"] = "Azure OpenAI returned 404 for deployment 'typo'"
    translate(app, job, model="typo")
    wait_step(app, state, job.id)
    assert "404" in job.error
    assert state["cleared"] == []  # the approval stays: the English did not change
    assert state["videos"]["v1"]["english_approved_by"] == "Ravi"
    assert job.path("review.json").read_text(encoding="utf-8") == before
    assert job.options.get("model") is None  # the failed attempt's deployment did not stick

    state["translate_error"], state["english"] = None, "New English"
    translate(app, job)
    wait_step(app, state, job.id)
    assert [c["reason"] for c in state["cleared"]] == ["re-translated"]
    assert state["cleared"][0]["review_then"] == before  # withdrawn before the new English was written
    assert state["videos"]["v1"]["english_approved_by"] is None
    assert "New English 1" in job.path("review.json").read_text(encoding="utf-8")


def test_save_says_english_is_stale_only_when_it_is(env):
    app, state = env
    job = generate(app, state)
    state["videos"]["v1"]["telugu_approved_at"] = "2026-11-25T00:00:00Z"
    translate(app, job)
    wait_step(app, state, job.id)
    save = lambda cues: app.save_source(job.id, cues=cues, x_reviewer="", _=None)["translation_stale"]

    cues = app.get_source(job.id, _=None)
    assert save(cues) is False  # saved without a change
    cues[0]["text"] += " x"
    assert save(cues) is True
    cues[0]["text"] = cues[0]["text"][:-2]
    assert save(cues) is False  # back to the Telugu the English was made from


def test_approval_refuses_text_the_reviewer_has_not_seen(env, monkeypatch):
    app, state = env
    job = generate(app, state)
    approved = []
    monkeypatch.setattr(app.db, "approve", lambda vid, lang, texts, job_id, reviewer: approved.append(lang))
    approve = lambda stage, seen: app.approve_job(job.id, body={"stage": stage, "reviewer": "A", "seen": seen}, _=None)
    telugu_page = lambda: [[c["start"], c["end"], c["text"]] for c in app.get_source(job.id, _=None)]

    page_a = telugu_page()  # reviewer A opens the job
    cues = app.get_source(job.id, _=None)  # reviewer B changes a cue and saves
    cues[0]["text"] += " x"
    app.save_source(job.id, cues=cues, x_reviewer="", _=None)
    with pytest.raises(app.HTTPException) as err:
        approve("telugu", page_a)  # A approves from the page loaded before B's edit
    assert err.value.status_code == 409 and "Reload" in err.value.detail
    with pytest.raises(app.HTTPException):
        approve("telugu", None)  # a page from before this check sends nothing: refused too
    assert approved == []
    approve("telugu", telugu_page())  # after reloading
    assert approved == ["te"]

    state["videos"]["v1"]["telugu_approved_at"] = "2026-11-25T00:00:00Z"
    translate(app, job)
    wait_step(app, state, job.id)
    page_a = [[r["cue"], r["english"], r["correction"]] for r in app.get_review(job.id, _=None)]
    app.save_review(job.id, edits=[{"cue": 1, "correction": "Greetings"}], x_reviewer="", _=None)
    with pytest.raises(app.HTTPException) as err:
        approve("english", page_a)
    assert err.value.status_code == 409
    approve("english", [[r["cue"], r["english"], r["correction"]] for r in app.get_review(job.id, _=None)])
    assert approved == ["te", "en"]


def test_superseded_job_does_not_move_the_video(env):
    app, state = env
    job = generate(app, state)
    state["videos"]["v1"]["current_job_id"] = "a-newer-job"
    app._set_workflow("v1", job.id, "failed")
    app._persist(job)
    assert state["videos"]["v1"]["workflow_status"] == "telugu_review"


def test_non_ascii_password_is_rejected_not_a_server_error(env):
    app, _ = env
    with pytest.raises(app.HTTPException) as err:
        app.require_password("caf\xe9")
    assert err.value.status_code == 401
    assert app._same_secret("pw", "pw") and not app._same_secret("caf\xe9", "pw")
