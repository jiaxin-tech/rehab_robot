import base64
from datetime import timedelta
import hashlib
import pytest

from scripts.readonly_lab_broker import Session, prepare, read_json, save_json, utc_now


def setup_session(tmp_path):
    directory = tmp_path / "session"
    prepare(directory)
    session = Session(directory)
    session.hello({"hostname": "OFFLINE_TEST_OLD_HOST"})
    return session


def test_authentication_binds_peer_and_token(tmp_path):
    s = setup_session(tmp_path)
    assert s.authorized("192.168.137.2", s.config["token"])
    assert not s.authorized("192.168.137.3", s.config["token"])
    assert not s.authorized("192.168.137.2", "incorrect")


def test_only_known_actions_and_successful_probe_before_capture(tmp_path):
    s = setup_session(tmp_path)
    for action, seconds in [("shell", None), ("capture", 600), ("probe", 10), ("capture", 10)]:
        with pytest.raises(ValueError):
            s.queue(action, seconds)
    task = s.queue("probe")
    assert s.deliver() == task
    assert s.deliver() is None  # No repeat acquisition after a lost response.
    s.receive(task["task_id"], dict(ok=True, result={}, files=[]))
    assert s.queue("capture", 10)["duration_s"] == 10


def test_failed_probe_latches_session_and_keeps_cost(tmp_path):
    s = setup_session(tmp_path)
    task = s.queue("probe")
    s.deliver()
    s.receive(task["task_id"], dict(ok=False, error="actual read failed", files=[]))
    with pytest.raises(ValueError, match="latched"):
        s.queue("probe")
    assert len(s.status()["tasks"]) == 1 and s.status()["failed"]


def test_upload_rejects_unknown_paths_and_wrong_hash_before_writes(tmp_path):
    s = setup_session(tmp_path)
    task = s.queue("probe")
    s.deliver()
    for name, digest in [("../outside.py", None), ("probe.json", "0" * 64)]:
        item = dict(name=name, content_base64=base64.b64encode(b"{}").decode())
        if digest:
            item["sha256"] = digest
        with pytest.raises(ValueError):
            s.receive(task["task_id"], dict(ok=True, files=[item]))
    assert not (s.directory / "outside.py").exists()
    item = dict(name="probe.json", content_base64=base64.b64encode(b"{}").decode(),
                sha256=hashlib.sha256(b"{}").hexdigest())
    s.receive(task["task_id"], dict(ok=True, result={}, files=[item]))
    with pytest.raises(ValueError):
        s.receive(task["task_id"], dict(ok=True, files=[]))


def test_successful_capture_requires_raw_file_and_expired_session_stops(tmp_path):
    s = setup_session(tmp_path)
    task = s.queue("probe")
    s.deliver()
    s.receive(task["task_id"], dict(ok=True, files=[]))
    capture = s.queue("capture", 30)
    s.deliver()
    with pytest.raises(ValueError, match="raw data"):
        s.receive(capture["task_id"], dict(ok=True, files=[]))
    s.config["expires_at"] = (utc_now() - timedelta(seconds=1)).isoformat()
    assert not s.active()
    assert s.deliver() is None
    with pytest.raises(ValueError, match="expired"):
        s.queue("probe")


def test_malformed_upload_entries_rejected_without_partial_files(tmp_path):
    s = setup_session(tmp_path)
    task = s.queue("probe")
    s.deliver()
    for item in (None, [], "not-an-object"):
        with pytest.raises(ValueError, match="must be an object"):
            s.receive(task["task_id"], dict(ok=True, files=[item]))
    assert not (s.directory / "tasks" / task["task_id"] / "result.json").exists()
