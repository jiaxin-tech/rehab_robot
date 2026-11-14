"""Temporary, peer-bound delivery of two read-only leg acquisition operations.

This broker never runs client-supplied commands. The old computer makes outbound
HTTP requests; there is no desktop, shell, credential or motion interface.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

MAX_UPLOAD = 16 * 1024 * 1024
MAX_FILES = 10 * 1024 * 1024
ALLOWED_FILES = {"leg.csv", "segment_status.json", "quality.json",
                 "probe.json", "capture_summary.json", "host_info.json"}


def utc_now():
    return datetime.now(timezone.utc)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


class Session:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.config = read_json(self.directory / "connection.json")
        self.lock = threading.RLock()

    def active(self):
        return (utc_now() < datetime.fromisoformat(self.config["expires_at"])
                and not (self.directory / "stop_requested").exists())

    def authorized(self, peer, token):
        return (peer == self.config["client_ip"] and isinstance(token, str)
                and hmac.compare_digest(token, self.config["token"]))

    def status(self):
        with self.lock:
            jobs = []
            for folder in sorted((self.directory / "tasks").glob("*")):
                request = read_json(folder / "request.json")
                jobs.append(dict(task_id=request["task_id"], action=request["action"],
                                 delivered=(folder / "delivered.json").is_file(),
                                 result=read_json(folder / "result.json")
                                 if (folder / "result.json").is_file() else None))
            return dict(active=self.active(),
                        host=read_json(self.directory / "hello.json")
                        if (self.directory / "hello.json").is_file() else None,
                        failed=(self.directory / "failure_latched.json").is_file(),
                        tasks=jobs, expires_at=self.config["expires_at"],
                        scope="READ_ONLY_LEG_PROBE_AND_BOUNDED_CAPTURE",
                        shell_or_motion_available=False)

    def queue(self, action, duration=None):
        with self.lock:
            if action not in ("probe", "capture") or (action == "probe" and duration is not None):
                raise ValueError("only probe or bounded capture is supported")
            if action == "capture" and (type(duration) is not int or duration not in (10, 30)):
                raise ValueError("capture duration must be exactly 10 or 30 seconds")
            if not self.active():
                raise ValueError("session has expired or stopped")
            status = self.status()
            if not status["host"]:
                raise ValueError("wait for the old computer to connect")
            if status["failed"]:
                raise ValueError("failed task is latched; inspect its records")
            if any(job["result"] is None for job in status["tasks"]):
                raise ValueError("a task is already queued or in progress")
            if action == "capture" and not any(
                    job["action"] == "probe" and job["result"]
                    and job["result"]["ok"] is True for job in status["tasks"]):
                raise ValueError("capture requires a successful read-only probe in this session")
            task = dict(task_id=uuid.uuid4().hex, action=action,
                        expires_at=self.config["expires_at"], created_at=utc_now().isoformat())
            if action == "capture":
                task["duration_s"] = duration
            folder = self.directory / "tasks" / task["task_id"]
            folder.mkdir()
            save_json(folder / "request.json", task)
            return task

    def deliver(self):
        with self.lock:
            if not self.active() or (self.directory / "failure_latched.json").exists():
                return None
            for folder in sorted((self.directory / "tasks").glob("*")):
                if not (folder / "delivered.json").exists() and not (folder / "result.json").exists():
                    task = read_json(folder / "request.json")
                    # Never issue a physical acquisition twice after a lost response.
                    save_json(folder / "delivered.json", dict(time=utc_now().isoformat()))
                    return task
            return None

    def hello(self, value):
        if not isinstance(value, dict) or not isinstance(value.get("hostname"), str):
            raise ValueError("host report must contain hostname")
        with self.lock:
            target = self.directory / "hello.json"
            if target.exists() and read_json(target)["hostname"] != value["hostname"]:
                raise ValueError("session is already paired to a different host")
            save_json(target, dict(value, received_at=utc_now().isoformat()))

    def receive(self, task_id, value):
        if not isinstance(task_id, str) or len(task_id) != 32 or any(
                char not in "0123456789abcdef" for char in task_id):
            raise ValueError("unknown task identity")
        folder = self.directory / "tasks" / task_id
        with self.lock:
            if not (folder / "delivered.json").is_file() or (folder / "result.json").exists():
                raise ValueError("task was not delivered or already completed")
            if (not isinstance(value, dict) or type(value.get("ok")) is not bool
                    or not isinstance(value.get("files", []), list)):
                raise ValueError("invalid result schema")
            files, seen, total = [], set(), 0
            for item in value.get("files", []):
                if not isinstance(item, dict):
                    raise ValueError("result file must be an object")
                name = item.get("name")
                if name not in ALLOWED_FILES or name in seen:
                    raise ValueError("unexpected or duplicate result filename")
                payload = base64.b64decode(item["content_base64"], validate=True)
                total += len(payload)
                if total > MAX_FILES:
                    raise ValueError("result exceeds the bounded acquisition size")
                digest = hashlib.sha256(payload).hexdigest()
                if item.get("sha256", digest) != digest:
                    raise ValueError("result file hash mismatch")
                files.append((name, payload, digest))
                seen.add(name)
            task = read_json(folder / "request.json")
            if value["ok"] and task["action"] == "capture" and not {
                    "leg.csv", "segment_status.json"}.issubset(seen):
                raise ValueError("successful capture requires raw data and completion evidence")
            for name, payload, digest in files:
                with (folder / name).open("xb") as stream:
                    stream.write(payload)
            result = {key: value.get(key) for key in ("ok", "result", "error")}
            result.update(received_at=utc_now().isoformat(),
                          files={name: dict(sha256=digest, bytes=len(payload))
                                 for name, payload, digest in files})
            save_json(folder / "result.json", result)
            if not value["ok"]:
                save_json(self.directory / "failure_latched.json",
                          dict(task_id=task_id, error=value.get("error")))
            return result


def handler_for(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not print the connection token in URL access logs.

        def reply(self, code, value, content_type="application/json; charset=utf-8"):
            body = (json.dumps(value, ensure_ascii=False).encode("utf-8")
                    if content_type.startswith("application/json") else value)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def allowed_peer(self):
            return self.client_address[0] in (
                session.config["client_ip"], session.config["server_ip"])

        def do_GET(self):
            parsed = urlsplit(self.path)
            if not self.allowed_peer():
                return self.reply(403, {"error": "peer is not part of this session"})
            if not session.active():
                return self.reply(410, {"error": "session stopped or expired"})
            if parsed.path == "/":
                return self.reply(200, (session.directory / "index.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if parsed.path == "/download":
                self.send_response(200)
                payload = (session.directory / "readonly_leg_helper.zip").read_bytes()
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="readonly_leg_helper.zip"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/health":
                return self.reply(200, dict(active=True, scope="READ_ONLY_LEG_ACQUISITION"))
            token = self.headers.get("X-Lab-Token")
            if not session.authorized(self.client_address[0], token):
                return self.reply(403, {"error": "session authentication failed"})
            if parsed.path == "/task":
                return self.reply(200, session.deliver())
            self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self):
            parsed = urlsplit(self.path)
            if not session.active():
                return self.reply(410, {"error": "session stopped or expired"})
            if not session.authorized(self.client_address[0], self.headers.get("X-Lab-Token")):
                return self.reply(403, {"error": "session authentication failed"})
            try:
                self.connection.settimeout(10)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    return self.reply(413, {"error": "invalid or excessive result size"})
                value = json.loads(self.rfile.read(length))
                if parsed.path == "/hello":
                    session.hello(value)
                    return self.reply(200, {"accepted": True})
                if parsed.path == "/result":
                    task_id = parse_qs(parsed.query).get("task_id", [""])[0]
                    session.receive(task_id, value)
                    return self.reply(200, {"accepted": True})
                return self.reply(404, {"error": "unknown endpoint"})
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    return Handler


def prepare(directory, server_ip="192.168.137.1", client_ip="192.168.137.2", port=8765):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tasks").mkdir()
    config = dict(server_url=f"http://{server_ip}:{port}", server_ip=server_ip, client_ip=client_ip,
                  token=secrets.token_hex(32), expires_at=(utc_now() + timedelta(minutes=30)).isoformat())
    save_json(directory / "connection.json", config)
    return config


def serve(directory):
    session = Session(directory)
    port = urlsplit(session.config["server_url"]).port
    server = ThreadingHTTPServer((session.config["server_ip"], port), handler_for(session))
    server.daemon_threads = True
    server.timeout = 1
    print("Read-only leg helper ready at " + session.config["server_url"], flush=True)
    try:
        while session.active():
            server.handle_request()
    finally:
        server.server_close()
        save_json(session.directory / "stopped.json", dict(time=utc_now().isoformat()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "serve", "queue", "status", "stop"))
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--action", choices=("probe", "capture"))
    parser.add_argument("--duration-s", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.session_dir)
        print(json.dumps(dict(server_url=result["server_url"], expires_at=result["expires_at"])))
    elif args.command == "serve":
        serve(args.session_dir)
    else:
        session = Session(args.session_dir)
        if args.command == "queue":
            print(json.dumps(session.queue(args.action, args.duration_s)))
        elif args.command == "stop":
            (session.directory / "stop_requested").touch()
            print("Session stop requested")
        else:
            print(json.dumps(session.status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
