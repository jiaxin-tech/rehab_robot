"""Local operator commands for offline checks and bounded read-only ADS capture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
from uuid import uuid4

from scripts import readonly_lab_client as readonly

ROOT = Path(__file__).resolve().parents[1]
PROBE_MAX_AGE_S = 600


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", timeout=10, check=False)
    if result.returncode != 0:
        raise RuntimeError("无法记录当前Git版本，请先在完整代码仓库中运行。")
    return result.stdout.strip()


def offline_check(config_path):
    """Read files and runtime facts only; never construct an ADS session."""
    config_path = Path(config_path).resolve(strict=True)
    config = readonly.validate_config(readonly.read_json(config_path))
    checks = dict(windows=os.name == "nt", python_64bit=struct.calcsize("P") == 8,
                  dll_exists=Path(config["ads_dll_path"]).is_file(), configuration_valid=True)
    return config, dict(ok=all(checks.values()), checks=checks,
                        python_version=sys.version.split()[0], config_path=str(config_path),
                        config_sha256=sha256_file(config_path), hostname=socket.gethostname(),
                        configured_ams_net_id=config["ams_net_id"], device_connected=False)


def validate_probe_session(path, *, config_sha256, hostname, now=None):
    path = Path(path).resolve(strict=True)
    manifest = readonly.read_json(path / "manifest.json")
    status = readonly.read_json(path / "segment_status.json")
    if (manifest.get("action") != "probe" or manifest.get("status") != "COMPLETE"
            or manifest.get("ok") is not True or status.get("ok") is not True
            or status.get("status") != "COMPLETE"):
        raise ValueError("采集前必须提供成功的probe目录，失败记录不能用于放行。")
    if manifest.get("config_sha256") != config_sha256 or manifest.get("hostname") != hostname:
        raise ValueError("probe不属于本机当前配置，请在本机用同一配置重新probe。")
    if status.get("actual_local_ams_net_id") != readonly.EXPECTED_MAPPING["ams_net_id"] or status.get("toggle_changes", 0) <= 0:
        raise ValueError("probe未证明实际AMS一致及输入toggle变化。")
    completed = datetime.fromisoformat(status["completed_utc"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc) if now is None else now
    if completed.tzinfo is None or not 0 <= (now-completed).total_seconds() <= PROBE_MAX_AGE_S:
        raise ValueError("probe已超过10分钟或时间不一致，请重新probe。")
    expected_sources = {"manual_leg_check.py": sha256_file(Path(__file__)),
                        "readonly_lab_client.py": sha256_file(Path(readonly.__file__))}
    if manifest.get("source_sha256") != expected_sources:
        raise ValueError("只读采集代码已变更，请使用当前代码重新probe。")
    if manifest.get("segment_status_sha256") != sha256_file(path / "segment_status.json"):
        raise ValueError("probe状态文件与记录的SHA不一致。")
    return path


def run(action, *, config_path, output_dir=None, live=False, duration_s=None,
        probe_session=None, executor=readonly.execute_task):
    """Preserve every attempt in a new folder; return its manifest and status."""
    if action not in ("check", "probe", "capture"):
        raise ValueError("只支持check、probe、capture。")
    config_path = Path(config_path).resolve()
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+action+"_"+uuid4().hex[:8]
    output = Path(output_dir).resolve() if output_dir else ROOT / "results/manual_lab" / suffix
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("输出目录已存在，禁止覆盖或重放，请使用新目录：" + str(output))
    manifest = dict(schema_version=1, action=action, status="STARTING", ok=False,
                    hostname=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
                    live_requested=bool(live), motion_commanded=False, remote_helper_used=False,
                    duration_s=duration_s, config_path=str(config_path),
                    source_sha256={"manual_leg_check.py": sha256_file(Path(__file__)),
                                   "readonly_lab_client.py": sha256_file(Path(readonly.__file__))})
    # execute_task owns creation of the task directory; early failures create it
    # here, while every successful live task writes evidence before returning.
    try:
        manifest["git_head"] = git_head()
        config, check = offline_check(config_path)
        manifest.update(config_sha256=check["config_sha256"], offline_check=check)
        if not check["ok"]:
            raise RuntimeError("离线检查未通过：" + json.dumps(check["checks"], ensure_ascii=False))
        if action == "check":
            output.mkdir(exist_ok=False)
            result = dict(ok=True, result=check, error=None)
        else:
            if live is not True:
                raise ValueError("probe和capture会读取真实设备，必须显式加--live。")
            task = dict(task_id=suffix.replace(".", "_"), action=action)
            if action == "capture":
                if type(duration_s) is not int or duration_s not in (10, 30):
                    raise ValueError("采集时长只能为10或30秒。")
                if probe_session is None:
                    raise ValueError("采集必须给出--probe-session成功probe目录。")
                probe = validate_probe_session(probe_session, config_sha256=manifest["config_sha256"], hostname=manifest["hostname"])
                manifest["probe_session"] = str(probe)
                task["duration_s"] = duration_s
            result = executor(task, config, output)
        manifest.update(status="COMPLETE" if result["ok"] else "FAILED", ok=result["ok"],
                        error=result.get("error"), result=result.get("result"))
    except Exception as exc:
        manifest.update(status="FAILED", ok=False, error=f"{type(exc).__name__}: {exc}")
        output.mkdir(parents=True, exist_ok=True)
    except KeyboardInterrupt:
        manifest.update(status="FAILED", ok=False, error="KeyboardInterrupt: 操作员取消，本次不自动重试。")
        output.mkdir(parents=True, exist_ok=True)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    if (output / "segment_status.json").is_file():
        manifest["segment_status_sha256"] = sha256_file(output / "segment_status.json")
    manifest["output_files_sha256"] = {name: sha256_file(output / name)
        for name in ("leg.csv", "segment_status.json") if (output / name).is_file()}
    readonly.write_json(output / "manifest.json", manifest)
    return output, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="本机假腿只读检查与短时采集，不连接机器人，不发送运动或ADS写指令。")
    parser.add_argument("action", choices=("check", "probe", "capture"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/twincat_leg_capture.json")
    parser.add_argument("--output-dir", type=Path, help="必须不存在的新目录；默认results/manual_lab/时间戳动作")
    parser.add_argument("--live", action="store_true", help="明确允许本次只读ADS读取")
    parser.add_argument("--duration-s", type=int, choices=(10, 30))
    parser.add_argument("--probe-session", type=Path)
    args = parser.parse_args(argv)
    output, manifest = run(args.action, config_path=args.config, output_dir=args.output_dir,
                           live=args.live, duration_s=args.duration_s, probe_session=args.probe_session)
    print("记录目录：" + str(output))
    if manifest["ok"]:
        print("离线检查通过，未连接设备。" if args.action == "check" else "只读任务完成；原始数据和状态已保存。")
        if args.action == "probe":
            print("后续采集请使用此目录作为--probe-session，10分钟内有效。")
        return 0
    print("本次已停止，未自动重试。原因：" + str(manifest.get("error")), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
