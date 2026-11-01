#!/usr/bin/env bash
# tools/ensure_betahub.sh
# Ensures the Tami 70th Family Beta Hub (local port 7070 + Cloudflare quick tunnel)
# runs detached in its own session (PPID=1, file-backed logs, zero PTY dependency)
# and automatically replaces stale/expired *.trycloudflare.com quick tunnels whose
# edge DNS registration has dropped (NXDOMAIN / HTTP 000 / 530).
#
# Usage:
#   ~/code/tami/tools/ensure_betahub.sh          # ensure server + public tunnel are HTTP 200 right now
#   ~/code/tami/tools/ensure_betahub.sh --quiet  # silent when healthy (used by 60s LaunchAgent watchdog)
#   ~/code/tami/tools/ensure_betahub.sh --force-new-tunnel  # force-rotate quick tunnel URL

set -euo pipefail

PROJECT_ROOT="/Users/talsheaffer/code/tami"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

exec "${VENV_PYTHON}" -u - "$@" <<'PY'
from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class BetaHubSupervisorConfig:
    project_root: str = "/Users/talsheaffer/code/tami"
    venv_python: str = "/Users/talsheaffer/code/tami/.venv/bin/python"
    serve_script: str = "/Users/talsheaffer/code/tami/prototype/serve.py"
    cloudflared_bin: str = "/opt/homebrew/bin/cloudflared"
    port: int = 7070
    metrics_addr: str = "127.0.0.1:20241"
    cloudflare_edge_probe_url: str = "https://cloudflare.com/cdn-cgi/trace"
    local_timeout_sec: float = 3.0
    public_timeout_sec: float = 6.0
    local_startup_budget_sec: float = 8.0
    tunnel_startup_budget_sec: float = 25.0
    poll_interval_sec: float = 0.75
    hard_alarm_sec: int = 45


def _http_status(url: str, timeout_sec: float) -> int:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tami-betahub-supervisor/1.0", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return 0


def _dns_resolves(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        return len(infos) > 0
    except OSError:
        return False


def _get_quicktunnel_hostname(cfg: BetaHubSupervisorConfig) -> str | None:
    url = f"http://{cfg.metrics_addr}/quicktunnel"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        host = str(payload.get("hostname") or "").strip()
        return host or None
    except Exception:
        return None


def _kill_matching_processes(pattern: str) -> None:
    subprocess.run(
        ["/usr/bin/pkill", "-f", pattern],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=4.0,
        check=False,
    )


def _kill_port_listeners(port: int) -> None:
    res = subprocess.run(
        ["/usr/sbin/lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=4.0,
        check=False,
        text=True,
    )
    pids = [int(p) for p in res.stdout.split() if p.strip().isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(0.4)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _backend_sources_max_mtime(cfg: BetaHubSupervisorConfig) -> float:
    proto_dir = pathlib.Path(cfg.project_root) / "prototype"
    watched = [
        proto_dir / "serve.py",
        proto_dir / "beta_config.py",
        proto_dir / "feedback_store.py",
        proto_dir / "assets" / "scenes" / "scene_manifest.json",
    ]
    return max((p.stat().st_mtime for p in watched if p.exists()), default=0.0)


def _start_detached_server(cfg: BetaHubSupervisorConfig, state_dir: pathlib.Path) -> None:
    out = open(state_dir / "serve.log", "ab", buffering=0)
    err = open(state_dir / "serve.err.log", "ab", buffering=0)
    devnull = open(os.devnull, "rb")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.Popen(
        [
            cfg.venv_python,
            "-u",
            cfg.serve_script,
            "--port",
            str(cfg.port),
            "--host",
            "127.0.0.1",
            "--diag-log",
            str(state_dir / "mobile_diag.jsonl"),
        ],
        cwd=cfg.project_root,
        stdin=devnull,
        stdout=out,
        stderr=err,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    (state_dir / "serve.started_ts").write_text(f"{time.time():.6f}\n", encoding="utf-8")


def _restart_tunnel_process(cfg: BetaHubSupervisorConfig, state_dir: pathlib.Path) -> None:
    # Stop any stray terminal-spawned or hung cloudflared targeting port 7070 / metrics 20241
    _kill_matching_processes(rf"cloudflared.*127\.0\.0\.1:{cfg.port}")
    _kill_matching_processes(r"cloudflared.*127\.0\.0\.1:20241")
    time.sleep(0.5)

    # If the macOS LaunchAgent `com.tami.betahub.tunnel` is loaded, kickstart it cleanly;
    # otherwise spawn our own detached cloudflared session.
    uid = os.getuid()
    ks = subprocess.run(
        ["/bin/launchctl", "kickstart", "-k", f"gui/{uid}/com.tami.betahub.tunnel"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
        check=False,
    )
    if ks.returncode == 0:
        return

    out = open(state_dir / "cloudflared.log", "ab", buffering=0)
    err = open(state_dir / "cloudflared.err.log", "ab", buffering=0)
    devnull = open(os.devnull, "rb")
    subprocess.Popen(
        [
            cfg.cloudflared_bin,
            "tunnel",
            "--no-autoupdate",
            "--metrics",
            cfg.metrics_addr,
            "--url",
            f"http://127.0.0.1:{cfg.port}",
        ],
        cwd=cfg.project_root,
        stdin=devnull,
        stdout=out,
        stderr=err,
        start_new_session=True,
        close_fds=True,
    )


def _check_public_tunnel_healthy(host: str, cfg: BetaHubSupervisorConfig) -> tuple[bool, int]:
    if not _dns_resolves(host):
        return False, 0
    code = _http_status(f"https://{host}/", cfg.public_timeout_sec)
    return (code == 200), code


def main(argv: list[str]) -> int:
    cfg = BetaHubSupervisorConfig()
    signal.alarm(cfg.hard_alarm_sec)

    quiet = "--quiet" in argv
    force_new_tunnel = "--force-new-tunnel" in argv

    state_dir = pathlib.Path.home() / ".local" / "share" / "tami-betahub"
    state_dir.mkdir(parents=True, exist_ok=True)
    tunnel_url_file = state_dir / "tunnel_url.txt"
    local_url = f"http://127.0.0.1:{cfg.port}"

    repaired = False

    # 1. Ensure local server on port 7070 responds HTTP 200 AND runs up-to-date backend code
    started_ts_file = state_dir / "serve.started_ts"
    started_ts = 0.0
    if started_ts_file.exists():
        try:
            started_ts = float(started_ts_file.read_text(encoding="utf-8").strip())
        except ValueError:
            started_ts = 0.0
    code_stale = _backend_sources_max_mtime(cfg) > started_ts + 0.5

    local_code = _http_status(f"{local_url}/", cfg.local_timeout_sec)
    if local_code != 200 or code_stale or "--restart-server" in argv:
        repaired = True
        # Always terminate any orphan/detached serve.py holding --port 7070 BEFORE
        # kickstarting LaunchAgent so the new process binds 7070 rather than falling back to 7071.
        _kill_matching_processes(rf"prototype/serve\.py.*--port {cfg.port}([[:space:]]|$)")
        _kill_port_listeners(cfg.port)
        time.sleep(0.3)
        uid = os.getuid()
        ks_srv = subprocess.run(
            ["/bin/launchctl", "kickstart", "-k", f"gui/{uid}/com.tami.betahub.server"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
        if ks_srv.returncode == 0:
            started_ts_file.write_text(f"{time.time():.6f}\n", encoding="utf-8")
        else:
            _start_detached_server(cfg, state_dir)
        deadline = time.monotonic() + cfg.local_startup_budget_sec
        while time.monotonic() < deadline:
            time.sleep(cfg.poll_interval_sec)
            local_code = _http_status(f"{local_url}/", cfg.local_timeout_sec)
            if local_code == 200:
                break

    # 2. Ensure Cloudflare quick tunnel is BOTH registered locally AND publicly reachable (DNS + HTTP 200)
    host = None if force_new_tunnel else _get_quicktunnel_hostname(cfg)
    pub_ok = False
    pub_code = 0

    if host is not None:
        pub_ok, pub_code = _check_public_tunnel_healthy(host, cfg)
        if not pub_ok:
            # Confirm once more after a brief pause before rotating URL
            time.sleep(0.8)
            pub_ok, pub_code = _check_public_tunnel_healthy(host, cfg)

    # Rotate/restart tunnel if missing, forced, or publicly dead while Cloudflare edge is reachable
    if not pub_ok:
        edge_code = _http_status(cfg.cloudflare_edge_probe_url, 4.0)
        if host is None or force_new_tunnel or edge_code == 200:
            repaired = True
            old_host = host
            _restart_tunnel_process(cfg, state_dir)
            deadline = time.monotonic() + cfg.tunnel_startup_budget_sec
            while time.monotonic() < deadline:
                time.sleep(cfg.poll_interval_sec)
                candidate = _get_quicktunnel_hostname(cfg)
                if not candidate or (old_host and candidate == old_host):
                    continue
                host = candidate
                pub_ok, pub_code = _check_public_tunnel_healthy(host, cfg)
                if pub_ok:
                    break

    tunnel_url = f"https://{host}/" if host else ""
    if tunnel_url:
        tunnel_url_file.write_text(tunnel_url + "\n", encoding="utf-8")

    local_status = "HTTP 200 OK" if local_code == 200 else f"DOWN (HTTP {local_code:03d})"
    pub_status = "HTTP 200 OK" if pub_ok else f"UNREACHABLE (HTTP {pub_code:03d})"

    if not quiet or repaired or local_code != 200 or not pub_ok:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print("====================================================================")
        print(f"Tami 70th Family Beta Hub — Detached Persistent Status [{ts}]")
        if repaired:
            print("  Action       : Repaired / rotated stale service or tunnel")
        print(f"  Local Server : {local_url}/ ({local_status})")
        print(f"  Public URL   : {tunnel_url or 'unavailable'} ({pub_status})")
        if tunnel_url:
            base = re.sub(r"/+$", "", tunnel_url)
            print(f"  Wiki Portal  : {base}/wiki")
            print(f"  Photo Catalog: {base}/catalog")
        print(f"  Logs         : {state_dir}/serve.err.log")
        print("====================================================================")

    return 0 if (local_code == 200 and pub_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
