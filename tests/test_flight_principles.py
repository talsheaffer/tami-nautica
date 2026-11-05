#!/usr/bin/env python3
r"""
tests/test_flight_principles.py — Automated 3D Portal Flight & Scene Transition Invariant Gate

Enforces the 5 mandatory laws recorded in
`.agents/skills/coding-guidelines/references/flight_transition_principles.md`:
  1. Law 1 (Pure 3D Flight — Zero Distortion/Flash Effects): `TRANSITION_FRAGMENT_SHADER` and
     portal flight pipelines contain zero synthetic `ripple`, `directionalWarp`, or `specularFlash`
     distortions.
  2. Law 2 ($C^2$ Camera Kinematics & Rotated-Frustum `effectiveFitZ` Registration): arrival pose
     accounts for yaw/pitch/roll so 100% of the rotated frustum lies strictly inside the portal
     aperture with zero 1-frame clip-plane snaps, and UI transitions between portal-connected
     scenes execute true 3D portal flight.
  3. Law 3 (Chromatic & Radiometric Continuity): host `layer0` never fades toward `clearColor`
     (`uFadeToClear == 0.0`), overhanging front rims dissolve strictly via physical aperture fill
     ratio $\rho(Z_{\text{cam}})$, and `uTint` interpolates smoothly across $t \in [0, 1]$.
  4. Law 4 (Eager Advance Bundle & GPU Prewarming): outgoing direct portals, occluders, occluder
     depth maps, and veils are tracked in shared URL textures and prewarmed in advance;
     `updatePortalShelves` is idempotent across viewport scaling.
  5. Law 5 (Bijective Scene State Isomorphism): companion-fall positions AND 3D Euler rotations
     (`scroll_black_rod` at `-95°`), depth-displaced child occluders (`scene17_scroll_foot_occluder`),
     live dissolve animations, carpet rigs, and active-visitor occluder lock deferrals stay 100%
     consistent between standalone scenes and embedded dioramas.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import unittest
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR: Path = REPO_ROOT / "prototype"
SCENE_MANAGER_JS: Path = PROTOTYPE_DIR / "js" / "scene_manager.js"
MANIFEST_JSON: Path = PROTOTYPE_DIR / "assets" / "scenes" / "scene_manifest.json"
FLIGHT_PRINCIPLES_DOC: Path = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "coding-guidelines"
    / "references"
    / "flight_transition_principles.md"
)
SKILL_DOC: Path = REPO_ROOT / ".agents" / "skills" / "coding-guidelines" / "SKILL.md"
RULES_DOC: Path = REPO_ROOT / ".agents" / "rules" / "coding_guidelines.md"
CHROME_BIN: str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


@dataclass(frozen=True)
class FlightTestThresholds:
    """Canonical numerical tolerances and timeouts for flight transition invariant tests."""

    desktop_width: int = 1440
    desktop_height: int = 900
    desktop_fov_deg: float = 42.0
    mobile_width: int = 390
    mobile_height: int = 844
    mobile_fov_deg: float = 56.0
    mobile_portrait_yaw_deg: float = 7.5
    node_timeout_sec: float = 8.0
    cdp_connect_timeout_sec: float = 14.0
    cdp_socket_timeout_sec: float = 12.0
    cdp_ready_poll_timeout_sec: float = 2.0
    cdp_heavy_eval_timeout_sec: float = 90.0
    layers_ready_timeout_sec: float = 120.0
    proc_wait_timeout_sec: float = 3.0
    base_http_port: int = 7520
    base_cdp_port: int = 9520
    expected_rod_fall_deg: float = -95.0
    rotation_rad_tol: float = 1e-4
    position_world_tol: float = 1e-3
    tint_tol: float = 2e-3
    unclipped_plane_sentinel: float = 1e5
    active_visitor_regen_probe_ms: int = 30
    active_visitor_wait_sec: float = 0.12


THRESHOLDS = FlightTestThresholds()


class TestFlightPrinciplesStaticAndPureMath(unittest.TestCase):
    """Static specification, shader purity, and pure frustum math verification."""

    def test_flight_principles_documented_and_wired(self) -> None:
        self.assertTrue(
            FLIGHT_PRINCIPLES_DOC.is_file(),
            f"Missing canonical flight transition principles doc: {FLIGHT_PRINCIPLES_DOC}",
        )
        doc_text = FLIGHT_PRINCIPLES_DOC.read_text(encoding="utf-8")
        for required_phrase in (
            "Law 1 — Pure 3D Camera Flight",
            "Law 2 — $C^2$ Camera Kinematics",
            "Law 3 — Chromatic & Radiometric Continuity",
            "Law 4 — Eager Advance Bundle & GPU Prewarming",
            "Law 5 — Bijective Scene State Isomorphism",
        ):
            self.assertIn(required_phrase, doc_text)

        skill_text = SKILL_DOC.read_text(encoding="utf-8")
        self.assertIn("references/flight_transition_principles.md", skill_text)
        self.assertIn("tests/test_flight_principles.py", skill_text)

        rules_text = RULES_DOC.read_text(encoding="utf-8")
        self.assertIn("flight_transition_principles.md", rules_text)
        self.assertIn("tests/test_flight_principles.py", rules_text)

    def test_transition_shader_pure_flight_no_distortion_effects(self) -> None:
        js_text = SCENE_MANAGER_JS.read_text(encoding="utf-8")
        match = re.search(
            r"const\s+TRANSITION_FRAGMENT_SHADER\s*=\s*`([^`]*)`",
            js_text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "Could not locate TRANSITION_FRAGMENT_SHADER in scene_manager.js")
        shader_body = match.group(1) if match else ""

        for forbidden_token in (
            "ripple",
            "directionalWarp",
            "specularFlash",
            "uDisplacementScale",
            "uPointer",
            "sin(dist",
        ):
            self.assertNotIn(
                forbidden_token,
                shader_body,
                f"Forbidden shader distortion token '{forbidden_token}' found in TRANSITION_FRAGMENT_SHADER",
            )
            self.assertNotIn(
                forbidden_token,
                js_text,
                f"Forbidden distortion uniform/symbol '{forbidden_token}' still present in scene_manager.js",
            )

        # Ensure uTint exists in ROOM_DEPTH_FRAGMENT_SHADER and in every ROOM_DEPTH_FRAGMENT_SHADER material
        self.assertIn("uniform vec3 uTint;", js_text)
        shader_mat_blocks = js_text.split("fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER")
        self.assertGreaterEqual(len(shader_mat_blocks) - 1, 10)
        for idx, tail in enumerate(shader_mat_blocks[1:], start=1):
            window = tail[:1200]
            self.assertIn(
                "uTint:",
                window,
                f"ShaderMaterial #{idx} using ROOM_DEPTH_FRAGMENT_SHADER is missing uTint uniform",
            )

    def test_effective_fit_z_contains_rotated_mobile_and_rolled_frustums(self) -> None:
        node_bin = shutil.which("node")
        self.assertIsNotNone(node_bin, "node binary required for pure JS geometry verification")

        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        manifest_shelf_ids = [
            p["id"] for p in manifest["scenes"]["shelf_portal"]["portalShelves"]
        ]

        node_script = f"""
import {{
  SCENE_GEOMETRY_SPEC,
  computePortalWorldGeometry,
  computePortalEffectiveFitZ
}} from {json.dumps(SCENE_MANAGER_JS.as_uri())};
import fs from 'node:fs';

const src = fs.readFileSync({json.dumps(str(SCENE_MANAGER_JS))}, 'utf8');
const shelfBlock = src.match(/const DEFAULT_SHELF_PORTAL_SHELVES = Object\\.freeze\\(\\[(.*?)\\n\\]\\);/s);
if (!shelfBlock) {{
  throw new Error('Could not locate DEFAULT_SHELF_PORTAL_SHELVES in scene_manager.js');
}}
const defaultIds = [...shelfBlock[1].matchAll(/id:\\s*'([^']+)'/g)].map((m) => m[1]);

const aspectMobile = {THRESHOLDS.mobile_width} / {THRESHOLDS.mobile_height};
const fovMobile = {THRESHOLDS.mobile_fov_deg};
const geom = computePortalWorldGeometry(
  {{ innerBoxPx: [838.36, 431.55, 930.94, 521.36], worldZ: -284.0, recessDepth: 160.0 }},
  1.15,
  -44.0,
  380.0,
  -350.0,
  fovMobile,
  aspectMobile
);
const effMobileYaw = computePortalEffectiveFitZ(
  geom,
  fovMobile,
  aspectMobile,
  {THRESHOLDS.mobile_portrait_yaw_deg},
  0.0,
  0.0,
  0.0
);
const effDesktopZero = computePortalEffectiveFitZ(
  geom,
  {THRESHOLDS.desktop_fov_deg},
  {THRESHOLDS.desktop_width} / {THRESHOLDS.desktop_height},
  0.0,
  0.0,
  0.0,
  0.0
);

// Project all 4 frustum corners onto aperture plane z = geom.worldZ from z = geom.worldZ + effMobileYaw
const deg = Math.PI / 180.0;
const yaw = {THRESHOLDS.mobile_portrait_yaw_deg} * deg;
const cy = Math.cos(yaw);
const sy = Math.sin(yaw);
const tanY = Math.tan(0.5 * fovMobile * deg);
const tanX = tanY * aspectMobile;
let maxRatioX = 0;
let maxRatioY = 0;
for (const sx of [-1, 1]) {{
  for (const sySign of [-1, 1]) {{
    const vx = sx * tanX;
    const vy = sySign * tanY;
    const vz = -1.0;
    // YXZ with pitch=0, roll=0 -> Ry(yaw)
    const dx = cy * vx + sy * vz;
    const dy = vy;
    const dz = -sy * vx + cy * vz;
    const tHit = -effMobileYaw / dz;
    const hitX = dx * tHit;
    const hitY = dy * tHit;
    maxRatioX = Math.max(maxRatioX, Math.abs(hitX) / (0.5 * geom.width));
    maxRatioY = Math.max(maxRatioY, Math.abs(hitY) / (0.5 * geom.height));
  }}
}}

console.log(JSON.stringify({{
  defaultIds,
  fitZ: geom.fitZ,
  effMobileYaw,
  effDesktopZero,
  maxRatioX,
  maxRatioY,
  safetyRatio: SCENE_GEOMETRY_SPEC.portalApertureFitSafetyRatio
}}));
"""
        proc = subprocess.run(
            [node_bin or "node", "--input-type=module", "-e", node_script],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=THRESHOLDS.node_timeout_sec,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Node pure geometry evaluation failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        payload = json.loads(proc.stdout.strip())
        self.assertEqual(
            payload["defaultIds"],
            manifest_shelf_ids,
            "DEFAULT_SHELF_PORTAL_SHELVES order must match scene_manifest.json 1:1",
        )
        self.assertLess(
            payload["effMobileYaw"],
            payload["fitZ"],
            "Rotated mobile arrival yaw must pull effectiveFitZ closer than unrotated fitZ",
        )
        self.assertLessEqual(payload["maxRatioX"], 1.0)
        self.assertLessEqual(payload["maxRatioY"], 1.0)
        self.assertAlmostEqual(
            max(payload["maxRatioX"], payload["maxRatioY"]),
            payload["safetyRatio"],
            places=5,
        )

    def test_tzofen_window_keeps_front_bricks_and_frame_over_treetop_portal(self) -> None:
        """The Tzofen `fg` must open only the plaster seen through the window frame; the frame
        posts and the terracotta bricks standing in front of the opening stay opaque so they
        occlude the treetop portal (a hard rectangle cut once erased them)."""
        import numpy as np
        from PIL import Image

        from prototype.tzofen_tree_scenes_builder import default_tzofen_tree_bundle_config

        cfg = default_tzofen_tree_bundle_config().tzofen
        fg_path = PROTOTYPE_DIR / "assets" / "scenes" / f"{cfg.asset_prefix}_fg.png"
        if not fg_path.is_file():
            self.skipTest(f"Missing built Tzofen fg asset: {fg_path}")
        fg_alpha = np.asarray(Image.open(fg_path).convert("RGBA"))[:, :, 3].astype(np.float32)
        src = Image.open(REPO_ROOT / cfg.source_image_relpath).convert("RGB")
        src = src.resize((cfg.inner_dims.width, cfg.scaled_height_px), Image.Resampling.LANCZOS)
        rgb = np.zeros((cfg.inner_dims.height, cfg.inner_dims.width, 3), dtype=np.float32)
        y0 = cfg.inner_y_offset_px
        rgb[y0 : y0 + cfg.scaled_height_px] = np.asarray(src, dtype=np.float32)

        wx0, wy0, wx1, wy1 = (int(round(v)) for v in cfg.window_portal_inner_box_px)
        box_rgb = rgb[wy0:wy1, wx0:wx1]
        box_alpha = fg_alpha[wy0:wy1, wx0:wx1]
        r_minus_b = box_rgb[:, :, 0] - box_rgb[:, :, 2]
        luma = box_rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        brick = r_minus_b > 90.0
        wall = (luma > 238.0) & (r_minus_b > 8.0) & (r_minus_b < 20.0)
        self.assertGreater(int(brick.sum()), 500, "Expected terracotta bricks inside the window box")
        self.assertGreater(int(wall.sum()), 5000, "Expected plaster wall inside the window opening")
        self.assertGreater(
            float(np.median(box_alpha[brick])), 240.0, "Bricks in front of the window were cut out"
        )
        self.assertLess(
            float(np.median(box_alpha[wall])), 5.0, "Plaster seen through the window was not opened"
        )


class TestFlightPrinciplesWebGLRuntime(unittest.TestCase):
    """Empirical headless Chrome WebGL2 verification of flight continuity & scene state isomorphism."""

    server: Any = None
    http_port: int = 0
    server_thread: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(CHROME_BIN):
            raise unittest.SkipTest(f"Chrome binary not found at {CHROME_BIN}")

        from prototype.serve import find_free_port, start_server_thread

        cls.http_port = find_free_port(THRESHOLDS.base_http_port)
        cls.server, cls.http_port, cls.server_thread = start_server_thread(
            port=cls.http_port, directory=str(PROTOTYPE_DIR)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.server is not None:
            cls.server.shutdown()
            cls.server.server_close()

    def _launch_chrome_cdp(self, width: int, height: int) -> tuple[Any, ...]:
        from prototype.serve import find_free_port

        cdp_port = find_free_port(THRESHOLDS.base_cdp_port)
        tmp_dir = tempfile.mkdtemp(prefix="tami_flight_test_")
        url = f"http://127.0.0.1:{self.http_port}/index.html?fastLoad=1"
        cmd = [
            CHROME_BIN,
            "--headless=new",
            "--disable-extensions",
            f"--user-data-dir={tmp_dir}",
            f"--window-size={width},{height}",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            f"--remote-debugging-port={cdp_port}",
            url,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        sock: socket.socket | None = None
        try:
            tabs = None
            t0 = time.time()
            while time.time() - t0 < THRESHOLDS.cdp_connect_timeout_sec:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{cdp_port}/json", timeout=2.0
                    ) as resp:
                        cand = json.loads(resp.read())
                        if [t for t in cand if "index.html" in t.get("url", "")]:
                            tabs = cand
                            break
                except Exception:
                    time.sleep(0.2)

            self.assertIsNotNone(tabs, "Chrome CDP /json did not respond in time")
            target_tabs = [t for t in (tabs or []) if "index.html" in t.get("url", "")]
            ws_url = target_tabs[0]["webSocketDebuggerUrl"]
            host_port, ws_path = ws_url.replace("ws://", "").split("/", 1)
            host, port_str = host_port.split(":")

            sock = socket.create_connection((host, int(port_str)), timeout=8.0)
            handshake = (
                f"GET /{ws_path} HTTP/1.1\r\n"
                f"Host: {host}:{port_str}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            sock.sendall(handshake.encode("utf-8"))
            hs_resp = sock.recv(4096)
            self.assertIn(b"101", hs_resp)
        except Exception:
            if sock is not None:
                sock.close()
            proc.terminate()
            try:
                proc.wait(timeout=THRESHOLDS.proc_wait_timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        msg_counter = {"id": 1}

        def eval_js(
            expression: str,
            await_promise: bool = False,
            timeout_sec: float = THRESHOLDS.cdp_socket_timeout_sec,
        ) -> Any:
            assert sock is not None
            mid = msg_counter["id"]
            msg_counter["id"] += 1
            raw = json.dumps(
                {
                    "id": mid,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": await_promise,
                    },
                }
            ).encode("utf-8")
            frame = bytearray([0x81])
            n = len(raw)
            if n < 126:
                frame.append(0x80 | n)
            elif n < 65536:
                frame.append(0x80 | 126)
                frame.extend(struct.pack(">H", n))
            else:
                frame.append(0x80 | 127)
                frame.extend(struct.pack(">Q", n))
            mask_key = b"\x21\x43\x65\x87"
            frame.extend(mask_key)
            frame.extend(b ^ mask_key[i % 4] for i, b in enumerate(raw))
            sock.sendall(bytes(frame))

            sock.settimeout(timeout_sec)
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                hdr = sock.recv(2)
                if not hdr:
                    raise RuntimeError("CDP WebSocket closed unexpectedly")
                plen = hdr[1] & 0x7F
                if plen == 126:
                    plen = struct.unpack(">H", sock.recv(2))[0]
                elif plen == 127:
                    plen = struct.unpack(">Q", sock.recv(8))[0]
                buf = bytearray()
                while len(buf) < plen:
                    chunk = sock.recv(plen - len(buf))
                    if not chunk:
                        raise RuntimeError("CDP WebSocket truncated frame")
                    buf.extend(chunk)
                if (hdr[0] & 0x0F) == 0x8:
                    raise RuntimeError("CDP WebSocket received close frame")
                parsed = json.loads(buf.decode("utf-8", errors="replace"))
                if parsed.get("id") == mid:
                    if "exceptionDetails" in parsed.get("result", {}):
                        raise AssertionError(f"JS exception: {parsed['result']['exceptionDetails']}")
                    return parsed.get("result", {}).get("result", {}).get("value")
            raise TimeoutError(f"CDP command {mid} timed out")

        # Wait until initial scene layersReady
        t_ready = time.time()
        ready = False
        while time.time() - t_ready < THRESHOLDS.layers_ready_timeout_sec:
            # Headless SwiftShader can block the page's main thread for several seconds while
            # decoding/uploading the texture set, so a single unanswered poll is not a failure.
            try:
                if eval_js(
                    "Boolean(window.__PROTOTYPE_STATE__ && window.__PROTOTYPE_STATE__.layersReady)",
                    timeout_sec=THRESHOLDS.cdp_ready_poll_timeout_sec,
                ):
                    ready = True
                    break
            except (TimeoutError, socket.timeout):
                continue
            time.sleep(0.2)
        if not ready:
            # Close Chrome before failing, or every load timeout leaks a headless browser.
            self._close_chrome_cdp(proc, sock, tmp_dir)
        self.assertTrue(ready, "Prototype failed to reach layersReady within timeout")

        return proc, sock, tmp_dir, eval_js

    def _close_chrome_cdp(self, proc: Any, sock: Any, tmp_dir: str) -> None:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        proc.terminate()
        try:
            proc.wait(timeout=THRESHOLDS.proc_wait_timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_scroll_foot_flight_continuity_and_dropped_rod_state_consistency(self) -> None:
        proc, sock, tmp_dir, eval_js = self._launch_chrome_cdp(
            THRESHOLDS.desktop_width, THRESHOLDS.desktop_height
        )
        try:
            # 1. Verify bundle preloading includes outgoing portal purim_costume_1966 AND
            #    updatePortalShelves is idempotent after applyDynamicWidescreenScaling.
            prep_report = eval_js(
                """
                (async () => {
                  const sm = window.__tamiSceneManager;
                  await sm._loadSceneBundleReady('bene_beitcha_scroll_foot', false);
                  const reqIds = sm._getRequiredBundleSceneIds('bene_beitcha_scroll_foot');
                  const beforeInst = sm._portalInstances.find((p) => p.spec.id === 'cubby_scroll_foot');
                  const rawManifest = await fetch('assets/scenes/scene_manifest.json').then((r) => r.json());
                  sm.applyDynamicWidescreenScaling();
                  sm.updatePortalShelves(rawManifest.scenes.shelf_portal.portalShelves, 'shelf_portal');
                  const afterInst = sm._portalInstances.find((p) => p.spec.id === 'cubby_scroll_foot');
                  return {
                    reqIds,
                    purimReady: Boolean(sm.sceneObjects.purim_costume_1966.texturesReady),
                    sameCubbyInstanceAfterManifestSync: beforeInst === afterInst,
                    childOccUsesDepthShader: Boolean(
                      afterInst.childPortalMeshes[0] &&
                      afterInst.childPortalMeshes[0].childOccMat &&
                      afterInst.childPortalMeshes[0].childOccMat.uniforms &&
                      afterInst.childPortalMeshes[0].childOccMat.uniforms.uDepthMap
                    )
                  };
                })()
                """,
                await_promise=True,
                timeout_sec=THRESHOLDS.cdp_heavy_eval_timeout_sec,
            )
            self.assertIn("purim_costume_1966", prep_report["reqIds"])
            self.assertTrue(prep_report["purimReady"])
            self.assertTrue(
                prep_report["sameCubbyInstanceAfterManifestSync"],
                "updatePortalShelves tore down and rebuilt live cubbies despite identical canonical specs",
            )
            self.assertTrue(
                prep_report["childOccUsesDepthShader"],
                "Embedded diorama child occluder (ceramic foot) must use depth-displaced ShaderMaterial",
            )

            # 2. Launch forward flight from shelf_portal into bene_beitcha_scroll_foot via transitionTo()
            #    and sample frozenProgress across [0.0, 0.35, 0.78, 0.88, 0.96].
            fwd_samples = eval_js(
                """
                (() => {
                  const sm = window.__tamiSceneManager;
                  const started = sm.transitionTo('bene_beitcha_scroll_foot');
                  const isPortalFlight = Boolean(sm._portalFlight && !sm._portalFlight.reverse);
                  const cubbyInst = sm._portalInstances.find((p) => p.spec.id === 'cubby_scroll_foot');
                  const samples = [];
                  for (const fp of [0.0, 0.35, 0.78, 0.88, 0.96]) {
                    sm._portalFlight.frozenProgress = fp;
                    sm.animate();
                    cancelAnimationFrame(sm.animId);
                    const tint = cubbyInst.bgMat.uniforms.uTint.value;
                    samples.push({
                      t: fp,
                      camZ: sm.camera.position.z,
                      fadeToClear: sm.sceneObjects.shelf_portal.layer0.material.uniforms.uFadeToClear.value,
                      rimOp: sm.sceneObjects.shelf_portal.portalSurroundingOpacity,
                      maxClipConst: Math.max(...cubbyInst.clipPlanes.map((pl) => Math.abs(pl.constant))),
                      tint: [tint.r, tint.g, tint.b]
                    });
                  }
                  // Land at t = 1.0
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate();
                  cancelAnimationFrame(sm.animId);
                  return {
                    started,
                    isPortalFlight,
                    landedScene: sm.currentSceneId,
                    samples
                  };
                })()
                """,
                timeout_sec=THRESHOLDS.cdp_heavy_eval_timeout_sec,
            )
            self.assertTrue(fwd_samples["started"])
            self.assertTrue(
                fwd_samples["isPortalFlight"],
                "transitionTo('bene_beitcha_scroll_foot') from shelf_portal must use 3D _portalFlight",
            )
            self.assertEqual(fwd_samples["landedScene"], "bene_beitcha_scroll_foot")

            prev_blue = 0.0
            for sample in fwd_samples["samples"]:
                self.assertEqual(
                    sample["fadeToClear"],
                    0.0,
                    f"Host layer0 uFadeToClear darkened toward black at t={sample['t']}",
                )
                self.assertLess(
                    sample["maxClipConst"],
                    THRESHOLDS.unclipped_plane_sentinel,
                    f"Portal clip planes snapped off at t={sample['t']}",
                )
                self.assertGreaterEqual(
                    sample["tint"][2] + 1e-6,
                    prev_blue,
                    f"uTint blue channel failed monotonic warm->neutral progression at t={sample['t']}",
                )
                prev_blue = sample["tint"][2]

            self.assertAlmostEqual(
                fwd_samples["samples"][0]["tint"][2], 0.952, delta=THRESHOLDS.tint_tol
            )
            self.assertAlmostEqual(
                fwd_samples["samples"][-1]["tint"][2], 1.0, delta=0.01
            )

            # 3. Verify active-visitor occluder regen deferral while inside bene_beitcha_scroll_foot
            regen_deferred = eval_js(
                f"""
                (async () => {{
                  const sm = window.__tamiSceneManager;
                  const cubbySpec = sm._findPortalSpec('cubby_scroll_foot');
                  sm._scheduleOccluderRegen(cubbySpec, {THRESHOLDS.active_visitor_regen_probe_ms});
                  await new Promise((r) => setTimeout(r, {int(THRESHOLDS.active_visitor_wait_sec * 1000)}));
                  return {{
                    stillUnlockedWhileVisiting: !sm.isPortalLocked('cubby_scroll_foot'),
                    currentScene: sm.currentSceneId
                  }};
                }})()
                """,
                await_promise=True,
            )
            self.assertTrue(
                regen_deferred["stillUnlockedWhileVisiting"],
                "cubby_scroll_foot occluder regenerated and locked the cubby while visitor was inside target scene",
            )

            # 4. Drop the ceramic foot + black rod inside bene_beitcha_scroll_foot, then verify
            #    bijective 3D rotation & position consistency across reverse flight and resize!
            expected_rad = math.radians(THRESHOLDS.expected_rod_fall_deg)
            rod_report = eval_js(
                """
                (() => {
                  const sm = window.__tamiSceneManager;
                  sm.erasePortalOccluder('portal_scroll_foot_purim_1966');
                  // Advance dissolve animation past completion (500ms > 420ms)
                  const futureNow = performance.now() + 600.0;
                  sm._stepPortalOccluderAnimations(futureNow);

                  const standaloneRod = sm.sceneObjects.bene_beitcha_scroll_foot.extraLayers.find(
                    (el) => el.id === 'scroll_black_rod'
                  ).mesh;
                  const cubbyInst = sm._portalInstances.find((p) => p.spec.id === 'cubby_scroll_foot');
                  const dioramaRod = cubbyInst.extraMeshes.find(
                    (ex) => ex.sourceExtra.id === 'scroll_black_rod'
                  ).mesh;
                  const dioramaFoot = cubbyInst.childPortalMeshes[0].childOccMesh;

                  const beforeReverse = {
                    standaloneRotZ: standaloneRod.rotation.z,
                    dioramaRotZ: dioramaRod.rotation.z,
                    posDiff: standaloneRod.position.distanceTo(dioramaRod.position),
                    dioramaFootVisible: dioramaFoot.visible
                  };

                  // Trigger reverse flight back to shelf_portal via transitionTo('shelf_portal')
                  const revStarted = sm.transitionTo('shelf_portal');
                  const revIsPortalFlight = Boolean(sm._portalFlight && sm._portalFlight.reverse);
                  const revFlightRotations = [];
                  for (const fp of [0.02, 0.5, 0.95]) {
                    sm._portalFlight.frozenProgress = fp;
                    sm.animate();
                    cancelAnimationFrame(sm.animId);
                    revFlightRotations.push({
                      t: fp,
                      dioramaRotZ: dioramaRod.rotation.z,
                      posDiff: standaloneRod.position.distanceTo(dioramaRod.position),
                      dioramaFootVisible: dioramaFoot.visible,
                      maxClipConst: Math.max(...cubbyInst.clipPlanes.map((pl) => Math.abs(pl.constant))),
                      fadeToClear: sm.sceneObjects.shelf_portal.layer0.material.uniforms.uFadeToClear.value
                    });
                  }

                  // Complete reverse flight onto shelf_portal AND stress-test viewport resize
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate();
                  cancelAnimationFrame(sm.animId);
                  sm.applyDynamicWidescreenScaling();

                  const canvas = document.querySelector('canvas#spatial-canvas');
                  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

                  return {
                    beforeReverse,
                    revStarted,
                    revIsPortalFlight,
                    revFlightRotations,
                    afterLandingAndResize: {
                      currentScene: sm.currentSceneId,
                      standaloneRotZ: standaloneRod.rotation.z,
                      dioramaRotZ: dioramaRod.rotation.z,
                      posDiff: standaloneRod.position.distanceTo(dioramaRod.position),
                      dioramaFootVisible: dioramaFoot.visible,
                      restTintB: cubbyInst.bgMat.uniforms.uTint.value.b,
                      glError: gl ? gl.getError() : -1
                    }
                  };
                })()
                """,
                timeout_sec=THRESHOLDS.cdp_heavy_eval_timeout_sec,
            )

            self.assertAlmostEqual(
                rod_report["beforeReverse"]["standaloneRotZ"],
                expected_rad,
                delta=THRESHOLDS.rotation_rad_tol,
            )
            self.assertAlmostEqual(
                rod_report["beforeReverse"]["dioramaRotZ"],
                expected_rad,
                delta=THRESHOLDS.rotation_rad_tol,
                msg="Dropped black rod rotation in cubby diorama did not match standalone scene",
            )
            self.assertLessEqual(
                rod_report["beforeReverse"]["posDiff"], THRESHOLDS.position_world_tol
            )
            self.assertFalse(rod_report["beforeReverse"]["dioramaFootVisible"])

            self.assertTrue(rod_report["revStarted"])
            self.assertTrue(
                rod_report["revIsPortalFlight"],
                "transitionTo('shelf_portal') from bene_beitcha_scroll_foot must use reverse 3D _portalFlight",
            )

            for step in rod_report["revFlightRotations"]:
                self.assertAlmostEqual(
                    step["dioramaRotZ"],
                    expected_rad,
                    delta=THRESHOLDS.rotation_rad_tol,
                    msg=f"Black rod snapped vertical during reverse flight at t={step['t']}",
                )
                self.assertLessEqual(step["posDiff"], THRESHOLDS.position_world_tol)
                self.assertFalse(step["dioramaFootVisible"])
                self.assertEqual(step["fadeToClear"], 0.0)
                self.assertLess(step["maxClipConst"], THRESHOLDS.unclipped_plane_sentinel)

            after = rod_report["afterLandingAndResize"]
            self.assertEqual(after["currentScene"], "shelf_portal")
            self.assertAlmostEqual(
                after["dioramaRotZ"],
                expected_rad,
                delta=THRESHOLDS.rotation_rad_tol,
                msg="Black rod snapped vertical after landing on shelf_portal / resizing viewport",
            )
            self.assertLessEqual(after["posDiff"], THRESHOLDS.position_world_tol)
            self.assertFalse(after["dioramaFootVisible"])
            self.assertAlmostEqual(after["restTintB"], 0.952, delta=THRESHOLDS.tint_tol)
            self.assertEqual(after["glError"], 0)
        finally:
            self._close_chrome_cdp(proc, sock, tmp_dir)

    def test_wall_mounted_vitrine_flush_registration_and_nested_placement(self) -> None:
        """Vitrine hung rotated on Bar Giora's wall: diorama uses the wall-mount similarity, the
        flight lands exactly on its arrival camera pose (both directions), and the nested copy
        inside the shelf cubby sits at the same flush transform instead of full scene size."""
        proc, sock, tmp_dir, eval_js = self._launch_chrome_cdp(
            THRESHOLDS.desktop_width, THRESHOLDS.desktop_height
        )
        try:
            report = eval_js(
                """
                (async () => {
                  const sm = window.__tamiSceneManager;
                  await sm._loadSceneBundleReady('bar_giora_bedroom', false);
                  await sm._loadSceneBundleReady('vitrine_keys_pen', false);
                  sm._updatePortalShelfTransforms();
                  const inst = sm._portalInstances.find((p) => p.spec.id === 'portal_bar_giora_vitrine');
                  const wmx = inst.wallMountXform;
                  const cubby = sm._portalInstances.find((p) => p.spec.id === 'cubby_bar_giora');
                  const nested = cubby.childPortalMeshes
                    .map((cp) => cp.childWallMount)
                    .find((cwm) => cwm && cwm.childTargetId === 'vitrine_keys_pen');
                  const out = {
                    hasXform: Boolean(wmx),
                    groupPosErr: inst.dioramaGroup.position.distanceTo(wmx.tDio),
                    groupQuatDot: Math.abs(inst.dioramaGroup.quaternion.dot(wmx.qDio)),
                    groupScaleErr: Math.abs(inst.dioramaGroup.scale.x - wmx.lambdaP),
                    nestedPosErr: nested.group.position.distanceTo(wmx.tDio),
                    nestedQuatDot: Math.abs(nested.group.quaternion.dot(wmx.qDio)),
                    nestedScaleErr: Math.abs(nested.group.scale.x - wmx.lambdaP),
                    lambdaP: wmx.lambdaP,
                    wallYawDeg: (() => {
                      const n = wmx.tDio.clone().set(0, 0, 1).applyQuaternion(wmx.qDio);
                      return (Math.atan2(n.x, n.z) * 180.0) / Math.PI;
                    })()
                  };

                  // Forward: bar_giora_bedroom -> vitrine_keys_pen, sample just before landing.
                  sm.transitionTo('bar_giora_bedroom');
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  sm.transitionTo('vitrine_keys_pen');
                  out.fwdHasQuatPath = Boolean(sm._portalFlight && sm._portalFlight.quatPath);
                  sm._portalFlight.frozenProgress = 0.9999;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.fwdArrivalPosErr = sm.camera.position.distanceTo(wmx.arrivalCamPos);
                  out.fwdArrivalQuatDot = Math.abs(sm.camera.quaternion.dot(wmx.arrivalCamQuat));
                  out.fwdOuterWall = inst.bgMat.uniforms.uOuterWallOpacity.value;
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.landed = sm.currentSceneId;

                  // Reverse: from the standalone vitrine rest pose, t = 0 maps onto the arrival pose.
                  sm.transitionTo('bar_giora_bedroom');
                  out.revHasQuatPath = Boolean(sm._portalFlight && sm._portalFlight.quatPath);
                  sm._portalFlight.frozenProgress = 0.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.revStartPosErr = sm.camera.position.distanceTo(wmx.arrivalCamPos);
                  out.revStartQuatDot = Math.abs(sm.camera.quaternion.dot(wmx.arrivalCamQuat));
                  out.revStartOuterWall = inst.bgMat.uniforms.uOuterWallOpacity.value;
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.revLanded = sm.currentSceneId;
                  out.revRestOuterWall = inst.bgMat.uniforms.uOuterWallOpacity.value;
                  return out;
                })()
                """,
                await_promise=True,
                timeout_sec=THRESHOLDS.cdp_heavy_eval_timeout_sec,
            )
            pos_tol = THRESHOLDS.position_world_tol * 10.0
            quat_min_dot = 1.0 - 1e-6
            self.assertTrue(report["hasXform"], "Wall-mount transform was never computed for the vitrine")
            self.assertLess(report["groupPosErr"], pos_tol)
            self.assertGreater(report["groupQuatDot"], quat_min_dot)
            self.assertLess(report["groupScaleErr"], 1e-6)
            self.assertLess(report["lambdaP"], 0.5, "Vitrine diorama must be scaled down onto the wall")
            # The wall under the painting recedes ~30.7 deg (measured from the painting's vanishing
            # point); the neural depth map alone flattens it to ~6.7 deg, which reads as not flush.
            self.assertAlmostEqual(report["wallYawDeg"], 30.7, delta=1.5)
            self.assertLess(
                report["nestedPosErr"], pos_tol, "Nested vitrine in shelf cubby not placed on the wall"
            )
            self.assertGreater(report["nestedQuatDot"], quat_min_dot)
            self.assertLess(
                report["nestedScaleErr"], 1e-6, "Nested vitrine in shelf cubby drawn at the wrong scale"
            )

            self.assertTrue(report["fwdHasQuatPath"])
            self.assertLess(report["fwdArrivalPosErr"], pos_tol)
            self.assertGreater(report["fwdArrivalQuatDot"], quat_min_dot)
            self.assertAlmostEqual(report["fwdOuterWall"], 1.0, delta=1e-3)
            self.assertEqual(report["landed"], "vitrine_keys_pen")

            self.assertTrue(report["revHasQuatPath"])
            self.assertLess(report["revStartPosErr"], pos_tol)
            self.assertGreater(report["revStartQuatDot"], quat_min_dot)
            self.assertAlmostEqual(report["revStartOuterWall"], 1.0, delta=1e-3)
            self.assertEqual(report["revLanded"], "bar_giora_bedroom")
            self.assertAlmostEqual(report["revRestOuterWall"], 0.0, delta=1e-6)
        finally:
            self._close_chrome_cdp(proc, sock, tmp_dir)


    def test_purim_photo_surface_mount_pure_flight(self) -> None:
        """The Purim print lies tilted on the scroll floor: Scene 18 is embedded on it (pixel-exact
        and flat at rest), the camera swerves head-on and lands exactly on the standalone rest
        view (full relief), with zero crossfade/transition-shader rendering in either direction."""
        proc, sock, tmp_dir, eval_js = self._launch_chrome_cdp(
            THRESHOLDS.desktop_width, THRESHOLDS.desktop_height
        )
        try:
            report = eval_js(
                """
                (async () => {
                  const sm = window.__tamiSceneManager;
                  await sm._loadSceneBundleReady('bene_beitcha_scroll_foot', false);
                  await sm._loadSceneBundleReady('purim_costume_1966', false);
                  sm.transitionTo('bene_beitcha_scroll_foot');
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  sm._updatePortalShelfTransforms();
                  sm.animate(); cancelAnimationFrame(sm.animId);

                  const V = sm.camera.position.constructor;
                  const inst = sm._portalInstances.find((p) => p.spec.id === 'portal_scroll_foot_purim_1966');
                  const wmx = inst.wallMountXform;
                  const host = sm.sceneObjects.bene_beitcha_scroll_foot;
                  const tgt = sm.sceneObjects.purim_costume_1966;
                  const layer = host.extraLayers.find((e) => e.id === 'pinned_purim_photo');
                  const W = sm.width;
                  const H = sm.height;
                  const scr = (p) => {
                    const q = p.clone().project(sm.camera);
                    return [((q.x + 1) / 2) * W, ((1 - q.y) / 2) * H];
                  };
                  inst.dioramaGroup.updateMatrixWorld(true);
                  let maxRestErrPx = 0;
                  for (const [[tu, tv], [hx, hy]] of inst.spec.wallMount.pointPairsInnerPx) {
                    const hp = sm._sampleLayerSurfaceWorldPoint(layer, hx, hy);
                    const tp = new V(0.5 * tgt.innerScale * (tu - 750), -0.5 * tgt.innerScale * (tv - 500), tgt.sceneBgZ)
                      .applyMatrix4(inst.dioramaGroup.matrixWorld);
                    const a = scr(hp);
                    const b = scr(tp);
                    maxRestErrPx = Math.max(maxRestErrPx, Math.hypot(a[0] - b[0], a[1] - b[1]));
                  }
                  const out = {
                    placed: Boolean(wmx),
                    embedDiorama: inst.embedDiorama,
                    maxRestErrPx,
                    restRelief: inst.midMat.uniforms.uDepthAmplitude.value
                  };

                  sm.erasePortalOccluder('portal_scroll_foot_purim_1966');
                  sm._stepPortalOccluderAnimations(performance.now() + 800.0);
                  sm.transitionTo('purim_costume_1966');
                  out.fwdIsEmbeddedFlight = Boolean(sm._portalFlight && sm._portalFlight.embedDiorama);
                  out.fwdHasQuatPath = Boolean(sm._portalFlight && sm._portalFlight.quatPath);
                  sm._portalFlight.frozenProgress = 0.9999;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.fwdArrivalPosErr = sm.camera.position.distanceTo(wmx.arrivalCamPos);
                  out.fwdArrivalQuatDot = Math.abs(sm.camera.quaternion.dot(wmx.arrivalCamQuat));
                  out.fwdRelief = inst.midMat.uniforms.uDepthAmplitude.value;
                  out.fullRelief = tgt.midDepthAmplitude;
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.landed = sm.currentSceneId;

                  sm.transitionTo('bene_beitcha_scroll_foot');
                  out.revIsEmbeddedFlight = Boolean(sm._portalFlight && sm._portalFlight.embedDiorama);
                  sm._portalFlight.frozenProgress = 0.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.revStartPosErr = sm.camera.position.distanceTo(wmx.arrivalCamPos);
                  out.revStartQuatDot = Math.abs(sm.camera.quaternion.dot(wmx.arrivalCamQuat));
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.revLanded = sm.currentSceneId;
                  out.revRestRelief = inst.midMat.uniforms.uDepthAmplitude.value;

                  const cubby = sm._portalInstances.find((p) => p.spec.id === 'cubby_scroll_foot');
                  const nested = cubby.childPortalMeshes.map((cp) => cp.childWallMount).find(Boolean);
                  out.nestedVisible = Boolean(nested && nested.group.visible);
                  out.nestedMatrixErr = nested
                    ? Math.max(...nested.group.matrix.elements.map((v, i) => Math.abs(v - inst.dioramaGroup.matrix.elements[i])))
                    : 1e9;
                  return out;
                })()
                """,
                await_promise=True,
                timeout_sec=THRESHOLDS.cdp_heavy_eval_timeout_sec,
            )
            pos_tol = THRESHOLDS.position_world_tol * 10.0
            quat_min_dot = 1.0 - 1e-6
            self.assertTrue(report["placed"], "Purim surface mount was never placed on the scroll floor photo")
            self.assertTrue(report["embedDiorama"], "Purim portal must be an embedded 3D flight, not a crossfade")
            self.assertLess(report["maxRestErrPx"], 0.5, "Embedded Purim scene not pixel-aligned with the print")
            self.assertAlmostEqual(report["restRelief"], 0.0, delta=1e-6)

            self.assertTrue(report["fwdIsEmbeddedFlight"])
            self.assertTrue(report["fwdHasQuatPath"])
            self.assertLess(report["fwdArrivalPosErr"], pos_tol)
            self.assertGreater(report["fwdArrivalQuatDot"], quat_min_dot)
            self.assertAlmostEqual(report["fwdRelief"], report["fullRelief"], delta=0.05)
            self.assertEqual(report["landed"], "purim_costume_1966")

            self.assertTrue(report["revIsEmbeddedFlight"])
            self.assertLess(report["revStartPosErr"], pos_tol)
            self.assertGreater(report["revStartQuatDot"], quat_min_dot)
            self.assertEqual(report["revLanded"], "bene_beitcha_scroll_foot")
            self.assertAlmostEqual(report["revRestRelief"], 0.0, delta=1e-6)

            self.assertTrue(report["nestedVisible"])
            self.assertLess(report["nestedMatrixErr"], 1e-4, "Nested Purim mount in shelf cubby differs from standalone")
        finally:
            self._close_chrome_cdp(proc, sock, tmp_dir)

    def test_tzofen_shelf_cubby_flight_shows_nested_treetop_window(self) -> None:
        """New bottom-left shelf cubby flies into Tzofen; the cubby's copy of Tzofen mirrors the
        treetop seen through the window (same transform as the standalone portal diorama,
        clipped to the window from the live camera), so landing has no white-window snap."""
        proc, sock, tmp_dir, eval_js = self._launch_chrome_cdp(
            THRESHOLDS.desktop_width, THRESHOLDS.desktop_height
        )
        try:
            report = eval_js(
                """
                (async () => {
                  const sm = window.__tamiSceneManager;
                  await sm._loadSceneBundleReady('tzofen_brick_table', false);
                  sm._updatePortalShelfTransforms();
                  const cubby = sm._portalInstances.find((p) => p.spec.id === 'cubby_tzofen');
                  const src = sm._portalInstances.find((p) => p.spec.id === 'portal_tzofen_tree_window');
                  const cp = cubby && cubby.childPortalMeshes.find(
                    (c) => c.childSpec.id === 'portal_tzofen_tree_window'
                  );
                  const mirror = cp && cp.nestedMirror;
                  const out = {
                    cubbyHost: cubby && cubby.hostSceneId,
                    cubbyEmbed: Boolean(cubby && cubby.embedDiorama),
                    hasMirror: Boolean(mirror),
                    mirrorMeshes: mirror ? mirror.pairs.length : 0,
                    sourceMeshes: src.dioramaGroup.children.filter((o) => o.isMesh || o.isPoints).length,
                    groupPosErr: mirror ? mirror.group.position.distanceTo(src.dioramaGroup.position) : -1,
                    groupScaleErr: mirror ? Math.abs(mirror.group.scale.x - src.dioramaGroup.scale.x) : -1,
                    placeholderHidden: !cp.childDioBgMesh || !cp.childDioBgMesh.visible
                  };
                  sm.transitionTo('tzofen_brick_table');
                  out.isPortalFlight = Boolean(sm._portalFlight && !sm._portalFlight.reverse);
                  sm._portalFlight.frozenProgress = 0.95;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.maxMirrorPlaneConst = Math.max(...mirror.planes.map((pl) => Math.abs(pl.constant)));
                  out.mirrorBgVisible = mirror.pairs[0].dst.visible;
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.landed = sm.currentSceneId;
                  sm.transitionTo('shelf_portal');
                  out.reverseIsPortalFlight = Boolean(sm._portalFlight && sm._portalFlight.reverse);
                  sm._portalFlight.frozenProgress = 1.0;
                  sm.animate(); cancelAnimationFrame(sm.animId);
                  out.backOn = sm.currentSceneId;
                  return out;
                })()
                """,
                await_promise=True,
                timeout_sec=THRESHOLDS.cdp_heavy_eval_timeout_sec,
            )
            self.assertEqual(report["cubbyHost"], "shelf_portal")
            self.assertTrue(report["cubbyEmbed"])
            self.assertTrue(report["hasMirror"], "Shelf cubby copy of Tzofen has no treetop mirror")
            self.assertEqual(report["mirrorMeshes"], report["sourceMeshes"])
            self.assertLess(report["groupPosErr"], THRESHOLDS.position_world_tol)
            self.assertLess(report["groupScaleErr"], 1e-9)
            self.assertTrue(report["placeholderHidden"])
            self.assertTrue(report["isPortalFlight"])
            self.assertTrue(report["mirrorBgVisible"])
            self.assertLess(report["maxMirrorPlaneConst"], THRESHOLDS.unclipped_plane_sentinel)
            self.assertEqual(report["landed"], "tzofen_brick_table")
            self.assertTrue(report["reverseIsPortalFlight"])
            self.assertEqual(report["backOn"], "shelf_portal")
        finally:
            self._close_chrome_cdp(proc, sock, tmp_dir)


if __name__ == "__main__":
    unittest.main()
