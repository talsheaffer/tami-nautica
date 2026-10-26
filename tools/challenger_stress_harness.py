#!/usr/bin/env python3
"""
tools/challenger_stress_harness.py — Independent Empirical Adversarial Challenger Harness
Executed by Milestone 5 Iteration 2 Challenger (m5_it2_challenger).

Oracles, stress tests, and assertions:
1. Dual-Viewport Headless Chrome: Desktop (1440x900) and Mobile (390x844)
2. Active polling on window.__tamiSceneManager.layersReady
3. Explicit Parallax Shifts: dx > 20, dy > 10, matrixWorld delta > 1.0
4. WebGL Pipeline & Context Integrity: gl.getError() === 0 at all phases
5. 21:9 Ultrawide (2560x1080) & 32:9 Super Ultrawide (5120x1440) dynamic scaling
6. Transition Storming & Fault Injection (30 rapid calls, invalid scene IDs, mid-transition interrupts)
7. Full WebGL Lifecycle Disposal & GPU Resource Reclaim
8. 100% Offline Self-Containment (Zero remote network requests captured via CDP Network domain)
"""

import os
import sys
import json
import time
import socket
import struct
import tempfile
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR = PROJECT_ROOT / "prototype"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def find_free_port(start_port=7800, max_attempts=50):
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ChromeSession:
    def __init__(self, http_port, width=1440, height=900, is_mobile=False):
        self.http_port = http_port
        self.width = width
        self.height = height
        self.is_mobile = is_mobile
        self.cdp_port = find_free_port(9500)
        self.tmp_dir = tempfile.mkdtemp(prefix="tami_challenger_")
        self.proc = None
        self.ws = None
        self.network_requests = []
        self.console_messages = []
        self.msg_id = 1000

    def start(self):
        url = f"http://127.0.0.1:{self.http_port}/index.html"
        cmd = [
            CHROME_BIN,
            "--headless=new",
            "--disable-extensions",
            f"--user-data-dir={self.tmp_dir}",
            f"--window-size={self.width},{self.height}",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            f"--remote-debugging-port={self.cdp_port}",
            url
        ]
        if self.is_mobile:
            cmd.append("--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1")

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Poll CDP /json endpoint
        tabs = None
        start_t = time.time()
        while time.time() - start_t < 15.0:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cdp_port}/json", timeout=2) as r:
                    tabs = json.loads(r.read())
                    if [t for t in tabs if "index.html" in t.get("url", "")]:
                        break
            except Exception:
                time.sleep(0.2)

        if not tabs:
            raise RuntimeError(f"Chrome CDP /json failed to respond on port {self.cdp_port}")

        target_tabs = [t for t in tabs if "index.html" in t.get("url", "")]
        if not target_tabs:
            raise RuntimeError("index.html tab not found in Chrome CDP")

        ws_url = target_tabs[0]["webSocketDebuggerUrl"]
        parts = ws_url.replace("ws://", "").split("/", 1)
        host, port_str = parts[0].split(":")
        path = "/" + parts[1]

        self.ws = socket.create_connection((host, int(port_str)), timeout=10.0)
        req = (f"GET {path} HTTP/1.1\r\n"
               f"Host: {host}:{port_str}\r\n"
               f"Upgrade: websocket\r\n"
               f"Connection: Upgrade\r\n"
               f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
               f"Sec-WebSocket-Version: 13\r\n\r\n")
        self.ws.sendall(req.encode("utf-8"))
        resp = self.ws.recv(4096)
        if b"101" not in resp:
            raise RuntimeError(f"CDP WebSocket handshake failed: {resp}")

        # Enable Network and Runtime domain
        self.send("Network.enable")
        self.read(target_id=self.msg_id)
        self.send("Runtime.enable")
        self.read(target_id=self.msg_id)

    def send(self, method, params=None):
        self.msg_id += 1
        msg = json.dumps({"id": self.msg_id, "method": method, "params": params or {}})
        data = msg.encode("utf-8")
        frame = bytearray([0x81])
        length = len(data)
        if length < 126:
            frame.append(0x80 | length)
        elif length < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack(">H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack(">Q", length))
        mask_key = b"\x12\x34\x56\x78"
        frame.extend(mask_key)
        frame.extend(bytearray(b ^ mask_key[i % 4] for i, b in enumerate(data)))
        self.ws.sendall(bytes(frame))
        return self.msg_id

    def read(self, target_id=None, timeout=10.0):
        start = time.time()
        self.ws.settimeout(timeout)
        while time.time() - start < timeout:
            try:
                header = self.ws.recv(2)
            except (socket.timeout, TimeoutError):
                return None
            if not header:
                return None
            length = header[1] & 0x7F
            if length == 126:
                length = struct.unpack(">H", self.ws.recv(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self.ws.recv(8))[0]
            payload = b""
            while len(payload) < length:
                chunk = self.ws.recv(length - len(payload))
                if not chunk:
                    break
                payload += chunk
            msg = json.loads(payload.decode("utf-8", errors="replace"))

            # Capture events
            method = msg.get("method", "")
            if method == "Network.requestWillBeSent":
                req_url = msg.get("params", {}).get("request", {}).get("url", "")
                self.network_requests.append(req_url)
            elif method == "Runtime.consoleAPICalled":
                self.console_messages.append(msg.get("params", {}))

            if target_id is None or msg.get("id") == target_id:
                return msg
        return None

    def evaluate(self, expr, timeout=10.0):
        mid = self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        res = self.read(target_id=mid, timeout=timeout)
        if not res or "result" not in res:
            return None
        r = res["result"].get("result", {})
        if "value" in r:
            return r["value"]
        return r

    def wait_for_layers_ready(self, timeout=15.0):
        poll_js = """
        (() => {
            const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
            const st = window.__PROTOTYPE_STATE__;
            return !!((sm && sm.layersReady) || (st && st.layersReady));
        })()
        """
        start = time.time()
        while time.time() - start < timeout:
            val = self.evaluate(poll_js)
            if val is True:
                return True
            time.sleep(0.2)
        return False

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def run_all_challenger_tests():
    from prototype.serve import start_server_thread

    print("======================================================================")
    print("M5 ITERATION 2 INDEPENDENT ADVERSARIAL CHALLENGER STRESS HARNESS")
    print("======================================================================")

    http_port = find_free_port(7700)
    server, http_port, server_thread = start_server_thread(port=http_port, directory=PROTOTYPE_DIR)
    print(f"[*] Started local prototype HTTP server on port {http_port}")

    failures = []

    try:
        # --------------------------------------------------------------------
        # TEST 1: Desktop Viewport (1440x900) Parallax, WebGL & Disposal
        # --------------------------------------------------------------------
        print("\n--- [TEST 1] Desktop Viewport 1440x900 ---")
        client = ChromeSession(http_port, width=1440, height=900, is_mobile=False)
        client.start()
        try:
            ready = client.wait_for_layers_ready(timeout=15.0)
            if not ready:
                failures.append("TEST 1: layersReady failed to become true within 15s")
            print(f"  [+] layersReady: {ready}")

            # Check baseline WebGL error state
            gl_err = client.evaluate("""
                (() => {
                    const canvas = document.querySelector('canvas#spatial-canvas');
                    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                    return gl ? gl.getError() : -1;
                })()
            """)
            print(f"  [+] Initial gl.getError(): {gl_err}")
            if gl_err != 0:
                failures.append(f"TEST 1: gl.getError() initially returned {gl_err}")

            # Extreme 1: Top-Left (-1.0, 1.0)
            client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                    sm.pointer.targetX = -1.0;
                    sm.pointer.targetY = 1.0;
                })()
            """)
            time.sleep(0.8)

            pos1 = client.evaluate("JSON.stringify(window.__PROTOTYPE_STATE__.cameraPosition)")
            pos1 = json.loads(pos1) if pos1 else {}
            mat1 = client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                    return sm && sm.camera ? Array.from(sm.camera.matrixWorld.elements) : [];
                })()
            """)

            # Extreme 2: Bottom-Right (1.0, -1.0)
            client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                    sm.pointer.targetX = 1.0;
                    sm.pointer.targetY = -1.0;
                })()
            """)
            time.sleep(0.8)

            pos2 = client.evaluate("JSON.stringify(window.__PROTOTYPE_STATE__.cameraPosition)")
            pos2 = json.loads(pos2) if pos2 else {}
            mat2 = client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                    return sm && sm.camera ? Array.from(sm.camera.matrixWorld.elements) : [];
                })()
            """)

            dx = abs(pos1.get("x", 0) - pos2.get("x", 0))
            dy = abs(pos1.get("y", 0) - pos2.get("y", 0))
            mat_delta = sum(abs(a - b) for a, b in zip(mat1, mat2))
            print(f"  [+] Parallax Deltas: dx = {dx:.2f} (required > 20), dy = {dy:.2f} (required > 10)")
            print(f"  [+] Camera matrixWorld Delta: {mat_delta:.4f} (required > 1.0)")

            if dx <= 20.0:
                failures.append(f"TEST 1: dx ({dx:.2f}) <= 20.0")
            if dy <= 10.0:
                failures.append(f"TEST 1: dy ({dy:.2f}) <= 10.0")
            if mat_delta <= 1.0:
                failures.append(f"TEST 1: camera matrixWorld delta ({mat_delta:.4f}) <= 1.0")

            # Transition to Denya
            print("  [*] Triggering transition to Scene 2 (denya)...")
            client.evaluate("window.__tamiSceneManager.transitionTo('denya')")
            start_t = time.time()
            trans_done = False
            while time.time() - start_t < 10.0:
                st = client.evaluate("JSON.stringify(window.__PROTOTYPE_STATE__)")
                st_data = json.loads(st) if st else {}
                if st_data.get("currentScene") == "denya" and not st_data.get("isTransitioning"):
                    trans_done = True
                    break
                time.sleep(0.2)

            print(f"  [+] Transition completed: {trans_done}")
            if not trans_done:
                failures.append("TEST 1: Transition to 'denya' failed to complete within 10s")

            # Check post-transition WebGL error
            gl_err2 = client.evaluate("""
                (() => {
                    const canvas = document.querySelector('canvas#spatial-canvas');
                    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                    return gl ? gl.getError() : -1;
                })()
            """)
            print(f"  [+] Post-transition gl.getError(): {gl_err2}")
            if gl_err2 != 0:
                failures.append(f"TEST 1: Post-transition gl.getError() returned {gl_err2}")

            # Verify Scene 2 reflection reveal
            has_refl = client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager;
                    return sm && sm.sceneObjects['denya'] && sm.sceneObjects['denya'].reflectionMesh !== undefined;
                })()
            """)
            print(f"  [+] Scene 2 reflection mesh present: {has_refl}")
            if not has_refl:
                failures.append("TEST 1: Scene 2 reflectionMesh missing")

            # Test dispose() lifecycle
            print("  [*] Invoking dispose() lifecycle...")
            dispose_res = client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager;
                    const canvas = document.querySelector('canvas#spatial-canvas');
                    const gl = canvas ? (canvas.getContext('webgl2') || canvas.getContext('webgl')) : null;
                    sm.dispose();
                    return {
                        animIdNull: sm.animId === null,
                        sceneObjectsEmpty: Object.keys(sm.sceneObjects).length === 0,
                        rtCurrentNull: sm.rtCurrent === null,
                        rtNextNull: sm.rtNext === null,
                        contextLost: gl ? gl.isContextLost() : false
                    };
                })()
            """)
            print(f"  [+] dispose() results: {dispose_res}")
            if not (dispose_res.get("animIdNull") and dispose_res.get("sceneObjectsEmpty") and dispose_res.get("rtCurrentNull") and dispose_res.get("rtNextNull") and dispose_res.get("contextLost")):
                failures.append(f"TEST 1: dispose() failed full cleanup: {dispose_res}")

        finally:
            client.close()

        # --------------------------------------------------------------------
        # TEST 2: Mobile Viewport (390x844) Touch & RTL
        # --------------------------------------------------------------------
        print("\n--- [TEST 2] Mobile Viewport 390x844 ---")
        m_client = ChromeSession(http_port, width=390, height=844, is_mobile=True)
        m_client.start()
        try:
            m_ready = m_client.wait_for_layers_ready(timeout=15.0)
            print(f"  [+] Mobile layersReady: {m_ready}")
            if not m_ready:
                failures.append("TEST 2: Mobile layersReady failed within 15s")

            # Verify HTML RTL direction
            dir_val = m_client.evaluate("document.documentElement.dir")
            print(f"  [+] Document dir: '{dir_val}'")
            if dir_val != "rtl":
                failures.append(f"TEST 2: Document dir is '{dir_val}', expected 'rtl'")

            # Simulate touch interactions: touchmove
            m_client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager;
                    sm._onTouchStart({ touches: [{ clientX: 195, clientY: 422 }] });
                    sm._onTouchMove({ touches: [{ clientX: 50, clientY: 700 }] });
                })()
            """)
            time.sleep(0.8)
            m_pos1 = m_client.evaluate("JSON.stringify(window.__PROTOTYPE_STATE__.cameraPosition)")
            m_pos1 = json.loads(m_pos1) if m_pos1 else {}

            m_client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager;
                    sm._onTouchMove({ touches: [{ clientX: 340, clientY: 100 }] });
                })()
            """)
            time.sleep(0.8)
            m_pos2 = m_client.evaluate("JSON.stringify(window.__PROTOTYPE_STATE__.cameraPosition)")
            m_pos2 = json.loads(m_pos2) if m_pos2 else {}

            m_dx = abs(m_pos1.get("x", 0) - m_pos2.get("x", 0))
            m_dy = abs(m_pos1.get("y", 0) - m_pos2.get("y", 0))
            print(f"  [+] Mobile Touch Parallax Deltas: dx = {m_dx:.2f} (required > 20), dy = {m_dy:.2f} (required > 10)")
            if m_dx <= 20.0:
                failures.append(f"TEST 2: Mobile dx ({m_dx:.2f}) <= 20.0")
            if m_dy <= 10.0:
                failures.append(f"TEST 2: Mobile dy ({m_dy:.2f}) <= 10.0")

            # Check gl.getError() == 0
            m_gl_err = m_client.evaluate("""
                (() => {
                    const canvas = document.querySelector('canvas#spatial-canvas');
                    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                    return gl ? gl.getError() : -1;
                })()
            """)
            print(f"  [+] Mobile gl.getError(): {m_gl_err}")
            if m_gl_err != 0:
                failures.append(f"TEST 2: Mobile gl.getError() returned {m_gl_err}")

        finally:
            m_client.close()

        # --------------------------------------------------------------------
        # TEST 3: Ultrawide Viewports (21:9 and 32:9)
        # --------------------------------------------------------------------
        print("\n--- [TEST 3] Ultrawide Viewports (21:9 / 2560x1080 and 32:9 / 5120x1440) ---")
        uw_client = ChromeSession(http_port, width=2560, height=1080, is_mobile=False)
        uw_client.start()
        try:
            uw_ready = uw_client.wait_for_layers_ready(timeout=15.0)
            print(f"  [+] 21:9 layersReady: {uw_ready}")
            if not uw_ready:
                failures.append("TEST 3: 21:9 layersReady failed within 15s")

            uw_info = uw_client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager;
                    const canvas = document.querySelector('canvas#spatial-canvas');
                    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                    return {
                        innerWidth: window.innerWidth,
                        innerHeight: window.innerHeight,
                        aspect: window.innerWidth / window.innerHeight,
                        layer0Scale: window.__PROTOTYPE_STATE__.layer0Scale,
                        glError: gl ? gl.getError() : -1
                    };
                })()
            """)
            print(f"  [+] 21:9 Metrics: {uw_info}")
            if uw_info.get("glError") != 0:
                failures.append(f"TEST 3: 21:9 glError returned {uw_info.get('glError')}")
            if uw_info.get("layer0Scale", 0) < 1.875:
                failures.append(f"TEST 3: 21:9 layer0Scale ({uw_info.get('layer0Scale')}) < 1.875")

            # Check 32:9 super ultrawide
            print("  [*] Simulating 32:9 Super Ultrawide (5120x1440)...")
            uw_client.send("Emulation.setDeviceMetricsOverride", {
                "width": 5120, "height": 1440, "deviceScaleFactor": 1, "mobile": False
            })
            uw_client.read(target_id=uw_client.msg_id)
            uw_client.evaluate("window.dispatchEvent(new Event('resize'))")
            time.sleep(0.5)

            uw32_info = uw_client.evaluate("""
                (() => {
                    const canvas = document.querySelector('canvas#spatial-canvas');
                    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                    return {
                        innerWidth: window.innerWidth,
                        innerHeight: window.innerHeight,
                        aspect: window.innerWidth / window.innerHeight,
                        layer0Scale: window.__PROTOTYPE_STATE__.layer0Scale,
                        glError: gl ? gl.getError() : -1
                    };
                })()
            """)
            print(f"  [+] 32:9 Metrics: {uw32_info}")
            if uw32_info.get("layer0Scale", 0) < 2.5:
                failures.append(f"TEST 3: 32:9 layer0Scale ({uw32_info.get('layer0Scale')}) < 2.5")
            if uw32_info.get("glError") != 0:
                failures.append(f"TEST 3: 32:9 glError returned {uw32_info.get('glError')}")

        finally:
            uw_client.close()

        # --------------------------------------------------------------------
        # TEST 4: Transition Storming & Fault Injection
        # --------------------------------------------------------------------
        print("\n--- [TEST 4] Rapid Transition Storming & Fault Injection ---")
        storm_client = ChromeSession(http_port, width=1440, height=900, is_mobile=False)
        storm_client.start()
        try:
            storm_ready = storm_client.wait_for_layers_ready(timeout=15.0)
            print(f"  [+] Storming layersReady: {storm_ready}")
            if not storm_ready:
                failures.append("TEST 4: Storming layersReady failed within 15s")

            # Inject 30 rapid valid and invalid transition calls
            storm_result = storm_client.evaluate("""
                (() => {
                    const sm = window.__tamiSceneManager;
                    const errors = [];
                    window.addEventListener('error', (e) => errors.push(e.message));

                    // 1. Rapid valid transitions
                    for (let i = 0; i < 20; i++) {
                        const target = (i % 2 === 0) ? 'denya' : 'herzliya';
                        try {
                            sm.transitionTo(target);
                        } catch (e) {
                            errors.push('valid: ' + e.message);
                        }
                    }

                    // 2. Fault injection: invalid scene IDs
                    const invalidTargets = ['invalid_scene', '', null, undefined, 42, {}, []];
                    invalidTargets.forEach(tgt => {
                        try {
                            sm.transitionTo(tgt);
                        } catch (e) {
                            errors.push('invalid: ' + e.message);
                        }
                    });

                    return {
                        immediateErrors: errors,
                        isTransitioning: sm.isTransitioning,
                        currentSceneId: sm.currentSceneId
                    };
                })()
            """)
            print(f"  [+] Storming immediate results: {storm_result}")
            if storm_result.get("immediateErrors"):
                failures.append(f"TEST 4: Immediate errors during transition storming: {storm_result.get('immediateErrors')}")

            # Wait for transition pipeline to settle
            settle_start = time.time()
            settled = False
            settled_data = {}
            while time.time() - settle_start < 10.0:
                settled_data = storm_client.evaluate("""
                    (() => {
                        const sm = window.__tamiSceneManager;
                        const canvas = document.querySelector('canvas#spatial-canvas');
                        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                        return {
                            currentScene: sm.currentSceneId,
                            isTransitioning: sm.isTransitioning,
                            glError: gl ? gl.getError() : -1,
                            validScene: sm.currentSceneId === 'herzliya' || sm.currentSceneId === 'denya'
                        };
                    })()
                """)
                if settled_data and not settled_data.get("isTransitioning"):
                    settled = True
                    break
                time.sleep(0.2)

            print(f"  [+] Storming settled state: {settled_data}")
            if not settled:
                failures.append("TEST 4: State failed to settle (isTransitioning stuck in true)")
            if not settled_data.get("validScene"):
                failures.append(f"TEST 4: Settled in invalid scene: {settled_data.get('currentScene')}")
            if settled_data.get("glError") != 0:
                failures.append(f"TEST 4: glError after storming: {settled_data.get('glError')}")

        finally:
            storm_client.close()

        # --------------------------------------------------------------------
        # TEST 5: 100% Offline Self-Containment Audit (Filesystem & Network)
        # --------------------------------------------------------------------
        print("\n--- [TEST 5] 100% Offline Self-Containment Audit ---")
        # 1. Audit filesystem for external network references
        ext_patterns = ["fonts.googleapis.com", "fonts.gstatic.com", "cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com"]
        for p in ext_patterns:
            grep_cmd = ["grep", "-rn", p, str(PROTOTYPE_DIR)]
            res = subprocess.run(grep_cmd, capture_output=True, text=True)
            if res.stdout.strip():
                failures.append(f"TEST 5: Found remote pattern '{p}' in prototype files: {res.stdout.strip()}")
            else:
                print(f"  [+] Zero instances of '{p}' in prototype/")

        # 2. Audit CDP network requests recorded across all sessions
        all_captured_urls = (
            client.network_requests + m_client.network_requests +
            uw_client.network_requests + storm_client.network_requests
        )
        external_requests = [
            u for u in all_captured_urls
            if not (u.startswith("http://127.0.0.1") or u.startswith("http://localhost") or u.startswith("data:"))
        ]
        print(f"  [+] Total captured CDP requests across sessions: {len(all_captured_urls)}")
        print(f"  [+] External requests detected: {len(external_requests)}")
        if external_requests:
            failures.append(f"TEST 5: External requests detected: {external_requests}")

    finally:
        server.shutdown()
        server.server_close()
        print("\n[*] Local prototype HTTP server shut down cleanly.")

    print("\n======================================================================")
    if failures:
        print(f"HARNESS RESULT: FAILED ({len(failures)} failures)")
        for f in failures:
            print(f"  [-] {f}")
        return False
    else:
        print("HARNESS RESULT: ALL EMPIRICAL CHALLENGES PASSED (100% CLEAN)")
        return True


if __name__ == "__main__":
    success = run_all_challenger_tests()
    sys.exit(0 if success else 1)
