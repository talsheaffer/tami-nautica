#!/usr/bin/env python3
"""
tests/test_beta_feedback.py — Automated Unit & Headless Chrome WebGL E2E Suite
for the Collaborative Family Beta Hub:
- Universal DOM UI Element Inspector & Composite HTML+WebGL Screenshots
- 3D Spatial Raycast Highlighting & Pin Reprojection
- Gyro Toggle ON/OFF Regression Test
- Read-Only Biographical Wiki Portal (`/wiki`, `/api/wiki`)
- Family Photo Drop-Folder Uploader & Live Catalog Enrichment (`/catalog`, `/api/uploads`, `/api/catalog`)
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from prototype.beta_config import BetaFeedbackConfig, default_beta_feedback_config
from prototype.feedback_store import (
    SAFE_LAYER_ID_RE,
    BetaFeedbackStore,
    FeedbackNotFoundError,
    FeedbackValidationError,
)
from prototype.serve import find_free_port, start_server_thread
from prototype.upload_service import FamilyPhotoUploadService, UploadValidationError
from prototype.wiki_service import WikiNotFoundError, WikiPortalService, WikiValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR = PROJECT_ROOT / "prototype"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _make_sample_png_data_url(width: int = 240, height: int = 180, color: tuple[int, int, int] = (44, 52, 64)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class TestBetaConfigSSOT(unittest.TestCase):
    """Verifies lossless round-trip serialization of BetaFeedbackConfig."""

    def test_config_roundtrip(self) -> None:
        cfg = default_beta_feedback_config()
        raw_json = cfg.to_json()
        restored = BetaFeedbackConfig.from_json(raw_json)
        self.assertEqual(cfg, restored)
        preset_ids = [p.preset_id for p in restored.reviewer_presets]
        self.assertEqual(preset_ids, ["tal", "naama", "gadi", "guy"])
        people_canonicals = [c for c, _ in restored.uploads.people_presets]
        self.assertIn("Ada Wertheimer", people_canonicals)
        for new_scene in (
            "scroll_palm_world",
            "humpty_dumpty_bundle",
            "humpty_dumpty_booth",
            "tzofen_brick_table",
            "herzliya_tree_canopy",
            "bene_beitcha_scroll_foot",
            "purim_costume_1966",
            "vitrine_keys_pen",
            "herzliya_lawn_carpet",
            "bar_giora_bedroom",
            "paris_cite_white_table",
            "toddler_tami_1958",
        ):
            self.assertIn(new_scene, restored.valid_scene_ids)

        disabled_cfg = default_beta_feedback_config(enabled=False)
        self.assertFalse(BetaFeedbackConfig.from_json(disabled_cfg.to_json()).enabled)


class TestWikiAndUploadServices(unittest.TestCase):
    """Verifies read-only WikiPortalService and sandboxed FamilyPhotoUploadService."""

    def test_wiki_portal_service_navigation_and_traversal_guard(self) -> None:
        svc = WikiPortalService(PROJECT_ROOT / "wiki")
        nav = svc.list_navigation()
        self.assertGreaterEqual(nav["total_pages"], 31)
        page = svc.get_page("places/haifa-greenboim-80-denya.md")
        self.assertIn("Greenboim", page.summary.title)
        self.assertGreater(len(page.markdown_body), 200)

        with self.assertRaises(WikiValidationError):
            svc.get_page("../MISSION.md")
        with self.assertRaises(WikiNotFoundError):
            svc.get_page("places/nonexistent.md")

    def test_family_upload_and_caption_update_in_isolated_sandbox(self) -> None:
        sandbox = Path(tempfile.mkdtemp(prefix="tami_upload_unit_"))
        try:
            cfg = default_beta_feedback_config()
            svc = FamilyPhotoUploadService(
                project_root=sandbox,
                drop_dir=sandbox / "drop_folder",
                media_dir=sandbox / "media",
                catalog_dir=sandbox / "catalog",
                limits=cfg.uploads,
            )

            res = svc.ingest_batch(
                {
                    "uploader": "גדי (אבא)",
                    "caption": "החצר של אהובה ומיכאל בהרצליה",
                    "date_or_era": "1974",
                    "places": ["Herzliya (parents' house & yard)"],
                    "people": ["Tamar 'Tami' Sheaffer", "Michael Fried", "Ahuva Fried"],
                    "files": [
                        {
                            "filename": "herzliya_yard_1974.png",
                            "data_url": _make_sample_png_data_url(320, 240, (90, 120, 80)),
                            "title": "חצר הבית בהרצליה",
                        }
                    ],
                }
            )
            self.assertEqual(res["ingested_count"], 1)
            self.assertEqual(res["duplicate_count"], 0)
            self.assertEqual(res["catalog_total_count"], 1)

            # Verify duplicate SHA-256 rejection
            dup_res = svc.ingest_batch(
                {
                    "uploader": "גיא",
                    "files": [
                        {
                            "filename": "same_photo_copy.png",
                            "data_url": _make_sample_png_data_url(320, 240, (90, 120, 80)),
                        }
                    ],
                }
            )
            self.assertEqual(dup_res["ingested_count"], 0)
            self.assertEqual(dup_res["duplicate_count"], 1)

            # Verify <150px rejection
            with self.assertRaises(UploadValidationError):
                svc.ingest_batch(
                    {
                        "uploader": "טל",
                        "files": [
                            {
                                "filename": "tiny_icon.png",
                                "data_url": _make_sample_png_data_url(80, 80),
                            }
                        ],
                    }
                )

            # Verify live caption/metadata enrichment on existing catalog item
            items = svc.get_catalog_items()
            self.assertEqual(len(items), 1)
            item_id = items[0]["id"]
            upd = svc.update_item_metadata(
                item_id,
                {
                    "editor": "גדי (אבא)",
                    "title": "עץ ההדר בחצר בהרצליה",
                    "caption": "מיכאל בנה כאן את הספסל ליד גן הילדים של אהובה",
                    "date_or_era": "1975",
                    "places": ["Herzliya (parents' house & yard)"],
                    "people": ["Michael Fried", "Ahuva Fried"],
                },
            )
            self.assertEqual(upd["item"]["title"], "עץ ההדר בחצר בהרצליה")
            self.assertIn("מיכאל בנה כאן", upd["item"]["caption"])
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)


class TestBetaFeedbackStore(unittest.TestCase):
    """Verifies domain validation, persistence, snapshot decoding, and traversal defenses."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="tami_feedback_unit_"))
        self.cfg = default_beta_feedback_config()
        self.store = BetaFeedbackStore(self.cfg, self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_list_resolve_and_delete_entry(self) -> None:
        snap_url = _make_sample_png_data_url()
        e1 = self.store.create_entry(
            {
                "author": "טל",
                "category_id": "visual_element",
                "scene_id": "bene_beitcha",
                "comment": "כדאי לחזק את ההשתקפות במראה של בנה ביתך.",
                "camera_pose": {
                    "x": 32.5,
                    "y": -8.0,
                    "z": 360.0,
                    "yaw_deg": -12.4,
                    "pitch_deg": 4.2,
                    "fov_deg": 45.0,
                },
                "highlight": {
                    "mode": "box",
                    "layer_id": "reflection",
                    "layer_label_he": "השתקפות מראה (בנה ביתך)",
                    "screen_rect_norm": {"x0": 0.62, "y0": 0.41, "x1": 0.74, "y1": 0.56},
                    "pin_screen_norm": [0.68, 0.485],
                    "world_point": [180.0, -20.0, -340.0],
                    "world_box_corners": [
                        [150.0, 10.0, -340.0],
                        [210.0, 10.0, -340.0],
                        [210.0, -50.0, -340.0],
                        [150.0, -50.0, -340.0],
                    ],
                    "texture_uv": [0.82, 0.47],
                    "texture_pixel": [1230, 530],
                },
                "snapshot_data_url": snap_url,
                "viewport": [1440, 900],
                "page_path": "/",
                "target_ref": "bene_beitcha",
            }
        )
        self.assertEqual(e1.seq_num, 1)
        self.assertEqual(e1.status, "open")
        self.assertIsNotNone(e1.snapshot_filename)
        snap_file = self.store.resolve_snapshot_path(e1.snapshot_filename or "")
        self.assertTrue(snap_file.is_file())

        e2 = self.store.create_entry(
            {
                "author": "גדי (אבא)",
                "category_id": "story_bio",
                "scene_id": "wiki",
                "comment": "הדירה בארלוזורוב 87 הייתה בקומה העליונה בפינת בר גיורא.",
                "page_path": "/wiki",
                "target_ref": "places/haifa-arlozorov-87.md",
            }
        )
        self.assertEqual(e2.seq_num, 2)
        self.assertEqual(e2.target_ref, "places/haifa-arlozorov-87.md")

        all_entries = self.store.list_entries()
        self.assertEqual(len(all_entries), 2)
        wiki_only = self.store.list_entries(scene_id="wiki")
        self.assertEqual(len(wiki_only), 1)
        self.assertEqual(wiki_only[0].entry_id, e2.entry_id)

        resolved = self.store.update_status(e1.entry_id, "resolved")
        self.assertEqual(resolved.status, "resolved")
        open_only = self.store.list_entries(include_resolved=False)
        self.assertEqual(len(open_only), 1)

        self.store.delete_entry(e1.entry_id)
        self.assertFalse(snap_file.exists())
        self.assertEqual(len(self.store.list_entries()), 1)

    def test_validation_rejects_bad_inputs_and_traversal(self) -> None:
        with self.assertRaises(FeedbackValidationError):
            self.store.create_entry(
                {"author": "", "category_id": "suggestion", "scene_id": "paper_capsules", "comment": "hi"}
            )
        with self.assertRaises(FeedbackValidationError):
            self.store.create_entry(
                {"author": "טל", "category_id": "unknown", "scene_id": "paper_capsules", "comment": "hi"}
            )
        with self.assertRaises(FeedbackValidationError):
            self.store.create_entry(
                {
                    "author": "טל",
                    "category_id": "suggestion",
                    "scene_id": "paper_capsules",
                    "comment": "bad image",
                    "snapshot_data_url": "data:image/jpeg;base64,bm90LWFuLWltYWdl",
                }
            )
        with self.assertRaises(FeedbackValidationError):
            self.store.resolve_snapshot_path("../secret.jpg")
        with self.assertRaises(FeedbackNotFoundError):
            self.store.update_status("fb-missing", "resolved")

    def test_creative_studio_modes_and_selected_media_ids(self) -> None:
        entry = self.store.create_entry(
            {
                "author": "גיא",
                "category_id": "suggestion",
                "scene_id": "catalog",
                "comment": "למזג את שתי הזוויות של הסלון בדניה לסצנה מרחבית אחת",
                "creation_mode": "fuse_scene",
                "selected_media_ids": ["IMG_0010", "IMG_0011"],
            }
        )
        self.assertEqual(entry.creation_mode, "fuse_scene")
        self.assertEqual(entry.selected_media_ids, ("IMG_0010", "IMG_0011"))

        reloaded = self.store.list_entries(scene_id="catalog")
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].creation_mode, "fuse_scene")
        self.assertEqual(reloaded[0].selected_media_ids, ("IMG_0010", "IMG_0011"))

        with self.assertRaises(FeedbackValidationError):
            self.store.create_entry(
                {
                    "author": "גיא",
                    "category_id": "suggestion",
                    "scene_id": "catalog",
                    "comment": "מצב לא קיים",
                    "creation_mode": "invalid_mode",
                }
            )

    def test_direct_screenshot_attachment_without_spatial_highlight(self) -> None:
        snap_url = _make_sample_png_data_url(402, 533, (28, 34, 42))
        entry = self.store.create_entry(
            {
                "author": "נעמה",
                "category_id": "ui_ux",
                "scene_id": "wandering_drawing",
                "comment": "העלאת צילום מסך ישירה מהטלפון ללא צורך בסימון נקודה במסך",
                "snapshot_data_url": snap_url,
                "viewport": [402, 533],
                "page_path": "/",
            }
        )
        self.assertIsNone(entry.highlight)
        self.assertIsNotNone(entry.snapshot_filename)
        snap_path = self.store.resolve_snapshot_path(entry.snapshot_filename or "")
        self.assertTrue(snap_path.is_file())

    def test_extra_layer_and_occluder_layer_ids_accepted_and_unsafe_rejected(self) -> None:
        dynamic_layer_ids = (
            "occluder_cubby_wandering_1",
            "occluder_cubby_scroll_foot",
            "occluder_fried_installation",
            "pipe_shell",
            "planted_couch_trio",
            "yard_railing",
            "refl_people",
            "extra:layer-01",
        )
        for idx, layer_id in enumerate(dynamic_layer_ids, start=1):
            entry = self.store.create_entry(
                {
                    "author": "טל",
                    "category_id": "visual_element",
                    "scene_id": "shelf_portal",
                    "comment": f"בדיקת סימון שכבה דינמית {layer_id}",
                    "highlight": {
                        "mode": "point",
                        "layer_id": layer_id,
                        "layer_label_he": f"שכבה מרחבית ({layer_id})",
                        "screen_rect_norm": {"x0": 0.4, "y0": 0.4, "x1": 0.5, "y1": 0.5},
                        "pin_screen_norm": [0.45, 0.45],
                        "world_point": [10.0 * idx, -5.0, -200.0],
                    },
                }
            )
            self.assertIsNotNone(entry.highlight)
            assert entry.highlight is not None
            self.assertEqual(entry.highlight.layer_id, layer_id)

        persisted = self.store.list_entries(scene_id="shelf_portal")
        self.assertEqual(
            [e.highlight.layer_id for e in persisted if e.highlight is not None],
            list(dynamic_layer_ids),
        )

        for bad_layer_id in ("", "   ", "../traversal", "layer with spaces", "<script>", "x" * 97):
            with self.assertRaises(FeedbackValidationError):
                self.store.create_entry(
                    {
                        "author": "טל",
                        "category_id": "visual_element",
                        "scene_id": "shelf_portal",
                        "comment": "שכבה לא תקינה",
                        "highlight": {
                            "mode": "point",
                            "layer_id": bad_layer_id,
                            "layer_label_he": "שכבה לא תקינה",
                            "screen_rect_norm": {"x0": 0.4, "y0": 0.4, "x1": 0.5, "y1": 0.5},
                            "pin_screen_norm": [0.45, 0.45],
                        },
                    }
                )

    def test_dynamic_manifest_scene_ids_accepted_without_restart(self) -> None:
        manifest_path = self.tmp_dir / "scene_manifest.json"
        live_store = BetaFeedbackStore(
            self.cfg, self.tmp_dir / "live_store", scene_manifest_path=manifest_path
        )

        with self.assertRaises(FeedbackValidationError):
            live_store.create_entry(
                {
                    "author": "טל",
                    "category_id": "visual_element",
                    "scene_id": "newly_built_portal_scene",
                    "comment": "לפני בניית הסצנה במניפסט",
                }
            )
        with self.assertRaises(FeedbackValidationError):
            live_store.list_entries(scene_id="newly_built_portal_scene")

        manifest_path.write_text(
            json.dumps(
                {
                    "scenes": {
                        "newly_built_portal_scene": {"title": "סצנה חדשה שנבנתה ללא אתחול שרת"}
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        created = live_store.create_entry(
            {
                "author": "טל",
                "category_id": "visual_element",
                "scene_id": "newly_built_portal_scene",
                "comment": "נשלח בהצלחה מיד לאחר עדכון scene_manifest.json ללא אתחול שרת",
            }
        )
        self.assertEqual(created.scene_id, "newly_built_portal_scene")
        listed = live_store.list_entries(scene_id="newly_built_portal_scene")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].entry_id, created.entry_id)

        summary_data = json.loads(live_store.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary_data["open_by_scene"]["newly_built_portal_scene"], 1)


class TestBetaFeedbackHttpAndChromeE2E(unittest.TestCase):
    """Verifies HTTP routes (/wiki, /catalog, /api/*) and Headless Chrome UI + 3D picking & Gyro toggle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.feedback_dir = Path(tempfile.mkdtemp(prefix="tami_beta_e2e_"))
        port = find_free_port(7520)
        cls.server, cls.port, cls.thread = start_server_thread(
            port=port,
            directory=PROTOTYPE_DIR,
            feedback_dir=cls.feedback_dir,
            enable_beta=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.feedback_dir, ignore_errors=True)

    def test_http_endpoints_portals_and_disabled_mode(self) -> None:
        base = f"http://127.0.0.1:{self.port}"
        with urllib.request.urlopen(f"{base}/api/beta-config", timeout=5) as resp:
            cfg_data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(cfg_data["enabled"])

        for route in ("/wiki", "/catalog", "/api/wiki", "/api/catalog"):
            with urllib.request.urlopen(f"{base}{route}", timeout=5) as r:
                self.assertEqual(r.status, 200)
                if route in ("/wiki", "/catalog"):
                    html_body = r.read().decode("utf-8")
                    self.assertIn("תמי-נאוטיקה", html_body)
                    self.assertIn("Tami-Nautica", html_body)

        # Disabled beta server check (clean gift presentation mode)
        dis_port = find_free_port(7560)
        dis_srv, dis_port, _ = start_server_thread(
            port=dis_port, directory=PROTOTYPE_DIR, enable_beta=False
        )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{dis_port}/api/beta-config", timeout=5
            ) as resp2:
                self.assertFalse(json.loads(resp2.read().decode("utf-8"))["enabled"])
            for protected in ("/api/feedback", "/wiki", "/catalog"):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"http://127.0.0.1:{dis_port}{protected}", timeout=5)
                self.assertEqual(ctx.exception.code, 404)
        finally:
            dis_srv.shutdown()
            dis_srv.server_close()

    def test_chrome_ui_element_box_pick_gyro_toggle_and_3d_pin(self) -> None:
        if not os.path.exists(CHROME_BIN):
            raise unittest.SkipTest(f"Chrome binary not found at {CHROME_BIN}")

        cdp_port = find_free_port(9520)
        tmp_profile = tempfile.mkdtemp(prefix="tami_chrome_beta_")
        url = f"http://127.0.0.1:{self.port}/index.html"
        cmd = [
            CHROME_BIN,
            "--headless=new",
            "--disable-extensions",
            f"--user-data-dir={tmp_profile}",
            "--window-size=1440,900",
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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL
        )
        s: socket.socket | None = None
        try:
            tabs = None
            t0 = time.time()
            while time.time() - t0 < 12.0:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=2) as r:
                        cand = json.loads(r.read())
                        if [t for t in cand if "index.html" in t.get("url", "")]:
                            tabs = cand
                            break
                except Exception:
                    time.sleep(0.2)

            self.assertIsNotNone(tabs, "CDP did not expose tabs")
            ws_url = [t for t in (tabs or []) if "index.html" in t.get("url", "")][0][
                "webSocketDebuggerUrl"
            ]
            parts = ws_url.replace("ws://", "").split("/", 1)
            host, p_str = parts[0].split(":")
            s = socket.create_connection((host, int(p_str)), timeout=6)
            upgrade = (
                f"GET /{parts[1]} HTTP/1.1\r\n"
                f"Host: {host}:{p_str}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            s.sendall(upgrade.encode("utf-8"))
            self.assertIn(b"101", s.recv(4096))

            def send_cdp(method: str, params: dict | None = None, msg_id: int = 1) -> None:
                data = json.dumps({"id": msg_id, "method": method, "params": params or {}}).encode(
                    "utf-8"
                )
                frame = bytearray([0x81])
                if len(data) < 126:
                    frame.append(0x80 | len(data))
                elif len(data) < 65536:
                    frame.append(0x80 | 126)
                    frame.extend(struct.pack(">H", len(data)))
                else:
                    frame.append(0x80 | 127)
                    frame.extend(struct.pack(">Q", len(data)))
                mask = b"\x21\x43\x65\x87"
                frame.extend(mask)
                frame.extend(bytearray(b ^ mask[i % 4] for i, b in enumerate(data)))
                s.sendall(bytes(frame))

            def read_cdp(target_id: int, timeout: float = 20.0) -> dict | None:
                s.settimeout(timeout)
                start = time.time()
                while time.time() - start < timeout:
                    hdr = s.recv(2)
                    if not hdr:
                        return None
                    ln = hdr[1] & 0x7F
                    if ln == 126:
                        ln = struct.unpack(">H", s.recv(2))[0]
                    elif ln == 127:
                        ln = struct.unpack(">Q", s.recv(8))[0]
                    buf = b""
                    while len(buf) < ln:
                        chunk = s.recv(ln - len(buf))
                        if not chunk:
                            break
                        buf += chunk
                    parsed = json.loads(buf.decode("utf-8", errors="replace"))
                    if parsed.get("id") == target_id:
                        return parsed
                return None

            mid = 10
            ready = False
            start_wait = time.time()
            while time.time() - start_wait < 25.0:
                mid += 1
                send_cdp(
                    "Runtime.evaluate",
                    {
                        "expression": "JSON.stringify({ready: !!(window.__SPATIAL_MANAGER__ && window.__SPATIAL_MANAGER__.layersReady && window.__BETA_FEEDBACK__ && window.__BETA_FEEDBACK__.config)})",
                        "returnByValue": True,
                    },
                    msg_id=mid,
                )
                resp = read_cdp(mid)
                if resp:
                    val = json.loads(resp.get("result", {}).get("result", {}).get("value", "{}"))
                    if val.get("ready"):
                        ready = True
                        break
                time.sleep(0.2)
            self.assertTrue(ready, "Prototype + BetaFeedbackOverlay did not initialize in time")

            e2e_expr = """
            (async () => {
                const bf = window.__BETA_FEEDBACK__;
                const sm = window.__SPATIAL_MANAGER__;

                // 1. Verify Gyro toggle ON -> OFF (including simulated iOS requestPermission)
                window.DeviceOrientationEvent = window.DeviceOrientationEvent || function() {};
                window.DeviceOrientationEvent.requestPermission = async () => 'granted';
                const gyroOn = await sm.enableGyroscope();
                const gyroOff = await sm.enableGyroscope();

                // 2. Verify UI Element Box-Selection over #gyro-toggle (inside collapsible #spatial-sidebar)
                const sidebarEl = document.getElementById('spatial-sidebar');
                if (sidebarEl) {
                    sidebarEl.style.transition = 'none';
                    document.getElementById('sidebar-toggle-btn').click();
                }
                const gyroBtn = document.getElementById('gyro-toggle');
                gyroBtn.scrollIntoView({ block: 'center', behavior: 'instant' });
                const gr = gyroBtn.getBoundingClientRect();
                bf.startHighlightMode();
                bf._updateLiveHoverInspector(gr.left + gr.width / 2, gr.top + gr.height / 2);
                const hoverInspectorVisible = !document.getElementById('beta-hover-inspector').classList.contains('hidden');
                bf._completeSpatialSelection(gr.left - 8, gr.top - 6, gr.right + 8, gr.bottom + 6);
                bf.authorInput.value = 'טל';
                bf.commentInput.value = 'בדיקת סימון כפתור ג׳יירו בממשק';
                await bf.submitFeedbackEntry();
                if (sidebarEl && !sidebarEl.classList.contains('collapsed')) {
                    document.getElementById('sidebar-close-btn').click();
                }

                // 3. Verify 3D Spatial Region Highlight + Pin Reprojection
                sm.setCameraPose({ x: 40, y: 10, z: 360, yawDeg: -8, pitchDeg: 3 }, true);
                bf.startHighlightMode();
                bf._completeSpatialSelection(560, 340, 760, 520);
                bf.authorInput.value = 'נעמה';
                bf.commentInput.value = 'בדיקת סימון אלמנט תלת-ממדי אוטומטית';
                await bf.submitFeedbackEntry();

                const pins = Array.from(document.querySelectorAll('.beta-pin-marker'));
                const spatialPin = pins.find(p => !p.classList.contains('ui-pin'));
                const pinX1 = spatialPin ? parseFloat(spatialPin.style.left) : -1;
                sm.setCameraPose({ x: -80, y: 0, z: 420, yawDeg: 14, pitchDeg: 0 }, true);
                bf._updateProjectedPinsAndHighlight();
                const pinX2 = spatialPin ? parseFloat(spatialPin.style.left) : -1;

                // 4. Verify clear resolve button labels ('✓ סמן כטופל' -> '↺ סמן כפתוח' + 'טופל ✓' badge)
                const openResolveBtn = bf.entriesListEl.querySelector('.resolve-btn');
                const openBtnLabel = openResolveBtn ? openResolveBtn.textContent.trim() : '';
                await bf.toggleResolveEntry(bf.entries[0]);
                const resolvedCard = bf.entriesListEl.querySelector(`.beta-entry-card[data-entry-id="${bf.entries[0].entry_id}"]`);
                const reopenBtnLabel = resolvedCard && resolvedCard.querySelector('.resolve-btn')
                    ? resolvedCard.querySelector('.resolve-btn').textContent.trim()
                    : '';
                const resolvedTagLabel = resolvedCard && resolvedCard.querySelector('.resolved-status-tag')
                    ? resolvedCard.querySelector('.resolved-status-tag').textContent.trim()
                    : '';

                // 5. Verify live quaternion gyro upward tilt (+pitch) & Android alpha===null persistence
                await sm.enableGyroscope();
                sm._onDeviceOrientation({ alpha: 120, beta: 60, gamma: 0 });
                sm._onDeviceOrientation({ alpha: 120, beta: 72, gamma: 0 });
                sm._onDeviceOrientation({ alpha: null, beta: 72, gamma: 0 });
                const gyroLivePitchDeg = sm._gyroTargetPitchDeg;
                const gyroLiveYawDeg = sm._gyroTargetYawDeg;
                await sm.enableGyroscope();

                return JSON.stringify({
                    gyroOn,
                    gyroOff,
                    hoverInspectorVisible,
                    entriesCount: bf.entries.length,
                    uiHighlightMode: bf.entries[0].highlight.mode,
                    uiSelector: bf.entries[0].highlight.ui_selector,
                    spatialLayerId: bf.entries[1].highlight.layer_id,
                    pinX1,
                    pinX2,
                    openBtnLabel,
                    reopenBtnLabel,
                    resolvedTagLabel,
                    gyroLivePitchDeg,
                    gyroLiveYawDeg
                });
            })()
            """
            mid += 1
            send_cdp(
                "Runtime.evaluate",
                {"expression": e2e_expr, "awaitPromise": True, "returnByValue": True},
                msg_id=mid,
            )
            res = read_cdp(mid, timeout=14.0)
            self.assertIsNotNone(res)
            out = json.loads((res or {}).get("result", {}).get("result", {}).get("value", "{}"))
            self.assertTrue(out["gyroOn"], f"Expected gyroOn=True: {out}")
            self.assertFalse(out["gyroOff"], f"Expected gyroOff=False on 2nd click: {out}")
            self.assertTrue(out["hoverInspectorVisible"], f"Hover inspector did not appear: {out}")
            self.assertEqual(out["entriesCount"], 2)
            self.assertEqual(out["uiHighlightMode"], "ui")
            self.assertEqual(out["uiSelector"], "#gyro-toggle")
            self.assertRegex(out["spatialLayerId"], SAFE_LAYER_ID_RE)
            self.assertNotEqual(out["spatialLayerId"], "ui")
            self.assertGreater(
                abs(out["pinX2"] - out["pinX1"]),
                15.0,
                f"3D pin failed to re-project when camera moved: {out}",
            )
            self.assertEqual(out["openBtnLabel"], "✓ סמן כטופל")
            self.assertEqual(out["reopenBtnLabel"], "↺ סמן כפתוח")
            self.assertEqual(out["resolvedTagLabel"], "טופל ✓")
            self.assertAlmostEqual(out["gyroLivePitchDeg"], 15.0, places=2)
            self.assertAlmostEqual(out["gyroLiveYawDeg"], 0.0, places=2)
        finally:
            if s is not None:
                s.close()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            shutil.rmtree(tmp_profile, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
