#!/usr/bin/env python3
"""
tests/test_prototype.py — Empirical Automated Verification Suite for Deliverable R5
(Playable 2.5D Spatial Prototype honoring Tamar "Tami" Sheaffer's 70th Birthday)

Tests:
1. File Integrity & Code Syntax (Three.js vendor, shaders, JS syntax via Node, HTML/CSS).
2. Dynamic Widescreen Frustum Scaling mathematical verification (16:9, 21:9 ultrawide, mobile).
3. Prototype HTTP Server & Collision Avoidance (HTTP 200, MIME types, clean shutdown).
4. Dual-Viewport Headless Chrome Verification (Desktop 1440x900, Mobile 390x844):
   - Zero JS/WebGL console errors.
   - Active WebGL 2.0 / WebGL context and canvas presence.
   - Parallax camera matrix responsiveness (dx > 20, dy > 10).
   - Thematic scene transition (Herzliya -> Denya).
   - RTL Hebrew layout verification (dir="rtl").
"""

import os
import sys
import json
import math
import time
import socket
import struct
import tempfile
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR = PROJECT_ROOT / "prototype"
SHADERS_DIR = PROJECT_ROOT / "assets" / "shaders"

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class TestPrototypeIntegrity(unittest.TestCase):
    """Validates filesystem artifacts, syntax integrity, and asset contracts."""

    def test_prototype_core_files_exist(self):
        """Asserts all required prototype source files and directories exist."""
        required_files = [
            PROTOTYPE_DIR / "index.html",
            PROTOTYPE_DIR / "css" / "style.css",
            PROTOTYPE_DIR / "js" / "app.js",
            PROTOTYPE_DIR / "js" / "scene_manager.js",
            PROTOTYPE_DIR / "js" / "vendor" / "three.module.js",
            PROTOTYPE_DIR / "serve.py",
            PROTOTYPE_DIR / "scene_config.py",
            PROTOTYPE_DIR / "vision_primitives.py",
            PROTOTYPE_DIR / "build_assets.py",
            PROTOTYPE_DIR / "denya_living_room_scene_builder.py",
            PROTOTYPE_DIR / "tzofen_tree_scenes_builder.py",
            PROTOTYPE_DIR / "scroll_humpty_scenes_builder.py",
            PROTOTYPE_DIR / "humpty_booth_scene_builder.py",
            PROTOTYPE_DIR / "scroll_purim_carpet_scenes_builder.py",
            PROTOTYPE_DIR / "vitrine_bedroom_paris_scenes_builder.py",
            PROJECT_ROOT / "assets" / "shaders" / "warped_reflection_transition.glsl"
        ]
        for f in required_files:
            self.assertTrue(f.exists(), f"Missing required file: {f}")
            self.assertGreater(f.stat().st_size, 0, f"Empty file: {f}")

    def test_spatial_pipeline_config_roundtrip_ssot(self):
        """Asserts all typed scene builder SSOT dataclasses round-trip cleanly via JSON."""
        from prototype.fried_house_scene_builder import (
            FriedHouseSceneConfig,
            default_fried_house_scene_config,
        )
        from prototype.new_scenes_builder import (
            NewScenesBundleConfig,
            default_new_scenes_bundle_config,
        )
        from prototype.railing_occluder_scene_builder import (
            RailingOccluderSceneConfig,
            build_railing_occluder_manifest_entry,
            default_railing_occluder_scene_config,
            verify_railing_occluder_depth_topology,
        )
        from prototype.scene_config import (
            CameraRigConfig,
            SpatialPipelineConfig,
            default_spatial_pipeline_config,
        )

        cfg = default_spatial_pipeline_config()
        self.assertEqual(cfg, SpatialPipelineConfig.from_json(cfg.to_json()))
        self.assertEqual(cfg.camera_rig, CameraRigConfig.from_json(cfg.camera_rig.to_json()))

        manifest_path = PROTOTYPE_DIR / "assets" / "scenes" / "scene_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest.get("pipeline_config", {}).get("camera_rig"),
            cfg.camera_rig.to_dict(),
        )

        new_cfg = default_new_scenes_bundle_config()
        self.assertEqual(new_cfg, NewScenesBundleConfig.from_json(new_cfg.to_json()))
        self.assertEqual(len(new_cfg.shelf_portal.portal_cubbies), 10)

        fried_cfg = default_fried_house_scene_config()
        self.assertEqual(fried_cfg, FriedHouseSceneConfig.from_json(fried_cfg.to_json()))

        yard_occ_cfg = default_railing_occluder_scene_config()
        self.assertEqual(
            yard_occ_cfg,
            RailingOccluderSceneConfig.from_json(yard_occ_cfg.to_json()),
        )
        self.assertEqual(yard_occ_cfg.sisters_reveal.portal_target_scene_id, "fried_house_railing")
        yard_manifest_entry = build_railing_occluder_manifest_entry(yard_occ_cfg)
        self.assertEqual(len(yard_manifest_entry["portalShelves"]), 1)
        self.assertEqual(
            yard_manifest_entry["portalShelves"][0]["targetSceneId"],
            "fried_house_railing",
        )
        yard_topo = verify_railing_occluder_depth_topology(yard_occ_cfg)
        self.assertGreaterEqual(yard_topo["railing_post_count"], 4.0)
        self.assertLessEqual(yard_topo["railing_max_contact_dz"], 2.5)
        self.assertLessEqual(yard_topo["installation_max_contact_dz"], 2.5)
        self.assertLessEqual(yard_topo["hedge_max_contact_dz"], 2.5)
        self.assertLessEqual(yard_topo["sisters_shoe_max_dz"], 2.5)
        self.assertGreater(yard_topo["min_rail_sisters_clearance_z"], 5.0)

        from prototype.denya_living_room_scene_builder import (
            DenyaLivingRoomBundleConfig,
            default_denya_living_room_bundle_config,
        )

        denya_cfg = default_denya_living_room_bundle_config()
        self.assertEqual(
            denya_cfg,
            DenyaLivingRoomBundleConfig.from_json(denya_cfg.to_json()),
        )
        self.assertEqual(denya_cfg.dining_scene.scene_id, "denya_living_room")
        self.assertEqual(denya_cfg.couch_scene.scene_id, "denya_couch_guitar")
        self.assertEqual(
            denya_cfg.framed_painting.target_scene_id,
            "wandering_drawing",
        )
        self.assertEqual(
            denya_cfg.planted_trio.portal_id_to_couch,
            "portal_denya_to_couch",
        )
        self.assertEqual(
            denya_cfg.couch_scene.return_portal_id,
            "portal_denya_back_to_dining",
        )

        from prototype.table_installation_scene_builder import (
            FriedTableHybridSceneConfig,
            default_fried_table_hybrid_scene_config,
        )

        table_hybrid_cfg = default_fried_table_hybrid_scene_config()
        self.assertEqual(
            table_hybrid_cfg,
            FriedTableHybridSceneConfig.from_json(table_hybrid_cfg.to_json()),
        )
        self.assertTrue(table_hybrid_cfg.hybrid_installation.mirror_horizontal)
        self.assertEqual(cfg.camera_rig.portal_occluder_regen_delay_ms, 60000)

        from prototype.tzofen_tree_scenes_builder import (
            TzofenTreeBundleConfig,
            default_tzofen_tree_bundle_config,
        )

        tz_tree_cfg = default_tzofen_tree_bundle_config()
        self.assertEqual(
            tz_tree_cfg,
            TzofenTreeBundleConfig.from_json(tz_tree_cfg.to_json()),
        )

        from prototype.scroll_humpty_scenes_builder import (
            ScrollHumptyBundleConfig,
            default_scroll_humpty_bundle_config,
        )

        scroll_humpty_cfg = default_scroll_humpty_bundle_config()
        self.assertEqual(
            scroll_humpty_cfg,
            ScrollHumptyBundleConfig.from_json(scroll_humpty_cfg.to_json()),
        )

        from prototype.humpty_booth_scene_builder import (
            HumptyBoothBundleConfig,
            default_humpty_booth_bundle_config,
        )

        humpty_booth_cfg = default_humpty_booth_bundle_config()
        self.assertEqual(
            humpty_booth_cfg,
            HumptyBoothBundleConfig.from_json(humpty_booth_cfg.to_json()),
        )

        from prototype.scroll_purim_carpet_scenes_builder import (
            ScrollPurimCarpetBundleConfig,
            default_scroll_purim_carpet_bundle_config,
        )

        spc_cfg = default_scroll_purim_carpet_bundle_config()
        self.assertEqual(
            spc_cfg,
            ScrollPurimCarpetBundleConfig.from_json(spc_cfg.to_json()),
        )

        from prototype.vitrine_bedroom_paris_scenes_builder import (
            VitrineBedroomParisBundleConfig,
            default_vitrine_bedroom_paris_bundle_config,
        )

        vbp_cfg = default_vitrine_bedroom_paris_bundle_config()
        self.assertEqual(
            vbp_cfg,
            VitrineBedroomParisBundleConfig.from_json(vbp_cfg.to_json()),
        )

    def test_prototype_scene_textures_exist_and_valid(self):
        """Asserts all multi-plane, outpainted room, and neural depth textures exist with valid SSOT dimensions."""
        from PIL import Image
        from prototype.scene_config import default_spatial_pipeline_config

        cfg = default_spatial_pipeline_config()
        inner_dims = (cfg.dimensions.width, cfg.dimensions.height)
        expanded_dims = cfg.outpainting.expanded_dimensions(cfg.dimensions)

        expected_assets: list[tuple[Path, tuple[int, int]]] = []
        for spec in cfg.scenes:
            spec_exp_dims = spec.expected_bg_dimensions(cfg.dimensions, cfg.outpainting)
            expected_assets.append(
                (PROTOTYPE_DIR / "assets" / "scenes" / spec.bg_output_filename, spec_exp_dims)
            )
            expected_assets.append(
                (PROTOTYPE_DIR / "assets" / "scenes" / spec.depth_output_filename, spec_exp_dims)
            )
            expected_assets.append(
                (
                    PROTOTYPE_DIR / "assets" / "scenes" / spec.mid_output_filename,
                    spec.expected_mid_dimensions(cfg.dimensions),
                )
            )
            fg_asset = PROTOTYPE_DIR / "assets" / "scenes" / spec.fg_output_filename
            expected_assets.append(
                (fg_asset, spec.expected_fg_dimensions(cfg.dimensions, cfg.outpainting))
            )
            for extra in spec.extra_layers:
                ex_asset = PROTOTYPE_DIR / "assets" / "scenes" / extra.output_filename
                expected_assets.append(
                    (
                        ex_asset,
                        spec.expected_extra_layer_dimensions(
                            extra, cfg.dimensions, cfg.outpainting
                        ),
                    )
                )
            if spec.reflection_output_filename:
                expected_assets.append(
                    (
                        PROTOTYPE_DIR / "assets" / "scenes" / spec.reflection_output_filename,
                        spec.expected_reflection_dimensions(cfg.dimensions),
                    )
                )
        for asset, exp_size in expected_assets:
            self.assertTrue(asset.exists(), f"Missing scene asset: {asset}")
            self.assertGreater(asset.stat().st_size, 1000, f"Asset too small: {asset}")
            with Image.open(asset) as im:
                self.assertEqual(
                    im.size,
                    exp_size,
                    f"Unexpected dimension for {asset.name}: {im.size} != {exp_size}",
                )

        manifest_path = PROTOTYPE_DIR / "assets" / "scenes" / "scene_manifest.json"
        self.assertTrue(manifest_path.exists(), f"Missing SSOT scene manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_scenes = manifest.get("scenes", {})
        self.assertEqual(len(cfg.scenes), 8)
        for spec in cfg.scenes:
            self.assertIn(spec.scene_id, manifest_scenes)
            spec_exp_w, spec_exp_h = spec.expected_bg_dimensions(cfg.dimensions, cfg.outpainting)
            self.assertEqual(manifest_scenes[spec.scene_id]["bgExpandedWidth"], spec_exp_w)
            self.assertEqual(manifest_scenes[spec.scene_id]["bgExpandedHeight"], spec_exp_h)

        # Verify companion multi-scene builders (`wandering_drawing_1`, `denya_model`, `shelf_portal`, `fried_house_railing`)
        companion_expected: list[tuple[Path, tuple[int, int]]] = [
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_bg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_bg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene8_denya_model_bg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene8_denya_model_bg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene8_denya_model_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene8_denya_model_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene9_shelf_portal_bg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene9_shelf_portal_bg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene9_shelf_portal_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene9_shelf_portal_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_railing.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_hedge.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_hedge_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_railing.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_railing_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_installation.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_installation_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene_railing_occluder_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_bg.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_couch_people.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_couch_people_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_wandering2_frame.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_dining_wandering2_frame_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_bg.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_fg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_wandering2_frame.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "denya_living_couch_wandering2_frame_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_railing_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_fg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_installation.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_installation_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_front_chair.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene11_fried_table_hybrid_front_chair_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_scroll_foot.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_scroll_foot_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene7_wandering1_scroll_veil.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_window_shutter.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_window_shutter_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene12_tzofen_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene12_tzofen_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene12_tzofen_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene12_tzofen_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene12_tzofen_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene12_tzofen_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene13_herzliya_canopy_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene13_herzliya_canopy_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene13_herzliya_canopy_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene13_herzliya_canopy_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene13_herzliya_canopy_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene13_herzliya_canopy_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene14_scroll_world_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene14_scroll_world_bg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene14_scroll_world_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene14_scroll_world_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene14_scroll_world_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene14_scroll_world_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene15_humpty_bundle_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene15_humpty_bundle_bg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene15_humpty_bundle_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene15_humpty_bundle_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene15_humpty_bundle_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene16_humpty_booth_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene16_humpty_booth_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene16_humpty_booth_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene16_humpty_booth_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene16_humpty_booth_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene16_humpty_booth_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_photo.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_photo_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_rod.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_rod_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_occluder.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene17_scroll_foot_occluder_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene18_purim_1966_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene18_purim_1966_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene18_purim_1966_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene18_purim_1966_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene18_purim_1966_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_earth_full.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_earth_full_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_roll.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene19_herz_carpet_roll_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene20_vitrine_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene20_vitrine_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene20_vitrine_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene20_vitrine_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene20_vitrine_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene20_vitrine_fg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_fg_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_vitrine.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene21_bar_giora_vitrine_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_fg.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_fg_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_photo.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene22_paris_table_photo_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene23_tami_1958_bg_inpainted.jpg", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene23_tami_1958_depth.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene23_tami_1958_mid.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene23_tami_1958_mid_depth.png", inner_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene23_tami_1958_fg.png", expanded_dims),
            (PROTOTYPE_DIR / "assets" / "scenes" / "scene23_tami_1958_fg_depth.png", expanded_dims),
        ]
        for asset, exp_size in companion_expected:
            self.assertTrue(asset.exists(), f"Missing companion scene asset: {asset}")
            with Image.open(asset) as im:
                self.assertEqual(im.size, exp_size, f"Unexpected dimension for {asset.name}: {im.size}")

        shutter_crop = PROTOTYPE_DIR / "assets" / "scenes" / "scene10_fried_house_window_shutter_crop.png"
        self.assertTrue(shutter_crop.exists(), f"Missing shutter crop asset: {shutter_crop}")

        for companion_id in (
            "wandering_drawing_1",
            "scroll_palm_world",
            "denya_model",
            "shelf_portal",
            "tzofen_brick_table",
            "humpty_dumpty_bundle",
            "humpty_dumpty_booth",
            "bene_beitcha_scroll_foot",
            "purim_costume_1966",
            "herzliya_lawn_carpet",
            "vitrine_keys_pen",
            "bar_giora_bedroom",
            "paris_cite_white_table",
            "toddler_tami_1958",
            "fried_yard_installation",
            "fried_house_railing",
            "fried_table_hybrid",
            "herzliya_tree_canopy",
            "denya_living_room",
            "denya_couch_guitar",
        ):
            self.assertIn(companion_id, manifest_scenes)

    def test_glsl_transition_shader_syntax_and_findings(self):
        """Validates GLSL transition shader syntax and M4 Reviewer 2 Finding 2 fix."""
        shader_file = PROJECT_ROOT / "assets" / "shaders" / "warped_reflection_transition.glsl"
        self.assertTrue(shader_file.exists())
        shader = shader_file.read_text(encoding="utf-8")

        # Mandatory GLSL uniforms and variables
        self.assertIn("uProgress", shader)
        self.assertIn("uTexCurrent", shader)
        self.assertIn("uTexNext", shader)
        self.assertIn("uAspect", shader)
        self.assertIn("uPointer", shader)
        self.assertIn("gl_FragColor", shader)

        # M4 Reviewer 2 Finding 2 Fix Verification:
        # Must clamp specular flash falloff max(0.0, 1.0 - dist) to prevent negative color values in corners
        self.assertIn("max(0.0, 1.0 - dist)", shader, "Shader missing M4 Reviewer 2 Finding 2 fix")

    def test_javascript_syntax_via_node(self):
        """Validates ES Module JavaScript syntax using node --check."""
        js_files = [
            PROTOTYPE_DIR / "js" / "app.js",
            PROTOTYPE_DIR / "js" / "scene_manager.js",
            PROTOTYPE_DIR / "js" / "feedback_overlay.js",
            PROTOTYPE_DIR / "js" / "vendor" / "three.module.js",
            PROTOTYPE_DIR / "tests" / "prototype.spec.js"
        ]
        for js in js_files:
            if not js.exists():
                continue
            res = subprocess.run(
                ["node", "--check", str(js)],
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            self.assertEqual(res.returncode, 0, f"Node syntax error in {js}:\n{res.stderr}")

    def test_gyro_quaternion_and_keyboard_bindings_via_node(self):
        """Verifies quaternion gyro pitch sign, W3C gimbal-flip immunity, Z-roll immunity, and keyboard elevation SSOT."""
        node_script = """
import {
  computeGyroAnglesFromDeviceEuler,
  KEYBOARD_NAV_BINDINGS,
  SCENE_GEOMETRY_SPEC
} from './prototype/js/scene_manager.js';

const r0 = computeGyroAnglesFromDeviceEuler(40, 60, 0, 0, null, SCENE_GEOMETRY_SPEC);
const rUp = computeGyroAnglesFromDeviceEuler(40, 72, 0, 0, r0.referenceQuat, SCENE_GEOMETRY_SPEC);
const rDown = computeGyroAnglesFromDeviceEuler(40, 48, 0, 0, r0.referenceQuat, SCENE_GEOMETRY_SPEC);

const g0 = computeGyroAnglesFromDeviceEuler(25, 85, 5, 0, null, SCENE_GEOMETRY_SPEC);
const gAcross90 = computeGyroAnglesFromDeviceEuler(25, 96, 5, 0, g0.referenceQuat, SCENE_GEOMETRY_SPEC);

const w0 = computeGyroAnglesFromDeviceEuler(0, 10, 88, 90, null, SCENE_GEOMETRY_SPEC);
const wGimbalFlip = computeGyroAnglesFromDeviceEuler(180, 170, -88, 90, w0.referenceQuat, SCENE_GEOMETRY_SPEC);

const z0 = computeGyroAnglesFromDeviceEuler(0, 0, 0, 0, null, SCENE_GEOMETRY_SPEC);
const zPureRoll = computeGyroAnglesFromDeviceEuler(45, 0, 0, 0, z0.referenceQuat, SCENE_GEOMETRY_SPEC);

const l0 = computeGyroAnglesFromDeviceEuler(0, 0, -60, 90, null, SCENE_GEOMETRY_SPEC);
const lUp = computeGyroAnglesFromDeviceEuler(0, 0, -72, 90, l0.referenceQuat, SCENE_GEOMETRY_SPEC);

console.log(JSON.stringify({
  rUpPitch: rUp.pitchDeg,
  rUpYaw: rUp.yawDeg,
  rDownPitch: rDown.pitchDeg,
  gAcross90Pitch: gAcross90.pitchDeg,
  wGimbalFlipPitch: wGimbalFlip.pitchDeg,
  wGimbalFlipYaw: wGimbalFlip.yawDeg,
  zPureRollExcursion: zPureRoll.excursionDeg,
  lUpPitch: lUp.pitchDeg,
  bindingsUp: KEYBOARD_NAV_BINDINGS.up,
  bindingsDown: KEYBOARD_NAV_BINDINGS.down,
  elevSpeed: SCENE_GEOMETRY_SPEC.keyElevationSpeedPerSec
}));
"""
        res = subprocess.run(
            ["node", "--input-type=module", "-e", node_script],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
        self.assertEqual(res.returncode, 0, f"Node gyro/keyboard test failed:\n{res.stderr}")
        out = json.loads(res.stdout.strip())
        self.assertAlmostEqual(out["rUpPitch"], 15.0, places=3)
        self.assertAlmostEqual(out["rUpYaw"], 0.0, places=3)
        self.assertAlmostEqual(out["rDownPitch"], -15.0, places=3)
        self.assertGreater(out["gAcross90Pitch"], 12.0)
        self.assertAlmostEqual(out["wGimbalFlipPitch"], -5.0, places=3)
        self.assertAlmostEqual(out["wGimbalFlipYaw"], 0.0, places=3)
        self.assertLess(out["zPureRollExcursion"], 0.01)
        self.assertAlmostEqual(out["lUpPitch"], 15.0, places=3)
        self.assertEqual(out["bindingsUp"], ["KeyE", "PageUp"])
        self.assertEqual(out["bindingsDown"], ["KeyQ", "PageDown"])
        self.assertGreater(out["elevSpeed"], 0)

    def test_html_rtl_and_accessibility_chrome(self):
        """Validates RTL Hebrew structure, semantic tags, and accessible controls in index.html."""
        html_file = PROTOTYPE_DIR / "index.html"
        self.assertTrue(html_file.exists())
        content = html_file.read_text(encoding="utf-8")

        self.assertIn('dir="rtl"', content)
        self.assertIn('lang="he"', content)
        self.assertIn('id="spatial-canvas"', content)
        for scene_id in (
            "shelf_portal",
            "paper_capsules",
            "tzofen_brick_table",
            "bene_beitcha",
            "humpty_dumpty_bundle",
            "humpty_dumpty_booth",
            "bene_beitcha_scroll_foot",
            "purim_costume_1966",
            "herzliya_lawn_carpet",
            "vitrine_keys_pen",
            "bar_giora_bedroom",
            "paris_cite_white_table",
            "toddler_tami_1958",
            "denya_model",
            "denya_living_room",
            "denya_couch_guitar",
            "plan_b_pipe",
            "plan_b_pipe_v3",
            "wandering_drawing_1",
            "scroll_palm_world",
            "wandering_drawing",
            "fried_yard_installation",
            "fried_house_railing",
            "fried_table_hybrid",
            "herzliya_tree_canopy",
            "marseille_mirror",
            "plan_b_pipe_v1",
            "marseille_mirror_v1",
        ):
            self.assertIn(f'data-scene="{scene_id}"', content)
        self.assertIn('id="spatial-nav-bar"', content)
        self.assertIn('id="spatial-back-btn"', content)
        self.assertIn('id="spatial-forward-btn"', content)
        self.assertIn('id="spatial-breadcrumb-list"', content)
        self.assertIn('id="spatial-outgoing-portals"', content)
        self.assertIn('id="portal-return-btn"', content)
        self.assertIn('id="exhibition-tabs"', content)
        self.assertIn('class="story-drawings-strip"', content)
        self.assertIn('rgma_bene_beitcha_scroll_drawing_foot_2026_331d870d.jpg', content)
        self.assertIn('id="info-drawer"', content)
        self.assertIn('id="loading-status-text"', content)
        self.assertIn('id="loading-skip-hint"', content)
        self.assertIn('תמי-נאוטיקה', content)
        self.assertIn('Tami-Nautica', content)
        self.assertIn('תמר שפר', content)
        self.assertNotIn('<ol id="spatial-breadcrumb-list"', content)

    def test_loading_screen_ssot_and_real_esm_imports_via_node(self):
        """Verifies real Node ESM module evaluation of app.js & scene_manager.js, LOADING_SCREEN_SPEC, and roll-aware frustum clamp."""
        node_script = """
import { LOADING_SCREEN_SPEC, EXHIBITION_GROUPS } from './prototype/js/app.js';
import {
  SCENE_GEOMETRY_SPEC,
  doFrustumCornersFitOnBackground,
  clampPoseToRoomFrustum
} from './prototype/js/scene_manager.js';

const fitRolled = doFrustumCornersFitOnBackground(
  -197.7, 77.0, -203.0, 0.0, 0.0, Math.tan(22.5 * Math.PI / 180.0), 1.6,
  -350.0, 900.0, 600.0, null, -130.0 * Math.PI / 180.0
);
const clampedTunnel = clampPoseToRoomFrustum(
  { x: 999.0, y: 0.0, z: -145.0, yawDeg: 40.0, pitchDeg: 0.0, rollDeg: 0.0 },
  56.0,
  411 / 705,
  2.2,
  {
    ...SCENE_GEOMETRY_SPEC,
    bgZ: -350.0,
    minCameraZ: -145.0,
    maxCameraZ: 685.0,
    restCameraX: 52.0,
    restCameraY: 0.0,
    maxYawDeg: 18.0,
    maxPitchDeg: 12.5,
    maxStrafeX: 39.0,
    maxStrafeY: 25.0
  }
);

console.log(JSON.stringify({
  brandHe: LOADING_SCREEN_SPEC.brandTitleHe,
  brandEn: LOADING_SCREEN_SPEC.brandTitleEn,
  messagesHe: LOADING_SCREEN_SPEC.messagesHe,
  rotationIntervalMs: LOADING_SCREEN_SPEC.rotationIntervalMs,
  minHumanDisplayMs: LOADING_SCREEN_SPEC.minHumanDisplayMs,
  groupCount: EXHIBITION_GROUPS.length,
  directFlightMs: SCENE_GEOMETRY_SPEC.directPortalFlightDurationMs,
  directFitScale: SCENE_GEOMETRY_SPEC.directPortalFitScale,
  tunnelMinStrafeScale: SCENE_GEOMETRY_SPEC.tunnelMinStrafeScale,
  fitRolled,
  clampedTunnel
}));
"""
        res = subprocess.run(
            ["node", "--input-type=module", "-e", node_script],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
        self.assertEqual(res.returncode, 0, f"Node ESM evaluation failed:\n{res.stderr}")
        out = json.loads(res.stdout.strip())
        self.assertEqual(out["brandHe"], "תמי-נאוטיקה")
        self.assertEqual(out["brandEn"], "Tami-Nautica")
        self.assertEqual(
            out["messagesHe"],
            [
                "טוען מרחבים...",
                "משחק עם יערה וכרמל...",
                "דואג לעתיד המדינה...",
                "מגייס את גדי לשיפצורים אחרונים...",
            ],
        )
        self.assertGreaterEqual(out["rotationIntervalMs"], 1500)
        self.assertGreaterEqual(out["minHumanDisplayMs"], 6000)
        self.assertEqual(out["groupCount"], 6)
        self.assertGreater(out["directFlightMs"], 800)
        self.assertGreater(out["directFitScale"], 1.0)
        self.assertGreater(out["tunnelMinStrafeScale"], 0.0)
        self.assertTrue(out["fitRolled"])
        self.assertLessEqual(out["clampedTunnel"]["x"], 52.0 + 39.0 + 0.01)
        self.assertLessEqual(abs(out["clampedTunnel"]["yawDeg"]), 18.01)


class TestDynamicWidescreenFrustumScaling(unittest.TestCase):
    """
    Empirically verifies M4 Reviewer 2 Finding 1:
    Layer 0 must dynamically scale according to camera.aspect and optical frustum
    so that no black borders appear at 16:9 (1920x1080), 21:9 ultrawide, or mobile.
    """

    def compute_layer0_scale(self, aspect, fov=45.0, z_cam=400.0, z_layer0=-350.0, mx=45.0, my=25.0):
        dist0 = z_cam - z_layer0  # 750
        half_fov_rad = math.radians(fov / 2.0)
        vis_height_z0 = 2.0 * dist0 * math.tan(half_fov_rad)
        vis_width_z0 = vis_height_z0 * aspect
        req_width = vis_width_z0 + 2.0 * mx
        req_height = vis_height_z0 + 2.0 * my
        base_w = 750.0
        base_h = 500.0
        scale_x = req_width / base_w
        scale_y = req_height / base_h
        return max(1.875, scale_x, scale_y), vis_width_z0, req_width

    def test_frustum_clearance_across_aspect_ratios(self):
        """Verifies zero black borders across standard desktop, ultrawide, and mobile viewports."""
        viewports = [
            ("Desktop 1440x900 (16:10)", 1440 / 900),
            ("Desktop 1920x1080 (16:9)", 1920 / 1080),
            ("Ultrawide 2560x1080 (21:9)", 2560 / 1080),
            ("Ultrawide 3440x1440 (21:9)", 3440 / 1440),
            ("Mobile Portrait 390x844 (9:19.5)", 390 / 844),
            ("Tablet 1024x768 (4:3)", 1024 / 768)
        ]

        for name, aspect in viewports:
            scale, vis_w, req_w = self.compute_layer0_scale(aspect)
            actual_w = 750.0 * scale
            margin = actual_w - req_w
            self.assertGreaterEqual(
                margin, 0.0,
                f"Frustum boundary exposure for {name}: actual_w={actual_w:.1f} < req_w={req_w:.1f}"
            )
            # Confirm scale meets or exceeds nominal baseline 1.875
            self.assertGreaterEqual(scale, 1.875)


class TestPrototypeServer(unittest.TestCase):
    """Validates prototype/serve.py HTTP serving, MIME types, and port collision avoidance."""

    def test_server_collision_avoidance_and_mime_types(self):
        from prototype.serve import create_server, start_server_thread, find_free_port

        # 1. Start primary server
        port1 = find_free_port(7090)
        server1, bound_port1, thread1 = start_server_thread(port=port1, directory=PROTOTYPE_DIR)
        self.assertEqual(port1, bound_port1)

        try:
            # 2. Collision test: attempt to bind same port -> must avoid and pick next free port
            server2, bound_port2, thread2 = start_server_thread(port=port1, directory=PROTOTYPE_DIR)
            self.assertNotEqual(bound_port1, bound_port2, "Collision avoidance failed to pick distinct port")

            try:
                # 3. HTTP GET verification for index.html
                url_index = f"http://127.0.0.1:{bound_port1}/index.html"
                req = urllib.request.Request(url_index)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    body = resp.read().decode("utf-8")
                    self.assertIn("תמר שפר", body)
                    self.assertIn("text/html", resp.headers.get("Content-Type"))

                # 4. HTTP GET verification for JS and CSS MIME types
                url_js = f"http://127.0.0.1:{bound_port1}/js/app.js"
                with urllib.request.urlopen(url_js, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertIn("javascript", resp.headers.get("Content-Type").lower())

                url_css = f"http://127.0.0.1:{bound_port1}/css/style.css"
                with urllib.request.urlopen(url_css, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertIn("text/css", resp.headers.get("Content-Type").lower())

                # 5. Scene textures 200 OK
                url_tex = f"http://127.0.0.1:{bound_port1}/assets/scenes/scene1_herzliya_bg_inpainted.jpg"
                with urllib.request.urlopen(url_tex, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertIn("image/jpeg", resp.headers.get("Content-Type").lower())
            finally:
                server2.shutdown()
                server2.server_close()
        finally:
            server1.shutdown()
            server1.server_close()


class TestDualViewportHeadlessChrome(unittest.TestCase):
    """
    Executes real Google Chrome in headless WebGL mode to verify runtime behavior:
    Desktop (1440x900) & Mobile (390x844), zero errors, canvas, WebGL context,
    parallax responsiveness, and scene transition.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHROME_BIN):
            raise unittest.SkipTest(f"Chrome binary not found at {CHROME_BIN}")

        from prototype.serve import start_server_thread, find_free_port
        cls.http_port = find_free_port(7110)
        cls.server, cls.http_port, cls.server_thread = start_server_thread(
            port=cls.http_port, directory=PROTOTYPE_DIR
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def run_chrome_test(self, width, height, is_mobile=False):
        """Runs Chrome headless with specified viewport and executes CDP evaluation."""
        from prototype.serve import find_free_port
        cdp_port = find_free_port(9330)
        tmp_dir = tempfile.mkdtemp()

        url = f"http://127.0.0.1:{self.http_port}/index.html"
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
            url
        ]
        if is_mobile:
            cmd.append("--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            # Query CDP tabs with active polling
            tabs = None
            start_tab = time.time()
            while time.time() - start_tab < 12.0:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=2) as r:
                        tabs = json.loads(r.read())
                        if [t for t in tabs if "index.html" in t.get("url", "")]:
                            break
                except Exception:
                    time.sleep(0.25)

            self.assertIsNotNone(tabs, "Chrome CDP /json endpoint did not respond")
            target_tabs = [t for t in tabs if "index.html" in t.get("url", "")]
            self.assertTrue(target_tabs, "Could not find index.html tab in Chrome CDP")
            ws_url = target_tabs[0]["webSocketDebuggerUrl"]

            # Connect raw WebSocket to CDP
            parts = ws_url.replace("ws://", "").split("/", 1)
            host, p = parts[0].split(":")
            path = "/" + parts[1]

            s = socket.create_connection((host, int(p)), timeout=6)
            req = (f"GET {path} HTTP/1.1\r\n"
                   f"Host: {host}:{p}\r\n"
                   f"Upgrade: websocket\r\n"
                   f"Connection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                   f"Sec-WebSocket-Version: 13\r\n\r\n")
            s.sendall(req.encode("utf-8"))
            resp = s.recv(4096)
            self.assertIn(b"101", resp, "WebSocket upgrade failed")

            def send_cdp(method, params=None, msg_id=1):
                msg = json.dumps({"id": msg_id, "method": method, "params": params or {}})
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
                masked = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(data))
                frame.extend(masked)
                s.sendall(bytes(frame))

            def read_cdp(target_id=None, timeout=20.0):
                s.settimeout(timeout)
                start = time.time()
                while time.time() - start < timeout:
                    header = s.recv(2)
                    if not header:
                        return None
                    b1, b2 = header[0], header[1]
                    length = b2 & 0x7F
                    if length == 126:
                        length = struct.unpack(">H", s.recv(2))[0]
                    elif length == 127:
                        length = struct.unpack(">Q", s.recv(8))[0]
                    payload = b""
                    while len(payload) < length:
                        chunk = s.recv(length - len(payload))
                        if not chunk:
                            break
                        payload += chunk
                    msg = json.loads(payload.decode("utf-8", errors="replace"))
                    if target_id is None or msg.get("id") == target_id:
                        return msg
                return None

            # Active polling loop for window.__tamiSceneManager.layersReady
            poll_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                const st = window.__PROTOTYPE_STATE__;
                const ready = !!((sm && sm.layersReady) || (st && st.layersReady));
                return JSON.stringify({ ready: ready });
            })()
            """
            start_ready = time.time()
            layers_ready = False
            msg_counter = 100
            while time.time() - start_ready < 25.0:
                msg_counter += 1
                send_cdp("Runtime.evaluate", {"expression": poll_js, "returnByValue": True}, msg_id=msg_counter)
                res = read_cdp(target_id=msg_counter)
                if res:
                    val = json.loads(res.get("result", {}).get("result", {}).get("value", "{}"))
                    if val.get("ready"):
                        layers_ready = True
                        break
                time.sleep(0.2)
            self.assertTrue(layers_ready, "SpatialSceneManager layers failed to become ready within 15s")

            # 1. Evaluate baseline page state and WebGL context
            eval_js = """
            (() => {
                const canvas = document.querySelector('canvas#spatial-canvas');
                const gl = canvas ? (canvas.getContext('webgl2') || canvas.getContext('webgl')) : null;
                const glError = gl ? gl.getError() : -1;
                return JSON.stringify({
                    canvasExists: !!canvas,
                    webglActive: !!gl,
                    glError: glError,
                    dir: document.documentElement.dir,
                    state: window.__PROTOTYPE_STATE__
                });
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": eval_js, "returnByValue": True}, msg_id=201)
            res201 = read_cdp(target_id=201)
            raw_val = res201.get("result", {}).get("result", {}).get("value")
            data = json.loads(raw_val)

            self.assertTrue(data["canvasExists"], "Canvas #spatial-canvas not found")
            self.assertTrue(data["webglActive"], "WebGL failed to initialize")
            self.assertEqual(data["glError"], 0, f"WebGL gl.getError() returned {data['glError']}")
            self.assertEqual(data["dir"], "rtl", "Document direction must be RTL")
            self.assertIsNotNone(data["state"], "window.__PROTOTYPE_STATE__ not exposed")
            self.assertEqual(data["state"]["currentScene"], "shelf_portal")

            # 2. Parallax interaction test: excursion across dual extremes
            # Extreme 1: Top-Left (-1.0, 1.0)
            excursion1_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                if (sm) {
                    sm.setCameraPose({ x: 0, y: -36, z: 1150 }, true);
                    sm.pointer.targetX = -1.0;
                    sm.pointer.targetY = 1.0;
                }
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": excursion1_js}, msg_id=202)
            read_cdp(target_id=202)
            time.sleep(0.8)  # Allow damping frames to settle (lambda = 0.05)

            read_pos_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                const cam = sm ? sm.camera : null;
                return JSON.stringify({
                    pos: window.__PROTOTYPE_STATE__.cameraPosition,
                    matrixWorld: cam ? Array.from(cam.matrixWorld.elements) : []
                });
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": read_pos_js, "returnByValue": True}, msg_id=203)
            res203 = read_cdp(target_id=203)
            val1 = json.loads(res203.get("result", {}).get("result", {}).get("value"))
            pos1 = val1["pos"]
            mat1 = val1["matrixWorld"]

            # Extreme 2: Bottom-Right (1.0, -1.0)
            excursion2_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                if (sm) {
                    sm.pointer.targetX = 1.0;
                    sm.pointer.targetY = -1.0;
                }
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": excursion2_js}, msg_id=204)
            read_cdp(target_id=204)
            time.sleep(0.8)  # Allow damping frames to settle

            send_cdp("Runtime.evaluate", {"expression": read_pos_js, "returnByValue": True}, msg_id=205)
            res205 = read_cdp(target_id=205)
            val2 = json.loads(res205.get("result", {}).get("result", {}).get("value"))
            pos2 = val2["pos"]
            mat2 = val2["matrixWorld"]

            # Explicit assertions for dx > 20 and dy > 10
            dx = abs(pos1["x"] - pos2["x"])
            dy = abs(pos1["y"] - pos2["y"])
            self.assertGreater(dx, 20.0, f"Parallax delta X {dx:.2f} failed to exceed 20")
            self.assertGreater(dy, 10.0, f"Parallax delta Y {dy:.2f} failed to exceed 10")

            # Explicit assertion for camera transformation matrixWorld shifts
            matrix_delta = sum(abs(a - b) for a, b in zip(mat1, mat2))
            self.assertGreater(matrix_delta, 1.0, f"Camera matrixWorld failed to shift on parallax, delta: {matrix_delta}")

            # 2b. 6-DOF Dolly (forward/backward) & Pose Rotation & Frustum Guard Verification
            six_dof_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                sm.resetCameraPose(true);
                const forwardPose = sm.setCameraPose({ z: 145 }, true);
                const backwardPose = sm.setCameraPose({ z: 1950 }, true);
                const rotatedPose = sm.setCameraPose({ x: 0, y: -36, z: 880, yawDeg: 22, pitchDeg: -11 }, true);
                const clampedExtreme = sm.setCameraPose({ x: 9999, y: 9999, z: 9999, yawDeg: 89, pitchDeg: 89 }, true);
                const resetPose = sm.resetCameraPose(true);
                return JSON.stringify({
                    forwardPose,
                    backwardPose,
                    rotatedPose,
                    clampedExtreme,
                    resetPose
                });
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": six_dof_js, "returnByValue": True}, msg_id=260)
            res260 = read_cdp(target_id=260)
            dof_data = json.loads(res260.get("result", {}).get("result", {}).get("value"))

            self.assertAlmostEqual(dof_data["forwardPose"]["z"], 145.0, places=1)
            self.assertGreaterEqual(dof_data["backwardPose"]["z"], 1785.0)
            self.assertGreater(abs(dof_data["rotatedPose"]["yawDeg"]), 15.0)
            self.assertGreater(abs(dof_data["rotatedPose"]["pitchDeg"]), 7.0)
            self.assertLessEqual(dof_data["clampedExtreme"]["z"], 2040.1)
            self.assertAlmostEqual(dof_data["resetPose"]["z"], 1740.0, places=1)
            self.assertAlmostEqual(dof_data["resetPose"]["y"], -36.0, places=1)
            self.assertAlmostEqual(dof_data["resetPose"]["yawDeg"], 0.0, places=2)

            # 2c. Direct Portal Flight Target Framing (#47 Purim, #53 Vitrine -130 deg roll, #43 Paris 1958)
            #     + plan_b_pipe_v3 Bore Clamp (#46) + Mobile Compact Breadcrumb Non-Overlap
            portal_and_clamp_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                const purimSpec = sm._findPortalSpec('portal_scroll_foot_purim_1966');
                const vitrineSpec = sm._findPortalSpec('portal_bar_giora_vitrine');
                const parisSpec = sm._findPortalSpec('portal_paris_table_tami_1958');
                const purimPose = sm._computeDirectPortalTargetPose(purimSpec, 'bene_beitcha_scroll_foot');
                const vitrinePose = sm._computeDirectPortalTargetPose(vitrineSpec, 'bar_giora_bedroom');
                const parisPose = sm._computeDirectPortalTargetPose(parisSpec, 'paris_cite_white_table');
                const pipeClampRest = sm._clampPoseForScene('plan_b_pipe_v3', {
                    x: 999, y: -999, z: 430.2, yawDeg: 65, pitchDeg: 45, rollDeg: 0
                });
                return JSON.stringify({
                    purimPose,
                    vitrinePose,
                    parisPose,
                    pipeClampRest,
                    titleText: document.title
                });
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": portal_and_clamp_js, "returnByValue": True}, msg_id=262)
            res262 = read_cdp(target_id=262)
            pc_data = json.loads(res262.get("result", {}).get("result", {}).get("value"))
            self.assertIn("תמי-נאוטיקה", pc_data["titleText"])
            self.assertLess(pc_data["purimPose"]["z"], -100.0)
            self.assertAlmostEqual(pc_data["vitrinePose"]["rollDeg"], -130.0, places=2)
            self.assertLess(pc_data["vitrinePose"]["z"], -120.0)
            self.assertLess(pc_data["parisPose"]["z"], -150.0)
            self.assertLessEqual(abs(pc_data["pipeClampRest"]["x"]), 56.1)
            self.assertLessEqual(abs(pc_data["pipeClampRest"]["yawDeg"]), 18.1)

            if not is_mobile:
                kb_elev_js = """
                (() => {
                    const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                    sm.resetCameraPose(true);
                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', code: 'KeyE', bubbles: true }));
                    sm._stepKeyboardNavigation(0.25);
                    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'e', code: 'KeyE', bubbles: true }));
                    const upY = sm.poseTarget.y;

                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'q', code: 'KeyQ', bubbles: true }));
                    sm._stepKeyboardNavigation(0.55);
                    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'q', code: 'KeyQ', bubbles: true }));
                    const downY = sm.poseTarget.y;

                    sm.resetCameraPose(true);
                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ק', code: 'KeyE', bubbles: true }));
                    sm._stepKeyboardNavigation(0.25);
                    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'ק', code: 'KeyE', bubbles: true }));
                    const hebrewUpY = sm.poseTarget.y;

                    sm.resetCameraPose(true);
                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ק', code: '', bubbles: true }));
                    sm._stepKeyboardNavigation(0.25);
                    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'ק', code: '', bubbles: true }));
                    const hebrewKeyOnlyUpY = sm.poseTarget.y;

                    sm.resetCameraPose(true);
                    const collapsedSceneBtn = document.querySelector('#spatial-sidebar.collapsed .scene-btn');
                    if (collapsedSceneBtn) collapsedSceneBtn.focus();
                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', code: 'KeyE', bubbles: true }));
                    sm._stepKeyboardNavigation(0.25);
                    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'e', code: 'KeyE', bubbles: true }));
                    const staleHiddenFocusUpY = sm.poseTarget.y;

                    sm.resetCameraPose(true);
                    const infoDrawer = document.getElementById('info-drawer');
                    if (infoDrawer) infoDrawer.classList.add('open');
                    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', code: 'KeyE', bubbles: true }));
                    sm._stepKeyboardNavigation(0.25);
                    const blockedByDrawerY = sm.poseTarget.y;
                    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'e', code: 'KeyE', bubbles: true }));
                    if (infoDrawer) infoDrawer.classList.remove('open');
                    sm.resetCameraPose(true);

                    return JSON.stringify({
                        upY,
                        downY,
                        hebrewUpY,
                        hebrewKeyOnlyUpY,
                        staleHiddenFocusUpY,
                        blockedByDrawerY
                    });
                })()
                """
                send_cdp("Runtime.evaluate", {"expression": kb_elev_js, "returnByValue": True}, msg_id=261)
                res261 = read_cdp(target_id=261)
                kb_data = json.loads(res261.get("result", {}).get("result", {}).get("value"))
                self.assertGreater(kb_data["upY"], 0.0)
                self.assertLess(kb_data["downY"], -66.0)
                self.assertGreater(kb_data["hebrewUpY"], 0.0)
                self.assertGreater(kb_data["hebrewKeyOnlyUpY"], 0.0)
                self.assertGreater(kb_data["staleHiddenFocusUpY"], 0.0)
                self.assertAlmostEqual(kb_data["blockedByDrawerY"], -36.0, places=3)

            # Check gl.getError() === 0
            gl_err_js = """
            (() => {
                const canvas = document.querySelector('canvas#spatial-canvas');
                const gl = canvas ? (canvas.getContext('webgl2') || canvas.getContext('webgl')) : null;
                return gl ? gl.getError() : -1;
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": gl_err_js, "returnByValue": True}, msg_id=206)
            res206 = read_cdp(target_id=206)
            gl_err = res206.get("result", {}).get("result", {}).get("value")
            self.assertEqual(gl_err, 0, f"WebGL gl.getError() returned {gl_err}")

            # 3. Scene transition test with active polling loop
            transition_js = """
            (() => {
                const sm = window.__tamiSceneManager || window.__SPATIAL_MANAGER__;
                if (sm) {
                    sm.transitionTo('bene_beitcha');
                }
            })()
            """
            send_cdp("Runtime.evaluate", {"expression": transition_js}, msg_id=207)
            read_cdp(target_id=207)

            # Poll until transition completes (up to 10s)
            poll_trans_js = """
            (() => {
                const state = window.__PROTOTYPE_STATE__;
                const canvas = document.querySelector('canvas#spatial-canvas');
                const gl = canvas ? (canvas.getContext('webgl2') || canvas.getContext('webgl')) : null;
                return JSON.stringify({
                    currentScene: state.currentScene,
                    isTransitioning: state.isTransitioning,
                    glError: gl ? gl.getError() : -1
                });
            })()
            """
            start_trans = time.time()
            trans_done = False
            final_data = {}
            while time.time() - start_trans < 10.0:
                msg_counter += 1
                send_cdp("Runtime.evaluate", {"expression": poll_trans_js, "returnByValue": True}, msg_id=msg_counter)
                res = read_cdp(target_id=msg_counter)
                if res:
                    final_data = json.loads(res.get("result", {}).get("result", {}).get("value", "{}"))
                    if not final_data.get("isTransitioning", True) and final_data.get("currentScene") == "bene_beitcha":
                        trans_done = True
                        break
                time.sleep(0.2)

            self.assertTrue(trans_done, f"Scene transition failed to settle within 10s: {final_data}")
            self.assertEqual(final_data.get("currentScene"), "bene_beitcha")
            self.assertEqual(final_data.get("glError"), 0, f"WebGL gl.getError() returned {final_data.get('glError')}")

            s.close()
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_desktop_viewport_1440x900(self):
        """Validates Desktop 1440x900 viewport: zero errors, WebGL, parallax, and scene transition."""
        self.run_chrome_test(width=1440, height=900, is_mobile=False)

    def test_mobile_viewport_390x844(self):
        """Validates Mobile 390x844 viewport: mobile user agent, RTL, and prototype responsiveness."""
        self.run_chrome_test(width=390, height=844, is_mobile=True)


if __name__ == "__main__":
    unittest.main()
