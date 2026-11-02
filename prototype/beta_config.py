#!/usr/bin/env python3
"""
prototype/beta_config.py
Canonical, hierarchical, serializable configuration data-models (SSOT)
for the collaborative Family Beta Hub: Suggestion Box, Universal DOM UI + 3D Spatial
Element Highlighting, Biographical Wiki Portal, and Drop-Folder Photo Uploader.
Shared across Python HTTP backend and JavaScript WebGL/Portal frontend via /api/beta-config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeedbackCategorySpec:
    category_id: str
    label_he: str
    icon: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackCategorySpec:
        return cls(
            category_id=str(data["category_id"]),
            label_he=str(data["label_he"]),
            icon=str(data["icon"]),
        )


@dataclass(frozen=True)
class ReviewerPresetSpec:
    preset_id: str
    label_he: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewerPresetSpec:
        return cls(
            preset_id=str(data["preset_id"]),
            label_he=str(data["label_he"]),
        )


@dataclass(frozen=True)
class PortalNavItemSpec:
    nav_id: str
    label_he: str
    icon: str
    href: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortalNavItemSpec:
        return cls(
            nav_id=str(data["nav_id"]),
            label_he=str(data["label_he"]),
            icon=str(data["icon"]),
            href=str(data["href"]),
        )


@dataclass(frozen=True)
class SpatialPickingConfig:
    alpha_hit_threshold: int
    alpha_sample_max_dim: int
    region_grid_samples_per_axis: int
    min_region_drag_px: int
    default_point_box_radius_norm: float
    snapshot_max_width_px: int
    snapshot_jpeg_quality: float
    snapshot_pillow_jpeg_quality: int
    highlight_flash_ms: int
    pin_near_clip_z: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialPickingConfig:
        return cls(
            alpha_hit_threshold=int(data["alpha_hit_threshold"]),
            alpha_sample_max_dim=int(data["alpha_sample_max_dim"]),
            region_grid_samples_per_axis=int(data["region_grid_samples_per_axis"]),
            min_region_drag_px=int(data["min_region_drag_px"]),
            default_point_box_radius_norm=float(data["default_point_box_radius_norm"]),
            snapshot_max_width_px=int(data["snapshot_max_width_px"]),
            snapshot_jpeg_quality=float(data["snapshot_jpeg_quality"]),
            snapshot_pillow_jpeg_quality=int(data["snapshot_pillow_jpeg_quality"]),
            highlight_flash_ms=int(data["highlight_flash_ms"]),
            pin_near_clip_z=float(data["pin_near_clip_z"]),
        )


@dataclass(frozen=True)
class FeedbackLimitsConfig:
    max_request_bytes: int
    max_comment_chars: int
    max_author_chars: int
    max_snapshot_bytes: int
    poll_interval_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackLimitsConfig:
        return cls(
            max_request_bytes=int(data["max_request_bytes"]),
            max_comment_chars=int(data["max_comment_chars"]),
            max_author_chars=int(data["max_author_chars"]),
            max_snapshot_bytes=int(data["max_snapshot_bytes"]),
            poll_interval_ms=int(data["poll_interval_ms"]),
        )


@dataclass(frozen=True)
class UploadLimitsConfig:
    max_upload_request_bytes: int
    max_single_file_bytes: int
    max_files_per_batch: int
    min_image_dim_px: int
    sips_convert_timeout_sec: int
    place_presets: tuple[tuple[str, str], ...]
    people_presets: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_upload_request_bytes": self.max_upload_request_bytes,
            "max_single_file_bytes": self.max_single_file_bytes,
            "max_files_per_batch": self.max_files_per_batch,
            "min_image_dim_px": self.min_image_dim_px,
            "sips_convert_timeout_sec": self.sips_convert_timeout_sec,
            "place_presets": [
                {"canonical": canonical, "label_he": label_he}
                for canonical, label_he in self.place_presets
            ],
            "people_presets": [
                {"canonical": canonical, "label_he": label_he}
                for canonical, label_he in self.people_presets
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadLimitsConfig:
        return cls(
            max_upload_request_bytes=int(data["max_upload_request_bytes"]),
            max_single_file_bytes=int(data["max_single_file_bytes"]),
            max_files_per_batch=int(data["max_files_per_batch"]),
            min_image_dim_px=int(data["min_image_dim_px"]),
            sips_convert_timeout_sec=int(data["sips_convert_timeout_sec"]),
            place_presets=tuple(
                (str(item["canonical"]), str(item["label_he"]))
                for item in data["place_presets"]
            ),
            people_presets=tuple(
                (str(item["canonical"]), str(item["label_he"]))
                for item in data["people_presets"]
            ),
        )


@dataclass(frozen=True)
class CreationModeSpec:
    mode_id: str
    label_he: str
    icon: str
    subtitle_he: str
    placeholder_he: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreationModeSpec:
        return cls(
            mode_id=str(data["mode_id"]),
            label_he=str(data["label_he"]),
            icon=str(data["icon"]),
            subtitle_he=str(data["subtitle_he"]),
            placeholder_he=str(data["placeholder_he"]),
        )


@dataclass(frozen=True)
class CreativeStudioConfig:
    max_selected_photos: int
    collage_canvas_width_px: int
    collage_padding_px: int
    collage_gap_px: int
    collage_header_height_px: int
    collage_jpeg_quality: float
    modes: tuple[CreationModeSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_selected_photos": self.max_selected_photos,
            "collage_canvas_width_px": self.collage_canvas_width_px,
            "collage_padding_px": self.collage_padding_px,
            "collage_gap_px": self.collage_gap_px,
            "collage_header_height_px": self.collage_header_height_px,
            "collage_jpeg_quality": self.collage_jpeg_quality,
            "modes": [m.to_dict() for m in self.modes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreativeStudioConfig:
        return cls(
            max_selected_photos=int(data["max_selected_photos"]),
            collage_canvas_width_px=int(data["collage_canvas_width_px"]),
            collage_padding_px=int(data["collage_padding_px"]),
            collage_gap_px=int(data["collage_gap_px"]),
            collage_header_height_px=int(data["collage_header_height_px"]),
            collage_jpeg_quality=float(data["collage_jpeg_quality"]),
            modes=tuple(CreationModeSpec.from_dict(m) for m in data["modes"]),
        )


@dataclass(frozen=True)
class BetaFeedbackConfig:
    enabled: bool
    beta_badge_label_he: str
    secret_notice_he: str
    storage_dir_relpath: str
    entries_filename: str
    summary_filename: str
    snapshots_subdir: str
    valid_scene_ids: tuple[str, ...]
    portal_nav_items: tuple[PortalNavItemSpec, ...]
    categories: tuple[FeedbackCategorySpec, ...]
    reviewer_presets: tuple[ReviewerPresetSpec, ...]
    picking: SpatialPickingConfig
    limits: FeedbackLimitsConfig
    uploads: UploadLimitsConfig
    creative_studio: CreativeStudioConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "beta_badge_label_he": self.beta_badge_label_he,
            "secret_notice_he": self.secret_notice_he,
            "storage_dir_relpath": self.storage_dir_relpath,
            "entries_filename": self.entries_filename,
            "summary_filename": self.summary_filename,
            "snapshots_subdir": self.snapshots_subdir,
            "valid_scene_ids": list(self.valid_scene_ids),
            "portal_nav_items": [n.to_dict() for n in self.portal_nav_items],
            "categories": [c.to_dict() for c in self.categories],
            "reviewer_presets": [r.to_dict() for r in self.reviewer_presets],
            "picking": self.picking.to_dict(),
            "limits": self.limits.to_dict(),
            "uploads": self.uploads.to_dict(),
            "creative_studio": self.creative_studio.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BetaFeedbackConfig:
        return cls(
            enabled=bool(data["enabled"]),
            beta_badge_label_he=str(data["beta_badge_label_he"]),
            secret_notice_he=str(data["secret_notice_he"]),
            storage_dir_relpath=str(data["storage_dir_relpath"]),
            entries_filename=str(data["entries_filename"]),
            summary_filename=str(data["summary_filename"]),
            snapshots_subdir=str(data["snapshots_subdir"]),
            valid_scene_ids=tuple(str(s) for s in data["valid_scene_ids"]),
            portal_nav_items=tuple(
                PortalNavItemSpec.from_dict(n) for n in data["portal_nav_items"]
            ),
            categories=tuple(FeedbackCategorySpec.from_dict(c) for c in data["categories"]),
            reviewer_presets=tuple(
                ReviewerPresetSpec.from_dict(r) for r in data["reviewer_presets"]
            ),
            picking=SpatialPickingConfig.from_dict(data["picking"]),
            limits=FeedbackLimitsConfig.from_dict(data["limits"]),
            uploads=UploadLimitsConfig.from_dict(data["uploads"]),
            creative_studio=CreativeStudioConfig.from_dict(data["creative_studio"]),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw_json: str) -> BetaFeedbackConfig:
        return cls.from_dict(json.loads(raw_json))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> BetaFeedbackConfig:
        return cls.from_json(path.read_text(encoding="utf-8"))


def default_beta_feedback_config(*, enabled: bool = True) -> BetaFeedbackConfig:
    """Returns the canonical SSOT Family Beta Hub & Highlighting configuration."""
    return BetaFeedbackConfig(
        enabled=enabled,
        beta_badge_label_he="בטא משפחתי",
        secret_notice_he="🤫 הפתעה לתמי ליום הולדת 70 — סודי! נא לא לשתף איתה",
        storage_dir_relpath="feedback",
        entries_filename="feedback_entries.jsonl",
        summary_filename="summary.json",
        snapshots_subdir="snapshots",
        valid_scene_ids=(
            "shelf_portal",
            "paper_capsules",
            "bene_beitcha",
            "bene_beitcha_scroll_foot",
            "purim_costume_1966",
            "vitrine_keys_pen",
            "denya_model",
            "denya_living_room",
            "denya_couch_guitar",
            "plan_b_pipe",
            "plan_b_pipe_v1",
            "plan_b_pipe_v3",
            "humpty_dumpty_booth",
            "wandering_drawing_1",
            "wandering_drawing",
            "paris_cite_white_table",
            "toddler_tami_1958",
            "fried_yard_installation",
            "fried_house_railing",
            "fried_table_hybrid",
            "herzliya_lawn_carpet",
            "bar_giora_bedroom",
            "marseille_mirror",
            "marseille_mirror_v1",
            "scroll_palm_world",
            "humpty_dumpty_bundle",
            "tzofen_brick_table",
            "herzliya_tree_canopy",
            "wiki",
            "catalog",
        ),
        portal_nav_items=(
            PortalNavItemSpec(
                nav_id="prototype",
                label_he="סיור מרחבי (3D)",
                icon="🏛️",
                href="/",
            ),
            PortalNavItemSpec(
                nav_id="wiki",
                label_he="ויקי ביוגרפי (31 ערכים)",
                icon="📖",
                href="/wiki",
            ),
            PortalNavItemSpec(
                nav_id="catalog",
                label_he="קטלוג והעלאת תמונות",
                icon="🖼️",
                href="/catalog",
            ),
        ),
        categories=(
            FeedbackCategorySpec(
                category_id="suggestion",
                label_he="רעיון / הצעה",
                icon="✨",
            ),
            FeedbackCategorySpec(
                category_id="ui_ux",
                label_he="ממשק וכפתורים (UI)",
                icon="🧩",
            ),
            FeedbackCategorySpec(
                category_id="visual_element",
                label_he="אלמנט מרחבי / תמונה",
                icon="🎨",
            ),
            FeedbackCategorySpec(
                category_id="navigation",
                label_he="ניווט ותנועה",
                icon="🧭",
            ),
            FeedbackCategorySpec(
                category_id="story_bio",
                label_he="תיקון / העשרה ביוגרפית",
                icon="📖",
            ),
        ),
        reviewer_presets=(
            ReviewerPresetSpec(preset_id="tal", label_he="טל"),
            ReviewerPresetSpec(preset_id="naama", label_he="נעמה"),
            ReviewerPresetSpec(preset_id="gadi", label_he="גדי (אבא)"),
            ReviewerPresetSpec(preset_id="guy", label_he="גיא"),
        ),
        picking=SpatialPickingConfig(
            alpha_hit_threshold=28,
            alpha_sample_max_dim=512,
            region_grid_samples_per_axis=5,
            min_region_drag_px=14,
            default_point_box_radius_norm=0.042,
            snapshot_max_width_px=1080,
            snapshot_jpeg_quality=0.84,
            snapshot_pillow_jpeg_quality=86,
            highlight_flash_ms=3000,
            pin_near_clip_z=5.0,
        ),
        limits=FeedbackLimitsConfig(
            max_request_bytes=2_400_000,
            max_comment_chars=2500,
            max_author_chars=64,
            max_snapshot_bytes=1_800_000,
            poll_interval_ms=12_000,
        ),
        uploads=UploadLimitsConfig(
            max_upload_request_bytes=95_000_000,
            max_single_file_bytes=35_000_000,
            max_files_per_batch=15,
            min_image_dim_px=150,
            sips_convert_timeout_sec=8,
            place_presets=(
                ("Herzliya (parents' house & yard)", "🏡 הרצליה — בית ההורים והחצר"),
                ("Haifa: Arlozorov", "🏢 חיפה — ארלוזורוב 87 (הדר)"),
                ("Haifa: Einstein", "🌳 חיפה — איינשטיין 33א (אחוזה)"),
                ("Haifa: Denya (80 Greenboim St & studio)", "🏛️ חיפה — גרינבוים 80, דניה (הבית והסטודיו)"),
                ("Paris (Cité des Arts)", "🗼 פריז — Cité des Arts"),
                ("Technion", "🎓 הטכניון / אלביט / רפאל"),
                ("Southern France", "🌻 דרום צרפת / טיולים בעולם"),
                ("Montreal", "🍁 מונטריאול / ניו יורק"),
                ("Ramat Gan Museum", "🖼️ מוזיאון רמת גן (בנה ביתך)"),
                ("Binyamin Gallery", "🎨 גלריה בנימין"),
            ),
            people_presets=(
                ("Tamar 'Tami' Sheaffer", "תמי (תמר)"),
                ("Gad Sheaffer", "גדי"),
                ("Michael Fried", "מיכאל פריד (סבא)"),
                ("Ahuva Fried", "אהובה פריד (סבתא)"),
                ("Er'ela Sadan", "אראלה"),
                ("Ada Wertheimer", "עדה ורטהיימר"),
                ("Guy Sheaffer", "גיא"),
                ("Tal Sheaffer", "טל"),
                ("Ya'ara", "יערה"),
                ("Carmel", "כרמל"),
                ("Beni", "בני"),
                ("Peri", "פרי"),
            ),
        ),
        creative_studio=CreativeStudioConfig(
            max_selected_photos=12,
            collage_canvas_width_px=1600,
            collage_padding_px=36,
            collage_gap_px=20,
            collage_header_height_px=108,
            collage_jpeg_quality=0.92,
            modes=(
                CreationModeSpec(
                    mode_id="collage",
                    label_he="קולאז׳ משפחתי",
                    icon="🖼️",
                    subtitle_he="יצירת קולאז׳ מעוצב להורדה מיידית או לשמירה בקטלוג",
                    placeholder_he="כותרת אופציונלית על הקולאז׳ (למשל: רגעים בדניה עם סבתא תמי)...",
                ),
                CreationModeSpec(
                    mode_id="fuse_scene",
                    label_he="מיזוג לסצנה מרחבית אחת",
                    icon="🏛️",
                    subtitle_he="חיבור מספר זוויות או תמונות של מקום/חדר לסצנת תלת־ממד בסיור",
                    placeholder_he="תארו כיצד לחבר את התמונות לסצנה אחת (למשל: אותו סלון בדניה משתי זוויות, חצר ובית, או שילוב פסל וזיכרון)...",
                ),
                CreationModeSpec(
                    mode_id="open_prompt",
                    label_he="תיאור מילולי פתוח",
                    icon="✍️",
                    subtitle_he="רעיון חופשי, סיפור משותף, ציר זמן או בקשה יצירתית מהתמונות שנבחרו",
                    placeholder_he="כתבו בחופשיות מה תרצו שניצור מהתמונות שנבחרו, או מה הסיפור המחבר ביניהן...",
                ),
            ),
        ),
    )
