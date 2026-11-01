#!/usr/bin/env python3
"""
scripts/ingest_whatsapp_export.py — Read-Only Family WhatsApp Export Parser & Curated Photo Ingestor

Reads `drop_folder/incoming/WhatsApp Chat - שפר.zip` strictly read-only, parses `_chat.txt`
for exact message/media coverage statistics, maps chat senders and surrounding ±2 message
context, and ingests curated family, Denya home/kitchen/dining/staircase, and exhibition
photos into `media/whatsapp_family/` with rich `.json` sidecars.

Privacy & Quality Guardrails:
- Strictly skips all screenshots, memes, stickers, receipts, links/news captures, and private
  70th-birthday coordination screenshots.
- Never mutates `drop_folder/incoming/WhatsApp Chat - שפר.zip`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP_PATH = PROJECT_ROOT / "drop_folder" / "incoming" / "WhatsApp Chat - שפר.zip"
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "media"
WHATSAPP_SUBDIR = "whatsapp_family"

MIN_IMAGE_DIM_PX = 150
SHA256_READ_CHUNK_BYTES = 65536
FILENAME_HASH_PREFIX_LEN = 8
SURROUNDING_CONTEXT_WINDOW = 2

PLACE_DENYA = "Haifa: Denya (80 Greenboim St & studio)"
PLACE_RAMAT_GAN = "Ramat Gan Museum"
PLACE_SOUTHERN_FRANCE = "Southern France"

PERSON_TAMI = "Tamar 'Tami' Sheaffer"
PERSON_GADI = "Gad Sheaffer"
PERSON_GUY = "Guy Sheaffer"
PERSON_TAL = "Tal Sheaffer"
PERSON_YAARA = "Ya'ara"
PERSON_CARMEL = "Carmel"

CHAT_LINE_RE = re.compile(
    r"^[\u200e\u200f]*\[(\d{2}/\d{2}/\d{4}),?\s+(\d{1,2}:\d{2}:\d{2})\]\s+([^:]+):\s*(.*)$"
)
ATTACHED_RE = re.compile(r"<attached:\s*([^>]+)>")
OMITTED_RE = re.compile(
    r"[\u200e\u200f]*(image|video|audio|gif|sticker|document)\s+omitted", re.IGNORECASE
)
SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_\-]+")

SENDER_CANONICAL_PEOPLE: dict[str, tuple[str, ...]] = {
    "אמא": (PERSON_TAMI,),
    "אבא": (PERSON_GADI,),
    "גיא": (PERSON_GUY,),
    "Tal Sheaffer": (PERSON_TAL,),
    "Wifey": (),
    "רונה פלדמן שפר": (),
}

SENDER_DISPLAY_HE: dict[str, str] = {
    "אמא": "תמי שפר (אמא)",
    "אבא": "גדי שפר (אבא)",
    "גיא": "גיא שפר",
    "Tal Sheaffer": "טל שפר",
    "Wifey": "נעמה",
    "רונה פלדמן שפר": "רונה פלדמן שפר",
}


@dataclass(frozen=True)
class WhatsAppIngestConfig:
    zip_path: Path = DEFAULT_ZIP_PATH
    media_dir: Path = DEFAULT_MEDIA_DIR
    min_dim_px: int = MIN_IMAGE_DIM_PX
    context_window: int = SURROUNDING_CONTEXT_WINDOW


@dataclass(frozen=True)
class CuratedWhatsAppPhoto:
    zip_filename: str
    title: str
    curated_description: str
    people: tuple[str, ...]
    places: tuple[str, ...]
    art_themes: tuple[str, ...] = ()


@dataclass
class ParsedChatMessage:
    date_ddmmyyyy: str
    time_hhmmss: str
    sender_raw: str
    body: str
    attached_filename: str | None = None

    @property
    def iso_date(self) -> str:
        dd, mm, yyyy = self.date_ddmmyyyy.split("/")
        return f"{yyyy}-{mm}-{dd}"


# Visually verified curated family, Denya home, exhibition, and travel photos from `WhatsApp Chat - שפר.zip`
CURATED_WHATSAPP_PHOTOS: tuple[CuratedWhatsAppPhoto, ...] = (
    # --- Denya Home (80 Greenboim St) Interior, Kitchen, Dining, Staircase & Porch ---
    CuratedWhatsAppPhoto(
        zip_filename="-0000001-PHOTO-2026-04-17-15-47-06.jpg",
        title="רות מכינה בצק במטבח",
        curated_description="רות מכינה כדורי בצק במטבח (צילום ששלח גיא שפר בקבוצת הוואטסאפ המשפחתית 'שפר').",
        people=(PERSON_GUY, "רות"),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000001-PHOTO-2026-04-18-09-14-14.jpg",
        title="דיוקן עצמי מחויך של תמי בבית בדניה ('עם צבע בשיער וחולצת עבודה')",
        curated_description="דיוקן עצמי הומוריסטי וחם ששלחה תמי לקבוצה המשפחתית מהבית בדניה.",
        people=(PERSON_TAMI,),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000053-PHOTO-2026-04-26-10-28-21.jpg",
        title="ספסל העץ והכלב במרפסת הבית בדניה",
        curated_description="פינת הישיבה החיצונית וספסל העץ במרפסת הבית בגרינבוים 80 בדניה.",
        people=(PERSON_GADI,),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000080-PHOTO-2026-04-28-11-23-12.jpg",
        title="גדי שקוע בקריאה בסלון הבית בדניה ('האיש שלי כל הזמן קורא')",
        curated_description="גדי שפר קורא בסלון הבית בגרינבוים 80 בדניה — תמונה מלאת חיבה שצילמה תמי.",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000555-PHOTO-2026-06-20-15-39-47.jpg",
        title="ארוחה משפחתית בסלון ובמטבח בגרינבוים 80 דניה — מבט רחב אל קיר החלונות והוואדי",
        curated_description="מבט פנים רחב של הבית בגרינבוים 80 בדניה: תמי, טל וגיא סביב שולחן האוכל, גדי במטבח העץ מאחור, גרם המדרגות וקיר הזכוכית המשקיף אל עצי הכרמל.",
        people=(PERSON_TAMI, PERSON_GADI, PERSON_GUY, PERSON_TAL),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000556-PHOTO-2026-06-20-15-39-49.jpg",
        title="פנים הבית בגרינבוים 80 דניה — פינת האוכל, מטבח העץ וגרם המדרגות",
        curated_description="מבט אנכי רחב על חלל המגורים הפתוח בגרינבוים 80 בדניה בעת מפגש משפחתי: שולחן האוכל, מטבח העץ וגרם המדרגות.",
        people=(PERSON_TAMI, PERSON_GADI, PERSON_GUY, PERSON_TAL),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000557-PHOTO-2026-06-20-15-39-51.jpg",
        title="מפגש משפחתי סביב שולחן האוכל בגרינבוים 80 דניה אל מול ירוק הכרמל",
        curated_description="תמי, גדי, גיא וטל סביב שולחן האוכל בחלל המרכזי של הבית בגרינבוים 80 בדניה מול החורש הירוק מבעד לחלונות הרצפה-תקרה.",
        people=(PERSON_TAMI, PERSON_GADI, PERSON_GUY, PERSON_TAL),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000584-PHOTO-2026-06-22-23-57-37.jpg",
        title="מאפי המשמש של תמי על דלפק העץ במטבח בדניה",
        curated_description="מגשי מאפים ביתיים עם משמשים שהכינה תמי על משטח העץ במטבח הבית בדניה.",
        people=(PERSON_TAMI,),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000746-PHOTO-2026-07-29-16-59-17.jpg",
        title="הנכדות יערה וכרמל מרכיבות פאזל למרגלות מדרגות העץ בבית בדניה",
        curated_description="יערה וכרמל יושבות על רצפת הסומסום למרגלות גרם המדרגות מעץ בבית סבתא תמי וסבא גדי בדניה ומרכיבות פאזל של 240 חלקים.",
        people=(PERSON_YAARA, PERSON_CARMEL, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000747-PHOTO-2026-07-29-16-59-25.jpg",
        title="יערה וכרמל משלימות פאזל פרפר (240 חלקים) בבית בדניה",
        curated_description="יערה וכרמל מציגות בגאווה את פאזל הפרפר בן 240 החלקים על הרצפה לצד מדרגות העץ בבית בגרינבוים 80 בדניה.",
        people=(PERSON_YAARA, PERSON_CARMEL, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),

    # --- Ramat Gan Museum Solo Exhibition ('בנה ביתך', May 2026) ---
    CuratedWhatsAppPhoto(
        zip_filename="00000097-PHOTO-2026-05-01-21-05-23.jpg",
        title="כרזת רחוב לתערוכת 'בנה ביתך' של תמר שפר ברמת גן ('תמי את סלב בחוצות רמת גן')",
        curated_description="שלט חוצות מואר ברמת גן לקראת פתיחת תערוכת היחיד 'בנה ביתך' של תמר שפר במוזיאון רמת גן לאמנות ישראלית.",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_RAMAT_GAN,),
        art_themes=("simulacra of home",),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000137-PHOTO-2026-05-06-19-42-01.jpg",
        title="תמי וגדי שפר מחובקים ומחייכים בערב פתיחת התערוכה 'בנה ביתך' במוזיאון רמת גן",
        curated_description="דיוקן זוגי קורן של תמי וגדי בערב הפתיחה החגיגי של תערוכת היחיד 'בנה ביתך' במוזיאון רמת גן (6 במאי 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_RAMAT_GAN,),
        art_themes=("simulacra of home",),
    ),

    # --- Family Portraits & Shared Travels (Venice Biennale, Balkans, Southern France) ---
    CuratedWhatsAppPhoto(
        zip_filename="00000207-PHOTO-2026-05-21-22-05-01.jpg",
        title="רהיטים חצויים בביאנלה בוונציה ('שאבו השראה מאמא')",
        curated_description="מיצב רהיטים חצויים שצילם גדי בוונציה, שעליו הגיב גיא בחיוך: 'חצוי – שאבו השראה מאמא'.",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(),
        art_themes=("misshapen furniture",),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000290-PHOTO-2026-05-25-13-20-24.jpg",
        title="תמי וגדי משתקפים במראה עתיקה בביאנלה בוונציה",
        curated_description="צילום עצמי אמנותי של תמי וגדי משתקפים במראה מעוטרת במהלך ביקורם בביאנלה לאמנות בוונציה (מאי 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(),
        art_themes=("broken/blurry reflections",),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000299-PHOTO-2026-05-25-13-44-44.jpg",
        title="דיוקן של תמי וגדי בוונציה (מאי 2026)",
        curated_description="תמי וגדי יחד בוונציה במהלך חופשת האמנות בביאנלה (מאי 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000468-PHOTO-2026-06-11-22-49-52.jpg",
        title="מסע בהרי הבלקן — נופי הרים וירוק (יוני 2026)",
        curated_description="צילום ששלחה תמי לקבוצה המשפחתית מטיולם של תמי וגדי בבלקן ('וואו נופים מהממים').",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000480-PHOTO-2026-06-12-15-53-56.jpg",
        title="הפסקת צהריים בטיול בבלקן (יוני 2026)",
        curated_description="תמי וגדי בהפסקת צהריים מול הנוף במהלך מסעם בבלקן (יוני 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000545-PHOTO-2026-06-18-17-08-24.jpg",
        title="הנכדות יערה וכרמל — צילום משפחתי (יוני 2026)",
        curated_description="יערה וכרמל בצילום משפחתי בהפרדה גבוהה ששיתפה רונה בקבוצת 'שפר'.",
        people=(PERSON_YAARA, PERSON_CARMEL),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000717-PHOTO-2026-07-26-09-43-58.jpg",
        title="דיוקן של גדי במסע לאסטוניה ('אוהבת איך שאתה נראה כאן!')",
        curated_description="גדי שפר בבוקר הנסיעה לטאלין, ותגובתה של תמי בקבוצה: 'אוהבת איך שאתה נראה כאן!' (יולי 2026).",
        people=(PERSON_GADI,),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000932-PHOTO-2026-08-28-12-59-37.jpg",
        title="יערה וכרמל — סוף חופשת הקיץ (אוגוסט 2026)",
        curated_description="דיוקן משותף חד ומלא שמחה של הנכדות יערה וכרמל מתוך קבוצת המשפחה.",
        people=(PERSON_YAARA, PERSON_CARMEL),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00000983-PHOTO-2026-09-07-12-09-39.jpg",
        title="תמי וגדי על שפת המצוק באתיופיה — זיכרון מסע משותף",
        curated_description="תמי וגדי יחד מול נופי המצוקים באתיופיה, תמונה ארכיונית ששיתף גדי בקבוצת המשפחה.",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001005-PHOTO-2026-09-10-16-25-57.jpg",
        title="'שנה טובה עם המתוקות' — ברכת ראש השנה עם יערה וכרמל",
        curated_description="גדי עם הנכדות יערה וכרמל בברכת 'שנה טובה עם המתוקות' לקראת ראש השנה תשפ\"ז.",
        people=(PERSON_GADI, PERSON_YAARA, PERSON_CARMEL),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001073-PHOTO-2026-09-16-20-46-16.jpg",
        title="סלפי מחויך של תמי ובנה הבכור גיא ('אתם מהממים!')",
        curated_description="דיוקן קרוב ומלא חום של תמי וגיא שפר מחייכים יחד (ספטמבר 2026).",
        people=(PERSON_TAMI, PERSON_GUY),
        places=(),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001107-PHOTO-2026-09-20-13-07-13.jpg",
        title="כלי קרמיקה של פיקאסו בדרום צרפת ('פיקאסו המוכשר המעצבן הזה')",
        curated_description="כלי קרמיקה של פיקאסו שצילמה תמי במהלך הטיול בדרום צרפת עם כיתובה המשעשע בקבוצה (ספטמבר 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_SOUTHERN_FRANCE,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001118-PHOTO-2026-09-20-18-51-06.jpg",
        title="תמי וגדי בטיול בדרום צרפת (ספטמבר 2026)",
        curated_description="צילום ממסעם של תמי וגדי בדרום צרפת ששותף בקבוצת המשפחה (ספטמבר 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_SOUTHERN_FRANCE,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001121-PHOTO-2026-09-20-19-19-34.jpg",
        title="ערב בדרום צרפת — מסעם של תמי וגדי (ספטמבר 2026)",
        curated_description="צילום ששיתפה תמי ממסעם בדרום צרפת ('תמונות מהממות! של כולם').",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_SOUTHERN_FRANCE,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001125-PHOTO-2026-09-22-16-47-37.jpg",
        title="האמפיתיאטרון הרומי בנים (Nîmes) — דרום צרפת",
        curated_description="הארנה הרומית בנים ('אנחנו ב-Nîmes') מתוך הטיול של תמי וגדי בדרום צרפת (ספטמבר 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_SOUTHERN_FRANCE,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001131-PHOTO-2026-09-24-14-49-14.jpg",
        title="להקת פלמינגו בשמורת הקמארג בדרום צרפת",
        curated_description="צילום של תמי משמורת הקמארג בפרובנס במהלך הטיול בדרום צרפת (ספטמבר 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_SOUTHERN_FRANCE,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001139-PHOTO-2026-09-26-15-28-39.jpg",
        title="ארוחת צהריים של תמי וגדי במסעדה בתכנון טדאו אנדו בפרובנס",
        curated_description="תמי וגדי בארוחת צהריים על שפת המים במסעדה שתכנן האדריכל טדאו אנדו בשאטו לה קוסט, דרום צרפת (ספטמבר 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_SOUTHERN_FRANCE,),
    ),
    CuratedWhatsAppPhoto(
        zip_filename="00001229-PHOTO-2026-10-17-21-36-02.jpg",
        title="יערה וכרמל משתובבות הפוכות על הספה",
        curated_description="הנכדות יערה וכרמל צוחקות הפוכות על הספה בצילום משפחתי ששיתף סבא גדי (אוקטובר 2026).",
        people=(PERSON_YAARA, PERSON_CARMEL, PERSON_GADI),
        places=(),
    ),
)


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(SHA256_READ_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest().lower()


def collect_existing_hashes(media_dir: Path, exclude_subdir: Path | None = None) -> dict[str, Path]:
    hashes: dict[str, Path] = {}
    resolved_excl = exclude_subdir.resolve() if exclude_subdir and exclude_subdir.exists() else None
    for root, _, files in os.walk(media_dir):
        root_path = Path(root).resolve()
        if resolved_excl and (root_path == resolved_excl or resolved_excl in root_path.parents):
            continue
        for fname in files:
            fpath = root_path / fname
            if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}:
                hashes[compute_sha256_file(fpath)] = fpath
    return hashes


def parse_whatsapp_chat(chat_text: str) -> tuple[list[ParsedChatMessage], dict[str, Any]]:
    messages: list[ParsedChatMessage] = []
    omitted_counts: dict[str, int] = {}
    sender_counts: dict[str, int] = {}
    raw_lines = chat_text.splitlines()

    for raw_line in raw_lines:
        m = CHAT_LINE_RE.match(raw_line)
        if m:
            date_s, time_s, sender_s, body_s = m.group(1), m.group(2), m.group(3).strip(), m.group(4)
            sender_clean = sender_s.strip("\u200e\u200f ")
            att_m = ATTACHED_RE.search(body_s)
            att_name = att_m.group(1).strip() if att_m else None
            msg = ParsedChatMessage(
                date_ddmmyyyy=date_s,
                time_hhmmss=time_s,
                sender_raw=sender_clean,
                body=body_s,
                attached_filename=att_name,
            )
            messages.append(msg)
            sender_counts[sender_clean] = sender_counts.get(sender_clean, 0) + 1
            for om in OMITTED_RE.finditer(body_s):
                kind = om.group(1).lower()
                omitted_counts[kind] = omitted_counts.get(kind, 0) + 1
        elif messages:
            messages[-1].body += "\n" + raw_line
            att_m = ATTACHED_RE.search(raw_line)
            if att_m and not messages[-1].attached_filename:
                messages[-1].attached_filename = att_m.group(1).strip()
            for om in OMITTED_RE.finditer(raw_line):
                kind = om.group(1).lower()
                omitted_counts[kind] = omitted_counts.get(kind, 0) + 1

    stats: dict[str, Any] = {
        "total_lines": len(raw_lines),
        "total_messages": len(messages),
        "first_timestamp": f"{messages[0].date_ddmmyyyy} {messages[0].time_hhmmss}" if messages else "",
        "last_timestamp": f"{messages[-1].date_ddmmyyyy} {messages[-1].time_hhmmss}" if messages else "",
        "sender_counts": sender_counts,
        "omitted_counts": omitted_counts,
    }
    return messages, stats


def clean_chat_body(raw_body: str) -> str:
    cleaned = ATTACHED_RE.sub("", raw_body)
    cleaned = OMITTED_RE.sub("", cleaned)
    cleaned = cleaned.replace("\u200e", "").replace("\u200f", "").strip()
    return " ".join(cleaned.split())


def format_surrounding_context(
    messages: list[ParsedChatMessage], msg_idx: int, window: int
) -> tuple[str, list[str]]:
    target = messages[msg_idx]
    inline_caption = clean_chat_body(target.body)
    snippets: list[str] = []
    start = max(0, msg_idx - window)
    end = min(len(messages), msg_idx + window + 1)
    for idx in range(start, end):
        if idx == msg_idx:
            continue
        m = messages[idx]
        txt = clean_chat_body(m.body)
        if not txt or "This message was deleted" in txt or "http" in txt:
            continue
        sender_he = SENDER_DISPLAY_HE.get(m.sender_raw, m.sender_raw)
        snippets.append(f"{sender_he}: '{txt[:110]}'")
    return inline_caption, snippets


def ingest_curated_whatsapp_export(cfg: WhatsAppIngestConfig) -> dict[str, Any]:
    if not cfg.zip_path.is_file():
        raise FileNotFoundError(f"WhatsApp export ZIP not found at {cfg.zip_path}")

    out_dir = cfg.media_dir / WHATSAPP_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = collect_existing_hashes(cfg.media_dir, exclude_subdir=out_dir)

    ingested: list[dict[str, Any]] = []
    skipped_duplicates: list[str] = []

    with zipfile.ZipFile(cfg.zip_path, mode="r") as zf:
        namelist = set(zf.namelist())
        if "_chat.txt" not in namelist:
            raise ValueError(f"_chat.txt missing inside {cfg.zip_path}")

        chat_text = zf.read("_chat.txt").decode("utf-8")
        messages, chat_stats = parse_whatsapp_chat(chat_text)

        msg_idx_by_attachment: dict[str, int] = {}
        for idx, msg in enumerate(messages):
            if msg.attached_filename:
                msg_idx_by_attachment[msg.attached_filename] = idx

        zip_image_count = sum(
            1 for n in namelist if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        chat_stats["attached_images_in_zip"] = zip_image_count

        for spec in CURATED_WHATSAPP_PHOTOS:
            if spec.zip_filename not in namelist:
                raise KeyError(f"Curated WhatsApp photo '{spec.zip_filename}' not found in {cfg.zip_path.name}")

            raw_bytes = zf.read(spec.zip_filename)
            with Image.open(io.BytesIO(raw_bytes)) as vimg:
                vimg.verify()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.load()
                width, height = img.size

            if width < cfg.min_dim_px or height < cfg.min_dim_px:
                raise ValueError(
                    f"Curated WhatsApp photo '{spec.zip_filename}' ({width}x{height}) is under {cfg.min_dim_px}px"
                )

            sha256_hex = compute_sha256_bytes(raw_bytes)
            if sha256_hex in seen_hashes:
                skipped_duplicates.append(spec.zip_filename)
                continue

            msg_idx = msg_idx_by_attachment.get(spec.zip_filename)
            if msg_idx is not None:
                msg = messages[msg_idx]
                iso_date = msg.iso_date
                sender_label = SENDER_DISPLAY_HE.get(msg.sender_raw, msg.sender_raw)
                inline_cap, ctx_snippets = format_surrounding_context(messages, msg_idx, cfg.context_window)
            else:
                iso_date = "2026"
                sender_label = "קבוצת משפחת שפר"
                inline_cap = ""
                ctx_snippets = []

            caption_parts = [spec.curated_description, f"נשלח בקבוצת הוואטסאפ המשפחתית 'שפר' ע\"י {sender_label} ({iso_date})."]
            if inline_cap:
                caption_parts.append(f"כיתוב בהודעה: \"{inline_cap}\".")
            if ctx_snippets:
                caption_parts.append("שיחה סמוכה: " + " · ".join(ctx_snippets[:2]))

            ext = Path(spec.zip_filename).suffix.lower() or ".jpg"
            stem_clean = SAFE_STEM_RE.sub("_", Path(spec.zip_filename).stem).strip("_-").lower()
            out_filename = f"wa_sheaffer_{stem_clean}_{sha256_hex[:FILENAME_HASH_PREFIX_LEN]}{ext}"
            img_path = out_dir / out_filename
            sidecar_path = out_dir / f"{out_filename}.json"

            sidecar_payload: dict[str, Any] = {
                "title": spec.title,
                "caption": " ".join(caption_parts),
                "source_url_or_path": f"drop_folder/incoming/WhatsApp Chat - שפר.zip#{spec.zip_filename}",
                "date_or_era": iso_date,
                "people": list(spec.people),
                "places": list(spec.places),
                "places_locked": True,
                "art_themes": list(spec.art_themes),
                "confidence_notes": (
                    f"CONFIRMED: Visually verified {width}x{height} family photo from WhatsApp 'שפר' export ({iso_date})."
                ),
            }

            img_path.write_bytes(raw_bytes)
            sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            seen_hashes[sha256_hex] = img_path
            ingested.append(
                {
                    "filename": out_filename,
                    "orig_zip_name": spec.zip_filename,
                    "width": width,
                    "height": height,
                    "date": iso_date,
                    "places": list(spec.places),
                    "title": spec.title,
                }
            )

    return {
        "chat_stats": chat_stats,
        "ingested": ingested,
        "skipped_duplicates": skipped_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest curated family photos from WhatsApp 'שפר' export ZIP.")
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP_PATH), help="Path to WhatsApp export ZIP")
    parser.add_argument("--media-dir", default=str(DEFAULT_MEDIA_DIR), help="Path to media archive root")
    args = parser.parse_args()

    cfg = WhatsAppIngestConfig(
        zip_path=Path(args.zip_path).resolve(),
        media_dir=Path(args.media_dir).resolve(),
    )
    report = ingest_curated_whatsapp_export(cfg)
    stats = report["chat_stats"]
    print(
        f"[WhatsApp 'שפר' Export] Span: {stats['first_timestamp']} -> {stats['last_timestamp']} | "
        f"Messages: {stats['total_messages']} | Attached images in ZIP: {stats['attached_images_in_zip']} | "
        f"Omitted media: {stats['omitted_counts']}"
    )
    print(f"[WhatsApp Ingest] Saved {len(report['ingested'])} curated family & home photos to media/{WHATSAPP_SUBDIR}/.")


if __name__ == "__main__":
    main()
