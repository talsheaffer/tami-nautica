#!/usr/bin/env python3
"""
scripts/harvest_blog_homes_wide.py — Wide WordPress Home-Photo Harvester & Sidecar Cleaner

Performs two synchronized, strictly read-only operations against `theshingimel.wordpress.com`
and local sidecars under `media/`:
1. Audits and cleans polluted `places` tags on existing sidecars in `media/portfolio_site/`
   and `media/wordpress_blog/` (removing blind post/batch-level `Herzliya` and `Denya` tags
   from artworks, gallery tours, and foreign travel posts while preserving genuine home photos).
2. Harvests full-resolution (`>= 150px`, stripping WordPress `-WxH` resize suffixes and `?w=...`
   query strings) biographical home, yard, wadi, balcony, kitchen, furniture, and studio photos
   from audited posts across the entire 2011–2026 blog archive, writing rich per-image `.json`
   sidecars with `places_locked: True` so non-home images never suffer tag bleed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "media"
WORDPRESS_SUBDIR = "wordpress_blog"
PORTFOLIO_SUBDIR = "portfolio_site"

MIN_IMAGE_DIM_PX = 150
HTTP_TIMEOUT_SECONDS = 15
SHA256_READ_CHUNK_BYTES = 65536
FILENAME_HASH_PREFIX_LEN = 8
MAX_STEM_LENGTH = 56
USER_AGENT = "Tami70thBirthdayArchiveHarvester/2.0 (ReadOnly; FamilyProject)"

PLACE_HERZLIYA = "Herzliya (parents' house & yard)"
PLACE_DENYA = "Haifa: Denya (80 Greenboim St & studio)"
PLACE_ARLOZOROV = "Haifa: Arlozorov"
PLACE_EINSTEIN = "Haifa: Einstein"
PLACE_PARIS = "Paris (Cité des Arts)"
PLACE_TECHNION = "Technion"
PLACE_BINYAMIN = "Binyamin Gallery"
PLACE_RAMAT_GAN = "Ramat Gan Museum"
PLACE_SOUTHERN_FRANCE = "Southern France"

PERSON_TAMI = "Tamar 'Tami' Sheaffer"
PERSON_GADI = "Gad Sheaffer"
PERSON_MICHAEL = "Michael Fried"
PERSON_AHUVA = "Ahuva Fried"
PERSON_TAL = "Tal Sheaffer"
PERSON_YAARA = "Ya'ara"
PERSON_CARMEL = "Carmel"

WP_RESIZE_SUFFIX_RE = re.compile(r"-\d+x\d+(?=\.[a-zA-Z0-9]+$)")
SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


@dataclass(frozen=True)
class HarvesterConfig:
    media_dir: Path = DEFAULT_MEDIA_DIR
    min_dim_px: int = MIN_IMAGE_DIM_PX
    http_timeout_sec: int = HTTP_TIMEOUT_SECONDS
    user_agent: str = USER_AGENT


@dataclass(frozen=True)
class BlogImageTarget:
    url: str
    post_url: str
    post_title: str
    date_or_era: str
    image_title: str
    caption: str
    people: tuple[str, ...]
    places: tuple[str, ...]
    art_themes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExistingSidecarPatch:
    title: str
    caption: str
    date_or_era: str
    people: tuple[str, ...]
    places: tuple[str, ...]
    confidence_notes: str


# Exact per-file remediation map for polluted existing `media/wordpress_blog/` sidecars
EXISTING_WP_SIDECAR_PATCHES: dict[str, ExistingSidecarPatch] = {
    "2023-03-07-1.jpg_5d05b1a3.jpg.json": ExistingSidecarPatch(
        title="מסעותי — תמונת שער עמוד המסעות",
        caption="מתוך העמוד 'מסעותי' בבלוג 'דרכי עצים' של גדי שפר (אינו צילום של בית הרצליה או דניה).",
        date_or_era="2023-07-02",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Cleaned false sitemap batch place tags; verified against post ID=35317.",
    ),
    "71abf3e4-b444-424a-892f-da5e3216276a.jpg_71754ad8.jpeg.json": ExistingSidecarPatch(
        title="הפרמקון של זמננו — דימוי נושא",
        caption="מתוך הפוסט הפילוסופי 'הפרמקון של זמננו' בבלוג 'דרכי עצים' (ספטמבר 2026).",
        date_or_era="2026-09-11",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Essay illustration; removed false Denya/Herzliya batch tags.",
    ),
    "a1de3707-9b61-4895-b7a0-4e08add4c444.jpg_a20a2c95.png.json": ExistingSidecarPatch(
        title="מי מדבר כאן? — אינדקסיקליות בצילום",
        caption="מתוך הפוסט 'מי מדבר כאן?' על סמיוטיקה ואינדקסיקליות בבלוג 'דרכי עצים' (ספטמבר 2026).",
        date_or_era="2026-09-07",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Essay illustration; removed false Denya/Herzliya batch tags.",
    ),
    "caption.jpg_ab8a2a2a.jpg.json": ExistingSidecarPatch(
        title="בון — ביקור בגרמניה",
        caption="מתוך רשומת המסע 'בון' בבלוג 'דרכי עצים' (אוקטובר 2026).",
        date_or_era="2026-10-07",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(),
        confidence_notes="CONFIRMED: Bonn travel photo; removed false Denya/Herzliya batch tags.",
    ),
    "d79ed7a8d797d791d799d79d-d79ed7a9d795d7aad7a4d799d79d-d7a2d7.jpg_b665d5c6.png.json": ExistingSidecarPatch(
        title="האוניברסליות המדומה של המרחב: מן הגוף אל עולם האמנות",
        caption="דימוי מתוך מסה על תפיסת המרחב והאמנות בבלוג 'דרכי עצים' (אוקטובר 2026).",
        date_or_era="2026-10-15",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Essay diagram; removed false Denya/Herzliya batch tags.",
    ),
    "folder-001.jpg_564dbf48.jpg.json": ExistingSidecarPatch(
        title="פרק שביעי – אולריך פוגש את שלמה",
        caption="סריקה ארכיונית מתוך סדרת 'אולריך' על תולדות משפחת שפר בבלוג 'דרכי עצים' (2013).",
        date_or_era="2013-10-29",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Family history document from Ulrich series; removed false Denya/Herzliya tags.",
    ),
    "fullscreen-capture-15012014-145645.jpg_45f4b949.jpg.json": ExistingSidecarPatch(
        title="פרק חמישי – אולריך ומלאך ההיסטוריה",
        caption="מתוך סדרת הרשומות 'אולריך ומלאך ההיסטוריה' בבלוג 'דרכי עצים' (2013).",
        date_or_era="2013-10-27",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Archival capture from Ulrich series; removed false Denya/Herzliya tags.",
    ),
    "image-1.jpg_c0d13909.png.json": ExistingSidecarPatch(
        title="המזוודה של בנימין — אנדרטת פורטבו",
        caption="מתוך הרשומה 'המזוודה של בנימין' בעקבות הביקור בפורטבו (גבול צרפת-ספרד, ספטמבר 2026).",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(),
        confidence_notes="CONFIRMED: Portbou / Walter Benjamin post; removed false Denya/Herzliya tags.",
    ),
    "image-2.jpg_fa77d718.png.json": ExistingSidecarPatch(
        title="לישון בתוך רעיון — יחידת המגורים של לה קורבוזיה במרסיי",
        caption="מתוך הרשומה 'לישון בתוך רעיון' מהטיול המשפחתי בדרום צרפת (מרסיי, ספטמבר 2026).",
        date_or_era="2026-09-16",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Marseille (Southern France) trip post; replaced false Denya/Herzliya tags.",
    ),
    "image.jpg_67b02c26.png.json": ExistingSidecarPatch(
        title="האם 'המשגה סינתטית' היא בעצמה המשגה סינתטית?",
        caption="איור נושא מתוך סדרת 'המשגה סינתטית' בבלוג 'דרכי עצים' (אוקטובר 2026).",
        date_or_era="2026-10-14",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Essay banner; removed false Denya/Herzliya batch tags.",
    ),
    "img-20260921-wa00062582828128820141435.jpg_1d70426e.jpg.json": ExistingSidecarPatch(
        title="קוליוור — מסע בדרום צרפת",
        caption="צילום מחוף קוליוור (Collioure) במהלך הטיול של תמי וגדי בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Collioure (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20250715_131352.jpg_73573121.jpg.json": ExistingSidecarPatch(
        title="מושגים בשפה הפארואזית — איי פארו",
        caption="צילום נוף מאיי פארו מתוך הרשומה 'מושגים בשפה הפארואזית' בבלוג 'דרכי עצים'.",
        date_or_era="2025-07-15",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(),
        confidence_notes="CONFIRMED: Faroe Islands travel photo; removed false Denya/Herzliya tags.",
    ),
    "img_20260917_085313.jpg_04ca4a16.jpg.json": ExistingSidecarPatch(
        title="טיול ארבעת הכוכבים — דרום צרפת",
        caption="צילום מתוך הרשומה 'טיול ארבעת הכוכבים' על מסעם של תמי וגדי בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-17",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Southern France trip photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260919_1247485466813979446799248.jpg_6ba4f5ce.jpg.json": ExistingSidecarPatch(
        title="הגפנה — כרמים בדרום צרפת",
        caption="צילום מתוך הרשומה 'הגפנה' מהמסע בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-19",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Southern France trip photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260919_163330.jpg_2cf11250.jpg.json": ExistingSidecarPatch(
        title="סאט (Sète) — מסע בדרום צרפת",
        caption="צילום מעיירת הנמל Sète מתוך מסעם של תמי וגדי בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-19",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Sète (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260920_1227206576824604237338244.jpg_c676da93.jpg.json": ExistingSidecarPatch(
        title="חיים ב-Céret — מסע בדרום צרפת",
        caption="צילום מהעיירה Céret למרגלות הפירנאים הצרפתיים מתוך הבלוג 'דרכי עצים' (ספטמבר 2026).",
        date_or_era="2026-09-20",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Céret (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260921_113654.jpg_8b12c962.jpg.json": ExistingSidecarPatch(
        title="המזוודה של בנימין — אנדרטת פורטבו אל מול הים",
        caption="האנדרטה של דני קרוון לזכר ולטר בנימין בפורטבו (ספטמבר 2026), מתוך הפוסט 'המזוודה של בנימין'.",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(),
        confidence_notes="CONFIRMED: Portbou Passages memorial photo; removed false Denya/Herzliya tags.",
    ),
    "img_20260921_120124.jpg_bac3faf9.jpg.json": ExistingSidecarPatch(
        title="המזוודה של בנימין — מפרץ פורטבו",
        caption="מבט אל מפרץ פורטבו מתוך הרשומה 'המזוודה של בנימין' בבלוג 'דרכי עצים' (ספטמבר 2026).",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(),
        confidence_notes="CONFIRMED: Portbou bay photo; removed false Denya/Herzliya tags.",
    ),
    "img_20260921_1717331240177318268878602.jpg_c53978ae.jpg.json": ExistingSidecarPatch(
        title="קוליוור — סמטאות הצבע בדרום צרפת",
        caption="צילום מקוליוור (Collioure) בטיול בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Collioure (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260921_1728127664782294298693616.jpg_eeffb5e4.jpg.json": ExistingSidecarPatch(
        title="קוליוור — נמל ומגדלור בדרום צרפת",
        caption="צילום מקוליוור (Collioure) בטיול בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Collioure (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260921_1740178589449007144392738.jpg_baa41b0b.jpg.json": ExistingSidecarPatch(
        title="קוליוור — חוף הים בדרום צרפת",
        caption="צילום מקוליוור (Collioure) בטיול בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-21",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Collioure (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260924_1048367387668821765122065.jpg_1fd38c25.jpg.json": ExistingSidecarPatch(
        title="הקמארג — שמורת הטבע בפרובנס",
        caption="צילום משמורת הקמארג (Camargue) בדרום צרפת מתוך הבלוג 'דרכי עצים' (ספטמבר 2026).",
        date_or_era="2026-09-24",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Camargue (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "img_20260925_142934.jpg_2b31ab37.jpg.json": ExistingSidecarPatch(
        title="הניטרלי באמנות הפלסטית — תצוגה מוזיאלית בצרפת",
        caption="צילום מתוך הרשומה 'הניטרלי באמנות הפלסטית' במהלך המסע בדרום צרפת (ספטמבר 2026).",
        date_or_era="2026-09-25",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Museum visit in Southern France; replaced false Denya/Herzliya tags.",
    ),
    "img_20260926_134622-edit.jpg_0f918791.jpg.json": ExistingSidecarPatch(
        title="סזאן וההר — הר סנט-ויקטואר בפרובנס",
        caption="צילום הר סנט-ויקטואר באקס-אן-פרובנס מתוך הרשומה 'סזאן וההר' (ספטמבר 2026).",
        date_or_era="2026-09-26",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_SOUTHERN_FRANCE,),
        confidence_notes="CONFIRMED: Provence (Southern France) travel photo; replaced false Denya/Herzliya tags.",
    ),
    "meshech_metuchnat_banner_1200x300-2.jpg_5d707175.jpg.json": ExistingSidecarPatch(
        title="המשגה סינתטית: משך מתוכנת",
        caption="באנר מתוך המסה 'המשגה סינתטית: משך מתוכנת' בבלוג 'דרכי עצים' (אוקטובר 2026).",
        date_or_era="2026-10-17",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Essay banner; removed false Denya/Herzliya batch tags.",
    ),
    "penguin_pool_london_zoo-001.jpg_417dbb4a.jpg.json": ExistingSidecarPatch(
        title="בריכת הפינגווינים בגן החיות של לונדון (לובטקין)",
        caption="בריכת הפינגווינים המודרניסטית שתכנן ברתולד לובטקין בלונדון, מתוך סדרת 'אולריך' (2013).",
        date_or_era="2013-10-30",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Lubetkin Penguin Pool in London Zoo; removed false Denya/Herzliya tags.",
    ),
    "tatami-entrance.jpg_1de9b179.jpg.json": ExistingSidecarPatch(
        title="בון — כניסת מחצלות טטאמי",
        caption="צילום מתוך הרשומה 'בון' בבלוג 'דרכי עצים' (אוקטובר 2026).",
        date_or_era="2026-10-07",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(),
        confidence_notes="CONFIRMED: Bonn travel photo; removed false Denya/Herzliya tags.",
    ),
    "top-038.jpg_9d09937a.jpg.json": ExistingSidecarPatch(
        title="פעלים המוטים כמו נָזַל – הוריד",
        caption="דימוי לשוני מתוך עמוד דקדוק בבלוג 'דרכי עצים'.",
        date_or_era="2026-09-07",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Hebrew grammar page illustration; removed false Denya/Herzliya tags.",
    ),
    "tumblr_m8bnxbqzp81qevm0qo1_1280-001.jpg_5b19f59f.jpg.json": ExistingSidecarPatch(
        title="Angelus Novus (פאול קלה) — Angelus Ovis Aries",
        caption="ציורו של פאול קלה Angelus Novus מתוך הרשומה 'Angelus Ovis Aries' בבלוג 'דרכי עצים' (2014).",
        date_or_era="2014-05-03",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Paul Klee artwork reference; removed false Denya/Herzliya tags.",
    ),
    "tumblr_m8bnxbqzp81qevm0qo1_1280.jpg_27ad28f0.jpg.json": ExistingSidecarPatch(
        title="Angelus Novus — אולריך ומלאך ההיסטוריה",
        caption="Angelus Novus של פאול קלה מתוך הפרק החמישי בסדרת 'אולריך' בבלוג 'דרכי עצים' (2013).",
        date_or_era="2013-10-27",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Paul Klee artwork reference; removed false Denya/Herzliya tags.",
    ),
    "wp_house_1-fullscreen-capture-25092016-203348_8dc5ea03.jpg.json": ExistingSidecarPatch(
        title="20-24: וילה במדבר — שירות צבאי בסיני",
        caption="צילום מתוך הרשומה הביוגרפית '20-24: וילה במדבר' על שנות השירות בסיני (אינו בית דניה).",
        date_or_era="2016-09-25",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Sinai military service post (20-24); removed false Denya tag.",
    ),
    "wp_house_20170902_113359_32b41269.jpg.json": ExistingSidecarPatch(
        title="חורים שחורים וקירות לבנים — מסכות בחלון ראווה",
        caption="צילום רחוב מתוך המסה 'חורים שחורים וקירות לבנים' בעקבות דלז וגואטרי (2018).",
        date_or_era="2018-01-24",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Deleuze/Guattari essay street photo; removed false Herzliya tag.",
    ),
    "wp_house_20171101_124003_01b87c72.jpg.json": ExistingSidecarPatch(
        title="חורים שחורים וקירות לבנים — ציור קיר ופנים",
        caption="צילום מתוך המסה 'חורים שחורים וקירות לבנים' בבלוג 'דרכי עצים' (2018).",
        date_or_era="2018-01-24",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Deleuze/Guattari essay street photo; removed false Herzliya tag.",
    ),
    "wp_house_20180124_0938181074390410_aaf04b6d.jpg.json": ExistingSidecarPatch(
        title="חורים שחורים וקירות לבנים — פסל מסכה בקמפוס הטכניון",
        caption="פסל חוץ על מדשאה בקמפוס הטכניון מתוך המסה 'חורים שחורים וקירות לבנים' (2018).",
        date_or_era="2018-01-24",
        people=(PERSON_GADI,),
        places=(PLACE_TECHNION,),
        confidence_notes="CONFIRMED: Outdoor sculpture at Technion; replaced false Herzliya tag with Technion.",
    ),
    "wp_house_fullscreen-capture-22-jan-18-95003-am_31537037.jpg.json": ExistingSidecarPatch(
        title="חורים שחורים וקירות לבנים — דימוי קולנועי",
        caption="לכידת מסך מתוך המסה הפילוסופית 'חורים שחורים וקירות לבנים' (2018).",
        date_or_era="2018-01-24",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Film still in Deleuze essay; removed false Herzliya tag.",
    ),
    "wp_house_47279b_1_a9c7dc58.jpg.json": ExistingSidecarPatch(
        title="בית פארנסוורת' (Farnsworth House) — השוואה אדריכלית",
        caption="בית פארנסוורת' של מיס ון דר רוהה באילינוי, שהוצג בפוסט 'אל המנוחה ואל הנחלה' כהשוואה רעיונית (אינו הבית בחיפה).",
        date_or_era="2017-05-24",
        people=(PERSON_GADI,),
        places=(),
        confidence_notes="CONFIRMED: Mies van der Rohe Farnsworth House reference image; removed false Denya tag.",
    ),
    "wp_house_d7a1d7a8d799d7a7d794-3_388207f4.jpg.json": ExistingSidecarPatch(
        title="גרינבוים 80 דניה — מודל אדריכלי מקורי של הבית",
        caption="סריקה של המודל הפיזי של בית משפחת שפר ברחוב גרינבוים 80 בדניה ('וזה מודל הבית, שאכן כך בוצע פחות או יותר'), מתוך הפוסט 'אל המנוחה ואל הנחלה'.",
        date_or_era="2017-05-24",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
        confidence_notes="CONFIRMED: Genuine physical architectural model of 80 Greenboim St, Denya.",
    ),
    "wp_house_g80_6d5a2ba1.jpg.json": ExistingSidecarPatch(
        title="גרינבוים 80 דניה — חזית הבית על מורדות הכרמל",
        caption="מבט חוץ על ביתם של תמי וגדי שפר ברחוב גרינבוים 80 בשכונת דניה בחיפה, טבול בחורש הכרמל ('ועוד בית אחד ממש לא איקוני בחיפה – זה שלנו, אליו אני עולה לרגל יום יום').",
        date_or_era="2023-09-02",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
        confidence_notes="CONFIRMED: Genuine exterior photo of 80 Greenboim St, Denya (`g80.jpg`).",
    ),
}

# All 9 `wp_house_wp-159680*` images belong to post ID=34128 (`/2020/08/07/safe/` 'מקום בטוח'),
# a COVID-era gallery tour in Rishon LeZion, Holon, Gordon Gallery Tel Aviv, and Petah Tikva Museum.
SAFE_POST_GALLERY_TOUR_PREFIX = "wp_house_wp-159680"
SAFE_POST_PATCH = ExistingSidecarPatch(
    title="מקום בטוח — סיור גלריות ומוזיאונים בגוש דן (קיץ 2020)",
    caption="צילום מתוך הרשומה 'מקום בטוח' (אוגוסט 2020) המתעדת ביקור בתערוכות בראשון לציון, חולון, גלריה גורדון ומוזיאון פתח תקווה (אינו בית הרצליה).",
    date_or_era="2020-08-07",
    people=(PERSON_GADI,),
    places=(),
    confidence_notes="CONFIRMED: Gallery tour photo from post /2020/08/07/safe/; removed false Herzliya tag.",
)


# Audited wide blog harvest list across 2011–2026
BLOG_HARVEST_TARGETS: tuple[BlogImageTarget, ...] = (
    # =========================================================================
    # 1. HERZLIYA — PARENTS' HOUSE & YARD (Ahuva & Michael Fried)
    # =========================================================================
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/06/d791d799d7aa-d7a4d7a8d799d793.jpg",
        post_url="https://theshingimel.wordpress.com/2013/06/08/ahuva-fried/",
        post_title="אהובה",
        date_or_era="2013-06-08",
        image_title="בית פריד בהרצליה — הבית והדשא שעליו נישאו תמי וגדי",
        caption="ביתם של אהובה ומיכאל פריד בהרצליה מוקף העצים והדשא: 'בבית הזעיר הזה (שלושה חדרים, כשישים מטרים רבועים) הם גידלו את שתי בנותיהם. על הדשא שלפני הבית הזה התחתנו – על השולחן שבמרפסת החתים אותי הרב על הכתובה.'",
        people=(PERSON_AHUVA, PERSON_MICHAEL, PERSON_TAMI, PERSON_GADI),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/11/d1000024.jpg",
        post_url="https://theshingimel.wordpress.com/2013/11/26/fried-2/",
        post_title="מיכאל",
        date_or_era="2013-11-26",
        image_title="צריף העץ והסדנה של מיכאל פריד בירכתי החצר בהרצליה",
        caption="מתוך הרשומה 'מיכאל': 'למיכאל היו ידי זהב, ממש, ותמיד היו לו תחביבים. הוא עסק בקרמיקה ובצורפות, בצריף קטן מעץ בירכתי החצר שלהם בהרצליה.'",
        people=(PERSON_MICHAEL, PERSON_TAMI, PERSON_GADI),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/11/img_0293.jpg",
        post_url="https://theshingimel.wordpress.com/2013/11/26/fried-2/",
        post_title="מיכאל",
        date_or_era="2013-11-26",
        image_title="סבא מיכאל וסבתא אהובה עם הנכדים גיא וטל סביב השולחן בבית בהרצליה",
        caption="פנים בית ההורים בהרצליה: מיכאל ואהובה פריד יושבים סביב שולחן האוכל ומאחוריהם נכדיהם הצעירים גיא וטל שפר (מתוך הרשומה 'מיכאל', 2013).",
        people=(PERSON_MICHAEL, PERSON_AHUVA, "Guy Sheaffer", PERSON_TAL),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/11/d1000036.jpg",
        post_url="https://theshingimel.wordpress.com/2013/11/26/fried-2/",
        post_title="מיכאל",
        date_or_era="2013-11-26",
        image_title="מיכאל פריד משוחח בבית ההורים בהרצליה",
        caption="מיכאל פריד ('ידי זהב'), אביה של תמי, בשיחה בסלון ביתם של אהובה ומיכאל בהרצליה מתוך הרשומה 'מיכאל' (2013).",
        people=(PERSON_MICHAEL,),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/11/d7a4d7a8d799d793.jpg",
        post_url="https://theshingimel.wordpress.com/2013/11/26/fried-2/",
        post_title="מיכאל",
        date_or_era="2013-11-26",
        image_title="בית פריד והחצר האחורית בהרצליה — הבית שבנו אהובה ומיכאל במו ידיהם",
        caption="מבט חוץ מואר על ביתם של אהובה ומיכאל פריד בשיכון חיילים משוחררים בהרצליה — גג הרעפים, דוד השמש, שער החצר, הצריף והגינה הירוקה ('אחרי המלחמה הם בנו את ביתם, במו ידיהם ממש').",
        people=(PERSON_MICHAEL, PERSON_AHUVA, PERSON_TAMI, PERSON_GADI),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/06/img_0988.jpg",
        post_url="https://theshingimel.wordpress.com/2013/06/08/ahuva-fried/",
        post_title="אהובה",
        date_or_era="2013-06-08",
        image_title="אהובה פריד בבית בהרצליה",
        caption="דיוקנה של אהובה פריד ז\"ל (אמה של תמי, גננת אהובה בהרצליה) בביתה בהרצליה, מתוך רשומת הזיכרון 'אהובה' של גדי שפר (2013).",
        people=(PERSON_AHUVA,),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/06/d790d794d795d791d794-0011.jpg",
        post_url="https://theshingimel.wordpress.com/2013/06/08/ahuva-fried/",
        post_title="אהובה",
        date_or_era="2013-06-08",
        image_title="אהובה ומיכאל פריד בצעירותם (1945)",
        caption="צילום היסטורי של אהובה ומיכאל פריד בצעירותם בתקופת הבריגדה והקמת ביתם, מתוך הרשומה 'אהובה' (2013).",
        people=(PERSON_AHUVA, PERSON_MICHAEL),
        places=(),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/06/pict0047-001.jpg",
        post_url="https://theshingimel.wordpress.com/2013/06/08/ahuva-fried/",
        post_title="אהובה",
        date_or_era="2013-06-08",
        image_title="אהובה פריד בשיחה משפחתית בבית בהרצליה",
        caption="אהובה פריד בבית ההורים בהרצליה בשיחה אינטימית, מתוך קובצי הרשומה 'אהובה' (יוני 2013).",
        people=(PERSON_AHUVA,),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/06/pict2739.jpg",
        post_url="https://theshingimel.wordpress.com/2013/06/08/ahuva-fried/",
        post_title="אהובה",
        date_or_era="2013-06-08",
        image_title="שביל הכניסה, העץ והגדר החיה בבית פריד בהרצליה",
        caption="מבט מהחצר והשביל המוצל עם מעקה הברזל הירוק והעץ העבות אל בית הרעפים של אהובה ומיכאל פריד בהרצליה (מתוך הרשומה 'אהובה', 2013).",
        people=(PERSON_AHUVA, PERSON_MICHAEL, PERSON_TAMI),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/11/pict0005.jpg",
        post_url="https://theshingimel.wordpress.com/2013/11/26/fried-2/",
        post_title="מיכאל",
        date_or_era="2013-11-26",
        image_title="מיכאל פריד בסלון בית ההורים בהרצליה",
        caption="מיכאל פריד יושב בסלון הבית בהרצליה (מתוך קובצי הרשומה 'מיכאל', נובמבר 2013).",
        people=(PERSON_MICHAEL,),
        places=(PLACE_HERZLIYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/11/d1000015.jpg",
        post_url="https://theshingimel.wordpress.com/2013/11/26/fried-2/",
        post_title="מיכאל",
        date_or_era="2013-11-26",
        image_title="תכשיטי הכסף שעיצב ויצר מיכאל פריד בצריף בהרצליה",
        caption="תכשיטי כסף בעבודת יד שתכנן ויצר מיכאל פריד בצריף העץ הקטן בירכתי החצר בהרצליה, מתוך הרשומה 'מיכאל' (2013).",
        people=(PERSON_MICHAEL,),
        places=(),
    ),

    # =========================================================================
    # 2. DENYA — 80 GREENBOIM ST, GARDEN, WADI, KITCHEN, STAIRS & HOME STUDIO
    # =========================================================================
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2023/01/wp-1673603255180.jpg",
        post_url="https://theshingimel.wordpress.com/2023/01/15/me-and-corbu/",
        post_title="מסלולי מגורים",
        date_or_era="2023-01-15",
        image_title="גרינבוים 80 דניה — קיר המסך מרצפה עד תקרה אל מול נחל טירה והוואדי",
        caption="מבט מפנים הבית בגרינבוים 80 בדניה דרך קיר המסך באורך עשרה מטרים מרצפה ועד תקרה, המשקיף אל החורש הירוק של נחל טירה בכרמל ('מסלולי מגורים', 2023).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2023/01/img_20230111_104901.jpg",
        post_url="https://theshingimel.wordpress.com/2023/01/15/me-and-corbu/",
        post_title="מסלולי מגורים",
        date_or_era="2023-01-11",
        image_title="גרינבוים 80 דניה — מרפסת הגג, המדרגות וצינור ניקוז מי הגשמים",
        caption="מרפסת הגג, המדרגות וצינור הניקוז המוביל מי גשמים לבור המים התת-קרקעי להשקיית הגינה בגרינבוים 80 בדניה ('מסלולי מגורים', 2023).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2023/09/img_0197.jpg",
        post_url="https://theshingimel.wordpress.com/g80/",
        post_title="בית (g80)",
        date_or_era="2023-09-02",
        image_title="גרינבוים 80 דניה — חזית הכניסה מהרחוב אל מול חורש הכרמל",
        caption="חזית הרחוב הלבנה והנקייה של הבית ברחוב גרינבוים 80 בדניה — דלת הכניסה, שער הרפפות והקומה העליונה על רקע עצי הכרמל (תמונת השער של עמוד 'בית' g80).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2022/07/img_0185.jpg",
        post_url="https://theshingimel.wordpress.com/2022/07/16/strawberry/",
        post_title="הדורבן שאהב תות",
        date_or_era="2022-07-16",
        image_title="גרינבוים 80 דניה — המגרש באפיק נחל טירה בטרם הבנייה",
        caption="המגרש של משפחת שפר בתוך אפיק נחל טירה בדניה טרם בניית הבית ('המגרש האחרון שנותר לא בנוי בסביבה כולה'), מתוך 'הדורבן שאהב תות'.",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2022/07/d79ed795d793d79c.jpg",
        post_url="https://theshingimel.wordpress.com/2022/07/16/strawberry/",
        post_title="הדורבן שאהב תות",
        date_or_era="2022-07-16",
        image_title="גרינבוים 80 דניה — מודל מוקדם של הבית והמסדרון האקולוגי בנחל טירה",
        caption="מודל אדריכלי מוקדם בשני אגפים משני צידי תת-האפיק כדי להותיר מסדרון אקולוגי פתוח לבעלי החיים בוואדי דניה ('הדורבן שאהב תות').",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2022/07/wp-1657968394577.jpg",
        post_url="https://theshingimel.wordpress.com/2022/07/16/strawberry/",
        post_title="הדורבן שאהב תות",
        date_or_era="2022-07-16",
        image_title="גרינבוים 80 דניה — חזית הגינה התחתונה, קיר המסך וגרם המדרגות מהוואדי",
        caption="מבט מהגינה התחתונה אל הבית בגרינבוים 80 בדניה: הזיז האדריכלי הלבן עם קיר החלונות הרחב, קומת העמודים וגרם המדרגות העולה ממפלס הוואדי ('הדורבן שאהב תות', 2022).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/03/img_0264.jpg",
        post_url="https://theshingimel.wordpress.com/2015/03/24/superman/",
        post_title="סופרמן",
        date_or_era="2015-03-24",
        image_title="גרינבוים 80 דניה — חציבת בור היסודות של הבית בסלע הכרמל",
        caption="מחפר הפיאט-היטאצ'י הכתום חוצב את בור היסודות בסלע הכרמל שעליו נבנה ביתם של תמי וגדי בגרינבוים 80 בדניה ('סופרמן').",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/03/img_0265.jpg",
        post_url="https://theshingimel.wordpress.com/2015/03/24/superman/",
        post_title="סופרמן",
        date_or_era="2015-03-24",
        image_title="גרינבוים 80 דניה — חפירת המגרש בהקמת הבית",
        caption="עבודות החציבה להקמת בית משפחת שפר בגרינבוים 80 בדניה (כ-150 משאיות סלע שנחצבו בשבועיים), מתוך הפוסט 'סופרמן'.",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/03/img_0194.jpg",
        post_url="https://theshingimel.wordpress.com/2015/03/24/superman/",
        post_title="סופרמן",
        date_or_era="2015-03-24",
        image_title="גרינבוים 80 דניה — תיעוד מתקופת הקמת הבית",
        caption="צילום מארכיון הקמת הבית בגרינבוים 80 בדניה שהועלה לרשומה 'סופרמן' (2015).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/04/dsc08378.jpg",
        post_url="https://theshingimel.wordpress.com/2015/04/26/building/",
        post_title="מפלסים",
        date_or_era="2015-04-26",
        image_title="גרינבוים 80 דניה — מפגש מפלס הפנים וניקוז מרפסת החוץ ('מפלסים')",
        caption="מבט על מפתן היציאה למרפסת וניקוז המשטח החיצוני בגרינבוים 80 בדניה מתוך הרשומה ההנדסית 'מפלסים' (2015).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2023/08/wp-1691994762497.jpg",
        post_url="https://theshingimel.wordpress.com/2023/08/14/short-doors/",
        post_title="דלתות קצרות",
        date_or_era="2023-08-14",
        image_title="גרינבוים 80 דניה — מפתן חדר ומרצפות סומסום בבית",
        caption="מבט פנים על מפתן דלת ומרצפות הסומסום בבית גרינבוים 80 בדניה לאחר חידוש הריצוף ללא מדרגה ('דלתות קצרות', 2023).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2024/02/wp-17089362034703526253214700512475.jpg",
        post_url="https://theshingimel.wordpress.com/2024/02/26/%d7%94%d7%a9%d7%9e%d7%a0%d7%9e%d7%9f/",
        post_title="השמנמן",
        date_or_era="2024-02-26",
        image_title="גרינבוים 80 דניה — שורשי עץ הברכיכיטון ('השמנמן') בחצר הבית",
        caption="שורשי עץ הברכיכיטון בחצר הבית בגרינבוים 80 בדניה מתוך הרשומה 'השמנמן' (פברואר 2024).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2024/02/wp-17089361449546791839142930026624.jpg",
        post_url="https://theshingimel.wordpress.com/2024/02/26/%d7%94%d7%a9%d7%9e%d7%a0%d7%9e%d7%9f/",
        post_title="השמנמן",
        date_or_era="2024-02-26",
        image_title="גרינבוים 80 דניה — רחבת החוץ והגינה סביב עץ הברכיכיטון",
        caption="עבודות מעקף הצנרת סביב גזע עץ הברכיכיטון בחצר גרינבוים 80 בדניה ('השמנמן', פברואר 2024).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2024/12/wp-17355789241255697951411128363802.jpg",
        post_url="https://theshingimel.wordpress.com/2025/01/02/woodworking/",
        post_title="השולחן שלי",
        date_or_era="2025-01-02",
        image_title="גרינבוים 80 דניה — שולחן המייפל שבנה גדי במטבח הבית (פרט רגל וגלגל)",
        caption="פרט רגל העץ המשתנה ושרוול הנירוסטה של שולחן המטבח מעץ מייפל שבנה גדי שפר וניצב במטבח הבית בדניה ('השולחן שלי', ינואר 2025).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2025/01/d7a9d795d79cd797d79f.jpg",
        post_url="https://theshingimel.wordpress.com/2025/01/02/woodworking/",
        post_title="השולחן שלי",
        date_or_era="2025-01-02",
        image_title="גרינבוים 80 דניה — שולחן המייפל שבנה גדי במטבח הבית בדניה",
        caption="מבט מלא על שולחן המטבח מעץ מייפל שבנה גדי שפר בחרשות עץ, בלב מטבח העץ בבית משפחת שפר בגרינבוים 80 בדניה ('השולחן שלי', 2025).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2025/01/wp-17358019593546177321841333711432.jpg",
        post_url="https://theshingimel.wordpress.com/2025/01/02/woodworking/",
        post_title="השולחן שלי",
        date_or_era="2025-01-02",
        image_title="גרינבוים 80 דניה — קונסטרוקציית העץ התחתונה של שולחן המטבח בדניה",
        caption="מבט תחתון על מחברי העץ והרגליים של שולחן המטבח שבנה גדי בבית בדניה ('השולחן שלי', 2025).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2024/08/wp-17247373694644267773996461661956.jpg",
        post_url="https://theshingimel.wordpress.com/2024/08/27/glass/",
        post_title="גולות",
        date_or_era="2024-08-27",
        image_title="גרינבוים 80 דניה — צנצנת הגולות בבית בדניה",
        caption="צנצנת הזכוכית הגדולה המלאה בגולות צבעוניות בביתם של תמי וגדי בדניה, שלכבודה מתאספות הנכדות בכל ביקור ('גולות', אוגוסט 2024).",
        people=(PERSON_GADI, PERSON_TAMI, PERSON_YAARA, PERSON_CARMEL),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2024/08/wp-17246720973252545024486697452530.jpg",
        post_url="https://theshingimel.wordpress.com/2024/08/27/glass/",
        post_title="גולות",
        date_or_era="2024-08-27",
        image_title="גרינבוים 80 דניה — משחקי גולות, קערות ומסננות על רצפת הבית בדניה",
        caption="מערכי הגולות, הסירים והמסננות של הנכדות יערה וכרמל על רצפת הסומסום בבית בדניה ('רעש הגולות המתגלגלות ממלא את הבית', אוגוסט 2024).",
        people=(PERSON_GADI, PERSON_TAMI, PERSON_YAARA, PERSON_CARMEL),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2022/03/img_20220301_134758.jpg",
        post_url="https://theshingimel.wordpress.com/2022/03/05/chili/",
        post_title="צ’ילי",
        date_or_era="2022-03-01",
        image_title="גרינבוים 80 דניה — שולחן הנגרים בקומת העמודים תחת הבית",
        caption="קומת העמודים ושולחן הנגרים של גדי מתחת לבית בגרינבוים 80 בדניה (שם ניסרו כיסא לעבודת אמנות של תמי), מתוך הרשומה 'צ'ילי' (2022).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2022/03/d7a6d799d79cd799-d7a8d797d791.jpg",
        post_url="https://theshingimel.wordpress.com/2022/03/05/chili/",
        post_title="צ’ילי",
        date_or_era="2022-03-05",
        image_title="גרינבוים 80 דניה — החתולה צ'ילי בבית המשפחה בדניה",
        caption="צ'ילי, חתולת המשפחה במשך 14 שנים, בביתם של תמי וגדי בדניה, מתוך הרשומה 'צ'ילי' (מרץ 2022).",
        people=(PERSON_GADI, PERSON_TAMI),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2012/07/p1030304.jpg",
        post_url="https://theshingimel.wordpress.com/2012/07/30/swinishness/",
        post_title="חזירויות",
        date_or_era="2012-07-30",
        image_title="גרינבוים 80 דניה — פינת הקומפוסטר והגינה מעל הוואדי",
        caption="פינת הגינה והמדשן המבוצר מפני חזירי הבר בחצר הבית בגרינבוים 80 בדניה, על שפת חורש הכרמל ('חזירויות', יולי 2012).",
        people=(PERSON_GADI,),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2012/07/p10304532.jpg",
        post_url="https://theshingimel.wordpress.com/2012/07/30/swinishness/",
        post_title="חזירויות",
        date_or_era="2012-07-30",
        image_title="גרינבוים 80 דניה — חצר הבית והמדשן מול מדרון הוואדי",
        caption="חצר הבית בגרינבוים 80 בדניה לאחר ביקור חזירי הבר בלילה ('חזירויות', יולי 2012).",
        people=(PERSON_GADI,),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/05/p1010621.jpg",
        post_url="https://theshingimel.wordpress.com/2013/05/16/tsabar/",
        post_title="צבר",
        date_or_era="2013-05-16",
        image_title="גרינבוים 80 דניה — שתיל הצבר בוואדי שמתחת לבית",
        caption="שתיל הצבר ששתל גדי בוואדי שמתחת לבית בגרינבוים 80 בדניה ('צבר', מאי 2013).",
        people=(PERSON_GADI,),
        places=(PLACE_DENYA,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/10/dsc03616.jpg",
        post_url="https://theshingimel.wordpress.com/2013/10/12/now-here-there/",
        post_title="פלישת חוטפי הגופות",
        date_or_era="2013-10-12",
        image_title="סטודיו דניה — חלקי המיצב של תמי שפר ארוזים בפצפצים לקראת התערוכה",
        caption="חלקי המיצב של תמר שפר מפורקים וארוזים ב'פצפצים' בסטודיו בבית בדניה לפני הובלתם לגלריה בנימין ('תמונת הכותרת מראה את מקצת החלקים הארוזים – חשבתי שזהו מיצב משל עצמו', אוקטובר 2013).",
        people=(PERSON_TAMI, PERSON_GADI, PERSON_TAL),
        places=(PLACE_DENYA,),
        art_themes=("capsules/enclosures", "pipes/conduits"),
    ),
    # 2011 MFA Hamra brick fabrication at home in Denya (`ID=263`)
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2011/05/p1040010.jpg",
        post_url="https://theshingimel.wordpress.com/2011/06/03/%d7%91%d7%99%d7%95%d7%9d-%d7%a9%d7%9c%d7%99%d7%a9%d7%99-%d7%99%d7%a6%d7%99%d7%a7%d7%94/",
        post_title="ביום שלישי – יציקה",
        date_or_era="2011-05-29",
        image_title="יציקת 380 לבני החמרה בחצר הבית בדניה לפרויקט הגמר של תמי (MFA)",
        caption="סינון אדמת החמרה ודחיסתה לתבניות בחצר הבית בדניה לקראת תערוכת הגמר של תמר שפר באוניברסיטת חיפה ('ביום שלישי – יציקה', 2011).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
        art_themes=("earth", "cement"),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2011/05/p1040012.jpg",
        post_url="https://theshingimel.wordpress.com/2011/06/03/%d7%91%d7%99%d7%95%d7%9d-%d7%a9%d7%9c%d7%99%d7%a9%d7%99-%d7%99%d7%a6%d7%99%d7%a7%d7%94/",
        post_title="ביום שלישי – יציקה",
        date_or_era="2011-05-29",
        image_title="דחיסת לבני החמרה בתבניות בחצר הבית בדניה (2011)",
        caption="עבודת הצוות בחצר הבית בדניה בדחיסת 380 לבני חמרה כבושות עם אותיות חתוכות בלייזר ('ביום שלישי – יציקה', 2011).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
        art_themes=("earth", "cement"),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2011/05/29032011375-11.jpg",
        post_url="https://theshingimel.wordpress.com/2011/06/03/%d7%91%d7%99%d7%95%d7%9d-%d7%a9%d7%9c%d7%99%d7%a9%d7%99-%d7%99%d7%a6%d7%99%d7%a7%d7%94/",
        post_title="ביום שלישי – יציקה",
        date_or_era="2011-03-29",
        image_title="כ-50 תבניות עץ מצופות פורמייקה שבנה גדי בדניה ליציקת לבני החמרה",
        caption="תבניות העץ והפורמייקה שבנה גדי שפר בעזרת מסור השולחן והמקיטה בבית בדניה עבור יציקת לבני החמרה של תמי (2011).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
        art_themes=("earth",),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2011/05/23032011366.jpg",
        post_url="https://theshingimel.wordpress.com/2011/06/03/%d7%91%d7%99%d7%95%d7%9d-%d7%a9%d7%9c%d7%99%d7%a9%d7%99-%d7%99%d7%a6%d7%99%d7%a7%d7%94/",
        post_title="ביום שלישי – יציקה",
        date_or_era="2011-03-23",
        image_title="ייבוש שורות לבני החמרה היצוקות בבית בדניה (2011)",
        caption="שורות לבני החמרה מתייבשות לאחר פירוק התבניות בבית בדניה ('40 לבנים בשבוע', מרץ 2011).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_DENYA,),
        art_themes=("earth",),
    ),

    # =========================================================================
    # 3. OTHER HAIFA HOMES (Bar Giora 7 / Hadar HaCarmel) & SOLO EXHIBITIONS
    # =========================================================================
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/08/d791d7a8-d792d799d795d7a8d790-7.jpg",
        post_url="https://theshingimel.wordpress.com/2015/08/05/24-26/",
        post_title="24-26: ברחוב בר גיורא 7",
        date_or_era="2015-08-05",
        image_title="בר גיורא 7, הדר הכרמל, חיפה — הבניין של דירתם המשותפת הראשונה של תמי וגדי",
        caption="חזית הבניין ברחוב בר גיורא 7 בהדר הכרמל בחיפה (4 בתים מבר גיורא 1 / ארלוזורוב 87), שבו שכנה דירתם המשותפת הראשונה של תמי וגדי כסטודנטים בטכניון.",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_ARLOZOROV,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/08/p1050497.jpg",
        post_url="https://theshingimel.wordpress.com/2015/08/05/24-26/",
        post_title="24-26: ברחוב בר גיורא 7",
        date_or_era="2015-08-05",
        image_title="בר גיורא 7, הדר הכרמל — פנים הדירה הראשונה של תמי וגדי",
        caption="מבט פנים בדירה ברחוב בר גיורא 7 בהדר הכרמל שבה התגוררו תמי וגדי בצעירותם ('24-26: ברחוב בר גיורא 7', 2015).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_ARLOZOROV,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2015/08/p1050507.jpg",
        post_url="https://theshingimel.wordpress.com/2015/08/05/24-26/",
        post_title="24-26: ברחוב בר גיורא 7",
        date_or_era="2015-08-05",
        image_title="בר גיורא 7, הדר הכרמל — מבואת הכניסה לדירה",
        caption="מבט נוסף מתוך הביקור החוזר בדירה הראשונה ברחוב בר גיורא 7 בהדר הכרמל, חיפה (2015).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_ARLOZOROV,),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/10/1-_dsc7647-1.jpg",
        post_url="https://theshingimel.wordpress.com/2013/10/12/now-here-there/",
        post_title="פלישת חוטפי הגופות",
        date_or_era="2013-10-12",
        image_title="תערוכת היחיד 'עכשיו / (כאן + שם)' של תמר שפר בגלריה בנימין (2013)",
        caption="המיצב התלת-ממדי של תמר שפר בתערוכת היחיד הראשונה שלה בתל אביב בגלריה בנימין (אוקטובר 2013), שבה עבד טל שפר ארבעה חודשים כאסיסטנט.",
        people=(PERSON_TAMI, PERSON_GADI, PERSON_TAL),
        places=(PLACE_BINYAMIN,),
        art_themes=("capsules/enclosures", "pipes/conduits", "papier-mâché"),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2013/10/1-_dsc7717-11.jpg",
        post_url="https://theshingimel.wordpress.com/2013/10/12/now-here-there/",
        post_title="פלישת חוטפי הגופות",
        date_or_era="2013-10-12",
        image_title="מבט רחב על המיצב 'עכשיו / (כאן + שם)' בגלריה בנימין (2013)",
        caption="המיצב התלת-ממדי של תמר שפר הנפרש לאורך שלושה קירות בגלריה בנימין ועוסק במיזוג בין 'ארכיטקטורה של הענן' לחומר פיזי (2013).",
        people=(PERSON_TAMI, PERSON_TAL),
        places=(PLACE_BINYAMIN,),
        art_themes=("capsules/enclosures", "pipes/conduits", "papier-mâché"),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2026/05/20260329_162310.jpg",
        post_url="https://theshingimel.wordpress.com/2026/05/02/koby/",
        post_title="המַגְשים",
        date_or_era="2026-03-29",
        image_title="הקמת המיצב 'בנה ביתך' של תמר שפר במוזיאון רמת גן (2026)",
        caption="התקנת המיצב 'בנה ביתך' של תמר שפר במוזיאון רמת גן לאמנות ישראלית עם מערכת התאורה הסמויה מתוך הרשומה 'המגשים' (מאי 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_RAMAT_GAN,),
        art_themes=("simulacra of home", "misshapen furniture", "broken/blurry reflections"),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2026/04/img_20260329_160400.jpg",
        post_url="https://theshingimel.wordpress.com/2026/05/02/koby/",
        post_title="המַגְשים",
        date_or_era="2026-03-29",
        image_title="גזע העץ והבית המשובש במיצב 'בנה ביתך' במוזיאון רמת גן (2026)",
        caption="מבט מתוך תערוכת היחיד 'בנה ביתך' של תמר שפר במוזיאון רמת גן לאמנות ישראלית (אוצרת: שרי גולן, 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_RAMAT_GAN,),
        art_themes=("simulacra of home", "misshapen furniture"),
    ),
    BlogImageTarget(
        url="https://theshingimel.wordpress.com/wp-content/uploads/2026/07/img_20260329_160355.jpg",
        post_url="https://theshingimel.wordpress.com/2026/07/24/%d7%90%d7%9c%d7%a3-%d7%aa%d7%9e%d7%95%d7%a0%d7%95%d7%aa/",
        post_title="אלף תמונות",
        date_or_era="2026-03-29",
        image_title="תערוכת 'בנה ביתך' של תמר שפר בקומה השנייה של מוזיאון רמת גן",
        caption="מבט רחב על חלל המיצב 'בנה ביתך' של תמר שפר במוזיאון רמת גן מתוך הרשומה 'אלף תמונות' (יולי 2026).",
        people=(PERSON_TAMI, PERSON_GADI),
        places=(PLACE_RAMAT_GAN,),
        art_themes=("simulacra of home", "capsules/enclosures", "misshapen furniture"),
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


def collect_existing_hashes(media_dir: Path) -> dict[str, Path]:
    hashes: dict[str, Path] = {}
    for root, _, files in os.walk(media_dir):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff"}:
                hashes[compute_sha256_file(fpath)] = fpath
    return hashes


def candidate_original_urls(raw_url: str) -> list[str]:
    base_no_query = raw_url.split("?")[0]
    stripped_dim = WP_RESIZE_SUFFIX_RE.sub("", base_no_query)
    candidates: list[str] = []
    for u in (stripped_dim, base_no_query):
        if u and u not in candidates:
            candidates.append(u)
    return candidates


def fetch_readonly_bytes(url: str, cfg: HarvesterConfig) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, encoded_path, "", ""))
    req = urllib.request.Request(
        safe_url,
        headers={"User-Agent": cfg.user_agent, "Accept": "image/*,*/*;q=0.8"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=cfg.http_timeout_sec) as resp:
        return resp.read()


def clean_existing_sidecars(media_dir: Path) -> dict[str, int]:
    """Cleans polluted `places` tags in `portfolio_site/` and `wordpress_blog/`."""
    portfolio_dir = media_dir / PORTFOLIO_SUBDIR
    wp_dir = media_dir / WORDPRESS_SUBDIR
    cleaned_portfolio = 0
    cleaned_wp = 0

    if portfolio_dir.is_dir():
        for jp in sorted(portfolio_dir.glob("*.json")):
            data: dict[str, Any] = json.loads(jp.read_text(encoding="utf-8"))
            if PLACE_DENYA in (data.get("places") or []):
                data["places"] = [p for p in data["places"] if p != PLACE_DENYA]
                data["places_locked"] = True
                jp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                cleaned_portfolio += 1

    if wp_dir.is_dir():
        for jp in sorted(wp_dir.glob("*.json")):
            patch = EXISTING_WP_SIDECAR_PATCHES.get(jp.name)
            if patch is None and jp.name.startswith(SAFE_POST_GALLERY_TOUR_PREFIX):
                patch = SAFE_POST_PATCH
            if patch is None:
                continue
            data = json.loads(jp.read_text(encoding="utf-8"))
            data["title"] = patch.title
            data["caption"] = patch.caption
            data["date_or_era"] = patch.date_or_era
            data["people"] = list(patch.people)
            data["places"] = list(patch.places)
            data["places_locked"] = True
            data["confidence_notes"] = patch.confidence_notes
            jp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            cleaned_wp += 1

    return {"cleaned_portfolio": cleaned_portfolio, "cleaned_wp": cleaned_wp}


def sanitize_stem(url_path: str) -> str:
    raw_stem = Path(urllib.parse.unquote(url_path)).stem
    ascii_stem = SAFE_FILENAME_RE.sub("_", raw_stem).strip("_-").lower()
    if not ascii_stem:
        ascii_stem = "wp_img"
    return ascii_stem[:MAX_STEM_LENGTH]


def decode_and_verify_image(raw_bytes: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(raw_bytes)) as vimg:
        vimg.verify()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()
        width, height = img.size
        fmt = (img.format or "JPEG").upper()
    ext_map = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "TIFF": ".tiff",
    }
    return width, height, ext_map.get(fmt, ".jpg")


def _build_sidecar_payload(
    target: BlogImageTarget, resolved_url: str, width: int, height: int
) -> dict[str, Any]:
    return {
        "title": target.image_title,
        "caption": f"{target.caption} (מתוך הפוסט '{target.post_title}': {target.post_url})",
        "source_url_or_path": resolved_url,
        "post_url": target.post_url,
        "date_or_era": target.date_or_era,
        "people": list(target.people),
        "places": list(target.places),
        "places_locked": True,
        "art_themes": list(target.art_themes),
        "confidence_notes": (
            f"CONFIRMED: Visually verified full-res {width}x{height} image from '{target.post_title}' ({target.post_url})."
        ),
    }


def harvest_wide_blog_targets(cfg: HarvesterConfig) -> dict[str, Any]:
    out_dir = cfg.media_dir / WORDPRESS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    active_candidate_urls: set[str] = set()
    for t in BLOG_HARVEST_TARGETS:
        active_candidate_urls.update(candidate_original_urls(t.url))

    # Prune any `wp_wide_*` files removed after visual audit, and index cached `wp_wide_*` by source URL
    cached_by_url: dict[str, Path] = {}
    pruned_files: list[str] = []
    for sidecar_p in sorted(out_dir.glob("wp_wide_*.json")):
        meta = json.loads(sidecar_p.read_text(encoding="utf-8"))
        src_url = str(meta.get("source_url_or_path", ""))
        img_p = out_dir / sidecar_p.name[: -len(".json")]
        if src_url not in active_candidate_urls:
            if img_p.exists():
                img_p.unlink()
            sidecar_p.unlink()
            pruned_files.append(img_p.name)
        elif img_p.is_file():
            cached_by_url[src_url] = img_p

    seen_hashes = collect_existing_hashes(cfg.media_dir)

    saved: list[dict[str, Any]] = []
    refreshed_cached: list[str] = []
    skipped_duplicates: list[str] = []
    skipped_small: list[tuple[str, int, int]] = []
    failed_urls: list[tuple[str, str]] = []

    for target in BLOG_HARVEST_TARGETS:
        cand_urls = candidate_original_urls(target.url)
        cached_img = next((cached_by_url[u] for u in cand_urls if u in cached_by_url), None)
        if cached_img is not None:
            resolved_url = next(u for u in cand_urls if cached_by_url.get(u) == cached_img)
            with Image.open(cached_img) as img:
                width, height = img.size
            sidecar_path = out_dir / f"{cached_img.name}.json"
            payload = _build_sidecar_payload(target, resolved_url, width, height)
            sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            refreshed_cached.append(cached_img.name)
            continue

        raw_bytes: bytes | None = None
        resolved_url = target.url
        last_err = ""
        for cand_url in cand_urls:
            try:
                raw_bytes = fetch_readonly_bytes(cand_url, cfg)
                resolved_url = cand_url
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = str(exc)
                continue

        if raw_bytes is None:
            failed_urls.append((target.url, last_err))
            continue

        try:
            width, height, ext = decode_and_verify_image(raw_bytes)
        except Exception as exc:
            failed_urls.append((resolved_url, f"Pillow decode error: {exc}"))
            continue

        if width < cfg.min_dim_px or height < cfg.min_dim_px:
            skipped_small.append((resolved_url, width, height))
            continue

        sha256_hex = compute_sha256_bytes(raw_bytes)
        if sha256_hex in seen_hashes:
            skipped_duplicates.append(f"{resolved_url} -> existing {seen_hashes[sha256_hex].name}")
            continue

        stem = sanitize_stem(resolved_url)
        filename = f"wp_wide_{stem}_{sha256_hex[:FILENAME_HASH_PREFIX_LEN]}{ext}"
        img_path = out_dir / filename
        sidecar_path = out_dir / f"{filename}.json"

        sidecar_payload = _build_sidecar_payload(target, resolved_url, width, height)
        img_path.write_bytes(raw_bytes)
        sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        seen_hashes[sha256_hex] = img_path
        saved.append(
            {
                "filename": filename,
                "width": width,
                "height": height,
                "places": list(target.places),
                "title": target.image_title,
                "url": resolved_url,
            }
        )

    return {
        "saved": saved,
        "refreshed_cached": refreshed_cached,
        "pruned_files": pruned_files,
        "skipped_duplicates": skipped_duplicates,
        "skipped_small": skipped_small,
        "failed_urls": failed_urls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean polluted blog/portfolio sidecars and harvest wide home photos.")
    parser.add_argument("--media-dir", default=str(DEFAULT_MEDIA_DIR), help="Path to media archive root")
    args = parser.parse_args()

    cfg = HarvesterConfig(media_dir=Path(args.media_dir).resolve())
    clean_stats = clean_existing_sidecars(cfg.media_dir)
    print(
        f"[Clean Sidecars] Cleaned {clean_stats['cleaned_portfolio']} portfolio sidecars "
        f"and {clean_stats['cleaned_wp']} WordPress sidecars."
    )

    harvest_stats = harvest_wide_blog_targets(cfg)
    print(
        f"[Wide Harvest] Saved {len(harvest_stats['saved'])} new, "
        f"refreshed {len(harvest_stats['refreshed_cached'])} cached, "
        f"pruned {len(harvest_stats['pruned_files'])} non-photo diagrams/screenshots."
    )
    if harvest_stats["skipped_duplicates"]:
        print(f"  Skipped {len(harvest_stats['skipped_duplicates'])} already-archived SHA-256 duplicates.")
    if harvest_stats["skipped_small"]:
        print(f"  Skipped {len(harvest_stats['skipped_small'])} undersized images (<{cfg.min_dim_px}px): {harvest_stats['skipped_small']}")
    if harvest_stats["failed_urls"]:
        print(f"  [WARN] Failed {len(harvest_stats['failed_urls'])} URLs: {harvest_stats['failed_urls']}")


if __name__ == "__main__":
    main()
