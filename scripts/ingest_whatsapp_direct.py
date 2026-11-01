#!/usr/bin/env python3
"""
scripts/ingest_whatsapp_direct.py — Read-Only Direct WhatsApp Photo Ingestion Pipeline

Reads selected childhood, family, and costume photographs sent directly to Tal Sheaffer
via WhatsApp from Ada Wertheimer and Gad Sheaffer ("אבא"), validates image integrity,
deduplicates via SHA-256, and ingests them into `media/whatsapp_family/` with rich
descriptive Hebrew/English metadata JSON sidecars and upright derivative rotation flags.

Strict Read-Only Safety:
- Accesses local WhatsApp storage strictly read-only (?mode=ro or filesystem read).
- Never mutates external files or databases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHATSAPP_STORAGE_ROOT = Path(
    "/Users/talsheaffer/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/Message"
)
MEDIA_TARGET_DIR = PROJECT_ROOT / "media" / "whatsapp_family"
DROP_FOLDER_INCOMING = PROJECT_ROOT / "drop_folder" / "incoming"

PERSON_TAMI = "Tamar 'Tami' Sheaffer"
PERSON_GADI = "Gad Sheaffer"
PERSON_ADA = "Ada Wertheimer"
PERSON_ERELA = "Er'ela Sadan"
PERSON_AHUVA = "Ahuva Fried"

PLACE_HERZLIYA = "Herzliya (parents' house & yard)"


@dataclass(frozen=True)
class CuratedDirectWhatsAppPhoto:
    source_rel_path: str
    target_filename: str
    title: str
    caption: str
    date_or_era: str
    people: tuple[str, ...]
    places: tuple[str, ...]
    display_rotation_cw_deg: int = 0
    art_themes: tuple[str, ...] = ()
    confidence_notes: str = ""
    source_url_or_path: str = ""

    def to_sidecar_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "caption": self.caption,
            "source_url_or_path": self.source_url_or_path,
            "date_or_era": self.date_or_era,
            "people": list(self.people),
            "people_locked": True,
            "places": list(self.places),
            "places_locked": True,
            "art_themes": list(self.art_themes),
            "art_themes_locked": True,
            "display_rotation_cw_deg": self.display_rotation_cw_deg,
            "confidence_notes": self.confidence_notes,
        }


CURATED_DIRECT_PHOTOS: tuple[CuratedDirectWhatsAppPhoto, ...] = (
    # --- Ada Wertheimer (WhatsApp Chat 168641974784033@lid) ---
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/0/e/0e15321d-78d2-4020-8382-fdd93e2bf256.jpg",
        target_filename="wa_ada_wertheimer_1957_family_table_0e15321d.jpg",
        title="ארוחה משפחתית סביב השולחן בחצר (1957)",
        caption="תצלום משפחתי בשחור-לבן משנת 1957 ששלחה עדה ורטהיימר. מבוגרים ופעוטות מסביב לשולחן ערוך בחצר בהרצליה; משמאל אב מחזיק תינוקת, במרכז ילדה, ומימין אישה מחזיקה פעוטה (תמי הפעוטה בחיק המבוגרים). כיתוב מקורי מעדה: \"1957\".",
        date_or_era="1957",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Archival print photographed and captioned '1957' by childhood friend Ada Wertheimer via WhatsApp.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50210#0e15321d",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/5/3/530bc045-bc41-4c4b-9b34-cfc6703c5d8b.jpg",
        target_filename="wa_ada_wertheimer_1958_tami_standing_530bc045.jpg",
        title="תמי הפעוטה עומדת בחדר (1958)",
        caption="דיוקן פעוטה מתוק של תמי שפר משנת 1958, עומדת ואוחזת בסורגי לול/מיטה בסוודר בהיר, עם שיער חלק קצר וללא תלתלים כפי שציינה עדה (\"זאת הקטנטונת בלי התלתלים\"). כיתוב מקורי מעדה: \"1958\".",
        date_or_era="1958",
        people=(PERSON_TAMI,),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=270,
        confidence_notes="CONFIRMED: Archival print photographed and captioned '1958' by Ada Wertheimer via WhatsApp; upright derivative rotated 270 deg CW.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50215#530bc045",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/b/a/ba34d1e3-2554-4faf-a49d-ebdbf5e35701.jpg",
        target_filename="wa_ada_wertheimer_1958_tami_ada_bench_hats_ba34d1e3.jpg",
        title="תמי ועדה בכובעי שמש לבנים על ספסל בחצר (1958)",
        caption="תמי שפר ועדה ורטהיימר כפעוטות בשנת 1958, יושבות יחד מחויכות על ספסל עץ בחצר הבית בהרצליה, שתיהן חובשות כובעי שמש בהירים. נשלח ע\"י עדה ורטהיימר בוואטסאפ עם ההערה \"גם אלה מ-1958\".",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=270,
        confidence_notes="CONFIRMED: Archival print from 1958 sent by Ada Wertheimer via WhatsApp; upright derivative rotated 270 deg CW.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50211#ba34d1e3",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/f/3/f3922aa2-b0e1-4dd3-a64a-0c073c7ffe35.jpg",
        target_filename="wa_ada_wertheimer_1958_tami_ahuva_outdoor_f3922aa2.jpg",
        title="תמי הפעוטה בחיק אמה אהובה וקרובת משפחה בחצר (1958)",
        caption="תמי הפעוטה בכובע שמש לבן נישאת בחיק אמה אהובה פריד ובת משפחה על רקע קיר האבן והחצר בהרצליה בשנת 1958. נשלח ע\"י עדה ורטהיימר בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_AHUVA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=270,
        confidence_notes="CONFIRMED: Archival 1958 print sent by Ada Wertheimer via WhatsApp; upright derivative rotated 270 deg CW.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50212#f3922aa2",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/6/b/6b7d6bab-f38c-45d8-8d2b-cde7ea3034fa.jpg",
        target_filename="wa_ada_wertheimer_1958_tami_ada_sand_bucket_6b7d6bab.jpg",
        title="תמי ועדה משחקות בחול בחצר בהרצליה (1958)",
        caption="תמי (בשיער חלק, יושבת עם דלי משחק ביד) ועדה (משמאל עם תלתלים, יושבת על דרגש קטן) משחקות בחול בחצר הבית וגן הילדים של אהובה פריד בהרצליה בשנת 1958. נשלח ע\"י עדה ורטהיימר בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Archival print from 1958 sent by Ada Wertheimer via WhatsApp.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50213#6b7d6bab",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/1/6/16442124-1b25-45aa-a143-301726d8aad2.jpg",
        target_filename="wa_ada_wertheimer_1958_tami_ada_hats_closeup_16442124.jpg",
        title="תקריב תמי ועדה בכובעי שמש לבנים בחצר (1958)",
        caption="תצלום תקריב אינטימי של תמי (משמאל, בכובע בד לבן) ועדה (מימין, מביטה בה) בחצר בהרצליה בשנת 1958. נשלח ע\"י עדה ורטהיימר בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=270,
        confidence_notes="CONFIRMED: Archival 1958 print sent by Ada Wertheimer via WhatsApp; upright derivative rotated 270 deg CW.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50214#16442124",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/1/6/16571624-c079-4bf4-94a0-028365e6ac1b.jpg",
        target_filename="wa_ada_wertheimer_ca1966_tami_ada_purim_standing_16571624.jpg",
        title="תמי ועדה בתחפושת עומדות בטבע (סביבות 1966)",
        caption="תמי שפר ועדה ורטהיימר בסביבות גיל 10 בתחפושת (רועים / פורים) עם כובעי ברט, שכמיות, שמלות וסלסילת קש בטבע. כיתוב מקורי מעדה בוואטסאפ: \"בלי תאריך, משהו כמו גיל 10?\".",
        date_or_era="ca. 1966",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Archival print sent by Ada Wertheimer via WhatsApp, estimated ca. 1966 (age 10).",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50217#16571624",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/168641974784033@lid/9/2/925f457b-1307-4825-9700-211ff4aba0bf.jpg",
        target_filename="wa_ada_wertheimer_ca1966_tami_ada_purim_kneeling_925f457b.jpg",
        title="תמי ועדה כורעות בתחפושת עם סלסילה (סביבות 1966 - צילום עדה)",
        caption="תמי שפר ועדה ורטהיימר כורעות על הדשא בתחפושת עם סלסילת קש בסביבות גיל 10 (1966 בקירוב). תצלום שצילמה עדה ורטהיימר מההדפס המקורי ושלחה בוואטסאפ.",
        date_or_era="ca. 1966",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=270,
        confidence_notes="CONFIRMED: Archival print sent by Ada Wertheimer via WhatsApp; upright derivative rotated 270 deg CW.",
        source_url_or_path="whatsapp://chat/168641974784033@lid/msg/50218#925f457b",
    ),

    # --- Aba / Gad Sheaffer (WhatsApp Chat 972547885114@s.whatsapp.net) ---
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/4/8/486fb437-3ccb-42cb-9838-fa3b56606de5.jpg",
        target_filename="wa_gadi_childhood_1958_tami_ada_garden_486fb437.jpg",
        title="תמי הפעוטה ועדה עובדות בגינה בהרצליה (1958)",
        caption="תמי שפר (מימין, בסוודר בהיר ומכנסיים עם כלי גינון קטן) ועדה ורטהיימר (משמאל, בשמלת סרפן כהה ותלתלים) בגינה על רקע חלונות הבית בהרצליה. צולם מהדפס מקורי עם שולי זיגזג ע\"י גדי שפר (אבא) ונשלח בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Original physical photograph from 1958 preserved in Sheaffer family album, photographed and sent by Gad Sheaffer.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50185#486fb437",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/b/5/b5f7c67c-f196-4f59-bbda-3cb064888c95.jpg",
        target_filename="wa_gadi_childhood_1958_tami_ada_closeup_b5f7c67c.jpg",
        title="תקריב תמי ועדה עם כובע שמש בחצר (1958)",
        caption="תצלום תקריב בהיר וצלול של תמי (משמאל, בכובע בד וסרפן) ועדה (מימין, בתלתלים שופעים מחזיקה כובע שמש) בחצר בהרצליה בקיץ 1958. צולם מהדפס מקורי ע\"י גדי שפר ונשלח בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Original physical print photographed by Gad Sheaffer via WhatsApp.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50186#b5f7c67c",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/7/e/7ef90cdd-c958-47c2-8b57-9b097f1d92a2.jpg",
        target_filename="wa_gadi_childhood_1958_tami_ada_garden_var_7ef90cdd.jpg",
        title="תמי ועדה בגינה ליד חלונות הבית בהרצליה (גרסה שנייה, 1958)",
        caption="גרסת הדפס נוספת של תמי ועדה עובדות בגינה בהרצליה בשנת 1958, במסגור רחב יותר החושף את שולי הנייר המזוגזגים והצל של הבנות. נשלח ע\"י גדי שפר בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Alternate physical print of 1958 garden scene photographed by Gad Sheaffer.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50187#7ef90cdd",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/8/8/886ba0d8-906b-47ac-95cd-e170f9715636.jpg",
        target_filename="wa_gadi_childhood_1958_tami_bed_singer_machine_886ba0d8.jpg",
        title="תמי הפעוטה במיטה לצד אחותה/אמה ומכונת תפירה זינגר (1958)",
        caption="תצלום פנים חם ומרגש: תמי הפעוטה יחפה בפיג'מה יושבת על מיטה ליד אחותה אראלה / אמה אהובה השוכבת, וליד המיטה ניצבת מכונת תפירה עתיקה מתוצרת Singer עם דוושת ברזל. צולם מהדפס מקורי ע\"י גדי שפר ונשלח בוואטסאפ.",
        date_or_era="1958",
        people=(PERSON_TAMI, PERSON_ERELA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Rare interior print of Fried family home with vintage Singer sewing machine, photographed by Gad Sheaffer.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50188#886ba0d8",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/a/6/a60f7a0e-04a4-4e7f-8f49-ad3741eb35b7.jpg",
        target_filename="wa_gadi_childhood_ca1960_four_girls_stairs_a60f7a0e.jpg",
        title="ארבע הבנות על מדרגות האבן בחצר בהרצליה (סביבות 1960)",
        caption="ארבע הבנות יושבות יחד על מדרגות אבן בחצר הבית בהרצליה בסביבות 1959-1960: שתי בנות בוגרות למעלה (כולל אראלה סדן ועדה ורטהיימר), ושתי פעוטות למטה. גדי שפר ציין בהודעה: \"פה ארבעתן / תמי ימין למטה\".",
        date_or_era="ca. 1960",
        people=(PERSON_TAMI, PERSON_ERELA, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Four girls on stone stairs, with Tami bottom-right confirmed directly by Gad Sheaffer on WhatsApp.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50189#a60f7a0e",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/e/5/e5d88035-05f2-4582-8912-70acebaea3d4.jpg",
        target_filename="wa_gadi_childhood_ca1966_tami_ada_car_purim_e5d88035.jpg",
        title="תמי ועדה בתחפושות פורים ליד מכונית (סביבות 1966)",
        caption="תמי שפר ועדה ורטהיימר בפורים בסביבות גיל 10 ליד מכונית ישנה. תמי משמאל מחופשת לאישה מבוגרת עם כובע, משקפיים, שכמית תחרה ותיק יד; עדה מימין בתלבושת מסורתית. גדי שפר ציין: \"בוקר טוב. מזהה מי זו תמי? ... עם הכובע? כן\".",
        date_or_era="ca. 1966",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: Purim costume photograph confirmed by Gad Sheaffer ('Tami with the hat') and Tal Sheaffer via WhatsApp.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50190#e5d88035",
    ),
    CuratedDirectWhatsAppPhoto(
        source_rel_path="Media/9985983512751@lid/c/4/c4a60365-4d0b-4c41-9e81-0ddd029a8e80.jpg",
        target_filename="wa_gadi_childhood_ca1966_tami_ada_purim_kneeling_c4a60365.jpg",
        title="תמי ועדה כורעות בתחפושת עם סלסילת קש (סביבות 1966 - צילום גדי)",
        caption="הדפס מקורי חד ואיכותי של תמי שפר ועדה ורטהיימר כורעות בטבע בתחפושת רועים/פורים עם סלסילת קש בסביבות גיל 10 (1966 בקירוב). צולם מההדפס המשפחתי ע\"י גדי שפר ונשלח בוואטסאפ.",
        date_or_era="ca. 1966",
        people=(PERSON_TAMI, PERSON_ADA),
        places=(PLACE_HERZLIYA,),
        display_rotation_cw_deg=0,
        confidence_notes="CONFIRMED: High-resolution physical print captured by Gad Sheaffer.",
        source_url_or_path="whatsapp://chat/972547885114@s.whatsapp.net/msg/50191#c4a60365",
    ),
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def run_ingestion(dry_run: bool = False) -> int:
    print(f"=== Running WhatsApp Direct Photo Ingestion (dry_run={dry_run}) ===")
    MEDIA_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    DROP_FOLDER_INCOMING.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, str] = {}
    ingested_count = 0

    for item in CURATED_DIRECT_PHOTOS:
        source_path = WHATSAPP_STORAGE_ROOT / item.source_rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"WhatsApp source image not found: {source_path}")

        # Validate non-corruption and dimensions
        with Image.open(source_path) as img:
            img.verify()
        with Image.open(source_path) as img:
            img.load()
            w, h = img.size

        if w < 150 or h < 150:
            raise ValueError(f"Image {source_path} below 150px threshold ({w}x{h})")

        sha256 = compute_sha256(source_path)
        if sha256 in seen_hashes:
            raise ValueError(
                f"SHA-256 collision detected: {item.target_filename} matches {seen_hashes[sha256]}"
            )
        seen_hashes[sha256] = item.target_filename

        target_img_path = MEDIA_TARGET_DIR / item.target_filename
        target_sidecar_path = MEDIA_TARGET_DIR / f"{item.target_filename}.json"
        drop_incoming_path = DROP_FOLDER_INCOMING / item.target_filename

        print(
            f"  [+] Ingesting: {item.target_filename} ({w}x{h}, {sha256[:12]}..., rot={item.display_rotation_cw_deg})"
        )
        print(f"      Title: {item.title}")
        print(f"      Era:   {item.date_or_era}")
        print(f"      People: {', '.join(item.people)}")

        if not dry_run:
            # 1. Copy to media/whatsapp_family/
            shutil.copy2(source_path, target_img_path)
            # 2. Write JSON sidecar
            sidecar_data = item.to_sidecar_dict()
            with open(target_sidecar_path, "w", encoding="utf-8") as sf:
                json.dump(sidecar_data, sf, ensure_ascii=False, indent=2)
            # 3. Copy to drop_folder/incoming/
            shutil.copy2(source_path, drop_incoming_path)

        ingested_count += 1

    print(f"\nSuccessfully validated and staged {ingested_count} archival WhatsApp photos.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest direct WhatsApp photos into catalog.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files.")
    args = parser.parse_args()
    sys.exit(run_ingestion(dry_run=args.dry_run))
