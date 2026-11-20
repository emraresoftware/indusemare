#!/usr/bin/env python3
"""Render 1200x630 FaunaTech Doğal Atlas v2.0 OpenGraph / Twitter social preview card with real Indus TTF glyphs."""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "portal" / "assets" / "og-indusemare.png"
ART_DIR = Path(os.environ["INDUS_ARTIFACT_DIR"]) if os.environ.get("INDUS_ARTIFACT_DIR") else None


def main() -> None:
    glyphs = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")
    cp_map = {}
    for _, r in glyphs.iterrows():
        sid = str(r["sign_id"]).strip().zfill(3)
        cps = [int(h, 16) for h in str(r["codepoints"]).split() if h]
        if cps:
            cp_map[sid] = "".join(chr(c) for c in cps)

    indus_xl = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 52)
    indus_md = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 36)
    try:
        f_serif_lg = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 44)
        f_serif_md = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", 22)
        f_sans_b = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        f_sans = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        f_mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
        f_mono_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except Exception:
        f_serif_lg = f_serif_md = f_sans_b = f_sans = f_mono = f_mono_sm = ImageFont.load_default()

    W, H = 1200, 630
    img = Image.new("RGB", (W, H), "#132a1f")  # Deep Pine Seal canvas
    draw = ImageDraw.Draw(img)

    # Subtle layered botanical panels (zero neon, FaunaTech Doğal Atlas v2.0)
    draw.rectangle([(0, 0), (W, 10)], fill="#d7efaf")  # Taze Filiz top accent line
    draw.rounded_rectangle([(36, 36), (W - 36, H - 36)], radius=24, fill="#183427", outline="#2d5440", width=2)

    # Top eyebrow pill
    draw.rounded_rectangle([(68, 64), (612, 98)], radius=17, fill="#224534", outline="#4c7a5f", width=1)
    draw.text((86, 72), "INDUS-DEDUCT v1.0  |  OPEN BENCHMARK  |  11/11 VERIFIED", fill="#d7efaf", font=f_mono)

    # Domain top right
    draw.text((926, 72), "indus.emare.dev", fill="#d7efaf", font=f_sans_b)

    # Main Title
    draw.text((68, 122), "Calibrated Computational Epigraphy", fill="#f6f7f3", font=f_serif_lg)
    draw.text((68, 174), "for the Indus Valley Script", fill="#d7efaf", font=f_serif_lg)
    draw.text((68, 236), "5,679 inscriptions · Zero fabricated readings · 5-way calibration vs Sumerian & Linear B", fill="#b6c9bc", font=f_serif_md)

    # 3 Light-paper ("Sabah Keteni / Duru Kâğıt") highlight cards inside the deep pine seal
    card_y = 294
    card_h = 224
    cards_x = [(68, 412), (428, 772), (788, 1132)]

    # Card 1: 5-Way Calibration
    x0, x1 = cards_x[0]
    draw.rounded_rectangle([(x0, card_y), (x1, card_y + card_h)], radius=16, fill="#f6f7f3", outline="#dfe5dc", width=1)
    draw.text((x0 + 20, card_y + 18), "1. 5-WAY CALIBRATION", fill="#315b41", font=f_mono)
    draw.text((x0 + 20, card_y + 48), "z_H1 entropy RETIRED (-245s neg.)", fill="#8a4b35", font=f_sans_b)
    draw.text((x0 + 20, card_y + 82), "Validated discriminators:", fill="#43554a", font=f_sans)
    draw.text((x0 + 20, card_y + 110), "Kober MI:  +53.7s  (vs +0.07s)", fill="#182c22", font=f_mono)
    draw.text((x0 + 20, card_y + 136), "Unseen-3g: +6.38pp (vs -2.95pp)", fill="#182c22", font=f_mono)
    draw.text((x0 + 20, card_y + 162), "Affix 2x:  207:1   (vs 0:0)", fill="#182c22", font=f_mono)
    draw.text((x0 + 20, card_y + 190), "23-Seal CSP: k=2 anchors needed", fill="#68766b", font=f_mono_sm)

    # Card 2: 700-Year Paleographic Evolution (LIVE REAL GLYPHS!)
    x0, x1 = cards_x[1]
    draw.rounded_rectangle([(x0, card_y), (x1, card_y + card_h)], radius=16, fill="#f6f7f3", outline="#dfe5dc", width=1)
    draw.text((x0 + 20, card_y + 18), "2. 700-YEAR GLYPH DRIFT (FDR*)", fill="#315b41", font=f_mono)
    # Row A: 838 -> 803 -> 806
    draw.text((x0 + 30, card_y + 46), cp_map.get("838", "?"), fill="#182c22", font=indus_xl)
    draw.text((x0 + 104, card_y + 68), "->", fill="#8a4b35", font=f_sans_b)
    draw.text((x0 + 144, card_y + 46), cp_map.get("803", "?"), fill="#182c22", font=indus_xl)
    draw.text((x0 + 218, card_y + 68), "->", fill="#8a4b35", font=f_sans_b)
    draw.text((x0 + 258, card_y + 46), cp_map.get("806", "?"), fill="#315b41", font=indus_xl)
    draw.text((x0 + 24, card_y + 114), "#838 -> #803 -> #806 (q=0.034*)", fill="#43554a", font=f_mono_sm)
    # Row B: 156 -> 154
    draw.text((x0 + 68, card_y + 136), cp_map.get("156", "?"), fill="#182c22", font=indus_md)
    draw.text((x0 + 146, card_y + 150), "->", fill="#8a4b35", font=f_sans_b)
    draw.text((x0 + 210, card_y + 136), cp_map.get("154", "?"), fill="#315b41", font=indus_md)
    draw.text((x0 + 24, card_y + 190), "#156 -> #154 (q=0.0083* · MD+HA)", fill="#43554a", font=f_mono_sm)

    # Card 3: 690 Restored + Internal Rosetta + Export #091
    x0, x1 = cards_x[2]
    draw.rounded_rectangle([(x0, card_y), (x1, card_y + card_h)], radius=16, fill="#f6f7f3", outline="#dfe5dc", width=1)
    draw.text((x0 + 20, card_y + 18), "3. ROSETTA, TRADE & LOCO AI", fill="#315b41", font=f_mono)
    # Show #341 (quadruped) + #091 (twins) + #790 (ring) + #740 (jar)
    draw.text((x0 + 26, card_y + 48), cp_map.get("341", "?"), fill="#8a4b35", font=indus_md)
    draw.text((x0 + 102, card_y + 48), cp_map.get("091", "?"), fill="#315b41", font=indus_md)
    draw.text((x0 + 184, card_y + 48), cp_map.get("790", "?"), fill="#182c22", font=indus_md)
    draw.text((x0 + 256, card_y + 48), cp_map.get("740", "?"), fill="#182c22", font=indus_md)
    draw.text((x0 + 24, card_y + 98), "#341 TAB:C   #091 42x   #790   #740", fill="#68766b", font=f_mono_sm)
    draw.text((x0 + 20, card_y + 128), "TAB:C Icon Lock: 96.2% (+19.7s)", fill="#182c22", font=f_mono)
    draw.text((x0 + 20, card_y + 156), "690 Damaged Seals Restored", fill="#315b41", font=f_sans_b)
    draw.text((x0 + 20, card_y + 186), "12-City LOCO: 57.5% Top-1 / 82.9% T5", fill="#43554a", font=f_mono_sm)

    # Footer bar
    draw.text((68, 542), "github.com/emraresoftware/indusemare  ·  macOS & Linux SHA-256 Reproducible  ·  MIT / CC-BY-4.0", fill="#9db5a5", font=f_mono)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)
    if ART_DIR and ART_DIR.exists():
        img.save(ART_DIR / "og-indusemare.png")
    print("Saved OG social card:", OUT_PATH, OUT_PATH.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
