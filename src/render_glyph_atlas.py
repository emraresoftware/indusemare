#!/usr/bin/env python3
"""
INDUS-DEDUCT — Render Visual Glyph Atlas of All Frequent ICIT Signs (freq >= 5)
==============================================================================
Sorts by ICIT numeric ID (which groups morphologically related sign families!)
and renders crisp, high-resolution contact sheets for visual inspection +
elemental/pictographic categorization.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports" / "atlas"
ART_DIR = Path("/Users/emre/.gemini/antigravity/brain/969a4497-dabc-4f45-8c95-dcaebdbcb608")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    glyphs = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")
    glyphs["freq_all"] = glyphs["freq_all"].astype(int)
    glyphs["sid_int"] = glyphs["sign_id"].astype(int)

    # Keep all signs with freq >= 5 that have a valid codepoint in the font
    sub = glyphs[(glyphs["freq_all"] >= 5) & (glyphs["codepoints"].str.len() > 0)].sort_values("sid_int")
    print(f"Frequent renderable signs (freq >= 5): {len(sub)} (total tokens: {sub['freq_all'].sum()})")

    # Also compute quick slot role (Init/Med/Fin) from lines_clean.csv for each sign
    lines = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    lines["is_strict_analytic"] = lines["is_strict_analytic"].astype(int)
    strict = lines[(lines["is_strict_analytic"] == 1) & (lines["direction"].isin(["R/L", "L/R", "BUS"]))]
    pos = {}
    for seq_str in strict["clean_reading_seq"].unique():
        seq = seq_str.split()
        n = len(seq)
        for i, s in enumerate(seq):
            d = pos.setdefault(s, [0, 0, 0])
            if i == 0:
                d[0] += 1
            elif i == n - 1:
                d[2] += 1
            else:
                d[1] += 1

    indus_font = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 46)
    try:
        id_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        sub_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        id_font = sub_font = title_font = ImageFont.load_default()

    records = sub.to_dict("records")
    per_page = 100
    cols = 10
    rows = 10
    cw, ch = 128, 112
    n_pages = math.ceil(len(records) / per_page)

    for p in range(n_pages):
        chunk = records[p * per_page : (p + 1) * per_page]
        img = Image.new("RGB", (cols * cw, rows * ch + 34), "white")
        draw = ImageDraw.Draw(img)
        s_min, s_max = chunk[0]["sign_id"], chunk[-1]["sign_id"]
        draw.text(
            (12, 8),
            f"INDUS-DEDUCT Visual Atlas — Page {p+1}/{n_pages} (ICIT {s_min}..{s_max}, freq>=5)",
            fill="black",
            font=title_font,
        )
        for idx, rec in enumerate(chunk):
            r, c = divmod(idx, cols)
            x0, y0 = c * cw + 3, r * ch + 36
            sid = rec["sign_id"]
            freq = rec["freq_all"]
            cps = [int(h, 16) for h in rec["codepoints"].split() if h]
            glyph_str = "".join(chr(cp) for cp in cps)

            # Determine dominant slot badge
            p_arr = pos.get(sid, [0, 0, 0])
            ptot = sum(p_arr)
            if ptot >= 5:
                pi, pm, pf = p_arr[0] / ptot, p_arr[1] / ptot, p_arr[2] / ptot
                if pf >= 0.60:
                    badge, bcol = "SON", "#B03A2E"
                elif pi >= 0.50:
                    badge, bcol = "BAS", "#1E8449"
                elif pm >= 0.65:
                    badge, bcol = "ORT", "#2874A6"
                else:
                    badge, bcol = "ESN", "#6C3483"
            else:
                badge, bcol = "", "#888888"

            draw.rectangle([x0, y0, x0 + cw - 6, y0 + ch - 6], outline="#DDDDDD", width=1)
            draw.text((x0 + 5, y0 + 3), f"{sid} ({freq})", fill="#1A5276", font=id_font)
            if badge:
                draw.text((x0 + cw - 35, y0 + 4), badge, fill=bcol, font=sub_font)
            draw.text((x0 + 34, y0 + 24), glyph_str, fill="black", font=indus_font)

        out_path = REPORTS_DIR / f"atlas_p{p+1}.png"
        img.save(out_path)
        img.save(ART_DIR / f"atlas_p{p+1}.png")
        print(f"Saved {out_path} ({len(chunk)} signs)")


if __name__ == "__main__":
    main()
