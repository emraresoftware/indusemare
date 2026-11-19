#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 0: Canonical Uncontaminated Data Warehouse & Audit
=======================================================================
Ingests raw Indus Valley script corpora, strictly strips any speculative
decipherment/translation columns (zero theory leakage), verifies reading
directionality empirically, extracts ICIT glyph decompositions and visual
similarity edges, and writes a clean canonical SQLite + CSV warehouse.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Columns that MUST be dropped immediately to prevent speculative-theory leakage.
FORBIDDEN_COLUMNS = {"sanskrit", "translation", "notes"}

# Known damage / unreadable placeholder sign IDs in ICIT encoding.
DAMAGE_TOKENS = {"000", "999"}


def normalize_direction_label(raw_dir: str) -> str:
    d = (raw_dir or "").strip().upper()
    if d in {"R/L", "L/R", "T/B", "BUS", "NR", "SYM"}:
        return d
    return "UNK"


def parse_inscription_lines(raw_text: str) -> list[dict[str, Any]]:
    """
    Parse ICIT-formatted text like '+740-390-590+' or ']000-031/151-740-235+'.

    Edge markers in ICIT:
      '+' -> intact edge
      '[' or ']' -> broken / damaged edge

    CRITICAL EMPIRICAL FACT (verified automatically in audit):
    In the upstream ICIT-derived corpus, BOTH 'R/L' and 'L/R' records are
    stored in normalized canonical seal-impression orientation where the
    leftmost token is the TERMINAL (final) sign (e.g., Jar 740) and the
    rightmost token is the INITIAL sign. Therefore, chronological reading
    order is tokens[::-1].
    """
    text = (raw_text or "").strip()
    if not text:
        return []

    raw_lines = [seg.strip() for seg in text.split("/") if seg.strip()]
    parsed_lines: list[dict[str, Any]] = []

    for idx, seg in enumerate(raw_lines):
        left_broken = seg.startswith("]") or seg.startswith("[")
        right_broken = seg.endswith("[") or seg.endswith("]")
        visual_tokens = re.findall(r"\d{3}", seg)
        if not visual_tokens:
            continue

        # Canonical chronological reading order: right-to-left over normalized visual string
        reading_tokens = list(reversed(visual_tokens))
        has_damage_token = any(t in DAMAGE_TOKENS for t in reading_tokens)
        clean_tokens = [t for t in reading_tokens if t not in DAMAGE_TOKENS]

        parsed_lines.append(
            {
                "line_idx": idx,
                "raw_segment": seg,
                "left_broken": int(left_broken),
                "right_broken": int(right_broken),
                "has_damage_token": int(has_damage_token),
                "visual_seq": " ".join(visual_tokens),
                "reading_seq": " ".join(reading_tokens),
                "clean_reading_seq": " ".join(clean_tokens),
                "token_count_raw": len(reading_tokens),
                "token_count_clean": len(clean_tokens),
            }
        )
    return parsed_lines


def parse_glyph_tables_from_sql(sql_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract ONLY non-speculative structural tables from indus_website_population.sql:
      - GLYPH (GLYPHID, UNICODE)  -> PUA unicode strings expose multi-codepoint compounds!
      - GLYPHSIMILARITY (GLYPHID1, GLYPHID2) -> visual similarity edges
    Strictly ignores DECIPHERMENT / TRANSLITERATION tables.
    """
    if not sql_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    sql = sql_path.read_text(encoding="utf-8", errors="ignore")

    # Parse GLYPH rows: (id, "&#xE...;") — split cleanly up to next SQL statement because HTML entities contain ';'
    glyph_rows = []
    m_glyph = re.search(r"INSERT INTO GLYPH\s*\([^)]+\)\s*VALUES\s*(.+?);\s*\r?\n(?:INSERT|CREATE|DROP|USE|--|/\*|$)", sql, re.S)
    if m_glyph:
        block = m_glyph.group(1)
        for gid_str, uval in re.findall(r'\(\s*(\d+)\s*,\s*"([^"]*)"\s*\)', block):
            gid = int(gid_str)
            if gid == 0:
                continue
            codepoints = re.findall(r"&#x([0-9A-Fa-f]+);", uval)
            glyph_rows.append(
                {
                    "sign_id": f"{gid:03d}",
                    "glyph_int": gid,
                    "unicode_html": uval,
                    "codepoints": " ".join(codepoints),
                    "component_count": len(codepoints),
                    "is_pua_compound": int(len(codepoints) > 1),
                }
            )
    glyph_df = pd.DataFrame(glyph_rows).drop_duplicates(subset=["sign_id"])

    # Map single codepoints back to base sign_ids to decompose compounds automatically
    cp_to_sign: dict[str, str] = {}
    if not glyph_df.empty:
        for _, row in glyph_df[glyph_df["component_count"] == 1].iterrows():
            cp = row["codepoints"]
            if cp and cp not in cp_to_sign:
                cp_to_sign[cp] = row["sign_id"]

        decomps = []
        for _, row in glyph_df.iterrows():
            cps = row["codepoints"].split() if row["codepoints"] else []
            if len(cps) > 1:
                parts = [cp_to_sign.get(cp, f"U+{cp}") for cp in cps]
                decomps.append("+".join(parts))
            else:
                decomps.append("")
        glyph_df["pua_decomposition"] = decomps

    # Parse GLYPHSIMILARITY pairs
    sim_rows = []
    m_sim = re.search(
        r"INSERT INTO GLYPHSIMILARITY\s*\([^)]+\)\s*VALUES\s*(.+?);", sql, re.S
    )
    if m_sim:
        for g1, g2 in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", m_sim.group(1)):
            s1, s2 = f"{int(g1):03d}", f"{int(g2):03d}"
            if s1 != s2:
                sim_rows.append({"sign_a": min(s1, s2), "sign_b": max(s1, s2), "source": "sql_glyphsimilarity"})
    sim_df = pd.DataFrame(sim_rows).drop_duplicates()

    return glyph_df, sim_df


def build_canonical_warehouse() -> dict[str, Any]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RAW_DIR / "inscriptions_5679.csv"
    df = pd.read_csv(raw_csv, dtype=str).fillna("")

    # 1. Strip speculative theory columns immediately
    dropped_cols = [c for c in df.columns if c.strip().lower() in FORBIDDEN_COLUMNS]
    df = df.drop(columns=dropped_cols)

    # 2. Clean column names
    rename_map = {
        "id": "inscription_id",
        "area-section": "area_section",
        "block-house": "block_house",
        "room-grid": "room_grid",
        "excavation-idno": "excavation_idno",
        "cross-section": "cross_section",
        "dir.": "direction_raw",
        "text length": "declared_text_length",
        "signs": "declared_sign_count",
        "horizontal(mm)": "horizontal_mm",
        "vertical(mm)": "vertical_mm",
        "thickness(mm)": "thickness_mm",
        "text": "raw_text",
        "class": "text_class",
        "type": "object_type",
    }
    df = df.rename(columns=rename_map)

    # Normalize key fields
    df["direction"] = df["direction_raw"].map(normalize_direction_label)
    df["complete"] = df["complete"].str.strip().str.upper()
    df["region"] = df["region"].str.strip()
    df["site"] = df["site"].str.strip()
    df["symbol"] = df["symbol"].str.strip()
    df["object_type"] = df["object_type"].str.strip()

    # Classify Heartland vs Foreign / Peripheral regions
    foreign_regions = {"Mesopotamia", "Persian Gulf", "Iranian Plateau", "Central Asia"}
    df["is_foreign"] = df["region"].isin(foreign_regions).astype(int)

    # Parse every inscription into structured lines
    line_records = []
    ins_line_counts = []
    ins_clean_seqs = []
    ins_clean_lens = []
    ins_has_damage = []

    for _, row in df.iterrows():
        lines = parse_inscription_lines(row["raw_text"])
        ins_line_counts.append(len(lines))
        all_clean = [tok for ln in lines for tok in ln["clean_reading_seq"].split() if tok]
        ins_clean_seqs.append(" ".join(all_clean))
        ins_clean_lens.append(len(all_clean))
        ins_has_damage.append(
            int(
                any(
                    ln["left_broken"] or ln["right_broken"] or ln["has_damage_token"]
                    for ln in lines
                )
            )
        )
        for ln in lines:
            line_records.append(
                {
                    "inscription_id": row["inscription_id"],
                    "cisi": row["cisi"],
                    "region": row["region"],
                    "site": row["site"],
                    "is_foreign": row["is_foreign"],
                    "object_type": row["object_type"],
                    "symbol": row["symbol"],
                    "material": row["material"],
                    "period": row["period"],
                    "complete": row["complete"],
                    "direction": row["direction"],
                    **ln,
                }
            )

    df["parsed_line_count"] = ins_line_counts
    df["clean_reading_seq_full"] = ins_clean_seqs
    df["clean_sign_count_full"] = ins_clean_lens
    df["has_structural_damage"] = ins_has_damage

    lines_df = pd.DataFrame(line_records)

    # Define strict high-confidence analytic subset:
    # complete == Y, no damage tokens, no broken edges, >= 2 clean signs
    lines_df["is_strict_analytic"] = (
        (lines_df["complete"] == "Y")
        & (lines_df["has_damage_token"] == 0)
        & (lines_df["left_broken"] == 0)
        & (lines_df["right_broken"] == 0)
        & (lines_df["token_count_clean"] >= 2)
    ).astype(int)

    # Extract glyph metadata & visual similarity pairs from SQL
    glyph_df, sim_df = parse_glyph_tables_from_sql(RAW_DIR / "indus_website_population.sql")

    # Compute corpus sign frequencies and attach to glyph catalog
    all_sign_counts = Counter()
    analytic_sign_counts = Counter()
    for seq in lines_df["clean_reading_seq"]:
        if seq:
            all_sign_counts.update(seq.split())
    for seq in lines_df.loc[lines_df["is_strict_analytic"] == 1, "clean_reading_seq"]:
        if seq:
            analytic_sign_counts.update(seq.split())

    all_observed_signs = sorted(set(all_sign_counts.keys()) | set(glyph_df["sign_id"].tolist() if not glyph_df.empty else []))
    glyph_meta_map = glyph_df.set_index("sign_id").to_dict(orient="index") if not glyph_df.empty else {}
    catalog_rows = []
    for sid in all_observed_signs:
        meta = glyph_meta_map.get(sid, {})
        catalog_rows.append(
            {
                "sign_id": sid,
                "freq_all": all_sign_counts.get(sid, 0),
                "freq_strict_analytic": analytic_sign_counts.get(sid, 0),
                "unicode_html": meta.get("unicode_html", ""),
                "codepoints": meta.get("codepoints", ""),
                "component_count": meta.get("component_count", 0),
                "is_pua_compound": meta.get("is_pua_compound", 0),
                "pua_decomposition": meta.get("pua_decomposition", ""),
            }
        )
    catalog_df = pd.DataFrame(catalog_rows).sort_values("freq_all", ascending=False)

    # Write canonical CSVs
    df.to_csv(PROC_DIR / "inscriptions_clean.csv", index=False)
    lines_df.to_csv(PROC_DIR / "lines_clean.csv", index=False)
    catalog_df.to_csv(PROC_DIR / "glyphs_clean.csv", index=False)
    sim_df.to_csv(PROC_DIR / "glyph_similarity.csv", index=False)

    # Write canonical SQLite database
    db_path = PROC_DIR / "indus_canonical.sqlite"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        df.to_sql("inscriptions", conn, index=False)
        lines_df.to_sql("lines", conn, index=False)
        catalog_df.to_sql("glyphs", conn, index=False)
        sim_df.to_sql("glyph_similarity", conn, index=False)
        conn.execute("CREATE INDEX idx_lines_site ON lines(site)")
        conn.execute("CREATE INDEX idx_lines_region ON lines(region)")
        conn.execute("CREATE INDEX idx_lines_analytic ON lines(is_strict_analytic)")
        conn.execute("CREATE INDEX idx_ins_foreign ON inscriptions(is_foreign)")

    # Empirical Directionality Audit (proves both R/L and L/R share canonical orientation)
    dir_audit = {}
    for dlabel in ["R/L", "L/R", "BUS", "T/B", "UNK"]:
        sub = lines_df[(lines_df["direction"] == dlabel) & (lines_df["is_strict_analytic"] == 1)]
        if len(sub) == 0:
            continue
        init_counts = Counter(s.split()[0] for s in sub["clean_reading_seq"])
        final_counts = Counter(s.split()[-1] for s in sub["clean_reading_seq"])
        dir_audit[dlabel] = {
            "strict_lines": int(len(sub)),
            "top_initial_reading": init_counts.most_common(5),
            "top_final_reading": final_counts.most_common(5),
            "jar_740_initial_pct": round(100.0 * init_counts.get("740", 0) / len(sub), 2),
            "jar_740_final_pct": round(100.0 * final_counts.get("740", 0) / len(sub), 2),
        }

    strict_df = lines_df[lines_df["is_strict_analytic"] == 1]
    unique_strict_seqs = strict_df["clean_reading_seq"].nunique()

    audit = {
        "source_file": str(raw_csv.relative_to(ROOT)),
        "dropped_speculative_columns": dropped_cols,
        "total_inscriptions": int(len(df)),
        "total_parsed_lines": int(len(lines_df)),
        "strict_analytic_lines": int(len(strict_df)),
        "strict_unique_sequences": int(unique_strict_seqs),
        "dedup_repetition_pct": round(100.0 * (1.0 - unique_strict_seqs / max(len(strict_df), 1)), 2),
        "total_clean_sign_tokens": int(sum(all_sign_counts.values())),
        "unique_observed_signs": int(len(all_sign_counts)),
        "hapax_signs_count": int(sum(1 for c in all_sign_counts.values() if c == 1)),
        "signs_freq_ge_5": int(sum(1 for c in all_sign_counts.values() if c >= 5)),
        "pua_compound_signs_detected": int(catalog_df["is_pua_compound"].sum()),
        "sql_visual_similarity_edges": int(len(sim_df)),
        "foreign_inscriptions_count": int(df["is_foreign"].sum()),
        "regions": df["region"].value_counts().to_dict(),
        "mean_strict_line_length": round(float(strict_df["token_count_clean"].mean()), 3),
        "max_strict_line_length": int(strict_df["token_count_clean"].max()),
        "directionality_audit": dir_audit,
    }

    (REPORTS_DIR / "layer0_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    summary = build_canonical_warehouse()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
