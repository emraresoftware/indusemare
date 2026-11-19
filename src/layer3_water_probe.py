#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 3A: Top-Down "WATER" (SU) Multi-Axis Triangulation Probe
=============================================================================
Tests Emre's semantic-anchor hypothesis ("find WATER top-down, then cascade")
across 4 independent archaeological & structural axes WITHOUT cherry-picking:
  Axis 1: Aquatic Iconography (Gavi/gharial, Water Buffalo, Boat/Ship, Fish scenes)
  Axis 2: Hydraulic & Maritime Excavation Contexts (Mohenjo-daro SD = Great Bath,
          Lothal = Dockyard/Port, Dholavira = Reservoirs)
  Axis 3: Liquid/Water Vessel Artifacts (POT:T:g, POT:T:s — pointed-base goblets & jars)
  Axis 4: Cross-Axis Intersection Score (signs enriched across >= 2 independent water axes!)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def enrich_binary(
    df: pd.DataFrame, mask: pd.Series, label: str, min_hits: int = 3
) -> list[dict[str, Any]]:
    """Compute sign enrichment inside mask vs outside mask on deduplicated sequences."""
    dedup = df.assign(_in=mask.astype(int)).drop_duplicates(subset=["_in", "clean_reading_seq"])
    in_df = dedup[dedup["_in"] == 1]
    out_df = dedup[dedup["_in"] == 0]
    n_in, n_out = len(in_df), len(out_df)
    if n_in < 5:
        return []

    c_in = Counter()
    c_out = Counter()
    for s in in_df["clean_reading_seq"]:
        for t in set(s.split()):
            c_in[t] += 1
    for s in out_df["clean_reading_seq"]:
        for t in set(s.split()):
            c_out[t] += 1

    results = []
    for sign, a in c_in.items():
        if a < min_hits:
            continue
        b = n_in - a
        c = c_out[sign]
        d = n_out - c
        p_in = a / n_in
        p_out = c / max(n_out, 1)
        lift = p_in / max(p_out, 1e-6)
        if lift < 1.5:
            continue
        # Use Fisher exact for small counts, chi2 otherwise
        if min(a, b, c, d) < 5:
            _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        else:
            _, pval, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)
        if pval < 0.05:
            results.append(
                {
                    "axis": label,
                    "sign_id": sign,
                    "in_hits": int(a),
                    "in_total": int(n_in),
                    "out_hits": int(c),
                    "out_total": int(n_out),
                    "in_pct": round(100 * p_in, 2),
                    "out_pct": round(100 * p_out, 2),
                    "lift": round(lift, 2),
                    "p_value": float(f"{pval:.3e}"),
                }
            )
    results.sort(key=lambda x: (x["p_value"], -x["lift"]))
    return results


def main() -> None:
    ins_df = pd.read_csv(PROC_DIR / "inscriptions_clean.csv", dtype=str).fillna("")
    lines_df = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")

    # Merge area_section & cult onto lines for excavation-block analysis
    meta = ins_df[["inscription_id", "area_section", "block_house", "cult", "material", "shape"]]
    df = lines_df.merge(meta, on="inscription_id", how="left", suffixes=("", "_ins"))
    df = df[df["clean_reading_seq"].str.len() > 0].copy()

    print("=== ALL SYMBOLS (top 40) ===")
    print(ins_df["symbol"].value_counts().head(40).to_dict())
    print("\n=== ALL CULT VALUES (top 25) ===")
    print(ins_df["cult"].value_counts().head(25).to_dict())
    print("\n=== MOHENJO-DARO AREA SECTIONS (top 20) ===")
    md = ins_df[ins_df["site"] == "Mohenjo-daro"]
    print(md["area_section"].value_counts().head(20).to_dict())
    print("\n=== POTTERY TYPES ===")
    print(ins_df[ins_df["object_type"].str.startswith("POT")]["object_type"].value_counts().to_dict())

    # Define 6 independent water/hydraulic/maritime test masks
    axes: dict[str, pd.Series] = {
        "AXIS_1A_GHARIAL_GAVI": df["symbol"].str.contains("Gavi", case=False, na=False),
        "AXIS_1B_WATER_BUFFALO": df["symbol"].str.contains("Buf", case=False, na=False),
        "AXIS_1C_BOAT_FISH_AQUATIC": df["symbol"].str.contains("Boat|Fish|Croc|Ship|Tort|Turt|Crab|Aqua", case=False, na=False),
        "AXIS_2A_GREAT_BATH_SD": (df["site"] == "Mohenjo-daro") & (df["area_section"].str.contains(r"\bSD\b", case=False, na=False)),
        "AXIS_2B_MARITIME_PORTS_LOTHAL_DHOLAVIRA": df["site"].isin(["Lothal", "Dholavira", "Bala-kot", "Shikarpur", "Gola Dhoro (Bagasra)"]),
        "AXIS_3_POTTERY_VESSELS": df["object_type"].str.startswith("POT"),
    }

    all_hits = {}
    sign_axis_scores: dict[str, list[dict[str, Any]]] = {}

    for ax_name, mask in axes.items():
        n_rows = int(mask.sum())
        hits = enrich_binary(df, mask, ax_name, min_hits=3)
        all_hits[ax_name] = {"matching_lines": n_rows, "top_enriched": hits[:15]}
        print(f"\n=== {ax_name} (lines={n_rows}) ===")
        for h in hits[:8]:
            print(f"  Sign {h['sign_id']} | in={h['in_hits']}/{h['in_total']} ({h['in_pct']}%) vs out={h['out_pct']}% | lift={h['lift']}x | p={h['p_value']}")
        for h in hits:
            if h["p_value"] < 0.01:
                sign_axis_scores.setdefault(h["sign_id"], []).append(h)

    # Cross-axis triangulation: signs appearing strongly across >= 2 independent water axes
    triangulated = []
    for sign_id, entries in sign_axis_scores.items():
        if len(entries) >= 2:
            # Combined Fisher-style score = sum of -log10(p)
            import math
            score = sum(-math.log10(max(e["p_value"], 1e-300)) for e in entries)
            triangulated.append(
                {
                    "sign_id": sign_id,
                    "axes_hit_count": len(entries),
                    "axes": [e["axis"] for e in entries],
                    "combined_neg_log10_p": round(score, 2),
                    "details": entries,
                }
            )
    triangulated.sort(key=lambda x: (-x["axes_hit_count"], -x["combined_neg_log10_p"]))

    print("\n=== CROSS-AXIS WATER TRIANGULATION (SIGNS HIT ON >= 2 INDEPENDENT AXES, p<0.01) ===")
    for t in triangulated[:15]:
        print(f"  Sign {t['sign_id']} | axes={t['axes_hit_count']} ({', '.join(t['axes'])}) | score={t['combined_neg_log10_p']}")

    out = {"axes": all_hits, "cross_axis_triangulated": triangulated}
    (REPORTS_DIR / "layer3_water_probe.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
