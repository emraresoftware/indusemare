#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 3F: The 4 Blind Spots Audit
================================================
Directly tests Emre's 4 newly identified blind spots on the full 5,679-row corpus:
  1. FAUNA-FIRST TAXONOMY (All 18+ animal species grouped by ecological niche:
     Aquatic, Wild Predator/Megafauna, Domestic/Agri, Mythical/Composite, Bull1 subtypes)
     -> Does each animal species carry a distinct lexical/sign signature?
  2. TOHUM & TUREVLERI (Seed / Kernel / Germination 803-840 family + Sprout/Grain 380-426)
     -> Grammar, co-occurrence, and ecological links of the 500+ occurrence seed cluster!
  3. YAPI SISTEMI & URBAN HIERARCHY (Architectural signs 315-326, 572-578, 615-642
     + Excavation-sector Citadel vs Granary/Workshops vs Lower Town in Harappa & Mohenjo-daro!)
  4. AHLAK / DUZEN / DENGE SISTEMI (Balance/Yoke 151-161, 6-Spoked Wheel 820,
     Symmetry 550/690/692/840/845 across Citadel vs Lower Town vs Tags vs Tablets!)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Ecological / Behavioral Fauna Grouping (Fauna-First!)
FAUNA_NICHES: dict[str, list[str]] = {
    "NICHE_1_MYTHIC_UNICORN": ["Bull1:W", "Bull1", "Bull1:J", "Bull1:S", "Bull1:I", "Bull1:L", "Bull1:U", "Bull1:II", "Bull1:V", "Bull1:X", "Bull1:T"],
    "NICHE_2_DOMESTIC_BOVID": ["Bult", "Gaur", "Zebu", "Bull", "Bull2", "Bull3"],
    "NICHE_3_WILD_MEGAFAUNA": ["Elep", "Rhin", "Tigr"],
    "NICHE_4_AQUATIC_RIVER": ["Gavi", "Fish", "Buff"],
    "NICHE_5_SMALL_PASTORAL": ["Hare", "Goat", "Goat:2", "Goat:3", "Goat:4"],
    "NICHE_6_COMPOSITE_DEITY": ["Mult", "Comp", "CompBull", "Anth", "Scene"],
    "NICHE_7_FLORA_PHYT": ["Phyt"],
}

SEED_FAMILY = ["803", "804", "806", "808", "812", "831", "832", "838", "840", "842", "844"]
SPROUT_GRAIN_FAMILY = ["380", "382", "384", "388", "390", "391", "411", "413", "415", "416", "417", "420", "421", "422", "423", "426", "760"]
ARCH_FAMILY = ["315", "317", "318", "320", "321", "323", "326", "360", "362", "572", "575", "578", "615", "617", "621", "625", "626", "630", "632", "634", "636", "637", "642"]
ORDER_BALANCE_FAMILY = ["151", "153", "154", "155", "156", "158", "159", "160", "161", "820", "821", "540", "550", "690", "692", "845"]


def enrich_group(df: pd.DataFrame, mask: pd.Series, min_hits: int = 3) -> list[dict[str, Any]]:
    dedup = df.assign(_in=mask.astype(int)).drop_duplicates(subset=["_in", "clean_reading_seq"])
    in_df = dedup[dedup["_in"] == 1]
    out_df = dedup[dedup["_in"] == 0]
    n_in, n_out = len(in_df), len(out_df)
    if n_in < 5:
        return []
    c_in = Counter(t for s in in_df["clean_reading_seq"] for t in set(s.split()))
    c_out = Counter(t for s in out_df["clean_reading_seq"] for t in set(s.split()))
    hits = []
    for sign, a in c_in.items():
        if a < min_hits:
            continue
        b = n_in - a
        c = c_out[sign]
        d = n_out - c
        p_in = a / n_in
        p_out = c / max(n_out, 1)
        lift = p_in / max(p_out, 1e-6)
        if lift < 1.6:
            continue
        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        if pval < 0.02:
            hits.append(
                {
                    "sign": sign,
                    "in": f"{a}/{n_in} ({100*p_in:.1f}%)",
                    "out_pct": round(100 * p_out, 2),
                    "lift": round(lift, 2),
                    "p_value": float(f"{pval:.3e}"),
                }
            )
    hits.sort(key=lambda x: x["p_value"])
    return hits


def main() -> None:
    ins = pd.read_csv(PROC_DIR / "inscriptions_clean.csv", dtype=str).fillna("")
    lines = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    lines["is_strict_analytic"] = lines["is_strict_analytic"].astype(int)
    df = lines.merge(
        ins[["inscription_id", "area_section", "block_house", "cult", "material", "shape", "period", "phase"]],
        on="inscription_id",
        how="left",
        suffixes=("", "_ins"),
    )
    df = df[df["clean_reading_seq"].str.len() > 0].copy()

    # =========================================================================
    # 1. FAUNA-FIRST TAXONOMY: PER-SPECIES & PER-NICHE LEXICAL SIGNATURES
    # =========================================================================
    niche_results = {}
    print("=== 1. FAUNA-FIRST ECOLOGICAL NICHE SIGNATURES ===")
    for niche, syms in FAUNA_NICHES.items():
        mask = df["symbol"].isin(syms)
        hits = enrich_group(df, mask, min_hits=3)
        niche_results[niche] = {"lines": int(mask.sum()), "top_signs": hits[:8]}
        print(f"-- {niche} (lines={int(mask.sum())}) --")
        for h in hits[:5]:
            print(f"   Sign {h['sign']} | in={h['in']} vs out={h['out_pct']}% | lift={h['lift']}x | p={h['p_value']}")

    # Individual species signatures (for key distinct species!)
    species_results = {}
    print("\n=== 1B. INDIVIDUAL SPECIES SIGNATURES (Zebu, Gaur, Bult, Elep, Rhin, Tigr, Hare, Goat, Buff) ===")
    for sp in ["Zebu", "Gaur", "Bult", "Elep", "Rhin", "Tigr", "Hare", "Buff", "Gavi", "Fish"]:
        mask = df["symbol"] == sp
        hits = enrich_group(df, mask, min_hits=2)
        species_results[sp] = {"lines": int(mask.sum()), "top_signs": hits[:6]}
        top_str = ", ".join(f"{h['sign']}({h['lift']}x, p={h['p_value']})" for h in hits[:4])
        print(f"  {sp:6s} (n={int(mask.sum()):3d}): {top_str or 'no strong single-sign lock'}")

    # =========================================================================
    # 2. TOHUM & TUREVLERI (SEED 803-840 + SPROUT/GRAIN 380-426)
    # =========================================================================
    strict = df[(df["is_strict_analytic"] == 1) & (df["direction"].isin(["R/L", "L/R", "BUS"]))]
    unique_seqs = [s.split() for s in strict["clean_reading_seq"].unique()]

    seed_set = set(SEED_FAMILY)
    sprout_set = set(SPROUT_GRAIN_FAMILY)
    seed_mask = df["clean_reading_seq"].apply(lambda s: any(t in seed_set for t in s.split()))
    sprout_mask = df["clean_reading_seq"].apply(lambda s: any(t in sprout_set for t in s.split()))

    # Test seed <-> sprout co-occurrence and directional flow (Seed -> Sprout vs Sprout -> Seed)
    s2sp = sp2s = 0
    seed_pos = Counter()
    seed_left = Counter()
    seed_right = Counter()
    for seq in unique_seqs:
        n = len(seq)
        for i, t in enumerate(seq):
            if t in seed_set:
                seed_pos["init" if i == 0 else ("fin" if i == n - 1 else "med")] += 1
                if i > 0:
                    seed_left[seq[i - 1]] += 1
                if i < n - 1:
                    seed_right[seq[i + 1]] += 1
        for a, b in zip(seq[:-1], seq[1:]):
            if a in seed_set and b in sprout_set:
                s2sp += 1
            if a in sprout_set and b in seed_set:
                sp2s += 1

    seed_obj = df[seed_mask]["object_type"].value_counts().head(6).to_dict()
    seed_sym = df[seed_mask]["symbol"].replace("", "NONE").value_counts().head(6).to_dict()
    print("\n=== 2. TOHUM & TUREVLERI (SEED 803-840 FAMILY) ===")
    print(f"  Seed-family lines: {int(seed_mask.sum())} | Sprout/Grain lines: {int(sprout_mask.sum())}")
    print(f"  Seed positions (unique strict): {dict(seed_pos)}")
    print(f"  Seed top left neighbors : {seed_left.most_common(6)}")
    print(f"  Seed top right neighbors: {seed_right.most_common(6)}")
    print(f"  Directional flow: Seed->Sprout = {s2sp} vs Sprout->Seed = {sp2s}")
    print(f"  Seed object types: {seed_obj}")
    print(f"  Seed motifs      : {seed_sym}")

    # =========================================================================
    # 3. YAPI SISTEMI & URBAN HIERARCHY (CITADEL vs GRANARY vs LOWER TOWN!)
    # =========================================================================
    print("\n=== 3. YAPI SISTEMI — HARAPPA & MOHENJO-DARO URBAN SECTORS ===")
    har = df[df["site"] == "Harappa"]
    md = df[df["site"] == "Mohenjo-daro"]
    print("  Harappa top area_sections:", har["area_section"].value_counts().head(15).to_dict())
    print("  Mohenjo-daro top area_sections:", md["area_section"].value_counts().head(12).to_dict())

    # Harappa: Mound AB (Citadel/Walled Acropolis) vs Mound F (Great Granary/Workshops) vs Mound E/ET (Lower Town/Crafts/Gate)
    # Mohenjo-daro: SD / L (Citadel / Great Bath / Pillared Hall) vs DK / HR / VS (Lower Town residential & commercial)
    urban_zones: dict[str, pd.Series] = {
        "HAR_CITADEL_MOUND_AB": (df["site"] == "Harappa") & (df["area_section"].str.contains(r"AB", case=False, na=False)),
        "HAR_GRANARY_WORKSHOPS_F": (df["site"] == "Harappa") & (df["area_section"].str.contains(r"^F|Mound F|\bF\b", case=False, na=False)),
        "HAR_LOWER_TOWN_E_ET": (df["site"] == "Harappa") & (df["area_section"].str.contains(r"^E|ET|Mound E", case=False, na=False)),
        "MD_CITADEL_SD_L": (df["site"] == "Mohenjo-daro") & (df["area_section"].str.contains(r"SD|^L-", case=False, na=False)),
        "MD_LOWER_DKG_SOUTH": (df["site"] == "Mohenjo-daro") & (df["area_section"].str.contains(r"DKG", case=False, na=False)),
        "MD_LOWER_HR_VS_RESIDENTIAL": (df["site"] == "Mohenjo-daro") & (df["area_section"].str.contains(r"HR|VS", case=False, na=False)),
    }

    # Test Citadel (AB + SD + L) vs Lower Town (E/ET + DK + HR + VS) across ALL signs AND across our 4 Value Families!
    citadel_mask = urban_zones["HAR_CITADEL_MOUND_AB"] | urban_zones["MD_CITADEL_SD_L"]
    granary_mask = urban_zones["HAR_GRANARY_WORKSHOPS_F"]
    lower_mask = urban_zones["HAR_LOWER_TOWN_E_ET"] | urban_zones["MD_LOWER_DKG_SOUTH"] | urban_zones["MD_LOWER_HR_VS_RESIDENTIAL"]

    print(f"\n  Zone line counts: Citadel={int(citadel_mask.sum())} | Granary/F={int(granary_mask.sum())} | LowerTown={int(lower_mask.sum())}")

    urban_enrichment = {}
    for zname, zmask in [("CITADEL_ALL", citadel_mask), ("GRANARY_WORKSHOPS_F", granary_mask), ("LOWER_TOWN_ALL", lower_mask)]:
        sub_only = df[citadel_mask | granary_mask | lower_mask].copy()
        zm = zmask.loc[sub_only.index]
        hits = enrich_group(sub_only, zm, min_hits=4)
        urban_enrichment[zname] = {"lines": int(zm.sum()), "top_signs": hits[:8]}
        print(f"  -- {zname} (vs other mapped urban zones, n={int(zm.sum())}) --")
        for h in hits[:5]:
            print(f"     Sign {h['sign']} | in={h['in']} vs out={h['out_pct']}% | lift={h['lift']}x | p={h['p_value']}")

    # Compare the 4 Structural/Value Families (Seed, Arch, Order/Balance, Fish/Star) across Citadel vs Granary vs Lower Town!
    arch_set = set(ARCH_FAMILY)
    order_set = set(ORDER_BALANCE_FAMILY)
    fish_set = {"220", "226", "231", "233", "235", "240"}
    fam_zone_table = {}
    print("\n=== 4. AHLAK/DUZEN, YAPI, TOHUM & YILDIZ BY URBAN ZONE (CITADEL vs GRANARY vs LOWER TOWN) ===")
    for fname, fset in [("TOHUM_SEED", seed_set), ("YAPI_ARCH", arch_set), ("AHLAK_DENGE_ORDER", order_set), ("YILDIZ_FISH", fish_set)]:
        row = {}
        for zname, zmask in [("Citadel", citadel_mask), ("Granary_F", granary_mask), ("LowerTown", lower_mask)]:
            zdf = df[zmask].drop_duplicates(subset=["clean_reading_seq"])
            hits = sum(any(t in fset for t in s.split()) for s in zdf["clean_reading_seq"])
            tot = len(zdf)
            row[zname] = f"{hits}/{tot} ({100*hits/max(tot,1):.1f}%)"
        fam_zone_table[fname] = row
        print(f"  {fname:20s}: {row}")

    # Also check exact object-type distribution across Citadel vs Granary vs Lower Town (what WAS written in each zone?)
    zone_obj = {}
    for zname, zmask in [("Citadel", citadel_mask), ("Granary_F", granary_mask), ("LowerTown", lower_mask)]:
        zone_obj[zname] = df[zmask]["object_type"].value_counts(normalize=True).head(5).mul(100).round(1).to_dict()
        print(f"  Object types in {zname:10s}: {zone_obj[zname]}")

    out = {
        "fauna_niche_signatures": niche_results,
        "individual_species_signatures": species_results,
        "seed_family_analysis": {
            "seed_lines": int(seed_mask.sum()),
            "sprout_lines": int(sprout_mask.sum()),
            "seed_positions": dict(seed_pos),
            "seed_top_left": seed_left.most_common(10),
            "seed_top_right": seed_right.most_common(10),
            "seed_to_sprout": s2sp,
            "sprout_to_seed": sp2s,
            "seed_objects": seed_obj,
            "seed_motifs": seed_sym,
        },
        "urban_hierarchy_enrichment": urban_enrichment,
        "value_families_by_urban_zone": fam_zone_table,
        "object_types_by_urban_zone": zone_obj,
    }
    (REPORTS_DIR / "layer3f_fauna_seed_arch_order.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
