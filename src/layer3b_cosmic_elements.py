#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 3B: 7 Primal Elements Statistical Verification Engine
==========================================================================
Tests Emre's 7 primal semantic targets:
  1. SU (Water / Vessel / Water-Carrier / Stream)
  2. YILDIZ (Star / Fish=*min hypothesis / Rayed Disks)
  3. GOKYUZU (Sky / Vault / Roof / Sun / Crescent Moon)
  4. ATES (Fire / Flame / Hearth / Brazier / Altar / Kiln)
  5. TOPRAK (Earth / Plowed Field / Enclosure / Plant / Plow)
  6. HAVA & NEFES (Air / Wind / Breath / Bird / Wing / Spiral)
  + SAYI (Numerals / Stroke Counters)

Runs 4 falsifiable tests on the clean 5,679-inscription corpus:
  Test 1: Exact Numeral -> Fish (*min = Star/Constellation) significance test
  Test 2: Archaeological domain enrichment (Kalibangan Fire Altars, Copper/Kiln,
          Phyt/Plant motifs, Aquatic motifs, Pottery vessels, Cult stands)
  Test 3: Per-sign grammatical profile for every candidate in all 7 families
  Test 4: 7x7 Element-to-Element PMI Transition Network (which elements glue together?)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Visually cataloged candidate families from the 294-sign ICIT Atlas (strictly hypotheses to TEST!)
ELEMENT_FAMILIES: dict[str, list[str]] = {
    "SU_KAP_AKIS": [
        "700", "697", "698", "702", "703", "704", "705", "706",
        "740", "741", "742", "745", "746", "747", "752",
        "151", "154", "155", "156", "158", "161", "110", "111", "103", "927",
    ],
    "YILDIZ_BALIK": [
        "219", "220", "221", "222", "226", "231", "232", "233", "234",
        "235", "236", "240", "241", "242", "243",
        "794", "820", "821",
    ],
    "GOKYUZU_GUNES_AY": [
        "480", "482", "484", "572",
        "790", "791", "792", "817", "824", "840", "845",
        "900", "904", "905", "906", "920", "921", "923", "924",
    ],
    "ATES_OCAK_SUNAK": [
        "460", "500", "501", "503", "504",
        "585", "590", "592", "595", "597",
        "853", "861", "868", "869", "877", "880",
    ],
    "TOPRAK_TARLA_BITKI": [
        "615", "617", "621", "625", "626", "630", "632", "634", "636", "637", "642",
        "380", "382", "384", "388", "390", "391", "411", "413", "415", "416",
        "465", "467", "468", "472",
    ],
    "HAVA_NEFES_KUS": [
        "244", "255", "257", "260", "263", "267", "269", "272",
        "100", "101", "140",
        "165", "167", "175", "176",
        "455", "456", "540", "545", "550", "552", "554", "555", "556",
    ],
    "SAYI_OLCU": [
        "001", "002", "003", "004", "005", "006", "007",
        "013", "014", "015", "016", "017", "018", "019",
        "031", "032", "033", "034", "035", "036", "055", "056",
    ],
}

SHORT_NUMERALS = {
    "001": 1, "002": 2, "003": 3, "004": 4, "005": 5, "006": 6, "007": 7,
    "013": 3, "014": 4, "015": 5, "016": 6, "017": 7, "018": 8, "019": 9,
    "031": 1, "032": 2, "033": 3, "034": 4, "035": 5, "036": 6, "055": 12,
}
FISH_SIGNS = {"219", "220", "221", "222", "226", "231", "232", "233", "234", "235", "236", "240", "241", "242", "243"}


def test_numeral_fish(unique_seqs: list[list[str]]) -> dict[str, Any]:
    """Test whether [Numeral] -> [Fish] occurs far above random expectation (Parpola *min=star test)."""
    total_bi = 0
    num_any = 0
    any_fish = 0
    num_fish = 0
    fish_num = 0
    pair_counts = Counter()
    val_counts = Counter()

    for seq in unique_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            total_bi += 1
            a_num = a in SHORT_NUMERALS
            b_num = b in SHORT_NUMERALS
            a_fish = a in FISH_SIGNS
            b_fish = b in FISH_SIGNS
            if a_num:
                num_any += 1
            if b_fish:
                any_fish += 1
            if a_num and b_fish:
                num_fish += 1
                pair_counts[(a, b)] += 1
                val_counts[SHORT_NUMERALS[a]] += 1
            if a_fish and b_num:
                fish_num += 1

    # Fisher exact on 2x2: (a is Numeral) x (b is Fish)
    a_cell = num_fish
    b_cell = num_any - num_fish
    c_cell = any_fish - num_fish
    d_cell = total_bi - num_any - c_cell
    expected = (num_any * any_fish) / max(total_bi, 1)
    _, pval = fisher_exact([[a_cell, b_cell], [c_cell, d_cell]], alternative="greater")

    return {
        "total_bigrams_unique_seqs": total_bi,
        "observed_numeral_to_fish": num_fish,
        "expected_by_chance": round(expected, 2),
        "lift": round(num_fish / max(expected, 1e-6), 2),
        "fisher_p_value": float(f"{pval:.3e}"),
        "reverse_fish_to_numeral": fish_num,
        "by_numeric_value_before_fish": dict(sorted(val_counts.items())),
        "top_numeral_fish_pairs": [
            {"pair": f"{a} ({SHORT_NUMERALS[a]}) -> {b}", "count": int(c)}
            for (a, b), c in pair_counts.most_common(15)
        ],
    }


def test_archaeological_domains(df: pd.DataFrame) -> dict[str, Any]:
    """Check which signs spike in Fire, Earth/Plant, Sky/Cult, Water, and Air archaeological contexts."""
    domains: dict[str, pd.Series] = {
        "ATES_KALIBANGAN_FIRE_ALTAR_CITY": df["site"] == "Kalibangan",
        "ATES_COPPER_METAL_KILN": df["object_type"].isin(["TAB:C"]) | df["material"].str.contains("Copper|Bronze|Metal", case=False, na=False),
        "ATES_FAIENCE_VITREOUS_KILN": df["material"].str.contains("Faience|Paste|Vitreous|Stoneware", case=False, na=False),
        "TOPRAK_PLANT_PHYT_MOTIF": df["symbol"].str.contains("Phyt|Tree|Leaf", case=False, na=False) | df["cult"].str.contains("Plant", case=False, na=False),
        "GOK_CULT_STAND_BRAZIER_SAN_SAF": df["cult"].isin(["SAN", "SAF", "RAN", "RAF", "RBF", "HAF", "HAN"]),
        "GOK_TROUGH_OFFERING": df["cult"] == "Trough",
        "SU_GHARIAL_FISH_MOTIF": df["symbol"].isin(["Gavi", "Fish"]),
        "SU_POTTERY_VESSELS": df["object_type"].str.startswith("POT"),
    }

    out = {}
    for dname, mask in domains.items():
        dedup = df.assign(_in=mask.astype(int)).drop_duplicates(subset=["_in", "clean_reading_seq"])
        in_df = dedup[dedup["_in"] == 1]
        out_df = dedup[dedup["_in"] == 0]
        n_in, n_out = len(in_df), len(out_df)
        c_in = Counter(t for s in in_df["clean_reading_seq"] for t in set(s.split()))
        c_out = Counter(t for s in out_df["clean_reading_seq"] for t in set(s.split()))

        hits = []
        for sign, a in c_in.items():
            if a < 4:
                continue
            b = n_in - a
            c = c_out[sign]
            d = n_out - c
            p_in = a / max(n_in, 1)
            p_out = c / max(n_out, 1)
            lift = p_in / max(p_out, 1e-6)
            if lift < 1.8:
                continue
            _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
            if pval < 0.005:
                hits.append(
                    {
                        "sign_id": sign,
                        "in": f"{a}/{n_in} ({100*p_in:.1f}%)",
                        "out_pct": round(100 * p_out, 2),
                        "lift": round(lift, 2),
                        "p_value": float(f"{pval:.3e}"),
                    }
                )
        hits.sort(key=lambda x: x["p_value"])
        out[dname] = {"n_unique_docs": int(n_in), "top_signs": hits[:10]}
    return out


def test_element_interaction_network(unique_seqs: list[list[str]]) -> dict[str, Any]:
    """Compute Family-A -> Family-B transition counts and PMI across all 7 elemental families."""
    sign_to_fam = {}
    for fam, signs in ELEMENT_FAMILIES.items():
        for s in signs:
            sign_to_fam[s] = fam

    fam_bi = Counter()
    fam_left = Counter()
    fam_right = Counter()
    total_tagged_bi = 0
    top_cross_pairs = defaultdict(Counter)

    for seq in unique_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            fa = sign_to_fam.get(a)
            fb = sign_to_fam.get(b)
            if fa and fb:
                total_tagged_bi += 1
                fam_bi[(fa, fb)] += 1
                fam_left[fa] += 1
                fam_right[fb] += 1
                top_cross_pairs[(fa, fb)][f"{a}->{b}"] += 1

    links = []
    for (fa, fb), cnt in fam_bi.items():
        if cnt < 10:
            continue
        p_ab = cnt / total_tagged_bi
        p_a = fam_left[fa] / total_tagged_bi
        p_b = fam_right[fb] / total_tagged_bi
        pmi = math.log2(p_ab / (p_a * p_b))
        links.append(
            {
                "from": fa,
                "to": fb,
                "count": int(cnt),
                "pmi": round(pmi, 3),
                "examples": top_cross_pairs[(fa, fb)].most_common(4),
            }
        )
    links.sort(key=lambda x: -x["pmi"])
    return {
        "total_element_to_element_bigrams": total_tagged_bi,
        "strongest_positive_element_links": links[:15],
        "strongest_repulsions_avoided_pairs": list(reversed(links[-8:])),
    }


def main() -> None:
    ins_df = pd.read_csv(PROC_DIR / "inscriptions_clean.csv", dtype=str).fillna("")
    lines_df = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    lines_df["is_strict_analytic"] = lines_df["is_strict_analytic"].astype(int)

    df = lines_df.merge(ins_df[["inscription_id", "cult", "area_section"]], on="inscription_id", how="left")
    df = df[df["clean_reading_seq"].str.len() > 0].copy()

    strict = df[(df["is_strict_analytic"] == 1) & (df["direction"].isin(["R/L", "L/R", "BUS"]))]
    unique_seqs = [s.split() for s in strict["clean_reading_seq"].unique()]

    num_fish = test_numeral_fish(unique_seqs)
    arch = test_archaeological_domains(df)
    net = test_element_interaction_network(unique_seqs)

    # Also inspect the top 5 most frequent signs & their neighbors inside each of the 7 families
    uni = Counter(t for seq in unique_seqs for t in seq)
    left_nb = defaultdict(Counter)
    right_nb = defaultdict(Counter)
    pos_c = defaultdict(Counter)
    for seq in unique_seqs:
        n = len(seq)
        for i, t in enumerate(seq):
            pos_c[t]["init" if i == 0 else ("fin" if i == n - 1 else "med")] += 1
            if i > 0:
                left_nb[t][seq[i - 1]] += 1
            if i < n - 1:
                right_nb[t][seq[i + 1]] += 1

    fam_profiles = {}
    for fam, signs in ELEMENT_FAMILIES.items():
        top_s = sorted(signs, key=lambda s: -uni[s])[:6]
        fam_profiles[fam] = [
            {
                "sign": s,
                "freq_unique": int(uni[s]),
                "pos": dict(pos_c[s]),
                "top_left": left_nb[s].most_common(3),
                "top_right": right_nb[s].most_common(3),
            }
            for s in top_s
            if uni[s] > 0
        ]

    res = {
        "test1_numeral_fish_star": num_fish,
        "test2_archaeological_domains": arch,
        "test3_element_interaction_network": net,
        "family_top_sign_profiles": fam_profiles,
    }
    (REPORTS_DIR / "layer3b_elements_findings.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== TEST 1: NUMERAL -> FISH (STAR / CONSTELLATION *MIN TEST) ===")
    print(json.dumps(num_fish, indent=2))

    print("\n=== TEST 2: ARCHAEOLOGICAL DOMAIN ENRICHMENT (TOP 5 PER DOMAIN) ===")
    for k, v in arch.items():
        print(f"-- {k} (n={v['n_unique_docs']}) --")
        for h in v["top_signs"][:5]:
            print(f"   Sign {h['sign_id']} | in={h['in']} vs out={h['out_pct']}% | lift={h['lift']}x | p={h['p_value']}")

    print("\n=== TEST 3: 7x7 ELEMENT INTERACTION NETWORK (TOP POSITIVE LINKS) ===")
    for lk in net["strongest_positive_element_links"][:10]:
        print(f"  {lk['from']:20s} -> {lk['to']:20s} | n={lk['count']:4d} | PMI={lk['pmi']:+.3f} | ex={lk['examples'][:3]}")
    print("AVOIDED / REPULSION PAIRS:")
    for lk in net["strongest_repulsions_avoided_pairs"][:5]:
        print(f"  {lk['from']:20s} -> {lk['to']:20s} | n={lk['count']:4d} | PMI={lk['pmi']:+.3f}")


if __name__ == "__main__":
    main()
