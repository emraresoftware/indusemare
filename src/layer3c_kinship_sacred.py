#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 3C: Kinship, Gender, Lineage & Sacred/Deity Structural Probe
=================================================================================
Tests Emre's core-value concepts strictly from corpus structure & archaeology:
  (anne, baba, kardes, cocuk/ogul, kiz/kadin, erkek, ata/dede, Allah/Tanri)
4 independent, zero-hallucination tests:
  Test 1: Binary Opposition / Mutual-Exclusion Matrix of Top Terminal Signs
          (e.g., 740 Jar vs 400 Comb vs 520 Arrow vs 090 Man vs 151/156 Yoke-Man)
  Test 2: Mutual-Exclusion Matrix of the 5 Pure-Initial Openers (817, 820, 861, 920, 692)
          — testing if they form a closed paradigm of Deity/Clan/Honorific determinatives
  Test 3: Relational Connector Finder ("-in / oglu / kizi / ile")
          — Medial signs with maximum BOTH-SIDED neighbor entropy min(H_left, H_right)
  Test 4: Archaeology x Value Mapping:
          - BNGL (50 stoneware bangles -> female/elite personal ornament!)
          - IMPL / TAB:C (tools/weapons/copper -> male/craft!)
          - Scene / Anth / Comp / Mult (deity/ritual/mythic scenes -> Tanri/Kutsal!)
          - Human-figure signs (090..161) full distribution & co-occurrence!
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


def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values())


def mutual_exclusion_matrix(unique_seqs: list[list[str]], signs: list[str]) -> list[dict[str, Any]]:
    """Test pairwise co-occurrence vs expectation (odds ratio / lift < 1 = avoidance/opposition!)."""
    n = len(unique_seqs)
    doc_sets = [set(s) for s in unique_seqs]
    counts = {sg: sum(sg in ds for ds in doc_sets) for sg in signs}
    rows = []
    for i in range(len(signs)):
        for j in range(i + 1, len(signs)):
            sa, sb = signs[i], signs[j]
            both = sum((sa in ds and sb in ds) for ds in doc_sets)
            ca, cb = counts[sa], counts[sb]
            exp = (ca * cb) / max(n, 1)
            ratio = both / max(exp, 1e-6)
            # Two-sided Fisher exact
            a = both
            b = ca - both
            c = cb - both
            d = n - ca - cb + both
            _, p_less = fisher_exact([[a, b], [c, d]], alternative="less")
            _, p_gt = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "pair": f"{sa} vs {sb}",
                    "count_a": ca,
                    "count_b": cb,
                    "both_obs": both,
                    "both_exp": round(exp, 2),
                    "obs_exp_ratio": round(ratio, 3),
                    "p_avoid": float(f"{p_less:.3e}"),
                    "p_attract": float(f"{p_gt:.3e}"),
                }
            )
    rows.sort(key=lambda x: x["p_avoid"])
    return rows


def find_relational_connectors(unique_seqs: list[list[str]]) -> list[dict[str, Any]]:
    """Find medial signs that bridge diverse left and right elements (e.g. 'of / son of / and')."""
    left_nb = defaultdict(Counter)
    right_nb = defaultdict(Counter)
    pos_c = defaultdict(Counter)
    totals = Counter()
    for seq in unique_seqs:
        n = len(seq)
        for i, tok in enumerate(seq):
            totals[tok] += 1
            pos_c[tok]["init" if i == 0 else ("fin" if i == n - 1 else "med")] += 1
            if i > 0:
                left_nb[tok][seq[i - 1]] += 1
            if i < n - 1:
                right_nb[tok][seq[i + 1]] += 1

    res = []
    for s, tot in totals.items():
        if tot < 30:
            continue
        p_med = pos_c[s]["med"] / tot
        if p_med < 0.55:
            continue
        hl = entropy(left_nb[s])
        hr = entropy(right_nb[s])
        res.append(
            {
                "sign": s,
                "freq": int(tot),
                "p_init": round(pos_c[s]["init"] / tot, 3),
                "p_med": round(p_med, 3),
                "p_fin": round(pos_c[s]["fin"] / tot, 3),
                "H_left_bits": round(hl, 3),
                "H_right_bits": round(hr, 3),
                "min_H_bridge": round(min(hl, hr), 3),
                "unique_left": len(left_nb[s]),
                "unique_right": len(right_nb[s]),
                "top_left": left_nb[s].most_common(3),
                "top_right": right_nb[s].most_common(3),
            }
        )
    res.sort(key=lambda x: -x["min_H_bridge"])
    return res


def domain_enrichment(df: pd.DataFrame, mask: pd.Series, min_hits: int = 3) -> list[dict[str, Any]]:
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
        if lift < 1.5:
            continue
        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        if pval < 0.05:
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
    df = lines.merge(ins[["inscription_id", "cult", "material", "shape"]], on="inscription_id", how="left", suffixes=("", "_ins"))
    df = df[df["clean_reading_seq"].str.len() > 0].copy()

    strict = df[(df["is_strict_analytic"] == 1) & (df["direction"].isin(["R/L", "L/R", "BUS"]))]
    unique_seqs = [s.split() for s in strict["clean_reading_seq"].unique()]

    # Test 1: Terminal Oppositions (740 Jar, 400 Comb, 520 Arrow, 090 Man, 151/156 Yoke-Man, 617 Double-Field, 700 Basin)
    term_signs = ["740", "400", "520", "090", "151", "156", "617", "700"]
    term_ex = mutual_exclusion_matrix(unique_seqs, term_signs)

    # Test 2: Pure Initial Openers (817, 820, 861, 920, 692)
    init_signs = ["817", "820", "861", "920", "692"]
    init_ex = mutual_exclusion_matrix(unique_seqs, init_signs)

    # Test 3: Relational Connectors ("son of / of / kinship link")
    connectors = find_relational_connectors(unique_seqs)

    # Test 4: Value-linked Archaeological Contexts
    contexts = {
        "BANGLES_BNGL_PERSONAL_ORNAMENT": df["object_type"] == "BNGL",
        "IVORY_BONE_RODS_ROD": df["object_type"] == "ROD",
        "TOOLS_WEAPONS_IMPL": df["object_type"] == "IMPL",
        "DEITY_RITUAL_SCENES_Scene_Anth": df["symbol"].isin(["Scene", "Anth"]),
        "MYTHIC_COMPOSITE_DEITY_Comp_Mult": df["symbol"].isin(["Comp", "CompBull", "Mult"]),
        "TIGR_ELEP_RHIN_POWER_ANIMALS": df["symbol"].isin(["Tigr", "Elep", "Rhin"]),
    }
    ctx_hits = {}
    for cname, mask in contexts.items():
        h = domain_enrichment(df, mask, min_hits=2)
        ctx_hits[cname] = {"lines": int(mask.sum()), "top": h[:10]}

    # Also check ALL inscriptions on BNGL (bangles) and Scene/Anth specifically for 740 vs 400 vs human signs!
    bngl_sub = df[df["object_type"] == "BNGL"]
    bngl_seqs = bngl_sub[["inscription_id", "site", "clean_reading_seq"]].drop_duplicates().head(25).to_dict("records")

    # Human signs family (090..161) co-occurrence & slot summary
    human_signs = ["090", "091", "095", "097", "100", "104", "110", "111", "125", "130", "136", "137", "140", "142", "151", "154", "156", "158", "161"]
    human_stats = []
    for hs in human_signs:
        c = sum(hs in s for s in unique_seqs)
        if c == 0:
            continue
        p_i = sum(s[0] == hs for s in unique_seqs) / c
        p_f = sum(s[-1] == hs for s in unique_seqs) / c
        with_740 = sum((hs in s and "740" in s) for s in unique_seqs)
        with_400 = sum((hs in s and "400" in s) for s in unique_seqs)
        human_stats.append({
            "sign": hs, "docs": c, "p_init": round(p_i, 2), "p_final": round(p_f, 2),
            "with_740": with_740, "with_400": with_400
        })
    human_stats.sort(key=lambda x: -x["docs"])

    out = {
        "test1_terminal_oppositions": term_ex[:15],
        "test2_initial_openers_exclusion": init_ex,
        "test3_relational_connectors": connectors[:15],
        "test4_archaeology_contexts": ctx_hits,
        "bngl_samples": bngl_seqs,
        "human_signs_profile": human_stats,
    }
    (REPORTS_DIR / "layer3c_kinship_sacred.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== TEST 1: TOP TERMINAL MUTUAL EXCLUSIONS (BINARY OPPOSITIONS!) ===")
    for r in term_ex[:12]:
        print(f"  {r['pair']:12s} | obs={r['both_obs']:3d} vs exp={r['both_exp']:5.1f} (ratio={r['obs_exp_ratio']:.2f}) | p_avoid={r['p_avoid']}")

    print("\n=== TEST 2: 5 PURE-INITIAL OPENERS (DEITY/CLAN DETERMINATIVES?) ===")
    for r in init_ex:
        print(f"  {r['pair']:12s} | obs={r['both_obs']:3d} vs exp={r['both_exp']:5.1f} (ratio={r['obs_exp_ratio']:.2f}) | p_avoid={r['p_avoid']}")

    print("\n=== TEST 3: TOP RELATIONAL CONNECTORS ('-in / oglu / kizi / ile' BRIDGE SIGNS!) ===")
    for r in connectors[:10]:
        print(f"  Sign {r['sign']} (n={r['freq']:3d}, med={r['p_med']:.2f}) | H_L={r['H_left_bits']:.2f} H_R={r['H_right_bits']:.2f} minH={r['min_H_bridge']:.2f} | L={r['top_left'][:2]} R={r['top_right'][:2]}")

    print("\n=== TEST 4: VALUE CONTEXTS (BANGLES, RODS, TOOLS, DEITY SCENES) ===")
    for cname, v in ctx_hits.items():
        print(f"-- {cname} (lines={v['lines']}) --")
        for h in v["top"][:5]:
            print(f"   Sign {h['sign']} | in={h['in']} vs out={h['out_pct']}% | lift={h['lift']}x | p={h['p_value']}")

    print("\n=== HUMAN FIGURE FAMILY (090..161) vs 740 (JAR) & 400 (COMB) ===")
    for h in human_stats[:12]:
        print(f"  Sign {h['sign']} | docs={h['docs']:3d} | init={h['p_init']:.2f} fin={h['p_final']:.2f} | with_740={h['with_740']:2d} | with_400={h['with_400']:2d}")


if __name__ == "__main__":
    main()
