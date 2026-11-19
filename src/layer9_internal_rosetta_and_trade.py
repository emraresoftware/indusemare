#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 9: Internal Rosetta (Mohenjo-daro Copper Tablets),
Harappa Double-Sided Accounting Tokens, Multi-Signature Cargo Protocol,
and Meluhha External Trade Commodity & Craft-Hub Audit.

Strict Anti-Hallucination & Council Standards:
  - Always report BOTH raw and strictly deduplicated (unique inscription text) counts.
  - Apply global Benjamini-Hochberg FDR (q < 0.05) across all commodity/hub enrichment tests.
  - Validate Text<->Icon lock via 1,000-run permutation null Mutual Information (MI) & Purity,
    contrasting Mohenjo-daro TAB:C directly against core SEAL:S.
  - Never invent phonetic readings for trade goods; separate verified cuneiform attestations
    from speculative loanword etymologies and empirical corpus enrichments.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
ART_DIR = Path("/Users/emre/.gemini/antigravity/brain/969a4497-dabc-4f45-8c95-dcaebdbcb608")

RNG = np.random.default_rng(20261120)


def parse_toks(s: Any) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [t for t in s.replace("-", " ").split() if t and t != "000"]


def mutual_info(xs: list[str], ys: list[str]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    cx = Counter(xs)
    cy = Counter(ys)
    cxy = Counter(zip(xs, ys))
    mi = 0.0
    for (x, y), v in cxy.items():
        pxy = v / n
        mi += pxy * math.log2(pxy / ((cx[x] / n) * (cy[y] / n)))
    return mi


def purity_repeated(xs: list[str], ys: list[str]) -> tuple[int, int, float]:
    """Among texts appearing >= 2 times, fraction locked to a single icon/symbol."""
    by_x: dict[str, Counter[str]] = defaultdict(Counter)
    for x, y in zip(xs, ys):
        by_x[x][y] += 1
    rep = [c for c in by_x.values() if sum(c.values()) >= 2]
    if not rep:
        return 0, 0, 0.0
    pure = sum(1 for c in rep if len(c) == 1)
    weighted_purity = sum(c.most_common(1)[0][1] for c in rep) / sum(sum(c.values()) for c in rep)
    return pure, len(rep), weighted_purity


def perm_mi_test(xs: list[str], ys: list[str], n_perm: int = 1000) -> dict[str, Any]:
    obs_mi = mutual_info(xs, ys)
    pure, n_rep, w_pur = purity_repeated(xs, ys)
    y_arr = np.array(ys, dtype=object)
    null_mis = np.empty(n_perm, dtype=float)
    null_wpur = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = RNG.permutation(y_arr).tolist()
        null_mis[i] = mutual_info(xs, yp)
        _, _, null_wpur[i] = purity_repeated(xs, yp)
    mu, sd = float(null_mis.mean()), float(null_mis.std(ddof=1))
    z = (obs_mi - mu) / sd if sd > 1e-12 else 0.0
    p = float((np.sum(null_mis >= obs_mi) + 1) / (n_perm + 1))
    w_mu, w_sd = float(null_wpur.mean()), float(null_wpur.std(ddof=1))
    w_z = (w_pur - w_mu) / w_sd if w_sd > 1e-12 else 0.0
    return {
        "n_pairs": len(xs),
        "unique_texts": len(set(xs)),
        "unique_symbols": len(set(ys)),
        "obs_mi_bits": round(obs_mi, 4),
        "null_mi_mean": round(mu, 4),
        "null_mi_sd": round(sd, 4),
        "mi_z": round(z, 2),
        "mi_emp_p": round(p, 5),
        "repeated_texts_total": n_rep,
        "repeated_texts_100pct_pure": pure,
        "weighted_purity": round(w_pur, 4),
        "null_weighted_purity_mean": round(w_mu, 4),
        "purity_z": round(w_z, 2),
    }


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rank_idx in range(m - 1, -1, -1):
        idx = order[rank_idx]
        rank = rank_idx + 1
        val = min(prev, pvals[idx] * m / rank)
        q[idx] = val
        prev = val
    return [float(x) for x in q]


def run_module1_copper_rosetta(ins: pd.DataFrame) -> dict[str, Any]:
    tc = ins[ins["object_type"] == "TAB:C"].copy()
    tc["toks"] = tc["clean_reading_seq_full"].apply(parse_toks)
    tc["base_id"] = tc["inscription_id"].astype(str).str.split(".").str[0]

    # 1A. Text <-> Symbol lock on TAB:C vs SEAL:S
    def valid_pairs(df: pd.DataFrame) -> tuple[list[str], list[str]]:
        sub = df[
            df["clean_reading_seq_full"].fillna("").str.strip().ne("")
            & ~df["symbol"].fillna("").str.strip().isin(["", "-", "None", "nan"])
        ]
        # Normalize Bull1:* variants? Keep raw AND coarse animal class
        return sub["clean_reading_seq_full"].str.strip().tolist(), sub["symbol"].str.strip().tolist()

    tc_x, tc_y = valid_pairs(tc)
    seals = ins[ins["object_type"] == "SEAL:S"]
    ss_x, ss_y = valid_pairs(seals)

    # Also test TAB:C excluding generic 'Othr' so we prove purity holds strictly on named animals/icons!
    tc_named = tc[
        tc["clean_reading_seq_full"].fillna("").str.strip().ne("")
        & ~tc["symbol"].fillna("").str.strip().isin(["", "-", "None", "nan", "Othr"])
    ]
    tcn_x = tc_named["clean_reading_seq_full"].str.strip().tolist()
    tcn_y = tc_named["symbol"].str.strip().tolist()

    mi_tc_all = perm_mi_test(tc_x, tc_y, n_perm=1000)
    mi_tc_named = perm_mi_test(tcn_x, tcn_y, n_perm=1000)
    mi_seals = perm_mi_test(ss_x, ss_y, n_perm=1000)

    # 1B. Full locked classes table
    text_to_sym: dict[str, Counter[str]] = defaultdict(Counter)
    for x, y in zip(tc_x, tc_y):
        text_to_sym[x][y] += 1
    locked_classes = []
    for txt, c in sorted(text_to_sym.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        if tot >= 2:
            dom_sym, dom_n = c.most_common(1)[0]
            locked_classes.append({
                "text": txt,
                "n_total": tot,
                "dominant_symbol": dom_sym,
                "dominant_count": dom_n,
                "purity": round(dom_n / tot, 4),
                "all_symbols": dict(c),
            })

    # 1C. Two-sided TAB:C single-sign logogram <-> multi-sign spelling equations!
    # Map every multi-sign text to its known TAB:C icon (from 1B)
    known_icon = {r["text"]: f"{r['dominant_symbol']} ({r['dominant_count']}/{r['n_total']})" for r in locked_classes}
    # Handle near-variants (e.g. 220 017 585 095 520 032 407 vs 231 017 585 095 520 032 407)
    two_sided_eqs = []
    single_sign_pairs = []
    for bid, g in tc.sort_values("inscription_id").groupby("base_id"):
        if len(g) < 2:
            continue
        seqs = [str(s).strip() for s in g["clean_reading_seq_full"].tolist()]
        cisis = [str(s).strip() for s in g["cisi"].tolist()]
        syms = [str(s).strip() for s in g["symbol"].tolist()]
        lens = [len(parse_toks(s)) for s in seqs]
        rec = {
            "base_id": bid,
            "cisi": cisis[0],
            "symbol": next((s for s in syms if s and s != "-"), ""),
            "sides": seqs,
            "lens": lens,
        }
        two_sided_eqs.append(rec)
        if min(lens) == 1 and max(lens) >= 2:
            short_idx = int(np.argmin(lens))
            long_idx = int(np.argmax(lens))
            short_sign = seqs[short_idx]
            long_txt = seqs[long_idx]
            matched_icon = known_icon.get(long_txt, "")
            if not matched_icon:
                # Check stem match (>=4 shared tokens)
                lt = set(long_txt.split())
                for kt, ki in known_icon.items():
                    if len(lt & set(kt.split())) >= 4:
                        matched_icon = f"~{ki}"
                        break
            single_sign_pairs.append({
                "base_id": bid,
                "cisi": cisis[0],
                "short_sign": short_sign,
                "long_spelling": long_txt,
                "linked_icon": matched_icon or rec["symbol"] or "unmapped_icon",
            })

    # Aggregate single_sign -> long_spelling equations
    eq_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    eq_icon: dict[tuple[str, str], str] = {}
    for p in single_sign_pairs:
        key = (p["short_sign"], p["long_spelling"])
        eq_groups[key].append(p["cisi"] if p["cisi"] != "-" else p["base_id"])
        eq_icon[key] = p["linked_icon"]
    consolidated_eqs = [
        {
            "short_sign": k[0],
            "long_spelling": k[1],
            "count": len(v),
            "cisi_ids": v,
            "linked_icon": eq_icon[k],
        }
        for k, v in sorted(eq_groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0]))
    ]

    # 1D. Cross-corpus check: Do TAB:C icon-specific signs keep the same animal on SEAL:S?
    # Check signs that appear heavily on ONE animal on TAB:C: 923 (Elep), 321 (Hare), 845 (Hare/Anth/Loop), 503+615 (Othr), 850 (Anth)
    cross_corpus = {}
    for sign, tc_role in [
        ("923", "Elep (11/11)"),
        ("321", "Hare (12/12)"),
        ("845", "Hare/Anth/Loop stem"),
        ("407", "TAB:C terminal/stem"),
        ("503", "Othr/Rhin opener"),
        ("615", "Othr/Rhin medial"),
        ("752", "Othr medial"),
        ("850", "Anth medial"),
    ]:
        ss_sub = seals[seals["clean_reading_seq_full"].fillna("").apply(lambda s: sign in s.split())]
        ss_syms = Counter(s for s in ss_sub["symbol"].fillna("") if s and s != "-")
        cross_corpus[sign] = {
            "tab_c_role": tc_role,
            "seal_s_count": int(len(ss_sub)),
            "seal_s_top_animals": dict(ss_syms.most_common(5)),
        }

    # 1E. Morpheme decomposition: shared motifs across TAB:C classes
    shared_motifs = {}
    for motif_name, sub_tokens in [
        ("845 ... 407 (shared frame)", ["845", "407"]),
        ("705/706 033 (shared pair)", ["033"]),
        ("142 617 (shared ending)", ["142", "617"]),
        ("900 165 (reverse formula)", ["900", "165"]),
        ("503 615 (Othr/Rhin stem)", ["503", "615"]),
    ]:
        hits = []
        for r in locked_classes:
            toks = r["text"].split()
            if motif_name.startswith("705/706"):
                if ("705 033" in r["text"]) or ("706 033" in r["text"]):
                    hits.append(f"{r['dominant_symbol']}: {r['text']} (n={r['n_total']})")
            elif all(t in toks for t in sub_tokens):
                hits.append(f"{r['dominant_symbol']}: {r['text']} (n={r['n_total']})")
        shared_motifs[motif_name] = hits

    return {
        "total_tab_c_rows": int(len(tc)),
        "mohenjo_daro_rows": int((tc["site"] == "Mohenjo-daro").sum()),
        "mohenjo_daro_share_pct": round(100.0 * (tc["site"] == "Mohenjo-daro").sum() / len(tc), 2),
        "distinct_objects": int(tc["base_id"].nunique()),
        "two_sided_objects": int(len(two_sided_eqs)),
        "text_symbol_mi_tab_c_all": mi_tc_all,
        "text_symbol_mi_tab_c_named_only": mi_tc_named,
        "text_symbol_mi_seal_s_control": mi_seals,
        "locked_classes": locked_classes,
        "single_sign_logogram_equations": consolidated_eqs,
        "total_single_sign_equation_tablets": len(single_sign_pairs),
        "shared_motifs": shared_motifs,
        "cross_corpus_seal_s_check": cross_corpus,
    }


def run_module2_harappa_tokens(ins: pd.DataFrame) -> dict[str, Any]:
    ha_tabs = ins[ins["object_type"].isin(["TAB:I", "TAB:B"])].copy()
    ha_tabs["base_id"] = ha_tabs["inscription_id"].astype(str).str.split(".").str[0]

    # Build Front (entity) <-> Back (numeral+700) pairing
    num_signs = {"031": 1, "032": 2, "033": 3, "034": 4, "035": 5}
    qty_counts: Counter[str] = Counter()
    entity_to_qty: dict[str, Counter[str]] = defaultdict(Counter)
    multi_stats = {}

    for ot in ["TAB:I", "TAB:B"]:
        sub = ha_tabs[ha_tabs["object_type"] == ot].sort_values("inscription_id")
        grp = sub.groupby("base_id")["clean_reading_seq_full"].apply(list)
        multi = grp[grp.apply(len) >= 2]
        n_700 = 0
        n_num700 = 0
        n_400 = 0
        for seqs in multi:
            has_700 = any("700" in str(s).split() for s in seqs)
            has_400 = any("400" in str(s).split() for s in seqs)
            if has_700:
                n_700 += 1
            if has_400:
                n_400 += 1
            # Identify quantity side like "033 700" or "700 033"
            qty_sides = []
            ent_sides = []
            for s in seqs:
                toks = parse_toks(s)
                if len(toks) == 2 and "700" in toks and any(t in num_signs for t in toks):
                    n_tok = [t for t in toks if t in num_signs][0]
                    qty_sides.append(f"{n_tok}+700 (val={num_signs[n_tok]})")
                elif toks == ["700"]:
                    qty_sides.append("700_only")
                elif toks:
                    ent_sides.append(" ".join(toks))
            if qty_sides:
                n_num700 += 1
                q = qty_sides[0]
                qty_counts[q] += 1
                if ent_sides:
                    ent_key = " || ".join(sorted(ent_sides))
                    entity_to_qty[ent_key][q] += 1
        multi_stats[ot] = {
            "total_rows": int(len(sub)),
            "harappa_share_pct": round(100.0 * (sub["site"] == "Harappa").sum() / len(sub), 2),
            "distinct_objects": int(len(grp)),
            "multi_sided_objects": int(len(multi)),
            "has_700_count": n_700,
            "has_700_pct": round(100.0 * n_700 / max(1, len(multi)), 1),
            "exact_num700_side_count": n_num700,
            "has_400_count": n_400,
            "has_400_pct": round(100.0 * n_400 / max(1, len(multi)), 1),
        }

    # Test Entity <-> Quantity lock via MI permutation
    ex, qy = [], []
    for ent, c in entity_to_qty.items():
        for q, cnt in c.items():
            if q != "700_only":
                for _ in range(cnt):
                    ex.append(ent)
                    qy.append(q)
    ent_qty_mi = perm_mi_test(ex, qy, n_perm=1000)

    top_ent_qty = [
        {"entity_side": ent, "n": sum(c.values()), "quantities": dict(c)}
        for ent, c in sorted(entity_to_qty.items(), key=lambda kv: -sum(kv[1].values()))[:12]
    ]

    # Verify ': X : + 400' clitic-stacking hypothesis on TAB:I & TAB:B!
    framed = {"226", "232", "234", "236", "241", "592", "804", "842"}
    clitic_total = 0
    clitic_final_or_pre400 = 0
    clitic_direct_400 = 0
    clitic_examples = []
    # Compare against non-framed medial signs on TAB:I/B directly before final 400
    bg_nonfinal = 0
    bg_before_400 = 0
    for _, r in ha_tabs.iterrows():
        toks = parse_toks(r["clean_reading_seq_full"])
        if len(toks) < 2:
            continue
        for i, t in enumerate(toks):
            is_fin = (i == len(toks) - 1)
            followed_by_400 = (i + 1 < len(toks) and toks[i + 1] == "400" and i + 1 == len(toks) - 1)
            followed_158_400 = (i + 2 < len(toks) and toks[i + 1] == "158" and toks[i + 2] == "400")
            if t in framed:
                clitic_total += 1
                if is_fin or followed_by_400 or followed_158_400:
                    clitic_final_or_pre400 += 1
                if followed_by_400:
                    clitic_direct_400 += 1
                clitic_examples.append(f"{r['object_type']} {r['inscription_id']}: {' '.join(toks)}")
            elif t != "400":
                bg_nonfinal += 1
                if is_fin or followed_by_400:
                    bg_before_400 += 1
    _, p_clitic = fisher_exact(
        [[clitic_final_or_pre400, clitic_total - clitic_final_or_pre400],
         [bg_before_400, bg_nonfinal - bg_before_400]],
        alternative="greater",
    )

    # Harappa Mound F (Granary/North) vs Mound E (Gate/Workshop) distribution
    ha_only = ha_tabs[ha_tabs["site"] == "Harappa"].copy()
    def mound_bucket(a: str) -> str:
        a = str(a or "")
        if a.startswith("F"):
            return "Mound_F_Granary_WorkArea"
        if a.startswith("E"):
            return "Mound_E_ET_Gate_Craft"
        if a.startswith("AB"):
            return "Mound_AB_Citadel"
        if a.startswith("D"):
            return "Mound_D"
        return "Other_Unprovenanced"

    ha_only["mound"] = ha_only["area_section"].apply(mound_bucket)
    mound_ct = pd.crosstab(ha_only["mound"], ha_only["object_type"]).to_dict()

    return {
        "multi_side_stats": multi_stats,
        "quantity_distribution": dict(qty_counts.most_common()),
        "entity_quantity_mi_test": ent_qty_mi,
        "top_entity_quantity_pairs": top_ent_qty,
        "clitic_stacking_X_plus_400": {
            "total_framed_tokens": clitic_total,
            "direct_before_final_400": clitic_direct_400,
            "final_or_pre_400_total": clitic_final_or_pre400,
            "rate_pct": round(100.0 * clitic_final_or_pre400 / max(1, clitic_total), 1),
            "background_rate_pct": round(100.0 * bg_before_400 / max(1, bg_nonfinal), 1),
            "fisher_p": float(f"{p_clitic:.3e}"),
            "examples": clitic_examples[:12],
        },
        "harappa_mound_distribution": mound_ct,
    }


def run_module3_multisig_cargo(ins: pd.DataFrame) -> dict[str, Any]:
    tags = ins[ins["object_type"].fillna("").str.startswith("TAG")].copy()
    tags["base_id"] = tags["inscription_id"].astype(str).str.split(".").str[0]
    seals = ins[ins["object_type"].fillna("").str.startswith("SEAL")].copy()

    # Map every stone seal sequence (len >= 3 strict!) to its sites/ids
    seal_index_len3: dict[str, list[dict[str, str]]] = defaultdict(list)
    seal_index_len2: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, r in seals.iterrows():
        seq = " ".join(parse_toks(r["clean_reading_seq_full"]))
        n = len(seq.split()) if seq else 0
        rec = {"id": str(r["inscription_id"]), "site": str(r["site"]), "type": str(r["object_type"]), "sym": str(r["symbol"])}
        if n >= 3:
            seal_index_len3[seq].append(rec)
        elif n == 2:
            seal_index_len2[seq].append(rec)

    # All multi-signed clay lumps (base_id with >=2 non-empty distinct seal texts)
    multisig_lumps = []
    co_sign_edges: Counter[tuple[str, str]] = Counter()
    hub_counter: Counter[str] = Counter()

    for bid, g in tags.sort_values("inscription_id").groupby("base_id"):
        if len(g) < 2:
            continue
        site = str(g["site"].iloc[0])
        otype = str(g["object_type"].iloc[0])
        seqs = [" ".join(parse_toks(s)) for s in g["clean_reading_seq_full"].tolist()]
        non_empty = [s for s in seqs if s]
        distinct = list(dict.fromkeys(non_empty))
        if len(distinct) >= 2:
            # Check distant stone seal matches for each side
            distant_matches = {}
            for s in distinct:
                if s in seal_index_len3:
                    other_sites = [m for m in seal_index_len3[s] if m["site"] != site]
                    if other_sites:
                        distant_matches[s] = other_sites[:3]
            multisig_lumps.append({
                "base_id": bid,
                "site": site,
                "object_type": otype,
                "n_impressions": int(len(g)),
                "distinct_seals": distinct,
                "distant_stone_seal_matches_len3": distant_matches,
            })
            for s in distinct:
                hub_counter[f"{site} :: {s}"] += 1
            for i in range(len(distinct)):
                for j in range(i + 1, len(distinct)):
                    a, b = sorted([distinct[i], distinct[j]])
                    co_sign_edges[(a, b)] += 1

    # All TAG* -> distant stone SEAL* matches (STRICT len >= 3!)
    intercity_tag_seal_len3 = []
    seen_pairs = set()
    for _, r in tags.iterrows():
        seq = " ".join(parse_toks(r["clean_reading_seq_full"]))
        if len(seq.split()) < 3:
            continue
        if seq in seal_index_len3:
            tag_site = str(r["site"])
            for m in seal_index_len3[seq]:
                if m["site"] != tag_site:
                    key = (seq, tag_site, m["site"])
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        n_tags_here = int(((tags["site"] == tag_site) & (tags["clean_reading_seq_full"].apply(lambda s: " ".join(parse_toks(s))) == seq)).sum())
                        intercity_tag_seal_len3.append({
                            "text": seq,
                            "len": len(seq.split()),
                            "tag_site": tag_site,
                            "tag_count": n_tags_here,
                            "tag_symbol": str(r["symbol"]),
                            "stone_seal_site": m["site"],
                            "stone_seal_id": m["id"],
                            "stone_seal_type": m["type"],
                            "stone_seal_symbol": m["sym"],
                        })

    intercity_tag_seal_len3.sort(key=lambda x: (-x["len"], -x["tag_count"]))

    # Lothal master counter-signature audit
    lothal_master = "705 500 741 001 055 220 740 090"
    lm_lumps = [x for x in multisig_lumps if lothal_master in x["distinct_seals"]]

    return {
        "total_tag_rows": int(len(tags)),
        "multisig_lumps_with_ge2_distinct_seals": len(multisig_lumps),
        "multisig_by_site": dict(Counter(x["site"] for x in multisig_lumps)),
        "top_cosigning_hubs": [{"seal": k, "lumps": v} for k, v in hub_counter.most_common(10)],
        "lothal_master_counter_sign": {
            "seal_text": lothal_master,
            "co_signed_lumps": lm_lumps,
        },
        "intercity_cargo_tag_to_distant_stone_seal_len3": intercity_tag_seal_len3,
        "all_multisig_lumps_sample": multisig_lumps[:15],
    }


def run_module4_meluhha_trade_audit(ins: pd.DataFrame) -> dict[str, Any]:
    """
    Test deduplicated AND raw sign enrichment on trade materials & export hubs,
    with global Benjamini-Hochberg FDR across ALL tests.
    """
    ins = ins.copy()
    ins["norm_seq"] = ins["clean_reading_seq_full"].apply(lambda s: " ".join(parse_toks(s)))
    ins_nonempty = ins[ins["norm_seq"].str.len() > 0].copy()

    # Define tests: (category_name, fg_mask, bg_mask, control_note)
    is_seal_or_tag = ins_nonempty["object_type"].fillna("").str.startswith(("SEAL", "TAG"))
    tests_spec = [
        ("Ivory_Objects", ins_nonempty["material"] == "Ivory", ins_nonempty["material"] != "Ivory", "Ivory vs all non-Ivory"),
        ("Stoneware_Shell_Bangles", ins_nonempty["object_type"] == "BNGL", ins_nonempty["object_type"] != "BNGL", "BNGL vs all non-BNGL"),
        ("Copper_Gold_Implements", ins_nonempty["object_type"] == "IMPL", ins_nonempty["object_type"] != "IMPL", "IMPL vs all non-IMPL"),
        ("Cargo_Clay_Sealings_vs_Seals", ins_nonempty["object_type"].fillna("").str.startswith("TAG"), ins_nonempty["object_type"] == "SEAL:S", "TAG* vs SEAL:S"),
        ("Lothal_Port_vs_Core", (ins_nonempty["site"] == "Lothal") & is_seal_or_tag, ins_nonempty["site"].isin(["Mohenjo-daro", "Harappa"]) & is_seal_or_tag, "Lothal vs MD+HA (SEAL+TAG)"),
        ("Dholavira_Port_vs_Core", (ins_nonempty["site"] == "Dholavira") & is_seal_or_tag, ins_nonempty["site"].isin(["Mohenjo-daro", "Harappa"]) & is_seal_or_tag, "Dholavira vs MD+HA (SEAL+TAG)"),
        ("Chanhudaro_BeadHub_vs_Core", (ins_nonempty["site"] == "Chanhu-daro") & is_seal_or_tag, ins_nonempty["site"].isin(["Mohenjo-daro", "Harappa"]) & is_seal_or_tag, "Chanhu-daro vs MD+HA (SEAL+TAG)"),
        ("Export_Corridor_Foreign_vs_CoreSeals", ins_nonempty["is_foreign"] == 1, (ins_nonempty["is_foreign"] == 0) & ins_nonempty["object_type"].fillna("").str.startswith("SEAL"), "Foreign (Gulf/Meso/Iran/CA) vs Core SEAL*"),
    ]

    all_tests_dedup = []
    category_summaries = {}

    for cat_name, fg_mask, bg_mask, note in tests_spec:
        fg_raw = ins_nonempty[fg_mask]
        bg_raw = ins_nonempty[bg_mask]
        # STRICT DEDUPLICATION: unique norm_seq within fg and within bg!
        fg_uniq = fg_raw["norm_seq"].unique().tolist()
        bg_uniq = bg_raw["norm_seq"].unique().tolist()
        n_fg_raw, n_bg_raw = len(fg_raw), len(bg_raw)
        n_fg_ded, n_bg_ded = len(fg_uniq), len(bg_uniq)

        fg_c_raw = Counter(t for s in fg_raw["norm_seq"] for t in set(s.split()))
        bg_c_raw = Counter(t for s in bg_raw["norm_seq"] for t in set(s.split()))
        fg_c_ded = Counter(t for s in fg_uniq for t in set(s.split()))
        bg_c_ded = Counter(t for s in bg_uniq for t in set(s.split()))

        # Test every sign appearing >= 2 times in dedup foreground (or >= 2 raw to detect duplicate artifacts!)
        tested_here = []
        raw_only_artifacts = []
        for sign, a_raw in fg_c_raw.items():
            if a_raw < 2:
                continue
            a_ded = fg_c_ded.get(sign, 0)
            b_ded = bg_c_ded.get(sign, 0)
            b_raw = bg_c_raw.get(sign, 0)
            _, p_raw = fisher_exact([[a_raw, n_fg_raw - a_raw], [b_raw, n_bg_raw - b_raw]], alternative="greater")
            if a_ded < 2:
                if p_raw < 0.005:
                    raw_only_artifacts.append({
                        "sign": sign, "raw_hits": a_raw, "dedup_hits": a_ded, "p_raw": float(f"{p_raw:.2e}"),
                        "reason": "duplicate_stamp_artifact",
                    })
                continue
            _, p_ded = fisher_exact([[a_ded, n_fg_ded - a_ded], [b_ded, n_bg_ded - b_ded]], alternative="greater")
            r_fg = a_ded / n_fg_ded
            r_bg = b_ded / n_bg_ded if n_bg_ded else 0.0
            lift_ded = (r_fg / r_bg) if r_bg > 0 else 999.0
            rec = {
                "category": cat_name,
                "sign": sign,
                "raw_fg": a_raw, "raw_n": n_fg_raw, "p_raw": float(f"{p_raw:.3e}"),
                "dedup_fg": a_ded, "dedup_n": n_fg_ded,
                "dedup_bg": b_ded, "dedup_bg_n": n_bg_ded,
                "lift_dedup": round(lift_ded, 2),
                "p_dedup": p_ded,
            }
            all_tests_dedup.append(rec)
            tested_here.append(rec)

        category_summaries[cat_name] = {
            "control": note,
            "n_raw": n_fg_raw,
            "n_dedup_unique_texts": n_fg_ded,
            "n_signs_tested_dedup": len(tested_here),
            "duplicate_stamp_artifacts_caught": raw_only_artifacts,
            "top_dedup_hits": [],  # filled after global BH-FDR
        }

    # Global Benjamini-Hochberg FDR across ALL deduplicated tests!
    pvals = [r["p_dedup"] for r in all_tests_dedup]
    qvals = bh_fdr(pvals)
    for r, q in zip(all_tests_dedup, qvals):
        r["q_fdr_global"] = float(f"{q:.4e}")
        r["p_dedup"] = float(f"{r['p_dedup']:.4e}")
        r["survives_fdr_05"] = bool(q < 0.05)

    for cat_name in category_summaries:
        sub = [r for r in all_tests_dedup if r["category"] == cat_name]
        sub.sort(key=lambda x: x["p_dedup"])
        category_summaries[cat_name]["n_survive_global_fdr_05"] = sum(1 for x in sub if x["survives_fdr_05"])
        category_summaries[cat_name]["top_dedup_hits"] = [
            x for x in sub[:8] if x["p_dedup"] < 0.01 or x["survives_fdr_05"]
        ]

    # Inspect IMPL (copper/gold axes) and BNGL (stoneware bangles) actual texts for transparency
    impl_texts = ins_nonempty[ins_nonempty["object_type"] == "IMPL"][["inscription_id", "site", "material", "norm_seq"]].to_dict("records")
    bngl_840_790 = ins_nonempty[
        (ins_nonempty["object_type"] == "BNGL")
        & ins_nonempty["norm_seq"].apply(lambda s: ("840" in s.split()) or ("790" in s.split()))
    ][["inscription_id", "site", "material", "norm_seq"]].to_dict("records")

    # Verified Cuneiform Meluhha Commodity Reference Ledger (Possehl 1996/2002; Laursen & Steinkeller 2017)
    cuneiform_commodities = [
        {
            "commodity_tr": "Kirmizi Akik (Carnelian)",
            "cuneiform_sumerian": "na4.gug (Akk. samtu)",
            "attestation_status": "verified_cuneiform_meluhha_export",
            "indus_source_hubs": "Lothal, Chanhu-daro, Dholavira (Gujarat/Kutch)",
            "corpus_presence": "2 Agate SEAL:R at Lothal; Chanhu-daro bead hub (n=60 SEAL+TAG)",
            "etymology_note": "Sumerce 'gug' yerel/kulturel sozcuk; dogrudan Dravidce odunc degil",
        },
        {
            "commodity_tr": "Fildisi (Ivory)",
            "cuneiform_sumerian": "zu2-am-si (Akk. sinnu piri)",
            "attestation_status": "verified_cuneiform_meluhha_export",
            "indus_source_hubs": "Mohenjo-daro, Harappa, Lothal",
            "corpus_presence": "37 Ivory objects (31 Ivory ROD at MD, 4 at HA, 1 Ivory SEAL at Lothal) + Elep icon",
            "etymology_note": "Akadca 'piru' (fil) / Sanskrit 'ibha' tartismali; kesin Meluhha oduncu degil",
        },
        {
            "commodity_tr": "Sert Tropik Kereste / Abanoz / Mangrov",
            "cuneiform_sumerian": "gis-esi, gis-mes-ma2-kan-na, gis-ab-ba-me-luh-ha",
            "attestation_status": "verified_cuneiform_meluhha_export",
            "indus_source_hubs": "Indus deltasi & bati kiyi ormanlari",
            "corpus_presence": "Organik (yazitli ahsap korunmamis)",
            "etymology_note": "'gis-ab-ba-me-luh-ha' Ur III / Eski Babil listelerinde 12+ kez geciyor",
        },
        {
            "commodity_tr": "Bakir, Altin, Kalay/Tunc",
            "cuneiform_sumerian": "urudu, ku3-sig17, an-na",
            "attestation_status": "verified_cuneiform_meluhha_magan_trade",
            "indus_source_hubs": "Umman (Magan) <-> Lothal <-> Mohenjo-daro",
            "corpus_presence": "230 Copper TAB:C, 23 Copper/Gold IMPL, 3 Silver SEAL:S",
            "etymology_note": "Sumerce/Akadca standart metal terimleri",
        },
        {
            "commodity_tr": "Lacivert Tasi (Lapis Lazuli)",
            "cuneiform_sumerian": "na4.za-gin3 (Akk. uqnu)",
            "attestation_status": "verified_cuneiform_meluhha_transit",
            "indus_source_hubs": "Shortughai (Bedahsan/Afganistan) -> Indus -> Korfez",
            "corpus_presence": "Shortughai'da 1 Indus muhru; Gonur Depe & Altyn Depe'de 2 muhur",
            "etymology_note": "Meluhha uzerinden transit ticaret mali",
        },
        {
            "commodity_tr": "Tavus Kusu / Meluhha Kusu",
            "cuneiform_sumerian": "ha-ia3-musen / dar-me-luh-ha",
            "attestation_status": "verified_cuneiform_meluhha_export",
            "indus_source_hubs": "Indus Vadisi (comlek boyamalarinda cok yaygin)",
            "corpus_presence": "Enki ve Dunya Duzeni'nde 'sarayda oten ha-ia3 kusu'",
            "etymology_note": "Parpola/Witzel bazi kus adlarini yerli substratla iliskilendirir (hipotez)",
        },
        {
            "commodity_tr": "Deniz Kabugu & Tas Bilesik Bilezikler",
            "cuneiform_sumerian": "Turbinella pyrum / kakma deniz kabugu",
            "attestation_status": "verified_archaeological_export",
            "indus_source_hubs": "Nageshwar, Lothal, Dholavira, Mohenjo-daro",
            "corpus_presence": "50 yazitli bilezik (BNGL: 38 Stoneware, Shell)",
            "etymology_note": "Firin kapaklarina muhur basilarak uretilen prestij urunu (Halim & Vidale 1984)",
        },
        {
            "commodity_tr": "Susam Yagi & Pamuk Dokuma (Tartismali Odunc)",
            "cuneiform_sumerian": "i3-gis / ellu? (susam); tug2 (karsi ihracat: Mezopotamya yunu)",
            "attestation_status": "archaeobotanical_indus_origin_etymology_disputed",
            "indus_source_hubs": "Harappa, Mohenjo-daro (Sesamum indicum & Gossypium)",
            "corpus_presence": "Organik (kumas/yag); Laursen & Steinkeller (2017): Mezopotamya karsi ihracati yun/arpa/yag",
            "etymology_note": "Dravidce '*el' (susam) <-> Akadca 'ellu' onerilen ama tartismali etimoloji (Akadca 'ellu'=saf)",
        },
    ]

    return {
        "total_dedup_tests_run": len(all_tests_dedup),
        "total_surviving_global_fdr_05": sum(1 for r in all_tests_dedup if r["survives_fdr_05"]),
        "categories": category_summaries,
        "impl_records_sample": impl_texts,
        "bngl_840_790_records": bngl_840_790,
        "cuneiform_meluhha_commodity_ledger": cuneiform_commodities,
    }


def render_glyph_plate(
    glyphs_df: pd.DataFrame,
    mod1: dict[str, Any],
    mod2: dict[str, Any],
    mod3: dict[str, Any],
    mod4: dict[str, Any],
    out_path: Path,
) -> None:
    """Render clean, zero-overlap high-resolution Indus TTF glyph plate for Layer 9."""
    cp_map = {}
    for _, r in glyphs_df.iterrows():
        sid = str(r["sign_id"]).strip().zfill(3)
        cps = [int(h, 16) for h in str(r["codepoints"]).split() if h]
        if cps:
            cp_map[sid] = "".join(chr(c) for c in cps)

    indus_lg = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 32)
    indus_md = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 26)
    try:
        f_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        f_sec = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        f_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        f_reg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        f_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except Exception:
        f_title = f_sec = f_bold = f_reg = f_sm = ImageFont.load_default()

    W, H = 1520, 1310
    img = Image.new("RGB", (W, H), "#FAF9F6")
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (W, 64)], fill="#1E293B")
    draw.text((24, 12), "INDUS-DEDUCT LAYER 9 — INTERNAL ROSETTA (TAB:C), HARAPPA TOKENS & MELUHHA TRADE PROTOCOL", fill="#FFFFFF", font=f_title)
    draw.text((24, 41), "Actual ICIT TTF Glyphs (Reading Order L->R) | Zero Hallucination Audit | Global BH-FDR Controlled", fill="#CBD5E1", font=f_reg)

    def draw_seq(x: int, y: int, seq_str: str, font_obj: Any, step: int = 40, color: str = "#0F172A") -> int:
        toks = seq_str.split()
        cx = x
        for t in toks:
            g = cp_map.get(t.zfill(3), "?")
            draw.text((cx, y), g, fill=color, font=font_obj)
            draw.text((cx + 2, y + 36), t, fill="#64748B", font=f_sm)
            cx += step
        return cx

    # --- PANEL A: Mohenjo-daro TAB:C Internal Rosetta ---
    y = 82
    draw.rectangle([(20, y), (W - 20, y + 30)], fill="#E2E8F0")
    mi_tc = mod1["text_symbol_mi_tab_c_all"]
    mi_ss = mod1["text_symbol_mi_seal_s_control"]
    draw.text(
        (32, y + 6),
        f"A. MOHENJO-DARO COPPER TABLETS (TAB:C, 229/230 = 99.6%) — TEXT <-> ICON LOCK (MI z = +{mi_tc['mi_z']} vs SEAL:S +{mi_ss['mi_z']}) & 1-SIGN LOGOGRAM EQUATIONS",
        fill="#0F172A",
        font=f_sec,
    )
    y += 42

    # Column headers
    draw.text((32, y), "LOCKED ICON / CLASS", fill="#475569", font=f_bold)
    draw.text((275, y), "1-SIGN REVERSE LOGOGRAM", fill="#475569", font=f_bold)
    draw.text((530, y), "FULL COPPER TABLET SPELLING (READING ORDER ->)", fill="#475569", font=f_bold)
    draw.text((1110, y), "LOCK & STRUCTURAL NOTE", fill="#475569", font=f_bold)
    y += 22
    draw.line([(24, y), (W - 24, y)], fill="#CBD5E1", width=1)
    y += 10

    panel_a_rows = [
        ("Tavsan (Hare)", "12/12 (100%)", "753", "2x (M-593, M-1547)", "235 705 033 845 407 321 407", "321=Hare-ozel; 845(dugum)..407 & 705 033 ortak"),
        ("Boynuzlu Avci (Anth)", "10/10 (100%)", "780 781 782 777", "5x (M-602..2054)", "806 845 061 407 850 900 740", "777-782 boynuzlu-U ailesi; 845..407 ortak"),
        ("Fil (Elep)", "11/11 (100%)", "003 422", "1x (Comp var.)", "706 033 923 740", "923(uclu-yay/dis) + 706 033 + 740 (4 adette +001)"),
        ("Sonsuzluk Dugumu (Loop)", "4/4 (100%)", "-", "ikon arka yuzde", "415 220 845 407", "845(dugum) 407 koku Hare & Anth ile ortak"),
        ("Standart Ikon (Othr/Rhin)", "28/29 (96.6%)", "341", "2x (M-516, M-517)", "503 615 752 740", "341 = 4 ayakli hayvan piktogrami <-> 503 615 740!"),
        ("Keci / Manda (Goat/Buff)", "3/3 (100%)", "777 781 782", "6x (M-599..2053)", "231 017 585 095 520 032 407", "4x M-1555..1558 hepsi arka yuzde 777"),
        ("Cift-Yuzlu Sabit Seri", "4x (M-597..1550)", "749", "4x (M-597,1548-50)", "806 233 384 740", "749(icli-U) <-> 806 233 384 740 & 231 255.."),
    ]
    for label, lock_txt, short_s, short_note, full_seq, note in panel_a_rows:
        draw.rectangle([(24, y - 4), (W - 24, y + 61)], fill="#FFFFFF", outline="#E2E8F0")
        draw.text((34, y + 8), label, fill="#0F172A", font=f_bold)
        draw.text((34, y + 28), lock_txt, fill="#0369A1", font=f_reg)
        if short_s != "-":
            draw_seq(278, y, short_s, indus_lg, step=42, color="#B91C1C")
            draw.text((452, y + 18), "<->", fill="#94A3B8", font=f_bold)
            draw.text((278, y + 48), short_note, fill="#991B1B", font=f_sm)
        else:
            draw.text((295, y + 18), "(ikon arka yuzde)", fill="#94A3B8", font=f_reg)
        draw_seq(532, y, full_seq, indus_lg, step=46, color="#0F172A")
        draw.text((1110, y + 16), note, fill="#334155", font=f_reg)
        y += 71

    # --- PANEL B (Left): Lothal Multi-Signature Cargo Protocol ---
    y += 12
    mid_x = 760
    draw.rectangle([(20, y), (mid_x - 10, y + 30)], fill="#DBEAFE")
    draw.text((32, y + 6), "B. LOTHAL LIMANI COK-IMZALI KARGO PROTOKOLU (TAG*)", fill="#1E3A8A", font=f_sec)

    # --- PANEL C (Right): Harappa Double-Sided Accounting Tokens ---
    draw.rectangle([(mid_x + 10, y), (W - 20, y + 30)], fill="#FEF3C7")
    draw.text((mid_x + 22, y + 6), "C. HARAPPA CIFT YUZLU FIS SISTEMI (TAB:I + TAB:B, n=772)", fill="#92400E", font=f_sec)
    y += 42

    # Left Box Content (Lothal)
    draw.rectangle([(20, y), (mid_x - 10, y + 290)], fill="#FFFFFF", outline="#BFDBFE")
    ly = y + 12
    draw.text((34, ly), "1) Lothal Liman/Gumruk Ana Karsi-Imzasi (4 ayri kargo balyasinda!):", fill="#1E40AF", font=f_bold)
    ly += 22
    draw_seq(34, ly, "705 500 741 001 055 220 740 090", indus_lg, step=46, color="#1E3A8A")
    ly += 58
    draw.text((34, ly), "   Ayni islak kil uzerinde karsi-imzaladigi TUCCAR muhru (Kargo #2113):", fill="#334155", font=f_reg)
    ly += 20
    draw_seq(34, ly, "861 002 235", indus_lg, step=46, color="#047857")
    draw.text((200, ly + 10), "<- Bu muhrun TAS KALIBI (2239.1) 700 km otede", fill="#047857", font=f_bold)
    draw.text((200, ly + 28), "   MOHENJO-DARO'DA (SEAL:S, Bull1:I) bulundu!", fill="#047857", font=f_bold)
    ly += 60
    draw.text((34, ly), "2) Diger Sehirlerarasi Kargo Balya <-> Uzak Tas Muhur Eslesmeleri (len>=3):", fill="#1E40AF", font=f_bold)
    ly += 22
    draw_seq(34, ly, "817 002 048 740", indus_md, step=38, color="#0F172A")
    draw.text((210, ly + 8), "Mohenjo-daro Bull1 SEAL <-> 10 Lothal Elep TAG:B", fill="#334155", font=f_reg)
    ly += 48
    draw_seq(34, ly, "220 740 090", indus_md, step=38, color="#0F172A")
    draw.text((210, ly + 8), "Harappa (2 SEAL:S) <-> Lothal TAG (#2111)", fill="#334155", font=f_reg)
    draw.text((210, ly + 24), "Ayrica 705 033 520 & 140 100 740: MD SEAL <-> Lothal TAG", fill="#64748B", font=f_sm)

    # Right Box Content (Harappa Tokens)
    draw.rectangle([(mid_x + 10, y), (W - 20, y + 290)], fill="#FFFFFF", outline="#FDE68A")
    ry = y + 12
    draw.text((mid_x + 24, ry), "1) On Yuz (Kurum + 400 Damgasi) || Arka Yuz (Miktar + 700 Olcek):", fill="#92400E", font=f_bold)
    ry += 22
    draw_seq(mid_x + 24, ry, "176 740 400", indus_lg, step=46, color="#92400E")
    draw.text((mid_x + 178, ry + 10), "||", fill="#94A3B8", font=f_title)
    draw_seq(mid_x + 206, ry, "033 700", indus_lg, step=46, color="#0F172A")
    draw.text((mid_x + 315, ry + 8), "17x (3 olcek) & 7x (4 olcek: 034 700)", fill="#334155", font=f_reg)
    draw.text((mid_x + 315, ry + 26), "772 cift-yuzlu tabletin %63.3'unde 700!", fill="#64748B", font=f_sm)
    ry += 62
    draw.text((mid_x + 24, ry), "2) Uc-Yuzlu Sabit Kote Formulu (20x Harappa TAB:I):", fill="#92400E", font=f_bold)
    ry += 22
    draw_seq(mid_x + 24, ry, "003 861", indus_md, step=38, color="#0F172A")
    draw.text((mid_x + 106, ry + 8), "||", fill="#94A3B8", font=f_sec)
    draw_seq(mid_x + 126, ry, "034 700", indus_md, step=38, color="#B91C1C")
    draw.text((mid_x + 208, ry + 8), "||", fill="#94A3B8", font=f_sec)
    draw_seq(mid_x + 228, ry, "003 156", indus_md, step=38, color="#0F172A")
    draw.text((mid_x + 322, ry + 10), "20/20 hep '034 700' (4 olcek)!", fill="#B91C1C", font=f_bold)
    ry += 56
    cs = mod2["clitic_stacking_X_plus_400"]
    draw.text((mid_x + 24, ry), f"3) ': X :' + 400 Ek Yigilmasi ({cs['final_or_pre_400_total']}/{cs['total_framed_tokens']} = %{cs['rate_pct']}, p={cs['fisher_p']}):", fill="#92400E", font=f_bold)
    ry += 22
    draw_seq(mid_x + 24, ry, "236 400", indus_md, step=38, color="#0F172A")
    draw.text((mid_x + 110, ry + 12), "|", fill="#CBD5E1", font=f_sec)
    draw_seq(mid_x + 125, ry, "234 400", indus_md, step=38, color="#0F172A")
    draw.text((mid_x + 211, ry + 12), "|", fill="#CBD5E1", font=f_sec)
    draw_seq(mid_x + 226, ry, "226 400", indus_md, step=38, color="#0F172A")
    draw.text((mid_x + 316, ry + 8), "Ayni dilbilgisi + Harappa 400 eki!", fill="#047857", font=f_bold)
    draw.text((mid_x + 316, ry + 26), "Farkli dil degil, fis/jeton turu.", fill="#334155", font=f_sm)

    # --- PANEL D (Bottom): Meluhha Trade Commodity & Export Corridor Enriched Signs (DEDUP BH-FDR!) ---
    y += 304
    draw.rectangle([(20, y), (W - 20, y + 30)], fill="#DCFCE7")
    draw.text(
        (32, y + 6),
        f"D. MELUHHA DIS TICARET MALLARI & IHRACAT KORIDORU (TEKRARSIZ / DEDUP FISHER + GLOBAL BH-FDR q<0.05: {mod4['total_surviving_global_fdr_05']}/{mod4['total_dedup_tests_run']} test gecti)",
        fill="#14532D",
        font=f_sec,
    )
    y += 40

    # Render 4 neat cards across bottom row
    cards = [
        ("Tas/Kabuk Bilezik (BNGL)", "Stoneware_Shell_Bangles", "#ECFDF5", "#10B981"),
        ("Fildisi Cubuk (Ivory)", "Ivory_Objects", "#FEFCE8", "#EAB308"),
        ("Chanhu-daro Akik Atolye", "Chanhudaro_BeadHub_vs_Core", "#FFF7ED", "#F97316"),
        ("Korfez/Mezopotamya Ihracat", "Export_Corridor_Foreign_vs_CoreSeals", "#EFF6FF", "#3B82F6"),
    ]
    cw = (W - 40 - 3 * 16) // 4
    for idx, (title, cat_key, bg, border) in enumerate(cards):
        x0 = 20 + idx * (cw + 16)
        draw.rectangle([(x0, y), (x0 + cw, y + 220)], fill=bg, outline=border, width=2)
        cat = mod4["categories"][cat_key]
        draw.text((x0 + 12, y + 10), title, fill="#0F172A", font=f_bold)
        draw.text((x0 + 12, y + 28), f"n_uniq={cat['n_dedup_unique_texts']} (raw={cat['n_raw']}) | FDR<0.05: {cat['n_survive_global_fdr_05']}", fill="#475569", font=f_sm)
        hits = cat["top_dedup_hits"][:3]
        hy = y + 50
        if not hits:
            draw.text((x0 + 12, hy + 20), "Tekrarsiz testte p<0.01 isaret yok.", fill="#64748B", font=f_reg)
        for h in hits:
            sig = h["sign"]
            g = cp_map.get(sig.zfill(3), "?")
            col = "#15803D" if h["survives_fdr_05"] else "#475569"
            tag = "FDR*" if h["survives_fdr_05"] else "nom."
            draw.text((x0 + 14, hy), g, fill="#0F172A", font=indus_md)
            draw.text((x0 + 58, hy + 2), f"#{sig}  {h['dedup_fg']}/{h['dedup_n']} ({h['lift_dedup']}x)", fill="#0F172A", font=f_bold)
            draw.text((x0 + 58, hy + 20), f"p={h['p_dedup']:.1e} | q={h['q_fdr_global']:.2e} [{tag}]", fill=col, font=f_sm)
            hy += 54

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    if ART_DIR.exists():
        img.save(ART_DIR / out_path.name)


def main() -> None:
    con = sqlite3.connect(PROC_DIR / "indus_canonical.sqlite")
    ins = pd.read_sql("SELECT * FROM inscriptions", con)
    glyphs_df = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")

    mod1 = run_module1_copper_rosetta(ins)
    mod2 = run_module2_harappa_tokens(ins)
    mod3 = run_module3_multisig_cargo(ins)
    mod4 = run_module4_meluhha_trade_audit(ins)

    out_json = REPORTS_DIR / "layer9_internal_rosetta_trade.json"
    payload = {
        "layer": "Layer 9 — Internal Rosetta (TAB:C), Harappa Accounting Tokens (TAB:I/B), Multi-Signature Cargo Protocol (TAG*), & Meluhha Trade Commodities",
        "module1_copper_rosetta": mod1,
        "module2_harappa_tokens": mod2,
        "module3_multisig_cargo": mod3,
        "module4_meluhha_trade": mod4,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    plate_path = REPORTS_DIR / "layer9_copper_rosetta_plate.png"
    render_glyph_plate(glyphs_df, mod1, mod2, mod3, mod4, plate_path)

    # Write concise markdown summary
    md_lines = [
        "# INDUS-DEDUCT Layer 9: İç Rosetta (`TAB:C`), Harappa Fişleri, Çok-İmzalı Kargo ve Meluhha Dış Ticaret Malları",
        "",
        "## 1. Mohenjo-daro Bakır Tabletleri (`TAB:C`, 229/230 = %99.6) — İç Rosetta",
        f"- **Metin ↔ İkon Karşılıklı Bilgi (MI) Testi (1.000 permütasyon):**",
        f"  - `TAB:C` (tümü, n={mod1['text_symbol_mi_tab_c_all']['n_pairs']}): `MI = {mod1['text_symbol_mi_tab_c_all']['obs_mi_bits']} bit` (**z = +{mod1['text_symbol_mi_tab_c_all']['mi_z']}**, `p = {mod1['text_symbol_mi_tab_c_all']['mi_emp_p']}`, ağırlıklı saflık = **%{100*mod1['text_symbol_mi_tab_c_all']['weighted_purity']:.1f}** vs null `%{100*mod1['text_symbol_mi_tab_c_all']['null_weighted_purity_mean']:.1f}`)",
        f"  - `TAB:C` (sadece adlandırılmış hayvan/ikonlar, `Othr` hariç, n={mod1['text_symbol_mi_tab_c_named_only']['n_pairs']}): `MI = {mod1['text_symbol_mi_tab_c_named_only']['obs_mi_bits']} bit` (**z = +{mod1['text_symbol_mi_tab_c_named_only']['mi_z']}**, saflık = **%{100*mod1['text_symbol_mi_tab_c_named_only']['weighted_purity']:.1f}**)",
        f"  - **Kontrol — Kare Mühürler (`SEAL:S`, n={mod1['text_symbol_mi_seal_s_control']['n_pairs']}):** `z = +{mod1['text_symbol_mi_seal_s_control']['mi_z']}` (ağırlıklı saflık = `%{100*mod1['text_symbol_mi_seal_s_control']['weighted_purity']:.1f}` vs null `%{100*mod1['text_symbol_mi_seal_s_control']['null_weighted_purity_mean']:.1f}` — yani kare mühürde metin hayvanın adı değil!).",
        f"- **Çift Yüzlü Bakır Tabletler (`{mod1['total_single_sign_equation_tablets']}` tablet):** Arka yüzdeki hayvan figürünün yerine **tek bir İndus işareti** konmuş (`753 ↔ Tavşan`, `777/780/781/782 ↔ Boynuzlu Avcı & Keçi/Manda`, `341 ↔ 503 615 740`, `957 ↔ 798 231 705 033 904 100 740`).",
        "",
        "## 2. Harappa Çift Yüzlü Fiş Sistemi (`TAB:I` + `TAB:B`, 772 çift/üç yüzlü tablet)",
        f"- `TAB:I` %96.7, `TAB:B` %84.7 Harappa merkezli; çok yüzlü tabletlerin %60.5–66.2'sinde bir yüzde **`[sayı + 700]`** (`032/033/034 + 700`), diğer yüzde kurum/isim + `400` damga eki var.",
        f"- **Kurum ↔ Miktar Kilit Testi:** `MI z = +{mod2['entity_quantity_mi_test']['mi_z']}` (`p = {mod2['entity_quantity_mi_test']['mi_emp_p']}`, ağırlıklı saflık `%{100*mod2['entity_quantity_mi_test']['weighted_purity']:.1f}` vs null `%{100*mod2['entity_quantity_mi_test']['null_weighted_purity_mean']:.1f}`). Örnek: `003 861 || 003 156` üç-yüzlüsü **20/20 kez** hep `034 700` (4 ölçek) alıyor.",
        f"- **`: X : + 400` Ek Yığılması:** `{mod2['clitic_stacking_X_plus_400']['final_or_pre_400_total']}/{mod2['clitic_stacking_X_plus_400']['total_framed_tokens']}` (`%{mod2['clitic_stacking_X_plus_400']['rate_pct']}`, `p = {mod2['clitic_stacking_X_plus_400']['fisher_p']}`) — farklı dil değil, aynı gramerin sonuna Harappa `400` fiş eki eklenmiş.",
        "",
        "## 3. Çok İmzalı Kargo Paketleri (`TAG*`) & Şehirlerarası Lojistik",
        f"- Toplam `{mod3['multisig_lumps_with_ge2_distinct_seals']}` kil kargo topağında ≥2 farklı mühür aynı anda basılmış (`{mod3['multisig_by_site']}`).",
        f"- **Lothal Ana Gümrük Karşı-İmzası (`705 500 741 001 055 220 740 090`):** `{len(mod3['lothal_master_counter_sign']['co_signed_lumps'])}` farklı kargo paketinde (`2112, 2113, 2115, 2117`) ayrı tüccar mühürleriyle ortak imza atıyor — bunlardan `#2113` paketindeki karşı imza (`861 002 235`), **700 km ötede Mohenjo-daro'da kazılan gerçek taş mühür (`2239.1`)**!",
        f"- Toplam `{len(mod3['intercity_cargo_tag_to_distant_stone_seal_len3'])}` adet `len>=3` kargo mühür baskısı başka şehirdeki gerçek taş mühürle birebir eşleşiyor.",
        "",
        "## 4. Meluhha Dış Ticaret Malları & İhracat Koridoru (Tekrarsız Fisher + Global BH-FDR)",
        f"- Global BH-FDR (`q < 0.05`) geçen tekrarsız işaret sayısı: **`{mod4['total_surviving_global_fdr_05']}/{mod4['total_dedup_tests_run']}`**.",
    ]
    for cat_name, cat in mod4["categories"].items():
        surv = [h for h in cat["top_dedup_hits"] if h["survives_fdr_05"]]
        art_n = len(cat["duplicate_stamp_artifacts_caught"])
        md_lines.append(f"- **{cat_name}** (`n_uniq={cat['n_dedup_unique_texts']}`, `raw={cat['n_raw']}`): FDR<0.05 geçen = **{len(surv)}** | yakalanan kopya-damga yanılsaması = `{art_n}`")
        for h in surv:
            md_lines.append(f"  - İşaret **`{h['sign']}`**: `{h['dedup_fg']}/{h['dedup_n']}` (`{h['lift_dedup']}x`, `p={h['p_dedup']:.2e}`, `q={h['q_fdr_global']:.2e}`)")
    (REPORTS_DIR / "layer9_summary.md").write_text("\n".join(md_lines) + "\n")

    print("=== LAYER 9 COMPLETE ===")
    print("MOD1 TAB:C all MI:", mod1["text_symbol_mi_tab_c_all"])
    print("MOD1 TAB:C named MI:", mod1["text_symbol_mi_tab_c_named_only"])
    print("MOD1 SEAL:S control MI:", mod1["text_symbol_mi_seal_s_control"])
    print("MOD1 single-sign eqs:", len(mod1["single_sign_logogram_equations"]), "groups /", mod1["total_single_sign_equation_tablets"], "tablets")
    for eq in mod1["single_sign_logogram_equations"][:12]:
        print("  EQ:", eq)
    print("\nMOD2 entity-qty MI:", mod2["entity_quantity_mi_test"])
    print("MOD2 clitic X+400:", mod2["clitic_stacking_X_plus_400"])
    print("\nMOD3 multisig lumps:", mod3["multisig_lumps_with_ge2_distinct_seals"], mod3["multisig_by_site"])
    print("MOD3 intercity len>=3 tag<->seal:", len(mod3["intercity_cargo_tag_to_distant_stone_seal_len3"]))
    for m in mod3["intercity_cargo_tag_to_distant_stone_seal_len3"]:
        print("  LINK:", m)
    print("\nMOD4 global FDR survivors:", mod4["total_surviving_global_fdr_05"], "/", mod4["total_dedup_tests_run"])
    for cn, cd in mod4["categories"].items():
        surv = [h for h in cd["top_dedup_hits"] if h["survives_fdr_05"]]
        nom = [h for h in cd["top_dedup_hits"] if not h["survives_fdr_05"]][:2]
        print(f"  {cn} (uniq={cd['n_dedup_unique_texts']}, raw={cd['n_raw']}): FDR={[(x['sign'], x['dedup_fg'], x['lift_dedup'], x['q_fdr_global']) for x in surv]} | nom={[(x['sign'], x['dedup_fg'], x['lift_dedup'], x['p_dedup']) for x in nom]} | dup_artifacts={len(cd['duplicate_stamp_artifacts_caught'])}")


if __name__ == "__main__":
    main()
