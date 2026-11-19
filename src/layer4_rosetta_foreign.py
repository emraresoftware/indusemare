#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 4: Mesopotamia / Persian Gulf / Iranian Plateau / Central Asia
"Rosetta" Foreign Seals Structural Analysis & Museum-Grade Visual Plate
====================================================================================
Strictly separates:
  (A) VERIFIED STATISTICAL FACTS computed directly from the uncontaminated canonical
      5,679-inscription ICIT warehouse (41 foreign-region inscriptions / 42 lines),
      with length-matched bootstrap (B=5,000), controlled 2x2 motif/shape comparisons,
      and adversarial direction-invariant sensitivity checks;
  (B) VERIFIED CUNEIFORM REFERENCE TABLE of Sumerian/Akkadian/Amorite Meluhha & Dilmun
      personal names, titles, deities, and trade lexicon from primary Assyriological
      sources (Gadd 1932; Parpola, Parpola & Brunswig 1977; Glassner 1984, 2002;
      Parpola 1994; Possehl 1996; Vidale 2005; Vermaak 2008; Laursen 2010;
      Laursen & Steinkeller 2017; Marchesi 2017);
  (C) FALSIFIABLE CANDIDATE HYPOTHESES mapping structural constraints onto the top
      high-surprisal foreign seals (explicitly NOT claiming decipherment).

Outputs:
  - reports/layer4_rosetta_findings.json
  - reports/rosetta_foreign_seals.png
  - /Users/emre/.gemini/antigravity/brain/969a4497-dabc-4f45-8c95-dcaebdbcb608/rosetta_foreign_seals.png
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
BRAIN_PNG_PATH = Path(
    "/Users/emre/.gemini/antigravity/brain/969a4497-dabc-4f45-8c95-dcaebdbcb608/rosetta_foreign_seals.png"
)

BOS = "<BOS>"
EOS = "<EOS>"

# Morphological ICIT sign families (strictly visual/ICIT-numbered classes, zero phonetic assumptions)
ANTHRO_CLASS = {f"{i:03d}" for i in range(90, 165)}
TWIN_MAN_TRIAD = {"090", "091", "093"}
STROKES_CLASS = {f"{i:03d}" for i in range(1, 57)}
SHORT_STROKES_KEY = {"001", "002", "004"}
LONG_STROKES_KEY = {"031", "032", "033"}
TWELVE_STROKES = {"055"}
HL_TOP_SUFFIXES_NO_090 = {"740", "400", "520", "151", "407", "527", "154", "156"}
FISH_CLASS = {
    "219", "220", "221", "222", "226", "231", "232",
    "233", "234", "235", "236", "240", "241", "242", "243",
}

# Brief purely visual glyph glosses (verified directly against sk_indus_script-webfont.ttf)
VISUAL_GLOSS: dict[str, str] = {
    "001": "1 short raised stroke",
    "002": "2 short raised strokes",
    "003": "3 short raised strokes",
    "004": "4 short raised strokes",
    "013": "3 staggered short strokes",
    "031": "1 long vertical stroke",
    "032": "2 long vertical strokes",
    "033": "3 long vertical strokes",
    "055": "12 short strokes (3x4 grid)",
    "060": "double curved/bracket strokes",
    "071": "vertical staff with double crossbar",
    "090": "standing human figure (man)",
    "091": "two conjoined humans holding hands (twins)",
    "093": "standing human variant with lower prong",
    "095": "man holding vertical staff",
    "099": "striding/walking human figure",
    "117": "armed human holding oval shield",
    "121": "stylized hatched/winged human figure",
    "125": "archer/human holding bow",
    "127": "human flanked by two arcs/bows",
    "150": "human with shoulder crossbar/yoke",
    "151": "water-carrier with yoke & two jars",
    "160": "carrier yoke with two vessels",
    "190": "6-legged insect/spider/scorpion",
    "220": "plain fish",
    "235": "roofed fish",
    "240": "rayed/finned fish",
    "255": "hatched wing/feather/branch",
    "328": "curved stringed bow/harp",
    "340": "quadruped animal (hapax legomenon)",
    "350": "three-pronged tulip/lotus bud",
    "390": "trident/forked branch",
    "400": "comb/rake",
    "405": "diagonal-tined branch",
    "407": "forked comb-branch",
    "415": "four-tined rake/comb on stem",
    "416": "double-tiered rake/plant",
    "520": "arrow/spearhead",
    "590": "box with top vertical stem (hearth/standard)",
    "617": "double 3x3 grid (compound 615+615)",
    "643": "rare ICIT sign (absent from TTF)",
    "700": "U-shaped cup/vessel",
    "740": "two-handled Jar (top Heartland terminal)",
    "803": "leaf/oval with central rib",
    "817": "notched oval/disk",
    "820": "six-spoked wheel/star-oval",
    "831": "almond/oval with 3 internal diagonal strokes",
    "840": "double overlapping ovals",
    "924": "segmented/ladder-filled crescent",
}


# ---------------------------------------------------------------------------
# 1. DATA LOADING & HEARTLAND SURPRISAL MODEL (REPRODUCING LAYER 1-2 EXACTLY)
# ---------------------------------------------------------------------------
def load_warehouse() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ins_df = pd.read_csv(PROC_DIR / "inscriptions_clean.csv", dtype=str).fillna("")
    lines_df = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    glyphs_df = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")

    for col in ["is_strict_analytic", "is_foreign", "token_count_clean", "has_damage_token", "left_broken", "right_broken"]:
        lines_df[col] = lines_df[col].astype(int)
    for col in ["is_foreign", "clean_sign_count_full", "parsed_line_count", "has_structural_damage"]:
        ins_df[col] = ins_df[col].astype(int)
    for col in ["freq_all", "freq_strict_analytic", "component_count", "is_pua_compound"]:
        glyphs_df[col] = glyphs_df[col].astype(int)

    l12_path = REPORTS_DIR / "layer1_2_findings.json"
    l12 = json.loads(l12_path.read_text(encoding="utf-8")) if l12_path.exists() else {}
    return ins_df, lines_df, glyphs_df, l12


def fit_bigram(seqs: list[list[str]]) -> tuple[Counter, defaultdict[str, Counter], Counter, int]:
    uni: Counter = Counter()
    bi: defaultdict[str, Counter] = defaultdict(Counter)
    prev_tot: Counter = Counter()
    for seq in seqs:
        padded = [BOS] + seq + [EOS]
        for t in seq:
            uni[t] += 1
        for a, b in zip(padded[:-1], padded[1:]):
            bi[a][b] += 1
            prev_tot[a] += 1
    return uni, bi, prev_tot, sum(uni.values())


def seq_surprisal(
    seq: list[str],
    uni: Counter,
    bi: defaultdict[str, Counter],
    prev_tot: Counter,
    n_uni: int,
) -> float:
    V = len(uni) + 1
    padded = [BOS] + seq + [EOS]
    total_bits = 0.0
    for a, b in zip(padded[:-1], padded[1:]):
        p_u = (uni[b] + 0.5) / (n_uni + 0.5 * V)
        pt = prev_tot[a]
        p_ab = (bi[a][b] + 2.0 * p_u) / (pt + 2.0) if pt > 0 else p_u
        total_bits -= math.log2(max(p_ab, 1e-12))
    return total_bits / (len(seq) + 1)


def score_foreign_corpus(
    ins_df: pd.DataFrame, lines_df: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter, defaultdict[str, Counter]]:
    hl_strict = lines_df[
        (lines_df["is_strict_analytic"] == 1)
        & (lines_df["is_foreign"] == 0)
        & (lines_df["direction"].isin(["R/L", "L/R", "BUS"]))
    ]
    hl_unique = [s.split() for s in hl_strict["clean_reading_seq"].unique()]

    # Exact 5-fold CV Heartland baseline (matches layer1_2_deductive_engine.py seed=42)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(hl_unique))
    folds = np.array_split(idx, 5)
    hl_scores_fwd: list[float] = []
    hl_scores_min: list[float] = []
    for fi in range(5):
        te = set(folds[fi].tolist())
        tr = [hl_unique[i] for i in range(len(hl_unique)) if i not in te]
        u, b, pt, nu = fit_bigram(tr)
        for i in folds[fi]:
            s = hl_unique[i]
            sf = seq_surprisal(s, u, b, pt, nu)
            sr = seq_surprisal(list(reversed(s)), u, b, pt, nu)
            hl_scores_fwd.append(sf)
            hl_scores_min.append(min(sf, sr))

    hl_mean = float(np.mean(hl_scores_fwd))
    hl_std = float(np.std(hl_scores_fwd) + 1e-12)
    hl_min_mean = float(np.mean(hl_scores_min))
    hl_min_std = float(np.std(hl_scores_min) + 1e-12)

    uni_all, bi_all, prev_all, n_uni_all = fit_bigram(hl_unique)

    # Full inventory of all 41 foreign inscriptions / 42 lines
    f_ins = ins_df[ins_df["is_foreign"] == 1].copy()
    f_lines = lines_df[lines_df["is_foreign"] == 1].merge(
        ins_df[["inscription_id", "raw_text", "shape", "cult", "excavation_idno", "area_section"]],
        on="inscription_id",
        how="left",
    )

    all_foreign_records: list[dict[str, Any]] = []
    usable_ranked: list[dict[str, Any]] = []

    for _, r in f_lines.iterrows():
        seq = r["clean_reading_seq"].split() if r["clean_reading_seq"] else []
        rec: dict[str, Any] = {
            "inscription_id": r["inscription_id"],
            "line_idx": int(r["line_idx"]),
            "region": r["region"],
            "site": r["site"],
            "object_type": r["object_type"],
            "shape": r["shape"],
            "symbol": r["symbol"] if r["symbol"] not in {"", "-"} else "",
            "material": r["material"],
            "complete": r["complete"],
            "direction": r["direction"],
            "has_damage_token": int(r["has_damage_token"]),
            "cisi": r["cisi"],
            "excavation_idno": r["excavation_idno"],
            "raw_text": r["raw_text"],
            "tokens": seq,
            "reading_seq": " -> ".join(seq),
            "len": len(seq),
        }
        if len(seq) >= 2:
            sf = seq_surprisal(seq, uni_all, bi_all, prev_all, n_uni_all)
            sr = seq_surprisal(list(reversed(seq)), uni_all, bi_all, prev_all, n_uni_all)
            zf = (sf - hl_mean) / hl_std
            zr = (sr - hl_mean) / hl_std
            padded = [BOS] + seq + [EOS]
            unseen_bi_list = [
                f"{a}->{b}" for a, b in zip(padded[:-1], padded[1:]) if bi_all[a][b] == 0
            ]
            unseen_bi = len(unseen_bi_list)
            total_bi = len(seq) + 1
            unseen_signs = [s for s in seq if uni_all[s] == 0]
            rec.update(
                {
                    "usable_for_surprisal": True,
                    "surprisal_bits": round(sf, 3),
                    "surprisal_rev_bits": round(sr, 3),
                    "min_dir_surprisal_bits": round(min(sf, sr), 3),
                    "z_score_vs_heartland": round(zf, 2),
                    "z_score_rev": round(zr, 2),
                    "unseen_bigrams": f"{unseen_bi}/{total_bi}",
                    "unseen_bigram_ratio": round(unseen_bi / total_bi, 3),
                    "unseen_bigram_count": unseen_bi,
                    "total_transitions": total_bi,
                    "unseen_signs": unseen_signs,
                    "has_repeated_sign": len(set(seq)) < len(seq),
                    "classification": (
                        "ROSETTA_ANOMALY_CANDIDATE"
                        if zf >= 1.5
                        else ("NATIVE_HARAPPAN_ABROAD" if zf <= 0.3 else "MIXED_HYBRID")
                    ),
                }
            )
            usable_ranked.append(rec)
        else:
            rec["usable_for_surprisal"] = False
            rec["classification"] = "SINGLE_SIGN_ONLY" if len(seq) == 1 else "DAMAGED_NO_CLEAN_SIGNS"
        all_foreign_records.append(rec)

    usable_ranked.sort(key=lambda x: (-x["z_score_vs_heartland"], -x["surprisal_bits"], x["inscription_id"]))
    for rank, item in enumerate(usable_ranked, 1):
        item["rank"] = rank

    # Mann-Whitney tests (canonical forward & adversarial direction-invariant min(fwd, rev))
    f_fwd_vals = [x["surprisal_bits"] for x in usable_ranked]
    f_min_vals = [x["min_dir_surprisal_bits"] for x in usable_ranked]
    _, p_fwd = mannwhitneyu(f_fwd_vals, hl_scores_fwd, alternative="greater")
    _, p_min = mannwhitneyu(f_min_vals, hl_scores_min, alternative="greater")

    # Subgroup surprisal summary (SEAL:C vs SEAL:S vs Gaur vs regions)
    subgroup_surp: dict[str, Any] = {}
    for sname, flt in [
        ("Foreign_ALL_usable", lambda x: True),
        ("Foreign_SEAL_C_circular", lambda x: x["object_type"] == "SEAL:C"),
        ("Foreign_SEAL_CY_cylinder", lambda x: x["object_type"] == "SEAL:CY"),
        ("Foreign_SEAL_S_square", lambda x: x["object_type"] == "SEAL:S"),
        ("Foreign_Gaur_motif", lambda x: x["symbol"] == "Gaur"),
        ("Foreign_Gulf_Meso", lambda x: x["region"] in {"Persian Gulf", "Mesopotamia"}),
        ("Region_Persian_Gulf", lambda x: x["region"] == "Persian Gulf"),
        ("Region_Mesopotamia", lambda x: x["region"] == "Mesopotamia"),
        ("Region_Iranian_Plateau", lambda x: x["region"] == "Iranian Plateau"),
        ("Region_Central_Asia", lambda x: x["region"] == "Central Asia"),
    ]:
        sub = [x for x in usable_ranked if flt(x)]
        if not sub:
            continue
        vals = [x["surprisal_bits"] for x in sub]
        mvals = [x["min_dir_surprisal_bits"] for x in sub]
        zvals = [x["z_score_vs_heartland"] for x in sub]
        ubr = [x["unseen_bigram_ratio"] for x in sub]
        _, pv = mannwhitneyu(vals, hl_scores_fwd, alternative="greater")
        _, pv_min = mannwhitneyu(mvals, hl_scores_min, alternative="greater")
        subgroup_surp[sname] = {
            "n": len(sub),
            "mean_surprisal_bits": round(float(np.mean(vals)), 3),
            "mean_z_score": round(float(np.mean(zvals)), 2),
            "median_z_score": round(float(np.median(zvals)), 2),
            "mean_unseen_bigram_ratio": round(float(np.mean(ubr)), 3),
            "mw_p_vs_heartland_fwd": float(f"{pv:.3e}"),
            "mw_p_vs_heartland_dir_invariant": float(f"{pv_min:.3e}"),
        }

    # Direct Mann-Whitney: Foreign SEAL:C vs Foreign SEAL:S
    circ_s = [x["surprisal_bits"] for x in usable_ranked if x["object_type"] == "SEAL:C"]
    sq_s = [x["surprisal_bits"] for x in usable_ranked if x["object_type"] == "SEAL:S"]
    _, p_circ_gt_sq = mannwhitneyu(circ_s, sq_s, alternative="greater")

    audit = {
        "total_foreign_inscriptions": int(len(f_ins)),
        "total_foreign_parsed_lines": int(len(f_lines)),
        "by_region_inscriptions": f_ins["region"].value_counts().to_dict(),
        "by_region_lines": f_lines["region"].value_counts().to_dict(),
        "usable_lines_ge2_signs": len(usable_ranked),
        "single_sign_lines": int(sum(1 for x in all_foreign_records if x["len"] == 1)),
        "damaged_empty_lines": int(sum(1 for x in all_foreign_records if x["len"] == 0)),
        "heartland_cv_baseline": {
            "n_unique_strict_heartland": len(hl_unique),
            "mean_surprisal_bits": round(hl_mean, 3),
            "std_surprisal_bits": round(hl_std, 3),
            "dir_invariant_min_mean_bits": round(hl_min_mean, 3),
            "dir_invariant_min_std_bits": round(hl_min_std, 3),
        },
        "foreign_vs_heartland_surprisal": {
            "foreign_mean_bits": round(float(np.mean(f_fwd_vals)), 3),
            "foreign_dir_invariant_min_mean_bits": round(float(np.mean(f_min_vals)), 3),
            "mann_whitney_p_fwd": float(f"{p_fwd:.3e}"),
            "mann_whitney_p_dir_invariant": float(f"{p_min:.3e}"),
            "foreign_SEAL_C_gt_SEAL_S_mw_p": float(f"{p_circ_gt_sq:.3e}"),
        },
        "by_subgroup_surprisal": subgroup_surp,
        "all_41_inscriptions_42_lines": all_foreign_records,
    }
    return audit, usable_ranked, uni_all, bi_all


# ---------------------------------------------------------------------------
# 2. LENGTH-MATCHED BOOTSTRAP & STRUCTURAL COMPARISONS (FOREIGN VS HEARTLAND)
# ---------------------------------------------------------------------------
def compute_sequence_metrics(seqs: list[list[str]]) -> dict[str, float]:
    toks = [t for s in seqs for t in s]
    n_tok = len(toks)
    uni = Counter(toks)
    V = len(uni)
    h0 = -sum((c / n_tok) * math.log2(c / n_tok) for c in uni.values()) if n_tok else 0.0
    # Miller-Madow first-order bias-corrected entropy
    h0_mm = h0 + (V - 1) / (2.0 * n_tok * math.log(2.0)) if n_tok > 0 else 0.0
    evenness = h0 / math.log2(V) if V > 1 else 0.0

    inits = [s[0] for s in seqs if s]
    ic = Counter(inits)
    ni = len(inits)
    h_init = -sum((c / ni) * math.log2(c / ni) for c in ic.values()) if ni else 0.0

    fins = [s[-1] for s in seqs if s]
    fc = Counter(fins)
    nf = len(fins)
    h_fin = -sum((c / nf) * math.log2(c / nf) for c in fc.values()) if nf else 0.0

    bi: Counter = Counter()
    prev: Counter = Counter()
    for s in seqs:
        pad = [BOS] + s + [EOS]
        for a, b in zip(pad[:-1], pad[1:]):
            bi[(a, b)] += 1
            prev[a] += 1
    nb = sum(bi.values())
    h1 = -sum((c / nb) * math.log2(c / prev[a]) for (a, b), c in bi.items()) if nb else 0.0

    rep = sum(1 for s in seqs if len(set(s)) < len(s)) / max(len(seqs), 1)
    ttr = V / max(n_tok, 1)
    return {
        "n_seq": float(len(seqs)),
        "n_tok": float(n_tok),
        "V": float(V),
        "TTR": ttr,
        "H0": h0,
        "H0_miller_madow": h0_mm,
        "H_evenness": evenness,
        "H_init": h_init,
        "H_fin": h_fin,
        "H1": h1,
        "within_seal_repeat_rate": rep,
    }


def run_structural_comparisons(lines_df: pd.DataFrame, n_boot: int = 5000) -> dict[str, Any]:
    hl_usable = lines_df[(lines_df["is_foreign"] == 0) & (lines_df["token_count_clean"] >= 2)]
    hl_dedup = lines_df[
        (lines_df["is_foreign"] == 0)
        & (lines_df["is_strict_analytic"] == 1)
        & (lines_df["direction"].isin(["R/L", "L/R", "BUS"]))
    ].drop_duplicates("clean_reading_seq")
    hl_bull1 = hl_usable[hl_usable["symbol"].str.startswith("Bull1")].drop_duplicates("clean_reading_seq")
    hl_gaur = hl_usable[hl_usable["symbol"] == "Gaur"].drop_duplicates("clean_reading_seq")

    f_usable = lines_df[(lines_df["is_foreign"] == 1) & (lines_df["token_count_clean"] >= 2)]
    f_gulf_meso = f_usable[f_usable["region"].isin(["Persian Gulf", "Mesopotamia"])]
    f_gaur = f_usable[f_usable["symbol"] == "Gaur"]
    f_circ = f_usable[f_usable["object_type"] == "SEAL:C"]
    f_circ_plus_cy = f_usable[f_usable["object_type"].isin(["SEAL:C", "SEAL:CY"])]
    f_sq = f_usable[f_usable["object_type"] == "SEAL:S"]

    groups: dict[str, pd.DataFrame] = {
        "HL_strict_dedup": hl_dedup,
        "HL_Bull1_dedup": hl_bull1,
        "HL_Gaur_dedup": hl_gaur,
        "Foreign_ALL_usable": f_usable,
        "Foreign_Gulf_Meso": f_gulf_meso,
        "Foreign_Gaur": f_gaur,
        "Foreign_SEAL_C": f_circ,
        "Foreign_SEAL_C_and_CY": f_circ_plus_cy,
        "Foreign_SEAL_S": f_sq,
    }

    # 2A. Raw metrics & terminal/initial sign breakdowns for every group
    group_profiles: dict[str, Any] = {}
    for gname, gdf in groups.items():
        seqs = [s.split() for s in gdf["clean_reading_seq"]]
        m = compute_sequence_metrics(seqs)
        fins = Counter(s[-1] for s in seqs)
        inits = Counter(s[0] for s in seqs)
        n = len(seqs)
        toks = [t for s in seqs for t in s]
        nt = max(len(toks), 1)
        group_profiles[gname] = {
            "n_inscriptions": n,
            "n_tokens": int(m["n_tok"]),
            "mean_length": round(m["n_tok"] / max(n, 1), 2),
            "vocab_V": int(m["V"]),
            "type_token_ratio": round(m["TTR"], 3),
            "H0_bits": round(m["H0"], 3),
            "H0_miller_madow_bits": round(m["H0_miller_madow"], 3),
            "H_evenness": round(m["H_evenness"], 3),
            "H_init_bits": round(m["H_init"], 3),
            "H_fin_bits": round(m["H_fin"], 3),
            "H_fin_max_possible_log2n": round(math.log2(n), 3) if n > 1 else 0.0,
            "unique_terminal_signs": len(fins),
            "H1_cond_bigram_bits": round(m["H1"], 3),
            "within_seal_repeat_rate": round(m["within_seal_repeat_rate"], 3),
            "jar_740_final_pct": round(100.0 * fins["740"] / max(n, 1), 1),
            "jar_740_any_pct": round(100.0 * sum(1 for s in seqs if "740" in s) / max(n, 1), 1),
            "hl_8_suffixes_final_pct": round(
                100.0 * sum(1 for s in seqs if s[-1] in HL_TOP_SUFFIXES_NO_090) / max(n, 1), 1
            ),
            "token_share_pct": {
                "anthropomorphic_090_164": round(100.0 * sum(1 for t in toks if t in ANTHRO_CLASS) / nt, 1),
                "strokes_001_056": round(100.0 * sum(1 for t in toks if t in STROKES_CLASS) / nt, 1),
                "jar_740": round(100.0 * sum(1 for t in toks if t == "740") / nt, 1),
                "fish_219_243": round(100.0 * sum(1 for t in toks if t in FISH_CLASS) / nt, 1),
            },
            "top_terminal_signs": [
                {"sign_id": k, "count": int(v), "pct": round(100.0 * v / n, 1)}
                for k, v in fins.most_common(8)
            ],
            "top_initial_signs": [
                {"sign_id": k, "count": int(v), "pct": round(100.0 * v / n, 1)}
                for k, v in inits.most_common(8)
            ],
        }

    # 2B. Length-matched & sample-size-matched bootstrap (B = 5,000)
    rng = np.random.default_rng(42)
    pools = {
        "HL_strict_dedup": [s.split() for s in hl_dedup["clean_reading_seq"]],
        "HL_Bull1_dedup": [s.split() for s in hl_bull1["clean_reading_seq"]],
        "HL_Gaur_dedup": [s.split() for s in hl_gaur["clean_reading_seq"]],
    }
    pool_by_len: dict[str, dict[int, list[list[str]]]] = {}
    for pname, plist in pools.items():
        d: dict[int, list[list[str]]] = defaultdict(list)
        for s in plist:
            d[len(s)].append(s)
        pool_by_len[pname] = d

    metrics_to_boot = ["V", "TTR", "H0", "H0_miller_madow", "H_init", "H_fin", "H1", "within_seal_repeat_rate"]
    bootstrap_results: dict[str, Any] = {}

    comparisons = [
        ("Foreign_ALL_usable", "HL_strict_dedup"),
        ("Foreign_ALL_usable", "HL_Bull1_dedup"),
        ("Foreign_Gulf_Meso", "HL_strict_dedup"),
        ("Foreign_Gaur", "HL_strict_dedup"),
        ("Foreign_Gaur", "HL_Gaur_dedup"),
        ("Foreign_SEAL_C", "HL_strict_dedup"),
        ("Foreign_SEAL_C", "HL_Bull1_dedup"),
        ("Foreign_SEAL_C_and_CY", "HL_strict_dedup"),
        ("Foreign_SEAL_S", "HL_strict_dedup"),
    ]

    for fname, pname in comparisons:
        fseqs = [s.split() for s in groups[fname]["clean_reading_seq"]]
        obs = compute_sequence_metrics(fseqs)
        lens = [len(s) for s in fseqs]
        all_p = pools[pname]
        by_l = pool_by_len[pname]
        boot_acc: dict[str, list[float]] = {k: [] for k in metrics_to_boot}
        for _ in range(n_boot):
            sample = []
            for L in lens:
                cands = by_l.get(L)
                if not cands:
                    cands = all_p
                sample.append(cands[int(rng.integers(len(cands)))])
            st = compute_sequence_metrics(sample)
            for k in metrics_to_boot:
                boot_acc[k].append(st[k])

        comp_res: dict[str, Any] = {"n_boot": n_boot, "n_inscriptions": len(fseqs), "n_tokens": int(obs["n_tok"])}
        for k in metrics_to_boot:
            arr = np.array(boot_acc[k], dtype=np.float64)
            mu = float(arr.mean())
            sd = float(arr.std() + 1e-12)
            z = (obs[k] - mu) / sd
            p_hi = float((np.sum(arr >= obs[k]) + 1) / (n_boot + 1))
            p_lo = float((np.sum(arr <= obs[k]) + 1) / (n_boot + 1))
            comp_res[k] = {
                "observed": round(obs[k], 4),
                "matched_baseline_mean": round(mu, 4),
                "matched_baseline_std": round(sd, 4),
                "z_score": round(z, 2),
                "p_greater": round(p_hi, 4),
                "p_less": round(p_lo, 4),
            }
        bootstrap_results[f"{fname}__vs__{pname}"] = comp_res

    # 2C. Controlled 2x2 Natural Experiment Tests (Fisher Exact)
    def class_count(df: pd.DataFrame, mode: str, sign_set: set[str] | None = None) -> tuple[int, int]:
        seqs = [s.split() for s in df["clean_reading_seq"]]
        n = len(seqs)
        if mode == "final_740":
            k = sum(1 for s in seqs if s[-1] == "740")
        elif mode == "final_hl_suffix":
            k = sum(1 for s in seqs if s[-1] in HL_TOP_SUFFIXES_NO_090)
        else:
            assert sign_set is not None
            k = sum(1 for s in seqs if set(s) & sign_set)
        return k, n

    feature_defs: list[tuple[str, str, set[str] | None]] = [
        ("twin_man_triad_090_091_093_any", "any", TWIN_MAN_TRIAD),
        ("anthropomorphic_090_164_any", "any", ANTHRO_CLASS),
        ("sign_091_double_man_any", "any", {"091"}),
        ("sign_055_12_strokes_any", "any", TWELVE_STROKES),
        ("jar_740_final", "final_740", None),
        ("jar_740_any", "any", {"740"}),
        ("hl_8_suffixes_final_excl_090", "final_hl_suffix", None),
        ("short_strokes_001_002_004_any", "any", SHORT_STROKES_KEY),
        ("long_strokes_031_032_033_any", "any", LONG_STROKES_KEY),
    ]

    controlled_2x2: dict[str, Any] = {}
    pairs_2x2 = [
        ("Foreign_SEAL_C_vs_HL_strict_dedup", "Foreign_SEAL_C", "HL_strict_dedup"),
        ("Foreign_Gaur_vs_HL_Gaur_motif_controlled", "Foreign_Gaur", "HL_Gaur_dedup"),
        ("Foreign_SEAL_C_vs_Foreign_SEAL_S_geography_controlled", "Foreign_SEAL_C", "Foreign_SEAL_S"),
        ("Foreign_ALL_vs_HL_strict_dedup", "Foreign_ALL_usable", "HL_strict_dedup"),
    ]
    for cname, gA, gB in pairs_2x2:
        rows_2x2 = {}
        for feat_name, mode, sset in feature_defs:
            kA, nA = class_count(groups[gA], mode, sset)
            kB, nB = class_count(groups[gB], mode, sset)
            pA, pB = kA / max(nA, 1), kB / max(nB, 1)
            _, pval = fisher_exact([[kA, nA - kA], [kB, nB - kB]], alternative="two-sided")
            lift = (pA / pB) if pB > 0 else (float("inf") if pA > 0 else 1.0)
            rows_2x2[feat_name] = {
                "group_A": f"{kA}/{nA} ({100*pA:.1f}%)",
                "group_B": f"{kB}/{nB} ({100*pB:.1f}%)",
                "lift": round(lift, 2) if math.isfinite(lift) else "INF",
                "fisher_two_sided_p": float(f"{pval:.3e}"),
            }
        controlled_2x2[cname] = rows_2x2

    # 2D. Individual Sign Enrichments & Depletions (Foreign usable & Foreign SEAL:C vs Heartland)
    def sign_enrichment_table(target_df: pd.DataFrame, ref_df: pd.DataFrame, min_cnt: int = 2) -> dict[str, Any]:
        c_t = Counter(tok for s in target_df["clean_reading_seq"] for tok in set(s.split()))
        c_r = Counter(tok for s in ref_df["clean_reading_seq"] for tok in set(s.split()))
        nt, nr = len(target_df), len(ref_df)
        enriched = []
        for s, a in c_t.items():
            if a < min_cnt:
                continue
            c = c_r[s]
            pt, pr = a / nt, c / nr
            lift = pt / max(pr, 1e-6)
            _, pv = fisher_exact([[a, nt - a], [c, nr - c]], alternative="greater")
            if pv < 0.10:
                enriched.append(
                    {
                        "sign_id": s,
                        "visual_gloss": VISUAL_GLOSS.get(s, ""),
                        "target_docs": f"{a}/{nt} ({100*pt:.1f}%)",
                        "ref_docs": f"{c}/{nr} ({100*pr:.2f}%)",
                        "lift": round(lift, 2),
                        "fisher_p_greater": float(f"{pv:.3e}"),
                    }
                )
        enriched.sort(key=lambda x: x["fisher_p_greater"])

        depleted = []
        for s, c in c_r.most_common(45):
            a = c_t[s]
            pt, pr = a / nt, c / nr
            _, pv = fisher_exact([[a, nt - a], [c, nr - c]], alternative="less")
            if pv < 0.20:
                depleted.append(
                    {
                        "sign_id": s,
                        "visual_gloss": VISUAL_GLOSS.get(s, ""),
                        "target_docs": f"{a}/{nt} ({100*pt:.1f}%)",
                        "ref_docs": f"{c}/{nr} ({100*pr:.2f}%)",
                        "ratio": round(pt / max(pr, 1e-6), 2),
                        "fisher_p_less": float(f"{pv:.3e}"),
                    }
                )
        depleted.sort(key=lambda x: x["fisher_p_less"])
        return {"enriched": enriched[:20], "depleted": depleted[:12]}

    sign_enrich = {
        "Foreign_ALL_usable_vs_HL_all_usable": sign_enrichment_table(f_usable, hl_usable, min_cnt=2),
        "Foreign_SEAL_C_vs_HL_strict_dedup": sign_enrichment_table(f_circ, hl_dedup, min_cnt=2),
    }

    # 2E. Verified-direction vs UNK-direction audit for 090 & 091 on Foreign SEAL:C
    circ_ver = f_circ[f_circ["direction"].isin(["R/L", "L/R", "BUS"])]
    circ_unk = f_circ[f_circ["direction"] == "UNK"]
    dir_audit_090_091 = {
        "verified_dir_SEAL_C_n": int(len(circ_ver)),
        "verified_dir_SEAL_C_091_initial": int(sum(1 for s in circ_ver["clean_reading_seq"] if s.split()[0] == "091")),
        "verified_dir_SEAL_C_090_positions": {
            "init": int(sum(1 for s in circ_ver["clean_reading_seq"] if s.split()[0] == "090")),
            "medial": int(sum(1 for s in circ_ver["clean_reading_seq"] if "090" in s.split()[1:-1])),
            "final": int(sum(1 for s in circ_ver["clean_reading_seq"] if s.split()[-1] == "090")),
        },
        "verified_dir_SEAL_C_unique_terminals": f"{len(set(s.split()[-1] for s in circ_ver['clean_reading_seq']))}/{len(circ_ver)}",
        "unk_dir_SEAL_C_n": int(len(circ_unk)),
        "unk_dir_SEAL_C_090_initial_caveat": [
            {
                "inscription_id": r["inscription_id"],
                "site": r["site"],
                "seq_canonical": r["clean_reading_seq"],
                "seq_reversed": " ".join(reversed(r["clean_reading_seq"].split())),
            }
            for _, r in circ_unk.iterrows()
        ],
    }

    return {
        "group_profiles": group_profiles,
        "length_matched_bootstrap": bootstrap_results,
        "controlled_2x2_fisher_tests": controlled_2x2,
        "sign_enrichments": sign_enrich,
        "direction_caveat_audit_090_091": dir_audit_090_091,
    }


# ---------------------------------------------------------------------------
# 3. RECURRING FOREIGN BIGRAMS, MOTIFS, NATIVE FORMULAS & UNPROVENANCED CHECK
# ---------------------------------------------------------------------------
def analyze_foreign_patterns(lines_df: pd.DataFrame) -> dict[str, Any]:
    f_all = lines_df[(lines_df["is_foreign"] == 1) & (lines_df["token_count_clean"] >= 1)]
    f_usable = lines_df[(lines_df["is_foreign"] == 1) & (lines_df["token_count_clean"] >= 2)]
    hl_non_other = lines_df[(lines_df["is_foreign"] == 0) & (lines_df["region"] != "Other") & (lines_df["token_count_clean"] >= 2)]
    other_df = lines_df[(lines_df["region"] == "Other") & (lines_df["token_count_clean"] >= 1)]

    # Directed bigrams in foreign usable
    f_bi = Counter()
    f_bi_ex: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, r in f_usable.iterrows():
        seq = r["clean_reading_seq"].split()
        for a, b in zip(seq[:-1], seq[1:]):
            f_bi[(a, b)] += 1
            f_bi_ex[(a, b)].append(f"{r['inscription_id']} ({r['site']}, {r['object_type']})")

    hl_bi = Counter()
    for s in hl_non_other["clean_reading_seq"]:
        seq = s.split()
        for a, b in zip(seq[:-1], seq[1:]):
            hl_bi[(a, b)] += 1

    nf_bi = sum(f_bi.values())
    nh_bi = sum(hl_bi.values())

    recurring_directed = []
    for (a, b), cnt in f_bi.most_common():
        if cnt < 2:
            continue
        hc = hl_bi[(a, b)]
        _, pv = fisher_exact([[cnt, nf_bi - cnt], [hc, nh_bi - hc]], alternative="greater")
        lift = (cnt / nf_bi) / max(hc / nh_bi, 1e-6) if hc > 0 else float("inf")
        recurring_directed.append(
            {
                "bigram": f"{a} -> {b}",
                "foreign_count": int(cnt),
                "heartland_count": int(hc),
                "lift": round(lift, 2) if math.isfinite(lift) else "INF",
                "fisher_p": float(f"{pv:.3e}"),
                "examples": f_bi_ex[(a, b)],
            }
        )

    # Curated foreign-signature / foreign-enriched structural motifs (verified across ALL 5,679 lines)
    pattern_specs = [
        ("091_double_man_family", "Sign 091 (Twins/Double-Man) anywhere", lambda s: "091" in s.split()),
        ("091_followed_by_stroke", "091 -> {031, 032, 033, 001} (Twins + stroke)", lambda s: any(p in s for p in ["091 031", "091 032", "091 033", "091 001"])),
        ("055_to_090_gulf_exclusive", "055 -> 090 (12-strokes -> Man, 0x Heartland!)", lambda s: "055 090" in s),
        ("001_to_090_gulf_exclusive", "001 -> 090 (1-stroke -> Man, 0x Heartland!)", lambda s: "001 090" in s),
        ("090_001_adjacent_either_dir", "001 ~ 090 adjacent (either direction)", lambda s: "001 090" in s or "090 001" in s),
        ("093_rare_man_variant", "Sign 093 anywhere (53.3x lift abroad)", lambda s: "093" in s.split()),
        ("093_to_031_foreign_only", "093 -> 031 (0x Heartland!)", lambda s: "093 031" in s),
        ("031_and_055_cooccur", "031 + 055 co-occurrence in same seal", lambda s: "031" in s.split() and "055" in s.split()),
        ("350_and_090_cooccur", "350 + 090 co-occurrence (Failaka & Ur)", lambda s: "350" in s.split() and "090" in s.split()),
        ("002_004_adjacent_either_dir", "002 ~ 004 adjacent (Umma, Salut, Ur)", lambda s: "002 004" in s or "004 002" in s),
        ("004_to_002_foreign_only", "004 -> 002 (Ur 3898.1, 0x Heartland!)", lambda s: "004 002" in s),
        ("924_segmented_crescent", "Sign 924 anywhere (Ur & Susa cylinder, 33.3x)", lambda s: "924" in s.split()),
        ("831_striped_oval", "Sign 831 anywhere (Luristan & Janabiyah, 16.7x)", lambda s: "831" in s.split()),
        ("590_390_740_native_formula_abroad", "590 -> 390 -> 740 (#1 Native Harappan trigram abroad!)", lambda s: "590 390 740" in s),
    ]

    motif_inventory = []
    for key, desc, fn in pattern_specs:
        m_for = f_all[f_all["clean_reading_seq"].apply(fn)]
        m_oth = other_df[other_df["clean_reading_seq"].apply(fn)]
        m_hl = hl_non_other[hl_non_other["clean_reading_seq"].apply(fn)]
        motif_inventory.append(
            {
                "motif_id": key,
                "description": desc,
                "foreign_count": int(len(m_for)),
                "unprovenanced_other_count": int(len(m_oth)),
                "heartland_count": int(len(m_hl)),
                "foreign_examples": [
                    f"{r['inscription_id']} ({r['site']}, {r['object_type']}, {r['symbol'] or '-'}): {r['clean_reading_seq']}"
                    for _, r in m_for.iterrows()
                ],
                "unprovenanced_examples": [
                    f"{r['inscription_id']} (Unknown, {r['object_type']}, {r['symbol'] or '-'}): {r['clean_reading_seq']}"
                    for _, r in m_oth.iterrows()
                ],
            }
        )

    # Unprovenanced Gulf-type circular/cylinder seals + Dholavira circular Gaur seal cross-check
    gulf_signature_signs = {"091", "090", "093", "055", "190", "160", "924", "831"}
    unprov_gulf_hits = []
    for _, r in lines_df[
        ((lines_df["region"] == "Other") & (lines_df["object_type"].isin(["SEAL:C", "SEAL:CY"])))
        | ((lines_df["site"] == "Dholavira") & (lines_df["object_type"] == "SEAL:C"))
    ].iterrows():
        toks = r["clean_reading_seq"].split()
        hits = sorted(set(toks) & gulf_signature_signs)
        unprov_gulf_hits.append(
            {
                "inscription_id": r["inscription_id"],
                "region": r["region"],
                "site": r["site"],
                "object_type": r["object_type"],
                "symbol": r["symbol"],
                "reading_seq": " -> ".join(toks),
                "shared_gulf_signature_signs": hits,
            }
        )

    return {
        "recurring_directed_bigrams_in_foreign": recurring_directed,
        "foreign_motif_and_formula_inventory": motif_inventory,
        "out_of_sample_unprovenanced_and_dholavira_circular_seals": unprov_gulf_hits,
    }


# ---------------------------------------------------------------------------
# 4. VERIFIED CUNEIFORM REFERENCE TABLE & STRUCTURAL HYPOTHESIS MAPPING
# ---------------------------------------------------------------------------
def build_cuneiform_reference_and_hypotheses() -> dict[str, Any]:
    verified_cuneiform_corpus = {
        "epistemic_status": "VERIFIED_PRIMARY_ASSYRIOLOGICAL_REFERENCES",
        "personal_names_and_titles_linked_to_meluhha_and_dilmun": [
            {
                "name_transliteration": "Šu-ilīšu (šu-i-li-su)",
                "cuneiform_title_or_context": "eme-bal me-luḫ-ḫa ('interpreter of the Meluhhan language')",
                "artifact_or_text_id": "Louvre AO 22310 (Collection de Clercq, Old Akkadian cylinder seal)",
                "language": "Akkadian personal name + Sumerian title",
                "morpheme_template": "Šu- + ilī + -šu ('He of his god') — prefixing Akkadian PN",
                "syllable_count_pn": 4,
                "total_seal_cuneiform_signs": 9,
                "scholarly_refs": ["Parpola, Parpola & Brunswig 1977", "Parpola 1994", "Possehl 1996"],
            },
            {
                "name_transliteration": "Lú-sunzida (lú-su₃-zi-da)",
                "cuneiform_title_or_context": "lú-me-luḫ-ḫa-ke₄ ('man of Meluhha'); paid 10 shekels silver to Urur for a broken tooth",
                "artifact_or_text_id": "Old Akkadian / Ur III legal-administrative record (Girsu/Lagash corpus)",
                "language": "Sumerian (likely literal calque of an Indus/Meluhhan name: 'Man of the just buffalo cow')",
                "morpheme_template": "Lú- ('Man of') + sun-zi-da — prefixing Sumerian PN",
                "syllable_count_pn": 4,
                "scholarly_refs": ["Parpola, Parpola & Brunswig 1977", "Parpola 1994", "Vidale 2005"],
            },
            {
                "name_transliteration": "Ur-Lama (Ur-dLamma)",
                "cuneiform_title_or_context": "dumu me-luḫ-ḫa ('son of Meluhha') in é-duru₅ me-luḫ-ḫa ('Meluhha village') at Guabba/Lagash",
                "artifact_or_text_id": "Ur III Girsu/Lagash administrative archives (ITT / MVN 7 420 = ITT 4 8024)",
                "language": "Sumerian adopted name (acculturated 2nd/3rd-generation Meluhhan diaspora)",
                "morpheme_template": "Ur- ('Servant/Man of') + dLamma (tutelary deity) — prefixing Sumerian PN",
                "syllable_count_pn": 3,
                "scholarly_refs": ["Parpola, Parpola & Brunswig 1977", "Vermaak 2008"],
            },
            {
                "name_transliteration": "Samar & Nanaza",
                "cuneiform_title_or_context": "Individuals explicitly designated as 'Meluhhans' working as royal bezoar-goat shepherds",
                "artifact_or_text_id": "Ur III administrative archives",
                "language": "Non-Sumerian / foreign-sounding anthroponyms in cuneiform transcription",
                "morpheme_template": "2–3 syllable non-Mesopotamian stems (Sa-mar, Na-na-za)",
                "syllable_count_pn": "2–3",
                "scholarly_refs": ["Laursen & Steinkeller 2017"],
            },
            {
                "name_transliteration": "Yaglī-El (ya-ag-li-DINGIR) & Rī'mum",
                "cuneiform_title_or_context": "É.GAL ya-ag-li-DINGIR ARAD dIn-za-ak ša a-kà-rum ('Palace of Yaglī-El, servant of Inzak of Agarum')",
                "artifact_or_text_id": "Royal Mounds of A'ali, Bahrain (steatite vessel inscriptions, Early Dilmun c. 1800–1700 BCE)",
                "language": "Amorite personal names in Babylonian cuneiform (Dilmun royal dynasty)",
                "morpheme_template": "[Prefix/Verb]-El + ARAD ('servant of') + dInzak (theophoric/title formula)",
                "syllable_count_pn": "2–3 (full seal/vessel legend 6–10 signs)",
                "scholarly_refs": ["Marchesi 2017", "Laursen 2017", "Glassner 1984, 2002"],
            },
            {
                "name_transliteration": "Inzak-gāmil / Anzak-gāmil",
                "cuneiform_title_or_context": "Dilmunite theophoric personal name attested in Mesopotamian / Sealand / Gulf archives",
                "artifact_or_text_id": "Old Babylonian / Sealand economic texts & Failaka/Dilmun corpus",
                "language": "Dilmunite-Akkadian hybrid theophoric PN ('Inzak spares')",
                "morpheme_template": "dInzak + Akkadian participle gāmil",
                "syllable_count_pn": 4,
                "scholarly_refs": ["Glassner 1984, 2002"],
            },
            {
                "name_transliteration": "Ea-nāṣir (É-a-na-ṣir) & Ilī-ippašra (Ì-lí-ip-pa-aš-ra)",
                "cuneiform_title_or_context": "Ur-Dilmun copper merchant (Old Babylonian Ur, c. 1750 BCE) & Dilmun governor (Kassite Nippur letters, c. 14th c. BCE)",
                "artifact_or_text_id": "UET V (Ur) & Nippur Kassite correspondence",
                "language": "Akkadian",
                "morpheme_template": "[Deity]-participle / Ì-lí- ('My god') + verb — 4–5 syllables",
                "syllable_count_pn": "4–5",
                "scholarly_refs": ["Oppenheim 1954", "Glassner 2002"],
            },
            {
                "name_transliteration": "Gadd 1 archaic cuneiform legend: 'sak-ku-si' / 'ka-lu' (provisional)",
                "cuneiform_title_or_context": "Circular grey-steatite Indus-style short-horned bull (gaur) & manger seal inscribed in ARCHAIC CUNEIFORM instead of Indus script!",
                "artifact_or_text_id": "Gadd 1932 No. 1 = BM 120573 = U.7683 (excavated at Ur)",
                "language": "Pre-Sargonic/Akkadian cuneiform on a Gulf/Indus-motif seal (direct typological bridge)",
                "morpheme_template": "3 cuneiform signs on an Indus short-horned bull circular stamp seal",
                "syllable_count_pn": 3,
                "scholarly_refs": ["Gadd 1932", "Parpola 1994", "Vidale 2005", "Laursen 2010"],
            },
        ],
        "meluhha_and_dilmun_lexicon_and_goods": [
            {"term": "Me-luḫ-ḫa (KI)", "meaning": "Indus Valley Civilization / Harappan maritime sphere", "syllables": 3, "refs": ["Parpola & Parpola 1975", "Possehl 1996"]},
            {"term": "Dilmun / Tilmun (NI.TUK.KI)", "meaning": "Bahrain / Eastern Arabia / Failaka maritime entrepôt", "syllables": 2, "refs": ["Glassner 2002", "Laursen 2010"]},
            {"term": "dInzak (Enzak) & Meskilak (Panipa)", "meaning": "Chief patron god & goddess of Dilmun (often in ARAD dInzak seal formulas)", "syllables": "2 / 3", "refs": ["Glassner 1984", "Marchesi 2017"]},
            {"term": "gug (sāmtu)", "meaning": "Carnelian (signature etched/long-barrel Indus export bead)", "syllables": 1, "refs": ["Possehl 1996", "Parpola 1994"]},
            {"term": "zú-am-si", "meaning": "Elephant ivory ('tooth of the wild bull/elephant')", "syllables": 3, "refs": ["Possehl 1996", "Laursen & Steinkeller 2017"]},
            {"term": "giš-ab-ba me-luḫ-ḫa", "meaning": "Meluhhan sea-wood / kušabku thorn-tree timber (tracked at Guabba)", "syllables": "3 (+3)", "refs": ["Parpola et al. 1977", "Vermaak 2008"]},
            {"term": "mes-magan (mêsu)", "meaning": "Prized dark hardwood timber from Magan/Meluhha orbit", "syllables": "3", "refs": ["Possehl 1996"]},
            {"term": "ḫa-i̯a-mušen", "meaning": "Exotic palace bird from Meluhha (widely identified as peacock)", "syllables": "2 (+det)", "refs": ["Parpola 1994"]},
            {"term": "dar-mušen", "meaning": "Francolin / multi-colored fowl of the highlands/Meluhha", "syllables": "1 (+det)", "refs": ["Parpola 1994"]},
        ],
        "mesopotamian_vs_heartland_structural_templates": {
            "sumerian_akkadian_personal_names": {
                "morphology": "PREFIXING / THEOPHORIC / COMPOUND (Ur-X, Lú-X, En-X, Šu-X, Ì-lí-X, [PN] ARAD [Deity])",
                "typical_syllable_length": "3–5 syllables for single PN; 6–9 signs for PN + patronymic/title",
                "terminal_constraint": "Open/diverse endings (no single dominant 35% terminal suffix across unrelated PNs)",
            },
            "heartland_harappan_seals": {
                "morphology": "RIGIDLY SUFFIXING (59.9% end in 8 suffixes; 34.8% end in Jar 740 alone)",
                "typical_sign_length": "4.52 signs mean (strict dedup)",
                "terminal_constraint": "Strongly collapsed terminal entropy H_fin (jar 740, comb 400, arrow 520)",
            },
        },
    }

    candidate_hypotheses = {
        "epistemic_warning": (
            "STRICTLY FALSIFIABLE STRUCTURAL HYPOTHESES — NOT DECIPHERMENT OR PHONETIC READINGS. "
            "No individual Indus sign is assigned a proven phonetic value below."
        ),
        "H1_phonetic_syllabic_reencoding_abroad": {
            "statement": (
                "Circular Gulf/Mesopotamian seals (SEAL:C, n=15) and the Susa cylinder (SEAL:CY) exhibit "
                "a simultaneous statistical shift predicted by phonetic/syllabic spelling of foreign "
                "(Sumerian/Akkadian/Dilmunite/Amorite/Elamite) names — extending Rao et al. (2009 PNAS/Science), "
                "Parpola (1994), Vidale (2005), and Laursen (2010)."
            ),
            "supporting_verified_evidence": [
                "Length-matched conditional bigram entropy H1 spikes +3.91σ on Foreign SEAL:C (1.441 vs 1.113±0.084 bits, B=5000 p=0.0002) while vocabulary V is compact (40 vs 43.45) -> higher combinatorial recombination of a reused sign pool.",
                "Terminal entropy H_fin hits the mathematical maximum log2(15)=3.907 bits on Foreign SEAL:C (15/15 unique endings, z=+2.65, B=5000 p=0.0006) as Heartland grammatical suffixes drop from 59.9% to 6.7% (p=2.83e-5).",
                "Controlled 2x2 natural experiment rules out both motif confounding (Heartland Gaur seals retain normal 32.4% final 740 & 59.0% suffixes vs Foreign Gaur 7.1%, p=2.74e-4) and geographic drift (Foreign Square SEAL:S seals retain normal syntax, H1 z=+1.18 n.s., H_fin z=+0.78 n.s., and exact 590->390->740 formulas).",
                "Direction-invariant min(fwd, rev) surprisal remains overwhelmingly higher abroad (7.949 vs 5.831 bits, Mann-Whitney p=1.18e-7).",
            ],
            "alternative_competing_hypotheses": [
                "Semi-literate or localized Dilmunite merchant pidgin / emblem drift (Laursen 2010): breakaway Harappan/Dilmunite seal cutters reusing familiar iconic signs (human figures, numerals) outside mainland grammatical rules without full syllabic encoding.",
                "Corporate/guild-marker system (Vidale 2005): short-horned bull (Gaur) + circular seal + 091/090 encoding merchant family/trade-house identity rather than full spoken sentences.",
            ],
        },
        "H2_anthropomorphic_prefix_slot_091_090": {
            "statement": (
                "The 15.7x surge of the human-figure triad {090, 091, 093} on Foreign SEAL:C (12/15 = 80.0% vs 5.1% Heartland, "
                "p=2.30e-13) and the initial-prefix positioning of 091 (Twins) on 3/11 verified-direction SEAL:C seals "
                "(depressing H_init by -2.22σ, B=5000 p=0.0316) structurally mirror prefixing Mesopotamian personal-name/title markers "
                "(e.g., Sumerian Lú- 'Man of', Ur- 'Servant of', Akkadian Šu- 'He of', or Dilmunite ARAD 'Servant of')."
            ),
            "critical_falsification_caveat": (
                "While 091's initial position is direction-verified (R/L on 2153.1 Luristan, 5227.1 & 5228.1 Karzakan), "
                "all three SEAL:C seals with initial 090 (147.1 Failaka, 5225.1 & 5231.1 Ur) have direction=UNK; reversing "
                "5225.1 (090 001 -> 001 090) restores 090 to terminal position matching verified-R/L Failaka 148.1 (... 001->090) "
                "and unprovenanced Gulf seal 5223.1 (... 001->090). However, 090 is still unambiguously MEDIAL on verified "
                "Qala'at al-Bahrain 3863.1 (L/R) and Susa 3881.1 (R/L), confirming partial syntactic displacement."
            ),
        },
        "H3_seal_by_seal_structural_constraint_mapping": [
            {
                "seal_id": "5232.1",
                "site": "Janabiyah (Bahrain / Dilmun)",
                "type_motif": "SEAL:C · Gaur · L/R · Complete",
                "seq": "071 -> 031 -> 831 -> 055 -> 121 -> 099 -> 055",
                "metrics": "len=7, z_fwd=+3.18, z_rev=+2.67, unseen=8/8 (100%)",
                "structural_constraints": [
                    "Longest complete circular Gulf seal (7 signs); anomalous in BOTH reading directions.",
                    "Exact periodic sign reduplication of 055 (12-strokes) at slots 4 & 7: [071-031-831]-055-[121-099]-055 (3+1+2+1 two-clause symmetry!).",
                    "Contains two ultra-rare anthropomorphic figures (121 & 099, each freq=2 in entire 5,679 corpus) + Gulf-enriched 831 (16.7x, shared with Luristan 2153.1).",
                    "Cuneiform structural parallel (length & bimorphemic repetition): matches a 6–7 sign two-part Dilmunite/Akkadian seal legend ([PN] + [patronymic/title] or repeated syllable/case marker -X ... -X).",
                ],
            },
            {
                "seal_id": "3863.1",
                "site": "Qala'at al-Bahrain (Dilmun capital)",
                "type_motif": "SEAL:C · Gaur · L/R · Complete",
                "seq": "190 -> 060 -> 055 -> 090 -> 160",
                "metrics": "len=5, z_fwd=+2.57, z_rev=+1.34, unseen=5/6 (83%)",
                "structural_constraints": [
                    "Contains Gulf-exclusive bigram 055 -> 090 (0 occurrences in entire Indus Heartland; shared ONLY with unprovenanced cylinder seal 3888.1).",
                    "Initial 190 (spider/insect) is shared with unprovenanced circular seal 3887.1 (190-091-603); terminal 160 (yoke-carrier) is shared with unprovenanced Gulf Gaur seal 5223.1.",
                    "5-sign length with zero Heartland suffixes aligns with a 4–5 syllable Dilmunite/Akkadian name (cf. Ilī-ippašra, Inzak-gāmil, Šu-ilīšu).",
                ],
            },
            {
                "seal_id": "3881.1 & 3882.1",
                "site": "Susa (Elam / Iranian Plateau)",
                "type_motif": "3881.1 SEAL:C (len=5, N) & 3882.1 SEAL:CY cylinder (len=8, Y) · both Gaur · R/L",
                "seq": "3881.1: 365->090->407->604->700 | 3882.1: 924->001->319->031->055->002->150->416",
                "metrics": "3881.1: z=+2.13 (5/6 unseen) | 3882.1: z=+1.48, z_rev=+1.62 (5/9 unseen)",
                "structural_constraints": [
                    "3882.1 is an 8-sign Elamite/Mesopotamian CYLINDER seal combining the Indus Gaur motif with Gulf-enriched signs 924 (33.3x, shared with Ur 3897.1), 031->055, short numerals 001/002, and yoke-man 150.",
                    "8-sign length matches a full two-register Mesopotamian/Elamite cylinder legend (PN + title/filiation; cf. Šu-ilīšu's 9-sign Louvre cylinder seal AO 22310).",
                ],
            },
            {
                "seal_id": "3897.1 & 3898.1",
                "site": "Ur (Southern Mesopotamia)",
                "type_motif": "Both SEAL:C · Gaur · R/L · Complete",
                "seq": "3897.1: 528->220->924->340->093 | 3898.1 (U.17649): 415->803->001->328->004->002",
                "metrics": "3897.1: len=5, z=+1.84 (4/6 unseen) | 3898.1: len=6, z=+1.72 (5/7 unseen)",
                "structural_constraints": [
                    "3897.1 contains the corpus-wide HAPAX quadruped 340 (freq=1), plus Gulf-enriched 924 (shared with Susa cylinder 3882.1) and 093 (53.3x, shared with Saar 5229.1). Length 5 fits a 4–5 syllable Sumerian/Akkadian personal name.",
                    "3898.1 (U.17649) ends in a unique stroke-dense block 001->328->004->002 (004->002 occurs 0x in Heartland; 328 is freq=2). Half of its signs (3/6) are short strokes {001, 004, 002}, consistent either with phonetic reuse of frequent CV/V stroke signs or trade/metrological notation.",
                ],
            },
            {
                "seal_id": "2153.1, 5227.1, 5228.1 (The '091 + 3-sign' Quartet Template)",
                "site": "Luristan (2153.1) & Karzakan, Bahrain (5227.1, 5228.1) + Saar (5229.1)",
                "type_motif": "All SEAL:C · Gaur (except Saar blank) · R/L · Complete",
                "seq": "2153.1: 091->840->413->831 | 5227.1: 091->031->455->220 | 5228.1: 091->032->001->033 | 5229.1: 055->220->091->001->093->031",
                "metrics": "2153.1 z=+1.84; 5229.1 z=+1.36; 5227.1 z=+0.59; 5228.1 z=+0.30",
                "structural_constraints": [
                    "Three independent complete circular Gaur seals share an identical 4-sign prefixing frame [091 + X + Y + Z] (and 5229.1 embeds [091 + 001 + 093 + 031] right after [055-220]!).",
                    "Notably, 5227.1, 5228.1, and 5229.1 all place a stroke numeral immediately after 091 (091->031, 091->032, 091->001) — a pattern appearing only ONCE in the entire 4,531-line Heartland (Harappa 697.1).",
                    "Structurally matches a 1-morpheme human/title prefix (Lú- / Ur- / Šu- / guild marker) + 3-sign personal name or numeral-qualified trade title.",
                ],
            },
            {
                "seal_id": "CONTROL: 1971.1 (Kish) & 4155.1 (Gonur Depe)",
                "site": "Mesopotamia (Kish) & Central Asia (Gonur Depe)",
                "type_motif": "SEAL:S (Square!) · Bull1:L & Elep · R/L · Complete",
                "seq": "1971.1: 416->840->060->003->220->590->390->740 | 4155.1: 285->002->235->220->125->590->390->740",
                "metrics": "1971.1: len=8, z=-0.66 (1/9 unseen) | 4155.1: len=8, z=-0.35 (1/9 unseen)",
                "structural_constraints": [
                    "BOTH 8-sign square seals found thousands of kilometers apart terminate in the exact #1 Heartland trigram 590 -> 390 -> 740 (preceded by plain fish 220 in Kish and 220->125 in Gonur Depe).",
                    "Provides a decisive falsification control: authentic Harappan merchants carrying square seals abroad kept 100% canonical Heartland suffix syntax. Therefore, the high surprisal of circular SEAL:C seals is NOT an artifact of distance or sample noise.",
                ],
            },
        ],
    }

    return {
        "verified_cuneiform_reference_table": verified_cuneiform_corpus,
        "candidate_hypotheses_falsifiable": candidate_hypotheses,
    }


# ---------------------------------------------------------------------------
# 5. MUSEUM-GRADE VISUAL PLATE RENDERER (TOP 18 FOREIGN SEALS)
# ---------------------------------------------------------------------------
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_rosetta_plate(
    top18: list[dict[str, Any]],
    glyphs_df: pd.DataFrame,
    audit: dict[str, Any],
    structural: dict[str, Any],
    out_paths: list[Path],
) -> None:
    """
    Render a museum-grade, zero-overlap catalogue plate of the Top 18 usable foreign
    inscriptions sorted by Heartland surprisal z-score, with genuine Indus glyphs
    rendered via sk_indus_script-webfont.ttf in chronological reading order (L->R)
    and ICIT numeric IDs directly underneath every glyph.
    """
    g_cp: dict[str, list[int]] = {}
    for _, r in glyphs_df.iterrows():
        cp_s = r["codepoints"].strip()
        if cp_s:
            g_cp[r["sign_id"]] = [int(h, 16) for h in cp_s.split() if h]

    # Foreign-enriched signature signs (highlighted in soft warm gold)
    FOREIGN_SIG_HIGHLIGHT = {"091", "090", "093", "055", "924", "831"}

    # Palette (Museum Parchment / Navy / Terracotta / Teal / Indigo)
    BG = "#FAF8F4"
    HEADER_BG = "#162436"
    GOLD = "#C89B3C"
    ROW_EVEN = "#FFFFFF"
    ROW_ODD = "#F5F2EB"
    BORDER = "#DDD6C7"
    INK = "#1B2430"
    MUTED = "#5A6577"

    REGION_COLORS = {
        "Persian Gulf": ("#0F766E", "#E6F4F1"),
        "Mesopotamia": ("#B45309", "#FEF3C7"),
        "Iranian Plateau": ("#4338CA", "#EEF2FF"),
        "Central Asia": ("#6D28D9", "#F3E8FF"),
    }

    # Fonts
    indus_ttf = str(RAW_DIR / "sk_indus_script-webfont.ttf")
    f_indus_lg = load_font(indus_ttf, 42)
    f_indus_md = load_font(indus_ttf, 34)
    f_title = load_font("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 25)
    f_subtitle = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    f_colhdr = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
    f_site = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    f_meta = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    f_badge = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 11)
    f_mono_b = load_font("/System/Library/Fonts/Menlo.ttc", 13)
    f_mono = load_font("/System/Library/Fonts/Menlo.ttc", 11)
    f_foot = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 12)

    # Geometry
    W = 1720
    HEADER_H = 114
    COLHDR_H = 38
    ROW_H = 84
    FOOTER_H = 68
    H = HEADER_H + COLHDR_H + len(top18) * ROW_H + FOOTER_H + 14

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Column X coordinates
    X_RANK = 18       # width 92  (#01 + ID)
    X_SITE = 114      # width 248 (Site + Region pill + dir/comp)
    X_MOTIF = 368     # width 178 (Motif + Object Type)
    X_METRIC = 552    # width 270 (z-score bar + unseen + bits + class)
    X_GLYPH = 834     # width 868 (up to 9 glyphs * 94px pitch = 846px)
    GLYPH_PITCH = 94
    GLYPH_BOX_W = 86
    GLYPH_BOX_H = 68

    # --- HEADER BANNER ---
    draw.rectangle([0, 0, W, HEADER_H], fill=HEADER_BG)
    draw.rectangle([0, HEADER_H - 4, W, HEADER_H], fill=GOLD)
    draw.text(
        (24, 18),
        'INDUS-DEDUCT Katman 4 — Yabanci Bolge "Rosetta" Muhurleri (Ilk 18 Surpriz Anomalisi)',
        fill="#FFFFFF",
        font=f_title,
    )
    hl_m = audit["heartland_cv_baseline"]["mean_surprisal_bits"]
    hl_s = audit["heartland_cv_baseline"]["std_surprisal_bits"]
    p_fwd = audit["foreign_vs_heartland_surprisal"]["mann_whitney_p_fwd"]
    p_inv = audit["foreign_vs_heartland_surprisal"]["mann_whitney_p_dir_invariant"]
    draw.text(
        (24, 54),
        f"Heartland 5-Kat CV Tabani: {hl_m:.2f} +/- {hl_s:.2f} bit/gecis  |  Yabanci (n=34) Ort: "
        f"{audit['foreign_vs_heartland_surprisal']['foreign_mean_bits']:.2f} bit  |  "
        f"Mann-Whitney p = {p_fwd:.2e} (yon-bagimsiz p = {p_inv:.2e})",
        fill="#D6E0EB",
        font=f_subtitle,
    )
    h1_z = structural["length_matched_bootstrap"]["Foreign_SEAL_C__vs__HL_strict_dedup"]["H1"]["z_score"]
    h1_p = structural["length_matched_bootstrap"]["Foreign_SEAL_C__vs__HL_strict_dedup"]["H1"]["p_greater"]
    draw.text(
        (24, 78),
        f"Dairesel Korfez/Mezopotamya (SEAL:C) muhurlerinde 740 Kavanoz soneki coker (%34.8 -> %6.7), "
        f"090/091/093 insan isaretleri 15.7x artar (p=2.3e-13) ve H1 kosullu entropi {h1_z:+.2f} sigma yukselir (p={h1_p:.4f}).",
        fill="#F3D89B",
        font=f_subtitle,
    )

    # --- COLUMN HEADER STRIP ---
    y_ch = HEADER_H
    draw.rectangle([0, y_ch, W, y_ch + COLHDR_H], fill="#EAE5DA")
    draw.line([0, y_ch + COLHDR_H - 1, W, y_ch + COLHDR_H - 1], fill="#C9C0AE", width=2)
    draw.text((X_RANK + 4, y_ch + 11), "SIRA / ID", fill=INK, font=f_colhdr)
    draw.text((X_SITE + 4, y_ch + 11), "YERLESIM & BOLGE", fill=INK, font=f_colhdr)
    draw.text((X_MOTIF + 4, y_ch + 11), "MOTIF & OBJE TIPI", fill=INK, font=f_colhdr)
    draw.text((X_METRIC + 4, y_ch + 11), "HEARTLAND SURPRIZ (z / GORULMEMIS)", fill=INK, font=f_colhdr)

    # Reading order arrow header
    draw.text((X_GLYPH + 4, y_ch + 11), "INDUS YAZITI  —  Okuma Sirasi ->  (soldan saga, altinda ICIT no.)", fill="#7C2D12", font=f_colhdr)

    # --- 18 SEAL ROWS ---
    y0 = y_ch + COLHDR_H
    for idx, rec in enumerate(top18):
        ry = y0 + idx * ROW_H
        bg = ROW_EVEN if idx % 2 == 0 else ROW_ODD
        draw.rectangle([0, ry, W, ry + ROW_H], fill=bg)
        draw.line([0, ry + ROW_H - 1, W, ry + ROW_H - 1], fill=BORDER, width=1)

        reg = rec["region"]
        rc_fg, rc_bg = REGION_COLORS.get(reg, ("#475569", "#F1F5F9"))
        # Left region accent bar
        draw.rectangle([0, ry, 7, ry + ROW_H - 1], fill=rc_fg)

        # 1. Rank & ID
        rank_str = f"#{rec['rank']:02d}"
        draw.text((X_RANK + 4, ry + 16), rank_str, fill=rc_fg, font=f_site)
        draw.text((X_RANK + 4, ry + 42), f"ID {rec['inscription_id']}", fill=MUTED, font=f_mono)

        # 2. Site & Region pill & Direction/Completeness
        site_txt = rec["site"]
        draw.text((X_SITE + 4, ry + 12), site_txt, fill=INK, font=f_site)

        # Region pill
        rb = draw.textbbox((0, 0), reg, font=f_badge)
        rw = (rb[2] - rb[0]) + 14
        draw.rounded_rectangle([X_SITE + 4, ry + 35, X_SITE + 4 + rw, ry + 55], radius=4, fill=rc_bg, outline=rc_fg, width=1)
        draw.text((X_SITE + 11, ry + 39), reg, fill=rc_fg, font=f_badge)

        comp_lbl = {"Y": "Tam", "N": "Eksik", "?": "Belirsiz"}.get(rec["complete"], rec["complete"])
        dir_lbl = rec["direction"]
        draw.text((X_SITE + 4, ry + 61), f"Yon: {dir_lbl}  |  Durum: {comp_lbl}  |  L={rec['len']}", fill=MUTED, font=f_meta)

        # 3. Motif & Object Type
        sym = rec["symbol"] if rec["symbol"] else "(Motifsiz)"
        otype = rec["object_type"]
        is_gaur = sym == "Gaur"
        is_circ = otype in {"SEAL:C", "SEAL:CY"}

        m_fg = "#92400E" if is_gaur else "#334155"
        m_bg = "#FEF3C7" if is_gaur else "#F1F5F9"
        mb = draw.textbbox((0, 0), sym, font=f_badge)
        mw = max(76, (mb[2] - mb[0]) + 16)
        draw.rounded_rectangle([X_MOTIF + 4, ry + 15, X_MOTIF + 4 + mw, ry + 37], radius=4, fill=m_bg, outline=m_fg, width=1)
        draw.text((X_MOTIF + 12, ry + 20), sym, fill=m_fg, font=f_badge)

        o_fg = "#0F766E" if is_circ else "#475569"
        o_bg = "#CCFBF1" if is_circ else "#E2E8F0"
        o_txt = f"{otype} (Daire)" if otype == "SEAL:C" else (f"{otype} (Silindir)" if otype == "SEAL:CY" else otype)
        ob = draw.textbbox((0, 0), o_txt, font=f_badge)
        ow = (ob[2] - ob[0]) + 14
        draw.rounded_rectangle([X_MOTIF + 4, ry + 45, X_MOTIF + 4 + ow, ry + 66], radius=4, fill=o_bg, outline=o_fg, width=1)
        draw.text((X_MOTIF + 11, ry + 50), o_txt, fill=o_fg, font=f_badge)

        # 4. Surprisal z-score gauge + Unseen bigram ratio + classification
        z = rec["z_score_vs_heartland"]
        is_rosetta = z >= 1.5
        bar_col = "#B91C1C" if is_rosetta else "#D97706"
        tag_txt = "ROSETTA" if is_rosetta else "HIBRIT"
        tag_bg = "#FEE2E2" if is_rosetta else "#FEF3C7"

        draw.text((X_METRIC + 4, ry + 11), f"z = {z:+.2f}", fill=bar_col, font=f_site)
        draw.text((X_METRIC + 94, ry + 13), f"({rec['surprisal_bits']:.2f} bit)", fill=MUTED, font=f_mono)

        # Classification mini-pill
        tb = draw.textbbox((0, 0), tag_txt, font=f_badge)
        tw = (tb[2] - tb[0]) + 12
        draw.rounded_rectangle([X_METRIC + 200, ry + 11, X_METRIC + 200 + tw, ry + 29], radius=4, fill=tag_bg, outline=bar_col, width=1)
        draw.text((X_METRIC + 206, ry + 14), tag_txt, fill=bar_col, font=f_badge)

        # Horizontal z-score bar (scale 0..3.5)
        bx0, by0, bw_max, bh = X_METRIC + 4, ry + 37, 250, 12
        draw.rounded_rectangle([bx0, by0, bx0 + bw_max, by0 + bh], radius=4, fill="#E5E1D8")
        fill_w = max(8, min(bw_max, int(bw_max * max(z, 0.0) / 3.5)))
        draw.rounded_rectangle([bx0, by0, bx0 + fill_w, by0 + bh], radius=4, fill=bar_col)
        # Reference tick at z = 1.5
        x_thr = bx0 + int(bw_max * 1.5 / 3.5)
        draw.line([x_thr, by0 - 2, x_thr, by0 + bh + 2], fill="#1E293B", width=1)

        ub_pct = int(round(100 * rec["unseen_bigram_ratio"]))
        draw.text(
            (X_METRIC + 4, ry + 56),
            f"Gorulmemis bigram: {rec['unseen_bigrams']} (%{ub_pct})  |  z_ters={rec['z_score_rev']:+.2f}",
            fill=MUTED,
            font=f_meta,
        )

        # 5. Actual Indus Glyphs in chronological reading order (L -> R)
        unseen_set = set(rec["unseen_signs"])
        for g_i, sid in enumerate(rec["tokens"]):
            gx0 = X_GLYPH + g_i * GLYPH_PITCH
            gy0 = ry + 8
            gx1 = gx0 + GLYPH_BOX_W
            gy1 = gy0 + GLYPH_BOX_H

            # Cell styling
            if sid in unseen_set:
                cell_fill = "#FFF1F2"
                cell_out = "#BE123C"
                out_w = 2
            elif sid in FOREIGN_SIG_HIGHLIGHT:
                cell_fill = "#FEF9E7"
                cell_out = "#C89B3C"
                out_w = 2
            else:
                cell_fill = "#FFFFFF"
                cell_out = "#D6CFC2"
                out_w = 1

            draw.rounded_rectangle([gx0, gy0, gx1, gy1], radius=6, fill=cell_fill, outline=cell_out, width=out_w)

            # Glyph rendering area: top 48px of the 68px box, ID label in bottom 18px
            cps = g_cp.get(sid, [])
            if cps:
                g_str = "".join(chr(cp) for cp in cps)
                # Use slightly smaller font for multi-codepoint wide compounds (like 617)
                use_font = f_indus_md if len(cps) > 1 else f_indus_lg
                bb = draw.textbbox((0, 0), g_str, font=use_font)
                gw = bb[2] - bb[0]
                gh = bb[3] - bb[1]
                # Shrink if wider than box interior
                if gw > GLYPH_BOX_W - 10 or gh > 44:
                    use_font = f_indus_md
                    bb = draw.textbbox((0, 0), g_str, font=use_font)
                    gw = bb[2] - bb[0]
                    gh = bb[3] - bb[1]
                tx = gx0 + (GLYPH_BOX_W - gw) / 2 - bb[0]
                ty = gy0 + 3 + (44 - gh) / 2 - bb[1]
                draw.text((tx, ty), g_str, fill="#111827", font=use_font)
            else:
                # Crisp placeholder for 643 (genuinely absent from sk_indus_script TTF)
                draw.rounded_rectangle([gx0 + 14, gy0 + 8, gx1 - 14, gy0 + 42], radius=3, fill="#F3F4F6", outline="#9CA3AF", width=1)
                ntxt = "N/A*"
                nb = draw.textbbox((0, 0), ntxt, font=f_mono)
                draw.text((gx0 + (GLYPH_BOX_W - (nb[2] - nb[0])) / 2, gy0 + 18), ntxt, fill="#6B7280", font=f_mono)

            # Subtle separator line and centered ICIT numeric ID underneath
            draw.line([gx0 + 8, gy0 + 49, gx1 - 8, gy0 + 49], fill="#E5E0D5", width=1)
            id_lbl = f"{sid}*" if not cps else sid
            ib = draw.textbbox((0, 0), id_lbl, font=f_mono_b)
            iw = ib[2] - ib[0]
            id_col = "#9F1239" if sid in unseen_set else ("#92400E" if sid in FOREIGN_SIG_HIGHLIGHT else "#1E3A5F")
            draw.text((gx0 + (GLYPH_BOX_W - iw) / 2, gy0 + 51), id_lbl, fill=id_col, font=f_mono_b)

            # Small connecting arrow between adjacent glyphs
            if g_i < len(rec["tokens"]) - 1:
                ax = gx1 + 1
                ay = gy0 + 26
                draw.text((ax, ay - 6), ">", fill="#B0A798", font=f_mono)

    # --- FOOTER LEGEND ---
    fy = y0 + len(top18) * ROW_H + 8
    draw.rectangle([0, fy - 4, W, H], fill="#EFECE4")
    draw.line([0, fy - 4, W, fy - 4], fill="#C9C0AE", width=2)

    # Legend swatches
    lx = 24
    # Gold swatch
    draw.rounded_rectangle([lx, fy + 8, lx + 20, fy + 24], radius=3, fill="#FEF9E7", outline="#C89B3C", width=2)
    draw.text((lx + 28, fy + 9), "Yabanci-zenginlestirilmis imza isaretleri (091, 090, 093, 055, 831, 924)", fill=INK, font=f_foot)
    lx += 485
    # Red swatch (dynamically collect all unseen_signs shown in Top 18)
    all_unseen_top18 = sorted({s for rec in top18 for s in rec["unseen_signs"]})
    unseen_lbl = ", ".join(all_unseen_top18)
    draw.rounded_rectangle([lx, fy + 8, lx + 20, fy + 24], radius=3, fill="#FFF1F2", outline="#BE123C", width=2)
    draw.text((lx + 28, fy + 9), f"Heartland korpusunda hic gorulmeyen isaretler ({unseen_lbl})", fill=INK, font=f_foot)
    lx += 485
    # Region swatches
    for rname, (rfg, _) in [("Persian Gulf", REGION_COLORS["Persian Gulf"]), ("Mesopotamia", REGION_COLORS["Mesopotamia"]), ("Iranian Plateau", REGION_COLORS["Iranian Plateau"])]:
        draw.rectangle([lx, fy + 10, lx + 12, fy + 22], fill=rfg)
        draw.text((lx + 17, fy + 9), rname, fill=INK, font=f_foot)
        lx += 135

    draw.text(
        (24, fy + 34),
        "Not: Glifler sk_indus_script-webfont.ttf ile kronolojik okuma sirasinda (soldan saga) cizilmistir. "
        "*643 (Kish 1970.1): nadir ICIT isareti (f=4), TTF font dosyasinda kod noktasi yoktur. "
        "z_ters = ters okuma yonundeki Heartland surpriz z-skoru (yon-duyarlilik kontrolu).",
        fill=MUTED,
        font=f_foot,
    )

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
def main() -> None:
    ins_df, lines_df, glyphs_df, l12 = load_warehouse()
    audit, usable_ranked, _, _ = score_foreign_corpus(ins_df, lines_df)
    structural = run_structural_comparisons(lines_df, n_boot=5000)
    patterns = analyze_foreign_patterns(lines_df)
    cuneiform_and_hyps = build_cuneiform_reference_and_hypotheses()

    # Cross-check exact reproduction against layer1_2_findings.json
    l12_f = l12.get("layer2_to_4_foreign_surprisal", {})
    repro_check = {
        "layer1_2_hl_mean": l12_f.get("heartland_cv_mean_surprisal"),
        "layer4_hl_mean": audit["heartland_cv_baseline"]["mean_surprisal_bits"],
        "layer1_2_foreign_mean": l12_f.get("foreign_mean_surprisal"),
        "layer4_foreign_mean": audit["foreign_vs_heartland_surprisal"]["foreign_mean_bits"],
        "exact_match": (
            l12_f.get("heartland_cv_mean_surprisal") == audit["heartland_cv_baseline"]["mean_surprisal_bits"]
            and l12_f.get("foreign_mean_surprisal") == audit["foreign_vs_heartland_surprisal"]["foreign_mean_bits"]
        ),
    }

    top18 = usable_ranked[:18]
    png_primary = REPORTS_DIR / "rosetta_foreign_seals.png"
    render_rosetta_plate(top18, glyphs_df, audit, structural, [png_primary, BRAIN_PNG_PATH])

    report = {
        "metadata": {
            "layer": "Layer 4 — Foreign-Region 'Rosetta' Seals Computational Epigraphy",
            "corpus_source": "data/processed/inscriptions_clean.csv + lines_clean.csv + glyphs_clean.csv",
            "reproduction_check_vs_layer1_2": repro_check,
            "outputs": {
                "json_report": str(REPORTS_DIR / "layer4_rosetta_findings.json"),
                "png_primary": str(png_primary),
                "png_brain_copy": str(BRAIN_PNG_PATH),
            },
        },
        "section_A_verified_statistical_facts": {
            "corpus_audit_and_surprisal": audit,
            "top_18_rosetta_ranked": top18,
            "native_harappan_abroad_bottom_8": list(reversed(usable_ranked[-8:])),
            "structural_comparisons": structural,
            "foreign_patterns_and_formulas": patterns,
        },
        "section_B_verified_cuneiform_reference": cuneiform_and_hyps["verified_cuneiform_reference_table"],
        "section_C_candidate_hypotheses_falsifiable": cuneiform_and_hyps["candidate_hypotheses_falsifiable"],
    }

    out_json = REPORTS_DIR / "layer4_rosetta_findings.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Executive console summary
    print("=== LAYER 4 ROSETTA FOREIGN SEALS — VERIFIED SUMMARY ===")
    print(f"Reproduction vs Layer 1-2 exact_match: {repro_check['exact_match']}")
    print(
        f"Foreign corpus: {audit['total_foreign_inscriptions']} inscriptions / "
        f"{audit['total_foreign_parsed_lines']} lines -> {audit['usable_lines_ge2_signs']} usable (>=2 signs)"
    )
    fvh = audit["foreign_vs_heartland_surprisal"]
    print(
        f"Surprisal: Foreign={fvh['foreign_mean_bits']} vs HL_CV={audit['heartland_cv_baseline']['mean_surprisal_bits']} bits "
        f"(p={fvh['mann_whitney_p_fwd']}; dir-invariant min p={fvh['mann_whitney_p_dir_invariant']})"
    )
    print(f"Foreign SEAL:C > Foreign SEAL:S surprisal Mann-Whitney p = {fvh['foreign_SEAL_C_gt_SEAL_S_mw_p']}")

    print("\n--- CONTROLLED 2x2 NATURAL EXPERIMENT (KEY SIGNALS) ---")
    for cname, tbl in structural["controlled_2x2_fisher_tests"].items():
        print(f"[{cname}]")
        for fk in ["twin_man_triad_090_091_093_any", "anthropomorphic_090_164_any", "sign_055_12_strokes_any", "jar_740_final", "hl_8_suffixes_final_excl_090"]:
            v = tbl[fk]
            print(f"  {fk:32s}: A={v['group_A']:16s} vs B={v['group_B']:18s} | lift={str(v['lift']):>6s} | p={v['fisher_two_sided_p']:.3e}")

    print("\n--- LENGTH-MATCHED BOOTSTRAP (B=5000) KEY METRICS ---")
    for bkey in ["Foreign_SEAL_C__vs__HL_strict_dedup", "Foreign_Gaur__vs__HL_Gaur_dedup", "Foreign_SEAL_S__vs__HL_strict_dedup"]:
        b = structural["length_matched_bootstrap"][bkey]
        print(f"[{bkey}] (n={b['n_inscriptions']}, toks={b['n_tokens']})")
        for mk in ["V", "H0", "H_init", "H_fin", "H1"]:
            m = b[mk]
            print(f"  {mk:6s}: obs={m['observed']:.3f} vs base={m['matched_baseline_mean']:.3f}±{m['matched_baseline_std']:.3f} | z={m['z_score']:+5.2f} | p_hi={m['p_greater']:.4f} p_lo={m['p_less']:.4f}")

    print(f"\nSaved JSON : {out_json}")
    print(f"Saved PNG  : {png_primary}")
    print(f"Copied PNG : {BRAIN_PNG_PATH}")


if __name__ == "__main__":
    main()
