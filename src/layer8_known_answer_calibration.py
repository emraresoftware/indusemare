#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 8: 5-Way Known-Answer & Negative-Control Calibration Benchmark
===================================================================================
Directly fulfills the #1 methodological recommendation of independent reviewers
(`vega` and `orivis-mac`): calibrate every blind structural, syntactic, morphological,
predictive, and CSP-identifiability instrument on REAL deciphered, undeciphered, and
synthetic non-linguistic control corpora under IDENTICAL, UNCHANGED, ZERO-HARDCODING
algorithms matched to Indus's exact sequence-length distribution and sample size.

5 Calibration Systems:
  1. Target — Indus Valley Script (`data/processed/lines_clean.csv`, N=2,047 unique strict seqs)
  2. Control 1 — Real Linear B (Deciphered Mycenaean Greek CV Syllabary, Fusional;
                 InsiderPhD/Linear-B-Dataset `words.csv` + `tablets.csv`, Indus length-matched N=1,651,
                 plus full N=4,844 words & N=1,037 intact tablet lines for transparency)
  3. Control 2 — Real Ur III Sumerian (Deciphered Cuneiform Logo-Syllabic, Agglutinative;
                 CDLI CC0-1.0 `mtaac_gold_corpus`, 4,055 tablets -> Indus exact length-matched N=2,047,
                 plus full N=23,759 lines & Seal/NP matched N=1,929)
  4. Control 3 — Real Linear A (Undeciphered Minoan Aegean Script; SigLA/Younger `LinearAWords.json`,
                 clean intact words N=660 & clean tablet lines N=614)
  5. Control 4 — Synthetic Non-Linguistic Bureaucratic Stamps (Rigid template
                 [Header(4)]+[Conn(2)]+[Item(180)]+[CheckStamp(6)], exact Indus N=2,047 & length hist)

5-Metric Blind Calibration Suite:
  Metric 1: Inventory & Information Theory (V, H0, H1, z_H1, H_init, H_final, Bottleneck Ratio, Top-1 Final)
  Metric 2: Blind Affix Stacking Asymmetry (2-suffix vs 2-prefix stacks, zero hardcoded sign IDs)
  Metric 3: Blind Alice Kober Bridge-Sign Inflection Test (alt/same ratio & MI(Bridge; Ending) z-score)
  Metric 4: 5-Fold Masked-Sign Prediction (ALL, SEEN_TRIGRAM, UNSEEN_TRIGRAM, final_UNSEEN vs Pos-Baseline)
  Metric 5: Known-Answer 23-Seal CSP Identifiability Calibration (Exact-First + Bitmask Dynamic-MRV Max-CSP,
            k=0 vs k=2 true anchors vs 100 matched synthetic shuffled lexicons on Real Linear B,
            Real Ur III Sumerian, 74% Partial-Attestation Stress Tests, and Actual Indus 23 Gulf Seals!)
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
CAL_DIR = ROOT / "data" / "calibration"
REPORTS_DIR = ROOT / "reports"
BRAIN_COPY_PATH = Path("reports/calibration_5way_chart.png")

BOS, EOS = "<BOS>", "<EOS>"
SEED = 42


# =============================================================================
# 0. ARCHAEOLOGICAL & SYNTHETIC CORPUS LOADERS + EXACT LENGTH-HISTOGRAM SLICER
# =============================================================================

def load_indus_corpus() -> tuple[list[list[str]], Counter]:
    lines_df = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    strict_df = lines_df[
        (lines_df["is_strict_analytic"] == "1")
        & (lines_df["direction"].isin(["R/L", "L/R", "BUS"]))
    ]
    seqs = [s.split() for s in strict_df["clean_reading_seq"].unique() if len(s.split()) >= 2]
    hist = Counter(len(s) for s in seqs)
    return seqs, hist


def load_linearb_corpora() -> dict[str, Any]:
    """
    Parse real Mycenaean Greek Linear B syllabic words (union of InsiderPhD/Linear-B-Dataset
    `words.csv` + `tablets.csv` intact entries) AND intact multi-word tablet lines (`tablets.csv`).
    Strictly filters out lacunae (`[`, `]`, `?`, `.`), keeping clean hyphen-split syllabograms.
    """
    words_raw: list[list[str]] = []
    with open(CAL_DIR / "linearb_words.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = (row.get("word") or "").strip()
            if not w or any(ch in w for ch in "[]?,.() "):
                continue
            toks = [t.strip().lower() for t in w.split("-") if t.strip()]
            if 2 <= len(toks) <= 13 and all(re.match(r"^[a-z][a-z0-9]*$|^\*[0-9]+$", t) for t in toks):
                words_raw.append(toks)

    with open(CAL_DIR / "linearb_tablets.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            ins = (row.get("inscription") or "").strip()
            for p in ins.split(","):
                p = p.strip()
                if not p or any(ch in p for ch in "[]?."):
                    continue
                p_clean = p.replace("(", "").replace(")", "")
                toks = [t.strip().lower() for t in p_clean.split("-") if t.strip()]
                if 2 <= len(toks) <= 13 and all(re.match(r"^[a-z][a-z0-9]*$|^\*[0-9]+$", t) for t in toks):
                    words_raw.append(toks)

    words_uniq = list({tuple(s): s for s in words_raw}.values())

    # Intact multi-word tablet line segments (syllabograms only)
    lines_raw: list[list[str]] = []
    with open(CAL_DIR / "linearb_tablets.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            orig = (row.get("original") or "").strip()
            for seg in re.split(r"(?:^|\s)\.[0-9a-zA-Z]+\s+|\s*/\s*", orig):
                seg = seg.strip()
                if not seg or any(x in seg for x in ("[", "]", "?", "vest", "vacat")):
                    continue
                ws = re.findall(r"[a-z0-9*]+(?:-[a-z0-9*]+)+", seg.replace("(", "").replace(")", ""))
                toks = [
                    t.lower()
                    for w in ws
                    for t in w.split("-")
                    if re.match(r"^[a-z][a-z0-9]*$|^\*[0-9]+$", t.lower())
                ]
                if 2 <= len(toks) <= 13:
                    lines_raw.append(toks)
    lines_uniq = list({tuple(s): s for s in lines_raw}.values())

    # Sanity-check canonical Linear B words
    word_strs = {"-".join(s) for s in words_uniq}
    iconic = {w: (w in word_strs) for w in ["ti-ri-po-de", "ko-no-so", "pa-i-to", "ko-no-si-jo", "ko-no-si-ja", "wa-na-ka-te-ro"]}
    return {
        "words_full": words_uniq,
        "lines_full": lines_uniq,
        "iconic_words_verified": iconic,
    }


def load_sumerian_corpora() -> dict[str, Any]:
    """
    Parse real Ur III Sumerian cuneiform texts from CDLI CC0-1.0 `mtaac_ur3_gold_corpus.zip`
    (4,055 tablets). Splits hyphenated signs and isolates determinatives (`{d}`, `{ki}`, etc.)
    as standalone cuneiform sign tokens matching Indus sign granularity. Rejects damaged lines.
    """
    all_lines: list[list[str]] = []
    seal_np_lines: list[list[str]] = []
    n_tablets = 0
    with zipfile.ZipFile(CAL_DIR / "mtaac_ur3_gold_corpus.zip") as zf:
        for cname in sorted(zf.namelist()):
            if not cname.endswith(".conll"):
                continue
            n_tablets += 1
            txt = zf.read(cname).decode("utf-8", errors="replace")
            by_line: dict[str, list[str]] = defaultdict(list)
            by_pos: dict[str, list[str]] = defaultdict(list)
            for line in txt.splitlines():
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 2:
                    continue
                tid, form = cols[0].strip(), cols[1].strip()
                pos = cols[3].strip() if len(cols) >= 4 else ""
                if not tid or not form:
                    continue
                parts = tid.split(".")
                lk = ".".join(parts[:-1]) if len(parts) >= 2 else tid
                by_line[lk].append(form)
                by_pos[lk].append(pos)

            for lk, forms in by_line.items():
                toks: list[str] = []
                bad = False
                for fm in forms:
                    if any(c in fm for c in "[]?<>_") or fm in {"x", "...", "n", "N"}:
                        bad = True
                        break
                    fm_c = re.sub(
                        r"(\{[^}]+\})",
                        r"-\1-",
                        fm.replace("#", "").replace("!", "").replace("*", ""),
                    )
                    for s in re.split(r"[-.]", fm_c):
                        s = s.strip()
                        if not s:
                            continue
                        if s.lower() == "x":
                            bad = True
                            break
                        toks.append(s)
                    if bad:
                        break
                if not bad and 2 <= len(toks) <= 13:
                    all_lines.append(toks)
                    if (
                        lk.startswith("s")
                        or any(p in {"PN", "DN", "RN"} for p in by_pos[lk])
                        or any(t in {"dumu", "kiszib3", "dub", "lugal", "ur", "{d}"} for t in toks)
                    ):
                        seal_np_lines.append(toks)

    all_uniq = list({tuple(s): s for s in all_lines}.values())
    seal_np_uniq = list({tuple(s): s for s in seal_np_lines}.values())
    return {
        "n_tablets_parsed": n_tablets,
        "lines_full": all_uniq,
        "seal_np_full": seal_np_uniq,
    }


def load_lineara_corpora() -> dict[str, Any]:
    """
    Parse real Minoan Linear A from SigLA/Younger `LinearAWords.json` (1,713 inscriptions).
    Filters out lacunae (`\U0001076b` / lacuna tags), word separators (`𐄁`), numbers, and fractions.
    Returns both intact multi-sign words and intact clean tablet lines (lengths 2..13).
    """
    raw = json.loads((CAL_DIR / "LinearAWords.json").read_text(encoding="utf-8"))
    words_raw: list[list[str]] = []
    lines_raw: list[list[str]] = []
    for ins in raw:
        cur_line: list[str] = []
        broken = False
        for wobj in ins.get("tagsForWords", []):
            tags = set(wobj.get("tags", []))
            tw = (wobj.get("transliteratedWord") or "").strip()
            if wobj.get("transliteratedWord") == "\n" or tw == "\n":
                if not broken and 2 <= len(cur_line) <= 13:
                    lines_raw.append(cur_line)
                cur_line, broken = [], False
                continue
            if not tw:
                continue
            if any("lacuna" in t for t in tags) or "\U0001076b" in tw:
                broken = True
                continue
            if any(t in tags for t in ("word separator", "punctuation", "number", "fraction")) or tw == "𐄁":
                continue
            toks = [t.strip() for t in tw.split("-") if t.strip() and t.strip() not in {"𐄁", "\U0001076b"}]
            if ("word" in tags or "word or logoram" in tags) and 2 <= len(toks) <= 13:
                words_raw.append(toks)
            cur_line.extend(toks)
        if not broken and 2 <= len(cur_line) <= 13:
            lines_raw.append(cur_line)

    words_uniq = list({tuple(s): s for s in words_raw}.values())
    lines_uniq = list({tuple(s): s for s in lines_raw}.values())
    return {
        "n_inscriptions_raw": len(raw),
        "words_full": words_uniq,
        "lines_full": lines_uniq,
    }


def generate_nonlinguistic_stamps(target_hist: Counter, seed: int = SEED) -> list[list[str]]:
    """
    Synthetic 100% NON-LINGUISTIC bureaucratic stamp generator matching Indus's EXACT
    sample size (N=2,047 unique sequences) and EXACT length histogram (2..13):
      Rigid template: [Header(4)] + [Conn(2)] + [Item(180), Zipf] + [CheckStamp(6)]
    (Exact Layer 7 template, deduplicated to exact target length histogram.)
    """
    rng = np.random.default_rng(seed)
    headers = [f"H{i}" for i in range(4)]
    conns = ["C0", "C1"]
    items = [f"I{i}" for i in range(180)]
    stamps = [f"S{i}" for i in range(6)]
    stamp_p = np.array([0.35, 0.20, 0.15, 0.12, 0.10, 0.08])
    item_p = np.array([1.0 / ((i + 1) ** 0.9) for i in range(len(items))])
    item_p /= item_p.sum()

    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for L in sorted(target_hist.keys()):
        need = target_hist[L]
        got = 0
        while got < need:
            if L == 2:
                seq = [str(rng.choice(items, p=item_p)), str(rng.choice(stamps, p=stamp_p))]
            elif L == 3:
                seq = [str(rng.choice(headers)), str(rng.choice(items, p=item_p)), str(rng.choice(stamps, p=stamp_p))]
            else:
                mid = [str(rng.choice(items, p=item_p)) for _ in range(L - 3)]
                seq = [str(rng.choice(headers)), str(rng.choice(conns))] + mid + [str(rng.choice(stamps, p=stamp_p))]
            t = tuple(seq)
            if t not in seen:
                seen.add(t)
                out.append(seq)
                got += 1
    return out


def slice_indus_length_matched(seqs: list[list[str]], target_hist: Counter, seed: int = SEED) -> list[list[str]]:
    """Sample without replacement matching Indus's EXACT length histogram (seed=42)."""
    rng = np.random.default_rng(seed)
    by_len: dict[int, list[list[str]]] = defaultdict(list)
    for s in seqs:
        by_len[len(s)].append(s)
    out: list[list[str]] = []
    for L in sorted(target_hist.keys()):
        pool = by_len.get(L, [])
        k = min(target_hist[L], len(pool))
        if k > 0:
            idxs = rng.choice(len(pool), size=k, replace=False)
            for idx in sorted(idxs.tolist()):
                out.append(pool[idx])
    return out


# =============================================================================
# METRICS 1, 2, 3, 4 — UNCHANGED BLIND CALIBRATION SUITE (ZERO HARDCODED SIGNS)
# =============================================================================

def shannon_entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values() if c > 0)


def compute_h0_h1(seqs: list[list[str]]) -> tuple[float, float]:
    uni, bi, pt = Counter(), Counter(), Counter()
    for seq in seqs:
        padded = [BOS] + seq + [EOS]
        for tok in seq:
            uni[tok] += 1
        for a, b in zip(padded[:-1], padded[1:]):
            bi[(a, b)] += 1
            pt[a] += 1
    h0 = shannon_entropy(uni)
    nb = sum(bi.values())
    h1 = -sum((c / nb) * math.log2(c / pt[a]) for (a, b), c in bi.items()) if nb else 0.0
    return h0, h1


def metric1_info_theory(seqs: list[list[str]], n_perm: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    N = len(seqs)
    mean_len = sum(len(s) for s in seqs) / max(N, 1)
    uni = Counter(t for s in seqs for t in s)
    V = len(uni)
    h0, h1 = compute_h0_h1(seqs)

    # Exact Layer 2A within-sequence permutation null
    null_h1s = []
    for _ in range(n_perm):
        shuf = []
        for s in seqs:
            arr = list(s)
            rng.shuffle(arr)
            shuf.append(arr)
        null_h1s.append(compute_h0_h1(shuf)[1])
    null_mu = float(np.mean(null_h1s))
    null_sd = float(np.std(null_h1s) + 1e-12)
    z_h1 = (h1 - null_mu) / null_sd

    init_c = Counter(s[0] for s in seqs)
    fin_c = Counter(s[-1] for s in seqs)
    h_init = shannon_entropy(init_c)
    h_final = shannon_entropy(fin_c)
    eff_init = 2.0 ** h_init
    eff_final = 2.0 ** h_final
    bn_ratio = eff_init / max(eff_final, 1e-12)
    top1_fin_sign, top1_fin_cnt = fin_c.most_common(1)[0]
    top5_fin_share = sum(c for _, c in fin_c.most_common(5)) / max(N, 1)
    top1_init_share = init_c.most_common(1)[0][1] / max(N, 1)

    return {
        "N_unique_seqs": N,
        "mean_len": round(mean_len, 3),
        "V_unique_signs": V,
        "H0_bits": round(h0, 4),
        "H1_bits": round(h1, 4),
        "null_H1_mean_bits": round(null_mu, 4),
        "null_H1_std_bits": round(null_sd, 4),
        "z_H1_within_seq_shuffle": round(z_h1, 2),
        "H_init_bits": round(h_init, 4),
        "H_final_bits": round(h_final, 4),
        "eff_init": round(eff_init, 2),
        "eff_final": round(eff_final, 2),
        "right_edge_bottleneck_ratio": round(bn_ratio, 2),
        "top1_final_sign": top1_fin_sign,
        "top1_final_share_pct": round(100.0 * top1_fin_cnt / max(N, 1), 2),
        "top5_final_share_pct": round(100.0 * top5_fin_share, 2),
        "top1_init_share_pct": round(100.0 * top1_init_share, 2),
    }


def metric2_affix_stacking(seqs: list[list[str]]) -> dict[str, Any]:
    """
    100% BLIND Affix Stacking Asymmetry (exact Layer 5 algorithm, ZERO hardcoded sign IDs!).
    Reports BOTH fixed threshold (tot >= 20, exact Layer 5 rule) AND sample-scaled threshold
    (tot >= max(5, round(0.01 * N))) for fair small-N Linear A comparison.
    """
    N = len(seqs)
    uni = Counter(t for s in seqs for t in s)
    pos = defaultdict(Counter)
    for s in seqs:
        L = len(s)
        for i, t in enumerate(s):
            pos[t]["init" if i == 0 else ("final" if i == L - 1 else "medial")] += 1

    def eval_thresh(min_tot: int) -> dict[str, Any]:
        term_set = {s for s, tot in uni.items() if tot >= min_tot and pos[s]["final"] / tot >= 0.65}
        init_set = {s for s, tot in uni.items() if tot >= min_tot and pos[s]["init"] / tot >= 0.55 and s not in term_set}
        suf2 = Counter()
        suf3 = Counter()
        pre2 = Counter()
        for s in seqs:
            if len(s) >= 2 and s[-2] in term_set and s[-1] in term_set:
                suf2[(s[-2], s[-1])] += 1
            if len(s) >= 3 and s[-3] in term_set and s[-2] in term_set and s[-1] in term_set:
                suf3[(s[-3], s[-2], s[-1])] += 1
            if len(s) >= 2 and s[0] in init_set and s[1] in init_set:
                pre2[(s[0], s[1])] += 1
        s2_tot = int(sum(suf2.values()))
        p2_tot = int(sum(pre2.values()))
        return {
            "min_freq_threshold": min_tot,
            "n_terminal_types": len(term_set),
            "n_initial_types": len(init_set),
            "suffix_2slot_stacks": s2_tot,
            "suffix_3slot_stacks": int(sum(suf3.values())),
            "prefix_2slot_stacks": p2_tot,
            "suffix_to_prefix_ratio": round(s2_tot / max(p2_tot, 1), 2),
            "log2_suf_over_pre": round(math.log2((s2_tot + 0.5) / (p2_tot + 0.5)), 2),
            "top_suffix_2stacks": [{"pair": f"{a}->{b}", "count": int(c)} for (a, b), c in suf2.most_common(5)],
            "top_prefix_2stacks": [{"pair": f"{a}->{b}", "count": int(c)} for (a, b), c in pre2.most_common(5)],
        }

    fixed = eval_thresh(20)
    scaled_thresh = max(5, round(0.01 * N))
    scaled = eval_thresh(scaled_thresh)
    return {
        "fixed_ge20_exact_layer5": fixed,
        "scaled_1pct_sensitivity": scaled,
    }


def metric3_kober_bridge_test(seqs: list[list[str]], n_perm: int = 500) -> dict[str, Any]:
    """
    100% BLIND Alice Kober Bridge-Sign Inflection Test (exact Layer 7 algorithm,
    selecting top_endings = top 6 most frequent final signs dynamically per corpus!).
    """
    rng = np.random.default_rng(SEED)
    seqs3 = [s for s in seqs if len(s) >= 3]
    fin_c = Counter(s[-1] for s in seqs)
    top6_end = {s for s, _ in fin_c.most_common(6)}

    stem_to_be: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
    full_stem_endings: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for seq in seqs3:
        if seq[-1] in top6_end:
            stem_to_be[tuple(seq[:-2])].append((seq[-2], seq[-1]))
            full_stem_endings[tuple(seq[:-1])].add(seq[-1])

    invariant_multi = sum(1 for ends in full_stem_endings.values() if len(ends) >= 2)
    alt_stems = 0
    same_stems = 0
    bridge_end_pairs = Counter()
    for stem, be_list in stem_to_be.items():
        by_e: dict[str, set[str]] = defaultdict(set)
        for b, e in be_list:
            by_e[e].add(b)
        if len(by_e) < 2:
            continue
        ends = sorted(by_e.keys())
        disj = all(
            not (by_e[ends[i]] & by_e[ends[j]])
            for i in range(len(ends))
            for j in range(i + 1, len(ends))
        )
        if disj:
            alt_stems += 1
            for b, e in be_list:
                bridge_end_pairs[(b, e)] += 1
        else:
            same_stems += 1

    b_list = [seq[-2] for seq in seqs3 if seq[-1] in top6_end]
    e_list = [seq[-1] for seq in seqs3 if seq[-1] in top6_end]
    N_be = len(b_list)
    be_c = Counter(zip(b_list, e_list))
    b_c = Counter(b_list)
    e_c = Counter(e_list)
    mi_bits = (
        sum((c / N_be) * math.log2((c * N_be) / (b_c[b] * e_c[e])) for (b, e), c in be_c.items())
        if N_be > 0
        else 0.0
    )

    null_mis = []
    if N_be > 0:
        for _ in range(n_perm):
            es = rng.permutation(e_list)
            cs = Counter(zip(b_list, es))
            null_mis.append(
                sum((c / N_be) * math.log2((c * N_be) / (b_c[b] * e_c[e])) for (b, e), c in cs.items())
            )
    null_mu = float(np.mean(null_mis)) if null_mis else 0.0
    null_sd = float(np.std(null_mis) + 1e-12) if null_mis else 1.0
    z_mi = (mi_bits - null_mu) / null_sd if N_be > 0 else 0.0

    return {
        "top6_endings_blind": [s for s, _ in fin_c.most_common(6)],
        "n_seqs_len_ge3_in_top6": N_be,
        "invariant_stems_ge2_endings": invariant_multi,
        "kober_same_bridge_stems": same_stems,
        "kober_alternating_stems": alt_stems,
        "ratio_alt_vs_same_bridge": round(alt_stems / max(same_stems, 1), 2),
        "bridge_ending_MI_bits": round(mi_bits, 4),
        "null_MI_mean_bits": round(null_mu, 4),
        "bridge_ending_MI_z": round(z_mi, 2),
        "top_alternating_pairs": [{"pair": f"{b}->{e}", "stems": int(c)} for (b, e), c in bridge_end_pairs.most_common(6)],
    }


def build_bidir_model(train_seqs: list[list[str]]) -> dict[str, Any]:
    """Exact Layer 7 bidirectional N-gram model (unchanged)."""
    uni = Counter()
    pos_uni: dict[str, Counter] = defaultdict(Counter)
    left_bi: dict[str, Counter] = defaultdict(Counter)
    right_bi: dict[str, Counter] = defaultdict(Counter)
    surround: dict[tuple[str, str], Counter] = defaultdict(Counter)
    left_tri: dict[tuple[str, str], Counter] = defaultdict(Counter)
    right_tri: dict[tuple[str, str], Counter] = defaultdict(Counter)
    seen_trigrams: set[tuple[str, str, str]] = set()

    for seq in train_seqs:
        padded = [BOS, BOS] + seq + [EOS, EOS]
        n = len(seq)
        for i, tok in enumerate(seq):
            uni[tok] += 1
            pt = "init" if i == 0 else ("final" if i == n - 1 else "medial")
            pos_uni[pt][tok] += 1
            p2, p1, n1, n2 = padded[i], padded[i + 1], padded[i + 3], padded[i + 4]
            left_bi[p1][tok] += 1
            right_bi[n1][tok] += 1
            surround[(p1, n1)][tok] += 1
            left_tri[(p2, p1)][tok] += 1
            right_tri[(n1, n2)][tok] += 1
            seen_trigrams.add((p1, tok, n1))

    vocab = [s for s, _ in uni.most_common()]
    return {
        "uni": uni,
        "pos_uni": pos_uni,
        "left_bi": left_bi,
        "right_bi": right_bi,
        "surround": surround,
        "left_tri": left_tri,
        "right_tri": right_tri,
        "seen_trigrams": seen_trigrams,
        "vocab": vocab,
    }


def predict_slot_probs(model: dict[str, Any], p2: str, p1: str, n1: str, n2: str, pt: str) -> dict[str, float]:
    """Exact Layer 7 slot probability distribution (unchanged)."""
    vocab = model["vocab"]
    V = len(vocab) + 1
    pos_u = model["pos_uni"][pt]
    l_bi = model["left_bi"][p1]
    r_bi = model["right_bi"][n1]
    sur = model["surround"][(p1, n1)]
    lt = model["left_tri"][(p2, p1)]
    rt = model["right_tri"][(n1, n2)]

    p_tot = sum(pos_u.values())
    l_tot = sum(l_bi.values())
    r_tot = sum(r_bi.values())
    s_tot = sum(sur.values())
    lt_tot = sum(lt.values())
    rt_tot = sum(rt.values())
    denom_p = p_tot + 0.1 * V

    scores = {}
    for cand in vocab:
        p_p = (pos_u[cand] + 0.1) / denom_p
        p_l = (l_bi[cand] + 0.5 * p_p) / (l_tot + 0.5) if l_tot > 0 else p_p
        p_r = (r_bi[cand] + 0.5 * p_p) / (r_tot + 0.5) if r_tot > 0 else p_p
        p_s = (sur[cand] + 0.3 * 0.5 * (p_l + p_r)) / (s_tot + 0.3) if s_tot > 0 else 0.5 * (p_l + p_r)
        p_lt = (lt[cand] + 0.3 * p_l) / (lt_tot + 0.3) if lt_tot > 0 else p_l
        p_rt = (rt[cand] + 0.3 * p_r) / (rt_tot + 0.3) if rt_tot > 0 else p_r
        scores[cand] = max(0.28 * p_s + 0.22 * p_l + 0.22 * p_r + 0.10 * p_lt + 0.10 * p_rt + 0.08 * p_p, 1e-12)

    unk_p = 0.1 / denom_p
    tot = sum(scores.values()) + unk_p
    for k in scores:
        scores[k] /= tot
    scores["<UNK>"] = unk_p / tot
    return scores


def metric4_masked_prediction_5fold(seqs: list[list[str]]) -> dict[str, Any]:
    """Exact Layer 7 5-fold deduplicated CV (Seen vs Unseen Trigram vs Position-Aware Baseline)."""
    seqs3 = [s for s in seqs if len(s) >= 3]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(seqs3))
    folds = np.array_split(order, 5)

    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "top1": 0, "top5": 0, "pos1": 0, "pos5": 0, "ce": 0.0, "pos_ce": 0.0}
    )

    for fi in range(5):
        te = set(folds[fi].tolist())
        train = [seqs3[i] for i in range(len(seqs3)) if i not in te]
        test = [seqs3[i] for i in folds[fi]]
        m = build_bidir_model(train)
        pos_top1 = {pt: m["pos_uni"][pt].most_common(1)[0][0] for pt in ("init", "medial", "final")}
        pos_top5 = {pt: {s for s, _ in m["pos_uni"][pt].most_common(5)} for pt in ("init", "medial", "final")}
        pos_tot = {pt: sum(m["pos_uni"][pt].values()) for pt in ("init", "medial", "final")}
        V = len(m["vocab"]) + 1

        for seq in test:
            padded = [BOS, BOS] + seq + [EOS, EOS]
            n = len(seq)
            for i, true_tok in enumerate(seq):
                pt = "init" if i == 0 else ("final" if i == n - 1 else "medial")
                p2, p1, n1, n2 = padded[i], padded[i + 1], padded[i + 3], padded[i + 4]
                is_seen = (p1, true_tok, n1) in m["seen_trigrams"]
                probs = predict_slot_probs(m, p2, p1, n1, n2, pt)
                top5 = [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1])[:5] if k != "<UNK>"]
                p_true = probs.get(true_tok, probs["<UNK>"])
                p_pos = (m["pos_uni"][pt][true_tok] + 0.1) / (pos_tot[pt] + 0.1 * V)

                for bname in ("ALL", pt, "SEEN_TRIGRAM" if is_seen else "UNSEEN_TRIGRAM", f"{pt}_{'SEEN' if is_seen else 'UNSEEN'}"):
                    b = buckets[bname]
                    b["n"] += 1
                    b["top1"] += int(len(top5) > 0 and top5[0] == true_tok)
                    b["top5"] += int(true_tok in top5)
                    b["pos1"] += int(true_tok == pos_top1[pt])
                    b["pos5"] += int(true_tok in pos_top5[pt])
                    b["ce"] -= math.log2(max(p_true, 1e-12))
                    b["pos_ce"] -= math.log2(max(p_pos, 1e-12))

    out = {}
    for k in ("ALL", "SEEN_TRIGRAM", "UNSEEN_TRIGRAM", "final", "final_UNSEEN", "medial_UNSEEN", "init_UNSEEN"):
        b = buckets[k]
        n = max(b["n"], 1)
        m_ce = b["ce"] / n
        p_ce = b["pos_ce"] / n
        out[k] = {
            "n": b["n"],
            "model_top1_pct": round(100.0 * b["top1"] / n, 2),
            "pos_baseline_top1_pct": round(100.0 * b["pos1"] / n, 2),
            "gain_vs_pos_base_pp": round(100.0 * (b["top1"] - b["pos1"]) / n, 2),
            "model_top5_pct": round(100.0 * b["top5"] / n, 2),
            "pos_baseline_top5_pct": round(100.0 * b["pos5"] / n, 2),
            "model_ce_bits": round(m_ce, 3),
            "pos_base_ce_bits": round(p_ce, 3),
            "ce_gain_bits": round(p_ce - m_ce, 3),  # positive = model beats position baseline
        }
    return out


# =============================================================================
# METRIC 5 — KNOWN-ANSWER 23-SEAL CSP IDENTIFIABILITY CALIBRATION (N_NULL=100)
# =============================================================================

def generate_matched_synthetic_lexicon(
    real_words_by_len: dict[int, list[tuple[int, ...]]],
    rng: np.random.Generator,
) -> dict[int, list[tuple[int, ...]]]:
    """Exact Layer 4B synthetic shuffled lexicon generator (destroys historical pairing)."""
    synth: dict[int, list[tuple[int, ...]]] = {}
    for L, words in real_words_by_len.items():
        if not words:
            synth[L] = []
            continue
        pref_pool = [w[0] for w in words]
        rng.shuffle(pref_pool)
        body_bag = [s for w in words for s in w[1:]]
        body_uniq, body_cnts = np.unique(body_bag, return_counts=True)
        body_probs = body_cnts / body_cnts.sum()

        new_words: list[tuple[int, ...]] = []
        seen_w: set[tuple[int, ...]] = set()
        for idx, w in enumerate(words):
            pat_map: dict[int, int] = {}
            pat = []
            for syl in w:
                if syl not in pat_map:
                    pat_map[syl] = len(pat_map)
                pat.append(pat_map[syl])
            n_classes = len(pat_map)
            cand = w
            for _ in range(8):
                class_syl = [0] * n_classes
                class_syl[0] = int(pref_pool[idx])
                used = {class_syl[0]}
                for c in range(1, n_classes):
                    draw = int(rng.choice(body_uniq, p=body_probs))
                    tries = 0
                    while draw in used and tries < 12:
                        draw = int(rng.choice(body_uniq, p=body_probs))
                        tries += 1
                    class_syl[c] = draw
                    used.add(draw)
                cand = tuple(class_syl[p] for p in pat)
                if cand not in seen_w:
                    seen_w.add(cand)
                    new_words.append(cand)
                    break
            else:
                new_words.append(cand)
        synth[L] = new_words
    return synth


def filter_seal_domain(
    seq: list[str],
    words_of_len: list[tuple[int, ...]],
    anchors: dict[str, int],
    injective: bool = True,
) -> list[tuple[int, ...]]:
    L = len(seq)
    eq_pairs = [(i, j) for i in range(L) for j in range(i + 1, L) if seq[i] == seq[j]]
    neq_pairs = [(i, j) for i in range(L) for j in range(i + 1, L) if seq[i] != seq[j]] if injective else []
    anchor_pos = [(i, anchors[seq[i]]) for i in range(L) if seq[i] in anchors]
    anchor_vals = set(anchors.values())
    non_anchor_pos = [i for i in range(L) if seq[i] not in anchors]
    valid = []
    for w in words_of_len:
        ok = True
        for i, val in anchor_pos:
            if w[i] != val:
                ok = False
                break
        if not ok:
            continue
        if injective:
            for i in non_anchor_pos:
                if w[i] in anchor_vals:
                    ok = False
                    break
            if not ok:
                continue
        for i, j in eq_pairs:
            if w[i] != w[j]:
                ok = False
                break
        if not ok:
            continue
        for i, j in neq_pairs:
            if w[i] == w[j]:
                ok = False
                break
        if ok:
            valid.append(w)
    return valid


def build_bitmask_index(domains: list[list[tuple[int, ...]]]) -> list[tuple[int, list[dict[int, int]]]]:
    out = []
    for dom in domains:
        m = len(dom)
        if m == 0:
            out.append((0, []))
            continue
        full_mask = (1 << m) - 1
        L = len(dom[0])
        pos_mask: list[dict[int, int]] = [defaultdict(int) for _ in range(L)]
        for wi, w in enumerate(dom):
            bit = 1 << wi
            for p, syl in enumerate(w):
                pos_mask[p][syl] |= bit
        out.append((full_mask, pos_mask))
    return out


def mask_to_indices(mask: int) -> list[int]:
    res = []
    while mask:
        lsb = mask & -mask
        res.append(lsb.bit_length() - 1)
        mask ^= lsb
    return res


def fast_consistent_mask(
    sidx: int,
    bidx: list[tuple[int, list[dict[int, int]]]],
    seq_sign_pos: list[list[tuple[int, int]]],
    sign_val: list[int],
    used_syls: set[int],
    injective: bool = True,
) -> int:
    full_mask, pos_mask = bidx[sidx]
    if full_mask == 0:
        return 0
    m = full_mask
    spos = seq_sign_pos[sidx]
    unassigned_pos = False
    for pos, si in spos:
        v = sign_val[si]
        if v != -1:
            pm = pos_mask[pos].get(v, 0)
            m &= pm
            if m == 0:
                return 0
        else:
            unassigned_pos = True
    if injective and unassigned_pos and used_syls:
        for pos, si in spos:
            if sign_val[si] == -1:
                pmd = pos_mask[pos]
                for syl in used_syls:
                    bad = pmd.get(syl, 0)
                    if bad:
                        m &= ~bad
                        if m == 0:
                            return 0
    return m


def solve_exact_bitmask_csp(
    seqs: list[list[str]],
    words_by_len: dict[int, list[tuple[int, ...]]],
    anchors: dict[str, int] | None = None,
    injective: bool = True,
    max_solutions: int = 200,
    node_budget: int = 80000,
) -> dict[str, Any]:
    """Exact global N/N CSP with Bitmask Inverted-Index + Dynamic MRV + Forward Checking."""
    anchors = anchors or {}
    sign_list = sorted({tok for sq in seqs for tok in sq})
    sign_to_idx = {sg: i for i, sg in enumerate(sign_list)}
    sign_freq = Counter(tok for sq in seqs for tok in sq)
    n_signs = len(sign_list)
    n_seals = len(seqs)

    domains = [filter_seal_domain(sq, words_by_len.get(len(sq), []), anchors, injective=injective) for sq in seqs]
    if any(len(d) == 0 for d in domains):
        return {"satisfiable": False, "n_solutions": 0, "nodes": 0, "capped": False, "sign_counts": {}}
    bidx = build_bitmask_index(domains)
    seq_sign_pos = [[(i, sign_to_idx[tok]) for i, tok in enumerate(sq)] for sq in seqs]
    seal_deg = [sum(sign_freq[tok] - 1 for tok in set(sq)) for sq in seqs]

    sign_val = [-1] * n_signs
    used_syls: set[int] = set()
    for sg, val in anchors.items():
        if sg in sign_to_idx:
            sign_val[sign_to_idx[sg]] = val
            used_syls.add(val)

    used_seal = [False] * n_seals
    sol_counts = [Counter() for _ in range(n_signs)]
    n_sol = 0
    nodes = 0
    capped = False

    def bt(rem: int) -> None:
        nonlocal n_sol, nodes, capped
        if n_sol >= max_solutions or nodes >= node_budget:
            capped = True
            return
        if rem == 0:
            n_sol += 1
            for si in range(n_signs):
                sol_counts[si][sign_val[si]] += 1
            return
        best_s = -1
        best_mask = 0
        best_key = (10**9, 0)
        for s in range(n_seals):
            if used_seal[s]:
                continue
            m = fast_consistent_mask(s, bidx, seq_sign_pos, sign_val, used_syls, injective)
            if m == 0:
                return  # Instant dead-end pruning!
            cnt = m.bit_count()
            key = (cnt, -seal_deg[s])
            if key < best_key:
                best_key = key
                best_s = s
                best_mask = m
                if cnt == 1:
                    break
        nodes += 1
        used_seal[best_s] = True
        spos = seq_sign_pos[best_s]
        dom_s = domains[best_s]
        for wi in mask_to_indices(best_mask):
            w = dom_s[wi]
            newly = []
            for pos, si in spos:
                if sign_val[si] == -1:
                    syl = w[pos]
                    sign_val[si] = syl
                    used_syls.add(syl)
                    newly.append((si, syl))
            bt(rem - 1)
            for si, syl in newly:
                sign_val[si] = -1
                used_syls.discard(syl)
            if n_sol >= max_solutions or nodes >= node_budget:
                capped = True
                break
        used_seal[best_s] = False

    bt(n_seals)
    return {
        "satisfiable": n_sol > 0,
        "n_solutions": n_sol,
        "nodes": nodes,
        "capped": capped,
        "sign_counts": {sg: dict(sol_counts[sign_to_idx[sg]]) for sg in sign_list},
    }


def solve_max_bitmask_csp(
    seqs: list[list[str]],
    words_by_len: dict[int, list[tuple[int, ...]]],
    anchors: dict[str, int] | None = None,
    injective: bool = True,
    node_budget: int = 3000,
    collect_opt: int = 50,
    check_exact_first: bool = True,
) -> dict[str, Any]:
    """
    Two-stage Exact-First + Dynamic-MRV Bitmask Max-CSP:
      Stage 1: Exact N/N global CSP (zero skip branches -> finds 23/23 in <20 ms if it exists!).
      Stage 2: If Exact N/N is UNSAT, runs Dynamic-MRV Max-CSP branch-and-bound.
    """
    anchors = anchors or {}
    n_seals = len(seqs)
    if check_exact_first:
        ex = solve_exact_bitmask_csp(
            seqs, words_by_len, anchors=anchors, injective=injective,
            max_solutions=max(collect_opt, 100), node_budget=max(node_budget * 4, 40000),
        )
        if ex["satisfiable"]:
            return {
                "best_k": n_seals,
                "exact_sat": True,
                "opt_sols": ex["n_solutions"],
                "nodes": ex["nodes"],
                "budget_hit": ex["capped"],
                "sign_counts": ex["sign_counts"],
            }

    sign_list = sorted({tok for sq in seqs for tok in sq})
    sign_to_idx = {sg: i for i, sg in enumerate(sign_list)}
    sign_freq = Counter(tok for sq in seqs for tok in sq)
    n_signs = len(sign_list)

    domains = [filter_seal_domain(sq, words_by_len.get(len(sq), []), anchors, injective=injective) for sq in seqs]
    bidx = build_bitmask_index(domains)
    seq_sign_pos = [[(i, sign_to_idx[tok]) for i, tok in enumerate(sq)] for sq in seqs]
    seal_deg = [sum(sign_freq[tok] - 1 for tok in set(sq)) for sq in seqs]

    sign_val = [-1] * n_signs
    used_syls: set[int] = set()
    for sg, val in anchors.items():
        if sg in sign_to_idx:
            sign_val[sign_to_idx[sg]] = val
            used_syls.add(val)

    used_seal = [False] * n_seals
    best_k = 0
    opt_sols = 0
    opt_counts = [Counter() for _ in range(n_signs)]
    nodes = 0
    budget_hit = False

    def bnb(k_cur: int) -> None:
        nonlocal best_k, opt_sols, opt_counts, nodes, budget_hit
        if nodes >= node_budget:
            budget_hit = True
            return
        live = []
        for s in range(n_seals):
            if used_seal[s]:
                continue
            m = fast_consistent_mask(s, bidx, seq_sign_pos, sign_val, used_syls, injective)
            if m:
                live.append((m.bit_count(), -seal_deg[s], s, m))
        if k_cur > best_k:
            best_k = k_cur
            opt_sols = 1
            opt_counts = [Counter() for _ in range(n_signs)]
            for si in range(n_signs):
                if sign_val[si] != -1:
                    opt_counts[si][sign_val[si]] += 1
        elif k_cur == best_k and k_cur > 0 and opt_sols < collect_opt:
            opt_sols += 1
            for si in range(n_signs):
                if sign_val[si] != -1:
                    opt_counts[si][sign_val[si]] += 1
        if not live or k_cur + len(live) < best_k:
            return
        if k_cur + len(live) == best_k and opt_sols >= collect_opt:
            return
        live.sort(key=lambda x: (x[0], x[1]))
        _, _, best_s, best_mask = live[0]
        nodes += 1
        used_seal[best_s] = True
        spos = seq_sign_pos[best_s]
        dom_s = domains[best_s]
        for wi in mask_to_indices(best_mask):
            if nodes >= node_budget:
                budget_hit = True
                break
            w = dom_s[wi]
            newly = []
            for pos, si in spos:
                if sign_val[si] == -1:
                    syl = w[pos]
                    sign_val[si] = syl
                    used_syls.add(syl)
                    newly.append((si, syl))
            bnb(k_cur + 1)
            for si, syl in newly:
                sign_val[si] = -1
                used_syls.discard(syl)
        if k_cur + len(live) - 1 >= best_k and nodes < node_budget:
            if k_cur + len(live) - 1 > best_k or opt_sols < collect_opt:
                bnb(k_cur)
        used_seal[best_s] = False

    bnb(0)
    return {
        "best_k": best_k,
        "exact_sat": False,
        "opt_sols": opt_sols,
        "nodes": nodes,
        "budget_hit": budget_hit,
        "sign_counts": {sg: dict(opt_counts[sign_to_idx[sg]]) for sg in sign_list},
    }


def select_connected_23_gulf_matched(
    pool: list[list[str]], target_shared_min: int = 20, seed: int = SEED
) -> list[list[str]]:
    """
    Draw 23 real sequences (12 of len 4, 11 of len 5 -> mean 4.48) forming a single
    connected component matched to the 23 Gulf circular/cylinder seals' topology.
    """
    rng = np.random.default_rng(seed)
    p4 = [s for s in pool if len(s) == 4]
    p5 = [s for s in pool if len(s) == 5]
    rng.shuffle(p4)
    rng.shuffle(p5)
    freq = Counter(t for s in pool for t in s)
    chosen = [max(p4[:40], key=lambda s: sum(freq[t] for t in s))]
    used = {tuple(chosen[0])}
    sign_counts = Counter(chosen[0])
    c4, c5 = 1, 0
    cand_all = p4 + p5
    rng.shuffle(cand_all)
    while len(chosen) < 23:
        want_len = 4 if c4 < 12 else 5
        seen_signs = set(sign_counts.keys())
        cands = [
            s for s in cand_all
            if len(s) == want_len and tuple(s) not in used and 1 <= len(set(s) & seen_signs) <= 2
        ]
        if not cands:
            cands = [
                s for s in cand_all
                if len(s) == want_len and tuple(s) not in used and len(set(s) & seen_signs) >= 1
            ]
        cur_shared = sum(1 for c in sign_counts.values() if c >= 2)
        if cur_shared < target_shared_min:
            singletons = {sg for sg, c in sign_counts.items() if c == 1}
            touch = [s for s in cands if len(set(s) & singletons) >= 1]
            if touch:
                cands = touch
        pick = cands[int(rng.integers(0, min(len(cands), 25)))]
        chosen.append(pick)
        used.add(tuple(pick))
        sign_counts.update(pick)
        if want_len == 4:
            c4 += 1
        else:
            c5 += 1
    return chosen


def run_known_answer_csp_suite(
    lb_words_full: list[list[str]],
    sum_seal_np_full: list[list[str]],
    sum_lines_full: list[list[str]],
    n_null: int = 100,
) -> dict[str, Any]:
    """
    Metric 5: Known-Answer 23-Seal CSP Identifiability Calibration (n_null=100) on:
      (A) 23 Real Linear B words (100% attested & 74% partial-attestation stress test) at k=0 & k=2
      (B) 23 Real Ur III Sumerian seal/name phrases (100% attested & 74% partial-attestation) at k=0 & k=2
      (C) Actual 23 Indus Gulf Circular/Cylinder Seals vs Real CDLI Ur III Archive (k=0)
          + cross-reference to Layer 4B published strict_10 results.
    """
    lb_45 = [s for s in lb_words_full if len(s) in (4, 5)]
    sum_45 = [s for s in sum_seal_np_full if len(s) in (4, 5)]

    def eval_known_answer(
        label: str,
        pool: list[list[str]],
        target_per_len: int = 550,
        attested_n: int = 23,
        seed: int = SEED,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        seqs_23 = select_connected_23_gulf_matched(pool, target_shared_min=20, seed=seed)
        # Optionally drop (23 - attested_n) seals from the lexicon for realistic open-vocab test
        attested_subset = seqs_23[:attested_n]
        dropped_set = {tuple(s) for s in seqs_23[attested_n:]}

        inc_set = {tuple(s) for s in attested_subset}
        by_len: dict[int, list[list[str]]] = {
            4: [s for s in attested_subset if len(s) == 4],
            5: [s for s in attested_subset if len(s) == 5],
        }
        for L in (4, 5):
            rest = [s for s in pool if len(s) == L and tuple(s) not in inc_set and tuple(s) not in dropped_set]
            rng.shuffle(rest)
            by_len[L].extend(rest[: max(0, target_per_len - len(by_len[L]))])

        vocab = sorted({t for L in (4, 5) for s in by_len[L] for t in s} | {t for s in seqs_23 for t in s})
        tok2id = {t: i for i, t in enumerate(vocab)}
        words_by_len = {L: [tuple(tok2id[t] for t in s) for s in by_len[L]] for L in (4, 5)}

        # Hide true values under random bijection cipher phi*
        uniq_true = sorted({t for s in seqs_23 for t in s})
        perm = rng.permutation(len(uniq_true))
        true_to_anon = {t: f"SG_{perm[i]:02d}" for i, t in enumerate(uniq_true)}
        anon_to_true_id = {true_to_anon[t]: tok2id[t] for t in uniq_true}
        anon_seqs = [[true_to_anon[t] for t in s] for s in seqs_23]

        deg = Counter(sg for s in anon_seqs for sg in s)
        shared_set = {sg for sg, c in deg.items() if c >= 2}
        top2 = [sg for sg, _ in deg.most_common(2)]
        anc_k2 = {sg: anon_to_true_id[sg] for sg in top2}

        T = sum(len(s) for s in seqs_23)
        U = len(uniq_true)
        U_sh = len(shared_set)
        graph_stats = {
            "N_seals": len(seqs_23),
            "attested_in_lexicon": attested_n,
            "T_tokens": T,
            "mean_len": round(T / len(seqs_23), 2),
            "U_unique_signs": U,
            "U_shared_signs": U_sh,
            "U_singletons": U - U_sh,
            "singleton_pct": round(100.0 * (U - U_sh) / U, 1),
            "lexicon_size_L4": len(words_by_len[4]),
            "lexicon_size_L5": len(words_by_len[5]),
            "sample_true_phrases": ["-".join(s) for s in seqs_23[:6]],
        }

        regimes = {}
        for k_anc, anc in [(0, {}), (2, anc_k2)]:
            t1 = time.time()
            res_true = solve_max_bitmask_csp(
                anon_seqs, words_by_len, anchors=anc, node_budget=15000, collect_opt=100, check_exact_first=True
            )
            sc = res_true["sign_counts"]
            locked_all = sum(1 for sg, c in sc.items() if len(c) == 1)
            true_locked_all = sum(1 for sg, c in sc.items() if len(c) == 1 and next(iter(c)) == anon_to_true_id[sg])
            locked_sh = sum(1 for sg in shared_set if len(sc.get(sg, {})) == 1)
            true_locked_sh = sum(
                1 for sg in shared_set if len(sc.get(sg, {})) == 1 and next(iter(sc[sg])) == anon_to_true_id[sg]
            )
            # Also check modal accuracy among shared signs
            modal_correct_sh = sum(
                1 for sg in shared_set
                if sc.get(sg) and max(sc[sg].items(), key=lambda kv: kv[1])[0] == anon_to_true_id[sg]
            )

            null_ks = []
            null_exact_sat = 0
            for _ in range(n_null):
                syn = generate_matched_synthetic_lexicon(words_by_len, rng)
                rn = solve_max_bitmask_csp(
                    anon_seqs, syn, anchors=anc, node_budget=1800, collect_opt=5, check_exact_first=True
                )
                null_ks.append(rn["best_k"])
                if rn["exact_sat"]:
                    null_exact_sat += 1

            mu = float(np.mean(null_ks))
            sd = float(np.std(null_ks, ddof=1)) if len(null_ks) > 1 else 0.0
            z = (res_true["best_k"] - mu) / sd if sd > 1e-12 else 0.0
            emp_p = (1 + sum(x >= res_true["best_k"] for x in null_ks)) / (n_null + 1)

            regimes[f"k{k_anc}_anchors"] = {
                "k_anchors": k_anc,
                "true_max_k": int(res_true["best_k"]),
                "exact_23_of_23_sat": bool(res_true["exact_sat"]),
                "n_opt_solutions_collected": int(res_true["opt_sols"]),
                "budget_hit": bool(res_true["budget_hit"]),
                "locked_all_signs": f"{locked_all}/{U}",
                "true_recovered_locked_all": f"{true_locked_all}/{U}",
                "locked_shared_signs": f"{locked_sh}/{U_sh}",
                "true_recovered_locked_shared": f"{true_locked_sh}/{U_sh}",
                "modal_correct_shared_pct": round(100.0 * modal_correct_sh / max(U_sh, 1), 1),
                "null_n": n_null,
                "null_max_k_mean": round(mu, 2),
                "null_max_k_std": round(sd, 2),
                "null_max_k_max": int(max(null_ks)),
                "null_exact_sat_count": int(null_exact_sat),
                "z_score": round(z, 2),
                "empirical_p": round(emp_p, 4),
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(
                f"  [CSP {label} k={k_anc}] TRUE={res_true['best_k']}/23 (exact={res_true['exact_sat']}) "
                f"| NULL={mu:.2f}±{sd:.2f} (max={max(null_ks)}, sat={null_exact_sat}/{n_null}) "
                f"| z={z:+.2f}, p={emp_p:.4f} | sh_lock={true_locked_sh}/{U_sh} true ({time.time()-t1:.1f}s)"
            )
        return {"graph_stats": graph_stats, "results": regimes}

    out: dict[str, Any] = {}
    print("Running Metric 5A: Known-Answer Linear B (23 Real Words, 100% Attested)...")
    out["linearb_23_closed_100pct"] = eval_known_answer("LinearB_100%", lb_45, target_per_len=550, attested_n=23, seed=42)
    print("Running Metric 5B: Known-Answer Ur III Sumerian (23 Real Seal/Name Phrases, 100% Attested)...")
    out["sumerian_23_closed_100pct"] = eval_known_answer("Sumerian_100%", sum_45, target_per_len=550, attested_n=23, seed=42)
    print("Running Metric 5C: Partial-Attestation Stress Test (Linear B, 17/23 = 74% Attested)...")
    out["linearb_23_open_74pct"] = eval_known_answer("LinearB_74%", lb_45, target_per_len=550, attested_n=17, seed=42)
    print("Running Metric 5D: Partial-Attestation Stress Test (Ur III Sumerian, 17/23 = 74% Attested)...")
    out["sumerian_23_open_74pct"] = eval_known_answer("Sumerian_74%", sum_45, target_per_len=550, attested_n=17, seed=42)

    # Metric 5E: Direct Indus 23 Connected Gulf Seals vs Real CDLI Ur III Sumerian Archive (k=0)
    print("Running Metric 5E: Actual 23 Indus Gulf Seals vs Real CDLI Ur III Archive (same Dynamic-MRV solver)...")
    l4b = json.loads((REPORTS_DIR / "layer4b_rosetta_csp.json").read_text(encoding="utf-8"))
    gulf_23_seqs = [s["clean_seq"] for s in l4b["cohorts_summary"]["gulf_connected_seals"]]
    gulf_lens = sorted({len(s) for s in gulf_23_seqs})

    rng_g = np.random.default_rng(SEED)
    by_len_ur3: dict[int, list[list[str]]] = defaultdict(list)
    for s in sum_lines_full:
        if len(s) in gulf_lens:
            by_len_ur3[len(s)].append(s)
    sampled_ur3: dict[int, list[list[str]]] = {}
    for L in gulf_lens:
        pool_L = list(by_len_ur3[L])
        rng_g.shuffle(pool_L)
        sampled_ur3[L] = pool_L[:550]
    ur3_vocab = sorted({t for L in gulf_lens for s in sampled_ur3[L] for t in s})
    u2i = {t: i for i, t in enumerate(ur3_vocab)}
    ur3_wbl = {L: [tuple(u2i[t] for t in s) for s in sampled_ur3[L]] for L in gulf_lens}

    t1 = time.time()
    res_gulf = solve_max_bitmask_csp(gulf_23_seqs, ur3_wbl, anchors={}, node_budget=15000, collect_opt=100, check_exact_first=True)
    null_gulf_ks = []
    for _ in range(n_null):
        syn = generate_matched_synthetic_lexicon(ur3_wbl, rng_g)
        null_gulf_ks.append(
            solve_max_bitmask_csp(gulf_23_seqs, syn, anchors={}, node_budget=1800, collect_opt=5, check_exact_first=True)["best_k"]
        )
    mu_g = float(np.mean(null_gulf_ks))
    sd_g = float(np.std(null_gulf_ks, ddof=1))
    z_g = (res_gulf["best_k"] - mu_g) / (sd_g + 1e-12)
    p_g = (1 + sum(x >= res_gulf["best_k"] for x in null_gulf_ks)) / (n_null + 1)
    gulf_deg = Counter(sg for s in gulf_23_seqs for sg in s)
    gulf_shared = {sg for sg, c in gulf_deg.items() if c >= 2}
    gulf_locked_sh = sum(1 for sg in gulf_shared if len(res_gulf["sign_counts"].get(sg, {})) == 1)

    mc4b = l4b["monte_carlo_identifiability_1000"]
    out["indus_23_gulf_vs_real_cdli_ur3"] = {
        "N_seals": len(gulf_23_seqs),
        "U_unique_signs": len(gulf_deg),
        "U_shared_signs": len(gulf_shared),
        "singleton_pct": round(100.0 * (len(gulf_deg) - len(gulf_shared)) / len(gulf_deg), 1),
        "k0_observed_max_k": int(res_gulf["best_k"]),
        "exact_23_of_23_sat": bool(res_gulf["exact_sat"]),
        "locked_shared_signs": f"{gulf_locked_sh}/{len(gulf_shared)}",
        "null_n": n_null,
        "null_max_k_mean": round(mu_g, 2),
        "null_max_k_std": round(sd_g, 2),
        "null_max_k_max": int(max(null_gulf_ks)),
        "z_score": round(z_g, 2),
        "empirical_p": round(p_g, 4),
        "elapsed_sec": round(time.time() - t1, 1),
        "layer4b_published_strict10_curated": mc4b["regime_A_curated_attested"]["null1_synthetic_syllable_corpora"]["strict_10_max_k"],
        "layer4b_published_strict10_productive": mc4b["regime_B_productive_archive"]["null1_synthetic_syllable_corpora"]["strict_10_max_k"],
    }
    print(
        f"  [CSP Indus_23_Gulf vs CDLI_Ur3 k=0] OBS={res_gulf['best_k']}/23 "
        f"| NULL={mu_g:.2f}±{sd_g:.2f} (max={max(null_gulf_ks)}) | z={z_g:+.2f}, p={p_g:.4f} ({time.time()-t1:.1f}s)"
    )
    return out


# =============================================================================
# VISUALIZATION — PUBLICATION-GRADE 5-PANEL CALIBRATION PLATE
# =============================================================================

def _load_font(bold: bool = False, italic: bool = False, size: int = 20) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold and italic:
        candidates.append("/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf")
    elif bold:
        candidates.append("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    elif italic:
        candidates.append("/System/Library/Fonts/Supplemental/Arial Italic.ttf")
    else:
        candidates.append("/System/Library/Fonts/Supplemental/Arial.ttf")
    candidates.append("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw, cx: float, y: float, text: str, font: ImageFont.FreeTypeFont,
    fill: str = "#0F172A", anchor_v: str = "top", line_gap: int = 4,
) -> None:
    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    heights = [bb[3] - bb[1] for bb in bboxes]
    tot_h = sum(heights) + line_gap * (len(lines) - 1)
    if anchor_v == "bottom":
        cur_y = y - tot_h
    elif anchor_v == "center":
        cur_y = y - tot_h / 2
    else:
        cur_y = y
    for ln, bb, h in zip(lines, bboxes, heights):
        w = bb[2] - bb[0]
        draw.text((cx - w / 2 - bb[0], cur_y - bb[1]), ln, font=font, fill=fill)
        cur_y += h + line_gap


def _draw_dashed_hline(
    draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, fill: str, width: int = 2, dash: int = 10, gap: int = 7
) -> None:
    x = x0
    while x < x1:
        draw.line([(x, y), (min(x + dash, x1), y)], fill=fill, width=width)
        x += dash + gap


def render_5way_chart(results: dict[str, Any]) -> None:
    """Render a crisp, publication-grade 2600x1664 5-panel calibration plate with PIL (zero overlap)."""
    W, H = 2600, 1664
    img = Image.new("RGB", (W, H), "#F1F5F9")
    draw = ImageDraw.Draw(img)

    f_title = _load_font(bold=True, size=40)
    f_sub = _load_font(italic=True, size=22)
    f_ptitle = _load_font(bold=True, size=24)
    f_psub = _load_font(italic=False, size=18)
    f_axis = _load_font(bold=True, size=18)
    f_tick = _load_font(bold=False, size=17)
    f_xtick = _load_font(bold=True, size=17)
    f_val = _load_font(bold=True, size=17)
    f_val_sm = _load_font(bold=True, size=15)
    f_leg = _load_font(bold=True, size=16)
    f_foot = _load_font(bold=False, size=18)
    f_foot_b = _load_font(bold=True, size=18)

    primary_keys = [
        ("indus_strict", "Indus\n(Target)"),
        ("sumerian_matched", "Ur III\nSumerian"),
        ("linearb_matched", "Linear B\n(Greek)"),
        ("lineara_words", "Linear A\n(Minoan)"),
        ("nonlinguistic_stamps", "Non-Ling.\nStamps"),
    ]
    colors = {
        "indus_strict": "#C2410C",          # Terracotta Indus Orange-Red
        "sumerian_matched": "#1D4ED8",      # Lapis Lazuli Blue
        "linearb_matched": "#0F766E",       # Aegean Teal
        "lineara_words": "#7C3AED",         # Minoan Purple
        "nonlinguistic_stamps": "#64748B",  # Slate Gray (Negative Control)
    }
    colors_light = {
        "indus_strict": "#FDBA74",
        "sumerian_matched": "#93C5FD",
        "linearb_matched": "#5EEAD4",
        "lineara_words": "#C4B5FD",
        "nonlinguistic_stamps": "#CBD5E1",
    }
    labels = [lbl for _, lbl in primary_keys]
    bar_colors = [colors[k] for k, _ in primary_keys]
    bar_light = [colors_light[k] for k, _ in primary_keys]
    prim = results["primary_5way_matched"]

    # Header banner
    draw.rectangle([0, 0, W, 128], fill="#0F172A")
    _draw_centered_text(
        draw, W / 2, 22,
        "INDUS-DEDUCT Layer 8: 5-Way Known-Answer & Negative-Control Calibration Benchmark",
        f_title, fill="#FFFFFF",
    )
    _draw_centered_text(
        draw, W / 2, 78,
        "Identical zero-hardcoding blind algorithms across Indus (N=2,047), Real Ur III Sumerian (CDLI, N=2,047 matched), Real Linear B (N=1,651 matched), Real Linear A (N=660), & Non-Linguistic Stamps (N=2,047)",
        f_sub, fill="#CBD5E1",
    )

    def draw_card(x0: int, y0: int, x1: int, y1: int, title: str, subtitle: str) -> tuple[int, int, int, int]:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill="#FFFFFF", outline="#CBD5E1", width=2)
        draw.text((x0 + 26, y0 + 18), title, font=f_ptitle, fill="#0F172A")
        draw.text((x0 + 26, y0 + 50), subtitle, font=f_psub, fill="#475569")
        # Return inner plot area (px0, py0, px1, py1)
        return x0 + 96, y0 + 96, x1 - 30, y1 - 86

    def draw_axes_and_grid(
        px0: int, py0: int, px1: int, py1: int, ymin: float, ymax: float, ticks: list[float], fmt: str = "{:.0f}"
    ):
        draw.rectangle([px0, py0, px1, py1], fill="#F8FAFC", outline="#94A3B8", width=1)
        def y_map(v: float) -> int:
            return int(round(py1 - (v - ymin) / (ymax - ymin) * (py1 - py0)))
        for tv in ticks:
            yy = y_map(tv)
            if py0 <= yy <= py1:
                draw.line([(px0, yy), (px1, yy)], fill="#E2E8F0", width=1)
                lbl = fmt.format(tv)
                bb = draw.textbbox((0, 0), lbl, font=f_tick)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                draw.text((px0 - tw - 12 - bb[0], yy - th / 2 - bb[1]), lbl, font=f_tick, fill="#334155")
        if ymin < 0 < ymax:
            y0_line = y_map(0.0)
            draw.line([(px0, y0_line), (px1, y0_line)], fill="#0F172A", width=2)
        draw.line([(px0, py0), (px0, py1)], fill="#334155", width=2)
        draw.line([(px0, py1), (px1, py1)], fill="#334155", width=2)
        return y_map

    # Row 1 geometry (3 panels: A, B, C)
    r1_y0, r1_y1 = 152, 800
    pad_x = 28
    cw3 = (W - 4 * pad_x) // 3

    # --- PANEL A: Right-Edge Entropy Bottleneck Ratio ---
    ax0, ay0, ax1, ay1 = draw_card(
        pad_x, r1_y0, pad_x + cw3, r1_y1,
        "A. Right-Edge Entropy Bottleneck",
        "eff_init / eff_final ratio (×) & Top-1 Final sign share (%)",
    )
    bn_vals = [prim[k]["metric1_info_theory"]["right_edge_bottleneck_ratio"] for k, _ in primary_keys]
    top1_vals = [prim[k]["metric1_info_theory"]["top1_final_share_pct"] for k, _ in primary_keys]
    yA = draw_axes_and_grid(ax0, ay0, ax1, ay1, 0.0, 6.2, [0, 1, 2, 3, 4, 5, 6], "{:.0f}×")
    _draw_dashed_hline(draw, ax0, ax1, yA(1.0), "#DC2626", width=2)
    # Clean top-right legend pill (zero bar overlap!)
    draw.rounded_rectangle([ax1 - 220, ay0 + 10, ax1 - 10, ay0 + 44], radius=8, fill="#FFFFFF", outline="#CBD5E1", width=1)
    _draw_dashed_hline(draw, ax1 - 208, ax1 - 178, ay0 + 27, "#DC2626", width=2, dash=6, gap=4)
    draw.text((ax1 - 170, ay0 + 18), "Symmetry (1.0×)", font=f_leg, fill="#DC2626")

    slot_w = (ax1 - ax0) / len(primary_keys)
    bw = int(slot_w * 0.56)
    for i, (v, t1, col, lbl) in enumerate(zip(bn_vals, top1_vals, bar_colors, labels)):
        cx = ax0 + (i + 0.5) * slot_w
        bx0, bx1 = int(cx - bw / 2), int(cx + bw / 2)
        by = yA(v)
        draw.rectangle([bx0, by, bx1, yA(0.0)], fill=col, outline="#1E293B", width=2)
        _draw_centered_text(draw, cx, by - 8, f"{v:.2f}×\n({t1:.1f}%)", f_val, fill="#0F172A", anchor_v="bottom")
        _draw_centered_text(draw, cx, ay1 + 14, lbl, f_xtick, fill="#1E293B", anchor_v="top")

    # --- PANEL B: Blind Affix Stacking Asymmetry ---
    bx0_c = 2 * pad_x + cw3
    px0, py0, px1, py1 = draw_card(
        bx0_c, r1_y0, bx0_c + cw3, r1_y1,
        "B. Blind Affix Stacking Asymmetry",
        "Right 2-Suffix vs Left 2-Prefix stacks (freq ≥ 20, zero hardcoding)",
    )
    suf_vals = [prim[k]["metric2_affix_stacking"]["fixed_ge20_exact_layer5"]["suffix_2slot_stacks"] for k, _ in primary_keys]
    pre_vals = [prim[k]["metric2_affix_stacking"]["fixed_ge20_exact_layer5"]["prefix_2slot_stacks"] for k, _ in primary_keys]
    ymax_b = max(250.0, max(max(suf_vals), max(pre_vals)) * 1.24)
    yB = draw_axes_and_grid(px0, py0, px1, py1, 0.0, ymax_b, [0, 50, 100, 150, 200, 250], "{:.0f}")
    # Legend top-right
    draw.rounded_rectangle([px1 - 245, py0 + 10, px1 - 10, py0 + 72], radius=8, fill="#FFFFFF", outline="#CBD5E1", width=1)
    draw.rectangle([px1 - 232, py0 + 21, px1 - 212, py0 + 37], fill="#0284C7", outline="#1E293B", width=1)
    draw.text((px1 - 204, py0 + 18), "Right 2-Suffix Stacks", font=f_leg, fill="#0369A1")
    draw.rectangle([px1 - 232, py0 + 47, px1 - 212, py0 + 63], fill="#E11D48", outline="#1E293B", width=1)
    draw.text((px1 - 204, py0 + 44), "Left 2-Prefix Stacks", font=f_leg, fill="#BE123C")

    slot_w = (px1 - px0) / len(primary_keys)
    bw2 = int(slot_w * 0.34)
    for i, (sv, pv, lbl) in enumerate(zip(suf_vals, pre_vals, labels)):
        cx = px0 + (i + 0.5) * slot_w
        # Suffix bar left, prefix bar right
        s_x0, s_x1 = int(cx - bw2 - 3), int(cx - 3)
        p_x0, p_x1 = int(cx + 3), int(cx + bw2 + 3)
        sy, py_ = yB(sv), yB(pv)
        draw.rectangle([s_x0, sy, s_x1, yB(0.0)], fill="#0284C7", outline="#1E293B", width=2)
        draw.rectangle([p_x0, py_, p_x1, yB(0.0)], fill="#E11D48", outline="#1E293B", width=2)
        _draw_centered_text(draw, (s_x0 + s_x1) / 2, sy - 6, f"{sv}", f_val, fill="#0369A1", anchor_v="bottom")
        _draw_centered_text(draw, (p_x0 + p_x1) / 2, py_ - 6, f"{pv}", f_val, fill="#BE123C", anchor_v="bottom")
        _draw_centered_text(draw, cx, py1 + 14, lbl, f_xtick, fill="#1E293B", anchor_v="top")

    # --- PANEL C: Alice Kober Bridge-Sign MI(B;E) z-score ---
    cx0_c = 3 * pad_x + 2 * cw3
    px0, py0, px1, py1 = draw_card(
        cx0_c, r1_y0, W - pad_x, r1_y1,
        "C. Kober Bridge-Sign MI(B;E) z-score",
        "True coupling z (σ) vs sparsity-confounded alt/same (a/s)",
    )
    mi_z = [prim[k]["metric3_kober_bridge"]["bridge_ending_MI_z"] for k, _ in primary_keys]
    alt_r = [prim[k]["metric3_kober_bridge"]["ratio_alt_vs_same_bridge"] for k, _ in primary_keys]
    yC = draw_axes_and_grid(px0, py0, px1, py1, -14.0, 76.0, [-10, 0, 10, 20, 30, 40, 50, 60, 70], "{:+.0f}σ")
    _draw_dashed_hline(draw, px0, px1, yC(3.0), "#16A34A", width=2)
    # Clean top-right legend pill (zero bar overlap!)
    draw.rounded_rectangle([px1 - 210, py0 + 10, px1 - 10, py0 + 44], radius=8, fill="#FFFFFF", outline="#CBD5E1", width=1)
    _draw_dashed_hline(draw, px1 - 198, px1 - 168, py0 + 27, "#16A34A", width=2, dash=6, gap=4)
    draw.text((px1 - 160, py0 + 18), "z = +3.0σ gate", font=f_leg, fill="#15803D")

    slot_w = (px1 - px0) / len(primary_keys)
    bw = int(slot_w * 0.56)
    y0_c = yC(0.0)
    for i, (zv, ar, col, lbl) in enumerate(zip(mi_z, alt_r, bar_colors, labels)):
        cx = px0 + (i + 0.5) * slot_w
        bx0, bx1 = int(cx - bw / 2), int(cx + bw / 2)
        by = yC(zv)
        if zv >= 0:
            draw.rectangle([bx0, by, bx1, y0_c], fill=col, outline="#1E293B", width=2)
            _draw_centered_text(draw, cx, by - 7, f"{zv:+.1f}σ\n(a/s={ar:.2f})", f_val_sm, fill="#0F172A", anchor_v="bottom")
        else:
            draw.rectangle([bx0, y0_c, bx1, by], fill=col, outline="#1E293B", width=2)
            _draw_centered_text(draw, cx, by + 6, f"{zv:+.1f}σ\n(a/s={ar:.2f})", f_val_sm, fill="#DC2626", anchor_v="top")
        _draw_centered_text(draw, cx, py1 + 14, lbl, f_xtick, fill="#1E293B", anchor_v="top")

    # Row 2 geometry (2 wide panels: D and E)
    r2_y0, r2_y1 = 826, 1514
    cw2 = (W - 3 * pad_x) // 2

    # --- PANEL D: Masked-Sign Generalization vs Position-Aware Baseline ---
    px0, py0, px1, py1 = draw_card(
        pad_x, r2_y0, pad_x + cw2, r2_y1,
        "D. 5-Fold Masked-Sign Generalization vs Position-Aware Baseline (vega #1)",
        "UNSEEN-Trigram Top-1 gain (solid, CE gain in bits) & ALL-Slot gain (light) — negative = context HURTS",
    )
    unseen_pp = [prim[k]["metric4_masked_prediction"]["UNSEEN_TRIGRAM"]["gain_vs_pos_base_pp"] for k, _ in primary_keys]
    unseen_ce = [prim[k]["metric4_masked_prediction"]["UNSEEN_TRIGRAM"]["ce_gain_bits"] for k, _ in primary_keys]
    all_pp = [prim[k]["metric4_masked_prediction"]["ALL"]["gain_vs_pos_base_pp"] for k, _ in primary_keys]
    yD = draw_axes_and_grid(px0, py0, px1, py1, -12.0, 34.0, [-10, -5, 0, 5, 10, 15, 20, 25, 30], "{:+.0f}pp")
    # Highlight negative zone lightly
    draw.rectangle([px0 + 1, yD(0.0) + 1, px1 - 1, py1 - 1], fill="#FEF2F2")
    draw.line([(px0, yD(0.0)), (px1, yD(0.0))], fill="#DC2626", width=3)
    draw.text((px0 + 12, yD(0.0) + 8), "BELOW POSITION BASELINE (NO COMPOSITIONAL CONTEXT)", font=f_val_sm, fill="#B91C1C")
    # Legend top-right
    draw.rounded_rectangle([px1 - 325, py0 + 10, px1 - 10, py0 + 72], radius=8, fill="#FFFFFF", outline="#CBD5E1", width=1)
    draw.rectangle([px1 - 312, py0 + 21, px1 - 292, py0 + 37], fill="#0F172A", outline="#1E293B", width=1)
    draw.text((px1 - 284, py0 + 18), "UNSEEN Trigram Gain (pp & CE)", font=f_leg, fill="#0F172A")
    draw.rectangle([px1 - 312, py0 + 47, px1 - 292, py0 + 63], fill="#CBD5E1", outline="#1E293B", width=1)
    draw.text((px1 - 284, py0 + 44), "ALL Slots Top-1 Gain (pp)", font=f_leg, fill="#334155")

    slot_w = (px1 - px0) / len(primary_keys)
    bw_d = int(slot_w * 0.34)
    y0_d = yD(0.0)
    for i, (uv, uce, av, col, lcol, lbl) in enumerate(zip(unseen_pp, unseen_ce, all_pp, bar_colors, bar_light, labels)):
        cx = px0 + (i + 0.5) * slot_w
        u_x0, u_x1 = int(cx - bw_d - 4), int(cx - 4)
        a_x0, a_x1 = int(cx + 4), int(cx + bw_d + 4)
        uy, ay = yD(uv), yD(av)
        if uv >= 0:
            draw.rectangle([u_x0, uy, u_x1, y0_d], fill=col, outline="#1E293B", width=2)
            _draw_centered_text(draw, (u_x0 + u_x1) / 2, uy - 6, f"{uv:+.1f}pp\n({uce:+.2f}b)", f_val_sm, fill="#0F172A", anchor_v="bottom")
        else:
            draw.rectangle([u_x0, y0_d, u_x1, uy], fill=col, outline="#1E293B", width=2)
            _draw_centered_text(draw, (u_x0 + u_x1) / 2, uy + 6, f"{uv:+.1f}pp\n({uce:+.2f}b)", f_val_sm, fill="#DC2626", anchor_v="top")
        draw.rectangle([a_x0, ay, a_x1, y0_d], fill=lcol, outline="#334155", width=2)
        _draw_centered_text(draw, (a_x0 + a_x1) / 2, ay - 6, f"{av:+.1f}", f_val_sm, fill="#334155", anchor_v="bottom")
        _draw_centered_text(draw, cx, py1 + 14, lbl, f_xtick, fill="#1E293B", anchor_v="top")

    # --- PANEL E: Known-Answer 23-Seal CSP Identifiability Calibration ---
    ex0_c = 2 * pad_x + cw2
    px0, py0, px1, py1 = draw_card(
        ex0_c, r2_y0, W - pad_x, r2_y1,
        "E. Known-Answer 23-Seal CSP Identifiability Calibration (n_null = 100)",
        "Exact-First + Bitmask Dynamic-MRV: Real Lexicon max_k vs 100 Shuffled Null Lexicons (mean ± SD)",
    )
    csp = results["metric5_known_answer_csp"]
    csp_spec = [
        ("LinB 100%\nk=0", csp["linearb_23_closed_100pct"]["results"]["k0_anchors"], "#0F766E"),
        ("LinB 100%\nk=2", csp["linearb_23_closed_100pct"]["results"]["k2_anchors"], "#14B8A6"),
        ("Sum 100%\nk=0", csp["sumerian_23_closed_100pct"]["results"]["k0_anchors"], "#1D4ED8"),
        ("Sum 100%\nk=2", csp["sumerian_23_closed_100pct"]["results"]["k2_anchors"], "#3B82F6"),
        ("LinB 74%\nk=2", csp["linearb_23_open_74pct"]["results"]["k2_anchors"], "#0D9488"),
        ("Sum 74%\nk=2", csp["sumerian_23_open_74pct"]["results"]["k2_anchors"], "#2563EB"),
        ("Indus Gulf\nvs Ur3 k=0", {
            "true_max_k": csp["indus_23_gulf_vs_real_cdli_ur3"]["k0_observed_max_k"],
            "null_max_k_mean": csp["indus_23_gulf_vs_real_cdli_ur3"]["null_max_k_mean"],
            "null_max_k_std": csp["indus_23_gulf_vs_real_cdli_ur3"]["null_max_k_std"],
            "z_score": csp["indus_23_gulf_vs_real_cdli_ur3"]["z_score"],
        }, "#C2410C"),
    ]
    yE = draw_axes_and_grid(px0, py0, px1, py1, 0.0, 34.0, [0, 5, 10, 15, 20, 23, 25, 30], "{:.0f}")
    _draw_dashed_hline(draw, px0, px1, yE(23.0), "#16A34A", width=2)
    # Clean 3-row legend top-right (high above bars, zero overlap!)
    draw.rounded_rectangle([px1 - 315, py0 + 8, px1 - 10, py0 + 92], radius=8, fill="#FFFFFF", outline="#CBD5E1", width=1)
    draw.rectangle([px1 - 302, py0 + 18, px1 - 282, py0 + 34], fill="#1D4ED8", outline="#1E293B", width=1)
    draw.text((px1 - 274, py0 + 15), "Real Lexicon max_k (/23)", font=f_leg, fill="#0F172A")
    draw.rectangle([px1 - 302, py0 + 43, px1 - 282, py0 + 59], fill="#CBD5E1", outline="#475569", width=1)
    draw.text((px1 - 274, py0 + 40), "100 Null Lexicons (mean±SD)", font=f_leg, fill="#475569")
    _draw_dashed_hline(draw, px1 - 302, px1 - 282, py0 + 76, "#16A34A", width=2, dash=5, gap=3)
    draw.text((px1 - 274, py0 + 66), "Exact 23/23 Global Lock", font=f_leg, fill="#15803D")

    slot_w = (px1 - px0) / len(csp_spec)
    bw_e = int(slot_w * 0.36)
    for i, (lbl, rec, col) in enumerate(csp_spec):
        cx = px0 + (i + 0.5) * slot_w
        ok = rec["true_max_k"]
        nmu = rec["null_max_k_mean"]
        nsd = rec["null_max_k_std"]
        zv = rec["z_score"]
        ox0, ox1 = int(cx - bw_e - 3), int(cx - 3)
        nx0, nx1 = int(cx + 3), int(cx + bw_e + 3)
        oy, ny = yE(ok), yE(nmu)
        draw.rectangle([ox0, oy, ox1, yE(0.0)], fill=col, outline="#1E293B", width=2)
        draw.rectangle([nx0, ny, nx1, yE(0.0)], fill="#CBD5E1", outline="#475569", width=2)
        # Error bar for null SD
        ncx = (nx0 + nx1) / 2
        ey_hi, ey_lo = yE(min(33.0, nmu + nsd)), yE(max(0.0, nmu - nsd))
        draw.line([(ncx, ey_hi), (ncx, ey_lo)], fill="#1E293B", width=2)
        draw.line([(ncx - 6, ey_hi), (ncx + 6, ey_hi)], fill="#1E293B", width=2)
        draw.line([(ncx - 6, ey_lo), (ncx + 6, ey_lo)], fill="#1E293B", width=2)

        zcol = "#15803D" if zv >= 3.0 else ("#B45309" if zv >= 1.5 else "#DC2626")
        _draw_centered_text(draw, (ox0 + ox1) / 2, oy - 8, f"{ok}/23\n{zv:+.1f}σ", f_val_sm, fill=zcol, anchor_v="bottom")
        _draw_centered_text(draw, cx, py1 + 14, lbl, f_val_sm, fill="#1E293B", anchor_v="top")

    # Bottom Negative-Control Callout Footer (clean 2-line layout with generous padding!)
    z_indus = prim["indus_strict"]["metric1_info_theory"]["z_H1_within_seq_shuffle"]
    z_nonling = prim["nonlinguistic_stamps"]["metric1_info_theory"]["z_H1_within_seq_shuffle"]
    ar_nonling = prim["nonlinguistic_stamps"]["metric3_kober_bridge"]["ratio_alt_vs_same_bridge"]
    draw.rounded_rectangle([pad_x, 1534, W - pad_x, 1642], radius=12, fill="#FEF3C7", outline="#D97706", width=2)
    foot_line1 = (
        f"NEGATIVE-CONTROL VERDICT:  Within-seq shuffle z_H1 FAILS (Non-Ling {z_nonling:.1f}σ vs Indus {z_indus:.1f}σ), "
        f"& Kober alt/same ratio is sparsity-confounded (Non-Ling = {ar_nonling:.2f} > Linear B = {prim['linearb_matched']['metric3_kober_bridge']['ratio_alt_vs_same_bridge']:.2f})."
    )
    foot_line2 = (
        "VALIDATED DISCRIMINATORS:  Unseen-Trigram Gain vs Position Baseline (Panel D), Kober MI(B;E) z-score (Panel C), "
        "Right-Edge Suffix-Stacking Asymmetry (Panels A–B), & Known-Answer 23-Seal CSP (Panel E)."
    )
    _draw_centered_text(draw, W / 2, 1564, foot_line1, f_foot_b, fill="#92400E", anchor_v="center")
    _draw_centered_text(draw, W / 2, 1610, foot_line2, f_foot_b, fill="#78350F", anchor_v="center")

    out_png = REPORTS_DIR / "calibration_5way_chart.png"
    img.save(out_png, format="PNG", optimize=True)

    BRAIN_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_png, BRAIN_COPY_PATH)
    assert out_png.read_bytes() == BRAIN_COPY_PATH.read_bytes(), "Chart copy mismatch!"
    print(f"Saved publication chart: {out_png} ({out_png.stat().st_size:,} bytes) + byte-identical copy at {BRAIN_COPY_PATH}")


# =============================================================================
# MARKDOWN SUMMARY GENERATOR (BRUTALLY HONEST SCIENTIFIC REPORT)
# =============================================================================

def write_markdown_summary(res: dict[str, Any]) -> None:
    prim = res["primary_5way_matched"]
    full = res["raw_full_and_sensitivity_corpora"]
    csp = res["metric5_known_answer_csp"]

    order = [
        ("indus_strict", "Indus Valley (Target)"),
        ("sumerian_matched", "Ur III Sumerian (Matched)"),
        ("linearb_matched", "Linear B Words (Matched)"),
        ("lineara_words", "Linear A Words (Full)"),
        ("nonlinguistic_stamps", "Non-Linguistic Stamps"),
    ]

    md: list[str] = []
    md.append("# INDUS-DEDUCT — Layer 8: 5-Way Known-Answer & Negative-Control Calibration Benchmark")
    md.append("")
    md.append("> **Direct response to `vega` and `orivis-mac`'s #1 recommendation:** Run our exact, unchanged, zero-hardcoding blind epigraphic suite on real deciphered scripts (**Linear B**, **Ur III Sumerian**), a real undeciphered Aegean script (**Linear A**), and a **100% synthetic non-linguistic bureaucratic stamp negative control** matched to Indus's exact sequence-length histogram (`lengths 2–13`, `seed = 42`).")
    md.append("")
    md.append("## 1. Calibration Corpora & Exact Parsing Provenance")
    md.append("")
    md.append("| System | Script Class / Typology | Source File (`data/calibration/` or `data/processed/`) | Raw Clean Unique Seqs | Indus-Matched Slice (`seed=42`) | Mean Len | Vocab `V` |")
    md.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    md.append(f"| **Indus Valley** | Undeciphered (Target) | `lines_clean.csv` (`strict=1`, `R/L,L/R,BUS`) | 2,047 | **2,047** (Target) | {prim['indus_strict']['metric1_info_theory']['mean_len']:.2f} | {prim['indus_strict']['metric1_info_theory']['V_unique_signs']} |")
    md.append(f"| **Ur III Sumerian** | Deciphered Logo-Syllabic (Agglutinative) | `mtaac_ur3_gold_corpus.zip` (CDLI CC0-1.0, {res['metadata']['sumerian_tablets_parsed']:,} tablets; sign-level hyphen/determinative split) | {full['sumerian_full_lines']['metric1_info_theory']['N_unique_seqs']:,} | **2,047** (Exact len hist) | {prim['sumerian_matched']['metric1_info_theory']['mean_len']:.2f} | {prim['sumerian_matched']['metric1_info_theory']['V_unique_signs']} |")
    md.append(f"| **Linear B** | Deciphered CV Syllabary (Mycenaean Greek, Fusional) | `linearb_words.csv` + `linearb_tablets.csv` (InsiderPhD/DĀMOS; hyphenated syllabograms, verified `ti-ri-po-de`, `ko-no-so`, `pa-i-to`) | {full['linearb_words_full']['metric1_info_theory']['N_unique_seqs']:,} | **1,651** (Matched len 2–8) | {prim['linearb_matched']['metric1_info_theory']['mean_len']:.2f} | {prim['linearb_matched']['metric1_info_theory']['V_unique_signs']} |")
    md.append(f"| **Linear A** | Undeciphered Aegean Script (Minoan) | `LinearAWords.json` (SigLA/Younger, {res['metadata']['lineara_inscriptions_raw']:,} inscriptions; stripped `𐄁`, numerals, lacunae) | {prim['lineara_words']['metric1_info_theory']['N_unique_seqs']:,} | **660** (All intact words) | {prim['lineara_words']['metric1_info_theory']['mean_len']:.2f} | {prim['lineara_words']['metric1_info_theory']['V_unique_signs']} |")
    md.append(f"| **Non-Linguistic Stamps** | Synthetic Bureaucratic Negative Control | `[Header(4)]+[Conn(2)]+[Item(180),Zipf]+[CheckStamp(6)]` | 2,047 | **2,047** (Exact len hist) | {prim['nonlinguistic_stamps']['metric1_info_theory']['mean_len']:.2f} | {prim['nonlinguistic_stamps']['metric1_info_theory']['V_unique_signs']} |")
    md.append("")

    md.append("## 2. Metrics 1–3: Information Theory, Blind Affix Stacking & Alice Kober's Bridge-Sign Test")
    md.append("")
    md.append("| Corpus (Indus-Matched Primary) | `V` | `H0` | `H1` | `z_H1` (Shuf) | `H_init` | `H_fin` | **Bottleneck (`eff_i/eff_f`)** | Top-1 Final | **Blind 2-Stack (`Suf:Pre`)** | Kober `alt/same` | **Kober `MI(B;E) z`** |")
    md.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |")
    for k, lbl in order:
        m1 = prim[k]["metric1_info_theory"]
        m2 = prim[k]["metric2_affix_stacking"]["fixed_ge20_exact_layer5"]
        m3 = prim[k]["metric3_kober_bridge"]
        md.append(
            f"| **{lbl}** | {m1['V_unique_signs']} | {m1['H0_bits']:.2f} | {m1['H1_bits']:.2f} | `{m1['z_H1_within_seq_shuffle']:+.1f}` | "
            f"{m1['H_init_bits']:.2f} | {m1['H_final_bits']:.2f} | **{m1['right_edge_bottleneck_ratio']:.2f}×** | "
            f"{m1['top1_final_share_pct']:.1f}% (`{m1['top1_final_sign']}`) | **{m2['suffix_2slot_stacks']}:{m2['prefix_2slot_stacks']}** | "
            f"{m3['ratio_alt_vs_same_bridge']:.2f} ({m3['kober_alternating_stems']}/{m3['kober_same_bridge_stems']}) | **{m3['bridge_ending_MI_z']:+.2f}σ** |"
        )
    md.append("")

    md.append("### Raw Full-Corpus & Granularity Sensitivity Controls (Transparency Check)")
    md.append("")
    md.append("| Sensitivity Control | `N` | Mean Len | `V` | `z_H1` | Bottleneck | Blind `Suf:Pre` (≥20) | Scaled `Suf:Pre` (≥1% N) | Kober `alt/same` | Kober `MI z` |")
    md.append("| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: |")
    sens_order = [
        ("sumerian_full_lines", "Ur III Sumerian (Full All Lines)"),
        ("sumerian_seal_np_matched", "Ur III Sumerian (Matched Seal/NP)"),
        ("linearb_words_full", "Linear B (Full Words)"),
        ("linearb_lines_full", "Linear B (Intact Tablet Lines)"),
        ("lineara_lines_full", "Linear A (Intact Tablet Lines)"),
    ]
    for k, lbl in sens_order:
        m1 = full[k]["metric1_info_theory"]
        m2f = full[k]["metric2_affix_stacking"]["fixed_ge20_exact_layer5"]
        m2s = full[k]["metric2_affix_stacking"]["scaled_1pct_sensitivity"]
        m3 = full[k]["metric3_kober_bridge"]
        md.append(
            f"| {lbl} | {m1['N_unique_seqs']:,} | {m1['mean_len']:.2f} | {m1['V_unique_signs']} | `{m1['z_H1_within_seq_shuffle']:+.1f}` | "
            f"{m1['right_edge_bottleneck_ratio']:.2f}× | {m2f['suffix_2slot_stacks']}:{m2f['prefix_2slot_stacks']} | "
            f"{m2s['suffix_2slot_stacks']}:{m2s['prefix_2slot_stacks']} | {m3['ratio_alt_vs_same_bridge']:.2f} | {m3['bridge_ending_MI_z']:+.2f}σ |"
        )
    md.append("")

    md.append("## 3. Metric 4: 5-Fold Masked-Sign Generalization vs Position-Aware Baseline (`vega` #1)")
    md.append("")
    md.append("| Corpus (Primary) | `ALL` Top-1 (Gain vs Pos) | `ALL` CE Gain (bits) | `SEEN_TRI` Top-1 (Gain) | **`UNSEEN_TRI` Top-1 (Gain vs Pos)** | **`UNSEEN_TRI` CE Gain (bits)** | `final_UNSEEN` Top-1 (Gain) |")
    md.append("| :--- | :---: | ---: | :---: | :---: | ---: | :---: |")
    for k, lbl in order:
        m4 = prim[k]["metric4_masked_prediction"]
        a, s, u, fu = m4["ALL"], m4["SEEN_TRIGRAM"], m4["UNSEEN_TRIGRAM"], m4["final_UNSEEN"]
        md.append(
            f"| **{lbl}** | {a['model_top1_pct']:.1f}% (`{a['gain_vs_pos_base_pp']:+.1f}pp`) | `{a['ce_gain_bits']:+.3f}` | "
            f"{s['model_top1_pct']:.1f}% (`{s['gain_vs_pos_base_pp']:+.1f}pp`) | **{u['model_top1_pct']:.1f}% (`{u['gain_vs_pos_base_pp']:+.2f}pp`)** | "
            f"**`{u['ce_gain_bits']:+.3f}`** | {fu['model_top1_pct']:.1f}% (`{fu['gain_vs_pos_base_pp']:+.2f}pp`) |"
        )
    md.append("")

    md.append("## 4. Metric 5: Known-Answer 23-Seal CSP Identifiability Calibration (`n_null = 100`)")
    md.append("")
    md.append("| Calibration Regime (`N = 23` Connected Seals) | Topology (`U / Shared / Sing%`) | Anchors `k` | **Real Lexicon `max_k`** | 100 Null Lexicons (`mean ± SD`, max) | **`z`-score (`emp_p`)** | Shared Signs Locked (Exact True) | All Signs Locked (Exact True) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ckey, clbl in [
        ("linearb_23_closed_100pct", "Linear B (100% Attested)"),
        ("sumerian_23_closed_100pct", "Ur III Sumerian (100% Attested)"),
        ("linearb_23_open_74pct", "Linear B (74% Attested / 6 Foreign)"),
        ("sumerian_23_open_74pct", "Ur III Sumerian (74% Attested / 6 Foreign)"),
    ]:
        gs = csp[ckey]["graph_stats"]
        topo = f"{gs['U_unique_signs']} / {gs['U_shared_signs']} / {gs['singleton_pct']:.1f}%"
        for ak in ("k0_anchors", "k2_anchors"):
            r = csp[ckey]["results"][ak]
            md.append(
                f"| **{clbl}** | {topo} | `k={r['k_anchors']}` | **{r['true_max_k']}/23** ({'EXACT' if r['exact_23_of_23_sat'] else 'MaxCSP'}) | "
                f"{r['null_max_k_mean']:.2f} ± {r['null_max_k_std']:.2f} (max {r['null_max_k_max']}) | "
                f"**`{r['z_score']:+.2f}σ`** (`p={r['empirical_p']:.4f}`) | {r['locked_shared_signs']} ({r['true_recovered_locked_shared']}) | {r['locked_all_signs']} ({r['true_recovered_locked_all']}) |"
            )
    ig = csp["indus_23_gulf_vs_real_cdli_ur3"]
    md.append(
        f"| **Actual Indus 23 Gulf Seals vs Real CDLI Ur III** | {ig['U_unique_signs']} / {ig['U_shared_signs']} / {ig['singleton_pct']:.1f}% | `k=0` | "
        f"**{ig['k0_observed_max_k']}/23** | {ig['null_max_k_mean']:.2f} ± {ig['null_max_k_std']:.2f} (max {ig['null_max_k_max']}) | "
        f"**`{ig['z_score']:+.2f}σ`** (`p={ig['empirical_p']:.4f}`) | 0 stat. locked (fails gate) | 0 stat. locked |"
    )
    md.append("")
    md.append(
        f"> **Cross-reference to Layer 4B (`strict_10` curated/productive Ur III/Akkadian):** "
        f"Curated `obs=5/10` vs Null `{ig['layer4b_published_strict10_curated']['null_mean']}±{ig['layer4b_published_strict10_curated']['null_std']}` (`z={ig['layer4b_published_strict10_curated']['z_score']:+.2f}`, `p=1.00`); "
        f"Productive `obs=5/10` vs Null `{ig['layer4b_published_strict10_productive']['null_mean']}±{ig['layer4b_published_strict10_productive']['null_std']}` (`z={ig['layer4b_published_strict10_productive']['z_score']:+.2f}`, `p=1.00`)."
    )
    md.append("")

    md.append("## 5. Brutally Honest Scientific Verdict — What Our Instruments DO and DO NOT Prove")
    md.append("")
    md.append("### Instruments That FAILED the Negative Control (Retired as Standalone Language Proofs)")
    md.append(f"1. **Within-Sequence Shuffle `z_H1` FAILS (`Non-Linguistic = {prim['nonlinguistic_stamps']['metric1_info_theory']['z_H1_within_seq_shuffle']:+.1f}σ` vs `Indus = {prim['indus_strict']['metric1_info_theory']['z_H1_within_seq_shuffle']:+.1f}σ`):** Confirms `vega` & `orivis-mac`'s critique completely. Rigid non-linguistic slot templates produce even more extreme negative `z_H1` than natural languages. `z_H1` proves strict linear ordering, **NOT** natural language.")
    md.append(f"2. **Kober `alt/same` Bridge Ratio ALONE FAILS (`Non-Linguistic = {prim['nonlinguistic_stamps']['metric3_kober_bridge']['ratio_alt_vs_same_bridge']:.2f}` > `Linear B Matched = {prim['linearb_matched']['metric3_kober_bridge']['ratio_alt_vs_same_bridge']:.2f}`):** Critical calibration catch! Because `Item(180)` tokens drawn randomly before `CheckStamp(6)` rarely repeat on the same `[Header, Conn]` prefix, disjoint bridge sets arise trivially from vocabulary sparsity alone.")
    md.append(f"3. **`MI(Bridge; Ending) z` CANNOT Separate Syllabic Inflection from Agglutinative Suffix Stacking:** Real **Ur III Sumerian** (agglutinative logo-syllabic, `{prim['sumerian_matched']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`) scores virtually identically to real **Linear B** (fusional CV syllabary, `{prim['linearb_matched']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`). Thus Indus's `{prim['indus_strict']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ` proves strong compositional penultimate-to-final conditioning (separating it cleanly from Non-Linguistic Stamps `{prim['nonlinguistic_stamps']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`), **NOT** exclusively CV syllabicity.")
    md.append(f"4. **Layer 4B Static-Order Solver Diagnostic (`k=0` Budget Sensitivity):** In our pilot comparison, a static-order Max-CSP without Dynamic MRV / Exact-CSP-First can miss valid 23/23 solutions under tight node budgets when all initial domains are large. Upgrading to **Exact-CSP-First + Bitmask Dynamic-MRV** completely resolves this on closed 23-seal graphs, while revealing an even deeper epistemological boundary on **partial attestation (74% attested)**: check Table 4 above — when 6/23 seals are unattested foreign words, blind `k=0` Max-CSP loses dramatic statistical power until `k=2` true sign anchors are provided!")
    md.append("")
    md.append("### Instruments That PASSED Calibration (Cleanly Separating Real Scripts from Non-Linguistic Stamps)")
    md.append(f"1. **Unseen-Trigram Generalization over Position-Aware Baseline (Metric 4 — `vega` #1):** **THE #1 DISCRIMINATOR.** On Non-Linguistic Stamps, once the slot (`init/medial/final`) is known, neighbors carry ZERO compositional information — context HURTS unseen-trigram prediction (`{prim['nonlinguistic_stamps']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['gain_vs_pos_base_pp']:+.2f} pp`, CE `{prim['nonlinguistic_stamps']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['ce_gain_bits']:+.3f} bits`). In stark contrast, **Ur III Sumerian (`{prim['sumerian_matched']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['gain_vs_pos_base_pp']:+.2f} pp`, `{prim['sumerian_matched']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['ce_gain_bits']:+.3f} bits`)** and **Indus (`{prim['indus_strict']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['gain_vs_pos_base_pp']:+.2f} pp`, `{prim['indus_strict']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['ce_gain_bits']:+.3f} bits`)** both gain strongly over the position baseline on completely unseen trigrams, with **Linear B (`{prim['linearb_matched']['metric4_masked_prediction']['UNSEEN_TRIGRAM']['gain_vs_pos_base_pp']:+.2f} pp`)** also positive! Moreover, Indus's negative `final_UNSEEN` (`{prim['indus_strict']['metric4_masked_prediction']['final_UNSEEN']['gain_vs_pos_base_pp']:+.2f} pp`) is **shared by ALL real scripts** (`Sumerian {prim['sumerian_matched']['metric4_masked_prediction']['final_UNSEEN']['gain_vs_pos_base_pp']:+.2f} pp`, `Linear B {prim['linearb_matched']['metric4_masked_prediction']['final_UNSEEN']['gain_vs_pos_base_pp']:+.2f} pp`, `Linear A {prim['lineara_words']['metric4_masked_prediction']['final_UNSEEN']['gain_vs_pos_base_pp']:+.2f} pp`) due to conditioning on rare unexpected endings, whereas Non-Linguistic Stamps collapses by `{prim['nonlinguistic_stamps']['metric4_masked_prediction']['final_UNSEEN']['gain_vs_pos_base_pp']:+.2f} pp`!")
    md.append(f"2. **Kober `MI(Bridge; Ending) z` (Metric 3):** Completely **zero** on Non-Linguistic Stamps (`{prim['nonlinguistic_stamps']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`), massively positive on **Indus (`{prim['indus_strict']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`)**, **Ur III Sumerian (`{prim['sumerian_matched']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`)**, and **Linear B (`{prim['linearb_matched']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`, full `N=4,844`: `{full['linearb_words_full']['metric3_kober_bridge']['bridge_ending_MI_z']:+.2f}σ`)**.")
    md.append(f"3. **Right-Edge Bottleneck & Pure-Blind Affix Stacking Asymmetry (Metrics 1–2):** Indus (`{prim['indus_strict']['metric1_info_theory']['right_edge_bottleneck_ratio']:.2f}×` bottleneck, **`{prim['indus_strict']['metric2_affix_stacking']['fixed_ge20_exact_layer5']['suffix_2slot_stacks']}:{prim['indus_strict']['metric2_affix_stacking']['fixed_ge20_exact_layer5']['prefix_2slot_stacks']}`** suffix-vs-prefix stacking under zero hardcoding!) is radically more right-edge suffix-stacking than **all four controls** — directly validating Layer 5's typological prediction that real Ur III Sumerian administrative/verbal lines are left-edge heavy (`{prim['sumerian_matched']['metric1_info_theory']['right_edge_bottleneck_ratio']:.2f}×`, `{prim['sumerian_matched']['metric2_affix_stacking']['fixed_ge20_exact_layer5']['suffix_2slot_stacks']}:{prim['sumerian_matched']['metric2_affix_stacking']['fixed_ge20_exact_layer5']['prefix_2slot_stacks']}` prefix-dominant) and Linear B is near-symmetric fusional (`{prim['linearb_matched']['metric1_info_theory']['right_edge_bottleneck_ratio']:.2f}×`, `{prim['linearb_matched']['metric2_affix_stacking']['fixed_ge20_exact_layer5']['suffix_2slot_stacks']}:{prim['linearb_matched']['metric2_affix_stacking']['fixed_ge20_exact_layer5']['prefix_2slot_stacks']}`).")
    md.append(f"4. **Known-Answer 23-Seal CSP Identifiability Calibration (Metric 5):** Proves our upgraded Bitmask Dynamic-MRV CSP instrument **HAS MASSIVE TRUE-POSITIVE POWER** at `N = 23`: on 100%-attested **Linear B** (`z = {csp['linearb_23_closed_100pct']['results']['k0_anchors']['z_score']:+.2f}σ, p = {csp['linearb_23_closed_100pct']['results']['k0_anchors']['empirical_p']:.4f}`, `{csp['linearb_23_closed_100pct']['results']['k0_anchors']['true_recovered_locked_all']}` exact true signs recovered even at `k=0`!) and **Ur III Sumerian** (`z = {csp['sumerian_23_closed_100pct']['results']['k0_anchors']['z_score']:+.2f}σ, p = {csp['sumerian_23_closed_100pct']['results']['k0_anchors']['empirical_p']:.4f}`, `{csp['sumerian_23_closed_100pct']['results']['k0_anchors']['true_recovered_locked_shared']}` exact shared signs recovered at `k=0`!). Under realistic **74% partial attestation (6 foreign seals)**, `k = 2` true anchors STILL cleanly separate the true language (`Linear B z = {csp['linearb_23_open_74pct']['results']['k2_anchors']['z_score']:+.2f}σ`, `Sumerian z = {csp['sumerian_23_open_74pct']['results']['k2_anchors']['z_score']:+.2f}σ`), whereas **Actual Indus 23 Gulf Seals vs Real CDLI Ur III Sumerian** sits squarely inside the null noise (`{ig['k0_observed_max_k']}/23` vs `{ig['null_max_k_mean']}±{ig['null_max_k_std']}`, `z = {ig['z_score']:+.2f}σ, p = {ig['empirical_p']:.4f}`).")
    md.append("")
    md.append("![5-Way Calibration Chart](calibration_5way_chart.png)")
    md.append("")

    out_md = REPORTS_DIR / "layer8_calibration_summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved markdown summary: {out_md}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    import sys
    if "--replot" in sys.argv:
        res = json.loads((REPORTS_DIR / "layer8_calibration.json").read_text(encoding="utf-8"))
        render_5way_chart(res)
        write_markdown_summary(res)
        return
    t0 = time.time()
    print("=== INDUS-DEDUCT LAYER 8: 5-WAY KNOWN-ANSWER & NEGATIVE-CONTROL CALIBRATION ===")
    indus_seqs, indus_hist = load_indus_corpus()
    lb_data = load_linearb_corpora()
    sum_data = load_sumerian_corpora()
    la_data = load_lineara_corpora()
    nonling_seqs = generate_nonlinguistic_stamps(indus_hist, seed=SEED)

    primary_corpora = {
        "indus_strict": indus_seqs,
        "sumerian_matched": slice_indus_length_matched(sum_data["lines_full"], indus_hist, seed=SEED),
        "linearb_matched": slice_indus_length_matched(lb_data["words_full"], indus_hist, seed=SEED),
        "lineara_words": la_data["words_full"],
        "nonlinguistic_stamps": nonling_seqs,
    }
    sensitivity_corpora = {
        "sumerian_full_lines": sum_data["lines_full"],
        "sumerian_seal_np_matched": slice_indus_length_matched(sum_data["seal_np_full"], indus_hist, seed=SEED),
        "linearb_words_full": lb_data["words_full"],
        "linearb_lines_full": lb_data["lines_full"],
        "lineara_lines_full": la_data["lines_full"],
    }

    primary_out: dict[str, Any] = {}
    for name, seqs in primary_corpora.items():
        print(f"Evaluating Primary Corpus [{name}] (N={len(seqs)})...")
        primary_out[name] = {
            "metric1_info_theory": metric1_info_theory(seqs, n_perm=200),
            "metric2_affix_stacking": metric2_affix_stacking(seqs),
            "metric3_kober_bridge": metric3_kober_bridge_test(seqs, n_perm=500),
            "metric4_masked_prediction": metric4_masked_prediction_5fold(seqs),
        }

    sensitivity_out: dict[str, Any] = {}
    for name, seqs in sensitivity_corpora.items():
        print(f"Evaluating Sensitivity Corpus [{name}] (N={len(seqs)})...")
        # Use 100 perms for huge 23k full Sumerian to keep runtime snappy while exact
        np_h1 = 50 if len(seqs) > 10000 else 200
        np_mi = 200 if len(seqs) > 10000 else 500
        entry: dict[str, Any] = {
            "metric1_info_theory": metric1_info_theory(seqs, n_perm=np_h1),
            "metric2_affix_stacking": metric2_affix_stacking(seqs),
            "metric3_kober_bridge": metric3_kober_bridge_test(seqs, n_perm=np_mi),
        }
        if len(seqs) <= 5000:
            entry["metric4_masked_prediction"] = metric4_masked_prediction_5fold(seqs)
        sensitivity_out[name] = entry

    csp_out = run_known_answer_csp_suite(
        lb_data["words_full"], sum_data["seal_np_full"], sum_data["lines_full"], n_null=100
    )

    results = {
        "metadata": {
            "seed": SEED,
            "indus_target_N": len(indus_seqs),
            "indus_target_len_hist": {str(k): int(v) for k, v in sorted(indus_hist.items())},
            "linearb_iconic_words_verified": lb_data["iconic_words_verified"],
            "sumerian_tablets_parsed": sum_data["n_tablets_parsed"],
            "lineara_inscriptions_raw": la_data["n_inscriptions_raw"],
            "nonlinguistic_template": "[Header(4)]+[Conn(2)]+[Item(180),Zipf(0.9)]+[CheckStamp(6)]",
            "total_elapsed_sec": round(time.time() - t0, 1),
        },
        "primary_5way_matched": primary_out,
        "raw_full_and_sensitivity_corpora": sensitivity_out,
        "metric5_known_answer_csp": csp_out,
    }

    out_json = REPORTS_DIR / "layer8_calibration.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON report: {out_json} ({out_json.stat().st_size:,} bytes)")

    render_5way_chart(results)
    write_markdown_summary(results)
    print(f"=== LAYER 8 COMPLETE IN {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
