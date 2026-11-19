#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layers 1 & 2: Top-Down Deductive Structural Engine
=================================================================
Performs strictly unsupervised, falsifiable computational epigraphy on the
uncontaminated canonical Indus corpus:
  [Layer 1] Compound decomposition & distributional allograph discovery
  [Layer 2A] Information-theoretic syntax proof vs. null permutation model
  [Layer 2B] Positional slot taxonomy (Initial / Medial / Terminal / Bound Morphemes)
  [Layer 2C] Out-of-sample Masked-Sign Prediction (Prize Criterion #2, zero leakage)
  [Layer 2D] Cross-modal Archaeological Correlations (Site x Sign, Motif x Sign, ObjectType x Sign)
  [Layer 2->4] Foreign-Seal Surprisal Ranker (Mesopotamia / Gulf / Iran / Central Asia Rosetta Targets)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

BOS = "<BOS>"
EOS = "<EOS>"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ins_df = pd.read_csv(PROC_DIR / "inscriptions_clean.csv", dtype=str).fillna("")
    lines_df = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    glyphs_df = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")
    sim_df = pd.read_csv(PROC_DIR / "glyph_similarity.csv", dtype=str).fillna("")
    for col in ["is_strict_analytic", "is_foreign", "token_count_clean", "has_damage_token"]:
        lines_df[col] = lines_df[col].astype(int)
    for col in ["freq_all", "freq_strict_analytic", "component_count", "is_pua_compound"]:
        glyphs_df[col] = glyphs_df[col].astype(int)
    return ins_df, lines_df, glyphs_df, sim_df


def get_oriented_strict_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Strict complete, undamaged lines with verified canonical reading orientation."""
    return lines_df[
        (lines_df["is_strict_analytic"] == 1)
        & (lines_df["direction"].isin(["R/L", "L/R", "BUS"]))
    ].copy()


# ---------------------------------------------------------------------------
# LAYER 1: GRAPHEME REDUCTION (COMPOUNDS + DISTRIBUTIONAL ALLOGRAPHS)
# ---------------------------------------------------------------------------
def run_layer1_graphemes(
    strict_df: pd.DataFrame, glyphs_df: pd.DataFrame, sim_df: pd.DataFrame
) -> dict[str, Any]:
    # Deduplicated sequences prevent formula inflation in distributional embeddings
    unique_seqs = [s.split() for s in strict_df["clean_reading_seq"].unique()]

    # 1A. PUA Compound breakdown
    compounds = glyphs_df[glyphs_df["is_pua_compound"] == 1].sort_values("freq_all", ascending=False)
    top_compounds = [
        {
            "sign_id": r["sign_id"],
            "freq_all": int(r["freq_all"]),
            "decomposition": r["pua_decomposition"],
            "components": int(r["component_count"]),
        }
        for _, r in compounds.head(25).iterrows()
    ]

    # 1B. Build Left/Right/Position context matrix for signs with freq >= 8 in unique seqs
    sign_counts = Counter(tok for seq in unique_seqs for tok in seq)
    vocab = sorted([s for s, c in sign_counts.items() if c >= 8])
    v_idx = {s: i for i, s in enumerate(vocab)}

    ctx_counts: dict[str, Counter] = defaultdict(Counter)
    cooccur_same_line: Counter = Counter()
    total_ctx_events = 0

    for seq in unique_seqs:
        padded = [BOS] + seq + [EOS]
        n = len(seq)
        # Track co-occurrence in same line for complementary distribution test
        uniq_in_line = sorted(set(seq))
        for i in range(len(uniq_in_line)):
            for j in range(i + 1, len(uniq_in_line)):
                cooccur_same_line[(uniq_in_line[i], uniq_in_line[j])] += 1

        for idx, tok in enumerate(seq):
            if tok not in v_idx:
                continue
            left_tok = padded[idx]
            right_tok = padded[idx + 2]
            pos_tag = (
                "POS_INIT" if idx == 0 else ("POS_FINAL" if idx == n - 1 else "POS_MEDIAL")
            )
            for feat in (f"L:{left_tok}", f"R:{right_tok}", pos_tag):
                ctx_counts[tok][feat] += 1
                total_ctx_events += 1

    # Feature vocabulary (context features appearing >= 3 times)
    feat_totals = Counter()
    for c in ctx_counts.values():
        feat_totals.update(c)
    feats = sorted([f for f, cnt in feat_totals.items() if cnt >= 3])
    f_idx = {f: i for i, f in enumerate(feats)}

    mat = np.zeros((len(vocab), len(feats)), dtype=np.float64)
    for s, c in ctx_counts.items():
        i = v_idx[s]
        for f, cnt in c.items():
            if f in f_idx:
                mat[i, f_idx[f]] = cnt

    # Positive Pointwise Mutual Information (PPMI)
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-12
    col_sums = mat.sum(axis=0, keepdims=True) + 1e-12
    total = mat.sum() + 1e-12
    expected = (row_sums @ col_sums) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(np.maximum(mat, 1e-12) / expected)
    ppmi = np.maximum(pmi, 0.0)
    ppmi[mat == 0] = 0.0

    n_comp = min(32, len(vocab) - 1, len(feats) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(ppmi)
    sim_mat = cosine_similarity(emb)

    # Known visual similarity edges
    vis_edges = {(r["sign_a"], r["sign_b"]) for _, r in sim_df.iterrows()}

    # Find top distributional allograph candidates (high cosine sim + complementary distribution)
    allograph_candidates = []
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            csim = float(sim_mat[i, j])
            if csim >= 0.72:
                sa, sb = vocab[i], vocab[j]
                pair = (min(sa, sb), max(sa, sb))
                cooc = cooccur_same_line.get(pair, 0)
                min_freq = min(sign_counts[sa], sign_counts[sb])
                cooc_ratio = cooc / max(min_freq, 1)
                if cooc_ratio <= 0.05:  # Complementary distribution: rarely/never co-occur
                    allograph_candidates.append(
                        {
                            "sign_a": sa,
                            "sign_b": sb,
                            "freq_a": int(sign_counts[sa]),
                            "freq_b": int(sign_counts[sb]),
                            "distributional_cosine": round(csim, 4),
                            "same_line_cooccur": int(cooc),
                            "visual_sim_confirmed": pair in vis_edges,
                        }
                    )
    allograph_candidates.sort(key=lambda x: (-int(x["visual_sim_confirmed"]), -x["distributional_cosine"]))

    # Also score visual similarity pairs that exist in our vocab
    visual_edge_checks = []
    for sa, sb in sorted(vis_edges):
        if sa in v_idx and sb in v_idx:
            csim = float(sim_mat[v_idx[sa], v_idx[sb]])
            cooc = cooccur_same_line.get((min(sa, sb), max(sa, sb)), 0)
            visual_edge_checks.append(
                {
                    "sign_a": sa,
                    "sign_b": sb,
                    "freq_a": int(sign_counts[sa]),
                    "freq_b": int(sign_counts[sb]),
                    "distributional_cosine": round(csim, 4),
                    "same_line_cooccur": int(cooc),
                }
            )
    visual_edge_checks.sort(key=lambda x: -x["distributional_cosine"])

    return {
        "unique_strict_sequences": len(unique_seqs),
        "pua_compound_total": int(len(compounds)),
        "top_pua_compounds": top_compounds,
        "distributional_vocab_size_ge8": len(vocab),
        "allograph_candidates_count": len(allograph_candidates),
        "top_allograph_candidates": allograph_candidates[:25],
        "visual_edges_tested_in_vocab": len(visual_edge_checks),
        "top_visual_edges_by_distributional_sim": visual_edge_checks[:15],
    }


# ---------------------------------------------------------------------------
# LAYER 2A: ENTROPY & NULL-MODEL SYNTAX PROOF
# ---------------------------------------------------------------------------
def compute_bigram_conditional_entropy(seqs: list[list[str]]) -> tuple[float, float]:
    """Compute unigram H(X) and conditional bigram H(X_t | X_{t-1}) over padded sequences."""
    uni = Counter()
    bi = Counter()
    prev_tot = Counter()
    for seq in seqs:
        padded = [BOS] + seq + [EOS]
        for tok in seq:
            uni[tok] += 1
        for a, b in zip(padded[:-1], padded[1:]):
            bi[(a, b)] += 1
            prev_tot[a] += 1

    n_uni = sum(uni.values())
    h0 = -sum((c / n_uni) * math.log2(c / n_uni) for c in uni.values())

    n_bi = sum(bi.values())
    h1 = 0.0
    for (a, b), cnt in bi.items():
        p_ab = cnt / n_bi
        p_b_given_a = cnt / prev_tot[a]
        h1 -= p_ab * math.log2(p_b_given_a)
    return h0, h1


def run_layer2a_entropy(strict_df: pd.DataFrame, n_perm: int = 200) -> dict[str, Any]:
    unique_seqs = [s.split() for s in strict_df["clean_reading_seq"].unique()]
    h0, h1_real = compute_bigram_conditional_entropy(unique_seqs)

    # Null model: randomly shuffle sign order WITHIN each inscription (preserving length & unigram distribution!)
    rng = np.random.default_rng(42)
    h1_null = []
    for _ in range(n_perm):
        shuf_seqs = []
        for seq in unique_seqs:
            arr = list(seq)
            rng.shuffle(arr)
            shuf_seqs.append(arr)
        _, h1_s = compute_bigram_conditional_entropy(shuf_seqs)
        h1_null.append(h1_s)

    null_mean = float(np.mean(h1_null))
    null_std = float(np.std(h1_null) + 1e-12)
    z_score = (h1_real - null_mean) / null_std

    # Also test Reverse vs Forward directional asymmetry: H(X_t | X_{t-1}) vs H(X_{t-1} | X_t)
    rev_seqs = [list(reversed(s)) for s in unique_seqs]
    _, h1_rev = compute_bigram_conditional_entropy(rev_seqs)

    return {
        "n_unique_sequences": len(unique_seqs),
        "unigram_entropy_H0_bits": round(h0, 4),
        "bigram_cond_entropy_H1_bits": round(h1_real, 4),
        "reverse_bigram_cond_entropy_bits": round(h1_rev, 4),
        "null_shuffled_H1_mean": round(null_mean, 4),
        "null_shuffled_H1_std": round(null_std, 4),
        "syntax_order_z_score": round(z_score, 2),
        "entropy_drop_vs_shuffled_bits": round(null_mean - h1_real, 4),
    }


# ---------------------------------------------------------------------------
# LAYER 2B: POSITIONAL SLOT GRAMMAR & BOUND MORPHEME DISCOVERY
# ---------------------------------------------------------------------------
def run_layer2b_grammar(strict_df: pd.DataFrame) -> dict[str, Any]:
    unique_seqs = [s.split() for s in strict_df["clean_reading_seq"].unique()]
    pos_counts: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    bigrams = Counter()
    trigrams = Counter()

    for seq in unique_seqs:
        n = len(seq)
        for i, tok in enumerate(seq):
            totals[tok] += 1
            if i == 0:
                pos_counts[tok]["init"] += 1
            elif i == n - 1:
                pos_counts[tok]["final"] += 1
            else:
                pos_counts[tok]["medial"] += 1
        for a, b in zip(seq[:-1], seq[1:]):
            bigrams[(a, b)] += 1
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            trigrams[(a, b, c)] += 1

    # Slot classification for signs with freq >= 20 in unique sequences
    slot_table = []
    for s, tot in totals.most_common():
        if tot < 20:
            continue
        p_init = pos_counts[s]["init"] / tot
        p_med = pos_counts[s]["medial"] / tot
        p_fin = pos_counts[s]["final"] / tot
        if p_fin >= 0.65:
            role = "TERMINAL_SUFFIX"
        elif p_init >= 0.55:
            role = "INITIAL_PREFIX"
        elif p_med >= 0.65:
            role = "MEDIAL_STEM_OR_LINK"
        else:
            role = "FLEXIBLE_CORE"
        slot_table.append(
            {
                "sign_id": s,
                "freq_unique": int(tot),
                "p_init": round(p_init, 3),
                "p_medial": round(p_med, 3),
                "p_final": round(p_fin, 3),
                "slot_role": role,
            }
        )

    # Top PMI bigrams (freq >= 10) -> tightly bound morpheme / compound word candidates
    n_bi = sum(bigrams.values())
    n_uni = sum(totals.values())
    pmi_bigrams = []
    for (a, b), cnt in bigrams.items():
        if cnt < 10:
            continue
        p_ab = cnt / n_bi
        p_a = totals[a] / n_uni
        p_b = totals[b] / n_uni
        pmi = math.log2(p_ab / (p_a * p_b))
        # Forward & backward transition probability (measures how tightly glued a->b is)
        p_b_given_a = cnt / totals[a]
        p_a_given_b = cnt / totals[b]
        pmi_bigrams.append(
            {
                "bigram": f"{a} -> {b}",
                "count": int(cnt),
                "pmi": round(pmi, 3),
                "P(b|a)": round(p_b_given_a, 3),
                "P(a|b)": round(p_a_given_b, 3),
            }
        )
    pmi_bigrams.sort(key=lambda x: -x["pmi"])

    top_trigrams = [
        {"trigram": f"{a} -> {b} -> {c}", "count": int(cnt)}
        for (a, b, c), cnt in trigrams.most_common(20)
    ]

    role_counts = Counter(r["slot_role"] for r in slot_table)
    return {
        "slot_role_summary_ge20": dict(role_counts),
        "top_terminal_signs": [r for r in slot_table if r["slot_role"] == "TERMINAL_SUFFIX"][:15],
        "top_initial_signs": [r for r in slot_table if r["slot_role"] == "INITIAL_PREFIX"][:15],
        "top_medial_signs": [r for r in slot_table if r["slot_role"] == "MEDIAL_STEM_OR_LINK"][:15],
        "top_pmi_bound_bigrams": pmi_bigrams[:20],
        "top_trigrams": top_trigrams,
    }


# ---------------------------------------------------------------------------
# LAYER 2C: OUT-OF-SAMPLE MASKED-SIGN PREDICTION (PRIZE CRITERION #2)
# ---------------------------------------------------------------------------
def run_layer2c_masked_prediction(strict_df: pd.DataFrame, k_folds: int = 5) -> dict[str, Any]:
    """
    Zero-leakage 5-fold CV on DEDUPLICATED unique sequences of length >= 3.
    Predicts each masked token using Bidirectional Interpolated Kneser-style N-gram
    (left bigram + right bigram + trigram surround + position prior).
    """
    unique_seqs = [s.split() for s in strict_df["clean_reading_seq"].unique() if len(s.split()) >= 3]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(unique_seqs))
    folds = np.array_split(order, k_folds)

    top1_hits = top5_hits = top10_hits = 0
    base_top1 = base_top5 = 0
    total_preds = 0
    by_pos = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})

    for fold_idx in range(k_folds):
        test_idx = set(folds[fold_idx].tolist())
        train_seqs = [unique_seqs[i] for i in range(len(unique_seqs)) if i not in test_idx]
        test_seqs = [unique_seqs[i] for i in folds[fold_idx]]

        uni = Counter()
        pos_uni = defaultdict(Counter)
        left_bi = defaultdict(Counter)   # prev -> curr
        right_bi = defaultdict(Counter)  # next -> curr
        surround = defaultdict(Counter)  # (prev, next) -> curr
        left_tri = defaultdict(Counter)  # (prev2, prev1) -> curr

        for seq in train_seqs:
            padded = [BOS, BOS] + seq + [EOS]
            n = len(seq)
            for i, tok in enumerate(seq):
                uni[tok] += 1
                ptag = "init" if i == 0 else ("final" if i == n - 1 else "medial")
                pos_uni[ptag][tok] += 1
                prev2, prev1, nxt = padded[i], padded[i + 1], padded[i + 3]
                left_bi[prev1][tok] += 1
                right_bi[nxt][tok] += 1
                surround[(prev1, nxt)][tok] += 1
                left_tri[(prev2, prev1)][tok] += 1

        vocab_list = [s for s, _ in uni.most_common()]
        n_uni = sum(uni.values())
        base_top1_sign = vocab_list[0]
        base_top5_set = set(vocab_list[:5])

        for seq in test_seqs:
            padded = [BOS, BOS] + seq + [EOS]
            n = len(seq)
            for i, true_tok in enumerate(seq):
                ptag = "init" if i == 0 else ("final" if i == n - 1 else "medial")
                prev2, prev1, nxt = padded[i], padded[i + 1], padded[i + 3]

                # Candidate pool: union of observed context candidates + top 30 unigrams
                cands = set(vocab_list[:30])
                cands.update(left_bi[prev1].keys())
                cands.update(right_bi[nxt].keys())
                cands.update(surround[(prev1, nxt)].keys())
                cands.update(left_tri[(prev2, prev1)].keys())

                l_tot = sum(left_bi[prev1].values())
                r_tot = sum(right_bi[nxt].values())
                s_tot = sum(surround[(prev1, nxt)].values())
                lt_tot = sum(left_tri[(prev2, prev1)].values())
                p_tot = sum(pos_uni[ptag].values())

                scored = []
                for cand in cands:
                    # Smoothed log-linear / linear mixture of exact directional evidence
                    p_u = (uni[cand] + 0.1) / (n_uni + 0.1 * len(vocab_list))
                    p_p = (pos_uni[ptag][cand] + 0.1) / (p_tot + 0.1 * len(vocab_list))
                    p_l = (left_bi[prev1][cand] + 0.5 * p_p) / (l_tot + 0.5) if l_tot > 0 else p_p
                    p_r = (right_bi[nxt][cand] + 0.5 * p_p) / (r_tot + 0.5) if r_tot > 0 else p_p
                    p_s = (surround[(prev1, nxt)][cand] + 0.3 * p_l) / (s_tot + 0.3) if s_tot > 0 else 0.5 * (p_l + p_r)
                    p_lt = (left_tri[(prev2, prev1)][cand] + 0.3 * p_l) / (lt_tot + 0.3) if lt_tot > 0 else p_l

                    # Geometric-arithmetic blend favoring strong bidirectional & surround matches
                    score = (
                        0.32 * math.log(p_s + 1e-12)
                        + 0.25 * math.log(p_l + 1e-12)
                        + 0.25 * math.log(p_r + 1e-12)
                        + 0.10 * math.log(p_lt + 1e-12)
                        + 0.08 * math.log(p_p + 1e-12)
                    )
                    scored.append((score, cand))

                scored.sort(reverse=True)
                preds = [c for _, c in scored[:10]]

                total_preds += 1
                by_pos[ptag]["n"] += 1
                if preds[0] == true_tok:
                    top1_hits += 1
                    by_pos[ptag]["top1"] += 1
                if true_tok in preds[:5]:
                    top5_hits += 1
                    by_pos[ptag]["top5"] += 1
                if true_tok in preds[:10]:
                    top10_hits += 1

                if true_tok == base_top1_sign:
                    base_top1 += 1
                if true_tok in base_top5_set:
                    base_top5 += 1

    return {
        "n_unique_sequences_len_ge3": len(unique_seqs),
        "total_masked_positions_evaluated": total_preds,
        "baseline_unigram_top1_pct": round(100.0 * base_top1 / total_preds, 2),
        "baseline_unigram_top5_pct": round(100.0 * base_top5 / total_preds, 2),
        "model_top1_pct": round(100.0 * top1_hits / total_preds, 2),
        "model_top5_pct": round(100.0 * top5_hits / total_preds, 2),
        "model_top10_pct": round(100.0 * top10_hits / total_preds, 2),
        "by_position": {
            k: {
                "n": v["n"],
                "top1_pct": round(100.0 * v["top1"] / max(v["n"], 1), 2),
                "top5_pct": round(100.0 * v["top5"] / max(v["n"], 1), 2),
            }
            for k, v in by_pos.items()
        },
    }


# ---------------------------------------------------------------------------
# LAYER 2D: ARCHAEOLOGICAL CORRELATIONS (SITE / MOTIF / OBJECT TYPE x SIGN)
# ---------------------------------------------------------------------------
def compute_enrichment(
    df: pd.DataFrame, group_col: str, target_groups: list[str], min_sign_freq: int = 15
) -> list[dict[str, Any]]:
    """Find signs statistically enriched in specific archaeological categories (chi2 + lift)."""
    # Deduplicate by (group_col, clean_reading_seq) to avoid hoarding of duplicate mass-produced tablets
    dedup = df.drop_duplicates(subset=[group_col, "clean_reading_seq"])
    sub = dedup[dedup[group_col].isin(target_groups)]

    group_sign = defaultdict(Counter)
    sign_tot = Counter()
    group_tot = Counter()
    for _, r in sub.iterrows():
        g = r[group_col]
        toks = set(r["clean_reading_seq"].split())  # inscription-level presence
        group_tot[g] += 1
        for t in toks:
            group_sign[g][t] += 1
            sign_tot[t] += 1

    total_docs = sum(group_tot.values())
    findings = []
    for sign, s_cnt in sign_tot.items():
        if s_cnt < min_sign_freq:
            continue
        for g in target_groups:
            g_cnt = group_tot[g]
            if g_cnt < 15:
                continue
            a = group_sign[g][sign]
            if a < 5:
                continue
            b = g_cnt - a
            c = s_cnt - a
            d = total_docs - g_cnt - c
            if min(a, b, c, d) < 0:
                continue
            chi2, pval, _, _ = chi2_contingency([[a, b], [c, d]], correction=True)
            p_in = a / g_cnt
            p_out = c / max(total_docs - g_cnt, 1)
            lift = p_in / max(p_out, 1e-6)
            if pval < 0.001 and lift >= 2.0:
                findings.append(
                    {
                        "category": g,
                        "sign_id": sign,
                        "in_group_docs": int(a),
                        "group_total_docs": int(g_cnt),
                        "in_group_pct": round(100.0 * p_in, 2),
                        "outside_pct": round(100.0 * p_out, 2),
                        "lift": round(lift, 2),
                        "chi2": round(float(chi2), 2),
                        "p_value": float(f"{pval:.3e}"),
                    }
                )
    findings.sort(key=lambda x: (-x["chi2"]))
    return findings


def run_layer2d_archaeology(strict_df: pd.DataFrame) -> dict[str, Any]:
    # Collapse unicorn subtypes Bull1:* -> Bull1 for cleaner motif signal, keep others explicit
    df = strict_df.copy()
    df["motif_clean"] = df["symbol"].apply(
        lambda s: "Bull1_Unicorn" if s.startswith("Bull1") else (s if s not in {"", "-", "None"} else "NO_MOTIF")
    )
    major_sites = ["Harappa", "Mohenjo-daro", "Dholavira", "Kalibangan", "Lothal", "Chanhu-daro"]
    major_motifs = ["Bull1_Unicorn", "Bult", "Gaur", "Elep", "Zebu", "Gavi", "Mult", "NO_MOTIF"]
    major_types = ["SEAL:S", "SEAL:R", "SEAL:C", "TAB:B", "TAB:I", "TAB:C", "POT:T:g", "TAG"]

    return {
        "site_enriched_signs": compute_enrichment(df, "site", major_sites, min_sign_freq=15)[:25],
        "motif_enriched_signs": compute_enrichment(df, "motif_clean", major_motifs, min_sign_freq=15)[:25],
        "object_type_enriched_signs": compute_enrichment(df, "object_type", major_types, min_sign_freq=15)[:25],
    }


# ---------------------------------------------------------------------------
# LAYER 2->4 BRIDGE: FOREIGN-SEAL SURPRISAL / ROSETTA CANDIDATE RANKER
# ---------------------------------------------------------------------------
def run_foreign_surprisal_ranker(lines_df: pd.DataFrame) -> dict[str, Any]:
    """
    Train smoothed bigram model strictly on deduplicated Indus Heartland strict sequences.
    Compare surprisal (bits/transition) of:
      1) Held-out Heartland sequences (5-fold CV baseline)
      2) Foreign-region sequences (Mesopotamia, Persian Gulf, Iranian Plateau, Central Asia)
    High z-score foreign seals diverge sharply from Harappan grammar -> prime candidates
    for phonetic spelling of foreign (Akkadian/Sumerian/Dilmun/Elamite) names!
    """
    heartland_strict = lines_df[
        (lines_df["is_strict_analytic"] == 1)
        & (lines_df["is_foreign"] == 0)
        & (lines_df["direction"].isin(["R/L", "L/R", "BUS"]))
    ]
    heartland_unique = [s.split() for s in heartland_strict["clean_reading_seq"].unique()]

    def fit_bigram(seqs: list[list[str]]) -> tuple[Counter, defaultdict[str, Counter], Counter, int]:
        uni = Counter()
        bi = defaultdict(Counter)
        prev_tot = Counter()
        for seq in seqs:
            padded = [BOS] + seq + [EOS]
            for t in seq:
                uni[t] += 1
            for a, b in zip(padded[:-1], padded[1:]):
                bi[a][b] += 1
                prev_tot[a] += 1
        return uni, bi, prev_tot, sum(uni.values())

    def seq_surprisal(seq: list[str], uni: Counter, bi: defaultdict, prev_tot: Counter, n_uni: int) -> float:
        V = len(uni) + 1
        padded = [BOS] + seq + [EOS]
        total_bits = 0.0
        for a, b in zip(padded[:-1], padded[1:]):
            p_u = (uni[b] + 0.5) / (n_uni + 0.5 * V)
            # Interpolated smoothed bigram probability
            pt = prev_tot[a]
            p_ab = (bi[a][b] + 2.0 * p_u) / (pt + 2.0) if pt > 0 else p_u
            total_bits -= math.log2(max(p_ab, 1e-12))
        return total_bits / (len(seq) + 1)

    # 5-fold CV to get unbiased Heartland surprisal baseline distribution
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(heartland_unique))
    folds = np.array_split(idx, 5)
    hl_scores = []
    for fi in range(5):
        te = set(folds[fi].tolist())
        tr = [heartland_unique[i] for i in range(len(heartland_unique)) if i not in te]
        uni, bi, prev_tot, n_uni = fit_bigram(tr)
        for i in folds[fi]:
            hl_scores.append(seq_surprisal(heartland_unique[i], uni, bi, prev_tot, n_uni))

    hl_mean = float(np.mean(hl_scores))
    hl_std = float(np.std(hl_scores) + 1e-12)

    # Fit full Heartland model to score every complete/usable foreign inscription (>= 2 clean signs)
    uni_all, bi_all, prev_all, n_uni_all = fit_bigram(heartland_unique)
    foreign_df = lines_df[
        (lines_df["is_foreign"] == 1) & (lines_df["token_count_clean"] >= 2)
    ].copy()

    ranked = []
    for _, r in foreign_df.iterrows():
        seq = r["clean_reading_seq"].split()
        surp = seq_surprisal(seq, uni_all, bi_all, prev_all, n_uni_all)
        z = (surp - hl_mean) / hl_std
        # Count how many bigrams in this foreign seal have NEVER been seen in the entire Indus Heartland!
        padded = [BOS] + seq + [EOS]
        unseen_bi = sum(1 for a, b in zip(padded[:-1], padded[1:]) if bi_all[a][b] == 0)
        unseen_signs = [s for s in seq if uni_all[s] == 0]
        ranked.append(
            {
                "inscription_id": r["inscription_id"],
                "region": r["region"],
                "site": r["site"],
                "object_type": r["object_type"],
                "symbol": r["symbol"],
                "complete": r["complete"],
                "direction": r["direction"],
                "reading_seq": " -> ".join(seq),
                "len": len(seq),
                "surprisal_bits": round(surp, 3),
                "z_score_vs_heartland": round(z, 2),
                "unseen_bigrams": f"{unseen_bi}/{len(seq)+1}",
                "unseen_signs": unseen_signs,
                "classification": (
                    "ROSETTA_ANOMALY_CANDIDATE"
                    if z >= 1.5
                    else ("NATIVE_HARAPPAN_ABROAD" if z <= 0.3 else "MIXED_HYBRID")
                ),
            }
        )

    ranked.sort(key=lambda x: -x["z_score_vs_heartland"])
    foreign_mean = float(np.mean([x["surprisal_bits"] for x in ranked])) if ranked else 0.0
    from scipy.stats import mannwhitneyu
    u_stat, p_val = mannwhitneyu([x["surprisal_bits"] for x in ranked], hl_scores, alternative="greater")

    return {
        "heartland_cv_mean_surprisal": round(hl_mean, 3),
        "heartland_cv_std_surprisal": round(hl_std, 3),
        "foreign_mean_surprisal": round(foreign_mean, 3),
        "mann_whitney_foreign_gt_heartland_p": float(f"{p_val:.3e}"),
        "foreign_count_evaluated": len(ranked),
        "top_rosetta_anomalies": ranked[:15],
        "most_native_harappan_abroad": list(reversed(ranked[-8:])),
    }


def main() -> None:
    _, lines_df, glyphs_df, sim_df = load_data()
    strict_df = get_oriented_strict_lines(lines_df)

    results = {
        "layer1_graphemes": run_layer1_graphemes(strict_df, glyphs_df, sim_df),
        "layer2a_entropy": run_layer2a_entropy(strict_df, n_perm=200),
        "layer2b_grammar": run_layer2b_grammar(strict_df),
        "layer2c_masked_prediction": run_layer2c_masked_prediction(strict_df, k_folds=5),
        "layer2d_archaeology": run_layer2d_archaeology(strict_df),
        "layer2_to_4_foreign_surprisal": run_foreign_surprisal_ranker(lines_df),
    }

    out_path = REPORTS_DIR / "layer1_2_findings.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print crisp executive console summary
    print("=== LAYER 1: GRAPHEMES ===")
    l1 = results["layer1_graphemes"]
    print(f"PUA compounds: {l1['pua_compound_total']} | Vocab(>=8): {l1['distributional_vocab_size_ge8']} | Allograph candidates: {l1['allograph_candidates_count']}")
    print("Top 5 allograph candidates:", l1["top_allograph_candidates"][:5])

    print("\n=== LAYER 2A: ENTROPY & NULL-MODEL ===")
    print(json.dumps(results["layer2a_entropy"], indent=2))

    print("\n=== LAYER 2B: SLOT GRAMMAR ===")
    print("Slot roles:", results["layer2b_grammar"]["slot_role_summary_ge20"])
    print("Top 5 terminal:", results["layer2b_grammar"]["top_terminal_signs"][:5])
    print("Top 5 initial:", results["layer2b_grammar"]["top_initial_signs"][:5])
    print("Top 5 PMI bound bigrams:", results["layer2b_grammar"]["top_pmi_bound_bigrams"][:5])

    print("\n=== LAYER 2C: MASKED PREDICTION (5-FOLD ZERO-LEAKAGE CV) ===")
    print(json.dumps(results["layer2c_masked_prediction"], indent=2))

    print("\n=== LAYER 2D: ARCHAEOLOGICAL ENRICHMENT (TOP 5 EACH) ===")
    print("Site-enriched:", results["layer2d_archaeology"]["site_enriched_signs"][:5])
    print("Motif-enriched:", results["layer2d_archaeology"]["motif_enriched_signs"][:5])
    print("Object-enriched:", results["layer2d_archaeology"]["object_type_enriched_signs"][:5])

    print("\n=== LAYER 2->4: FOREIGN SEAL SURPRISAL RANKER ===")
    fs = results["layer2_to_4_foreign_surprisal"]
    print(f"Heartland CV mean: {fs['heartland_cv_mean_surprisal']} ± {fs['heartland_cv_std_surprisal']} | Foreign mean: {fs['foreign_mean_surprisal']} | p={fs['mann_whitney_foreign_gt_heartland_p']}")
    print("TOP 8 ROSETTA ANOMALIES:")
    for item in fs["top_rosetta_anomalies"][:8]:
        print(f"  z={item['z_score_vs_heartland']:+5.2f} | {item['site']:18s} ({item['region']:15s}) | {item['symbol']:8s} | {item['reading_seq']} | unseen_bi={item['unseen_bigrams']}")
    print("MOST NATIVE HARAPPAN ABROAD:")
    for item in fs["most_native_harappan_abroad"][:5]:
        print(f"  z={item['z_score_vs_heartland']:+5.2f} | {item['site']:18s} ({item['region']:15s}) | {item['symbol']:8s} | {item['reading_seq']} | unseen_bi={item['unseen_bigrams']}")


if __name__ == "__main__":
    main()
