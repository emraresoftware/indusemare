#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 11B: Visual-Glyph + Sub-Grapheme + Object-Aware Neural MLP
Damaged-Inscription Restoration Engine & Open Restoration Catalog.

Strict Anti-Leakage Discipline:
  - Evaluates head-to-head against the exact Layer 7 N-gram baseline on the exact
    same 44 edge + 10 interior real damaged inscriptions under STRICT Leave-One-City-Out
    + Parallel-Sequence-Out (STRICT_LOCO) cross-validation.
  - All distributional embeddings, object priors, and MLP weights are fitted STRICTLY
    inside each held-out-city training fold (zero test-city leakage).
  - Ensemble mixing weights are fixed a priori (0.50 N-gram + 0.25 Object/Allograph/Skip
    + 0.25 Visual-Context Neural MLP) rather than tuned on the 54 evaluation cases.
  - Exports an Open Restoration Catalog (JSON + CSV) for every single-gap damaged
    inscription across the full 5,679-inscription corpus.
"""

from __future__ import annotations

import json
import math
import sqlite3
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

BOS = "<BOS>"
EOS = "<EOS>"
RNG_SEED = 20261120

FRAMED_SIGNS = {"226", "232", "234", "236", "241", "592", "804", "842"}
RIM_TICKS_740 = {"741", "742", "743", "744", "745", "746", "747"}


def obj_family(t: Any) -> str:
    if not isinstance(t, str) or not t.strip():
        return "OTHER"
    t = t.strip()
    if t.startswith("SEAL:S"): return "SEAL_S"
    if t.startswith("SEAL:R"): return "SEAL_R"
    if t.startswith("SEAL:C"): return "SEAL_C"
    if t.startswith("SEAL"): return "SEAL_O"
    if t.startswith("TAB:C"): return "TAB_C"
    if t.startswith("TAB:I"): return "TAB_I"
    if t.startswith("TAB:B"): return "TAB_B"
    if t.startswith("TAG"): return "TAG"
    if t.startswith("POT"): return "POT"
    return "OTHER"


OBJ_FAMILIES = ["SEAL_S", "SEAL_R", "SEAL_C", "SEAL_O", "TAB_C", "TAB_I", "TAB_B", "TAG", "POT", "OTHER"]
OBJ_IDX = {k: i for i, k in enumerate(OBJ_FAMILIES)}
PT_LIST = ["init", "medial", "final"]
PT_IDX = {k: i for i, k in enumerate(PT_LIST)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return [round(100.0 * max(0.0, center - half), 1), round(100.0 * min(1.0, center + half), 1)]


def build_visual_glyph_embeddings(glyphs_df: pd.DataFrame, dim: int = 20) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[str, float]]]]:
    """Render 32x32 Indus TTF bitmaps -> 20-dim visual SVD embeddings + top visual neighbors."""
    font = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 24)
    sids = []
    vecs = []
    for _, r in glyphs_df.iterrows():
        sid = str(r["sign_id"]).strip().zfill(3)
        cps = [int(h, 16) for h in str(r["codepoints"]).split() if h]
        if not cps:
            continue
        ch = "".join(chr(c) for c in cps)
        img = Image.new("L", (32, 32), 255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((32 - w) // 2 - bbox[0], (32 - h) // 2 - bbox[1]), ch, fill=0, font=font)
        arr = (255.0 - np.asarray(img, dtype=np.float32)) / 255.0
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            arr /= norm
        sids.append(sid)
        vecs.append(arr.ravel())

    X = np.vstack(vecs)
    svd = TruncatedSVD(n_components=dim, random_state=RNG_SEED)
    Z = svd.fit_transform(X)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = np.divide(Z, np.maximum(norms, 1e-8))

    emb: dict[str, np.ndarray] = {sid: Z_norm[i].astype(np.float32) for i, sid in enumerate(sids)}
    # Special tokens get deterministic orthogonal-ish unit vectors
    rng = np.random.default_rng(77)
    for sp in (BOS, EOS, "000"):
        v = rng.normal(size=dim).astype(np.float32)
        emb[sp] = v / max(float(np.linalg.norm(v)), 1e-8)

    # Top visual neighbors (cos >= 0.72) for allograph kernel smoothing
    sim_mat = Z_norm @ Z_norm.T
    vis_neighbors: dict[str, list[tuple[str, float]]] = {}
    for i, sid in enumerate(sids):
        row = sim_mat[i]
        top_idx = np.argsort(-row)[1:6]
        nbs = [(sids[j], float(row[j])) for j in top_idx if row[j] >= 0.72]
        if nbs:
            vis_neighbors[sid] = nbs
    return emb, vis_neighbors


def build_distributional_embeddings(train_seqs: list[list[str]], dim: int = 20) -> dict[str, np.ndarray]:
    """Fit 20-dim PPMI distributional SVD embeddings strictly on the training fold."""
    uni = Counter(t for s in train_seqs for t in s)
    vocab = [s for s, c in uni.most_common() if c >= 2]
    if len(vocab) < dim + 2:
        return {}
    v_idx = {s: i for i, s in enumerate(vocab)}
    V = len(vocab)
    # Co-occurrence with left/right immediate neighbors (context cols = 2*V + 2 special)
    ctx_list = [BOS, EOS] + vocab
    c_idx = {s: i for i, s in enumerate(ctx_list)}
    C = len(ctx_list)
    mat = np.zeros((V, 2 * C), dtype=np.float32)
    for seq in train_seqs:
        padded = [BOS] + seq + [EOS]
        for i, tok in enumerate(seq):
            if tok not in v_idx:
                continue
            r = v_idx[tok]
            l_tok, r_tok = padded[i], padded[i + 2]
            if l_tok in c_idx:
                mat[r, c_idx[l_tok]] += 1.0
            if r_tok in c_idx:
                mat[r, C + c_idx[r_tok]] += 1.0
    tot = float(mat.sum()) + 1e-8
    row_s = mat.sum(axis=1, keepdims=True) + 1e-8
    col_s = mat.sum(axis=0, keepdims=True) + 1e-8
    pmi = np.log(np.maximum(mat * tot / (row_s * col_s), 1e-8))
    ppmi = np.maximum(pmi, 0.0)
    svd = TruncatedSVD(n_components=min(dim, V - 1), random_state=RNG_SEED)
    Z = svd.fit_transform(ppmi)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = np.divide(Z, np.maximum(norms, 1e-8))
    out = {s: Z[i].astype(np.float32) for i, s in enumerate(vocab)}
    # Pad if dim > Z.shape[1]
    if Z.shape[1] < dim:
        for s in out:
            out[s] = np.pad(out[s], (0, dim - Z.shape[1]))
    return out


def build_exact_l7_ngram_model(train_rows: list[tuple[list[str], str]]) -> dict[str, Any]:
    """Exact Layer 7 bidirectional n-gram + object-family & skip-bigram extensions."""
    uni = Counter()
    pos_uni = defaultdict(Counter)
    obj_pos_uni = defaultdict(Counter)
    left_bi = defaultdict(Counter)
    right_bi = defaultdict(Counter)
    surround = defaultdict(Counter)
    left_tri = defaultdict(Counter)
    right_tri = defaultdict(Counter)
    skip_l2_r1 = defaultdict(Counter)
    skip_l1_r2 = defaultdict(Counter)

    for seq, ofam in train_rows:
        padded = [BOS, BOS] + seq + [EOS, EOS]
        n = len(seq)
        for i, tok in enumerate(seq):
            uni[tok] += 1
            pt = "init" if i == 0 else ("final" if i == n - 1 else "medial")
            pos_uni[pt][tok] += 1
            obj_pos_uni[(ofam, pt)][tok] += 1
            p2, p1, n1, n2 = padded[i], padded[i + 1], padded[i + 3], padded[i + 4]
            left_bi[p1][tok] += 1
            right_bi[n1][tok] += 1
            surround[(p1, n1)][tok] += 1
            left_tri[(p2, p1)][tok] += 1
            right_tri[(n1, n2)][tok] += 1
            skip_l2_r1[(p2, n1)][tok] += 1
            skip_l1_r2[(p1, n2)][tok] += 1

    return {
        "uni": uni,
        "pos_uni": pos_uni,
        "obj_pos_uni": obj_pos_uni,
        "left_bi": left_bi,
        "right_bi": right_bi,
        "surround": surround,
        "left_tri": left_tri,
        "right_tri": right_tri,
        "skip_l2_r1": skip_l2_r1,
        "skip_l1_r2": skip_l1_r2,
        "vocab": [s for s, _ in uni.most_common()],
        "n_uni": sum(uni.values()),
    }


def predict_l7_ngram(m: dict[str, Any], p2: str, p1: str, n1: str, n2: str, pt: str) -> dict[str, float]:
    """Exact Layer 7 interpolated bidirectional predictor."""
    uni = m["uni"]
    n_uni = max(1, m["n_uni"])
    pu = m["pos_uni"][pt]
    n_pu = sum(pu.values()) or 1
    lb = m["left_bi"].get(p1, {})
    n_lb = sum(lb.values())
    rb = m["right_bi"].get(n1, {})
    n_rb = sum(rb.values())
    sur = m["surround"].get((p1, n1), {})
    n_sur = sum(sur.values())
    lt = m["left_tri"].get((p2, p1), {})
    n_lt = sum(lt.values())
    rt = m["right_tri"].get((n1, n2), {})
    n_rt = sum(rt.values())

    probs = {}
    for w in m["vocab"]:
        p = 0.05 * (uni[w] / n_uni) + 0.10 * (pu.get(w, 0) / n_pu)
        if n_lb: p += 0.18 * (lb.get(w, 0) / n_lb)
        if n_rb: p += 0.18 * (rb.get(w, 0) / n_rb)
        if n_sur: p += 0.25 * (sur.get(w, 0) / n_sur)
        if n_lt: p += 0.12 * (lt.get(w, 0) / n_lt)
        if n_rt: p += 0.12 * (rt.get(w, 0) / n_rt)
        probs[w] = p
    tot = sum(probs.values()) or 1.0
    return {k: v / tot for k, v in probs.items()}


def predict_obj_allograph_skip(
    m: dict[str, Any],
    vis_nbs: dict[str, list[tuple[str, float]]],
    p2: str, p1: str, n1: str, n2: str, pt: str, ofam: str,
) -> dict[str, float]:
    """Object-family slot prior + skip-bigrams + visual-allograph neighborhood smoothing."""
    opu = m["obj_pos_uni"].get((ofam, pt), m["pos_uni"][pt])
    n_opu = sum(opu.values()) or 1
    pu = m["pos_uni"][pt]
    n_pu = sum(pu.values()) or 1
    sk1 = m["skip_l2_r1"].get((p2, n1), {})
    n_sk1 = sum(sk1.values())
    sk2 = m["skip_l1_r2"].get((p1, n2), {})
    n_sk2 = sum(sk2.values())

    # Visual-allograph backoff on left/right neighbors when exact neighbor is sparse
    allo_lb: Counter[str] = Counter()
    for nb, sim in vis_nbs.get(p1, []):
        for w, cnt in m["left_bi"].get(nb, {}).items():
            allo_lb[w] += cnt * sim
    n_alb = sum(allo_lb.values())

    allo_rb: Counter[str] = Counter()
    for nb, sim in vis_nbs.get(n1, []):
        for w, cnt in m["right_bi"].get(nb, {}).items():
            allo_rb[w] += cnt * sim
    n_arb = sum(allo_rb.values())

    # Sub-grapheme structural prior boost (from Layer 5B/9):
    # If next token is '400' or EOS on SEAL_S, ': X :' framed signs get structural compatibility;
    # if next token is NOT EOS (medial), 741-747 rim-ticks fit medial better than bare 740.
    probs = {}
    for w in m["vocab"]:
        p = 0.24 * (opu.get(w, 0) / n_opu) + 0.10 * (pu.get(w, 0) / n_pu)
        if n_sk1: p += 0.22 * (sk1.get(w, 0) / n_sk1)
        if n_sk2: p += 0.22 * (sk2.get(w, 0) / n_sk2)
        if n_alb: p += 0.11 * (allo_lb.get(w, 0) / n_alb)
        if n_arb: p += 0.11 * (allo_rb.get(w, 0) / n_arb)
        # Light sub-grapheme structural consistency (zero leakage)
        if pt == "final" and w in RIM_TICKS_740:
            p *= 0.25
        elif pt == "medial" and w in FRAMED_SIGNS and n1 != "400":
            p *= 0.35
        probs[w] = p
    tot = sum(probs.values()) or 1.0
    return {k: v / tot for k, v in probs.items()}


def featurize_slot(
    p2: str, p1: str, n1: str, n2: str, pt: str, ofam: str, seq_len: int, pos_idx: int,
    vis_emb: dict[str, np.ndarray], dist_emb: dict[str, np.ndarray], vis_dim: int = 20, dist_dim: int = 20,
) -> np.ndarray:
    z_v = np.zeros(vis_dim, dtype=np.float32)
    z_d = np.zeros(dist_dim, dtype=np.float32)
    oh_pt = np.zeros(len(PT_LIST), dtype=np.float32)
    oh_pt[PT_IDX[pt]] = 1.0
    oh_obj = np.zeros(len(OBJ_FAMILIES), dtype=np.float32)
    oh_obj[OBJ_IDX.get(ofam, OBJ_IDX["OTHER"])] = 1.0
    rel_pos = pos_idx / max(1, seq_len - 1)
    # Sub-grapheme context flags (does left/right neighbor contain 400, 740, numeral, framed?)
    ctx_flags = np.array([
        min(seq_len, 10) / 10.0,
        rel_pos,
        1.0 if p1 in FRAMED_SIGNS else 0.0,
        1.0 if n1 == "400" else 0.0,
        1.0 if p1 in RIM_TICKS_740 else 0.0,
        1.0 if n1 == "740" else 0.0,
        1.0 if p1 in {"031", "032", "033", "034", "035"} else 0.0,
        1.0 if p1 in {"817", "820", "861", "920"} else 0.0,
    ], dtype=np.float32)
    return np.concatenate([
        vis_emb.get(p2, z_v),
        vis_emb.get(p1, z_v),
        vis_emb.get(n1, z_v),
        vis_emb.get(n2, z_v),
        dist_emb.get(p1, z_d),
        dist_emb.get(n1, z_d),
        oh_pt,
        oh_obj,
        ctx_flags,
    ])


def train_visual_context_mlp(
    train_rows: list[tuple[list[str], str]],
    vis_emb: dict[str, np.ndarray],
    dist_emb: dict[str, np.ndarray],
    min_class_freq: int = 5,
) -> tuple[MLPClassifier, list[str]]:
    """Train visual+distributional+subgrapheme MLP over frequent classes (freq >= 5 = 95.7% tokens)."""
    uni = Counter(t for seq, _ in train_rows for t in seq)
    classes = sorted([s for s, c in uni.items() if c >= min_class_freq])
    c_to_i = {c: i for i, c in enumerate(classes)}

    X_list = []
    y_list = []
    for seq, ofam in train_rows:
        n = len(seq)
        padded = [BOS, BOS] + seq + [EOS, EOS]
        for i, tok in enumerate(seq):
            if tok not in c_to_i:
                continue
            pt = "init" if i == 0 else ("final" if i == n - 1 else "medial")
            X_list.append(featurize_slot(padded[i], padded[i + 1], padded[i + 3], padded[i + 4], pt, ofam, n, i, vis_emb, dist_emb))
            y_list.append(c_to_i[tok])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-2,
        batch_size=128,
        learning_rate_init=1.5e-3,
        max_iter=35,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RNG_SEED,
    )
    mlp.fit(X, y)
    return mlp, classes


def predict_mlp_probs(
    mlp: MLPClassifier, classes: list[str], feat: np.ndarray, full_vocab: list[str]
) -> dict[str, float]:
    p_vec = mlp.predict_proba(feat.reshape(1, -1))[0]
    out = {w: 1e-9 for w in full_vocab}
    for cls_idx, p in zip(mlp.classes_, p_vec):
        out[classes[int(cls_idx)]] = float(p)
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}


def main() -> None:
    lines = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    glyphs_df = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")
    lines["is_strict_analytic"] = lines["is_strict_analytic"].astype(int)
    lines["has_damage_token"] = lines["has_damage_token"].astype(int)
    lines["obj_fam"] = lines["object_type"].apply(obj_family)

    print("Building 32x32 visual glyph SVD embeddings...")
    vis_emb, vis_nbs = build_visual_glyph_embeddings(glyphs_df, dim=20)
    print(f"Visual embeddings ready: {len(vis_emb)} glyphs, {len(vis_nbs)} with close visual neighbors.")

    # Exact Layer 7 strict analytic training pool & damaged ground-truth cases
    strict = lines[
        (lines["is_strict_analytic"] == 1)
        & (lines["direction"].isin(["R/L", "L/R", "BUS"]))
        & (lines["clean_reading_seq"].str.split().apply(len) >= 2)
    ].copy()
    dmg = lines[
        (lines["has_damage_token"] == 1)
        & (lines["direction"].isin(["R/L", "L/R", "BUS"]))
    ].copy()

    # Build wildcard ground-truth index (exact same logic as Layer 7!)
    wildcard_map: dict[tuple[int, int, tuple[str, ...]], list[tuple[str, str, str]]] = defaultdict(list)
    for _, r in strict.drop_duplicates(subset=["site", "clean_reading_seq"]).iterrows():
        toks = r["clean_reading_seq"].split()
        L = len(toks)
        for i in range(L):
            key = (L, i, tuple(toks[:i] + ["?"] + toks[i + 1 :]))
            wildcard_map[key].append((r["site"], toks[i], r["clean_reading_seq"]))

    cases = []
    for _, r in dmg.iterrows():
        toks = r["reading_seq"].split()
        if toks.count("000") != 1 or "999" in toks:
            continue
        pos = toks.index("000")
        L = len(toks)
        key = (L, pos, tuple(toks[:pos] + ["?"] + toks[pos + 1 :]))
        matches = wildcard_map.get(key, [])
        if not matches:
            continue
        site = r["site"]
        cases.append({
            "id": r["inscription_id"],
            "site": site,
            "ofam": r["obj_fam"],
            "toks": toks,
            "pos": pos,
            "is_interior": 0 < pos < L - 1,
            "pt": "init" if pos == 0 else ("final" if pos == L - 1 else "medial"),
            "all_fills": sorted({m[1] for m in matches}),
            "cross_fills": sorted({m[1] for m in matches if m[0] != site}),
            "parallel_seqs": {m[2] for m in matches},
        })

    print(f"LOCO real damaged benchmark cases: total={len(cases)} (edge={sum(not c['is_interior'] for c in cases)}, interior={sum(c['is_interior'] for c in cases)})")

    # Pre-train ONE leak-free Visual+Distributional MLP per held-out city (strictly excluding held-out city AND ALL ground-truth parallel sequences!)
    all_parallel_seqs = set().union(*(c["parallel_seqs"] for c in cases))
    strict_dedup = strict.drop_duplicates(subset=["site", "clean_reading_seq", "obj_fam"])
    held_out_cities = sorted({c["site"] for c in cases})
    print(f"Training strictly leak-free LOCO models for {len(held_out_cities)} held-out cities: {held_out_cities}")

    city_models: dict[str, tuple[dict[str, np.ndarray], MLPClassifier, list[str]]] = {}
    for city in held_out_cities:
        tr_df = strict_dedup[(strict_dedup["site"] != city) & (~strict_dedup["clean_reading_seq"].isin(all_parallel_seqs))]
        tr_rows = [(s.split(), of) for s, of in zip(tr_df["clean_reading_seq"], tr_df["obj_fam"])]
        dist_e = build_distributional_embeddings([r[0] for r in tr_rows], dim=20)
        mlp, classes = train_visual_context_mlp(tr_rows, vis_emb, dist_e, min_class_freq=5)
        city_models[city] = (dist_e, mlp, classes)

    # Evaluate ALL 4 models (L7 N-gram, Object+Allograph+Skip, Neural MLP alone, and Fixed-Weight Hybrid Ensemble)
    # on each LOCO case (removing per-case parallel_seqs for exact N-gram match with L7!)
    case_preds: list[dict[str, Any]] = []
    for c in cases:
        tr_df = strict_dedup[(strict_dedup["site"] != c["site"]) & (~strict_dedup["clean_reading_seq"].isin(c["parallel_seqs"]))]
        # Deduplicate sequences as in L7 for N-gram, keeping object family for obj_pos_uni
        tr_rows_ng = [(s.split(), of) for s, of in tr_df.drop_duplicates(subset=["clean_reading_seq"])[["clean_reading_seq", "obj_fam"]].itertuples(index=False)]
        tr_rows_all = [(s.split(), of) for s, of in tr_df[["clean_reading_seq", "obj_fam"]].itertuples(index=False)]

        m_ng = build_exact_l7_ngram_model(tr_rows_ng)
        m_oa = build_exact_l7_ngram_model(tr_rows_all)
        dist_e, mlp, classes = city_models[c["site"]]

        toks = c["toks"]
        pos = c["pos"]
        padded = [BOS, BOS] + toks + [EOS, EOS]
        p2, p1, n1, n2 = padded[pos], padded[pos + 1], padded[pos + 3], padded[pos + 4]

        pr_ng = predict_l7_ngram(m_ng, p2, p1, n1, n2, c["pt"])
        pr_oa = predict_obj_allograph_skip(m_oa, vis_nbs, p2, p1, n1, n2, c["pt"], c["ofam"])
        feat = featurize_slot(p2, p1, n1, n2, c["pt"], c["ofam"], len(toks), pos, vis_emb, dist_e)
        pr_mlp = predict_mlp_probs(mlp, classes, feat, m_ng["vocab"])

        # Fixed a-priori ensemble (0.50 N-gram + 0.25 Object/Allograph/Skip + 0.25 Visual-Context MLP)
        pr_hyb = {
            w: 0.50 * pr_ng.get(w, 0.0) + 0.25 * pr_oa.get(w, 0.0) + 0.25 * pr_mlp.get(w, 0.0)
            for w in m_ng["vocab"]
        }
        # Rank helper
        def rank_top(d: dict[str, float]) -> list[str]:
            return [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1]) if k != "<UNK>"]

        case_preds.append({
            "case": c,
            "pos_top1": m_ng["pos_uni"][c["pt"]].most_common(1)[0][0],
            "ngram": rank_top(pr_ng),
            "obj_allograph": rank_top(pr_oa),
            "neural_mlp": rank_top(pr_mlp),
            "hybrid_ensemble": rank_top(pr_hyb),
        })

    def score_subset(pred_list: list[dict[str, Any]], fill_key: str) -> dict[str, Any]:
        n = len(pred_list)
        out: dict[str, Any] = {"n": n}
        for mname in ("ngram", "obj_allograph", "neural_mlp", "hybrid_ensemble"):
            t1 = t3 = t5 = 0
            for rec in pred_list:
                fills = set(rec["case"][fill_key])
                rk = rec[mname]
                if rk and rk[0] in fills: t1 += 1
                if any(x in fills for x in rk[:3]): t3 += 1
                if any(x in fills for x in rk[:5]): t5 += 1
            out[mname] = {
                "top1_n": t1, "top1_pct": round(100.0 * t1 / max(1, n), 1), "top1_ci95": wilson_ci(t1, n),
                "top3_n": t3, "top3_pct": round(100.0 * t3 / max(1, n), 1), "top3_ci95": wilson_ci(t3, n),
                "top5_n": t5, "top5_pct": round(100.0 * t5 / max(1, n), 1),
            }
        pb1 = sum(1 for rec in pred_list if rec["pos_top1"] in set(rec["case"][fill_key]))
        out["pos_baseline_top1_pct"] = round(100.0 * pb1 / max(1, n), 1)
        return out

    edge_all = [r for r in case_preds if not r["case"]["is_interior"]]
    edge_single = [r for r in edge_all if len(r["case"]["all_fills"]) == 1]
    edge_cross = [r for r in edge_all if len(r["case"]["cross_fills"]) >= 1]
    edge_cross_single = [r for r in edge_all if len(r["case"]["cross_fills"]) == 1]
    inter_all = [r for r in case_preds if r["case"]["is_interior"]]

    loco_results = {
        "edge_all_STRICT_LOCO": score_subset(edge_all, "all_fills"),
        "edge_single_unambiguous_STRICT_LOCO": score_subset(edge_single, "all_fills"),
        "edge_cross_site_STRICT_LOCO": score_subset(edge_cross, "cross_fills"),
        "edge_cross_site_single_STRICT_LOCO": score_subset(edge_cross_single, "cross_fills"),
        "interior_all_STRICT_LOCO": score_subset(inter_all, "all_fills"),
    }

    # --- PART 2: GENERATE OPEN DAMAGED-INSCRIPTION RESTORATION CATALOG (ALL SINGLE-GAP DAMAGED LINES!) ---
    print("Training full-corpus model for Open Damaged-Inscription Restoration Catalog...")
    full_rows_ng = [(s.split(), of) for s, of in strict_dedup.drop_duplicates(subset=["clean_reading_seq"])[["clean_reading_seq", "obj_fam"]].itertuples(index=False)]
    full_rows_all = [(s.split(), of) for s, of in strict_dedup[["clean_reading_seq", "obj_fam"]].itertuples(index=False)]
    full_m_ng = build_exact_l7_ngram_model(full_rows_ng)
    full_m_oa = build_exact_l7_ngram_model(full_rows_all)
    full_dist = build_distributional_embeddings([r[0] for r in full_rows_all], dim=20)
    full_mlp, full_classes = train_visual_context_mlp(full_rows_all, vis_emb, full_dist, min_class_freq=5)

    catalog = []
    for _, r in dmg.iterrows():
        toks = r["reading_seq"].split()
        if toks.count("000") != 1 or "999" in toks or len(toks) < 2:
            continue
        pos = toks.index("000")
        L = len(toks)
        pt = "init" if pos == 0 else ("final" if pos == L - 1 else "medial")
        padded = [BOS, BOS] + toks + [EOS, EOS]
        p2, p1, n1, n2 = padded[pos], padded[pos + 1], padded[pos + 3], padded[pos + 4]
        ofam = r["obj_fam"]

        pr_ng = predict_l7_ngram(full_m_ng, p2, p1, n1, n2, pt)
        pr_oa = predict_obj_allograph_skip(full_m_oa, vis_nbs, p2, p1, n1, n2, pt, ofam)
        feat = featurize_slot(p2, p1, n1, n2, pt, ofam, L, pos, vis_emb, full_dist)
        pr_mlp = predict_mlp_probs(full_mlp, full_classes, feat, full_m_ng["vocab"])

        pr_hyb = {
            w: 0.50 * pr_ng.get(w, 0.0) + 0.25 * pr_oa.get(w, 0.0) + 0.25 * pr_mlp.get(w, 0.0)
            for w in full_m_ng["vocab"]
        }
        tot = sum(pr_hyb.values()) or 1.0
        pr_hyb = {k: v / tot for k, v in pr_hyb.items()}
        top5 = sorted(pr_hyb.items(), key=lambda kv: -kv[1])[:5]
        # Normalized entropy confidence (0..1)
        ent = -sum(p * math.log2(p) for p in pr_hyb.values() if p > 1e-12)
        max_ent = math.log2(len(pr_hyb))
        conf = round(max(0.0, 1.0 - ent / max_ent), 3)

        # Check if exact intact parallel exists in corpus (marked transparently!)
        key = (L, pos, tuple(toks[:pos] + ["?"] + toks[pos + 1 :]))
        attested_parallels = sorted({m[1] for m in wildcard_map.get(key, [])})

        catalog.append({
            "inscription_id": r["inscription_id"],
            "cisi": r["cisi"],
            "site": r["site"],
            "region": r["region"],
            "object_type": r["object_type"],
            "symbol": r["symbol"],
            "damaged_reading_seq": " ".join(toks),
            "missing_index": pos,
            "slot_type": pt,
            "confidence_score": conf,
            "top1_sign": top5[0][0],
            "top1_prob": round(top5[0][1], 4),
            "top5_candidates": [{"sign": k, "prob": round(v, 4)} for k, v in top5],
            "attested_intact_parallel_fills": attested_parallels,
            "is_novel_unattested_restoration": len(attested_parallels) == 0,
        })

    catalog.sort(key=lambda x: (-x["top1_prob"], -x["confidence_score"]))
    n_novel = sum(1 for x in catalog if x["is_novel_unattested_restoration"])
    n_high_conf_novel = sum(1 for x in catalog if x["is_novel_unattested_restoration"] and x["top1_prob"] >= 0.40)

    # Save CSV + JSON
    cat_csv_rows = [
        {
            "inscription_id": x["inscription_id"], "cisi": x["cisi"], "site": x["site"], "region": x["region"],
            "object_type": x["object_type"], "damaged_reading_seq": x["damaged_reading_seq"],
            "slot_type": x["slot_type"], "top1_sign": x["top1_sign"], "top1_prob": x["top1_prob"],
            "top5": " | ".join(f"{c['sign']}:{c['prob']:.2f}" for c in x["top5_candidates"]),
            "confidence_score": x["confidence_score"],
            "novel_restoration": x["is_novel_unattested_restoration"],
            "attested_parallels": ",".join(x["attested_intact_parallel_fills"]),
        }
        for x in catalog
    ]
    pd.DataFrame(cat_csv_rows).to_csv(REPORTS_DIR / "damaged_inscriptions_restoration_catalog.csv", index=False)
    (REPORTS_DIR / "damaged_inscriptions_restoration_catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False)
    )

    summary_payload = {
        "layer": "Layer 11B — Visual-Glyph + Sub-Grapheme + Object-Aware Neural MLP Damaged-Inscription Restoration & Open Catalog",
        "model_architecture": {
            "visual_glyph_svd_dim": 20,
            "distributional_ppmi_svd_dim": 20,
            "mlp_hidden_layers": [128, 64],
            "ensemble_weights_fixed_apriori": {"ngram": 0.50, "obj_allograph_skip": 0.25, "visual_context_mlp": 0.25},
        },
        "strict_loco_head_to_head": loco_results,
        "open_restoration_catalog_stats": {
            "total_single_gap_damaged_inscriptions_restored": len(catalog),
            "novel_unattested_restorations": n_novel,
            "high_confidence_novel_restorations_p_ge_040": n_high_conf_novel,
            "sites_covered": len({x["site"] for x in catalog}),
            "top_10_high_conf_novel_examples": [x for x in catalog if x["is_novel_unattested_restoration"]][:10],
        },
    }
    (REPORTS_DIR / "layer11b_neural_restoration.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False)
    )

    print("\n=== LAYER 11B COMPLETE ===")
    for sub_name, d in loco_results.items():
        print(f"\n  {sub_name} (n={d['n']}, pos_base={d['pos_baseline_top1_pct']}%):")
        for m in ("ngram", "obj_allograph", "neural_mlp", "hybrid_ensemble"):
            r = d[m]
            print(f"    {m:16s} | Top-1={r['top1_n']:2d}/{d['n']} ({r['top1_pct']:5.1f}%) | Top-3={r['top3_n']:2d}/{d['n']} ({r['top3_pct']:5.1f}%) | Top-5={r['top5_n']:2d}/{d['n']} ({r['top5_pct']:5.1f}%)")
    print(f"\nOpen Restoration Catalog: {len(catalog)} real damaged inscriptions ({n_novel} novel unattested, {n_high_conf_novel} high-conf p>=0.40, across {len({x['site'] for x in catalog})} sites)")


if __name__ == "__main__":
    main()
