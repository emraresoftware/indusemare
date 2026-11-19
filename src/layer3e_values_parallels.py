#!/usr/bin/env python3
"""
INDUS-DEDUCT — Layer 3E: Daily Life, Work, Food, Land, War/Conflict & Semantic Parallel/Opposite Discovery
===========================================================================================================
Investigates Emre's concrete social/behavioral domains:
  1. YEMEK / BESIN / TAHIL (Food, Grain, Trough offerings, Storage/Shipping tags)
  2. IS / ZANAAT / TICARET (Work, Carriers, Looms/Frames, Plows, Trade sealings TAG/TAG:L, Metal/Kiln)
  3. TOPRAK / ARAZI (Fields 615/617, Enclosures 630-642, Plants)
  4. SAVAS / SILAH / KORUMA / TEHDIT (Bow-man 125, Bow/Arrow 890/892/519/520, Armed man 095/137,
     Weapons/Tools IMPL, Fierce Beasts Tigr/Rhin/Elep, Hero/Monster amulets TAB:C/Comp)
  5. PARALLEL & OPPOSITE (ANTONYM / PARADIGMATIC) PAIR ENGINE:
     - Same picture + Same context + Never together  = ALLOGRAPH (variant of same sign)
     - DIFFERENT picture + Same context + Never together = SEMANTIC PARALLEL / OPPOSITE PAIR!
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import fisher_exact
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

BOS, EOS = "<BOS>", "<EOS>"

DOMAIN_FAMILIES: dict[str, list[str]] = {
    "YEMEK_BESIN_TAHIL": [
        "411", "413", "415", "416", "417", "420", "421", "422", "423", "426",
        "700", "705", "706", "740", "741", "745", "697", "698", "220", "346", "347", "260",
    ],
    "IS_ZANAAT_TICARET": [
        "151", "153", "154", "155", "156", "158", "160", "161",  # Yoke-carriers / workers
        "315", "317", "318", "320", "321", "323", "326",        # Looms / ladders / craft frames
        "400", "401", "405", "407",                             # Combs / rakes / metal-craft
        "465", "467", "468", "472", "130",                      # Plows / hoes / farmer
        "711", "590", "595",                                    # Crucible / furnace / craft box
    ],
    "TOPRAK_ARAZI_MULK": [
        "615", "617", "621", "625", "626", "630", "632", "634", "636", "637", "642",
        "382", "384", "388", "390",
    ],
    "SAVAS_SILAH_GUC": [
        "125",                  # Man with bow & arrow!
        "890", "892",           # Bow & Bow-with-arrow!
        "519", "520", "521",    # Arrow / Double-arrow / Trident-spear!
        "095", "137", "142",    # Man with club / weapon / staff
        "690", "692",           # Crossed blades / X-frame
        "530", "540", "550",    # Double-axe / shield-bowtie?
    ],
}


def render_glyph_vec(glyph_str: str, font: ImageFont.FreeTypeFont, size: int = 48) -> np.ndarray:
    """Render glyph tightly cropped & centered into a normalized float vector for visual similarity."""
    canvas = Image.new("L", (120, 120), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 24), glyph_str, fill=255, font=font)
    arr = np.asarray(canvas, dtype=np.float32)
    ys, xs = np.where(arr > 20)
    if len(ys) == 0:
        return np.zeros(size * size, dtype=np.float32)
    crop = canvas.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    # Preserve aspect ratio inside square box
    w, h = crop.size
    scale = (size - 6) / max(w, h, 1)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = crop.resize((nw, nh), Image.Resampling.BILINEAR)
    box = Image.new("L", (size, size), 0)
    box.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    vec = np.asarray(box, dtype=np.float32).ravel() / 255.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


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


def find_semantic_parallels_and_opposites(
    unique_seqs: list[list[str]], glyphs_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Separate true SEMANTIC PARALLELS / OPPOSITES (different glyph, identical grammar context,
    mutually exclusive) from ALLOGRAPHS (same glyph, identical context, mutually exclusive).
    """
    sign_counts = Counter(t for s in unique_seqs for t in s)
    vocab = sorted([s for s, c in sign_counts.items() if c >= 12])
    v_idx = {s: i for i, s in enumerate(vocab)}

    ctx = defaultdict(Counter)
    cooc = Counter()
    for seq in unique_seqs:
        padded = [BOS] + seq + [EOS]
        n = len(seq)
        uset = sorted(set(seq))
        for i in range(len(uset)):
            for j in range(i + 1, len(uset)):
                cooc[(uset[i], uset[j])] += 1
        for i, t in enumerate(seq):
            if t not in v_idx:
                continue
            pt = "INIT" if i == 0 else ("FIN" if i == n - 1 else "MED")
            for f in (f"L:{padded[i]}", f"R:{padded[i+2]}", pt):
                ctx[t][f] += 1

    ftot = Counter()
    for c in ctx.values():
        ftot.update(c)
    feats = sorted([f for f, cnt in ftot.items() if cnt >= 3])
    f_idx = {f: i for i, f in enumerate(feats)}
    mat = np.zeros((len(vocab), len(feats)), dtype=np.float64)
    for s, c in ctx.items():
        for f, cnt in c.items():
            if f in f_idx:
                mat[v_idx[s], f_idx[f]] = cnt

    rs = mat.sum(axis=1, keepdims=True) + 1e-12
    cs = mat.sum(axis=0, keepdims=True) + 1e-12
    ppmi = np.maximum(np.log2(np.maximum(mat, 1e-12) / ((rs @ cs) / mat.sum())), 0.0)
    ppmi[mat == 0] = 0.0
    emb = TruncatedSVD(n_components=min(28, len(vocab) - 1), random_state=42).fit_transform(ppmi)
    d_sim = cosine_similarity(emb)

    # Visual similarity from rendered TTF glyphs
    font = ImageFont.truetype(str(RAW_DIR / "sk_indus_script-webfont.ttf"), 52)
    gmap = {}
    for _, r in glyphs_df.iterrows():
        cps = [int(h, 16) for h in r["codepoints"].split() if h]
        if cps:
            gmap[r["sign_id"]] = "".join(chr(c) for c in cps)
    vis_vecs = np.vstack([render_glyph_vec(gmap.get(s, ""), font) for s in vocab])
    v_sim = cosine_similarity(vis_vecs)

    # Compute exact mutual exclusion + contextual similarity for visually DIFFERENT signs!
    n_docs = len(unique_seqs)
    semantic_parallels = []
    visual_allographs = []
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            ds = float(d_sim[i, j])
            vs = float(v_sim[i, j])
            if ds < 0.62:
                continue
            sa, sb = vocab[i], vocab[j]
            ca, cb = sign_counts[sa], sign_counts[sb]
            both = cooc.get((min(sa, sb), max(sa, sb)), 0)
            exp = (ca * cb) / max(n_docs, 1)
            obs_exp = both / max(exp, 1e-6)
            if both > 0 and obs_exp > 0.35:
                continue  # Must avoid each other (mutually exclusive slot competitors!)

            entry = {
                "sign_a": sa,
                "sign_b": sb,
                "freq_a": int(ca),
                "freq_b": int(cb),
                "context_sim": round(ds, 3),
                "visual_sim": round(vs, 3),
                "cooccur_obs": int(both),
                "cooccur_exp": round(exp, 2),
            }
            if vs < 0.35:
                # Visually TOTALLY DIFFERENT signs that occupy the EXACT SAME grammatical slot & avoid each other!
                semantic_parallels.append(entry)
            elif vs >= 0.60:
                visual_allographs.append(entry)

    semantic_parallels.sort(key=lambda x: (-x["context_sim"], x["visual_sim"]))
    visual_allographs.sort(key=lambda x: -(x["context_sim"] + x["visual_sim"]))
    return {
        "vocab_size_ge12": len(vocab),
        "semantic_parallel_opposite_pairs": semantic_parallels[:30],
        "confirmed_visual_allographs_quick": visual_allographs[:20],
    }


def main() -> None:
    ins = pd.read_csv(PROC_DIR / "inscriptions_clean.csv", dtype=str).fillna("")
    lines = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    glyphs = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")
    lines["is_strict_analytic"] = lines["is_strict_analytic"].astype(int)

    df = lines.merge(ins[["inscription_id", "cult", "material", "shape"]], on="inscription_id", how="left", suffixes=("", "_ins"))
    df = df[df["clean_reading_seq"].str.len() > 0].copy()
    strict = df[(df["is_strict_analytic"] == 1) & (df["direction"].isin(["R/L", "L/R", "BUS"]))]
    unique_seqs = [s.split() for s in strict["clean_reading_seq"].unique()]

    # 1. Domain archaeological enrichments for Food, Trade/Work, Land, War/Weapons/Amulets
    contexts = {
        "YEMEK_TROUGH_FEEDING_OFFERING": df["cult"] == "Trough",
        "IS_TICARET_BALE_SEALINGS_TAG": df["object_type"].isin(["TAG", "TAG:L"]),
        "IS_COPPER_METAL_WORK_TAB_C": df["object_type"] == "TAB:C",
        "SAVAS_TOOLS_WEAPONS_IMPL": df["object_type"] == "IMPL",
        "SAVAS_FIERCE_BEASTS_TIGR_RHIN_ELEP": df["symbol"].isin(["Tigr", "Rhin", "Elep"]),
        "KORUMA_AMULET_TABLETS_TAB_B_I": df["object_type"].isin(["TAB:B", "TAB:I"]),
    }
    ctx_results = {}
    for k, m in contexts.items():
        ctx_results[k] = {"lines": int(m.sum()), "top_signs": domain_enrichment(df, m, min_hits=3)[:8]}

    # 2. Check War/Weapon signs (125 bow-man, 892 bow-arrow, 520 arrow, 519 2-arrows, 095 club-man)
    #    specifically: WHERE do they appear, and WHO do they co-occur with / avoid?
    weapon_signs = ["125", "892", "890", "520", "519", "095", "137"]
    weapon_profiles = []
    for ws in weapon_signs:
        sub = df[df["clean_reading_seq"].apply(lambda s: ws in s.split())]
        if len(sub) == 0:
            continue
        weapon_profiles.append(
            {
                "sign": ws,
                "total_lines": int(len(sub)),
                "top_object_types": sub["object_type"].value_counts().head(4).to_dict(),
                "top_symbols": sub["symbol"].replace("", "NONE").value_counts().head(4).to_dict(),
                "top_sites": sub["site"].value_counts().head(4).to_dict(),
            }
        )

    # 3. Domain-to-Domain Interaction Network (Food vs Work vs Land vs War!)
    s2d = {s: d for d, slist in DOMAIN_FAMILIES.items() for s in slist}
    dom_bi = Counter()
    dom_l = Counter()
    dom_r = Counter()
    tot_dbi = 0
    dom_ex = defaultdict(Counter)
    for seq in unique_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            da, db = s2d.get(a), s2d.get(b)
            if da and db:
                tot_dbi += 1
                dom_bi[(da, db)] += 1
                dom_l[da] += 1
                dom_r[db] += 1
                dom_ex[(da, db)][f"{a}->{b}"] += 1
    dom_links = []
    for (da, db), cnt in dom_bi.items():
        if cnt < 8:
            continue
        pmi = math.log2((cnt / tot_dbi) / ((dom_l[da] / tot_dbi) * (dom_r[db] / tot_dbi)))
        dom_links.append({"from": da, "to": db, "count": int(cnt), "pmi": round(pmi, 3), "ex": dom_ex[(da, db)].most_common(3)})
    dom_links.sort(key=lambda x: -x["pmi"])

    # 4. Semantic Parallel / Opposite Pair Discovery Engine
    par_res = find_semantic_parallels_and_opposites(unique_seqs, glyphs)

    out = {
        "archaeological_contexts": ctx_results,
        "weapon_war_sign_profiles": weapon_profiles,
        "domain_interaction_links": dom_links,
        "semantic_parallels_engine": par_res,
    }
    (REPORTS_DIR / "layer3e_values_parallels.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== ARCHAEOLOGICAL CONTEXTS (FOOD / TRADE / WORK / WAR / AMULETS) ===")
    for k, v in ctx_results.items():
        print(f"-- {k} (lines={v['lines']}) --")
        for h in v["top_signs"][:5]:
            print(f"   Sign {h['sign']} | in={h['in']} vs out={h['out_pct']}% | lift={h['lift']}x | p={h['p_value']}")

    print("\n=== WAR / WEAPON / POWER SIGN PROFILES ===")
    for w in weapon_profiles:
        print(f"  Sign {w['sign']} (n={w['total_lines']:3d}) | obj={w['top_object_types']} | sym={w['top_symbols']}")

    print("\n=== DOMAIN x DOMAIN INTERACTION (FOOD / WORK / LAND / WAR) ===")
    for lk in dom_links[:8]:
        print(f"  {lk['from']:20s} -> {lk['to']:20s} | n={lk['count']:4d} | PMI={lk['pmi']:+.3f} | ex={lk['ex']}")
    print("AVOIDED / OPPOSITE DOMAIN PAIRS:")
    for lk in list(reversed(dom_links[-5:])):
        print(f"  {lk['from']:20s} -> {lk['to']:20s} | n={lk['count']:4d} | PMI={lk['pmi']:+.3f}")

    print("\n=== TOP SEMANTIC PARALLEL / OPPOSITE PAIRS (DIFFERENT PICTURE + SAME GRAMMAR SLOT + MUTUALLY EXCLUSIVE!) ===")
    for p in par_res["semantic_parallel_opposite_pairs"][:15]:
        print(f"  {p['sign_a']} ({p['freq_a']}x) <-> {p['sign_b']} ({p['freq_b']}x) | ctx_sim={p['context_sim']:.3f} | vis_sim={p['visual_sim']:.3f} | cooc={p['cooccur_obs']} (exp {p['cooccur_exp']})")


if __name__ == "__main__":
    main()
