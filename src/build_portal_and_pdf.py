#!/usr/bin/env python3
"""
INDUS-DEDUCT — Portal Data Bundler, FaunaTech Doğal Atlas v2.0 Interactive Web Portal,
and Headless-Chrome Print-to-PDF Academic Preprint Compiler.
"""

from __future__ import annotations

import html
import json
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
PAPER_DIR = ROOT / "paper"
PORTAL_DIR = ROOT / "portal"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def build_portal_bundle() -> dict[str, Any]:
    glyphs = pd.read_csv(PROC_DIR / "glyphs_clean.csv", dtype=str).fillna("")
    lines = pd.read_csv(PROC_DIR / "lines_clean.csv", dtype=str).fillna("")
    lines["is_strict_analytic"] = lines["is_strict_analytic"].astype(int)

    # Positional rates for every sign on strict unique lines
    strict = lines[(lines["is_strict_analytic"] == 1) & (lines["direction"].isin(["R/L", "L/R", "BUS"]))]
    pos_counts: dict[str, list[int]] = {}
    for seq_str in strict["clean_reading_seq"].unique():
        seq = seq_str.split()
        n = len(seq)
        for i, s in enumerate(seq):
            d = pos_counts.setdefault(s, [0, 0, 0])
            if i == 0: d[0] += 1
            elif i == n - 1: d[2] += 1
            else: d[1] += 1

    # Load all report JSONs
    l8 = json.loads((REPORTS_DIR / "layer8_calibration.json").read_text())
    l9 = json.loads((REPORTS_DIR / "layer9_internal_rosetta_trade.json").read_text())
    l10 = json.loads((REPORTS_DIR / "layer10_copper_rosetta_deep.json").read_text())
    l11a = json.loads((REPORTS_DIR / "layer11a_external_gulf_diachronic.json").read_text())
    l11b = json.loads((REPORTS_DIR / "layer11b_neural_restoration.json").read_text())
    l12 = json.loads((REPORTS_DIR / "layer12_stratigraphic_time.json").read_text())
    catalog = json.loads((REPORTS_DIR / "damaged_inscriptions_restoration_catalog.json").read_text())

    # Special tag sets
    framed_set = {"226", "232", "234", "236", "241", "592", "804", "842"}
    rim_set = {"740", "741", "742", "743", "744", "745", "746", "747"}
    tabc_key_set = {"341", "753", "777", "780", "781", "782", "749", "845", "407", "923", "321", "503", "615", "752", "706", "705", "806", "850"}
    trade_fdr_set = {"091", "090", "004", "790", "064", "400", "700", "326", "031", "032", "128", "006", "930", "005"}
    opener_set = {"817", "820", "861", "920"}

    # Full codepoint map for all signs (used by portal live renderer)
    cp_map: dict[str, str] = {}
    atlas_signs: list[dict[str, Any]] = []
    glyphs["freq_all_int"] = glyphs["freq_all"].astype(int)
    glyphs["sid_int"] = glyphs["sign_id"].astype(int)
    for _, r in glyphs.sort_values("sid_int").iterrows():
        sid = str(r["sign_id"]).strip().zfill(3)
        cps = [int(h, 16) for h in str(r["codepoints"]).split() if h]
        ch = "".join(chr(c) for c in cps) if cps else ""
        if ch:
            cp_map[sid] = ch
        freq = int(r["freq_all_int"])
        if freq >= 5 and ch:
            p = pos_counts.get(sid, [0, 0, 0])
            tot = sum(p) or 1
            tags = []
            if sid in framed_set: tags.append(":X: çerçeve")
            if sid in rim_set: tags.append("740 küp ailesi")
            if sid in tabc_key_set: tags.append("TAB:C bakır")
            if sid in trade_fdr_set: tags.append("Ticaret FDR*")
            if sid in opener_set: tags.append("Elit açıcı")
            if p[2] / tot >= 0.45: role = "Bitiş"
            elif p[0] / tot >= 0.45: role = "Açıcı"
            else: role = "Orta"
            atlas_signs.append({
                "id": sid,
                "ch": ch,
                "freq": freq,
                "role": role,
                "init_pct": round(100 * p[0] / tot, 1),
                "med_pct": round(100 * p[1] / tot, 1),
                "fin_pct": round(100 * p[2] / tot, 1),
                "tags": tags,
            })

    # Compact 5-way calibration rows
    cal_rows = []
    labels = {
        "indus_strict": "İndus (Kanonik)",
        "sumerian_matched": "Ur III Sümerce (CDLI)",
        "linearb_matched": "Linear B (Miken Yun.)",
        "lineara_words": "Linear A (Minos)",
        "nonlinguistic_stamps": "Dil-Dışı Damgalar (Neg. Kontrol)",
    }
    for key, lbl in labels.items():
        d = l8["primary_5way_matched"][key]
        m1 = d["metric1_info_theory"]
        m2 = d["metric2_affix_stacking"]["fixed_ge20_exact_layer5"]
        m3 = d["metric3_kober_bridge"]
        m4 = d["metric4_masked_prediction"]
        cal_rows.append({
            "key": key,
            "name": lbl,
            "z_h1": m1["z_H1_within_seq_shuffle"],
            "rb": m1["right_edge_bottleneck_ratio"],
            "suf_pre": f"{m2['suffix_2slot_stacks']}:{m2['prefix_2slot_stacks']}",
            "kober_z": m3["bridge_ending_MI_z"],
            "unseen_pp": m4["UNSEEN_TRIGRAM"]["gain_vs_pos_base_pp"],
            "unseen_bits": m4["UNSEEN_TRIGRAM"]["ce_gain_bits"],
            "fin_unseen_pp": m4["final_UNSEEN"]["gain_vs_pos_base_pp"],
        })

    # Known-Answer CSP rows
    csp = l8["metric5_known_answer_csp"]
    csp_rows = [
        {"name": "Linear B (Kapalı %100)", "k": 0, "max_k": "23/23", "z": csp["linearb_23_closed_100pct"]["results"]["k0_anchors"]["z_score"], "locked": "52/52 (%100)", "status": "Tam Çözüm"},
        {"name": "Ur III Sümerce (Kapalı %100)", "k": 0, "max_k": "23/23", "z": csp["sumerian_23_closed_100pct"]["results"]["k0_anchors"]["z_score"], "locked": "22/22 (%100 ortak)", "status": "Tam Çözüm"},
        {"name": "Ur III Sümerce (%74 Açık Sözlük)", "k": 0, "max_k": "14/23", "z": csp["sumerian_23_open_74pct"]["results"]["k0_anchors"]["z_score"], "locked": "8/22 (%36.4)", "status": "Gürültüye Düşer"},
        {"name": "Ur III Sümerce (%74 Açık + 2 Çapa)", "k": 2, "max_k": "21/23", "z": csp["sumerian_23_open_74pct"]["results"]["k2_anchors"]["z_score"], "locked": "15/22 (%68.2)", "status": "Kırılma Başlar"},
        {"name": "İndus 23 Körfez Mührü vs CDLI Ur III", "k": 0, "max_k": "15/23", "z": csp["indus_23_gulf_vs_real_cdli_ur3"]["z_score"], "locked": "0/53 (DOF=30)", "status": "Belirsiz (Çapa Eksik)"},
    ]

    bundle = {
        "meta": {
            "inscriptions": 5679,
            "lines": 5805,
            "core_signs": "240–271",
            "tests_passing": "11/11",
            "restored_damaged_total": len(catalog),
        },
        "cp_map": cp_map,
        "atlas_signs": atlas_signs,
        "calibration_5way": cal_rows,
        "csp_known_answer": csp_rows,
        "copper_rosetta": {
            "mi_tabc_all": l9["module1_copper_rosetta"]["text_symbol_mi_tab_c_all"],
            "mi_tabc_named": l9["module1_copper_rosetta"]["text_symbol_mi_tab_c_named_only"],
            "mi_seals": l9["module1_copper_rosetta"]["text_symbol_mi_seal_s_control"],
            "equations": [
                {"icon": "Tavşan (Hare)", "lock": "12/12 (%100)", "short": "753", "full": "235 705 033 845 407 321 407", "note": "M-593, M-1547 · 321 Tavşan'a özel; 845..407 ortak"},
                {"icon": "Boynuzlu Avcı (Anth)", "lock": "10/10 (%100)", "short": "780 781 782 777", "full": "806 845 061 407 850 900 740", "note": "5 tablet · 777-782 çok-anlamlı kap ailesi"},
                {"icon": "Fil (Elep)", "lock": "11/11 (%100)", "short": "003 422", "full": "706 033 923 740", "note": "923 (üçlü yay/diş) + 706 033 + 740"},
                {"icon": "Standart İkon (Othr/Rhin)", "lock": "28/29 (%96.6)", "short": "341", "full": "503 615 752 740", "note": "341 = 4 ayaklı hayvan piktogramı ↔ 503 615 740"},
                {"icon": "Sonsuzluk Düğümü (Loop)", "lock": "4/4 (%100)", "short": "", "full": "415 220 845 407", "note": "845 (düğüm) + 407 ortak gövde"},
                {"icon": "Keçi / Manda (Goat/Buff)", "lock": "3/3 (%100)", "short": "777 781 782", "full": "231 017 585 095 520 032 407", "note": "M-1555..1558 hepsi arka yüzde 777"},
            ],
            "h1_refutation": l10["test1_tabc_morpheme_architecture"]["length_comparison_H1_vs_H2"],
            "four_family_null": l10["test3_blind_4family_phonetic_tournament"]["family_results"],
        },
        "harappa_tokens": l9["module2_harappa_tokens"],
        "multisig_cargo": l9["module3_multisig_cargo"],
        "trade_audit": {
            "fdr_total": l9["module4_meluhha_trade"]["total_surviving_global_fdr_05"],
            "tests_total": l9["module4_meluhha_trade"]["total_dedup_tests_run"],
            "sign_091": l10["test2_deep_trace_091_and_790"]["sign_091_combined_trade_logistics_test"],
            "cuneiform_ledger": l9["module4_meluhha_trade"]["cuneiform_meluhha_commodity_ledger"],
            "categories": {
                k: {
                    "n_uniq": v["n_dedup_unique_texts"],
                    "n_raw": v["n_raw"],
                    "fdr_hits": [h for h in v["top_dedup_hits"] if h["survives_fdr_05"]],
                    "dup_artifacts": len(v["duplicate_stamp_artifacts_caught"]),
                }
                for k, v in l9["module4_meluhha_trade"]["categories"].items()
            },
        },
        "gulf_escalation": l11a["ring_diagnostics"],
        "dof_curve": l11a["marginal_dof_escalation_curve"],
        "diachronic": l11a["diachronic_successor_audit"],
        "restoration_benchmark": l11b["strict_loco_head_to_head"],
        "restoration_stats": l11b["open_restoration_catalog_stats"],
        "restoration_catalog": catalog,
        "layer12_time": l12,
    }
    return bundle


def write_portal_html(bundle: dict[str, Any]) -> None:
    (PORTAL_DIR / "assets" / "charts").mkdir(parents=True, exist_ok=True)
    (PORTAL_DIR / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW_DIR / "sk_indus_script-webfont.ttf", PORTAL_DIR / "assets" / "indus-font.ttf")
    for img_name in ("calibration_5way_chart.png", "layer9_copper_rosetta_plate.png", "layer12_time_evolution_plate.png"):
        src = REPORTS_DIR / img_name
        if src.exists():
            shutil.copy2(src, PORTAL_DIR / "assets" / "charts" / img_name)

    (PORTAL_DIR / "data" / "indus_portal_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
    )

    portal_html = r"""<!DOCTYPE html>
<html lang="tr" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>INDUS-DEDUCT — Calibrated Computational Epigraphy for the Indus Valley Script</title>
  <meta name="description" content="5,679 Indus inscriptions, 5-way calibration vs Ur III Sumerian &amp; Linear B, 700-year BH-FDR paleographic drift (#838->#803->#806, #156->#154), Mohenjo-daro Copper Tablet Internal Rosetta, and 690 restored damaged inscriptions. 11/11 verified." />
  <link rel="canonical" href="https://indus.emare.dev/" />
  <link rel="icon" type="image/svg+xml" href="./assets/emare-ui/brand/emare-seal-pine.svg" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://indus.emare.dev/" />
  <meta property="og:title" content="INDUS-DEDUCT — Calibrated Computational Epigraphy for the Indus Valley Script" />
  <meta property="og:description" content="5,679 inscriptions · Zero fabricated readings · z_H1 retired · 5-way calibration vs Sumerian &amp; Linear B · 700-yr glyph drift · 690 damaged inscriptions restored." />
  <meta property="og:image" content="https://indus.emare.dev/assets/og-indusemare.png" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="INDUS-DEDUCT — Calibrated Open Benchmark for the Indus Valley Script" />
  <meta name="twitter:description" content="We tested all 5,679 Indus inscriptions against real Ur III Sumerian, Linear B, Linear A &amp; non-linguistic stamps — retiring z_H1 and uncovering 700-year paleographic drift." />
  <meta name="twitter:image" content="https://indus.emare.dev/assets/og-indusemare.png" />
  <link rel="stylesheet" href="./emare-tokens.css" />
  <style>
    @font-face {
      font-family: "IndusICIT";
      src: url("./assets/indus-font.ttf") format("truetype");
      font-display: swap;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--em-canvas);
      color: var(--em-ink);
      font-family: var(--em-font-sans);
      font-size: var(--em-text-base);
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    .indus {
      font-family: "IndusICIT", serif;
      font-size: 1.75rem;
      line-height: 1.1;
      letter-spacing: 0.14em;
      color: var(--em-ink);
    }
    .indus-lg { font-family: "IndusICIT", serif; font-size: 2.35rem; line-height: 1.1; }
    .mono { font-family: var(--em-font-mono); font-size: 0.84em; }
    .serif { font-family: var(--em-font-editorial); }

    header.topbar {
      position: sticky; top: 0; z-index: 40;
      background: var(--em-glass-strong);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--em-border);
    }
    .topbar-inner {
      max-width: 1260px; margin: 0 auto; padding: 14px 24px;
      display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand img { width: 38px; height: 38px; border-radius: 10px; box-shadow: var(--em-shadow-icon); }
    .brand-title { font-family: var(--em-font-editorial); font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; }
    .brand-sub { font-size: var(--em-text-xs); color: var(--em-ink-muted); }

    .actions { display: flex; align-items: center; gap: 10px; }
    .btn {
      display: inline-flex; align-items: center; gap: 7px;
      padding: 8px 14px; border-radius: var(--em-radius-pill);
      border: 1px solid var(--em-border); background: var(--em-surface);
      color: var(--em-ink); font-size: var(--em-text-sm); font-weight: 500;
      cursor: pointer; text-decoration: none;
      transition: all var(--em-dur-fast) var(--em-ease);
    }
    .btn:hover { border-color: var(--em-border-strong); background: var(--em-surface-subtle); }
    .btn-primary {
      background: var(--em-accent); border-color: var(--em-accent); color: var(--em-ink-inverse);
    }
    .btn-primary:hover { background: var(--em-accent-hover); border-color: var(--em-accent-hover); }

    .nav-tabs {
      max-width: 1260px; margin: 0 auto; padding: 0 24px;
      display: flex; gap: 4px; overflow-x: auto; border-top: 1px solid var(--em-border-subtle);
    }
    .tab-btn {
      padding: 11px 13px; border: none; background: transparent;
      color: var(--em-ink-secondary); font-family: var(--em-font-sans); font-size: var(--em-text-sm);
      font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap;
      transition: color var(--em-dur-fast), border-color var(--em-dur-fast);
    }
    .tab-btn:hover { color: var(--em-ink); }
    .tab-btn.active { color: var(--em-accent); border-bottom-color: var(--em-accent); font-weight: 600; }

    main { max-width: 1260px; margin: 0 auto; padding: 28px 24px 72px; }
    .panel { display: none; }
    .panel.active { display: block; animation: fadeIn 200ms var(--em-ease); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    .hero {
      background: linear-gradient(135deg, var(--em-accent-seal) 0%, var(--em-accent-seal-end) 100%);
      color: #f6f7f3; border-radius: var(--em-radius-xl); padding: 32px 36px;
      box-shadow: var(--em-shadow-md); margin-bottom: 24px;
    }
    .hero-eyebrow {
      display: inline-block; font-size: var(--em-text-xs); letter-spacing: 0.08em; text-transform: uppercase;
      color: var(--em-accent-sprout); background: rgba(215, 239, 175, 0.12);
      border: 1px solid rgba(215, 239, 175, 0.28); padding: 4px 10px; border-radius: var(--em-radius-pill);
      margin-bottom: 12px;
    }
    .hero h1 { font-family: var(--em-font-editorial); font-size: clamp(1.75rem, 3vw, 2.5rem); margin: 0 0 10px; font-weight: 500; line-height: 1.18; }
    .hero p { margin: 0; max-width: 78ch; color: rgba(246, 247, 243, 0.86); font-size: var(--em-text-md); }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 24px; }
    .kpi {
      background: var(--em-surface); border: 1px solid var(--em-border); border-radius: var(--em-radius-md);
      padding: 18px 20px; box-shadow: var(--em-shadow-xs);
    }
    .kpi-label { font-size: var(--em-text-xs); color: var(--em-ink-muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-val { font-family: var(--em-font-editorial); font-size: 1.8rem; font-weight: 600; color: var(--em-ink); margin: 4px 0 2px; }
    .kpi-note { font-size: var(--em-text-xs); color: var(--em-ink-secondary); }

    .card {
      background: var(--em-surface); border: 1px solid var(--em-border); border-radius: var(--em-radius-lg);
      padding: 22px 24px; box-shadow: var(--em-shadow-sm); margin-bottom: 20px;
    }
    .card h2 { font-family: var(--em-font-editorial); font-size: 1.38rem; font-weight: 600; margin: 0 0 6px; color: var(--em-ink); }
    .card .lead { font-size: var(--em-text-sm); color: var(--em-ink-secondary); margin: 0 0 16px; }

    .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: var(--em-text-sm); }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--em-border-subtle); vertical-align: middle; }
    th { font-size: var(--em-text-xs); text-transform: uppercase; letter-spacing: 0.04em; color: var(--em-ink-muted); background: var(--em-surface-subtle); }
    tr:hover td { background: var(--em-surface-subtle); }
    .badge {
      display: inline-block; padding: 2px 9px; border-radius: var(--em-radius-pill);
      font-size: var(--em-text-xs); font-weight: 600;
    }
    .badge-ok { background: var(--em-success-soft); color: var(--em-success); }
    .badge-warn { background: var(--em-warning-soft); color: var(--em-warning); }
    .badge-err { background: var(--em-danger-soft); color: var(--em-danger); }
    .badge-clay { background: var(--em-clay-soft); color: var(--em-clay); }
    .badge-soft { background: var(--em-accent-soft); color: var(--em-accent); }

    .plate-img {
      width: 100%; border-radius: var(--em-radius-md); border: 1px solid var(--em-border);
      display: block; background: #fff;
    }

    /* Atlas Grid */
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; align-items: center; }
    .input, select {
      padding: 9px 13px; border-radius: var(--em-radius-sm);
      border: 1px solid var(--em-border); background: var(--em-surface);
      color: var(--em-ink); font-family: var(--em-font-sans); font-size: var(--em-text-sm);
    }
    .atlas-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(148px, 1fr)); gap: 12px; }
    .glyph-card {
      background: var(--em-surface); border: 1px solid var(--em-border); border-radius: var(--em-radius-md);
      padding: 14px 12px; text-align: center; transition: border-color 150ms, transform 150ms;
    }
    .glyph-card:hover { border-color: var(--em-accent); transform: translateY(-2px); box-shadow: var(--em-shadow-sm); }
    .glyph-char { font-family: "IndusICIT", serif; font-size: 2.55rem; line-height: 1.15; color: var(--em-ink); min-height: 46px; }
    .glyph-id { font-family: var(--em-font-mono); font-size: var(--em-text-xs); font-weight: 600; color: var(--em-accent); }
    .glyph-meta { font-size: 0.72rem; color: var(--em-ink-muted); margin-top: 3px; }
    .bar-wrap { display: flex; height: 5px; border-radius: 99px; overflow: hidden; margin-top: 7px; background: var(--em-surface-muted); }
    .bar-i { background: var(--em-clay); }
    .bar-m { background: var(--em-border-strong); }
    .bar-f { background: var(--em-accent); }

    /* Restoration Cards */
    .rest-seq { display: inline-flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .tok-box {
      display: inline-flex; flex-direction: column; align-items: center;
      padding: 4px 7px; border-radius: 8px; border: 1px solid var(--em-border-subtle);
      background: var(--em-surface-subtle); min-width: 40px;
    }
    .tok-box.gap {
      background: var(--em-accent-soft); border: 1.5px solid var(--em-accent);
    }
    .tok-box .g { font-family: "IndusICIT", serif; font-size: 1.45rem; line-height: 1.1; }
    .tok-box .id { font-family: var(--em-font-mono); font-size: 0.68rem; color: var(--em-ink-muted); }
    .tok-box.gap .id { color: var(--em-accent); font-weight: 700; }

    /* Timeline */
    .timeline { border-left: 2px solid var(--em-accent); padding-left: 20px; display: grid; gap: 16px; }
    .tl-item { position: relative; background: var(--em-surface-subtle); border: 1px solid var(--em-border); border-radius: var(--em-radius-md); padding: 14px 18px; }
    .tl-item::before {
      content: ""; position: absolute; left: -27px; top: 18px; width: 12px; height: 12px;
      border-radius: 50%; background: var(--em-accent); border: 2px solid var(--em-canvas);
    }
    .tl-era { font-family: var(--em-font-mono); font-size: var(--em-text-xs); color: var(--em-accent); font-weight: 700; }
    .tl-reg { font-size: var(--em-text-xs); color: var(--em-ink-muted); margin-left: 8px; }
    footer { text-align: center; padding: 24px; color: var(--em-ink-muted); font-size: var(--em-text-xs); border-top: 1px solid var(--em-border); }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <img src="./assets/emare-ui/brand/emare-seal-pine.svg" alt="Emare Mühür" />
        <div>
          <div class="brand-title">INDUS-DEDUCT</div>
          <div class="brand-sub" data-i18n="brandSub">FaunaTech Doğal Atlas v2.0 · Katman 0–12 Açık Hesaplamalı Epigrafi Portalı</div>
        </div>
      </div>
      <div class="actions">
        <a class="btn" href="https://doi.org/10.5281/zenodo.24383348" target="_blank" rel="noopener">DOI</a>
        <a class="btn" href="https://github.com/emraresoftware/indusemare" target="_blank" rel="noopener">GitHub</a>
        <a class="btn" href="https://github.com/emraresoftware/indusemare/blob/main/CLAIM_LEDGER.md" target="_blank" rel="noopener" data-i18n="btnLedger">İddia Defteri</a>
        <a class="btn btn-primary" href="./assets/INDUS_DEDUCT_PREPRINT.pdf" target="_blank" data-i18n="btnPdf">PDF Makale</a>
        <button class="btn" id="langToggle" type="button">EN</button>
        <button class="btn" id="themeToggle" type="button">Orman Gecesi</button>
      </div>
    </div>
    <nav class="nav-tabs" role="tablist">
      <button class="tab-btn active" data-tab="overview" data-i18n="tab1">1. Genel &amp; 5'li Kalibrasyon</button>
      <button class="tab-btn" data-tab="atlas" data-i18n="tab2">2. Canlı Glif Atlası (294)</button>
      <button class="tab-btn" data-tab="rosetta" data-i18n="tab3">3. İç Rosetta &amp; Meluhha Ticareti</button>
      <button class="tab-btn" data-tab="restore" data-i18n="tab4">4. Kırık Yazıt Tamamlama (690)</button>
      <button class="tab-btn" data-tab="diachronic" data-i18n="tab5">5. İndus Sonrası Dil &amp; Körfez Duvarı</button>
      <button class="tab-btn" data-tab="time12" data-i18n="tab6">6. 700 Yıllık Evrim &amp; Denizcilik</button>
    </nav>
  </header>

  <main>
    <!-- TAB 1: OVERVIEW & CALIBRATION -->
    <section class="panel active" id="tab-overview">
      <div class="hero">
        <span class="hero-eyebrow" data-i18n="heroEyebrow">Sıfır Halüsinasyon · 11/11 Doğrulanmış · Mac &amp; Linux Bit-Bit SHA-256 · v1.0.0</span>
        <h1 data-i18n="heroTitle">İndus Vadisi Yazısı İçin Kalibre Edilmiş Açık Benchmark</h1>
        <p data-i18n="heroDesc">5.679 yazıt üzerinde görsel glif analizi, 5'li bilinen-cevap &amp; dil-dışı kalibrasyon, Mohenjo-daro bakır tablet iç Rosettası, 700 yıllık paleografik evrim, Lothal çok-imzalı kargo ağı ve 690 kırık yazıtlık açık onarım kataloğu. Kanıtlanmış yapısal gerçekler ile kanıtlanmamış sesli okumalar kesin çizgiyle ayrılmıştır.</p>
      </div>

      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label" data-i18n="kpi1L">Kanonik Korpus</div><div class="kpi-val" data-i18n="kpi1V">5.679</div><div class="kpi-note" data-i18n="kpi1N">5.805 satır · sıfır kuram kirliliği</div></div>
        <div class="kpi"><div class="kpi-label" data-i18n="kpi2L">Çekirdek İşaret (Tipoloji)</div><div class="kpi-val">240–271</div><div class="kpi-note" data-i18n="kpi2N">Logo-heceli · %95.7 token kapsamı</div></div>
        <div class="kpi"><div class="kpi-label" data-i18n="kpi3L">Kober Köprü Sinyali</div><div class="kpi-val">+53.67σ</div><div class="kpi-note" data-i18n="kpi3N">Dil-dışı kontrol: +0.07σ (sıfır)</div></div>
        <div class="kpi"><div class="kpi-label" data-i18n="kpi4L">Bakır Tablet İç Kilidi</div><div class="kpi-val" data-i18n="kpi4V">%96.2</div><div class="kpi-note" data-i18n="kpi4N">MI z = +19.66σ (SEAL:S +2.79σ)</div></div>
        <div class="kpi"><div class="kpi-label" data-i18n="kpi5L">LOCO Kırık Yazıt Onarımı</div><div class="kpi-val" data-i18n="kpi5V">%57.5</div><div class="kpi-note" data-i18n="kpi5N">Top-1 (n=275) · Top-5 %82.9 · 12 şehir</div></div>
      </div>

      <div class="card">
        <h2 data-i18n="calTitle">Katman 8: 5'li Kalibrasyon (Eşit Boyut &amp; Uzunluk — N = 2.047)</h2>
        <p class="lead" data-i18n="calLead">İndus yazısının gerçek Sümer (Ur III CDLI), Linear B ve Linear A ile aynı kümeye oturup Dil-Dışı Damgalardan nasıl kesin ayrıştığı. z_H1 emekliye ayrıldı; 3 doğrulanmış ayırt edici öne çıktı.</p>
        <div style="overflow-x:auto">
          <table>
            <thead>
              <tr>
                <th data-i18n="thCorpus">Korpus</th>
                <th data-i18n="thZh1">z_H1 (Emekli)</th>
                <th data-i18n="thRb">Sağ-Kenar Daralma</th>
                <th data-i18n="thAffix">2-Ek Yığılması (Son:Ön)</th>
                <th data-i18n="thKober">Kober Köprü MI z</th>
                <th data-i18n="thUnseen">Görülmemiş Trigram Kazancı</th>
                <th><code>final_UNSEEN</code></th>
              </tr>
            </thead>
            <tbody id="calTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2 data-i18n="cspTitle">23-Mühürlü Bilinen-Cevap (Known-Answer) Max-CSP Doğrulaması</h2>
        <p class="lead" data-i18n="cspLead">Aynı çözücü kapalı Linear B ve Ur III Sümercede %100 kilit açarken, %74 açık sözlüklü gerçek Sümercede 0 çapayla gürültüye düşüp 2 çapayla (%68.2) çözülüyor — İndus Körfez duvarı algoritmik değil, fiziksel çapa eksikliği.</p>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th data-i18n="thRegime">23-Mühürlü Graf Rejimi</th><th data-i18n="thAnchor">Çapa (k)</th><th data-i18n="thBest">En İyi Eşleşme</th><th data-i18n="thZ">z-skoru</th><th data-i18n="thLocked">Doğru Kilitlenen</th><th data-i18n="thVerdict">Karar</th></tr></thead>
            <tbody id="cspTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2 data-i18n="calChartTitle">5'li Kalibrasyon Görsel Özeti</h2>
        <img class="plate-img" src="./assets/charts/calibration_5way_chart.png" alt="Layer 8 5-Way Calibration Chart" />
      </div>
    </section>

    <!-- TAB 2: LIVE GLYPH ATLAS -->
    <section class="panel" id="tab-atlas">
      <div class="card">
        <h2 data-i18n="atlasH2">Canlı İndus Glif Atlası (294 Sık İşaret = %95.7 Korpus Kapsamı)</h2>
        <p class="lead">Gerçek ICIT TTF yazı tipiyle canlı çizim. Alt çubuklar: <span class="badge badge-clay">Açıcı</span> · <span class="badge" style="background:var(--em-surface-muted)">Orta</span> · <span class="badge badge-ok">Bitiş</span> oranlarını gösterir.</p>
        <div class="controls">
          <input class="input" id="atlasSearch" placeholder="İşaret no ara (ör. 740, 091, 790)..." style="min-width:240px" />
          <select id="atlasRole">
            <option value="ALL">Tüm Pozisyonlar</option>
            <option value="Açıcı">Açıcı (Init ≥ %45)</option>
            <option value="Orta">Orta (Medial)</option>
            <option value="Bitiş">Bitiş (Final ≥ %45)</option>
          </select>
          <select id="atlasTag">
            <option value="ALL">Tüm Özel Etiketler</option>
            <option value=":X: çerçeve">: X : 4-Çizgili Çerçeve (8)</option>
            <option value="740 küp ailesi">740 Küp Ailesi</option>
            <option value="TAB:C bakır">TAB:C Bakır Tablet Anahtarları</option>
            <option value="Ticaret FDR*">Ticaret Global BH-FDR*</option>
            <option value="Elit açıcı">4 Elit Açıcı</option>
          </select>
          <span class="mono" id="atlasCount" style="color:var(--em-ink-muted)"></span>
        </div>
        <div class="atlas-grid" id="atlasGrid"></div>
      </div>
    </section>

    <!-- TAB 3: INTERNAL ROSETTA & MELUHHA TRADE -->
    <section class="panel" id="tab-rosetta">
      <div class="card">
        <h2>Mohenjo-daro Bakır Tabletleri (TAB:C, 229/230) — İç Rosetta &amp; H1 Reddi</h2>
        <p class="lead">Metin ↔ ikon kilidi <strong>+19.66σ (%96.2 saflık)</strong>; 23 çift-yüzlü bakır tablette arka yüz hayvanı yerine tek bir İndus işareti konmuş. Katman 10 bu metinlerin yalın hayvan adı (H1) değil, <strong>bileşik ritüel/kurumsal formül (H2)</strong> olduğunu 4 bağımsız testle (4 dil ailesi kör fonetik null dahil) kanıtladı.</p>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>Kilitli İkon</th><th>Saflık</th><th>Tek-İşaret Karşılığı</th><th>Tam Yazım (Okuma Yönü →)</th><th>Yapısal Not</th></tr></thead>
            <tbody id="tabcTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <h2>Çift İnsan #091 — En Güçlü İhracat &amp; Kargo Başlığı</h2>
          <p class="lead">Tüm 5.679 yazıtta yalnızca 28 kez geçer; <strong>%0 bitiş</strong>, %82.1 açıcı/tekil. Tekrarsız olarak <strong>%45.0</strong>'ı (hamda %57.1) ihracat mührü, yuvarlak Körfez mührü veya kargo balyasında (<strong>11.89×, p = 1.48e-6</strong>; Körfez'de <strong>42.35×, q = 2.86e-4</strong>).</p>
          <div style="display:flex;align-items:center;gap:18px;padding:14px;border-radius:12px;background:var(--em-surface-subtle);border:1px solid var(--em-border)">
            <div class="indus-lg" id="glyph091"></div>
            <div>
              <div><strong class="mono">#091</strong> · <span class="badge badge-ok">Global BH-FDR q = 2.86e-4</span></div>
              <div style="font-size:var(--em-text-xs);color:var(--em-ink-secondary);margin-top:4px">Massimo Vidale'nin Failaka/Bahreyn ikizler/tüccar hipoteziyle bağımsız örtüşme.</div>
            </div>
          </div>
        </div>
        <div class="card">
          <h2>Lothal Çok-İmzalı Kargo &amp; Harappa Fiş Kilidi</h2>
          <p class="lead"><strong>26 çok-imzalı kil balyası</strong> (Lothal: 12, Dholavira: 6) ve <strong>9 şehirlerarası kargo ↔ taş mühür</strong> bağlantısı. Lothal ana gümrük mührü 4 kargoda karşı-imza atıyor (biri 700 km ötedeki Mohenjo-daro taş mührü <code>861 002 235</code>).</p>
          <p class="lead" style="margin-bottom:0"><strong>Harappa 772 çift-yüzlü fişi:</strong> Kurum ↔ sabit kota kilidi <strong>MI z = +9.55σ</strong> ve <code>: X : + 400</code> ek yığılması <strong>20/21 (%95.2, p = 1.16e-8)</strong>.</p>
        </div>
      </div>

      <div class="card">
        <h2>Meluhha Dış Ticaret Malları (Çivi Yazısı Kayıtları &amp; Tekrarsız BH-FDR)</h2>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>Ticari Mal</th><th>Sümerce / Akadca</th><th>İndus Kaynak / Arkeolojik İz</th><th>Etimoloji / Durum</th></tr></thead>
            <tbody id="tradeTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2>Katman 9 Görsel Glif Levhası (İç Rosetta, Lothal Kargo, Harappa Fişi &amp; Ticaret)</h2>
        <img class="plate-img" src="./assets/charts/layer9_copper_rosetta_plate.png" alt="Layer 9 Visual Glyph Plate" />
      </div>
    </section>

    <!-- TAB 4: DAMAGED INSCRIPTION RESTORATION -->
    <section class="panel" id="tab-restore">
      <div class="card">
        <h2>Katman 11B: Sıkı Şehir-Dışarıda (LOCO) Hasarlı Yazıt Onarım Kıyaslaması</h2>
        <p class="lead">12 şehir dışarıda bırakılarak (sıfır sızıntı, sabit ön-ağırlıklar) 275 kenar + 10 iç kırık üzerinde görsel glif SVD + nesne-türü + nöral MLP hibrit motorunun saf n-gram ile kıyaslaması.</p>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>LOCO Alt Kümesi</th><th>n</th><th>Pozisyon Tabanı</th><th>N-gram Top-1 / Top-5</th><th>Nöral MLP Top-1</th><th>Hibrit Motor Top-1 / Top-5</th></tr></thead>
            <tbody id="locoTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2>Açık Kırık Yazıt Onarım Kataloğu (690 Gerçek Hasarlı Yazıt · 25 Şehir)</h2>
        <p class="lead">Kırık (<code>000</code>) konumu Orman Çamı çerçevesiyle işaretlenmiş, en yüksek olasılıklı 5 aday gerçek İndus glifleriyle listelenmiştir.</p>
        <div class="controls">
          <select id="restSite"><option value="ALL">Tüm Şehirler (25)</option></select>
          <select id="restType">
            <option value="NOVEL">Yalnızca Özgün (Eşi Olmayan, 405)</option>
            <option value="ALL">Tümü (690)</option>
            <option value="ATTESTED">Birebir Paraleli Olanlar (285)</option>
          </select>
          <input class="input" id="restSearch" placeholder="ID veya işaret ara (ör. 740, Lothal)..." style="min-width:220px" />
          <span class="mono" id="restCount" style="color:var(--em-ink-muted)"></span>
        </div>
        <div style="overflow-x:auto">
          <table>
            <thead><tr><th>ID / Şehir / Tür</th><th>Kırık Dizi (En İyi Tamamlama Gösterili)</th><th>Top-5 Aday (Olasılık)</th><th>Güven</th></tr></thead>
            <tbody id="restTableBody"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 5: DIACHRONIC & GULF WALL -->
    <section class="panel" id="tab-diachronic">
      <div class="card">
        <h2>İndus Dilinden Sonra Aynı Bölgede Hangi Dil Geldi? (MÖ 2600 → MÖ 300)</h2>
        <p class="lead">İndus Vadisi'nde (Saptasindhu) Harappa sonrası belgelenen ilk dil <strong>Erken Veda Sanskritçesidir (Rigveda)</strong> — ancak yerel Harappa sonrası nüfustan ~300–380 Hint-Avrupa-dışı kelime ve üstdamaksıl (retrofleks) sesleri devralmıştır.</p>
        <div class="timeline" id="timelineWrap"></div>
      </div>

      <div class="two-col">
        <div class="card">
          <h2>Rigveda'daki İndus Sonrası Alt-Katman (Substrate) Fosilleri</h2>
          <table>
            <thead><tr><th>Kelime</th><th>Anlam</th><th>Harappa Bağlantısı</th></tr></thead>
            <tbody id="substrateBody"></tbody>
          </table>
        </div>
        <div class="card">
          <h2>Katman 11A: Yurt Dışı Kare vs Yuvarlak Mühür &amp; DOF Tırmanışı</h2>
          <p class="lead">Yurt dışı <strong>Kare Harappa mühürleri</strong> (`n=12`) kendi <code>740</code> gramerini (%25) koruyup <code>090/091</code> hiç kullanmazken (`0/12`), <strong>Yuvarlak Körfez mühürleri</strong> (`n=17`) <code>740</code>'ı yarıya düşürüp <code>090/091</code>'e (11/17) geçer.</p>
          <table>
            <thead><tr><th>Halka</th><th>Mühür (E)</th><th>Tekil İşaret (U)</th><th>DOF</th><th>Tekil %</th><th>740 Bitiş</th><th>091 / 090</th></tr></thead>
            <tbody id="gulfRingBody"></tbody>
          </table>
          <div style="margin-top:14px;padding:12px 14px;border-radius:10px;background:var(--em-warning-soft);color:var(--em-warning);font-size:var(--em-text-xs)">
            <strong>Witzel (Para-Munda Ön-Ek) vs Proto-Dravid (Son-Ek) Kararı:</strong> Kendi Katman 8 ölçümümüz İndus'un <strong>207:1</strong> oranında neredeyse saf <strong>son-ekli</strong> (4.96× sağ-kenar daralması) olduğunu gösterdi; bu durum ön-ekli Para-Munda yerine son-ekli Proto-Dravid (veya kayıp son-ekli bir aile) ile uyumludur.
          </div>
        </div>
      </div>
    </section>
    <!-- TAB 6: 700-YEAR STRATIGRAPHIC EVOLUTION & MARITIME SPLIT -->
    <section class="panel" id="tab-time12">
      <div class="two-col">
        <div class="card">
          <h2>İlk Nicel Paleografik Evrim Zincirleri (BH-FDR q &lt; 0.05)</h2>
          <p class="lead">Mohenjo-daro alan-içi normalleştirilmiş derinlik (n=1.253) &amp; tek-datum DKG(S) (n=595) ile Harappa Period 2→5 (Spearman rho = +0.289, p = 0.0376) çapraz doğrulaması:</p>
          <div id="paleoChains" style="display:grid;gap:12px"></div>
        </div>
        <div class="card">
          <h2>Denizcilik (Kargo/Körfez) vs Irmak Başkentleri (Tabletler)</h2>
          <p class="lead">Tabletlerin %98.9'u iç nehir şehirlerinde (Harappa Gavi timsahı 62/66, Balık 47/47); gerçek açık-deniz lojistiği <strong>TAG* (%65.3)</strong> ve <strong>SEAL:C (~%66)</strong> üzerinde.</p>
          <table>
            <thead><tr><th>Belge Türü</th><th>Deniz Kıyısı &amp; Körfez</th><th>Aşağı İndus (MD)</th><th>İç Kara (Harappa vb.)</th></tr></thead>
            <tbody>
              <tr><td><strong>TAB:I (Minyatür)</strong></td><td class="mono">19 (%2.0)</td><td class="mono">8</td><td class="mono"><strong>925 (%97.2)</strong></td></tr>
              <tr><td><strong>TAB:B (Kabartma)</strong></td><td class="mono">5 (%0.5)</td><td class="mono">150</td><td class="mono"><strong>946 (%85.4)</strong></td></tr>
              <tr><td><strong>TAB:C (Bakır)</strong></td><td class="mono">0 (%0.0)</td><td class="mono"><strong>229 (%99.6)</strong></td><td class="mono">1</td></tr>
              <tr style="background:var(--em-accent-soft)"><td><strong>TAG* (Kargo Kil)</strong></td><td class="mono"><strong>171 (%65.3)</strong></td><td class="mono">29</td><td class="mono">56</td></tr>
              <tr style="background:var(--em-accent-soft)"><td><strong>SEAL:C/CY (Yuvarlak)</strong></td><td class="mono"><strong>16 (+11 Körfez)</strong></td><td class="mono">9</td><td class="mono">5</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2>4 Eksenli Hilal Merdiveni (#900 → #904 → #923, TAB:C Hariç Tekrarsız)</h2>
        <table>
          <thead><tr><th>İşaret</th><th>Görsel</th><th>n</th><th>Solunda Balık/Yıldız</th><th>Solunda Sayı</th><th>Sağında 740 Küp</th><th>MD Zaman Sırası</th></tr></thead>
          <tbody id="crescentBody"></tbody>
        </table>
      </div>

      <div class="card">
        <h2>Katman 12: 700 Yıllık Stratigrafik Zaman Evrimi Levhası</h2>
        <img class="plate-img" src="./assets/charts/layer12_time_evolution_plate.png" alt="Layer 12 Stratigraphic Time Evolution Plate" />
      </div>
    </section>
  </main>

  <footer data-i18n="footerTxt">
    INDUS-DEDUCT Açık Benchmark (v1.0.0-preprint) · FaunaTech Doğal Atlas v2.0 (emare-ui) · 11/11 Claim-Ledger Doğrulamalı · Sıfır Uydurma Okuma
  </footer>

  <script>
    const I18N = {
      tr: {
        brandSub: "FaunaTech Doğal Atlas v2.0 · Katman 0–12 Açık Hesaplamalı Epigrafi Portalı",
        btnLedger: "İddia Defteri",
        btnPdf: "PDF Makale",
        tab1: "1. Genel & Kalibrasyon",
        tab2: "2. Glif Atlası (294)",
        tab3: "3. İç Rosetta & Ticaret",
        tab4: "4. Kırık Onarım (690)",
        tab5: "5. Ardıl Dil & Körfez",
        tab6: "6. 700 Yıl & Denizcilik",
        heroEyebrow: "Sıfır Halüsinasyon · 11/11 Doğrulanmış · DOI: 10.5281/zenodo.24383348 · v1.0.1",
        heroTitle: "İndus Vadisi Yazısı İçin Kalibre Edilmiş Açık Benchmark",
        heroDesc: "5.679 yazıt üzerinde görsel glif analizi, 5'li bilinen-cevap & dil-dışı kalibrasyon, Mohenjo-daro bakır tablet iç Rosettası, 700 yıllık paleografik evrim, Lothal çok-imzalı kargo ağı ve 690 kırık yazıtlık açık onarım kataloğu. Kanıtlanmış yapısal gerçekler ile kanıtlanmamış sesli okumalar kesin çizgiyle ayrılmıştır.",
        kpi1L: "Kanonik Korpus", kpi1V: "5.679", kpi1N: "5.805 satır · sıfır kuram kirliliği",
        kpi2L: "Çekirdek İşaret (Tipoloji)", kpi2N: "Logo-heceli · %95.7 token kapsamı",
        kpi3L: "Kober Köprü Sinyali", kpi3N: "Dil-dışı kontrol: +0.07σ (sıfır)",
        kpi4L: "Bakır Tablet İç Kilidi", kpi4V: "%96.2", kpi4N: "MI z = +19.66σ (SEAL:S +2.79σ)",
        kpi5L: "LOCO Kırık Yazıt Onarımı", kpi5V: "%57.5", kpi5N: "Top-1 (n=275) · Top-5 %82.9 · 12 şehir",
        calTitle: "Katman 8: 5'li Kalibrasyon (Eşit Boyut & Uzunluk — N = 2.047)",
        calLead: "İndus yazısının gerçek Sümer (Ur III CDLI), Linear B ve Linear A ile aynı kümeye oturup Dil-Dışı Damgalardan nasıl kesin ayrıştığı. z_H1 emekliye ayrıldı; 3 doğrulanmış ayırt edici öne çıktı.",
        thCorpus: "Korpus", thZh1: "z_H1 (Emekli)", thRb: "Sağ-Kenar Daralma", thAffix: "2-Ek Yığılması (Son:Ön)",
        thKober: "Kober Köprü MI z", thUnseen: "Görülmemiş Trigram Kazancı",
        cspTitle: "23-Mühürlü Bilinen-Cevap (Known-Answer) Max-CSP Doğrulaması",
        cspLead: "Aynı çözücü kapalı Linear B ve Ur III Sümercede %100 kilit açarken, %74 açık sözlüklü gerçek Sümercede 0 çapayla gürültüye düşüp 2 çapayla (%68.2) çözülüyor — İndus Körfez duvarı algoritmik değil, fiziksel çapa eksikliği.",
        thRegime: "23-Mühürlü Graf Rejimi", thAnchor: "Çapa (k)", thBest: "En İyi Eşleşme", thZ: "z-skoru", thLocked: "Doğru Kilitlenen", thVerdict: "Karar",
        calChartTitle: "5'li Kalibrasyon Görsel Özeti",
        atlasH2: "Canlı İndus Glif Atlası (294 Sık İşaret = %95.7 Korpus Kapsamı)",
        themeDark: "Orman Gecesi", themeLight: "Sabah Keteni",
        footerTxt: "INDUS-DEDUCT Açık Benchmark (v1.0.1 · DOI: 10.5281/zenodo.24383348) · FaunaTech Doğal Atlas v2.0 (emare-ui) · 11/11 Doğrulamalı · Sıfır Uydurma Okuma"
      },
      en: {
        brandSub: "FaunaTech Natural Atlas v2.0 · Layers 0–12 Open Computational Epigraphy Portal",
        btnLedger: "Claim Ledger",
        btnPdf: "Preprint PDF",
        tab1: "1. Overview & Calibration",
        tab2: "2. Glyph Atlas (294)",
        tab3: "3. Rosetta & Trade",
        tab4: "4. Restorations (690)",
        tab5: "5. Gulf & Successors",
        tab6: "6. 700-Yr Drift & Sea",
        heroEyebrow: "Zero Hallucination · 11/11 Verified · DOI: 10.5281/zenodo.24383348 · v1.0.1",
        heroTitle: "A Calibrated Open Benchmark for the Indus Valley Script",
        heroDesc: "13-layer reproducible computational epigraphy across 5,679 inscriptions: 5-way calibration vs. real Ur III Sumerian, Linear B, Linear A & non-linguistic stamps; Mohenjo-daro Copper Tablet internal Rosetta; 700-year BH-FDR paleographic drift; Lothal multi-signature cargo network; and 690 strict-LOCO damaged restorations. Proven structure is strictly separated from unproven phonetic readings.",
        kpi1L: "Canonical Corpus", kpi1V: "5,679", kpi1N: "5,805 lines · zero speculative leakage",
        kpi2L: "Core Graphemes", kpi2N: "Logo-syllabic · 95.7% token coverage",
        kpi3L: "Kober Bridge MI z", kpi3N: "Non-linguistic control: +0.07σ (flat zero)",
        kpi4L: "Copper Tablet Lock", kpi4V: "96.2%", kpi4N: "MI z = +19.66σ (vs. SEAL:S +2.79σ)",
        kpi5L: "LOCO Restoration", kpi5V: "57.5%", kpi5N: "Top-1 (n=275) · Top-5 82.9% · 12 cities",
        calTitle: "Layer 8: Five-Way Matched Calibration (N = 2,047 Sequences)",
        calLead: "Indus clusters with real Ur III Sumerian (CDLI) and Linear B while separating cleanly from structured Non-Linguistic Stamps. Within-sequence z_H1 is retired (-245.3σ on non-linguistic stamps); three validated discriminators remain.",
        thCorpus: "Corpus", thZh1: "z_H1 (Retired)", thRb: "Right-Edge Bottleneck", thAffix: "2-Affix Stack (Suf:Pre)",
        thKober: "Kober Bridge MI z", thUnseen: "Unseen-Trigram Gain",
        cspTitle: "23-Seal Known-Answer Max-CSP Identifiability Calibration",
        cspLead: "The exact same solver locks 23/23 (100%) on closed Linear B and Ur III Sumerian at k=0, collapses to noise on realistic 74% open Sumerian at k=0, and rescues 21/23 (68.2% shared signs) at k=2 anchors — proving the Gulf decipherment wall is physical anchor scarcity, not algorithmic failure.",
        thRegime: "23-Seal Graph Regime", thAnchor: "Anchors (k)", thBest: "Best Match", thZ: "z-score", thLocked: "True Locked", thVerdict: "Verdict",
        calChartTitle: "Five-Way Calibration Visual Summary",
        atlasH2: "Live Indus Glyph Atlas (294 Core Signs = 95.7% Corpus Coverage)",
        themeDark: "Forest Night", themeLight: "Morning Linen",
        footerTxt: "INDUS-DEDUCT Open Benchmark (v1.0.1 · DOI: 10.5281/zenodo.24383348) · FaunaTech Natural Atlas v2.0 (emare-ui) · 11/11 Verified · Zero Fabricated Readings"
      }
    };
    const CAL_NAMES_EN = {
      indus_strict: "Indus (Canonical)",
      sumerian_matched: "Ur III Sumerian (CDLI)",
      linearb_matched: "Linear B (Mycenaean Greek)",
      lineara_words: "Linear A (Minoan)",
      nonlinguistic_stamps: "Non-Linguistic Stamps (Neg. Control)"
    };
    const CSP_MAP_EN = {
      "Linear B (Kapalı %100)": "Linear B (Closed 100%)",
      "Ur III Sümerce (Kapalı %100)": "Ur III Sumerian (Closed 100%)",
      "Ur III Sümerce (Açık %74)": "Ur III Sumerian (Open 74%)",
      "İndus Körfez Bileşeni (Gerçek)": "Indus Gulf Component (Real)",
      "Tam Çözüm": "Exact Lock",
      "Gürültü (Çapasız)": "Noise (Unanchored)",
      "Çapayla Kırılma!": "Anchor Breakthrough!",
      "Belirsiz (DOF=30)": "Underdetermined (DOF=30)"
    };
    const urlLang = new URLSearchParams(location.search).get("lang");
    let curLang = urlLang === "en" || urlLang === "tr" ? urlLang : ((navigator.language || "").toLowerCase().startsWith("tr") ? "tr" : "en");
    const langBtn = document.getElementById("langToggle");
    const themeBtn = document.getElementById("themeToggle");
    let portalData = null;

    function syncThemeBtn() {
      const isDark = document.documentElement.getAttribute("data-theme") === "dark";
      const dict = I18N[curLang];
      themeBtn.textContent = isDark ? dict.themeLight : dict.themeDark;
    }

    function renderCalTables() {
      if (!portalData) return;
      const B = portalData;
      document.getElementById("calTableBody").innerHTML = B.calibration_5way.map(r => {
        const isIndus = r.key === "indus_strict";
        const isNeg = r.key === "nonlinguistic_stamps";
        const bg = isIndus ? 'style="background:var(--em-accent-soft);font-weight:600"' : (isNeg ? 'style="background:var(--em-danger-soft)"' : "");
        const label = curLang === "en" ? (CAL_NAMES_EN[r.key] || r.name) : r.name;
        return `<tr ${bg}>
          <td>${label}</td>
          <td class="mono">${r.z_h1.toFixed(2)}σ</td>
          <td class="mono">${r.rb.toFixed(2)}×</td>
          <td class="mono">${r.suf_pre}</td>
          <td class="mono"><strong>${r.kober_z >= 0 ? "+" : ""}${r.kober_z.toFixed(2)}σ</strong></td>
          <td class="mono">${r.unseen_pp >= 0 ? "+" : ""}${r.unseen_pp.toFixed(2)} pp (${r.unseen_bits >= 0 ? "+" : ""}${r.unseen_bits.toFixed(3)} b)</td>
          <td class="mono">${r.fin_unseen_pp.toFixed(2)} pp</td>
        </tr>`;
      }).join("");

      document.getElementById("cspTableBody").innerHTML = B.csp_known_answer.map(r => {
        const cls = r.status.includes("Tam") || r.status.includes("Kırılma") ? "badge-ok" : (r.status.includes("Gürültü") ? "badge-err" : "badge-warn");
        const rName = curLang === "en" ? (CSP_MAP_EN[r.name] || r.name) : r.name;
        const rStatus = curLang === "en" ? (CSP_MAP_EN[r.status] || r.status) : r.status;
        return `<tr><td>${rName}</td><td class="mono">k=${r.k}</td><td class="mono">${r.max_k}</td><td class="mono">${r.z >= 0 ? "+" : ""}${r.z.toFixed(2)}σ</td><td class="mono">${r.locked}</td><td><span class="badge ${cls}">${rStatus}</span></td></tr>`;
      }).join("");
    }

    function applyLang(l) {
      curLang = l;
      document.documentElement.lang = l;
      const dict = I18N[l];
      document.querySelectorAll("[data-i18n]").forEach(el => {
        const k = el.getAttribute("data-i18n");
        if (dict[k]) el.textContent = dict[k];
      });
      langBtn.textContent = l === "tr" ? "EN" : "TR";
      syncThemeBtn();
      renderCalTables();
    }
    langBtn.addEventListener("click", () => applyLang(curLang === "tr" ? "en" : "tr"));
    applyLang(curLang);

    themeBtn.addEventListener("click", () => {
      const root = document.documentElement;
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      syncThemeBtn();
    });

    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      });
    });

    function renderSeqInline(seqStr, cpMap) {
      if (!seqStr) return '<span style="color:var(--em-ink-muted)">—</span>';
      return seqStr.split(/\s+/).filter(Boolean).map(t => {
        const g = cpMap[t.padStart(3, "0")] || "?";
        return `<span class="tok-box"><span class="g">${g}</span><span class="id">${t}</span></span>`;
      }).join("");
    }

    fetch("./data/indus_portal_bundle.json").then(r => r.json()).then(B => {
      portalData = B;
      const cp = B.cp_map;
      document.getElementById("glyph091").textContent = cp["091"] || "091";

      // 1 & 2. Calibration 5-way & CSP Known-Answer (language-aware)
      renderCalTables();

      // 3. Atlas
      function drawAtlas() {
        const q = document.getElementById("atlasSearch").value.trim().toLowerCase();
        const role = document.getElementById("atlasRole").value;
        const tag = document.getElementById("atlasTag").value;
        const rows = B.atlas_signs.filter(s => {
          if (q && !s.id.includes(q) && !s.tags.join(" ").toLowerCase().includes(q)) return false;
          if (role !== "ALL" && s.role !== role) return false;
          if (tag !== "ALL" && !s.tags.includes(tag)) return false;
          return true;
        });
        document.getElementById("atlasCount").textContent = `${rows.length} / ${B.atlas_signs.length} işaret`;
        document.getElementById("atlasGrid").innerHTML = rows.map(s => `
          <div class="glyph-card">
            <div class="glyph-char">${s.ch}</div>
            <div class="glyph-id">#${s.id} · ${s.freq}×</div>
            <div class="glyph-meta">${s.role} (${s.init_pct}/${s.med_pct}/${s.fin_pct}%)</div>
            <div class="bar-wrap"><div class="bar-i" style="width:${s.init_pct}%"></div><div class="bar-m" style="width:${s.med_pct}%"></div><div class="bar-f" style="width:${s.fin_pct}%"></div></div>
            ${s.tags.length ? `<div style="margin-top:6px">${s.tags.map(t => `<span class="badge badge-soft">${t}</span>`).join(" ")}</div>` : ""}
          </div>
        `).join("");
      }
      ["atlasSearch", "atlasRole", "atlasTag"].forEach(id => document.getElementById(id).addEventListener("input", drawAtlas));
      drawAtlas();

      // 4. TAB:C & Trade
      document.getElementById("tabcTableBody").innerHTML = B.copper_rosetta.equations.map(e => `
        <tr>
          <td><strong>${e.icon}</strong></td>
          <td><span class="badge badge-ok">${e.lock}</span></td>
          <td><div class="rest-seq">${renderSeqInline(e.short, cp)}</div></td>
          <td><div class="rest-seq">${renderSeqInline(e.full, cp)}</div></td>
          <td style="font-size:var(--em-text-xs);color:var(--em-ink-secondary)">${e.note}</td>
        </tr>
      `).join("");

      document.getElementById("tradeTableBody").innerHTML = B.trade_audit.cuneiform_ledger.map(t => `
        <tr>
          <td><strong>${t.commodity_tr}</strong></td>
          <td class="mono">${t.cuneiform_sumerian}</td>
          <td style="font-size:var(--em-text-xs)">${t.indus_source_hubs}<br/><span style="color:var(--em-ink-muted)">${t.corpus_presence}</span></td>
          <td style="font-size:var(--em-text-xs);color:var(--em-ink-secondary)">${t.etymology_note}</td>
        </tr>
      `).join("");

      // 5. LOCO benchmark & 690 Catalog
      const locoNames = {
        edge_all_STRICT_LOCO: "Kenar Kırıkları (Tümü, 12 Şehir LOCO)",
        edge_single_unambiguous_STRICT_LOCO: "Kenar — Tek Anlamlı",
        edge_cross_site_STRICT_LOCO: "Kenar — Çapraz Şehir",
        edge_cross_site_single_STRICT_LOCO: "Kenar — Çapraz Şehir Tekil",
        interior_all_STRICT_LOCO: "İç Kırıklar (En Zor)"
      };
      document.getElementById("locoTableBody").innerHTML = Object.entries(B.restoration_benchmark).map(([k, v]) => `
        <tr>
          <td><strong>${locoNames[k] || k}</strong></td>
          <td class="mono">${v.n}</td>
          <td class="mono">${v.pos_baseline_top1_pct}%</td>
          <td class="mono">${v.ngram.top1_pct}% / ${v.ngram.top5_pct}%</td>
          <td class="mono">${v.neural_mlp.top1_pct}%</td>
          <td class="mono"><span class="badge badge-ok">${v.hybrid_ensemble.top1_pct}% / ${v.hybrid_ensemble.top5_pct}%</span></td>
        </tr>
      `).join("");

      const sites = [...new Set(B.restoration_catalog.map(x => x.site))].sort();
      const siteSel = document.getElementById("restSite");
      sites.forEach(s => { const o = document.createElement("option"); o.value = s; o.textContent = s; siteSel.appendChild(o); });

      function drawRest() {
        const sVal = siteSel.value;
        const tVal = document.getElementById("restType").value;
        const q = document.getElementById("restSearch").value.trim().toLowerCase();
        const rows = B.restoration_catalog.filter(x => {
          if (sVal !== "ALL" && x.site !== sVal) return false;
          if (tVal === "NOVEL" && !x.is_novel_unattested_restoration) return false;
          if (tVal === "ATTESTED" && x.is_novel_unattested_restoration) return false;
          if (q && !(x.inscription_id.toLowerCase().includes(q) || x.site.toLowerCase().includes(q) || x.damaged_reading_seq.includes(q) || x.top1_sign.includes(q))) return false;
          return true;
        });
        document.getElementById("restCount").textContent = `${rows.length} yazıt gösteriliyor (ilk 60)`;
        document.getElementById("restTableBody").innerHTML = rows.slice(0, 60).map(x => {
          const seqHtml = x.damaged_reading_seq.split(" ").map((t, idx) => {
            if (idx === x.missing_index) {
              const g = cp[x.top1_sign.padStart(3, "0")] || "?";
              return `<span class="tok-box gap" title="Önerilen #${x.top1_sign} (${(100*x.top1_prob).toFixed(1)}%)"><span class="g">${g}</span><span class="id">→${x.top1_sign}</span></span>`;
            }
            const g = cp[t.padStart(3, "0")] || "?";
            return `<span class="tok-box"><span class="g">${g}</span><span class="id">${t}</span></span>`;
          }).join("");
          const cands = x.top5_candidates.map(c => `<span class="mono" style="margin-right:6px"><strong>${c.sign}</strong>:${(100*c.prob).toFixed(0)}%</span>`).join("");
          const tag = x.is_novel_unattested_restoration ? `<span class="badge badge-soft">Özgün</span>` : `<span class="badge badge-ok">Paralel: ${x.attested_intact_parallel_fills.join(",")}</span>`;
          return `<tr>
            <td><strong class="mono">${x.inscription_id}</strong> ${x.cisi !== "-" ? `<span class="mono" style="color:var(--em-ink-muted)">(${x.cisi})</span>` : ""}<br/><span style="font-size:var(--em-text-xs);color:var(--em-ink-secondary)">${x.site} · ${x.object_type}</span><br/>${tag}</td>
            <td><div class="rest-seq">${seqHtml}</div></td>
            <td>${cands}</td>
            <td class="mono">${(100*x.top1_prob).toFixed(1)}%<br/><span style="font-size:0.72rem;color:var(--em-ink-muted)">H-güven ${x.confidence_score}</span></td>
          </tr>`;
        }).join("");
      }
      ["restSite", "restType", "restSearch"].forEach(id => document.getElementById(id).addEventListener("input", drawRest));
      drawRest();

      // 6. Diachronic & Gulf
      document.getElementById("timelineWrap").innerHTML = B.diachronic.timeline_stages.map(s => `
        <div class="tl-item">
          <div><span class="tl-era">${s.era}</span><span class="tl-reg">${s.region}</span></div>
          <div style="margin-top:5px;font-size:var(--em-text-sm)">${s.language_status}</div>
        </div>
      `).join("");

      document.getElementById("substrateBody").innerHTML = B.diachronic.rigvedic_post_harappan_substrate_fossils.map(w => `
        <tr><td class="mono"><strong>${w.word}</strong></td><td>${w.meaning}</td><td style="font-size:var(--em-text-xs);color:var(--em-ink-secondary)">${w.harappan_link}</td></tr>
      `).join("");

      const ringLabels = {
        ring1_gulf_cc: "Halka 1: 24 Körfez Yuvarlak CC",
        ring2_all_foreign_plus_unk_round: "Halka 2: Tüm Yurt Dışı (42)",
        ring3_all_foreign_plus_core_round_cyl: "Halka 3: +İç Yuvarlak/Silindir (55)",
        foreign_square_seals_only: "Yurt Dışı Sadece KARE (12)",
        foreign_round_cyl_seals_only: "Yurt Dışı Sadece YUVARLAK (17)"
      };
      document.getElementById("gulfRingBody").innerHTML = Object.entries(B.gulf_escalation).map(([k, d]) => `
        <tr>
          <td style="font-size:var(--em-text-xs)"><strong>${ringLabels[k] || k}</strong></td>
          <td class="mono">${d.E_seals}</td>
          <td class="mono">${d.U_unique_signs}</td>
          <td class="mono"><strong>${d.DOF_U_minus_rank}</strong></td>
          <td class="mono">${d.singleton_pct}%</td>
          <td class="mono">${d.jar_740_final_pct}%</td>
          <td class="mono">${d.has_091_seals} / ${d.has_090_seals}</td>
        </tr>
      `).join("");

      // 7. Layer 12 Paleographic Chains & Crescent Ladder
      document.getElementById("paleoChains").innerHTML = `
        <div style="padding:12px 14px;border-radius:12px;background:var(--em-clay-soft);border:1px solid var(--em-border)">
          <div style="font-weight:600;color:var(--em-clay);margin-bottom:6px">1) Taşıyıcı Kayması (vis=0.88 · BH-FDR q = 0.0083*)</div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
            ${renderSeqInline("156", cp)} <span class="mono">sıra 0.51 · DKG(S) -15.05 ft</span>
            <strong style="color:var(--em-clay)">→</strong>
            ${renderSeqInline("154", cp)} <span class="mono">sıra 0.86 · DKG(S) -7.55 ft (p=0.0021)</span>
          </div>
        </div>
        <div style="padding:12px 14px;border-radius:12px;background:var(--em-success-soft);border:1px solid var(--em-border)">
          <div style="font-weight:600;color:var(--em-success);margin-bottom:6px">2) 3 Aşamalı Filizlenme / Yaprak Evrimi (vis=0.87–0.96 · BH-FDR q = 0.0339*)</div>
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            ${renderSeqInline("838", cp)} <span class="mono">(0.22)</span> <strong>→</strong>
            ${renderSeqInline("803", cp)} <span class="mono">(0.52)</span> <strong>→</strong>
            ${renderSeqInline("806", cp)} <span class="mono">(0.67 · Harappa #1 Geç q=0.0106*)</span>
          </div>
        </div>
      `;
      const cr = B.layer12_time.calendar_and_crescent_audit.crescent_progression_non_tabc;
      document.getElementById("crescentBody").innerHTML = ["900", "904", "923"].map(sid => {
        const c = cr[sid];
        return `<tr>
          <td class="mono"><strong>#${sid}</strong> · ${c.label}</td>
          <td>${renderSeqInline(sid, cp)}</td>
          <td class="mono">${c.dedup_non_tabc_count}</td>
          <td class="mono"><strong>%${c.preceded_by_fish_pct}</strong></td>
          <td class="mono">%${c.preceded_by_numeral_pct}</td>
          <td class="mono"><strong>%${c.followed_by_740_pct}</strong></td>
          <td class="mono">${c.md_within_area_med_rel_time}</td>
        </tr>`;
      }).join("");
    });
  </script>
</body>
</html>
"""
    (PORTAL_DIR / "index.html").write_text(portal_html, encoding="utf-8")


def md_to_print_html(md_text: str) -> str:
    """Minimal deterministic Markdown-to-HTML converter for our clean preprint structure."""
    lines = md_text.splitlines()
    out = []
    in_ul = False
    in_ol = False
    in_code = False
    in_table = False
    table_rows: list[list[str]] = []

    def inline_fmt(txt: str) -> str:
        txt = html.escape(txt)
        import re
        txt = re.sub(r"`([^`]+)`", r"<code>\1</code>", txt)
        txt = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", txt)
        txt = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", txt)
        return txt

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        out.append("<table><thead><tr>" + "".join(f"<th>{inline_fmt(c)}</th>" for c in table_rows[0]) + "</tr></thead><tbody>")
        for row in table_rows[2:]:
            out.append("<tr>" + "".join(f"<td>{inline_fmt(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table>")
        in_table = False
        table_rows = []

    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            flush_table()
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            continue
        if in_code:
            out.append(html.escape(ln))
            continue
        if s.startswith("|") and s.endswith("|"):
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            cells = [c.strip() for c in s.strip("|").split("|")]
            in_table = True
            table_rows.append(cells)
            continue
        else:
            flush_table()
        if not s:
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            continue
        if s == "---":
            out.append("<hr/>")
        elif s.startswith("### "):
            out.append(f"<h3>{inline_fmt(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{inline_fmt(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{inline_fmt(s[2:])}</h1>")
        elif s.startswith("- "):
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline_fmt(s[2:])}</li>")
        elif len(s) > 3 and s[0].isdigit() and s[1:3] in (". ", ") ") or (len(s) > 4 and s[:2].isdigit() and s[2:4] == ". "):
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol: out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline_fmt(s.split('. ', 1)[1])}</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            out.append(f"<p>{inline_fmt(s)}</p>")
    flush_table()
    if in_ul: out.append("</ul>")
    if in_ol: out.append("</ol>")
    return "\n".join(out)


def build_preprint_pdf() -> Path:
    md_path = PAPER_DIR / "INDUS_DEDUCT_PREPRINT.md"
    md_text = md_path.read_text(encoding="utf-8")

    # Append Layer 11A & Layer 11B addendum directly to print HTML (and keep preprint synchronized!)
    body_html = md_to_print_html(md_text)
    cal_img = "../reports/calibration_5way_chart.png"
    l9_img = "../reports/layer9_copper_rosetta_plate.png"
    l12_img = "../reports/layer12_time_evolution_plate.png"

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>INDUS-DEDUCT Open Benchmark Preprint</title>
  <style>
    @page {{ size: A4; margin: 16mm 15mm 18mm 15mm; }}
    body {{
      font-family: "Georgia", "Iowan Old Style", serif;
      color: #182c22;
      background: #ffffff;
      font-size: 9.6pt;
      line-height: 1.48;
      margin: 0;
    }}
    h1 {{
      font-size: 16.5pt; line-height: 1.22; color: #182c22;
      border-bottom: 2px solid #315b41; padding-bottom: 8px; margin-bottom: 10px;
    }}
    h2 {{
      font-size: 12pt; color: #315b41; margin-top: 16px; margin-bottom: 6px;
      border-bottom: 1px solid #dfe5dc; padding-bottom: 3px; page-break-after: avoid;
    }}
    h3 {{ font-size: 10.3pt; color: #182c22; margin-top: 11px; margin-bottom: 4px; page-break-after: avoid; }}
    p {{ margin: 0 0 7px; text-align: justify; }}
    ul, ol {{ margin: 4px 0 8px 18px; padding: 0; }}
    li {{ margin-bottom: 3px; }}
    code {{
      font-family: "Menlo", "JetBrains Mono", monospace; font-size: 8.3pt;
      background: #eef2eb; padding: 1px 4px; border-radius: 3px; color: #182c22;
    }}
    pre {{
      background: #f6f7f3; border: 1px solid #dfe5dc; border-radius: 6px;
      padding: 8px 10px; font-size: 8pt; white-space: pre-wrap; word-break: break-word; page-break-inside: avoid;
    }}
    pre code {{ background: transparent; padding: 0; }}
    table {{
      width: 100%; border-collapse: collapse; margin: 9px 0 12px;
      font-family: "Avenir Next", -apple-system, sans-serif; font-size: 8pt;
      page-break-inside: avoid;
    }}
    th, td {{ border: 1px solid #dfe5dc; padding: 4.5px 6px; text-align: left; }}
    th {{ background: #eef2eb; color: #182c22; font-weight: 600; }}
    hr {{ border: none; border-top: 1px solid #dfe5dc; margin: 12px 0; }}
    .figure {{
      margin: 14px 0; padding: 8px; border: 1px solid #dfe5dc; border-radius: 6px;
      background: #fafaf7; page-break-inside: avoid; text-align: center;
    }}
    .figure img {{ width: 100%; max-height: 220mm; object-fit: contain; }}
    .figcap {{
      font-family: "Avenir Next", sans-serif; font-size: 8pt; color: #43554a;
      margin-top: 6px; text-align: left;
    }}
  </style>
</head>
<body>
  {body_html}
  <div class="figure">
    <img src="{cal_img}" alt="Layer 8 Five-Way Calibration" />
    <div class="figcap"><strong>Figure 1 (Layer 8).</strong> Five-way matched calibration comparing Indus Strict against real Ur III Sumerian (CDLI), Linear B, Linear A, and structured Non-Linguistic Stamps across retired (z_H1) and validated discriminators (Unseen-Trigram generalization, Kober Bridge–Ending MI z, 2-Affix stacking) plus 23-seal Known-Answer Max-CSP validation.</div>
  </div>
  <div class="figure">
    <img src="{l9_img}" alt="Layer 9 Internal Rosetta and Meluhha Trade Plate" />
    <div class="figcap"><strong>Figure 2 (Layer 9).</strong> Actual ICIT TTF glyph plate illustrating Mohenjo-daro Copper Tablet (TAB:C) Text-to-Icon locks &amp; single-sign reverse equations (#341 quadruped pictograph, #753 Hare, #777–782 class vessels), Lothal multi-signature cargo co-signing with Mohenjo-daro stone seal #2239.1, Harappa double-sided ration tokens (: X : + 400 stacking), and global BH-FDR trade/material enrichments (#091 Twin-Humans, #790 Ring).</div>
  </div>
  <div class="figure">
    <img src="{l12_img}" alt="Layer 12 Stratigraphic Time Evolution Plate" />
    <div class="figcap"><strong>Figure 3 (Layer 12).</strong> 700-year de-confounded stratigraphic time evolution across Mohenjo-daro (within-area rank-normalized depth n=1,253 &amp; single-datum DKG(S) n=595) and Harappa (Period 2–5, Spearman rho=+0.289, p=0.0376), displaying the two global BH-FDR paleographic drift chains (#156→#154 q=0.0083; 3-stage sprouting chain #838→#803→#806 q=0.0339), frozen SEAL:S core grammar quartiles, and the four-axis Crescent ladder (#900→#904→#923).</div>
  </div>
</body>
</html>
"""
    html_out = PAPER_DIR / "INDUS_DEDUCT_PREPRINT.html"
    pdf_out = PAPER_DIR / "INDUS_DEDUCT_PREPRINT.pdf"
    html_out.write_text(full_html, encoding="utf-8")

    subprocess.run(
        [
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_out}",
            html_out.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.copy2(pdf_out, PORTAL_DIR / "assets" / "INDUS_DEDUCT_PREPRINT.pdf")
    return pdf_out


def main() -> None:
    bundle = build_portal_bundle()
    write_portal_html(bundle)
    pdf_path = build_preprint_pdf()
    print(f"Portal ready: {PORTAL_DIR / 'index.html'} (atlas={len(bundle['atlas_signs'])} glyphs, catalog={len(bundle['restoration_catalog'])} damaged inscriptions)")
    print(f"PDF Preprint compiled: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
