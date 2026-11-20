# INDUS-DEDUCT (`v1.0.0-preprint`)

**A Calibrated, Anti-Hallucination Computational Benchmark for the Indus Valley (Harappan) Script**

[![Live Portal](https://img.shields.io/badge/live%20portal-indus.emare.dev-315b41)](https://indus.emare.dev/?lang=en)
[![Tests](https://img.shields.io/badge/claim--ledger-11%2F11%20PASS-2b6142)]()
[![Cross-OS](https://img.shields.io/badge/repro-macOS%20%26%20Linux%20SHA--256-315b41)]()
[![License](https://img.shields.io/badge/license-MIT%20%2F%20CC--BY--4.0-182c22)]()

> **Live Interactive Explorer (EN / TR):** [**https://indus.emare.dev/?lang=en**](https://indus.emare.dev/?lang=en) · **PDF Preprint:** [`INDUS_DEDUCT_PREPRINT.pdf`](paper/INDUS_DEDUCT_PREPRINT.pdf) · **1-Page Claim Ledger:** [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md)

`INDUS-DEDUCT` is an end-to-end, reproducible 13-layer computational epigraphy benchmark across **5,679 Indus inscriptions** (`5,805` parsed lines; `713` raw signs). Every pipeline stage obeys a strict anti-hallucination contract:
- **Zero speculative upstream leakage:** Upstream `sanskrit`, `translation`, and `notes` columns are stripped prior to modeling and guarded by `test_01_zero_speculative_contamination`.
- **Calibrated against real scripts & negative controls:** Benchmarked at matched size and length (`N = 2,047`) against **Ur III Sumerian (CDLI)**, **Linear B**, **Linear A**, and a **structured non-linguistic stamp negative control**.
- **Zero fabricated phonetic readings:** We explicitly separate verified structural/stratigraphic proofs (Tier A) from comparative/typological leads (Tier B) and retired/falsified claims (Tier C). See [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md).

---

## Key Deliverables

| Deliverable | Path | Description |
|---|---|---|
| **Full Academic Preprint (PDF)** | [`paper/INDUS_DEDUCT_PREPRINT.pdf`](paper/INDUS_DEDUCT_PREPRINT.pdf) | 11-page peer-review-ready preprint with 3 embedded high-res figures (Layers 0–12) |
| **Executive Claim Ledger** | [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md) | 1-page Tier A (Proven) / Tier B (Supported) / Tier C (Retired/Null) referee map |
| **Interactive Web Portal** | [`portal/index.html`](portal/index.html) | 6-tab live FaunaTech Doğal Atlas v2.0 browser (294 live TTF glyphs, 690 restorations, calibration & stratigraphy) |
| **Open Damaged Restoration Catalog** | [`reports/damaged_inscriptions_restoration_catalog.csv`](reports/damaged_inscriptions_restoration_catalog.csv) | Calibrated Top-5 restorations for all 690 single-gap damaged inscriptions across 25 sites |
| **Automated Verification Suite** | [`tests/test_claim_ledger.py`](tests/test_claim_ledger.py) | 11/11 machine-verified invariants + SHA-256 integrity (`DATA_MANIFEST.json`) |

---

## Headline Verified Findings (Tier A)

1. **Five-Way Calibration (Layer 8):** Retires within-sequence conditional entropy `z_H1` (`Non-Linguistic Stamps = -245.34σ` vs. `Indus = -108.85σ`) and establishes three validated discriminators separating Indus and attested natural scripts from structured non-linguistic stamps:
   - **Unseen-trigram generalization** over position baseline: `Indus +6.38 pp (+0.432 bits)`, `Ur III Sumerian +10.71 pp`, `Linear B +0.81 pp` vs. `Non-Linguistic -2.95 pp (-0.960 bits)`.
   - **Kober bridge–ending `MI z`:** `Indus +53.67σ`, `Ur III Sumerian +20.40σ`, `Linear B +18.47σ` vs. `Non-Linguistic +0.07σ`.
   - **Blind 2-affix stacking (`Suffix:Prefix`):** `Indus 207:1` (`4.96×` right-edge bottleneck) vs. `Ur III Sumerian 4:39`, `Linear B 22:11`, `Non-Linguistic 0:0`.
2. **Logo-Syllabic Core & Sub-Graphemic Grammar (Layers 1B, 5B, 7):** `N_core = 240–271` graphemes (`95.7%` tokens); `: X :` four-dash frame converts medial stems (`2.9%`) to terminals (`62.2%`, 8/8 pairs, `p = 4.38e-45`) and transfers zero-shot across unseen base glyphs (`LOBO Δlog-loss = -31.3%`).
3. **700-Year Paleographic Drift & Cross-City Sync (Layer 12):** De-confounding Mohenjo-daro excavation datums (`n = 1,253` within-area; `DKG(S)` `n = 595`) and replicating against Harappa `Period 2–5` (`n = 637`, `Spearman rho = +0.289, p = 0.0376`) uncovers two BH-FDR paleographic drift chains: **`#156 → #154` (`q = 0.0083`)** and the 3-stage sprouting chain **`#838 → #803 → #806` (`q = 0.0339`, `#806` #1 late sign in Harappa `q = 0.0106`)**, while core `SEAL:S` grammar remains frozen across 700 years (`p > 0.09`).
4. **Document Registries & Maritime vs. Riverine Logistics (Layers 9–12):**
   - **Mohenjo-daro Copper Tablets (`TAB:C`, 99.6% MD-exclusive):** `Text ↔ Icon MI z = +19.66σ` (`96.2%` purity) + 23 single-sign reverse equations (`#341` quadruped `↔ 503 615 740`), yet 4 tests refute bare animal nouns (`H1`) in favor of compound institutional/ritual formulas (`H2`).
   - **Harappa Double-Sided Tokens (`TAB:I/B`, 772 objects):** Fixed `Entity ↔ Quantity` lock (`+9.55σ`) and `: X : + 400` clitic stacking (`95.2%`, `p = 1.16e-8`).
   - **Riverine Tablets vs. Maritime Cargo:** `98.9%` of tablets occur inland (enriched in freshwater `Gavi` gharial `62/66` and `Fish` `47/47`), whereas `65.3%` of cargo sealings (`TAG*`) and `~66%` of circular Gulf seals (`SEAL:C`) are coastal/maritime — anchored by Twin-Humans **`#091`** (`0%` final, `11.89×` trade/cargo lift `p = 1.48e-6`, `42.35×` in foreign corridor `q = 2.86e-4`).
5. **Exact Phonetic Decipherment Boundary (Layers 4B, 5B, 8, 11A):** Known-answer 23-seal Max-CSP locks `23/23` (`100%`) of closed Linear B (`+22.75σ`) and Ur III Sumerian (`+7.00σ`) at `k = 0`, fails at `k = 0` under realistic 74% open Sumerian (`z = -0.11σ`), and recovers `21/23` (`+4.97σ`, `68.2%` shared signs) at **`k = 2` bilingual anchors**. Expanding foreign Indus seals `24 → 42 → 55` strictly increases `DOF` (`29 → 43 → 45`), proving `k ≥ 2` bilingual anchors are required.
6. **Leak-Free Visual-Neural LOCO Damaged Restoration (Layer 11B):** Across `285` real damaged inscriptions held out over `12` cities (`STRICT_LOCO`), our visual-glyph + neural MLP hybrid improves `edge_all` Top-1 from `53.5% → 57.5%` (`Top-5 82.9%`, baseline `10.2%`) and `interior_all` Top-1 from `30.0% → 40.0%` (`Top-3 60.0%`).

---

## Quickstart & Full Reproduction

### 1. Zero-Dependency Claim-Ledger Check (Works on Fresh Clone, No `pip` Required)

```bash
git clone https://github.com/emraresoftware/indusemare.git
cd indusemare
python3 -m unittest tests/test_claim_ledger.py -v
```

This immediately verifies all analytical invariants across Layers 0–12 (`reports/*.json`) and SHA-256 checksums for bundled `data/external/*` tables. To respect primary epigraphic concordance rights (ICIT / CISI), raw upstream corpus dumps are not redistributed inside the repository (`DATA_MANIFEST.json` records their exact SHA-256 hashes for full local verification).

### 2. Full End-to-End Pipeline Reproduction (with Local Upstream Corpora)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Place upstream raw corpora under data/raw/ and data/calibration/ (verified via DATA_MANIFEST.json)
.venv/bin/python src/layer0_ingest.py

# Strict 11/11 verification including all 17 DATA_MANIFEST.json SHA-256 hashes + SQLite checks
INDUS_STRICT_CORPUS=1 python3 -m unittest tests/test_claim_ledger.py -v

# Launch the bilingual (EN / TR) interactive FaunaTech Doğal Atlas portal locally
python3 -m http.server 4188 --bind 127.0.0.1 --directory portal
```

## Upstream Data Attribution
- **Indus Corpus:** Derived from the Interactive Corpus of Indus Texts (ICIT; Bryan K. Wells & Andreas Fuls) and Corpus of Indus Seals and Inscriptions (CISI; Asko Parpola et al.), stripped of all speculative columns in Layer 0.
- **Calibration Corpora:** Ur III Sumerian gold corpus (CDLI / MTAAC), Linear B transliterated words/tablets, and Linear A open corpus (`LinearAWords.json`).
