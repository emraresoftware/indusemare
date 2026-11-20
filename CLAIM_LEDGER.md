# INDUS-DEDUCT Executive Claim Ledger (`v1.0.0-preprint`)

> **Referee Policy:** Every claim in `INDUS-DEDUCT` is assigned to exactly one epistemological tier. No Tier B comparative lead is ever presented as a Tier A proof, and every Tier C negative/retired result is preserved openly in automated tests (`tests/test_claim_ledger.py`, 11/11 PASS).

---

## Tier A — Proven Structural, Stratigraphic & Mathematical Findings *(Machine-Verified, BH-FDR / Permutations / Cross-OS SHA-256)*

| # | Claim | Key Quantitative Proof | Automated Test & Artifact |
|---|---|---|---|
| **A1** | Canonical corpus has zero speculative contamination; both `R/L` and `L/R` ICIT rows are stored in canonical visual order. | `5,679` inscriptions, `5,805` lines; Jar `740` leftmost in `29.91%` (`R/L`) & `24.54%` (`L/R`). | `test_01`, `test_03` · `reports/layer0_audit.json` |
| **A2** | Functional core inventory is **logo-syllabic** (pure syllabary ruled out). | `N_core = 240–271` (`223–254` excl. PUA compounds) covering `95.7%` tokens; allograph collapse beats random merges by `5.77` PPL. | `reports/layer1b_visual_allographs.json` |
| **A3** | Indus separates cleanly from structured non-linguistic stamps and clusters with attested natural scripts on 3 validated discriminators. | Unseen-trigram: Indus `+6.38 pp` / Sumer `+10.71 pp` vs. Non-Ling `-2.95 pp`. Kober `MI z`: Indus `+53.67σ` / Sumer `+20.40σ` / Linear B `+18.47σ` vs. Non-Ling `+0.07σ`. 2-affix stacking: Indus `207:1` vs. Non-Ling `0:0`. | `test_07` · `reports/layer8_calibration.json` |
| **A4** | Productive sub-graphemic grammar: `: X :` four-dash frame converts medial stems to terminals zero-shot on unseen base glyphs; `741–747` rim ticks invert finality. | `: X :` `2.9% → 62.2%` final (8/8 pairs, `p = 4.38e-45`); LOBO `Δlog-loss = -0.0687 (-31.3%)`. Rim ticks `71.9% → 2.4%` (`p = 3.43e-88`). | `test_06` · `reports/layer5b_expanded_phonetic.json`, `reports/layer7_council_audit.json` |
| **A5** | Strict Leave-One-City-Out (LOCO) damaged-inscription restoration strongly beats position baselines; visual-neural hybrid beats N-gram. | 12-city LOCO (`n = 275`): Hybrid `57.5%` Top-1 / `82.9%` Top-5 vs. N-gram `53.5%` / `79.6%` (baseline `10.2%`). Interior (`n = 10`): `40.0%` Top-1 / `60.0%` Top-3. `690`-inscription open catalog across 25 sites. | `test_06`, `test_10` · `reports/layer11b_neural_restoration.json` |
| **A6** | Mohenjo-daro Copper Tablets (`TAB:C`, 99.6% MD) lock Text ↔ Icon (`+19.66σ`, `96.2%` purity) with 23 single-sign reverse equations (`#341` quadruped ↔ `503 615 740`). | `TAB:C` `MI z = +18.39σ` (named `+19.66σ`) vs. `SEAL:S` `+2.79σ`. | `test_08` · `reports/layer9_internal_rosetta_trade.json` |
| **A7** | Harappa double-sided tokens (`TAB:I/B`, 772 objects) encode fixed Entity ↔ Quantity quotas in the **same** suffix grammar (`+ 400` clitic). | `Entity ↔ Qty MI z = +9.55σ` (`p = 0.001`); `: X : + 400` stacking `20/21 = 95.2%` (`p = 1.16e-8`). | `test_08` · `reports/layer9_internal_rosetta_trade.json` |
| **A8** | Inter-city logistics & maritime/export diaspora split: Twin-Humans `#091` is an export/cargo header (`0%` final); Ring `#790` (`○`) enriches stoneware bangles. | 26 multi-sig clay lumps; Lothal master counter-sign co-signs Mohenjo-daro stone seal `861 002 235`. `#091`: `11.89×` trade lift (`p = 1.48e-6`), `42.35×` foreign (`q = 2.86e-4`); foreign square `0/12` vs. circular `11/17` `#090/#091`. `#790`: `9.13×` on `BNGL` (`q = 0.032`). | `test_08`, `test_09`, `test_10` · `reports/layer9..11a*.json` |
| **A9** | 700-year paleographic evolution chains survive global BH-FDR & cross-city MD–Harappa replication while `SEAL:S` core grammar remains frozen. | `#156 → #154` (`q = 0.0083`; `DKG(S)` `-15.05 → -7.55 ft, p = 0.0021`); `#838 → #803 → #806` (`q = 0.0339`, `#806` Harappa #1 late `q = 0.0106`); MD–Harappa sync `rho = +0.289 (p = 0.0376)`; `SEAL:S` quartiles `p > 0.09`. | `test_11` · `reports/layer12_stratigraphic_time.json` |
| **A10** | The Gulf phonetic decipherment barrier is a physical anchor/open-vocab limit, not solver blindness. | Known-answer 23-seal CSP locks `23/23` closed Linear B (`+22.75σ`) & Sumerian (`+7.00σ`) at `k = 0`, fails on 74% open Sumerian at `k = 0` (`z = -0.11σ`), rescues at `k = 2` (`+4.97σ`). Foreign Indus expansion `24 → 55` increases `DOF` (`29 → 45`). | `test_05`, `test_07`, `test_10` · `reports/layer4b/8/11a*.json` |

---

## Tier B — Strongly Supported Comparative / Typological Leads *(Not Yet Conclusive Proof)*

| # | Hypothesis | Supporting Evidence | Remaining Gap to Tier A |
|---|---|---|---|
| **B1** | **Proto-Dravidian** is the strongest language-family candidate for the Indus elite script. | #1 typology (`0.925`), `207:1` exclusively suffixing (disfavors prefixing Para-Munda), Rakhigarhi aDNA (`0%` Steppe), expanded 13-probe Mode 2B `z = +6.86` (13/13 LOPO `> 5.0`), Brahui & Rigvedic retroflex/agricultural substrates. | Strict single-primary-lemma Mode 1 scores `z = +3.18` (`p = 0.004`), below `z > 5.0` gate. |
| **B2** | Carrier `#156` (`*kā`?) & Fish `#220` (`*mīn`?) visual-rebus roots. | Only two exact primary roots passing Mode 1; `#156` co-occurs `2.8×` with vessel `#700` on opposite faces (`34.0%` vs. `12.2%`) and sits on small-rank seals (`467 mm²`). | Independent phonetic confirmation via `k ≥ 2` external/bilingual anchors. |
| **B3** | Crescent ladder (`#900 → #904 → #923`) encodes a structured cyclic/celestial modifier series. | Monotonic 4-axis gradient (`Fish 1.4% → 22.9% → 44.8%`, `740 31.9% → 45.8% → 89.7%`, `Numeral 43.1% → 37.5% → 13.8%`, `MD time 0.62 → 0.44 → 0.37`). | No explicit date/month/year formula attested yet on short 4–5 sign seals. |

---

## Tier C — Explicitly Retired Metrics, Falsified Hypotheses & Honest Nulls

| # | Retired / Falsified Claim | Empirical Falsification | Source |
|---|---|---|---|
| **C1** | Within-sequence conditional entropy `z_H1` proves natural language. | **RETIRED:** Non-Linguistic Stamps produce `-245.34σ` (more than 2× Indus `-108.85σ`). Raw Kober `alt/same` ratio likewise retired (`Non-Ling 2.41 > Linear B 1.88`). | `test_06`, `test_07` |
| **C2** | Published Tamil Nadu megalithic graffiti overlap proves direct Indus continuity on pottery. | **FALSIFIED (at complexity control):** Removing universal vertical tally strokes (`031–035`, `58.3%`) collapses the remaining 17 geometric signs to `OR = 1.16, p = 0.124` (`z = +0.30`). | `reports/layer3g_faunatech_external.json` |
| **C3** | Mohenjo-daro Copper Tablet (`TAB:C`) texts are bare animal names (`H1`). | **FALSIFIED (4 proofs):** Mean length `4.79`, polysemous reverse vessel marks (`777/781/782` shared by `Anth` & `Goat/Buff`), `0/11` `SEAL:S` animal retention, and flat null across all 4 language families (`p > 0.30`). | `test_09` · `reports/layer10_copper_rosetta_deep.json` |
| **C4** | Lothal and Dholavira have unique port-specific signs on `SEAL+TAG`. | **FALSIFIED (duplicate-stamp artifact):** After strict deduplication, both drop from illusory `p = 1e-10` to **`0` BH-FDR survivors**. | `test_08` · `reports/layer9_internal_rosetta_trade.json` |
| **C5** | Harappan tablets (`TAB*`) are primarily open-sea maritime records. | **FALSIFIED:** `98.9%` (`2,243/2,290`) occur in inland river capitals (Lothal dockyard has `3`), carrying strictly freshwater icons (`Gavi` `62/66`, `Fish` `47/47`). Maritime logistics live on `TAG*` (`65.3%`) & `SEAL:C` (`~66%`). | Layer 12 / Preprint §8.4 |
| **C6** | Any single Indus sign has a proven phonetic value today (`k = 0` Gulf CSP or Mode 1). | **ZERO PROVEN PHONETIC READINGS:** `0/53` Gulf signs locked (`p = 1.000` synthetic / `p = 0.188` CDLI Ur III); Mode 1 `< 5.0σ` across all 4 families. | `test_04`, `test_05`, `test_07` |
