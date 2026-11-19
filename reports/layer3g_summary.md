# INDUS-DEDUCT — Layer 3G: FaunaTech Ecological Traits, Tamil Nadu Megalithic Graffiti Visual-Complexity Control, & West Asian Seals Audit

> **Epistemic Standard (Brutally Honest):** Every claim below explicitly separates **unconfounded structural signal** from **artifact-type and visual-complexity confounding** using 1,000-permutation null models, Cochran-Mantel-Haenszel stratification, Leave-One-Species-Out validation, and exact MLE logistic regression.

## Executive Verdicts (At a Glance)

| Workstream | Brutally Honest Verdict | Core Verified Numbers |
|---|---|---|
| **EKSIK #3 — FaunaTech 18-Motif Trait Alignment** | **PARTIAL / CONFOUNDER-SEPARATED** (Naive global `p=0.001` is heavily confounded by artifact medium `V=0.331`; **3 unconfounded cross-species/within-stratum signals DO survive!**) | Object-stratified `body_size` `z=+3.96, p=0.001`; `TAB:B/I`-only all 5 traits `p=0.001`; **Species-level obj-residualized `domestic_herd` (`Zebu/Bull/Bult/Goat`) `Δ_sim=+0.360, p=0.001`**; `SEAL:S` `body_size` species-level `p=0.025`; LOSO `social_behavior` `83.3% vs 49.7%, p=0.012`; CMH `megafauna` `440` (`6.63x, p=6.80e-6`) & `621` (`7.30x, p=1.91e-4`); global 5-trait obj-resid Mantel `r=0.126, p=0.070` (borderline). |
| **EKSIK #2A — Tamil Nadu Megalithic Graffiti Test** | **REAL POTTERY CONCENTRATION (`2.61x`), BUT DOMINATED BY TALLY STROKES (`58.3%`) & LOW-STROKE SCRATCHABILITY** | Raw `M_shared (22)` `POT:T:g` vs `SEAL:S` `OR=2.61x (p=4.36e-36)`; **splitting out vertical tallies `031-035` (`5.05x, p=1.38e-57`) collapses the remaining 17 non-tally geometric shapes (`M17`) to `OR=1.16x (p=0.124)` & complexity-matched 1,000-perm null `perm_p=0.388`!** `M17` shows a textbook monotone Q1→Q4 complexity gradient (`1.57x → 1.11x → 0.93x → 0.57x`) and minimal pairs favor simple forms (`480` `19.4x` vs `482` `1.04x`; `850` `2.64x` vs `861` `0.68x`; `790` `2.79x` vs `820` `0.42x`). |
| **EKSIK #2B — Published Gulf / Mesopotamia Seals Audit** | **CANNOT REACH LAYER 4B `m*=50` THRESHOLD FROM EXISTING LITERATURE** | Our DB has `41` foreign rows / `E=23` Gulf circular-seal graph vs ~`28` inscribed Gulf seals in Laursen (2010) & ~`30-40` total West Asia seals globally (`CISI 3.4` forthcoming). Adding ALL ~`5-15` remaining published seals reaches at most `E=28-38` — still **`35-45` seals short** of `m*=50` (`73` total) even with `k=2` anchors, and **`>165` short** at `k=0`. |

## Part 1 — EKSIK #3: FaunaTech Ecological & Anatomical Trait Alignment (18 Indus Fauna Motifs)

- **FaunaTech Schema Alignment:** Verified read-only against Emare's live FaunaTech v2.0 codebase (`emare-dev-91:/var/www/elraenn/current/`: `FaunaTech-Spec.md`, `python/build_full_catalog.py`, `python/faunatech_engine.py`, `js/species-groups.js`). Saved aligned 18-motif biological matrix to `data/external/faunatech_indus_traits.json`.
- **Corpus Scope:** `2552` non-empty motif lines total; **`2392` lines** (`2318` unique inscriptions, **`2034`** deduplicated `(motif, object_type, seq)` entries) across all 18 fauna motifs (`SEAL:S` = `1554`, `TAB:B/I` = `306`).
- **Critical Confounder Audit (Motif × Object Type):** **Cramér's V = `0.3312`** (`chi2 = 4460.45`, `p = 0.0`). `Gavi` (94% `TAB:B/I`), `Fish` (100% `TAB:B/I`), `Phyt` (93% `TAB:B/I`), and `Hare` (92% `TAB:C`) occur almost exclusively on tablets, whereas `Bull1` (90%), `Zebu` (91%), `Tigr` (87%), and `Buff` (86%) occur on `SEAL:S`. Naive unstratified trait tests therefore confound ecology with artifact medium!

### 1A. Line-Level 1,000-Permutation Trait Tests (Raw vs Object-Stratified vs Within-`SEAL:S` vs Within-`TAB:B/I`)

| FaunaTech Dimension | Unstratified `z` (`p`) | Object-Type-Stratified `z` (`p`) | Strictly `SEAL:S` Only `z` (`p`) | Strictly `TAB:B/I` Only `z` (`p`) | 5-Fold CV BalAcc (`SEAL:S` vs Chance) |
|---|---:|---:|---:|---:|---:|
| `habitat` | `+12.10` (`p=0.001`) | **`-0.41` (`p=0.62438`)** | **`-1.24` (`p=0.92308`)** | `+6.43` (`p=0.001`) | `0.3399` vs `0.25` |
| `body_size` | `+9.77` (`p=0.001`) | **`+3.96` (`p=0.001`)** | **`+1.28` (`p=0.11089`)** | `+4.96` (`p=0.001`) | `0.4518` vs `0.3333` |
| `morphology` | `+5.00` (`p=0.001`) | **`+1.67` (`p=0.06194`)** | **`-0.17` (`p=0.52547`)** | `+5.17` (`p=0.001`) | `0.2184` vs `0.1667` |
| `trophic_role` | `+7.26` (`p=0.001`) | **`+1.72` (`p=0.05095`)** | **`+0.62` (`p=0.24875`)** | `+4.18` (`p=0.001`) | `0.2723` vs `0.25` |
| `social_behavior` | `+5.02` (`p=0.001`) | **`+1.74` (`p=0.05594`)** | **`-0.36` (`p=0.63437`)** | `+4.45` (`p=0.001`) | `0.54` vs `0.5` |

### 1B. Species-Level 1,000-Permutation Tests (HOLDING INDIVIDUAL SPECIES & OBJECT TYPE CONSTANT)

- **18×18 Species Sign-Similarity vs FaunaTech Gower Trait-Similarity Mantel Test (1,000 species permutations):**
  - All 18 species (Raw TF-IDF): `r = 0.1393`, `rho = 0.138`, `z = +1.84`, **`p = 0.04795`**
  - All 18 species (**Object-Type-Residualized**): `r = 0.1259`, `rho = 0.1357`, `z = +1.55`, **`p = 0.06993`**
  - Strictly `SEAL:S`-only species (`n = 11` species with ≥7 seals): `r = 0.1459`, `rho = 0.1621`, `z = +0.94`, **`p = 0.18581`**

| Biological Dimension / Niche | Within-vs-Between Species `Δ_sim` (Obj-Resid) | Species-Perm `p` (Obj-Resid) | Within-vs-Between `Δ_sim` (`SEAL:S`-Only) | Species-Perm `p` (`SEAL:S`) | LOSO Acc (Obj-Resid vs Null, `p`) |
|---|---:|---:|---:|---:|---:|
| `habitat` | `-0.0009` | `0.50649` | `0.0664` | `0.13087` | `0.3333` vs `0.2136` (`p=0.21479`) |
| `body_size` | `0.0112` | `0.32667` | `0.092` | `0.02498` | `0.4444` vs `0.3391` (`p=0.28172`) |
| `morphology` | `0.089` | `0.05894` | `-0.0615` | `0.76923` | `0.0909` vs `0.0512` (`p=0.3986`) |
| `trophic_role` | `0.0548` | `0.05495` | `0.0255` | `0.24376` | `0.5` vs `0.2524` (`p=0.04296`) |
| `social_behavior` | `0.0247` | `0.18681` | `0.01` | `0.41658` | `0.8333` vs `0.497` (`p=0.01199`) |

**Binary Niche Species-Pair Clustering (`Obj-Residualized` 1,000 species-label permutations):**
- **`aquatic`** (`3` species, `3` cross-species pairs): Raw `Δ_sim=-0.0123` (`p=0.61838`) → Obj-Resid `Δ_sim=0.0695` (`p=0.27672`)
- **`megafauna`** (`4` species, `6` cross-species pairs): Raw `Δ_sim=0.0796` (`p=0.28971`) → Obj-Resid `Δ_sim=0.0758` (`p=0.14585`)
- **`horned_bovid`** (`5` species, `10` cross-species pairs): Raw `Δ_sim=0.0662` (`p=0.32468`) → Obj-Resid `Δ_sim=0.1089` (`p=0.06793`)
- **`solitary_apex`** (`2` species, `1` cross-species pairs): Raw `Δ_sim=0.0667` (`p=0.4046`) → Obj-Resid `Δ_sim=0.1772` (`p=0.15285`)
- **`domestic_herd`** (`4` species, `6` cross-species pairs): Raw `Δ_sim=0.019` (`p=0.53946`) → Obj-Resid `Δ_sim=0.3598` (`p=0.001`)
- **`mythic_composite`** (`5` species, `10` cross-species pairs): Raw `Δ_sim=0.0791` (`p=0.25874`) → Obj-Resid `Δ_sim=-0.0304` (`p=0.62837`)

### 1C. Forensic Check: Aquatic `Gavi`+`Fish` Inside Tablets (`TAB:B/I`) & CMH Object-Controlled Sign Anchors

- Sign `700` strictly inside `TAB:B/I`: Aquatic (`Gavi`/`Fish`) = `14/69 (20.3%)` vs Non-Aquatic Motifs on Tablets = `31/237 (13.1%)` (`OR = 1.69x`, `p = 0.09996`)
- Sign `400` strictly inside `TAB:B/I`: Aquatic (`Gavi`/`Fish`) = `24/69 (34.8%)` vs Non-Aquatic Motifs on Tablets = `46/237 (19.4%)` (`OR = 2.21x`, `p = 0.007193`)
- Sign `690` strictly inside `TAB:B/I`: Aquatic (`Gavi`/`Fish`) = `5/69 (7.2%)` vs Non-Aquatic Motifs on Tablets = `3/237 (1.3%)` (`OR = 6.09x`, `p = 0.01624`)
- Sign `032` strictly inside `TAB:B/I`: Aquatic (`Gavi`/`Fish`) = `14/69 (20.3%)` vs Non-Aquatic Motifs on Tablets = `26/237 (11.0%)` (`OR = 2.07x`, `p = 0.03839`)
- Sign `034` strictly inside `TAB:B/I`: Aquatic (`Gavi`/`Fish`) = `3/69 (4.3%)` vs Non-Aquatic Motifs on Tablets = `4/237 (1.7%)` (`OR = 2.65x`, `p = 0.1925`)

- **`aquatic`** (`n=95` dedup lines; `3` signs survive CMH object-type control at `p<0.05`): `690` (CMH OR=5.6265x, p=0.0055607), `017` (CMH OR=5.4811x, p=0.034276), `400` (CMH OR=1.9497x, p=0.038057)
- **`megafauna`** (`n=141` dedup lines; `7` signs survive CMH object-type control at `p<0.05`): `440` (CMH OR=6.6267x, p=6.7961e-06), `621` (CMH OR=7.2992x, p=0.00019061), `400` (CMH OR=2.1018x, p=0.014472), `760` (CMH OR=2.3598x, p=0.016639)
- **`horned_bovid`** (`n=355` dedup lines; `4` signs survive CMH object-type control at `p<0.05`): `035` (CMH OR=5.8492x, p=0.0022433), `171` (CMH OR=5.1674x, p=0.0073101), `924` (CMH OR=23.421x, p=0.0083199), `055` (CMH OR=2.8128x, p=0.046371)
- **`solitary_apex`** (`n=61` dedup lines; `2` signs survive CMH object-type control at `p<0.05`): `440` (CMH OR=11.2797x, p=1.0435e-05), `400` (CMH OR=2.9453x, p=0.0018882)
- **`domestic_herd`** (`n=230` dedup lines; `4` signs survive CMH object-type control at `p<0.05`): `035` (CMH OR=8.1934x, p=0.00073569), `171` (CMH OR=7.3028x, p=0.003869), `924` (CMH OR=30.1838x, p=0.012007), `231` (CMH OR=2.3959x, p=0.042241)
- **`mythic_composite`** (`n=1440` dedup lines; `1` signs survive CMH object-type control at `p<0.05`): `705` (CMH OR=1.9719x, p=0.040616)

---
## Part 2 — EKSIK #2A: Tamil Nadu Megalithic Graffiti Quantitative Test & Visual Stroke-Complexity Confounder Control

- **Published External Benchmarks (`data/external/tamil_nadu_graffiti_published.json`):**
  - **B. B. Lal (1960, *Ancient India* No. 16):** Out of **61** surveyed pottery graffiti symbols, **`47 / 61` (`77.05%` of the full surveyed set)** are shared across Harappan-Chalcolithic and South Indian Megalithic horizons (within Lal's specific subset denominators, **89%** of Megalithic symbols trace back to Harappan-Chalcolithic precursors and **85%** of Harappan-Chalcolithic symbols continue into the Megalithic), with explicit non-phonetic caution.
  - **K. Rajan & R. Sivanantham (2025, TN State Dept. of Archaeology):** `15,184` potsherds across `140` Tamil Nadu sites (`42` base signs, `544` variants, `1,521` composites); authors report **~90%** potsherd broad similarity and **~60%** base-sign parallels. **Open Data Audit:** No machine-readable CSV/JSON/GitHub dataset has been officially released (print/Tamil Digital Library flipbook only).
- **Visual Glyph Stroke-Complexity Extraction:** Computed directly from `687` cleanly rendered TTF glyphs (`28` empty/`.notdef` `U+2047` excluded; typographic guide dots stripped, with raw sensitivity check).

### 2A. Raw Concentration on Harappan Pottery Graffiti (`POT:T:g`, n=539) vs Steatite Seals (`SEAL:S`, n=1,848)

| Sign Category | `POT:T:g` Share (All Tokens) | `SEAL:S` Share (All Tokens) | Raw OR (`POT` vs `SEAL`) | Fisher `p` | Dedup-Seq OR (`p`) |
|---|---:|---:|---:|---:|---:|
| **Megalithic-Shared All (22 signs)** | `40.59%` (`360`) | `20.72%` (`1532`) | **`2.6138x`** | `4.3632e-36` | `1.8945x` (`p=5.1742e-12`) |
| ↳ Vertical Tally Strokes (`031-035`, 5 signs) | `23.68%` (`210`) | `5.79%` (`428`) | **`5.0486x`** | `1.3831e-57` | `3.1311x` (`p=2.2128e-18`) |
| ↳ Non-Tally Geometric (`520,480,690,220,700,850,550...`, 17 signs) | `16.91%` (`150`) | `14.93%` (`1104`) | **`1.1596x`** | `0.12436` | `1.1612x` (`p=0.18074`) |
| **Complex Indus-Specific Named (44 signs: Jar, Humans, Diacritic Fish)** | `21.42%` (`190`) | `32.82%` (`2427`) | **`0.5579x`** | `1.2637e-12` | `0.5683x` (`p=4.0569e-09`) |
| Complex Indus + All PUA Compounds | `23.34%` (`207`) | `33.84%` (`2502`) | **`0.5952x`** | `1.3407e-10` | `0.6084x` (`p=1.2101e-07`) |

- **Direct Head-to-Head (`Megalithic-Shared 22` vs `Complex Indus-Specific 44`):** `POT:T:g` = `360 vs 190 (65.5% Megalithic)` vs `SEAL:S` = `1532 vs 2427 (38.7% Megalithic)` (**OR = `3.0016x`**, `p = 2.9775e-32`).

### 2B. Visual Stroke-Complexity Confounder Test (Is It Cultural Continuity or Simple Scratchable Shapes?)

- **Are Megalithic-shared signs visually simpler?** **YES:** `M_shared (22)` mean complexity `z = -0.399` vs `Complex Indus (44)` `z = 0.192` (Cliff's `δ = -0.357`, `p = 0.010029`) and vs all other `665` Indus signs `z = 0.013` (Cliff's `δ = -0.23`, `p = 0.033098`).
- **Does visual complexity predict `POT:T:g` vs `SEAL:S`?** **YES at token level & in frequency-adjusted sign OLS, weak/non-significant in raw unweighted Spearman:** Pottery tokens are significantly visually simpler than seal tokens (`z = -0.764` vs `-0.641`, Mann-Whitney `p = 8.1952e-06`); across all `190` frequent signs, raw unweighted Spearman is negative but non-significant (`ρ = -0.0981`, `p = 0.17809`; Sobel edge `ρ = -0.0827`, `p = 0.25668`), turning significant in OLS once sign frequency is controlled (`β = -0.162`, `p = 0.0185`).

**Key Diagnostic — Monotone Complexity Gradient Inside Non-Tally Geometric Shapes (`M17`, Excluding `031-035`):**

| Complexity Quartile | `M17` Pottery Rate vs Non-`M17` | Within-Quartile OR (`POT` vs `SEAL`) | Fisher `p` | Representative `M17` Signs |
|---|---:|---:|---:|---|
| `Q1_Simplest` | `17.35%` vs `11.83%` | **`1.569x`** | `0.000761` | `480` chevron (19.4x), `700` U-bowl (18.6x), `411` branch (12.6x), `850` diamond (2.64x), `892` bow (2.25x), `520` arrow (0.38x) |
| `Q2_LowMid` | `9.66%` vs `8.9%` | **`1.105x`** | `0.6371` | `790` oval (2.79x), `220` fish (0.93x), `861` peaked diamond (0.68x) |
| `Q3_HighMid` | `8.93%` vs `9.91%` | **`0.926x`** | `0.8653` | `550` hourglass (1.10x), `482` double-chevron (1.04x), `690` double-X (0.25x) |
| `Q4_Complex` | `5.12%` vs `8.95%` | **`0.568x`** | `0.08882` | `615` 3×3 grid (0.67x), `820` 6-spoked wheel (0.42x), `692` outlined-X (0.40x) |

**Minimal-Pair Morphological Contrasts (Same Symbol Category, Simple vs Elaborated Variant on Pottery vs Seals):**

| Symbol Category | Simple Variant (`VC_z`, Pottery OR, `p`) | Elaborated Variant (`VC_z`, Pottery OR, `p`) | Simple/Complex OR Ratio |
|---|---|---|---:|
| `chevron_inverted_v` | `480` (`z=-1.092`, **`19.385x`**, `p=3.139e-11`) | `482` (`z=0.432`, `1.042x`, `p=1.0`) | **`18.6x`** |
| `diamond_lozenge` | `850` (`z=-0.8`, **`2.644x`**, `p=0.04443`) | `861` (`z=-0.534`, `0.676x`, `p=0.2425`) | **`3.91x`** |
| `circle_vs_6spoked_wheel` | `790` (`z=-0.379`, **`2.787x`**, `p=0.0837`) | `820` (`z=0.883`, `0.421x`, `p=0.01996`) | **`6.62x`** |

### 2C. Exact MLE Logistic Regression & Stratified Mantel-Haenszel Controlling for Visual Glyph Stroke-Complexity

| Model Specification (Outcome = `POT:T:g` vs `SEAL:S`) | Predictor | All-Tokens OR (`95% CI`) | All-Tokens `p` | Dedup-Seq OR (`95% CI`) | Dedup-Seq `p` |
|---|---|---:|---:|---:|---:|
| M0: Unadjusted | `is_megalithic` | **`2.6138x`** (`2.2603-3.0227`) | `2.1365e-38` | `1.8945x` (`1.5901-2.2571`) | `8.5956e-13` |
| M1: + Linear `visual_complexity_z` | `is_megalithic` | **`2.5813x`** (`2.2314-2.9861`) | `2.7636e-37` | `1.8826x` (`1.5796-2.2438`) | `1.5983e-12` |
| M2: + Full Stroke Metrics (ink, edge, crossings, holes, PUA) | `is_megalithic` | **`2.5403x`** (`2.1842-2.9543`) | `1.0467e-33` | `1.8979x` (`1.5877-2.2687`) | `1.9738e-12` |
| M3: + Nonlinear Complexity Octile Fixed Effects (8 bins) | `is_megalithic` | **`2.8423x`** (`2.4157-3.3442`) | `2.3563e-36` | `1.986x` (`1.6382-2.4077`) | `2.8436e-12` |
| M4 (Split): Non-Tally Geometric (17 signs) | adj Octile FE | `is_megalithic_notally` | **`1.688x`** (`1.3747-2.0727`) | `5.7869e-07` | `1.4232x` (`1.1236-1.8026`) | `0.0034262` |
| M4 (Split): Vertical Tally Strokes (`031-035`) | adj Octile FE | `is_tally_031_035` | **`6.1632x`** (`4.9819-7.6247`) | `5.5693e-63` | `3.5302x` (`2.7157-4.589`) | `4.3064e-21` |
| M6 (Split + Seq Length): Non-Tally Geometric | adj Octiles + Len | `is_megalithic_notally` | **`1.3996x`** (`1.1097-1.7652`) | `0.0045193` | `1.4086x` (`1.0881-1.8235`) | `0.0092842` |
| M6 (Split + Seq Length): Vertical Tally Strokes | adj Octiles + Len | `is_tally_031_035` | **`4.337x`** (`3.3335-5.6425`) | `8.4811e-28` | `3.5809x` (`2.6459-4.8462`) | `1.425e-16` |

- **Log-Odds Attenuation after Visual Complexity Control:** All-Tokens attenuates by **`1.3%`** (linear), **`3.0%`** (multivariate stroke), and **`-8.7%`** (octile FE); Deduplicated-Seq attenuates by **`1.0%`** / **`-0.3%`** / **`-7.4%`**.
- **Stratified Cochran-Mantel-Haenszel (CMH) Across Visual-Complexity Strata:**
  - `M_shared (22)` across Complexity Quartiles (All Tokens): **CMH OR = `2.6295x`** (`95% CI [2.2731, 3.0417]`, `p = 8.7676e-41`) | Octiles: **`2.6951x`** (`p = 5.3293e-38`) | Dedup-Seq Quartiles: **`1.9034x`** (`p = 3.7057e-13`)
  - **`M_notally (17 non-tally geometric shapes)`** across Complexity Quartiles: All Tokens **CMH OR = `1.211x`** (`95% CI [1.0039, 1.4609]`, `p = 0.050279`) | Dedup-Seq: **`1.1929x`** (`p = 0.12702`)
- **1,000-Permutation Visual-Complexity-Matched Null Model (Comparing `M_shared` against 1,000 random subsets of equally simple Indus signs!):**
  - `M22 vs all 687 rendered signs (complexity-quartile matched)`: Observed OR = **`2.6144x`** vs Null median `0.9898x` (95th pct `2.5403x`), `z = +1.75`, **`perm_p = 0.04396`**
  - `M22 vs freq>=10 rendered signs (complexity-quartile matched)`: Observed OR = **`2.6144x`** vs Null median `1.016x` (95th pct `2.0525x`), `z = +2.39`, **`perm_p = 0.01299`**
  - `M17 non-tally geometric vs freq>=10 rendered signs (complexity-quartile matched)`: Observed OR = **`1.1622x`** vs Null median `0.9986x` (95th pct `2.178x`), `z = +0.30`, **`perm_p = 0.38761`**

- **Sign-Type-Level OLS (`N = 190` frequent signs, weighting every sign equally):** Unadjusted `is_megalithic` `β = +1.048` (`p = 1.2723e-05`) → Adjusted for `visual_complexity_z` (`β = -0.162`, `p = 0.018521`), `log_tot_n`, and `is_pua_compound`: `is_megalithic` shifts to `β = +1.435` (`p = 3.3583e-09`). Splitting tallies vs non-tally shapes: `is_tally` `β = +2.804` (`p = 6.4215e-10`) vs `is_megalithic_notally` `β = +1.082` (`p = 2.4928e-05`).

---
## Part 3 — EKSIK #2B: Published Gulf / Mesopotamia / Dilmun Seals External Audit vs Layer 4B `m* = 50` Threshold

- **Our Corpus (`inscriptions_5679.csv`):** `41` foreign-region inscriptions (`{'Persian Gulf': 16, 'Mesopotamia': 11, 'Iranian Plateau': 11, 'Central Asia': 3}`), including `33` seals (`18` circular/cylinder, `14` square). Layer 4B's connected Gulf circular-seal constraint graph `G_gulf_cc` contains **`E = 23` seals** (`U = 53` signs, `35` singletons = `66.0%`, `DOF = 30`).
- **Global Published Literature Audit:**
  - **Gadd (1932, *Proc. British Academy* 18):** `18` Indus-style items from Ur & Babylonia (includes cuneiform `No. 1` BM 120573 + uninscribed Gulf seals + subset of authentic Indus-script seals).
  - **Laursen (2010, *AAE* 21):** Out of `120+` known circular Gulf Type seals across Bahrain, Failaka, Oman, Ur, Susa, and the Indus periphery, exactly **`28`** (`27-29`) carry authentic Indus script inscriptions.
  - **Parpola (1994) / CISI Vol. 3 (Parts 1-3 published 2010/2019/2022; Part 4 Gulf & West Asia forthcoming):** ~**`30-40`** total published West Asian Indus-inscribed seals worldwide across all shapes.
- **Honest Gap Assessment vs Layer 4B `m* = 50` Percolation Threshold:**
  - Remaining published circular Gulf/Mesopotamian seals not in our 23-seal graph: **`~5-15` seals** (yielding at most **`E = 28-38`** total globally).
  - Required for Layer 4B AC-3 percolation (`P_unique >= 95%`) WITH `k = 2` bilingual anchors: **`m* = 50` additional seals (`73` total)** → **`35-45 seals STILL MISSING`**.
  - Required WITHOUT a bilingual (`k = 0`): **`m* > 180` additional seals (`>203` total)** → **`>165 seals STILL MISSING`**.
  - **Brutally Honest Verdict:** Adding the remaining ~5-15 published Gulf/Mesopotamian circular seals from Gadd (1932), Laursen (2010; 28 inscribed Gulf Type seals), and CISI Vol. 3.4 (forthcoming) would bring our Gulf constraint graph from E=23 to at most E=28-38 — STILL 35-45 circular seals SHORT of Layer 4B's m*=50 additional (73 total) percolation threshold EVEN WITH k=2 bilingual anchors, and >165 seals short without a bilingual (k=0, m*>180). Exhausting all currently published West Asian seals cannot break the phonetic identifiability barrier; newly excavated seals or a genuine bilingual remain mandatory.

