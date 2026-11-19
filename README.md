# INDUS-DEDUCT — Top-Down Computational Epigraphy Engine

A strictly falsifiable, zero-leakage, 6-layer top-down (*tümdengelim*) computational system for analyzing and attempting structural/semantic decipherment of the Indus Valley (Harappan) Script.

## Core Scientific Rules
1. **Zero Theory Contamination:** Speculative translation columns (`sanskrit`, `translation`, `notes`) from upstream datasets are stripped at ingestion.
2. **Zero Train/Test Leakage:** All language models, masked-sign predictors, and enrichment tests deduplicate exact repeated inscriptions before splitting.
3. **Confounder Control:** Every archaeological/semantic hypothesis (e.g., *Water / Hydraulic* triangulation) is verified holding `site` and `object_type` constant.
4. **Anti-Hallucination Permutation Gate:** No phonetic hypothesis is accepted unless it beats 1,000 randomly permuted synthetic lexicons at `z > 5`.

## Directory Layout
- `data/raw/` — Immutable downloaded open corpora (`inscriptions_5679.csv`, SQL glyph catalog, Indus TTF font, Mahadevan 417 glyph PNGs)
- `data/processed/` — Uncontaminated canonical SQLite warehouse (`indus_canonical.sqlite`) and clean CSVs
- `src/layer0_ingest.py` — Canonical ingestion + empirical reading-direction proof
- `src/layer1_2_deductive_engine.py` — Layer 1 (compounds/allographs) + Layer 2 (syntax proof, slot grammar, masked prediction, archaeology enrichment, foreign-seal surprisal ranker)
- `src/layer3_water_probe.py` — Layer 3A multi-axis "Water (Su)" triangulation probe
- `reports/` — Reproducible JSON audit outputs and rendered glyph reference sheets
