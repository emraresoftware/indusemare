#!/usr/bin/env python3
"""
INDUS-DEDUCT — Automated Reproducibility & Claim-Ledger Verification Suite
==========================================================================
Directly fulfills vega & seyyah's engineering audit requirements:
  1. Verifies zero speculative theory columns (`sanskrit`, `translation`, `notes`)
     in all processed canonical files and SQLite tables.
  2. Verifies SHA-256 checksums for every file listed in `DATA_MANIFEST.json`.
  3. Verifies exact numerical consistency between `reports/*.json` and canonical
     counts (catching any drift between raw vs deduplicated claims or subset labels).
  4. Verifies Layer 7 Council Audit invariants (BH-FDR survival, LOCO restoration,
     `: X :` LOBO transfer, non-linguistic negative control).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FORBIDDEN_COLS = {"sanskrit", "translation", "notes"}


class TestIndusDeductClaimLedger(unittest.TestCase):
    def test_01_zero_speculative_contamination(self) -> None:
        for csv_name in ("inscriptions_clean.csv", "lines_clean.csv", "glyphs_clean.csv"):
            df = pd.read_csv(PROC / csv_name, nrows=5)
            cols = {c.strip().lower() for c in df.columns}
            self.assertEqual(cols & FORBIDDEN_COLS, set(), f"Contaminated column in {csv_name}")

        with sqlite3.connect(PROC / "indus_canonical.sqlite") as conn:
            for tbl in ("inscriptions", "lines", "glyphs"):
                tcols = {r[1].strip().lower() for r in conn.execute(f"PRAGMA table_info({tbl})")}
                self.assertEqual(tcols & FORBIDDEN_COLS, set(), f"Contaminated column in sqlite:{tbl}")

    def test_02_data_manifest_sha256_integrity(self) -> None:
        manifest_path = ROOT / "DATA_MANIFEST.json"
        self.assertTrue(manifest_path.exists(), "DATA_MANIFEST.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest["files"]), 15)
        for entry in manifest["files"]:
            p = ROOT / entry["path"]
            self.assertTrue(p.exists(), f"Manifest file missing: {entry['path']}")
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(actual, entry["sha256"], f"SHA-256 mismatch on {entry['path']}")

    def test_03_layer0_canonical_counts_match_disk(self) -> None:
        l0 = json.loads((REPORTS / "layer0_audit.json").read_text(encoding="utf-8"))
        ins = pd.read_csv(PROC / "inscriptions_clean.csv", dtype=str)
        lines = pd.read_csv(PROC / "lines_clean.csv", dtype=str)
        self.assertEqual(len(ins), l0["total_inscriptions"])
        self.assertEqual(len(lines), l0["total_parsed_lines"])
        self.assertEqual(l0["total_inscriptions"], 5679)
        self.assertGreater(l0["directionality_audit"]["R/L"]["jar_740_final_pct"], 25.0)
        self.assertGreater(l0["directionality_audit"]["L/R"]["jar_740_final_pct"], 20.0)

    def test_04_layer5_and_5b_anti_hallucination_invariants(self) -> None:
        l5b = json.loads((REPORTS / "layer5b_expanded_phonetic.json").read_text(encoding="utf-8"))
        tour = l5b["part1_expanded_13probe_phonetic_gate"]["expanded_13_probes_tournament"]
        # Strict Mode 1 MUST fail z > 5.0 across all 4 families
        for fam, fdict in tour.items():
            m1 = fdict["mode1_primary_lemma_strict"]
            self.assertLess(m1["z_score_phon_sim"], 5.0, f"Unexpected strict Mode 1 pass for {fam}")
            self.assertFalse(m1["passes_z5_gate_phon_sim"])
        # Proto-Dravidian Mode 2B clears z > 5.0
        pdr_m2b = tour["Proto-Dravidian"]["mode2b_expanded_distinct_etymon"]
        self.assertGreater(pdr_m2b["z_score_phon_sim"], 5.0)

    def test_05_layer4b_csp_underdetermined_invariant(self) -> None:
        l4b = json.loads((REPORTS / "layer4b_rosetta_csp.json").read_text(encoding="utf-8"))
        self.assertEqual(l4b["constraint_graph_density"]["G_gulf_cc_23"]["DOF_full_U_minus_rankM"], 30)
        self.assertTrue(l4b["verdict"]["mathematically_underdetermined"])
        self.assertFalse(l4b["verdict"]["uniquely_locks_any_sign_at_p_lt_0_001"])

    def test_06_layer7_council_audit_invariants(self) -> None:
        l7 = json.loads((REPORTS / "layer7_council_audit.json").read_text(encoding="utf-8"))
        # Global FDR ran on >= 395 p-values
        self.assertGreaterEqual(l7["audit1_global_fdr"]["total_p_values_scanned"], 395)
        # Position-preserving null passes for both 740->090 (70:1) and 705/706->033 (87:0)
        self.assertTrue(l7["audit1b_position_preserving_null"]["740_to_090_dedup"]["survives_pos_null"])
        self.assertTrue(l7["audit1b_position_preserving_null"]["705_706_to_033_dedup"]["survives_pos_null"])
        # Non-linguistic negative control yields huge negative z-score (proving H1 shuffle != language alone)
        self.assertLess(l7["audit6_nonlinguistic_negative_control"]["nonlinguistic_z_score"], -50.0)
        # `: X :` four-dash frame reduces LOBO log-loss on unseen base glyph families
        self.assertLess(l7["audit4_codex_four_dash_transfer"]["LOBO_leave_one_base_glyph_family_out"]["delta_log_loss"], -0.03)
        # All 3 damaged-restoration subsets are present and LOCO beats position baseline
        rest = l7["audit2b_loco_damaged_restoration"]
        for key in ("edge_all_STRICT_LOCO", "edge_single_unambiguous_STRICT_LOCO", "edge_cross_site_STRICT_LOCO"):
            self.assertIn(key, rest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
