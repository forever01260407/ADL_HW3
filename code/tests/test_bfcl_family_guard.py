"""Regression checks for task variants escaping the public-overlap exclusion."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from bfcl_family_guard import audit_file, excluded_families, family_of


class FamilyGuardTest(unittest.TestCase):
    def test_variant_and_expanded_step_share_family(self):
        for name in ("base", "miss_param", "miss_func", "long_context", "composite"):
            for suffix in ("", ":plain", ":state#12"):
                self.assertEqual(family_of(f"multi_turn_{name}_25{suffix}"), "bfcl_task_25")

    def test_unrelated_ids_do_not_collide_by_number(self):
        self.assertIsNone(family_of("live_irrelevance_25"))
        self.assertIsNone(family_of("glaive_25"))

    def test_unknown_multi_turn_category_fails_closed(self):
        with self.assertRaises(ValueError):
            excluded_families({"drop_ids": ["multi_turn_new_category_25"]})

    def test_related_variant_is_excluded_even_with_incorrect_metadata(self):
        blocked = excluded_families({"drop_ids": ["multi_turn_miss_param_25"]})
        row = {"id": "multi_turn_base_25:plain", "family": "wrong",
               "source": "bfcl_plain", "no_target": [1],
               "messages": [{"role": "user"}, {"role": "assistant"},
                            {"role": "tool"}, {"role": "assistant"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = audit_file(path, blocked)
        self.assertEqual(report["excluded_family_target_steps"], 1)
        self.assertEqual(report["excluded_family_records"][0]["id"], row["id"])
        self.assertEqual(report["family_metadata_mismatches"], [row["id"]])


if __name__ == "__main__":
    unittest.main()
