"""Synthetic-only tests; no public examples or GPU needed."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from check_predictions import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data.jsonl.gz"
        self.pred = Path(self.temp.name) / "pred.jsonl"
        with gzip.open(self.data, "wt", encoding="utf-8") as stream:
            for key in ("synthetic-a", "synthetic-b"):
                stream.write(json.dumps({"id": key}) + "\n")

    def write(self, keys, reason="stop"):
        self.pred.write_text("\n".join(json.dumps({
            "id": key, "output": "synthetic", "finish_reason": reason,
        }) for key in keys) + "\n", encoding="utf-8")

    def test_complete_and_capped_are_structurally_valid(self):
        self.write(["synthetic-a", "synthetic-b"], "length")
        self.assertEqual(audit(self.data, self.pred)["status"], "pass")

    def test_partial_is_pending_only_in_progress(self):
        self.write(["synthetic-a"])
        self.assertEqual(audit(self.data, self.pred, True)["status"], "pending")
        self.assertEqual(audit(self.data, self.pred)["errors"]["missing_ids"], 1)

    def test_duplicate_and_unknown_ids_fail(self):
        self.write(["synthetic-a", "synthetic-a", "synthetic-other"])
        errors = audit(self.data, self.pred, True)["errors"]
        self.assertEqual(errors["duplicate_id_rows"], 1)
        self.assertEqual(errors["unknown_id_rows"], 1)

    def test_active_partial_utf8_tail_is_deferred(self):
        self.write(["synthetic-a"])
        with self.pred.open("ab") as stream:
            stream.write(b'{"output":"\xe4\xb8')
        before = self.pred.read_bytes()
        result = audit(self.data, self.pred, True)
        self.assertEqual(result["status"], "pending")
        self.assertGreater(result["deferred_tail_bytes"], 0)
        self.assertEqual(audit(self.data, self.pred)["status"], "fail")
        self.assertEqual(self.pred.read_bytes(), before)

    def test_bad_fields_and_json_are_errors(self):
        self.pred.write_text('{"id":[],"output":2,"finish_reason":[]}\n{bad}\n', encoding="utf-8")
        errors = audit(self.data, self.pred, True)["errors"]
        for key in ("invalid_id_rows", "invalid_output_rows", "invalid_finish_reason_rows", "malformed_json_lines"):
            self.assertEqual(errors[key], 1)

    def test_absent_file_is_pending_only_in_progress(self):
        self.assertEqual(audit(self.data, self.pred, True)["status"], "pending")
        self.assertEqual(audit(self.data, self.pred)["status"], "fail")

    def test_final_record_needs_no_trailing_newline(self):
        self.write(["synthetic-a", "synthetic-b"])
        self.pred.write_bytes(self.pred.read_bytes().rstrip(b"\n"))
        self.assertEqual(audit(self.data, self.pred)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
