"""Scoring-contract regressions: intentional boundary cases, no model/GPU needed."""
import unittest
from audit_part1 import adapt_own, analyze, legacy_comparison, T


def gold(sid, after="user", call=True):
    return {"id": sid, "messages": [{"role": after, "content": "independent synthetic fixture"}],
            "acceptable_next": [{"name": "check", "arguments": {"n": 2}}] if call else [],
            "first_step_eligible": call and after == "user"}


class Contracts(unittest.TestCase):
    def test_first_text_counts_but_tool_chain_text_does_not(self):
        rows = [gold("first"), gold("chain", "tool"), gold("text", call=False)]
        preds = {r["id"]: {"id": r["id"], "output": "Checking.", "finish_reason": "stop"} for r in rows}
        report = analyze(rows, preds)
        self.assertAlmostEqual(report["headline_pct"], 100 * 2 / 12)
        self.assertEqual(report["groups"]["must_call"]["right"], 0)
        self.assertEqual(report["groups"]["eligible_first_call"]["right"], 1)

    def test_empty_capped_malformed_and_missing_do_not_receive_first_credit(self):
        rows = [gold(str(i)) for i in range(4)]
        preds = {"0": {"output": ""}, "1": {"output": "Checking.", "finish_reason": "length"},
                 "2": {"output": '<tool_call>{"bad":3}</tool_call>'}}
        self.assertEqual(analyze(rows, preds)["headline_pct"], 0)

    def test_default_and_alternative_call_use_staff_comparator(self):
        row = gold("a", "tool")
        row["defaults"] = {"check": {"n": 2}}
        row["acceptable_next"].append({"name": "alternate", "arguments": {}})
        for call in [{"name": "check", "arguments": {}}, {"name": "alternate", "arguments": {}}]:
            self.assertEqual(analyze([row], {"a": {"tool_calls": [call]}})["headline_pct"], 100)
        self.assertEqual(analyze([row], {"a": {"tool_calls": [{"name": "check", "arguments": {"n": True}}]}})["headline_pct"], 0)

    def test_text_plus_wrong_call_does_not_receive_first_credit(self):
        pred = {"a": {"text": "Checking.", "tool_calls": [{"name": "wrong", "arguments": {}}]}}
        self.assertEqual(analyze([gold("a")], pred)["headline_pct"], 0)

    def test_own_eligibility_and_defaults_and_after_validation(self):
        row = {"id": "own", "messages": [{"role": "user", "content": "check"}], "after": "user",
               "gold": {"calls": [{"name": "check", "arguments": '{"n": 2}'}]},
               "tools": [{"function": {"name": "check", "parameters": {"properties": {"n": {"default": 2}}}}}]}
        adapted = adapt_own(row)
        self.assertTrue(adapted["first_step_eligible"])
        self.assertEqual(adapted["defaults"], {"check": {"n": 2}})
        row["after"] = "tool"
        with self.assertRaises(ValueError):
            adapt_own(row)

    def test_counterfactual_does_not_claim_all_after_user_safe(self):
        row = gold("a")
        row["first_step_eligible"] = False  # preserve authoritative gold; never guess on official input
        pred = {"a": {"tool_calls": row["acceptable_next"]}}
        report = analyze([row], pred)
        self.assertEqual(report["headline_pct"], 100)
        self.assertEqual(report["counterfactuals"]["visible_after_user_all_steps"]["lost_steps"], 1)
        self.assertEqual(pred["a"]["tool_calls"], row["acceptable_next"])

    def test_exact_official_parity_and_no_call_blank_behavior(self):
        rows = [gold("a"), gold("b", "tool"), gold("c", call=False)]
        preds = {"a": {"output": "Confirming."}, "b": {"tool_calls": rows[1]["acceptable_next"]}, "c": {"output": ""}}
        for parser in ("lenient", "strict"):
            actual = analyze(rows, preds, parser)["official_result"]
            expected, _ = T.score_tools(rows, preds, parser)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["with_first_step_rule"]["all"]["weighted_step_acc"], 1)

    def test_duplicate_data_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze([gold("a"), gold("a")], {})

    def test_legacy_metric_is_separate_from_default_and_first_rule_changes(self):
        row = {"id": "a", "task": "part1", "messages": [{"role": "user", "content": "check"}],
               "gold": {"calls": [{"name": "check", "arguments": {"n": 2}}]}, "defaults": {"check": {"n": 2}}}
        report = legacy_comparison([row], {"a": {"tool_calls": [{"name": "check", "arguments": {}}]}}, "lenient")
        self.assertEqual(report["legacy_exact_pct"], 0)
        self.assertEqual(report["official_comparison_unweighted_without_first_rule_pct"], 100)
        self.assertEqual(report["comparison_rule_changes_before_first_credit"], {"call_comparison_change": 1})


if __name__ == "__main__":
    unittest.main()
