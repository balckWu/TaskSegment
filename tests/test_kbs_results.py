"""Check the statistical protocol and supported archived summary formats."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.kbs_results import COUNTS, SEEDS, aggregate, read_run


def example_rows():
    return [{"method": "unet", "seed": seed, "domain": domain, "n": count,
             "dice": (0.8 if domain == "thyroid" else 0.2) + offset * 0.1,
             "iou": 0.1 + offset * 0.1, "hd95": 20 + offset * 10}
            for offset, seed in enumerate(SEEDS) for domain, count in COUNTS.items()]


class KBSResultChecks(unittest.TestCase):
    def test_macro_is_domain_balanced_and_sd_uses_n_minus_one(self):
        result = aggregate(example_rows(), require_complete=False)["unet"]["macro"]
        self.assertAlmostEqual(result["dice"]["mean"], 0.4)
        self.assertAlmostEqual(result["dice"]["sd"], 0.1)
        self.assertEqual(result["hd95"], {"mean": 30, "sd": 10})

    def test_missing_duplicate_or_bad_count_is_rejected(self):
        rows = example_rows()
        cases = [rows[:-1], rows + [rows[0]]]
        wrong = copy.deepcopy(rows)
        wrong[0]["n"] = 1
        cases.append(wrong)
        for case in cases:
            with self.assertRaises(ValueError):
                aggregate(case, require_complete=False)

    def test_nonfinite_metrics_are_rejected(self):
        rows = example_rows()
        rows[0]["hd95"] = float("nan")
        with self.assertRaises(ValueError):
            aggregate(rows, require_complete=False)

    def test_baseline_n_and_main_count_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            predictions = run / "predictions/nested_baseline"
            predictions.mkdir(parents=True)
            summary = predictions / "summary_metrics.csv"
            for header, domain_header in [("n", "domain"), ("count", "organ")]:
                summary.write_text(f"{domain_header},{header},dice,iou,hd95\nthyroid,286,0.8,0.7,20\n")
                rows = read_run(run, "unet", 42)
                self.assertEqual(rows[0]["n"], 286)
                self.assertEqual(rows[0]["domain"], "thyroid")

    def test_iris_wrong_support_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "predictions").mkdir()
            (run / "predictions/metrics.json").write_text(json.dumps({
                "seed": 42, "eval_support_count": 1, "per_domain": {}}))
            with self.assertRaises(ValueError):
                read_run(run, "iris2d", 42)


if __name__ == "__main__":
    unittest.main()
