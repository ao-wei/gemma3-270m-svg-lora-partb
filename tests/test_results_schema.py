import json
import unittest
from pathlib import Path

import jsonschema

from student_kit.result_schema import add_passes, summarize


class ResultsSchemaTests(unittest.TestCase):
    def test_pass_rules_and_schema(self):
        reward = {
            "total": 0.8,
            "validity": 0.9,
            "fidelity": 0.6,
            "violations": [],
            "metadata": {"fatal": False},
        }
        output = add_passes({"raw_text": "<svg/>", "svg": "<svg/>", "reward": reward})
        sample = {"id": 0, "prompt": "x", "reference_svg": "<svg/>", "base": output, "tuned": dict(output)}
        summary = summarize([sample], "base")
        result = {
            "schema_version": 2,
            "environment": {},
            "model": {"base_path": "base", "adapter_path": "adapter"},
            "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048, "seed": 42},
            "counts": {"validation_samples": 1},
            "summary": {"base": summary, "tuned": summary},
            "samples": [sample],
        }
        schema = json.loads(Path("student_kit/results_schema_v2.json").read_text(encoding="utf-8"))
        jsonschema.validate(result, schema)
        self.assertTrue(output["passes"]["valid"])
        self.assertTrue(output["passes"]["quality"])
        self.assertEqual(summary["pass_rate"], 1.0)

    def test_degenerate_output_does_not_quality_pass(self):
        reward = {
            "total": 0.7,
            "validity": 0.9,
            "fidelity": 0.5,
            "violations": ["background_only_or_blank"],
            "metadata": {"fatal": False},
        }
        output = add_passes({"reward": reward})
        self.assertTrue(output["passes"]["valid"])
        self.assertFalse(output["passes"]["quality"])


if __name__ == "__main__":
    unittest.main()
