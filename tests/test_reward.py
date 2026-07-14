import unittest

from student_kit.reward import reward


PROMPT = "A navy #1B3A5C circular badge with a golden line and white center."
GOOD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="128" cy="128" r="100" fill="#1B3A5C"/><circle cx="128" cy="128" r="60" fill="white"/><line x1="80" y1="128" x2="176" y2="128" stroke="gold" stroke-width="8"/></svg>'


class RewardTests(unittest.TestCase):
    def test_good_svg_scores_high(self):
        result = reward(PROMPT, GOOD)
        self.assertGreater(result["total"], 0.85)
        self.assertFalse(result["metadata"]["fatal"])

    def test_unclosed_svg_is_fatal(self):
        result = reward(PROMPT, GOOD.replace("</svg>", ""))
        self.assertTrue(result["metadata"]["fatal"])
        self.assertLessEqual(result["total"], 0.1)

    def test_wrapper_text_is_penalized(self):
        clean = reward(PROMPT, GOOD)
        wrapped = reward(PROMPT, "Here it is:\n" + GOOD)
        self.assertLess(wrapped["total"], clean["total"])
        self.assertIn("non_svg_wrapper_text", wrapped["violations"])

    def test_wrong_svg_namespace_is_fatal(self):
        broken = GOOD.replace("http://www.w3.org/2000/svg", "http://www.w3.org/svg")
        result = reward(PROMPT, broken)
        self.assertTrue(result["metadata"]["fatal"])
        self.assertIn("invalid_svg_namespace", result["violations"])

    def test_script_is_fatal(self):
        unsafe = GOOD.replace("</svg>", '<script>alert(1)</script></svg>')
        result = reward(PROMPT, unsafe)
        self.assertTrue(result["metadata"]["fatal"])
        self.assertIn("forbidden_tag:script", result["violations"])

    def test_external_reference_is_fatal(self):
        unsafe = GOOD.replace("</svg>", '<use href="https://evil.example/x.svg#x"/></svg>')
        self.assertTrue(reward(PROMPT, unsafe)["metadata"]["fatal"])

    def test_non_finite_geometry_is_penalized(self):
        broken = GOOD.replace('cx="128"', 'cx="NaN"', 1)
        result = reward(PROMPT, broken)
        self.assertIn("non_finite_number", result["violations"])
        self.assertLess(result["components"]["geometry"], 1.0)

    def test_out_of_bounds_is_penalized(self):
        broken = GOOD.replace('cx="128"', 'cx="9999"', 1)
        result = reward(PROMPT, broken)
        self.assertIn("extreme_or_non_finite_geometry", result["violations"])

    def test_empty_and_single_shape_are_penalized(self):
        empty = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'
        single = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><rect x="0" y="0" width="256" height="256" fill="red"/></svg>'
        self.assertLess(reward(PROMPT, empty)["total"], reward(PROMPT, single)["total"])
        self.assertLess(reward(PROMPT, single)["total"], reward(PROMPT, GOOD)["total"])

    def test_color_fidelity_changes_score(self):
        wrong = GOOD.replace("#1B3A5C", "#FF00FF").replace("white", "black").replace("gold", "red")
        self.assertLess(reward(PROMPT, wrong)["fidelity"], reward(PROMPT, GOOD)["fidelity"])

    def test_repetition_is_penalized(self):
        repeated = GOOD.replace("</svg>", ("<circle cx=\"1\" cy=\"1\" r=\"1\" fill=\"red\"/>" * 80) + "</svg>")
        result = reward(PROMPT, repeated)
        self.assertIn("repetitive_elements", result["violations"])

    def test_repeated_canvas_cover_is_degenerate(self):
        circles = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">' + (
            '<circle cx="128" cy="128" r="128" fill="#888"/>' * 3
        ) + "</svg>"
        result = reward(PROMPT, circles)
        self.assertIn("background_only_or_blank", result["violations"])
        self.assertIn("repeated_identical_elements", result["violations"])
        self.assertLess(result["total"], 0.5)

    def test_global_and_pairwise_spatial_fidelity(self):
        spatial_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><circle cx="64" cy="128" r="25" fill="red"/><rect x="170" y="100" width="50" height="50" fill="blue"/></svg>'
        correct = reward("A circle left of a square", spatial_svg)
        wrong = reward("A square left of a circle", spatial_svg)
        self.assertGreater(correct["metadata"]["spatial_coverage"], wrong["metadata"]["spatial_coverage"])

    def test_ambiguous_spatial_relation_is_unscorable(self):
        ambiguous = GOOD.replace("</svg>", '<circle cx="30" cy="30" r="10" fill="red"/></svg>')
        result = reward("A circle left of a line", ambiguous)
        self.assertEqual(result["metadata"]["spatial_scorable"], 0)

    def test_unreasonable_stroke_width_is_penalized(self):
        broken = GOOD.replace('stroke-width="8"', 'stroke-width="80"')
        result = reward(PROMPT, broken)
        self.assertIn("unreasonable_stroke_width", result["violations"])
        self.assertLess(result["components"]["geometry"], 1.0)


if __name__ == "__main__":
    unittest.main()
