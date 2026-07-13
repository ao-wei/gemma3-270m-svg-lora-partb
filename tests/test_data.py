import unittest

from student_kit.data import CompletionOnlyCollator, filter_uninformative_prompts, simplify_svg_targets, split_train_dev, tokenize_example


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 3, 10, 11, 12]


class DataTests(unittest.TestCase):
    def test_assistant_only_masking(self):
        row = {"messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}
        tokenized = tokenize_example(row, FakeTokenizer(), max_length=10)
        self.assertEqual(tokenized["labels"], [-100, -100, -100, 10, 11, 12])

    def test_padding_labels_are_masked(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is installed in the training environment")
        collator = CompletionOnlyCollator(0)
        batch = collator([
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [1], "attention_mask": [1], "labels": [1]},
        ])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [1, -100]])

    def test_split_is_deterministic_and_disjoint(self):
        rows = [{"id": index} for index in range(20)]
        train_a, dev_a = split_train_dev(rows, 4, 42)
        train_b, dev_b = split_train_dev(rows, 4, 42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(dev_a, dev_b)
        self.assertFalse({x["id"] for x in train_a} & {x["id"] for x in dev_a})

    def test_uninformative_prompts_are_filtered_without_mutation(self):
        rows = [
            {"messages": [{}, {"content": "placeholder"}, {}]},
            {"messages": [{}, {"content": "A real logo prompt"}, {}]},
        ]
        kept, rejected = filter_uninformative_prompts(rows)
        self.assertEqual(rejected, [0])
        self.assertEqual(kept, [rows[1]])
        self.assertEqual(len(rows), 2)

    def test_simplified_target_is_complete_and_source_is_unchanged(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><defs/><circle cx="10" cy="10" r="5" fill="red"/><rect x="1" y="2" width="3" height="4" fill="url(#x)"/><path d="M0 0L1 1"/></svg>'
        rows = [{"messages": [{}, {}, {"content": svg}]}]
        result = simplify_svg_targets(rows, 2)
        self.assertEqual(rows[0]["messages"][2]["content"], svg)
        self.assertIn("</svg>", result[0]["messages"][2]["content"])
        self.assertNotIn("url(#x)", result[0]["messages"][2]["content"])


if __name__ == "__main__":
    unittest.main()
