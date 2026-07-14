import unittest

from student_kit.data import filter_uninformative_prompts, load_jsonl
from student_kit.common import is_bf16_compatibility_error, read_yaml


class TrainingContractTests(unittest.TestCase):
    def test_bf16_compatibility_detection_excludes_oom(self):
        self.assertTrue(is_bf16_compatibility_error(RuntimeError("MPS does not support bfloat16")))
        self.assertFalse(is_bf16_compatibility_error(RuntimeError("MPS backend out of memory")))

    def test_full_training_uses_217_rows(self):
        rows, rejected = filter_uninformative_prompts(load_jsonl("train.jsonl"))
        self.assertEqual(len(rows), 217)
        self.assertEqual(rejected, [79, 123])

    def test_all_full_targets_fit_3584(self):
        from transformers import AutoTokenizer

        rows, _ = filter_uninformative_prompts(load_jsonl("train.jsonl"))
        tokenizer = AutoTokenizer.from_pretrained("gemma3-270m-it", local_files_only=True)
        lengths = [
            len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False))
            for row in rows
        ]
        self.assertLessEqual(max(lengths), 3584)

    def test_final_config_is_full_data_complete_svg(self):
        config = read_yaml("train_config.yaml")
        self.assertEqual(config["dev_size"], 0)
        self.assertIsNone(config["max_train_samples"])
        self.assertFalse(config["simplify_targets"])
        self.assertTrue(config["require_no_truncation"])
        self.assertEqual(config["max_length"], 3584)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["init_adapter_path"], "adapter_curriculum_stage1")


if __name__ == "__main__":
    unittest.main()
