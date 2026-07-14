import unittest
from pathlib import Path


class AdapterLoadTests(unittest.TestCase):
    @unittest.skipUnless(Path("gemma3-270m-it").exists() and Path("adapter/adapter_config.json").exists(), "local model required")
    def test_adapter_loads_in_fresh_model(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained("gemma3-270m-it", local_files_only=True, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, "adapter")
        self.assertEqual(model.peft_config["default"].peft_type.value, "LORA")


if __name__ == "__main__":
    unittest.main()
