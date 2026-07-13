#!/usr/bin/env python3
"""LoRA supervised fine-tuning for Gemma 3 text models."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from common import device_name, read_yaml, set_seed, write_json
from data import CompletionOnlyCollator, filter_uninformative_prompts, load_jsonl, simplify_svg_targets, split_train_dev, tokenize_example


class TokenizedRows:
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return tokenize_example(self.rows[index], self.tokenizer, self.max_length)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--dev-size", type=int)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--shortest-first", action="store_true")
    parser.add_argument("--simplify-targets", action="store_true")
    parser.add_argument("--max-visible-elements", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    for key, value in {
        "output_dir": args.output_dir,
        "max_train_samples": args.max_train_samples,
        "seed": args.seed,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "eval_steps": args.eval_steps,
        "dev_size": args.dev_size,
        "early_stopping_patience": args.early_stopping_patience,
    }.items():
        if value is not None:
            config[key] = value
    if args.no_early_stopping:
        config["early_stopping"] = False
    if args.shortest_first:
        config["shortest_first"] = True
    if args.simplify_targets:
        config["simplify_targets"] = True
    if args.max_visible_elements is not None:
        config["max_visible_elements"] = args.max_visible_elements

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

    class MPSCacheCallback(TrainerCallback):
        """Release unused Metal allocations during long variable-length runs."""

        @staticmethod
        def _clear():
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        def on_step_end(self, args, state, control, **kwargs):
            self._clear()

        def on_substep_end(self, args, state, control, **kwargs):
            self._clear()

    seed = int(config["seed"])
    set_seed(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = load_jsonl(config["train_file"])
    rejected_rows = []
    if config.get("drop_uninformative_prompts", True):
        rows, rejected_rows = filter_uninformative_prompts(rows)
    if config.get("simplify_targets", False):
        rows = simplify_svg_targets(rows, int(config.get("max_visible_elements", 6)))
    dev_size = int(config["dev_size"])
    if dev_size:
        train_rows, dev_rows = split_train_dev(rows, dev_size, seed)
    else:
        train_rows, dev_rows = rows, []
    if config.get("shortest_first", False):
        def assistant_tokens(row):
            full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True)
            return len(full) - len(prompt)

        train_rows = sorted(train_rows, key=assistant_tokens)
    max_samples = config.get("max_train_samples")
    if max_samples:
        train_rows = train_rows[: int(max_samples)]

    max_length = int(config["max_length"])
    train_dataset = TokenizedRows(train_rows, tokenizer, max_length)
    eval_max_length = int(config.get("eval_max_length", max_length))
    dev_dataset = TokenizedRows(dev_rows, tokenizer, eval_max_length) if dev_rows else None
    device = device_name()
    dtype = torch.float32
    if config.get("precision") == "bf16" or (config.get("precision") == "auto" and device in {"cuda", "mps"}):
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], local_files_only=True, dtype=dtype, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=False,
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        max_grad_norm=float(config["max_grad_norm"]),
        logging_steps=int(config["logging_steps"]),
        eval_strategy="steps" if dev_rows else "no",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps" if dev_rows else "no",
        save_steps=int(config["eval_steps"]),
        save_total_limit=2,
        load_best_model_at_end=bool(dev_rows),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        prediction_loss_only=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        data_seed=seed,
        use_cpu=device == "cpu",
    )
    callbacks = [MPSCacheCallback()]
    if dev_rows and config.get("early_stopping", True):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"])))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate() if dev_rows else {}
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    metadata = {
        "config": config,
        "device": device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "rejected_source_rows": rejected_rows,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "elapsed_seconds": time.time() - started,
        "train_metrics": result.metrics,
        "eval_metrics": eval_metrics,
    }
    write_json(output_dir / "training_summary.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
