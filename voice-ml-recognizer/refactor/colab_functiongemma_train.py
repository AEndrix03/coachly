#!/usr/bin/env python3
"""
Colab T4 QLoRA training for Coachly JSON function-calling NLU.

Expected dataset format (jsonl):
- train.jsonl / val.jsonl / test.jsonl
- each row contains:
  {
    "text": "...",
    "label": {"action": "...", "items": [...]},
    "messages": [
      {"role":"system","content":"..."},
      {"role":"user","content":"..."},
      {"role":"assistant","content":"{...json...}"}
    ]
  }
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_CANDIDATES = [
    # Recommended default: open, small and stable on Colab T4.
    "Qwen/Qwen2.5-0.5B-Instruct",
    # Small FunctionGemma options.
    "google/functiongemma-270m-it",
    "unsloth/functiongemma-270m-it",
    # Extra fallback (larger, generally stronger).
    "Qwen/Qwen2.5-1.5B-Instruct",
]

VALID_ACTIONS = {"ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE", "UNKNOWN"}


@dataclass
class TrainConfig:
    data_dir: str = "refactor/data"
    train_file: str = "train.jsonl"   # override to use augmented file, e.g. train_aug.jsonl
    output_dir: str = "refactor/output/functiongemma_qlora"
    base_model: str = ""
    max_seq_len: int = 512
    num_epochs: int = 4
    lr: float = 2e-4
    warmup_ratio: float = 0.06
    train_batch_size: int = 2
    eval_batch_size: int = 2
    grad_accum: int = 8
    weight_decay: float = 0.01
    seed: int = 42
    save_merged: bool = False
    eval_samples: int = 120
    hf_token: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_id(requested: str, hf_token: str = "") -> str:
    if requested:
        return requested

    # Lazy import so the script still works if huggingface_hub is missing in edge environments.
    try:
        from huggingface_hub import model_info
    except Exception:
        return DEFAULT_MODEL_CANDIDATES[1]

    for candidate in DEFAULT_MODEL_CANDIDATES:
        try:
            info = model_info(candidate, token=hf_token if hf_token else None)
            if getattr(info, "gated", False) and not hf_token:
                print(f"Skip gated model without token: {candidate}")
                continue
            return candidate
        except Exception:
            continue
    return DEFAULT_MODEL_CANDIDATES[1]


def load_jsonl_splits(data_dir: str, train_file: str = "train.jsonl") -> DatasetDict:
    data_dir_path = Path(data_dir)
    files = {
        "train": str(data_dir_path / train_file),
        "validation": str(data_dir_path / "val.jsonl"),
        "test": str(data_dir_path / "test.jsonl"),
    }
    for name, p in files.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing split '{name}': {p}")
    return load_dataset("json", data_files=files)


def _safe_json_extract(raw: str) -> Optional[Dict]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def build_tokenized_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_seq_len: int) -> DatasetDict:
    def _map_row(row: Dict) -> Dict:
        msgs = row["messages"]
        prompt_msgs = msgs[:-1]
        full_msgs = msgs

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_enc = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Mask prompt tokens and train only on assistant completion tokens.
        prompt_len = min(len(prompt_ids), max(0, len(input_ids) - 1))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels[: len(input_ids)],
        }

    keep_cols = {"input_ids", "attention_mask", "labels"}
    out = dataset.map(_map_row, remove_columns=[c for c in dataset["train"].column_names if c not in keep_cols])
    return out


def build_model_and_tokenizer(base_model: str, hf_token: str = ""):
    token_arg = hf_token if hf_token else None
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=token_arg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token_arg,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    base_model = resolve_model_id(cfg.base_model, cfg.hf_token)
    print(f"Base model: {base_model}")
    print("Loading dataset...")
    ds_raw = load_jsonl_splits(cfg.data_dir, cfg.train_file)
    print(ds_raw)

    model, tokenizer = build_model_and_tokenizer(base_model, cfg.hf_token)
    ds_tok = build_tokenized_dataset(ds_raw, tokenizer, cfg.max_seq_len)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ta_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        logging_steps=20,
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
    )
    # transformers compatibility: older/newer versions renamed this argument
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    else:
        ta_kwargs["eval_strategy"] = "steps"
    args = TrainingArguments(**ta_kwargs)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Validation:", eval_metrics)

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to: {adapter_dir}")

    quick = quick_eval_json_action(
        model=trainer.model,
        tokenizer=tokenizer,
        test_rows=ds_raw["test"],
        max_samples=cfg.eval_samples,
    )
    with (output_dir / "quick_eval.json").open("w", encoding="utf-8") as f:
        json.dump(quick, f, ensure_ascii=False, indent=2)
    print("Quick test metrics:", quick)

    if cfg.save_merged:
        save_merged_model(base_model, adapter_dir, output_dir / "merged_fp16", cfg.hf_token)


@torch.no_grad()
def predict_label(model, tokenizer, user_text: str, system_prompt: str, max_new_tokens: int = 160) -> Dict:
    model.eval()
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    enc = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    out = model.generate(
        input_ids=enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][enc.shape[-1] :]
    gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    parsed = _safe_json_extract(gen_txt)
    return {"raw": gen_txt, "parsed": parsed}


def quick_eval_json_action(model, tokenizer, test_rows, max_samples: int = 120) -> Dict:
    total = min(max_samples, len(test_rows))
    chosen_idx = list(range(len(test_rows)))
    random.shuffle(chosen_idx)
    chosen_idx = chosen_idx[:total]

    valid_json = 0
    action_ok = 0
    item_count_mae_sum = 0.0

    for i, idx in enumerate(chosen_idx, start=1):
        row = test_rows[idx]
        pred = predict_label(model, tokenizer, row["text"], row["messages"][0]["content"])
        gold = row["label"]

        parsed = pred["parsed"]
        if parsed is None:
            continue
        valid_json += 1

        pred_action = parsed.get("action")
        if pred_action == gold.get("action"):
            action_ok += 1

        pred_items = parsed.get("items", [])
        gold_items = gold.get("items", [])
        if isinstance(pred_items, list):
            item_count_mae_sum += abs(len(pred_items) - len(gold_items))
        else:
            item_count_mae_sum += len(gold_items)

        if i % 20 == 0:
            print(f"Quick eval progress: {i}/{total}")

    valid_rate = valid_json / total if total else 0.0
    action_acc = action_ok / total if total else 0.0
    item_count_mae = item_count_mae_sum / total if total else math.nan

    return {
        "samples": total,
        "valid_json_rate": round(valid_rate, 4),
        "action_accuracy": round(action_acc, 4),
        "item_count_mae": round(item_count_mae, 4),
    }


def save_merged_model(base_model: str, adapter_dir: Path, out_dir: Path, hf_token: str = "") -> None:
    print("Merging adapter into full model...")
    token_arg = hf_token if hf_token else None
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu", token=token_arg)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    print(f"Merged model saved to: {out_dir}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="refactor/data")
    p.add_argument("--train_file",  type=str, default="train.jsonl", help="Train split filename inside data_dir (e.g. train_aug.jsonl)")
    p.add_argument("--output_dir", type=str, default="refactor/output/functiongemma_qlora")
    p.add_argument("--base_model", type=str, default="")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_merged", action="store_true")
    p.add_argument("--eval_samples", type=int, default=120)
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    a = p.parse_args()
    return TrainConfig(
        data_dir=a.data_dir,
        train_file=a.train_file,
        output_dir=a.output_dir,
        base_model=a.base_model,
        max_seq_len=a.max_seq_len,
        num_epochs=a.num_epochs,
        lr=a.lr,
        warmup_ratio=a.warmup_ratio,
        train_batch_size=a.train_batch_size,
        eval_batch_size=a.eval_batch_size,
        grad_accum=a.grad_accum,
        weight_decay=a.weight_decay,
        seed=a.seed,
        save_merged=a.save_merged,
        eval_samples=a.eval_samples,
        hf_token=a.hf_token,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
