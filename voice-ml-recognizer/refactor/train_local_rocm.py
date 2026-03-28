#!/usr/bin/env python3
"""
train_local_rocm.py — Local training for Coachly NLU on AMD GPU (Windows/ROCm)

Supporta due backend AMD su Windows:
  - ROCm (HIP SDK): torch.cuda con HSA_OVERRIDE_GFX_VERSION=10.3.0
  - DirectML:       torch_directml (fallback se ROCm non disponibile)

Nessun bitsandbytes / quantizzazione (non serve per 0.5B, evita problemi ROCm).
LoRA in fp16 + gradient checkpointing: ~4-5 GB VRAM, ok per RX 6600 8GB.

Usage:
  python train_local_rocm.py
  python train_local_rocm.py --data_dir data_v2 --train_file train_aug.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# ── GPU backend detection ─────────────────────────────────────────────────────
# RX 6600 = gfx1032: ROCm lo tratta come gfx1030 con questo override
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch

_USE_DML = False
_DEVICE  = "cpu"

if torch.cuda.is_available():
    _DEVICE = "cuda"
    print(f"[GPU] ROCm/CUDA: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    try:
        import torch_directml
        _DEVICE  = torch_directml.device()
        _USE_DML = True
        print(f"[GPU] DirectML: {torch_directml.device_name(0)}")
    except ImportError:
        print("[WARNING] Nessuna GPU rilevata — training su CPU (lento).")
        print("  Esegui: python setup_windows_gpu.ps1  (oppure: setup_windows_gpu.bat)")

assert str(_DEVICE) != "cpu", (
    "GPU non trovata. Esegui prima setup_windows_gpu.ps1 per installare le dipendenze GPU."
)

from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Coachly NLU. Convert workout speech-to-text into strict JSON.\n"
    "Return ONLY valid JSON, no markdown.\nSchema:\n"
    "{\n"
    '  "action": "ADD_EXERCISE|LOG_SET|UPDATE_SET|DELETE_EXERCISE|UNKNOWN",\n'
    '  "items": [\n    {\n'
    '      "exercise": string,\n      "sets": integer|null,\n'
    '      "reps": integer|null,\n      "weight": number|null,\n'
    '      "unit": "kg"|"lbs"|null,\n'
    '      "modifier": "to_failure"|"dropset"|"superset"|"amrap"|"pause"|null\n'
    "    }\n  ]\n}"
)


@dataclass
class TrainConfig:
    data_dir:     str   = "data_v2"
    train_file:   str   = "train_aug.jsonl"
    output_dir:   str   = "output/rocm_lora"
    base_model:   str   = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_len:  int   = 512
    num_epochs:   int   = 5
    lr:           float = 2e-4
    warmup_ratio: float = 0.05
    train_batch:  int   = 4    # riduci a 2 se OOM
    eval_batch:   int   = 4
    grad_accum:   int   = 4    # effective batch = 16
    weight_decay: float = 0.01
    lora_r:       int   = 16
    lora_alpha:   int   = 32
    lora_dropout: float = 0.05
    seed:         int   = 42
    eval_samples: int   = 200
    save_merged:  bool  = False


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(cfg: TrainConfig) -> DatasetDict:
    d = Path(cfg.data_dir)
    files = {
        "train":      str(d / cfg.train_file),
        "validation": str(d / "val.jsonl"),
        "test":       str(d / "test.jsonl"),
    }
    for name, p in files.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"Split '{name}' mancante: {p}")
    return load_dataset("json", data_files=files)


# ── Tokenisation ──────────────────────────────────────────────────────────────

def build_tokenize_fn(tokenizer, max_len: int):
    def tokenize(batch):
        input_ids_list, attn_list, labels_list = [], [], []
        for msgs in batch["messages"]:
            prompt = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True
            )
            full = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(full,   add_special_tokens=False)["input_ids"]

            if len(full_ids) > max_len:
                full_ids   = full_ids[:max_len]
                prompt_ids = prompt_ids[:max_len]

            n_prompt = len(prompt_ids)
            labels   = [-100] * n_prompt + full_ids[n_prompt:]
            pad      = max_len - len(full_ids)
            full_ids += [tokenizer.pad_token_id] * pad
            labels   += [-100] * pad
            attn      = [1] * (max_len - pad) + [0] * pad

            input_ids_list.append(full_ids)
            attn_list.append(attn)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": labels_list}

    return tokenize


# ── Model & LoRA setup ────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: TrainConfig):
    # DirectML non supporta device_map="auto" — carichiamo su CPU poi spostiamo
    if _USE_DML:
        dtype      = torch.float32   # DirectML non supporta fp16 per training
        device_arg = "cpu"
    else:
        dtype      = torch.float16
        device_arg = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=dtype,
        device_map=device_arg,
    )
    model.config.use_cache = False  # richiesto da gradient checkpointing

    if _USE_DML:
        model = model.to(_DEVICE)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Quick eval ────────────────────────────────────────────────────────────────

def quick_eval(model, tokenizer, ds_test, cfg: TrainConfig) -> Dict:
    device = next(model.parameters()).device
    model.eval()
    samples = ds_test.select(range(min(cfg.eval_samples, len(ds_test))))

    correct_action = total = json_ok = 0
    for ex in samples:
        msgs = ex["messages"]
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            pred, _ = json.JSONDecoder().raw_decode(raw)
            json_ok += 1
            if pred.get("action") == ex["action"]:
                correct_action += 1
        except Exception:
            pass
        total += 1

    result = {
        "total":          total,
        "action_acc":     round(correct_action / total, 4) if total else 0,
        "json_valid_pct": round(json_ok / total, 4) if total else 0,
    }
    print(f"\nQuick eval ({total} campioni): action_acc={result['action_acc']:.2%}  json_valid={result['json_valid_pct']:.2%}")
    return result


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig):
    import inspect
    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Carico i dati...")
    ds = load_splits(cfg)
    print(f"  train: {len(ds['train']):,}  val: {len(ds['validation']):,}  test: {len(ds['test']):,}")

    print("\n[2/4] Carico il modello...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print("\n[3/4] Tokenizzazione...")
    tok_fn = build_tokenize_fn(tokenizer, cfg.max_seq_len)
    ds_tok = ds.map(
        tok_fn, batched=True, batch_size=256,
        remove_columns=[c for c in ds["train"].column_names if c != "messages"],
    )
    ds_tok.set_format("torch")

    # DirectML non supporta fp16 né gradient_checkpointing stabile
    use_fp16 = not _USE_DML and torch.cuda.is_available()
    use_gc   = not _USE_DML

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch,
        per_device_eval_batch_size=cfg.eval_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=use_fp16,
        gradient_checkpointing=use_gc,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        dataloader_num_workers=0,
        seed=cfg.seed,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"],
        data_collator=collator,
    )
    sig = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer

    print(f"\n[4/4] Training (backend: {'DirectML' if _USE_DML else 'ROCm/CUDA'})...")
    t0 = time.time()
    Trainer(**trainer_kwargs).train()
    elapsed = time.time() - t0
    print(f"  Completato in {elapsed/60:.1f} min")

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  Adapter salvato in: {adapter_dir}")

    eval_res = quick_eval(model, tokenizer, ds["test"], cfg)
    eval_res["train_time_min"] = round(elapsed / 60, 1)
    (out / "eval.json").write_text(json.dumps(eval_res, indent=2))

    if cfg.save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"))
        tokenizer.save_pretrained(str(out / "merged"))
        print(f"  Modello merged in: {out / 'merged'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data_v2")
    p.add_argument("--train_file",   default="train_aug.jsonl")
    p.add_argument("--output_dir",   default="output/rocm_lora")
    p.add_argument("--base_model",   default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max_seq_len",  type=int,   default=512)
    p.add_argument("--num_epochs",   type=int,   default=5)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--train_batch",  type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_samples", type=int,   default=200)
    p.add_argument("--save_merged",  action="store_true")
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
