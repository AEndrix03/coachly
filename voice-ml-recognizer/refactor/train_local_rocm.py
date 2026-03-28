#!/usr/bin/env python3
"""
train_local_rocm.py — Local training for Coachly NLU on AMD GPU (Windows + DirectML)

Usa un training loop manuale per forzare l'uso del device DirectML (privateuseone:0),
perche' HF Trainer non supporta nativamento DirectML e cadrebbe su CPU.

Usage:
  python train_local_rocm.py
  python train_local_rocm.py --data_dir data_v2 --train_file train_aug.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

# ── GPU detection ─────────────────────────────────────────────────────────────

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

_DEVICE = None

if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
    print(f"[GPU] ROCm/CUDA  : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    try:
        import torch_directml as dml
        _DEVICE = dml.device()
        print(f"[GPU] DirectML   : {dml.device_name(0)}")
        # Smoke test
        _ = torch.ones(2, 2).to(_DEVICE) + torch.ones(2, 2).to(_DEVICE)
        print("[GPU] Smoke test : OK")
    except Exception as e:
        print(f"[ERROR] Nessuna GPU trovata: {e}")
        print("  Esegui setup_windows_gpu.ps1 per installare le dipendenze GPU.")
        raise SystemExit(1)

from accelerate import dispatch_model
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    max_seq_len:  int   = 256
    num_epochs:   int   = 5
    lr:           float = 2e-4
    warmup_steps: int   = 200
    train_batch:  int   = 2
    grad_accum:   int   = 8    # effective batch = 16
    gpu_layers:   int   = 12   # layer su DirectML (0-24); gli altri su CPU — regola per VRAM
    weight_decay: float = 0.01
    lora_r:       int   = 16
    lora_alpha:   int   = 32
    lora_dropout: float = 0.05
    seed:         int   = 42
    eval_samples: int   = 200
    save_merged:  bool  = False
    log_every:    int   = 50   # steps


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(cfg: TrainConfig):
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


def collate_fn(batch):
    return {
        "input_ids":      torch.tensor([x["input_ids"]      for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
        "labels":         torch.tensor([x["labels"]         for x in batch], dtype=torch.long),
    }


# ── Model setup ───────────────────────────────────────────────────────────────

def _build_device_map(n_total_layers: int, gpu_layers: int) -> dict:
    """
    Mappa layer per layer: i primi gpu_layers su DirectML, il resto su CPU.
    embed_tokens e lm_head restano sempre su CPU (input/output del DataLoader).
    """
    gpu_str = str(_DEVICE)   # "privateuseone:0"
    dm = {"model.embed_tokens": "cpu"}
    for i in range(n_total_layers):
        dm[f"model.layers.{i}"] = gpu_str if i < gpu_layers else "cpu"
    dm["model.norm"] = "cpu"
    dm["lm_head"]    = "cpu"
    return dm


def load_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_layers = cfg.gpu_layers  # quanti layer vanno su GPU
    print(f"  Carico base model fp16 su CPU (split: {n_layers} layer su GPU, resto CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False

    # Applica LoRA prima del dispatch: i param LoRA vengono creati su CPU
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    # LoRA in fp32 per stabilità, tutto ancora su CPU
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # Dispatch: sposta i layer 0..n-1 su DirectML, il resto resta su CPU.
    # Le module name nel PEFT model hanno prefisso "base_model.model."
    total_layers = len(base.model.layers)
    gpu_str = str(_DEVICE)
    device_map = {
        "base_model.model.model.embed_tokens": "cpu",
        **{
            f"base_model.model.model.layers.{i}": (gpu_str if i < n_layers else "cpu")
            for i in range(total_layers)
        },
        "base_model.model.model.norm": "cpu",
        "base_model.model.lm_head":    "cpu",
    }
    print(f"  Dispatch: {n_layers}/{total_layers} layer su {gpu_str}, resto su CPU...")
    model = dispatch_model(model, device_map=device_map)
    return model, tokenizer


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float = 1e-5) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_loss(model, loader_val, _ignored_device=None) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader_val:
            # Input su CPU — dispatch_model sposta le attivazioni layer per layer
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += out.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def quick_eval_accuracy(model, tokenizer, ds_test, cfg: TrainConfig) -> Dict:
    model.eval()
    samples = ds_test.select(range(min(cfg.eval_samples, len(ds_test))))
    correct = total = json_ok = 0

    for ex in samples:
        msgs = ex["messages"]
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids  # stays on CPU; dispatch_model routes layers
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
                correct += 1
        except Exception:
            pass
        total += 1

    result = {
        "total":          total,
        "action_acc":     round(correct / total, 4) if total else 0,
        "json_valid_pct": round(json_ok / total, 4) if total else 0,
    }
    print(f"  action_acc={result['action_acc']:.2%}  json_valid={result['json_valid_pct']:.2%}")
    model.train()
    return result


# ── Manual training loop ──────────────────────────────────────────────────────

def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Carico i dati...")
    ds = load_splits(cfg)
    print(f"  train: {len(ds['train']):,}  val: {len(ds['validation']):,}  test: {len(ds['test']):,}")

    print(f"\n[2/4] Carico il modello...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print(f"\n[3/4] Tokenizzazione...")
    tok_fn = build_tokenize_fn(tokenizer, cfg.max_seq_len)
    remove_cols = [c for c in ds["train"].column_names if c != "messages"]
    ds_tok = ds.map(tok_fn, batched=True, batch_size=256, remove_columns=remove_cols)
    ds_tok = ds_tok.remove_columns(["messages"])
    ds_tok.set_format("numpy")

    loader_train = DataLoader(
        ds_tok["train"], batch_size=cfg.train_batch,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    loader_val = DataLoader(
        ds_tok["validation"], batch_size=cfg.train_batch * 2,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Due ottimizzatori separati: param LoRA su GPU e param LoRA su CPU
    # (AdamW non supporta param su device misti nello stesso gruppo)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gpu_params = [p for _, p in trainable if p.device.type != "cpu"]
    cpu_params = [p for _, p in trainable if p.device.type == "cpu"]
    print(f"  Param trainable: {len(gpu_params)} su GPU, {len(cpu_params)} su CPU")

    opts = []
    if gpu_params:
        opts.append(AdamW(gpu_params, lr=cfg.lr, weight_decay=cfg.weight_decay))
    if cpu_params:
        opts.append(AdamW(cpu_params, lr=cfg.lr, weight_decay=cfg.weight_decay))

    def set_lr(lr):
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = lr

    def opts_step():
        for opt in opts:
            opt.step()

    def opts_zero_grad():
        for opt in opts:
            opt.zero_grad()

    steps_per_epoch = math.ceil(len(loader_train) / cfg.grad_accum)
    total_steps     = steps_per_epoch * cfg.num_epochs

    print(f"\n[4/4] Training (split CPU+GPU) ...")
    print(f"  epochs={cfg.num_epochs}  steps/epoch={steps_per_epoch}  total={total_steps}")
    print(f"  batch={cfg.train_batch}  grad_accum={cfg.grad_accum}  effective_batch={cfg.train_batch * cfg.grad_accum}")

    best_val_loss = float("inf")
    global_step   = 0
    t0            = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        opts_zero_grad()
        running_loss = 0.0
        accum_count  = 0

        for step_in_epoch, batch in enumerate(loader_train, 1):
            # Input su CPU: embed_tokens e' su CPU, dispatch_model gestisce il resto
            ids  = batch["input_ids"]
            attn = batch["attention_mask"]
            lbls = batch["labels"]

            out  = model(input_ids=ids, attention_mask=attn, labels=lbls)
            loss = out.loss / cfg.grad_accum
            loss.backward()

            running_loss += out.loss.item()
            accum_count  += 1

            if accum_count == cfg.grad_accum or step_in_epoch == len(loader_train):
                lr_now = get_lr(global_step, cfg.warmup_steps, total_steps, cfg.lr)
                set_lr(lr_now)

                all_trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)

                opts_step()
                opts_zero_grad()

                global_step += 1
                accum_count  = 0

                if global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    elapsed  = time.time() - t0
                    eta      = elapsed / global_step * (total_steps - global_step)
                    print(f"  epoch {epoch}/{cfg.num_epochs}  "
                          f"step {global_step}/{total_steps}  "
                          f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                          f"ETA={eta/60:.0f}min")
                    running_loss = 0.0

        # End of epoch: eval (input su CPU, dispatch_model gestisce device)
        val_loss = eval_loss(model, loader_val, "cpu")
        elapsed  = (time.time() - t0) / 60
        print(f"\n  [Epoch {epoch}] val_loss={val_loss:.4f}  elapsed={elapsed:.0f}min")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = out_dir / "best_adapter"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            print(f"  Checkpoint salvato (val_loss={best_val_loss:.4f}): {ckpt}")

    total_min = (time.time() - t0) / 60
    print(f"\nTraining completato in {total_min:.0f} min")

    # Salva adapter finale
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter finale: {adapter_dir}")

    # Accuracy eval
    print(f"\nQuick eval su {cfg.eval_samples} campioni di test...")
    eval_res = quick_eval_accuracy(model, tokenizer, ds["test"], cfg)
    eval_res["train_time_min"]  = round(total_min, 1)
    eval_res["best_val_loss"]   = round(best_val_loss, 4)
    (out_dir / "eval.json").write_text(json.dumps(eval_res, indent=2))
    print(f"Eval salvato: {out_dir / 'eval.json'}")

    if cfg.save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_dir / "merged"))
        tokenizer.save_pretrained(str(out_dir / "merged"))
        print(f"Merged: {out_dir / 'merged'}")


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
    p.add_argument("--train_batch",  type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_samples", type=int,   default=200)
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument("--save_merged",  action="store_true")
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    cfg = parse_args()
    print(f"Output dir: {cfg.output_dir}")
    train(cfg)
