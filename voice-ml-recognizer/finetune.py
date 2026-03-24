"""
Coachly NLU Fine-tuning — XLM-RoBERTa Joint Intent + NER
Hardware target: AMD RX6600 (gfx1032) con ROCm 6

Modello: xlm-roberta-base
Task: Intent classification (CLS token) + Slot filling BIO NER (token level)
Loss: CrossEntropy intent + CrossEntropy NER (slot), combined

Setup ROCm:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
    pip install transformers datasets seqeval accelerate

Run:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python finetune.py
"""

import os
import json
import math
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ─── ROCm / RX6600 setup ───────────────────────────────────────────────────────
# RX6600 è gfx1032 (RDNA2); ROCm non la supporta ufficialmente ma con override funziona
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaModel,
    XLMRobertaConfig,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_report

# ─── CONFIG ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    model_name:        str   = "xlm-roberta-base"
    data_dir:          str   = "data"
    output_dir:        str   = "output/workout_nlu"
    max_length:        int   = 64        # frasi vocali brevi, 64 è sufficiente
    batch_size:        int   = 32        # RX6600 8GB: 32 safe con fp16
    num_epochs:        int   = 15
    learning_rate:     float = 2e-5
    weight_decay:      float = 0.01
    warmup_ratio:      float = 0.1
    intent_weight:     float = 1.0       # peso loss intent nella combined loss
    slot_weight:       float = 1.0       # peso loss NER
    grad_clip:         float = 1.0
    seed:              int   = 42
    use_fp16:          bool  = True      # Mixed precision — essenziale su ROCm
    patience:          int   = 4         # Early stopping su val F1 combinata
    log_every_steps:   int   = 50
    eval_every_epochs: int   = 1

CFG = TrainConfig()

# ─── LABEL MAPS ────────────────────────────────────────────────────────────────

def load_label_maps(path: str) -> Tuple[Dict,Dict,Dict,Dict]:
    with open(path) as f:
        lm = json.load(f)
    return lm["intent2id"], lm["id2intent"], lm["tag2id"], lm["id2tag"]

# ─── DATASET ───────────────────────────────────────────────────────────────────

class WorkoutNLUDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: XLMRobertaTokenizerFast,
        intent2id: Dict[str,int],
        tag2id: Dict[str,int],
        max_length: int = 64,
    ):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        self.tokenizer  = tokenizer
        self.intent2id  = intent2id
        self.tag2id     = tag2id
        self.max_length = max_length
        self.examples   = raw

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        words    : List[str] = ex["words"]
        ner_tags : List[str] = ex["ner_tags"]
        intent   : int       = self.intent2id[ex["intent"]]

        # Tokenize con word_ids per allineare i NER labels alle subword
        encoding = self.tokenizer(
            words,
            is_split_into_words = True,
            max_length           = self.max_length,
            padding              = "max_length",
            truncation           = True,
            return_tensors       = "pt",
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids       = encoding.word_ids(batch_index=0)  # None per special tokens

        # Allinea NER labels: prima subword → label reale, resto → -100 (ignorato)
        labels_ner = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                labels_ner.append(-100)
            elif wid != prev_word_id:
                tag = ner_tags[wid] if wid < len(ner_tags) else "O"
                labels_ner.append(self.tag2id.get(tag, self.tag2id["O"]))
            else:
                labels_ner.append(-100)  # subword successiva: ignora
            prev_word_id = wid

        labels_ner = torch.tensor(labels_ner, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "intent_label":   torch.tensor(intent, dtype=torch.long),
            "ner_labels":     labels_ner,
        }


# ─── MODEL ─────────────────────────────────────────────────────────────────────

class WorkoutNLUModel(nn.Module):
    """
    XLM-RoBERTa base con due teste:
      - Intent: classificazione su [CLS] token  → num_intents classi
      - NER:    classificazione token-level      → num_slot_labels classi
    """

    def __init__(
        self,
        model_name:        str,
        num_intents:       int,
        num_slot_labels:   int,
        dropout:           float = 0.1,
    ):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size  # 768 per base

        # Intent head
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_intents),
        )

        # NER/Slot head
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_slot_labels),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence = out.last_hidden_state   # [B, T, H]
        cls_repr = sequence[:, 0, :]       # [B, H]  — [CLS]

        intent_logits = self.intent_head(cls_repr)   # [B, num_intents]
        slot_logits   = self.slot_head(sequence)     # [B, T, num_slots]

        return intent_logits, slot_logits


# ─── TRAINING UTILS ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model:      WorkoutNLUModel,
    loader:     DataLoader,
    device:     torch.device,
    id2intent:  Dict[int,str],
    id2tag:     Dict[int,str],
    tag2id:     Dict[str,int],
) -> Dict[str, float]:
    model.eval()

    intent_correct = 0
    intent_total   = 0
    all_pred_tags: List[List[str]] = []
    all_true_tags: List[List[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels  = batch["intent_label"].to(device)
            ner_labels     = batch["ner_labels"].to(device)

            intent_logits, slot_logits = model(input_ids, attention_mask)

            # Intent accuracy
            pred_intents = intent_logits.argmax(dim=-1)
            intent_correct += (pred_intents == intent_labels).sum().item()
            intent_total   += intent_labels.size(0)

            # NER F1 (seqeval vuole stringhe, no -100)
            pred_tags_batch = slot_logits.argmax(dim=-1).cpu().numpy()  # [B, T]
            true_tags_batch = ner_labels.cpu().numpy()                  # [B, T]

            for pred_row, true_row in zip(pred_tags_batch, true_tags_batch):
                pred_seq, true_seq = [], []
                for p, t in zip(pred_row, true_row):
                    if t == -100:
                        continue
                    pred_seq.append(id2tag.get(p, "O"))
                    true_seq.append(id2tag.get(t, "O"))
                all_pred_tags.append(pred_seq)
                all_true_tags.append(true_seq)

    intent_acc = intent_correct / intent_total if intent_total > 0 else 0.0

    # seqeval F1 — ignora O in micro avg
    try:
        slot_f1 = seq_f1(all_true_tags, all_pred_tags, zero_division=0)
    except Exception:
        slot_f1 = 0.0

    combined = 0.5 * intent_acc + 0.5 * slot_f1

    return {
        "intent_acc": intent_acc,
        "slot_f1":    slot_f1,
        "combined":   combined,
    }


# ─── MAIN TRAIN ────────────────────────────────────────────────────────────────

def train():
    set_seed(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("ATTENZIONE: GPU non trovata, training su CPU")

    # ── Label maps ──────────────────────────────────────────────────────────────
    intent2id, id2intent, tag2id, id2tag = load_label_maps(
        os.path.join(CFG.data_dir, "label_maps.json")
    )
    num_intents     = len(intent2id)
    num_slot_labels = len(tag2id)
    print(f"Intents: {num_intents}  |  NER tags: {num_slot_labels}")

    # ── Tokenizer ───────────────────────────────────────────────────────────────
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(CFG.model_name)

    # ── Datasets ────────────────────────────────────────────────────────────────
    train_ds = WorkoutNLUDataset(
        os.path.join(CFG.data_dir, "train.json"),
        tokenizer, intent2id, tag2id, CFG.max_length,
    )
    val_ds = WorkoutNLUDataset(
        os.path.join(CFG.data_dir, "val.json"),
        tokenizer, intent2id, tag2id, CFG.max_length,
    )
    test_ds = WorkoutNLUDataset(
        os.path.join(CFG.data_dir, "test.json"),
        tokenizer, intent2id, tag2id, CFG.max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ───────────────────────────────────────────────────────────────────
    model = WorkoutNLUModel(
        model_name      = CFG.model_name,
        num_intents     = num_intents,
        num_slot_labels = num_slot_labels,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri totali:    {total_params:,}")
    print(f"Parametri trainable: {trainable_params:,}")

    # ── Optimizer & Scheduler ───────────────────────────────────────────────────
    # Differenzia learning rate: backbone più basso, heads più alto
    optimizer = AdamW([
        {"params": model.roberta.parameters(),    "lr": CFG.learning_rate},
        {"params": model.intent_head.parameters(),"lr": CFG.learning_rate * 5},
        {"params": model.slot_head.parameters(),  "lr": CFG.learning_rate * 5},
    ], weight_decay=CFG.weight_decay)

    total_steps   = len(train_loader) * CFG.num_epochs
    warmup_steps  = int(total_steps * CFG.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Loss ────────────────────────────────────────────────────────────────────
    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion   = nn.CrossEntropyLoss(ignore_index=-100)

    # ── fp16 scaler (ROCm supporta GradScaler come CUDA) ────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_fp16)

    # ── Training loop ───────────────────────────────────────────────────────────
    best_combined = 0.0
    patience_counter = 0
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training {CFG.num_epochs} epochs | batch {CFG.batch_size} | fp16={CFG.use_fp16}")
    print(f"{'='*60}\n")

    for epoch in range(1, CFG.num_epochs + 1):
        model.train()
        epoch_loss_intent = 0.0
        epoch_loss_slot   = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels  = batch["intent_label"].to(device)
            ner_labels     = batch["ner_labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=CFG.use_fp16):
                intent_logits, slot_logits = model(input_ids, attention_mask)

                loss_intent = intent_criterion(intent_logits, intent_labels)
                loss_slot   = slot_criterion(
                    slot_logits.view(-1, num_slot_labels),
                    ner_labels.view(-1),
                )
                loss = CFG.intent_weight * loss_intent + CFG.slot_weight * loss_slot

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss_intent += loss_intent.item()
            epoch_loss_slot   += loss_slot.item()
            global_step       += 1

            if step % CFG.log_every_steps == 0:
                avg_intent = epoch_loss_intent / step
                avg_slot   = epoch_loss_slot   / step
                lr_now     = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:2d} step {step:4d}/{len(train_loader)} "
                      f"| loss_intent={avg_intent:.4f} loss_slot={avg_slot:.4f} "
                      f"| lr={lr_now:.2e}")

        elapsed = time.time() - t0
        avg_i = epoch_loss_intent / len(train_loader)
        avg_s = epoch_loss_slot   / len(train_loader)
        print(f"\nEpoch {epoch:2d} — loss_intent={avg_i:.4f} loss_slot={avg_s:.4f} "
              f"— {elapsed:.1f}s")

        # ── Valutazione ─────────────────────────────────────────────────────────
        if epoch % CFG.eval_every_epochs == 0:
            metrics = evaluate(model, val_loader, device, id2intent, id2tag, tag2id)
            print(f"         Val → intent_acc={metrics['intent_acc']:.4f} "
                  f"slot_f1={metrics['slot_f1']:.4f} "
                  f"combined={metrics['combined']:.4f}")

            if metrics["combined"] > best_combined:
                best_combined   = metrics["combined"]
                patience_counter = 0
                # Salva best model
                torch.save({
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "metrics":        metrics,
                    "config":         CFG.__dict__,
                    "intent2id":      intent2id,
                    "tag2id":         tag2id,
                }, os.path.join(CFG.output_dir, "best_model.pt"))
                print(f"         ✓ Nuovo best: {best_combined:.4f} — salvato")
            else:
                patience_counter += 1
                print(f"         Patience {patience_counter}/{CFG.patience}")
                if patience_counter >= CFG.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print()

    # ── Test finale ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Test finale sul best model")
    checkpoint = torch.load(os.path.join(CFG.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, id2intent, id2tag, tag2id)
    print(f"Test → intent_acc={test_metrics['intent_acc']:.4f} "
          f"slot_f1={test_metrics['slot_f1']:.4f} "
          f"combined={test_metrics['combined']:.4f}")

    # Salva risultati
    results = {
        "best_val_combined": best_combined,
        "test_metrics":      test_metrics,
    }
    with open(os.path.join(CFG.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Export ONNX per deployment on-device ────────────────────────────────────
    export_onnx(model, tokenizer, device, CFG.output_dir)
    print("\nTraining completato ✓")


# ─── ONNX EXPORT ───────────────────────────────────────────────────────────────

def export_onnx(
    model:     WorkoutNLUModel,
    tokenizer: XLMRobertaTokenizerFast,
    device:    torch.device,
    output_dir: str,
):
    """
    Esporta il modello in ONNX per inference efficiente on-device o su mobile.
    Opzionalmente quantizzabile con onnxruntime quantize_dynamic.
    """
    print("\nEsportazione ONNX...")
    model.eval()
    model.cpu()  # Export su CPU per compatibilità massima

    # Dummy input
    dummy = tokenizer(
        ["add bench press 3 sets 10 reps"],
        is_split_into_words = False,
        max_length           = CFG.max_length,
        padding              = "max_length",
        truncation           = True,
        return_tensors       = "pt",
    )

    onnx_path = os.path.join(output_dir, "workout_nlu.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            args              = (dummy["input_ids"], dummy["attention_mask"]),
            f                 = onnx_path,
            input_names       = ["input_ids", "attention_mask"],
            output_names      = ["intent_logits", "slot_logits"],
            dynamic_axes      = {
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "intent_logits":  {0: "batch"},
                "slot_logits":    {0: "batch", 1: "seq"},
            },
            opset_version     = 17,
            do_constant_folding = True,
        )

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"ONNX salvato → {onnx_path} ({size_mb:.1f} MB)")

    # Quantizzazione INT8 (opzionale, riduce ~4x)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_path = os.path.join(output_dir, "workout_nlu_int8.onnx")
        quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
        q_size = os.path.getsize(q_path) / 1e6
        print(f"ONNX INT8 → {q_path} ({q_size:.1f} MB)")
    except ImportError:
        print("onnxruntime non disponibile, skip quantizzazione")

    model.to(device)  # Rimetti su device per eventuale uso successivo


# ─── INFERENCE DEMO ────────────────────────────────────────────────────────────

def inference_demo():
    """
    Esempio di inference con il modello salvato.
    Mostra come usarlo a runtime nell'app.
    """
    print("\n" + "="*60)
    print("Inference demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load("output/workout_nlu/best_model.pt", map_location=device)

    intent2id: Dict[str,int] = ckpt["intent2id"]
    tag2id:    Dict[str,int] = ckpt["tag2id"]
    id2intent = {v:k for k,v in intent2id.items()}
    id2tag    = {v:k for k,v in tag2id.items()}

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(CFG.model_name)
    model = WorkoutNLUModel(
        CFG.model_name,
        num_intents     = len(intent2id),
        num_slot_labels = len(tag2id),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_sentences = [
        ("it", "aggiungi squat 4 serie da 10 a 100 kg"),
        ("en", "add bench press 3 sets of 8 reps 80 kg"),
        ("fr", "ajoute développé couché 3 séries de 8 reps 60 kg"),
        ("de", "füge hinzu bankdrücken 3 sätze 10 wiederholungen 70 kg"),
        ("es", "agrega sentadilla 4 series de 12 reps 90 kg"),
        ("it", "fatto squat 10 reps"),
        ("en", "done bench press 8 reps 80 kg"),
        ("it", "rimuovi panca piana"),
        ("en", "how many calories did I burn"),
        ("it", "aggiungi panca piana 3 x 8 e trazioni 4 x 6"),
    ]

    for lang, text in test_sentences:
        words = text.split()
        enc = tokenizer(
            words,
            is_split_into_words = True,
            max_length           = CFG.max_length,
            padding              = "max_length",
            truncation           = True,
            return_tensors       = "pt",
        ).to(device)

        with torch.no_grad():
            intent_logits, slot_logits = model(enc["input_ids"], enc["attention_mask"])

        intent_probs = torch.softmax(intent_logits, dim=-1)[0]
        intent_pred  = intent_probs.argmax().item()
        intent_conf  = intent_probs[intent_pred].item()

        slot_preds = slot_logits.argmax(dim=-1)[0].cpu().numpy()
        word_ids   = enc.word_ids(batch_index=0)

        # Ricostruisci slot per word
        slot_per_word: Dict[int, str] = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in slot_per_word:
                tag = id2tag.get(slot_preds[pos], "O")
                if tag != "O":
                    slot_per_word[wid] = tag

        # Estrai entità
        entities: Dict[str, List] = {}
        current_entity = None
        current_words  = []
        for wid in sorted(set(w for w in word_ids if w is not None)):
            tag = slot_per_word.get(wid, "O")
            word = words[wid] if wid < len(words) else ""
            if tag.startswith("B-"):
                if current_entity:
                    entities.setdefault(current_entity, []).append(" ".join(current_words))
                current_entity = tag[2:]
                current_words  = [word]
            elif tag.startswith("I-") and current_entity:
                current_words.append(word)
            else:
                if current_entity:
                    entities.setdefault(current_entity, []).append(" ".join(current_words))
                    current_entity = None
                    current_words  = []
        if current_entity:
            entities.setdefault(current_entity, []).append(" ".join(current_words))

        print(f"\n[{lang}] \"{text}\"")
        print(f"  Intent:   {id2intent[intent_pred]} (conf={intent_conf:.3f})")
        print(f"  Entities: {entities}")


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        inference_demo()
    else:
        train()
        inference_demo()