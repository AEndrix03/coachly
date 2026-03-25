"""
Coachly NLU — Script per Google Colab (T4 GPU)
================================================
Copia questo file su Colab e lancia cella per cella,
oppure esegui tutto con: !python colab_train.py

Setup consigliato:
  Runtime → Change runtime type → T4 GPU

Il modello finale verrà salvato su Google Drive in:
  /content/drive/MyDrive/coachly_nlu/
"""

import os, sys, json, time, random, shutil
import numpy as np

# ─── 1. INSTALLA DIPENDENZE ─────────────────────────────────────────────────────
# (già installato su Colab: torch, torchvision)

def install_deps():
    import subprocess
    pkgs = ["transformers>=4.38.0", "seqeval>=1.2.2", "onnx>=1.15.0", "onnxruntime>=1.17.0"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    print("Dipendenze installate.")

# ─── 2. MOUNT DRIVE (opzionale, per salvare il modello) ─────────────────────────

def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("Drive montato.")
        return True
    except Exception:
        print("Drive non disponibile (non sei su Colab o hai saltato il mount).")
        return False

# ─── CONFIGURAZIONE ─────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class CFG:
    model_name:      str   = "xlm-roberta-base"
    data_dir:        str   = "data"
    output_dir:      str   = "output/workout_nlu"
    drive_output:    str   = "/content/drive/MyDrive/coachly_nlu"  # dove salvare su Drive
    max_length:      int   = 64
    batch_size:      int   = 32    # T4 regge 32 con fp16
    num_epochs:      int   = 15
    learning_rate:   float = 2e-5
    weight_decay:    float = 0.01
    warmup_ratio:    float = 0.1
    intent_weight:   float = 1.0
    slot_weight:     float = 1.0
    grad_clip:       float = 1.0
    seed:            int   = 42
    patience:        int   = 4
    log_every_steps: int   = 50

# ─── LABEL MAPS ─────────────────────────────────────────────────────────────────

INTENTS = ["ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE", "UNKNOWN"]
NER_TAGS = ["O", "B-EXE", "I-EXE", "B-SET", "B-REP", "B-WGT", "B-UNT", "B-MOD", "I-MOD"]

INTENT2ID = {k: i for i, k in enumerate(INTENTS)}
TAG2ID    = {k: i for i, k in enumerate(NER_TAGS)}

def load_label_maps(path):
    with open(path) as f:
        lm = json.load(f)
    return lm["intent2id"], lm["id2intent"], lm["tag2id"], lm["id2tag"]

# ─── DATASET ────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Dict, List, Tuple
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score as seq_f1

class WorkoutNLUDataset(Dataset):
    def __init__(self, path, tokenizer, intent2id, tag2id, max_length=64):
        with open(path, encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer  = tokenizer
        self.intent2id  = intent2id
        self.tag2id     = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex       = self.examples[idx]
        words    = ex["words"]
        ner_tags = ex["ner_tags"]
        intent   = self.intent2id[ex["intent"]]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids       = encoding.word_ids(batch_index=0)

        labels_ner = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                labels_ner.append(-100)
            elif wid != prev_wid:
                tag = ner_tags[wid] if wid < len(ner_tags) else "O"
                labels_ner.append(self.tag2id.get(tag, self.tag2id["O"]))
            else:
                labels_ner.append(-100)
            prev_wid = wid

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "intent_label":   torch.tensor(intent, dtype=torch.long),
            "ner_labels":     torch.tensor(labels_ner, dtype=torch.long),
        }

# ─── MODELLO ────────────────────────────────────────────────────────────────────

class WorkoutNLUModel(nn.Module):
    def __init__(self, model_name, num_intents, num_slot_labels, dropout=0.1):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size  # 768

        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_intents),
        )
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_slot_labels),
        )

    def forward(self, input_ids, attention_mask):
        out      = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq      = out.last_hidden_state       # [B, T, 768]
        cls      = seq[:, 0, :]               # [B, 768]
        return self.intent_head(cls), self.slot_head(seq)

# ─── EVALUATE ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, id2intent, id2tag, tag2id):
    model.eval()
    correct = total = 0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            il    = batch["intent_label"].to(device)
            nl    = batch["ner_labels"].to(device)

            int_log, slot_log = model(ids, mask)

            pred = int_log.argmax(-1)
            correct += (pred == il).sum().item()
            total   += il.size(0)

            pb = slot_log.argmax(-1).cpu().numpy()
            tb = nl.cpu().numpy()
            for pr, tr in zip(pb, tb):
                ps, ts = [], []
                for p, t in zip(pr, tr):
                    if t == -100: continue
                    ps.append(id2tag.get(p, "O"))
                    ts.append(id2tag.get(t, "O"))
                all_pred.append(ps)
                all_true.append(ts)

    acc = correct / total if total else 0.0
    try:
        f1 = seq_f1(all_true, all_pred, zero_division=0)
    except Exception:
        f1 = 0.0
    return {"intent_acc": acc, "slot_f1": f1, "combined": 0.5 * acc + 0.5 * f1}

# ─── EXPORT ONNX ────────────────────────────────────────────────────────────────

def export_onnx(model, tokenizer, device, output_dir):
    print("\nExport ONNX...")
    model.eval(); model.cpu()

    dummy = tokenizer(
        ["add bench press 3 sets 10 reps"],
        is_split_into_words=False,
        max_length=CFG.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    onnx_path = os.path.join(output_dir, "workout_nlu.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(dummy["input_ids"], dummy["attention_mask"]),
            f=onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["intent_logits", "slot_logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "intent_logits":  {0: "batch"},
                "slot_logits":    {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )
    print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = os.path.join(output_dir, "workout_nlu_int8.onnx")
        quantize_dynamic(onnx_path, q, weight_type=QuantType.QInt8)
        print(f"ONNX INT8: {q} ({os.path.getsize(q)/1e6:.1f} MB)")
    except Exception as e:
        print(f"Quantizzazione saltata: {e}")

    model.to(device)

# ─── DEMO INFERENCE ─────────────────────────────────────────────────────────────

def inference_demo():
    print("\n" + "=" * 60)
    print("DEMO INFERENCE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(os.path.join(CFG.output_dir, "best_model.pt"), map_location=device)

    intent2id = ckpt["intent2id"]
    tag2id    = ckpt["tag2id"]
    id2intent = {v: k for k, v in intent2id.items()}
    id2tag    = {v: k for k, v in tag2id.items()}

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(CFG.model_name)
    model = WorkoutNLUModel(
        CFG.model_name,
        num_intents=len(intent2id),
        num_slot_labels=len(tag2id),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sentences = [
        ("it", "aggiungi bench press 3 x 10 a cedimento"),
        ("it", "aggiungi squat 4 serie da 8 e push up 3 x 15 dropset"),
        ("it", "fatto deadlift 5 reps 140 kg"),
        ("it", "rimuovi lat machine"),
        ("en", "add bench press 3 sets of 8 reps 80 kg to failure"),
        ("en", "add pull ups 4 x 8 and dips 3 x 12 then leg raises 3 x 20"),
        ("en", "done bench press 8 reps 80 kg"),
        ("it", "quanto riposo tra le serie"),
        ("fr", "ajoute développé couché 3 séries de 8 reps 60 kg"),
        ("de", "füge hinzu bankdrücken 3 sätze 10 wiederholungen"),
    ]

    for lang, text in sentences:
        words = text.split()
        enc = tokenizer(
            words,
            is_split_into_words=True,
            max_length=CFG.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            int_log, slot_log = model(enc["input_ids"], enc["attention_mask"])

        probs  = torch.softmax(int_log, -1)[0]
        intent = id2intent[probs.argmax().item()]
        conf   = probs.max().item()

        slot_preds = slot_log.argmax(-1)[0].cpu().tolist()
        word_ids   = enc.word_ids(0)

        seen = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                seen[wid] = id2tag.get(slot_preds[pos], "O")

        entities = {}
        cur_type, cur_words = None, []
        for wid in sorted(seen):
            tag = seen[wid]
            if tag.startswith("B-"):
                if cur_type:
                    entities.setdefault(cur_type, []).append(" ".join(cur_words))
                cur_type, cur_words = tag[2:], [words[wid]]
            elif tag.startswith("I-") and cur_type:
                cur_words.append(words[wid])
            else:
                if cur_type:
                    entities.setdefault(cur_type, []).append(" ".join(cur_words))
                cur_type, cur_words = None, []
        if cur_type:
            entities.setdefault(cur_type, []).append(" ".join(cur_words))

        print(f'\n[{lang}] "{text}"')
        print(f"  Intent:   {intent} ({conf:.1%})")
        print(f"  Entities: {entities}")

# ─── TRAINING ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train():
    set_seed(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        use_fp16 = True
    else:
        device = torch.device("cpu")
        print("CPU mode")
        use_fp16 = False

    # Label maps
    intent2id, id2intent, tag2id, id2tag = load_label_maps(
        os.path.join(CFG.data_dir, "label_maps.json")
    )
    num_intents     = len(intent2id)
    num_slot_labels = len(tag2id)
    print(f"Intents: {num_intents} | NER tags: {num_slot_labels}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(CFG.model_name)

    # Datasets
    nw = 2 if torch.cuda.is_available() else 0
    train_ds = WorkoutNLUDataset(os.path.join(CFG.data_dir, "train.json"), tokenizer, intent2id, tag2id, CFG.max_length)
    val_ds   = WorkoutNLUDataset(os.path.join(CFG.data_dir, "val.json"),   tokenizer, intent2id, tag2id, CFG.max_length)
    test_ds  = WorkoutNLUDataset(os.path.join(CFG.data_dir, "test.json"),  tokenizer, intent2id, tag2id, CFG.max_length)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model = WorkoutNLUModel(CFG.model_name, num_intents, num_slot_labels).to(device)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer con learning rate differenziato backbone / heads
    optimizer = AdamW([
        {"params": model.roberta.parameters(),     "lr": CFG.learning_rate},
        {"params": model.intent_head.parameters(), "lr": CFG.learning_rate * 5},
        {"params": model.slot_head.parameters(),   "lr": CFG.learning_rate * 5},
    ], weight_decay=CFG.weight_decay)

    total_steps  = len(train_loader) * CFG.num_epochs
    warmup_steps = int(total_steps * CFG.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion   = nn.CrossEntropyLoss(ignore_index=-100)
    scaler           = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"\n{'='*60}")
    print(f"Training {CFG.num_epochs} epochs | batch {CFG.batch_size} | fp16={use_fp16}")
    print(f"{'='*60}\n")

    best_combined    = 0.0
    patience_counter = 0

    for epoch in range(1, CFG.num_epochs + 1):
        model.train()
        loss_i_tot = loss_s_tot = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            il   = batch["intent_label"].to(device)
            nl   = batch["ner_labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_fp16):
                int_log, slot_log = model(ids, mask)
                loss_i = intent_criterion(int_log, il)
                loss_s = slot_criterion(slot_log.view(-1, num_slot_labels), nl.view(-1))
                loss   = CFG.intent_weight * loss_i + CFG.slot_weight * loss_s

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_i_tot += loss_i.item()
            loss_s_tot += loss_s.item()

            if step % CFG.log_every_steps == 0:
                print(f"  Ep {epoch:2d} step {step:4d}/{len(train_loader)} "
                      f"| li={loss_i_tot/step:.4f} ls={loss_s_tot/step:.4f} "
                      f"| lr={scheduler.get_last_lr()[0]:.2e}")

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:2d} — li={loss_i_tot/len(train_loader):.4f} "
              f"ls={loss_s_tot/len(train_loader):.4f} — {elapsed:.1f}s")

        metrics = evaluate(model, val_loader, device, id2intent, id2tag, tag2id)
        print(f"         Val -> intent_acc={metrics['intent_acc']:.4f} "
              f"slot_f1={metrics['slot_f1']:.4f} combined={metrics['combined']:.4f}")

        if metrics["combined"] > best_combined:
            best_combined    = metrics["combined"]
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "intent2id":   intent2id,
                "tag2id":      tag2id,
            }, os.path.join(CFG.output_dir, "best_model.pt"))
            print(f"         [BEST] {best_combined:.4f} — salvato")
        else:
            patience_counter += 1
            print(f"         Patience {patience_counter}/{CFG.patience}")
            if patience_counter >= CFG.patience:
                print(f"\nEarly stopping a epoch {epoch}")
                break
        print()

    # Test finale
    print("=" * 60)
    ckpt = torch.load(os.path.join(CFG.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tm = evaluate(model, test_loader, device, id2intent, id2tag, tag2id)
    print(f"Test -> intent_acc={tm['intent_acc']:.4f} slot_f1={tm['slot_f1']:.4f} combined={tm['combined']:.4f}")

    with open(os.path.join(CFG.output_dir, "results.json"), "w") as f:
        json.dump({"best_val": best_combined, "test": tm}, f, indent=2)

    export_onnx(model, tokenizer, device, CFG.output_dir)
    print("\nTraining completato!")

# ─── SALVA SU DRIVE ─────────────────────────────────────────────────────────────

def save_to_drive():
    if not os.path.exists("/content/drive"):
        print("Drive non montato, skip.")
        return
    os.makedirs(CFG.drive_output, exist_ok=True)
    shutil.copytree(CFG.output_dir, CFG.drive_output, dirs_exist_ok=True)
    print(f"Modello salvato su Drive: {CFG.drive_output}")

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Coachly NLU — Colab Training ===\n")

    install_deps()
    mount_drive()

    # Se il dataset non esiste, generalo
    if not os.path.exists("data/train.json"):
        print("Dataset non trovato, genero...")
        import subprocess
        subprocess.run([sys.executable, "generate-dataset.py"], check=True)

    train()
    inference_demo()
    save_to_drive()
