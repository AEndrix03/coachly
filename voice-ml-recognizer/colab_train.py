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

import os, sys, json, time, random, shutil, traceback
import numpy as np

# ─── LOGGING SU FILE + STDOUT ────────────────────────────────────────────────────
# Tutto quello che viene stampato finisce anche nel log file.

class _Tee:
    """Duplica stdout/stderr su file."""
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh     = fh
    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        try: self._fh.write(data); self._fh.flush()
        except Exception: pass
    def flush(self):
        self._stream.flush()
        try: self._fh.flush()
        except Exception: pass
    def __getattr__(self, attr):
        return getattr(self._stream, attr)

def _setup_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh

# ─── 1. INSTALLA DIPENDENZE ─────────────────────────────────────────────────────

def install_deps():
    import subprocess
    pkgs = [
        "transformers>=4.38.0",
        "seqeval>=1.2.2",
        "onnx>=1.15.0",
        "onnxruntime>=1.17.0",
        "pytorch-crf>=0.7.2",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    print("Dipendenze installate.")

# ─── 2. MOUNT DRIVE ─────────────────────────────────────────────────────────────

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
    drive_output:    str   = "/content/drive/MyDrive/coachly_nlu"
    max_length:      int   = 96
    batch_size:      int   = 32
    accum_steps:     int   = 2        # batch effettivo = 64
    num_epochs:      int   = 20
    learning_rate:   float = 2e-5
    llrd_decay:      float = 0.9      # layer-wise LR decay
    weight_decay:    float = 0.01
    warmup_ratio:    float = 0.1
    label_smoothing: float = 0.1
    dropout:         float = 0.15
    intent_weight:   float = 1.0
    slot_weight:     float = 2.0
    grad_clip:       float = 1.0
    seed:            int   = 42
    patience:        int   = 6
    log_every_steps: int   = 50

# ─── LABEL MAPS ─────────────────────────────────────────────────────────────────

INTENTS = ["ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE", "UNKNOWN"]
NER_TAGS = ["O", "B-EXE", "I-EXE", "B-SET", "B-REP", "B-WGT", "B-UNT", "B-MOD", "I-MOD"]

def load_label_maps(path):
    with open(path) as f:
        lm = json.load(f)
    # JSON ha chiavi stringa → convertiamo in int
    id2intent = {int(k): v for k, v in lm["id2intent"].items()}
    id2tag    = {int(k): v for k, v in lm["id2tag"].items()}
    return lm["intent2id"], id2intent, lm["tag2id"], id2tag

# ─── DATASET ────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchcrf import CRF
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaModel,
    get_cosine_schedule_with_warmup,
)
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_report

class WorkoutNLUDataset(Dataset):
    def __init__(self, path, tokenizer, intent2id, tag2id, max_length=96):
        with open(path, encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer  = tokenizer
        self.intent2id  = intent2id
        self.tag2id     = tag2id
        self.max_length = max_length

        truncated = sum(
            1 for ex in self.examples
            if len(tokenizer(ex["words"], is_split_into_words=True)["input_ids"]) > max_length
        )
        if truncated:
            pct = 100 * truncated / len(self.examples)
            print(f"  WARNING {os.path.basename(path)}: {truncated}/{len(self.examples)} "
                  f"frasi troncate ({pct:.1f}%) — considera di aumentare max_length")

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

        # ner_labels:
        #   -100 → posizioni da ignorare per F1 (special tokens, subword non-primi)
        #    0   → "O" (usato dal CRF su quelle stesse posizioni)
        # Usiamo attention_mask come maschera CRF, quindi tutti i token
        # non-padding ricevono un'etichetta (O per speciali/subword, reale per first-subword).
        labels_ner    = []   # etichette reali, -100 dove ignorare
        crf_tags_list = []   # come sopra ma con 0 al posto di -100 (per CRF)
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                # token speciale [CLS]/[SEP]: etichetta O per CRF, -100 per F1
                labels_ner.append(-100)
                crf_tags_list.append(0)
            elif wid != prev_wid:
                # primo subword di una parola: etichetta reale
                tag = ner_tags[wid] if wid < len(ner_tags) else "O"
                t = self.tag2id.get(tag, 0)
                labels_ner.append(t)
                crf_tags_list.append(t)
            else:
                # subword successivo: ignora per F1, O per CRF
                labels_ner.append(-100)
                crf_tags_list.append(0)
            prev_wid = wid

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "intent_label":   torch.tensor(intent,        dtype=torch.long),
            "ner_labels":     torch.tensor(labels_ner,    dtype=torch.long),
            "crf_tags":       torch.tensor(crf_tags_list, dtype=torch.long),
        }

# ─── MODELLO ────────────────────────────────────────────────────────────────────

class WorkoutNLUModel(nn.Module):
    """
    XLM-RoBERTa + attention pooling per intent + CRF per slot NER.

    Attention pooling: il modello impara quanto peso dare a ogni token
    per classificare l'intent, invece di usare solo [CLS].

    CRF: impara le probabilità di transizione tra tag (es. O→B-EXE ok,
    O→I-EXE impossibile). Viterbi garantisce sequenze BIO sempre valide.
    """
    def __init__(self, model_name, num_intents, num_slot_labels, dropout=0.15):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size  # 768

        self.intent_attn = nn.Linear(hidden, 1)
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_intents),
        )
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_slot_labels),
        )
        self.crf = CRF(num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # [B, T, 768]

        # Attention pooling
        attn_scores  = self.intent_attn(seq).squeeze(-1)                  # [B, T]
        attn_scores  = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)                 # [B, T]
        pooled       = (attn_weights.unsqueeze(-1) * seq).sum(dim=1)      # [B, 768]

        intent_logits  = self.intent_head(pooled)  # [B, num_intents]
        slot_emissions = self.slot_head(seq)        # [B, T, num_tags]
        return intent_logits, slot_emissions

    def crf_loss(self, emissions, crf_tags, crf_mask):
        """
        crf_tags: [B, T] con valori in [0, num_tags-1] (nessun -100)
        crf_mask: [B, T] bool — deve avere True in [:, 0] per pytorch-crf
        """
        return -self.crf(emissions, crf_tags, mask=crf_mask, reduction='mean')

    def crf_decode(self, emissions, crf_mask):
        """Viterbi. Ritorna list[list[int]], len = numero di True per riga."""
        return self.crf.decode(emissions, mask=crf_mask)

# ─── OPTIMIZER CON LLRD ──────────────────────────────────────────────────────────

def build_optimizer(model, base_lr, weight_decay, llrd_decay):
    no_decay  = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    num_layers = model.roberta.config.num_hidden_layers  # 12
    groups     = []

    def _add(named_params, lr):
        wd = [p for n, p in named_params if not any(nd in n for nd in no_decay)]
        nd = [p for n, p in named_params if     any(nd in n for nd in no_decay)]
        if wd: groups.append({"params": wd, "lr": lr, "weight_decay": weight_decay})
        if nd: groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})

    _add(model.roberta.embeddings.named_parameters(),
         base_lr * (llrd_decay ** (num_layers + 1)))

    for i in range(num_layers):
        _add(model.roberta.encoder.layer[i].named_parameters(),
             base_lr * (llrd_decay ** (num_layers - i)))

    if hasattr(model.roberta, "pooler") and model.roberta.pooler is not None:
        _add(model.roberta.pooler.named_parameters(), base_lr)

    head_lr = base_lr * 5
    for component in [model.intent_attn, model.intent_head,
                      model.slot_head, model.crf]:
        _add(component.named_parameters(), head_lr)

    return AdamW(groups)

# ─── EVALUATE ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, id2intent, id2tag, verbose=False):
    """
    Valuta il modello. Se verbose=True stampa il classification report
    per entity type (es. quant'è preciso su EXE vs REP vs WGT).
    """
    model.eval()
    correct = total = 0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            ids      = batch["input_ids"].to(device)
            attn     = batch["attention_mask"].to(device)
            il       = batch["intent_label"].to(device)
            nl       = batch["ner_labels"].to(device)        # contiene -100
            crf_tags = batch["crf_tags"].to(device)          # nessun -100

            int_log, emissions = model(ids, attn)

            correct += (int_log.argmax(-1) == il).sum().item()
            total   += il.size(0)

            # CRF decode su tutta la sequenza non-padding (attention_mask)
            crf_mask = attn.bool()
            decoded  = model.crf_decode(emissions, crf_mask)  # list[list[int]]

            # Allinea predizioni con etichette: considera solo posizioni con nl != -100
            for pred_full, nl_row, attn_row in zip(
                decoded, nl.cpu().numpy(), attn.cpu().numpy()
            ):
                ps, ts = [], []
                pred_idx = 0
                for pos in range(len(attn_row)):
                    if attn_row[pos] == 0:
                        break                      # padding → fine sequenza
                    if nl_row[pos] != -100:        # posizione valida per F1
                        ps.append(id2tag[pred_full[pred_idx]])
                        ts.append(id2tag[nl_row[pos]])
                    pred_idx += 1
                all_pred.append(ps)
                all_true.append(ts)

    acc = correct / total if total else 0.0
    try:
        f1 = seq_f1(all_true, all_pred, zero_division=0)
        if verbose:
            print(seq_report(all_true, all_pred, zero_division=0))
    except Exception as e:
        print(f"[WARN] seq_f1 fallito: {e}")
        f1 = 0.0
    return {"intent_acc": acc, "slot_f1": f1, "combined": 0.5 * acc + 0.5 * f1}

# ─── EXPORT ONNX ────────────────────────────────────────────────────────────────

def export_onnx(model, tokenizer, device, output_dir):
    """
    Esporta backbone + teste lineari. Il CRF Viterbi non è ONNX-compatibile,
    ma dopo il training il modello produce raramente sequenze BIO invalide,
    quindi argmax è sufficiente in produzione.
    """
    print("\nExport ONNX...")

    class _NoCRF(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids, attention_mask)

    export_model = _NoCRF(model).cpu().eval()
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
            export_model,
            args=(dummy["input_ids"], dummy["attention_mask"]),
            f=onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["intent_logits", "slot_emissions"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "intent_logits":  {0: "batch"},
                "slot_emissions": {0: "batch", 1: "seq"},
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

def _decode_entities(words, slot_preds, word_ids, attention_mask_row, id2tag):
    """
    Allinea le predizioni CRF (una per token non-padding) alle parole originali
    e costruisce il dizionario di entità.
    """
    # Mappa word_index → tag (primo subword vince)
    seen = {}
    pred_idx = 0
    for pos, wid in enumerate(word_ids):
        if attention_mask_row[pos] == 0:
            break
        if wid is not None and wid not in seen:
            seen[wid] = id2tag.get(slot_preds[pred_idx], "O")
        pred_idx += 1

    entities = {}
    cur_type, cur_words = None, []
    for wid in sorted(seen):
        tag = seen[wid]
        if tag.startswith("B-"):
            if cur_type:
                entities.setdefault(cur_type, []).append(" ".join(cur_words))
            cur_type, cur_words = tag[2:], [words[wid]]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_words.append(words[wid])
        else:
            if cur_type:
                entities.setdefault(cur_type, []).append(" ".join(cur_words))
            cur_type  = tag[2:] if tag.startswith("B-") else None
            cur_words = [words[wid]] if cur_type else []
    if cur_type:
        entities.setdefault(cur_type, []).append(" ".join(cur_words))
    return entities


def inference_demo():
    print("\n" + "=" * 60)
    print("DEMO INFERENCE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(
        os.path.join(CFG.output_dir, "best_model.pt"),
        map_location=device, weights_only=False,
    )
    intent2id = ckpt["intent2id"]
    tag2id    = ckpt["tag2id"]
    id2intent = {v: k for k, v in intent2id.items()}
    id2tag    = {v: k for k, v in tag2id.items()}

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(CFG.model_name)
    model = WorkoutNLUModel(
        CFG.model_name, len(intent2id), len(tag2id), CFG.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sentences = [
        ("it", "aggiungi bench press 3 x 10 a cedimento"),
        ("it", "fatto deadlift 5 reps 140 kg"),
        ("it", "rimuovi lat machine"),
        ("en", "add bench press 3 sets of 8 reps 80 kg to failure"),
        ("en", "done bench press 8 reps 80 kg"),
        ("it", "quanto riposo tra le serie"),
        # Catene di esercizi
        ("it", "aggiungi squat 4 serie da 8 e push up 3 x 15 dropset"),
        ("it", "aggiungi bench press 3x10 e squat 4x8 e deadlift 5x5 a cedimento"),
        ("en", "add pull ups 4 x 8 and dips 3 x 12 then leg raises 3 x 20"),
        ("en", "add bench press 3 sets 10 reps and barbell row 4 sets 8 reps and pull ups 3 sets to failure"),
        # Multilingue
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
            int_log, emissions = model(enc["input_ids"], enc["attention_mask"])
            crf_mask   = enc["attention_mask"].bool()
            slot_preds = model.crf_decode(emissions, crf_mask)[0]  # list[int]

        probs  = torch.softmax(int_log, -1)[0]
        intent = id2intent[probs.argmax().item()]
        conf   = probs.max().item()

        entities = _decode_entities(
            words, slot_preds,
            enc.word_ids(0),
            enc["attention_mask"][0].cpu().tolist(),
            id2tag,
        )

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        use_fp16 = True
    else:
        device = torch.device("cpu")
        print("CPU mode (training molto lento senza GPU)")
        use_fp16 = False

    intent2id, id2intent, tag2id, id2tag = load_label_maps(
        os.path.join(CFG.data_dir, "label_maps.json")
    )
    num_intents     = len(intent2id)
    num_slot_labels = len(tag2id)
    print(f"Intents: {num_intents} | NER tags: {num_slot_labels}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(CFG.model_name)

    nw = 2 if torch.cuda.is_available() else 0
    train_ds = WorkoutNLUDataset(os.path.join(CFG.data_dir, "train.json"), tokenizer, intent2id, tag2id, CFG.max_length)
    val_ds   = WorkoutNLUDataset(os.path.join(CFG.data_dir, "val.json"),   tokenizer, intent2id, tag2id, CFG.max_length)
    test_ds  = WorkoutNLUDataset(os.path.join(CFG.data_dir, "test.json"),  tokenizer, intent2id, tag2id, CFG.max_length)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = WorkoutNLUModel(CFG.model_name, num_intents, num_slot_labels, CFG.dropout).to(device)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = build_optimizer(model, CFG.learning_rate, CFG.weight_decay, CFG.llrd_decay)

    effective_steps = (len(train_loader) // CFG.accum_steps) * CFG.num_epochs
    warmup_steps    = int(effective_steps * CFG.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, effective_steps)

    # Class weights per intent loss (bilanciamento classi slot)
    tag_counts = torch.zeros(num_slot_labels)
    for sample in train_ds:
        for t in sample["crf_tags"].tolist():
            tag_counts[t] += 1
    total_tok   = tag_counts.sum()
    slot_w      = (total_tok / (num_slot_labels * tag_counts.clamp(min=1))).to(device)
    print("Slot class weights:")
    for i in range(num_slot_labels):
        print(f"  {id2tag[i]:8s} → {slot_w[i].item():.2f}  (n={int(tag_counts[i])})")

    intent_criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"\n{'='*60}")
    print(f"Training {CFG.num_epochs} epochs | batch {CFG.batch_size} "
          f"(eff. {CFG.batch_size * CFG.accum_steps}) | fp16={use_fp16} | max_len={CFG.max_length}")
    print(f"{'='*60}\n")

    best_combined    = 0.0
    patience_counter = 0
    start_epoch      = 1

    # ── RESUME ────────────────────────────────────────────────────────────────
    resume_path_drive = os.path.join(CFG.drive_output, "resume_ckpt.pt")
    resume_path_local = os.path.join(CFG.output_dir,   "resume_ckpt.pt")

    for rp in [resume_path_drive, resume_path_local]:
        if os.path.exists(rp):
            print(f"Resume checkpoint trovato: {rp}")
            rc = torch.load(rp, map_location=device, weights_only=False)
            model.load_state_dict(rc["model_state"])
            optimizer.load_state_dict(rc["optimizer_state"])
            scheduler.load_state_dict(rc["scheduler_state"])
            scaler.load_state_dict(rc["scaler_state"])
            best_combined    = rc["best_combined"]
            patience_counter = rc["patience_counter"]
            start_epoch      = rc["epoch"] + 1
            print(f"  → riparto da epoch {start_epoch} | best={best_combined:.4f} | patience={patience_counter}/{CFG.patience}\n")
            if rp == resume_path_drive:
                shutil.copy2(rp, resume_path_local)
            break

    # ── LOOP ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG.num_epochs + 1):
        model.train()
        loss_i_tot = loss_s_tot = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            ids      = batch["input_ids"].to(device)
            attn     = batch["attention_mask"].to(device)
            il       = batch["intent_label"].to(device)
            crf_tags = batch["crf_tags"].to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                int_log, emissions = model(ids, attn)
                loss_i = intent_criterion(int_log, il)
                # CRF mask = attention_mask (tutti i token non-padding)
                # pytorch-crf richiede mask[:, 0] = True: attention_mask lo garantisce sempre
                crf_mask = attn.bool()
                loss_s   = model.crf_loss(emissions, crf_tags, crf_mask)
                loss     = (CFG.intent_weight * loss_i + CFG.slot_weight * loss_s) / CFG.accum_steps

            scaler.scale(loss).backward()

            if step % CFG.accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            loss_i_tot += loss_i.item()
            loss_s_tot += loss_s.item()

            if step % CFG.log_every_steps == 0:
                print(f"  Ep {epoch:2d} step {step:4d}/{len(train_loader)} "
                      f"| li={loss_i_tot/step:.4f} ls={loss_s_tot/step:.4f} "
                      f"| lr={scheduler.get_last_lr()[0]:.2e}")

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:2d} — li={loss_i_tot/len(train_loader):.4f} "
              f"ls={loss_s_tot/len(train_loader):.4f} — {elapsed:.1f}s")

        metrics = evaluate(model, val_loader, device, id2intent, id2tag)
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

        # ── SALVA RESUME ──────────────────────────────────────────────────────
        resume_state = {
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "best_combined":    best_combined,
            "patience_counter": patience_counter,
            "intent2id":        intent2id,
            "tag2id":           tag2id,
        }
        torch.save(resume_state, resume_path_local)
        if os.path.exists("/content/drive"):
            os.makedirs(CFG.drive_output, exist_ok=True)
            shutil.copy2(resume_path_local, resume_path_drive)
            best_src = os.path.join(CFG.output_dir, "best_model.pt")
            if os.path.exists(best_src):
                shutil.copy2(best_src, os.path.join(CFG.drive_output, "best_model.pt"))
            print(f"         Drive aggiornato (epoch {epoch})")

        print()

    # ── TEST FINALE ───────────────────────────────────────────────────────────
    print("=" * 60)
    ckpt = torch.load(
        os.path.join(CFG.output_dir, "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"])
    print("Test finale — per-entity report:")
    tm = evaluate(model, test_loader, device, id2intent, id2tag, verbose=True)
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
    os.makedirs("output/workout_nlu", exist_ok=True)
    _log_fh = _setup_log("output/workout_nlu/training.log")

    print("=" * 60)
    print("=== Coachly NLU — Colab Training ===")
    print(f"    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    try:
        install_deps()
        mount_drive()

        if not os.path.exists("data/train.json"):
            print("Dataset non trovato, genero...")
            import subprocess
            subprocess.run([sys.executable, "generate-dataset.py"], check=True)

        train()
        inference_demo()
        save_to_drive()

        print("\n[OK] Script terminato con successo.")

    except Exception:
        print("\n" + "=" * 60)
        print("[ERRORE FATALE] Traceback completo:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        print(f"Log salvato in: output/workout_nlu/training.log")
        sys.exit(1)

    finally:
        try: _log_fh.close()
        except Exception: pass
