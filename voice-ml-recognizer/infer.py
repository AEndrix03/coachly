"""
Coachly NLU — Inferenza locale
===============================
Supporta sia best_model.pt (PyTorch) che best_model.onnx (ONNX).

── Setup ──────────────────────────────────────────────────────────
    pip install torch transformers pytorch-crf onnx onnxruntime

── Prima volta: esporta in ONNX (una tantum, ~1 min) ──────────────
    python infer.py --model best_model.pt --export

── Poi usa sempre l'ONNX (veloce) ─────────────────────────────────
    python infer.py --model best_model.onnx "fatto deadlift 5 reps 140 kg"
    python infer.py --model best_model.onnx          # modalità interattiva
"""

import os, sys, json, argparse, warnings
import numpy as np

# Silenzia il warning HuggingFace sull'HF_TOKEN (non serve per il tokenizer)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

MAX_LENGTH = 96
MODEL_NAME = "xlm-roberta-base"
TOKENIZER_DIR = "tokenizer"   # cartella locale dove salvare il tokenizer

# ─── TOKENIZER (scarica una volta, poi usa locale) ────────────────────────────

def get_tokenizer():
    from transformers import XLMRobertaTokenizerFast
    if os.path.isdir(TOKENIZER_DIR):
        tok = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER_DIR)
    else:
        print(f"Download tokenizer (una volta sola)...")
        tok = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)
        tok.save_pretrained(TOKENIZER_DIR)
        print(f"Tokenizer salvato in ./{TOKENIZER_DIR}/")
    return tok

# ─── LABEL MAPS ───────────────────────────────────────────────────────────────

def load_labels(path="data/label_maps.json"):
    with open(path) as f:
        lm = json.load(f)
    id2intent = {int(k): v for k, v in lm["id2intent"].items()}
    id2tag    = {int(k): v for k, v in lm["id2tag"].items()}
    return lm["intent2id"], id2intent, lm["tag2id"], id2tag

# ─── ARCHITETTURA (deve corrispondere a colab_train.py) ───────────────────────

def _build_model(num_intents, num_slot_labels):
    """
    Costruisce il modello con:
    - eager attention (compatibile ONNX, nessun download pesi HF)
    - -1e4 invece di -inf nel masking (ONNX non gestisce bene inf → NaN)
    """
    import torch
    import torch.nn as nn
    from torchcrf import CRF
    from transformers import XLMRobertaConfig, XLMRobertaModel

    class WorkoutNLUModel(nn.Module):
        def __init__(self):
            super().__init__()
            config = XLMRobertaConfig.from_pretrained(
                TOKENIZER_DIR if os.path.isdir(TOKENIZER_DIR) else MODEL_NAME
            )
            # eager: evita SDPA (Flash Attention) — necessario per export ONNX
            config._attn_implementation = "eager"
            # I pesi RoBERTa vengono dal checkpoint, non da HuggingFace
            self.roberta     = XLMRobertaModel(config)
            hidden           = config.hidden_size

            self.intent_attn = nn.Linear(hidden, 1)
            self.intent_head = nn.Sequential(
                nn.Dropout(0.15), nn.Linear(hidden, hidden // 2),
                nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden // 2, num_intents),
            )
            self.slot_head = nn.Sequential(
                nn.Dropout(0.15), nn.Linear(hidden, hidden // 2),
                nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden // 2, num_slot_labels),
            )
            self.crf = CRF(num_slot_labels, batch_first=True)

        def forward(self, input_ids, attention_mask):
            out    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            seq    = out.last_hidden_state                          # [B, T, H]
            scores = self.intent_attn(seq).squeeze(-1)             # [B, T]
            # -1e4 invece di -inf: numericamente equivalente ma ONNX-safe (no NaN)
            scores = scores.masked_fill(attention_mask == 0, -1e4)
            pooled = (torch.softmax(scores, -1).unsqueeze(-1) * seq).sum(1)
            return self.intent_head(pooled), self.slot_head(seq)

    return WorkoutNLUModel()

# ─── BACKEND PyTorch (.pt) ────────────────────────────────────────────────────

class TorchBackend:
    def __init__(self, pt_path):
        import torch

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

        intent2id      = ckpt["intent2id"]
        tag2id         = ckpt["tag2id"]
        self.id2intent = {v: k for k, v in intent2id.items()}
        self.id2tag    = {v: k for k, v in tag2id.items()}

        self.tokenizer = get_tokenizer()
        self.model     = _build_model(len(intent2id), len(tag2id))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.torch = torch
        print(f"Modello PT caricato: {pt_path}")

    def predict(self, text):
        import torch
        words = text.strip().split()
        enc   = self.tokenizer(
            words, is_split_into_words=True,
            max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.inference_mode():
            int_log, emissions = self.model(enc["input_ids"], enc["attention_mask"])
            slot_preds = self.model.crf.decode(emissions, enc["attention_mask"].bool())[0]

        probs  = torch.softmax(int_log, -1)[0]
        intent = self.id2intent[probs.argmax().item()]
        conf   = probs.max().item()
        entities = _bio_to_entities(
            words, slot_preds, enc.word_ids(0),
            enc["attention_mask"][0].tolist(), self.id2tag,
        )
        return {"intent": intent, "confidence": conf, "entities": entities}

# ─── BACKEND ONNX (.onnx) ────────────────────────────────────────────────────

class OnnxBackend:
    def __init__(self, onnx_path, labels_path="data/label_maps.json"):
        import onnxruntime as ort

        _, self.id2intent, _, self.id2tag = load_labels(labels_path)
        self.tokenizer = get_tokenizer()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"Modello ONNX caricato: {onnx_path}")

    def predict(self, text):
        words = text.strip().split()
        enc   = self.tokenizer(
            words, is_split_into_words=True,
            max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="np",
        )
        int_log, slot_em = self.session.run(
            ["intent_logits", "slot_emissions"],
            {"input_ids":      enc["input_ids"].astype(np.int64),
             "attention_mask": enc["attention_mask"].astype(np.int64)},
        )
        probs  = _softmax(int_log[0])
        intent = self.id2intent[int(np.argmax(probs))]
        conf   = float(np.max(probs))

        slot_preds = np.argmax(slot_em[0], axis=-1).tolist()
        entities   = _bio_to_entities(
            words, slot_preds, enc.word_ids(0),
            enc["attention_mask"][0].tolist(), self.id2tag,
        )
        return {"intent": intent, "confidence": conf, "entities": entities}

# ─── UTILS ────────────────────────────────────────────────────────────────────

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _bio_to_entities(words, slot_preds, word_ids, attn, id2tag):
    seen = {}
    pred_idx = 0
    for pos, wid in enumerate(word_ids):
        if attn[pos] == 0:
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

def _print_result(text, r):
    print(f"  Intent   : {r['intent']} ({r['confidence']:.1%})")
    if r["entities"]:
        for etype, vals in r["entities"].items():
            print(f"  {etype:8s} : {', '.join(vals)}")
    else:
        print("  Entities : —")

# ─── EXPORT PT → ONNX ────────────────────────────────────────────────────────

def _verify_onnx(onnx_path, tokenizer):
    """Verifica che l'ONNX produca output non-NaN su una frase di test."""
    import onnxruntime as ort
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = tokenizer(
        ["add bench press 3 sets 10 reps"],
        is_split_into_words=False, max_length=MAX_LENGTH,
        padding="max_length", truncation=True, return_tensors="np",
    )
    out = sess.run(None, {
        "input_ids":      dummy["input_ids"].astype(np.int64),
        "attention_mask": dummy["attention_mask"].astype(np.int64),
    })
    if np.isnan(out[0]).any():
        return False, out[0]
    return True, out[0]


def export_to_onnx(pt_path, onnx_path=None):
    import torch, torch.nn as nn

    if onnx_path is None:
        onnx_path = pt_path.replace(".pt", ".onnx")
    int8_path = onnx_path.replace(".onnx", "_int8.onnx")

    print(f"Carico {pt_path}...")
    backend = TorchBackend(pt_path)
    model   = backend.model.eval()

    class _NoCRF(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, ids, mask): return self.m(ids, mask)

    export_model = _NoCRF(model)
    dummy_pt = backend.tokenizer(
        ["add bench press 3 sets 10 reps"],
        is_split_into_words=False, max_length=MAX_LENGTH,
        padding="max_length", truncation=True, return_tensors="pt",
    )

    # ── Tenta export con la nuova API dynamo (PyTorch 2.x) ───────────────────
    # Gestisce meglio i transformer moderni rispetto al legacy torch.onnx.export
    exported = False
    if hasattr(torch.onnx, "dynamo_export"):
        try:
            print(f"Esporto ONNX (dynamo) → {onnx_path}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prog = torch.onnx.dynamo_export(
                    export_model,
                    dummy_pt["input_ids"],
                    dummy_pt["attention_mask"],
                )
            prog.save(onnx_path)
            ok, logits = _verify_onnx(onnx_path, backend.tokenizer)
            if ok:
                print(f"  dynamo OK — logits: {logits[0].round(3)}")
                exported = True
            else:
                print("  dynamo: output NaN, tento metodo legacy...")
                os.remove(onnx_path)
        except Exception as e:
            print(f"  dynamo fallito ({e}), tento metodo legacy...")

    # ── Fallback: legacy torch.onnx.export ───────────────────────────────────
    if not exported:
        print(f"Esporto ONNX (legacy opset 14) → {onnx_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                torch.onnx.export(
                    export_model,
                    args=(dummy_pt["input_ids"], dummy_pt["attention_mask"]),
                    f=onnx_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["intent_logits", "slot_emissions"],
                    dynamic_axes={
                        "input_ids":      {0: "batch"},
                        "attention_mask": {0: "batch"},
                        "intent_logits":  {0: "batch"},
                        "slot_emissions": {0: "batch"},
                    },
                    opset_version=14,
                )
        ok, logits = _verify_onnx(onnx_path, backend.tokenizer)
        if ok:
            print(f"  legacy OK — logits: {logits[0].round(3)}")
        else:
            print("  ERRORE: entrambi i metodi producono NaN.")
            print("  Il modello ONNX non funzionerà — usa --model best_model.pt")
            os.remove(onnx_path)
            return None

    print(f"  {onnx_path} ({os.path.getsize(onnx_path)/1e6:.0f} MB)")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
        ok, _ = _verify_onnx(int8_path, backend.tokenizer)
        if ok:
            print(f"  {int8_path} ({os.path.getsize(int8_path)/1e6:.0f} MB)  ← usa questo")
            return int8_path
        else:
            print("  int8 NaN — uso float32")
            os.remove(int8_path)
    except Exception as e:
        print(f"  Quantizzazione saltata: {e}")
    return onnx_path

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text",     nargs="?", help="Frase da analizzare")
    parser.add_argument("--model",  default="best_model.pt")
    parser.add_argument("--labels", default="data/label_maps.json")
    parser.add_argument("--export", action="store_true",
                        help="Converti .pt → ONNX quantizzato (esegui una volta sola)")
    args = parser.parse_args()

    if args.export:
        if not args.model.endswith(".pt"):
            print("--export richiede un file .pt"); sys.exit(1)
        out = export_to_onnx(args.model)
        print(f"\nFatto. Usa:\n  python infer.py --model {out}")
        return

    # Auto-usa ONNX se già esportato (evita di caricare PyTorch)
    model_path = args.model
    if model_path.endswith(".pt"):
        for candidate in [
            model_path.replace(".pt", "_int8.onnx"),
            model_path.replace(".pt", ".onnx"),
        ]:
            if os.path.exists(candidate):
                print(f"[auto] uso {candidate}")
                model_path = candidate
                break

    backend = (TorchBackend(model_path) if model_path.endswith(".pt")
               else OnnxBackend(model_path, args.labels))
    print()

    if args.text:
        _print_result(args.text, backend.predict(args.text))
    else:
        print("Modalità interattiva — Ctrl+C per uscire\n")
        while True:
            try:
                text = input(">>> ").strip()
                if text:
                    _print_result(text, backend.predict(text))
                    print()
            except KeyboardInterrupt:
                print("\nArrivederci!")
                break

if __name__ == "__main__":
    main()
