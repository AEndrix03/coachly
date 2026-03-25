"""
Coachly NLU — Inferenza locale
===============================
Supporta sia best_model.pt (PyTorch) che workout_nlu_int8.onnx (ONNX).

── Con best_model.pt ──────────────────────────────────────────────
    pip install torch transformers pytorch-crf seqeval
    python infer.py --model best_model.pt

── Con ONNX (più leggero, no GPU) ─────────────────────────────────
    pip install onnxruntime transformers
    python infer.py --model workout_nlu_int8.onnx

File necessari:
    best_model.pt      ← da Drive → coachly_nlu/
    data/label_maps.json
"""

import sys, json, argparse
import numpy as np

MAX_LENGTH = 96
MODEL_NAME = "xlm-roberta-base"

# ─── LABEL MAPS ───────────────────────────────────────────────────────────────

def load_labels(path="data/label_maps.json"):
    with open(path) as f:
        lm = json.load(f)
    id2intent = {int(k): v for k, v in lm["id2intent"].items()}
    id2tag    = {int(k): v for k, v in lm["id2tag"].items()}
    return lm["intent2id"], id2intent, lm["tag2id"], id2tag

# ─── BACKEND PyTorch (.pt) ────────────────────────────────────────────────────

class TorchBackend:
    def __init__(self, pt_path):
        import torch
        import torch.nn as nn
        from torchcrf import CRF
        from transformers import XLMRobertaModel, XLMRobertaTokenizerFast

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

        intent2id = ckpt["intent2id"]
        tag2id    = ckpt["tag2id"]
        self.id2intent = {v: k for k, v in intent2id.items()}
        self.id2tag    = {int(k) if isinstance(k, str) else k: v for k, v in tag2id.items()}
        # id2tag dal checkpoint ha chiavi stringa se salvato da json, int altrimenti
        if all(isinstance(k, str) for k in self.id2tag):
            self.id2tag = {int(k): v for k, v in self.id2tag.items()}

        num_intents     = len(intent2id)
        num_slot_labels = len(tag2id)

        class WorkoutNLUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.roberta     = XLMRobertaModel.from_pretrained(MODEL_NAME)
                hidden           = self.roberta.config.hidden_size
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
                seq    = out.last_hidden_state
                scores = self.intent_attn(seq).squeeze(-1)
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
                pooled = (torch.softmax(scores, -1).unsqueeze(-1) * seq).sum(1)
                return self.intent_head(pooled), self.slot_head(seq)

        self.model = WorkoutNLUModel()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.torch     = torch
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)
        print(f"Modello PT caricato: {pt_path}")

    def predict(self, text):
        import torch
        words = text.strip().split()
        enc   = self.tokenizer(
            words, is_split_into_words=True,
            max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            int_log, emissions = self.model(enc["input_ids"], enc["attention_mask"])
            crf_mask   = enc["attention_mask"].bool()
            slot_preds = self.model.crf.decode(emissions, mask=crf_mask)[0]

        probs  = torch.softmax(int_log, -1)[0]
        intent = self.id2intent[probs.argmax().item()]
        conf   = probs.max().item()

        entities = _bio_to_entities(
            words, slot_preds,
            enc.word_ids(0),
            enc["attention_mask"][0].tolist(),
            self.id2tag,
        )
        return {"intent": intent, "confidence": conf, "entities": entities}

# ─── BACKEND ONNX (.onnx) ────────────────────────────────────────────────────

class OnnxBackend:
    def __init__(self, onnx_path, labels_path="data/label_maps.json"):
        import onnxruntime as ort
        from transformers import XLMRobertaTokenizerFast

        _, self.id2intent, _, self.id2tag = load_labels(labels_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self.session   = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)
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
            words, slot_preds,
            enc.word_ids(0),
            enc["attention_mask"][0].tolist(),
            self.id2tag,
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

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text",    nargs="?",            help="Frase da analizzare")
    parser.add_argument("--model", default="best_model.pt")
    parser.add_argument("--labels", default="data/label_maps.json")
    args = parser.parse_args()

    # Scegli backend in base all'estensione
    if args.model.endswith(".pt"):
        backend = TorchBackend(args.model)
    else:
        backend = OnnxBackend(args.model, args.labels)

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
