"""
Coachly NLU — Inferenza locale (CPU, no GPU)
=============================================
Requisiti minimi:
    pip install onnxruntime transformers

File necessari (scarica da Drive → coachly_nlu/):
    workout_nlu_int8.onnx   ← modello quantizzato (~280 MB)
    label_maps.json         ← dalla cartella data/

Uso:
    python infer.py
    python infer.py "aggiungi squat 4x8 e bench press 3x10"
"""

import sys, json, argparse
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────

ONNX_MODEL   = "workout_nlu_int8.onnx"   # o workout_nlu.onnx se non hai l'int8
LABEL_MAPS   = "data/label_maps.json"
MODEL_NAME   = "xlm-roberta-base"         # per il tokenizer (scaricato la prima volta)
MAX_LENGTH   = 96

# ─── CARICA MODELLO ───────────────────────────────────────────────────────────

def load_model(onnx_path=ONNX_MODEL):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    print(f"Modello caricato: {onnx_path}")
    return sess

def load_labels(path=LABEL_MAPS):
    with open(path) as f:
        lm = json.load(f)
    id2intent = {int(k): v for k, v in lm["id2intent"].items()}
    id2tag    = {int(k): v for k, v in lm["id2tag"].items()}
    return lm["intent2id"], id2intent, lm["tag2id"], id2tag

def load_tokenizer(model_name=MODEL_NAME):
    from transformers import XLMRobertaTokenizerFast
    tok = XLMRobertaTokenizerFast.from_pretrained(model_name)
    print(f"Tokenizer caricato: {model_name}")
    return tok

# ─── INFERENZA ────────────────────────────────────────────────────────────────

def predict(text: str, session, tokenizer, id2intent, id2tag, max_length=MAX_LENGTH):
    words = text.strip().split()
    enc = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    input_ids      = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)

    intent_logits, slot_emissions = session.run(
        ["intent_logits", "slot_emissions"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )

    # Intent
    probs  = _softmax(intent_logits[0])
    intent = id2intent[int(np.argmax(probs))]
    conf   = float(np.max(probs))

    # Slot: argmax per token, poi raggruppa per parola
    slot_preds = np.argmax(slot_emissions[0], axis=-1)  # [T]
    word_ids   = enc.word_ids(0)
    attn       = attention_mask[0]

    seen = {}
    for pos, wid in enumerate(word_ids):
        if attn[pos] == 0:
            break
        if wid is not None and wid not in seen:
            seen[wid] = id2tag.get(int(slot_preds[pos]), "O")

    entities = _bio_to_entities(words, seen)
    return {"intent": intent, "confidence": conf, "entities": entities}


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _bio_to_entities(words, seen):
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

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Frase da analizzare (opzionale)")
    parser.add_argument("--model", default=ONNX_MODEL)
    parser.add_argument("--labels", default=LABEL_MAPS)
    args = parser.parse_args()

    session   = load_model(args.model)
    _, id2intent, _, id2tag = load_labels(args.labels)
    tokenizer = load_tokenizer()
    print()

    if args.text:
        # Singola frase da riga di comando
        result = predict(args.text, session, tokenizer, id2intent, id2tag)
        _print_result(args.text, result)
    else:
        # Modalità interattiva
        print("Modalità interattiva — digita una frase (Ctrl+C per uscire)\n")
        while True:
            try:
                text = input(">>> ").strip()
                if not text:
                    continue
                result = predict(text, session, tokenizer, id2intent, id2tag)
                _print_result(text, result)
                print()
            except KeyboardInterrupt:
                print("\nArrivederci!")
                break


def _print_result(text, r):
    print(f'  Intent   : {r["intent"]} ({r["confidence"]:.1%})')
    if r["entities"]:
        for entity_type, values in r["entities"].items():
            print(f'  {entity_type:8s} : {", ".join(values)}')
    else:
        print(f'  Entities : —')


if __name__ == "__main__":
    main()
