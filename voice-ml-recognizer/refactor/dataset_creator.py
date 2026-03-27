#!/usr/bin/env python3
"""
High-quality synthetic dataset creator for Coachly NLU function-calling.

Goal:
- Better intent balance (avoid collapse to ADD_EXERCISE)
- More realistic STT-like utterances (fillers, corrections, mixed it/en terms)
- Structured JSON target that is easy to parse on-device

Output files:
- <output_dir>/train.jsonl
- <output_dir>/val.jsonl
- <output_dir>/test.jsonl
- <output_dir>/metadata.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SYSTEM_PROMPT = (
    "You are Coachly NLU. Convert workout speech-to-text into strict JSON.\n"
    "Return ONLY valid JSON, no markdown.\n"
    "Schema:\n"
    "{\n"
    '  "action": "ADD_EXERCISE|LOG_SET|UPDATE_SET|DELETE_EXERCISE|UNKNOWN",\n'
    '  "items": [\n'
    "    {\n"
    '      "exercise": string,\n'
    '      "sets": integer|null,\n'
    '      "reps": integer|null,\n'
    '      "weight": number|null,\n'
    '      "unit": "kg"|"lbs"|null,\n'
    '      "modifier": "to_failure"|"dropset"|"superset"|"amrap"|"pause"|null\n'
    "    }\n"
    "  ]\n"
    "}"
)

ACTION_VALUES = [
    "ADD_EXERCISE",
    "LOG_SET",
    "UPDATE_SET",
    "DELETE_EXERCISE",
    "UNKNOWN",
]

MODIFIERS = [
    ("to_failure", {"it": ["a cedimento", "fino a cedimento"], "en": ["to failure"]}),
    ("dropset", {"it": ["dropset", "drop set"], "en": ["dropset", "drop set"]}),
    ("superset", {"it": ["superset"], "en": ["superset"]}),
    ("amrap", {"it": ["amrap"], "en": ["amrap"]}),
    ("pause", {"it": ["con pausa"], "en": ["paused"]}),
]

EXERCISES = [
    {
        "canonical": "bench press",
        "it": ["panca piana", "bench press"],
        "en": ["bench press", "flat bench"],
    },
    {
        "canonical": "incline bench press",
        "it": ["panca inclinata", "incline bench press"],
        "en": ["incline bench press", "incline bench"],
    },
    {
        "canonical": "squat",
        "it": ["squat", "back squat"],
        "en": ["squat", "barbell squat"],
    },
    {
        "canonical": "deadlift",
        "it": ["deadlift", "stacco da terra", "stacco"],
        "en": ["deadlift"],
    },
    {
        "canonical": "lat pulldown",
        "it": ["lat machine", "lat pulldown"],
        "en": ["lat pulldown"],
    },
    {
        "canonical": "pull up",
        "it": ["trazioni", "pull up"],
        "en": ["pull up", "pull ups", "chin up"],
    },
    {
        "canonical": "push up",
        "it": ["push up", "piegamenti", "flessioni"],
        "en": ["push up", "push ups"],
    },
    {
        "canonical": "overhead press",
        "it": ["military press", "overhead press", "lento avanti"],
        "en": ["overhead press", "military press"],
    },
    {
        "canonical": "leg press",
        "it": ["leg press", "pressa"],
        "en": ["leg press"],
    },
    {
        "canonical": "romanian deadlift",
        "it": ["stacco rumeno", "rdl", "romanian deadlift"],
        "en": ["romanian deadlift", "rdl"],
    },
    {
        "canonical": "barbell row",
        "it": ["rematore", "barbell row"],
        "en": ["barbell row", "bent over row"],
    },
    {
        "canonical": "dumbbell curl",
        "it": ["curl con manubri", "dumbbell curl"],
        "en": ["dumbbell curl", "dumbbell curls"],
    },
    {
        "canonical": "tricep pushdown",
        "it": ["pushdown", "tricep pushdown", "pushdown ai cavi"],
        "en": ["tricep pushdown", "cable pushdown"],
    },
]

UNKNOWN_IT = [
    "quante calorie ho bruciato oggi",
    "quanto devo recuperare tra le serie",
    "mi fai una scheda petto tricipiti",
    "che muscoli allena il deadlift",
    "oggi sono stanco non mi alleno",
    "metti un timer da 90 secondi",
    "quanto dura un deload",
    "dammi un consiglio sul riscaldamento",
    "come migliorare la tecnica nello squat",
    "fammi vedere il workout di ieri",
]

UNKNOWN_EN = [
    "how many calories did i burn",
    "create a chest and triceps workout plan",
    "what muscles does deadlift train",
    "set a timer for 90 seconds",
    "how long should i rest between sets",
    "show me my workout history",
    "i am too tired today",
    "give me mobility warmup tips",
    "what is deload week",
    "how to improve squat depth",
]

DISFLUENCY_PREFIX = {
    "it": ["uhm", "allora", "ok", "dunque", "ehm"],
    "en": ["uhm", "okay", "well", "so"],
}

CORRECTION_PREFIX = {
    "it": ["no aspetta", "anzi", "correggo"],
    "en": ["wait", "no actually", "correction"],
}


@dataclass
class ExerciseItem:
    exercise: str
    sets: int | None
    reps: int | None
    weight: float | None
    unit: str | None
    modifier: str | None

    def to_json(self) -> Dict:
        return {
            "exercise": self.exercise,
            "sets": self.sets,
            "reps": self.reps,
            "weight": self.weight,
            "unit": self.unit,
            "modifier": self.modifier,
        }


def _rng_int(a: int, b: int) -> int:
    return random.randint(a, b)


def pick_exercise(lang: str, allow_codeswitch: bool) -> Tuple[Dict, str]:
    e = random.choice(EXERCISES)
    if lang == "it" and allow_codeswitch and random.random() < 0.22:
        return e, random.choice(e["en"])
    return e, random.choice(e[lang])


def pick_modifier(lang: str, p: float = 0.28) -> Tuple[str | None, str]:
    if random.random() > p:
        return None, ""
    mod_key, forms = random.choice(MODIFIERS)
    return mod_key, random.choice(forms[lang])


def pick_weight() -> Tuple[float | None, str | None]:
    if random.random() < 0.42:
        value = random.choice([20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140])
        unit = random.choice(["kg", "lbs"]) if random.random() < 0.16 else "kg"
        return float(value), unit
    return None, None


def render_set_rep(lang: str, sets: int, reps: int) -> str:
    if lang == "it":
        styles = [f"{sets}x{reps}", f"{sets} x {reps}", f"{sets} serie da {reps}", f"{sets} serie da {reps} reps"]
    else:
        styles = [f"{sets}x{reps}", f"{sets} x {reps}", f"{sets} sets of {reps}", f"{sets} sets {reps} reps"]
    return random.choice(styles)


def render_weight(lang: str, value: float | None, unit: str | None) -> str:
    if value is None or unit is None:
        return ""
    i = int(value) if float(value).is_integer() else value
    if lang == "it":
        styles = [f"{i} {unit}", f"a {i} {unit}", f"con {i} {unit}"]
    else:
        styles = [f"{i} {unit}", f"at {i} {unit}", f"with {i} {unit}"]
    return random.choice(styles)


def maybe_noisy_prefix(lang: str) -> str:
    chunk: List[str] = []
    if random.random() < 0.25:
        chunk.append(random.choice(DISFLUENCY_PREFIX[lang]))
    if random.random() < 0.17:
        chunk.append(random.choice(CORRECTION_PREFIX[lang]))
    return " ".join(chunk).strip()


def normalize_stt_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,:;])\s*", " ", text)
    text = text.replace("  ", " ")
    return text.strip()


def create_item_payload(action: str, canonical: str, sets: int | None, reps: int | None, weight: float | None, unit: str | None, modifier: str | None) -> Dict:
    if action == "DELETE_EXERCISE":
        return ExerciseItem(canonical, None, None, None, None, None).to_json()
    if action == "UNKNOWN":
        return {}
    return ExerciseItem(canonical, sets, reps, weight, unit, modifier).to_json()


def render_add(lang: str) -> Tuple[str, Dict]:
    n_items = random.choice([1, 1, 1, 2, 2, 3])
    chunks = []
    payload_items = []

    for _ in range(n_items):
        ex, ex_surface = pick_exercise(lang, allow_codeswitch=True)
        sets = random.choice([2, 3, 4, 5])
        reps = random.choice([5, 6, 8, 10, 12, 15])
        weight, unit = pick_weight()
        mod_key, mod_surface = pick_modifier(lang, p=0.35)
        base = f"{ex_surface} {render_set_rep(lang, sets, reps)}"
        weight_part = render_weight(lang, weight, unit)
        if weight_part:
            base = f"{base} {weight_part}"
        if mod_surface:
            base = f"{base} {mod_surface}"
        chunks.append(base.strip())
        payload_items.append(create_item_payload("ADD_EXERCISE", ex["canonical"], sets, reps, weight, unit, mod_key))

    joiner = " e " if lang == "it" else " and "
    body = joiner.join(chunks)
    lead = random.choice(
        ["aggiungi", "metti", "inserisci", "programma"] if lang == "it" else ["add", "insert", "put", "schedule"]
    )
    text = f"{lead} {body}"
    return text, {"action": "ADD_EXERCISE", "items": payload_items}


def render_log(lang: str) -> Tuple[str, Dict]:
    ex, ex_surface = pick_exercise(lang, allow_codeswitch=True)
    reps = random.choice([3, 5, 6, 8, 10, 12, 15])
    sets = random.choice([1, 1, 1, 2])
    weight, unit = pick_weight()
    mod_key, mod_surface = pick_modifier(lang, p=0.18)

    if lang == "it":
        templates = [
            f"fatto {ex_surface} {reps} reps",
            f"ho fatto {ex_surface} {sets}x{reps}",
            f"completato {ex_surface} {reps} ripetizioni",
            f"segna {ex_surface} {sets} x {reps}",
        ]
    else:
        templates = [
            f"done {ex_surface} {reps} reps",
            f"i did {ex_surface} {sets}x{reps}",
            f"log {ex_surface} {sets} x {reps}",
            f"completed {ex_surface} {reps} reps",
        ]

    text = random.choice(templates)
    weight_part = render_weight(lang, weight, unit)
    if weight_part:
        text = f"{text} {weight_part}"
    if mod_surface:
        text = f"{text} {mod_surface}"

    payload = {
        "action": "LOG_SET",
        "items": [create_item_payload("LOG_SET", ex["canonical"], sets, reps, weight, unit, mod_key)],
    }
    return text, payload


def render_update(lang: str) -> Tuple[str, Dict]:
    ex, ex_surface = pick_exercise(lang, allow_codeswitch=True)
    sets = random.choice([2, 3, 4, 5])
    reps = random.choice([5, 6, 8, 10, 12])
    weight, unit = pick_weight()
    mod_key, mod_surface = pick_modifier(lang, p=0.16)

    if lang == "it":
        templates = [
            f"modifica {ex_surface} a {render_set_rep(lang, sets, reps)}",
            f"cambia {ex_surface}: {render_set_rep(lang, sets, reps)}",
            f"aggiorna {ex_surface} {render_set_rep(lang, sets, reps)}",
            f"porta {ex_surface} a {render_set_rep(lang, sets, reps)}",
        ]
    else:
        templates = [
            f"update {ex_surface} to {render_set_rep(lang, sets, reps)}",
            f"change {ex_surface} to {render_set_rep(lang, sets, reps)}",
            f"set {ex_surface} to {render_set_rep(lang, sets, reps)}",
            f"modify {ex_surface}: {render_set_rep(lang, sets, reps)}",
        ]

    text = random.choice(templates)
    weight_part = render_weight(lang, weight, unit)
    if weight_part:
        text = f"{text} {weight_part}"
    if mod_surface:
        text = f"{text} {mod_surface}"

    payload = {
        "action": "UPDATE_SET",
        "items": [create_item_payload("UPDATE_SET", ex["canonical"], sets, reps, weight, unit, mod_key)],
    }
    return text, payload


def render_delete(lang: str) -> Tuple[str, Dict]:
    n_items = random.choice([1, 1, 2])
    chunks = []
    payload_items = []
    for _ in range(n_items):
        ex, ex_surface = pick_exercise(lang, allow_codeswitch=True)
        chunks.append(ex_surface)
        payload_items.append(create_item_payload("DELETE_EXERCISE", ex["canonical"], None, None, None, None, None))

    joiner = " e " if lang == "it" else " and "
    if lang == "it":
        lead = random.choice(["rimuovi", "togli", "elimina"])
        text = f"{lead} {joiner.join(chunks)}"
    else:
        lead = random.choice(["remove", "delete", "drop"])
        text = f"{lead} {joiner.join(chunks)}"
    return text, {"action": "DELETE_EXERCISE", "items": payload_items}


def render_unknown(lang: str) -> Tuple[str, Dict]:
    text = random.choice(UNKNOWN_IT if lang == "it" else UNKNOWN_EN)
    return text, {"action": "UNKNOWN", "items": []}


def maybe_apply_prefix(text: str, lang: str) -> str:
    prefix = maybe_noisy_prefix(lang)
    if not prefix:
        return text
    sep = ", " if random.random() < 0.55 else " "
    return f"{prefix}{sep}{text}"


def example_to_record(idx: int, lang: str, text: str, payload: Dict) -> Dict:
    user_text = normalize_stt_text(text)
    assistant_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": f"{lang}_{idx:07d}",
        "lang": lang,
        "action": payload["action"],
        "text": user_text,
        "label": payload,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def validate_record(rec: Dict) -> None:
    action = rec["label"]["action"]
    items = rec["label"]["items"]
    if action not in ACTION_VALUES:
        raise ValueError(f"invalid action: {action}")
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    if action in {"ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE"} and len(items) == 0:
        raise ValueError(f"action {action} must include at least one item")
    for item in items:
        if "exercise" not in item:
            raise ValueError("missing exercise field")
        if item.get("unit") not in {"kg", "lbs", None}:
            raise ValueError("invalid unit")
        if item.get("modifier") not in {"to_failure", "dropset", "superset", "amrap", "pause", None}:
            raise ValueError("invalid modifier")

    parsed = json.loads(rec["messages"][-1]["content"])
    if parsed["action"] != action:
        raise ValueError("assistant payload mismatch")


def stratified_split(records: List[Dict], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    bucket = defaultdict(list)
    for r in records:
        bucket[(r["lang"], r["action"])].append(r)

    rng = random.Random(seed)
    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []

    for key in sorted(bucket.keys()):
        chunk = bucket[key]
        rng.shuffle(chunk)
        n = len(chunk)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(chunk[:n_train])
        val.extend(chunk[n_train : n_train + n_val])
        test.extend(chunk[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def sample_generator(action: str, lang: str) -> Tuple[str, Dict]:
    if action == "ADD_EXERCISE":
        return render_add(lang)
    if action == "LOG_SET":
        return render_log(lang)
    if action == "UPDATE_SET":
        return render_update(lang)
    if action == "DELETE_EXERCISE":
        return render_delete(lang)
    return render_unknown(lang)


def generate_dataset(per_action_per_lang: int, unknown_per_lang: int, seed: int) -> List[Dict]:
    random.seed(seed)
    langs = ["it", "en"]
    actions = ["ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE"]

    unique_texts = set()
    records: List[Dict] = []
    rid = 0

    for lang in langs:
        for action in actions:
            target = per_action_per_lang
            created = 0
            attempts = 0
            while created < target and attempts < target * 30:
                attempts += 1
                text, payload = sample_generator(action, lang)
                text = maybe_apply_prefix(text, lang)
                rec = example_to_record(rid, lang, text, payload)
                key = rec["text"]
                if key in unique_texts:
                    continue
                validate_record(rec)
                records.append(rec)
                unique_texts.add(key)
                rid += 1
                created += 1

        created_unknown = 0
        attempts_unknown = 0
        while created_unknown < unknown_per_lang and attempts_unknown < unknown_per_lang * 40:
            attempts_unknown += 1
            text, payload = render_unknown(lang)
            text = maybe_apply_prefix(text, lang)
            rec = example_to_record(rid, lang, text, payload)
            if rec["text"] in unique_texts:
                continue
            validate_record(rec)
            records.append(rec)
            unique_texts.add(rec["text"])
            rid += 1
            created_unknown += 1

    return records


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stats(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    by_action = Counter(r["action"] for r in rows)
    by_lang = Counter(r["lang"] for r in rows)
    by_pair = Counter((r["lang"], r["action"]) for r in rows)
    return {
        "action": dict(sorted(by_action.items())),
        "lang": dict(sorted(by_lang.items())),
        "lang_action": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_pair.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic dataset for FunctionGemma fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="refactor/data")
    parser.add_argument("--per_action_per_lang", type=int, default=340, help="Samples per lang for ADD/LOG/UPDATE/DELETE.")
    parser.add_argument("--unknown_per_lang", type=int, default=170, help="Samples per lang for UNKNOWN.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = generate_dataset(args.per_action_per_lang, args.unknown_per_lang, args.seed)
    train, val, test = stratified_split(all_rows, seed=args.seed, train_ratio=0.8, val_ratio=0.1)

    # Final split-level duplicate check
    all_texts = [r["text"] for r in train + val + test]
    if len(all_texts) != len(set(all_texts)):
        raise RuntimeError("duplicate texts found after split")

    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)
    _write_jsonl(output_dir / "test.jsonl", test)

    metadata = {
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "actions": ACTION_VALUES,
        "sizes": {"all": len(all_rows), "train": len(train), "val": len(val), "test": len(test)},
        "stats": {"all": _stats(all_rows), "train": _stats(train), "val": _stats(val), "test": _stats(test)},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Dataset created in: {output_dir}")
    print(json.dumps(metadata["sizes"], indent=2))
    print("Action distribution (all):", metadata["stats"]["all"]["action"])


if __name__ == "__main__":
    main()

