#!/usr/bin/env python3
"""
augment.py — Text-level augmentation for Coachly NLU dataset

Applies surface transformations to expand an existing JSONL file.
The label JSON is preserved exactly — only the user text changes.

Usage:
  python augment.py --input data_v2/train.jsonl --output data_v2/train_aug.jsonl --factor 1.5
  # factor=1.5 means the output has 1.5× the input rows (50% extra augmented)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

# ─── Transformation tables ────────────────────────────────────────────────────

_NUMS_IT = {"1":"uno","2":"due","3":"tre","4":"quattro","5":"cinque","6":"sei","7":"sette","8":"otto","9":"nove","10":"dieci","12":"dodici","15":"quindici","20":"venti"}
_NUMS_EN = {"1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine","10":"ten","12":"twelve","15":"fifteen","20":"twenty"}
_NUMS_IT_REV = {v: k for k, v in _NUMS_IT.items()}
_NUMS_EN_REV = {v: k for k, v in _NUMS_EN.items()}

_DISFLUENCIES_IT = ["", "uhm ", "allora ", "ok ", "dunque ", "ehm ", "quindi ", "tipo "]
_DISFLUENCIES_EN = ["", "uhm ", "okay ", "well ", "so ", "like ", "alright "]

# Notation swaps: "4x8" ↔ "4 sets of 8" / "4 serie da 8"
_NOTATION_EN = ["{n} sets of {r}", "{n} x {r}", "{n}x{r}", "{n} sets x {r} reps", "{n} sets {r} reps"]
_NOTATION_IT = ["{n} serie da {r}", "{n}x{r}", "{n} x {r}", "{n} serie per {r} rep", "{n}x{r} reps"]

# Weight format swaps
_WEIGHT_IT = ["da {w}kg", "{w}kg", "da {w} kg", "a {w} chili", "con {w}kg"]
_WEIGHT_EN = ["at {w}kg", "{w}kg", "with {w}kg", "at {w} kg", "{w} kg"]


# ─── Augmentation ops ─────────────────────────────────────────────────────────

def _add_disfluency(text: str, lang: str) -> str:
    """Prepend a random disfluency if not already present."""
    pool = _DISFLUENCIES_IT if lang == "it" else _DISFLUENCIES_EN
    prefix = random.choice(pool)
    if not prefix:
        return text
    for p in pool:
        if p and text.startswith(p):
            return text  # already has one
    return (prefix + text).strip()


def _swap_number_words(text: str, lang: str) -> str:
    """Randomly swap digits ↔ written words for small numbers."""
    if lang == "it":
        fwd, rev = _NUMS_IT, _NUMS_IT_REV
    else:
        fwd, rev = _NUMS_EN, _NUMS_EN_REV

    for digit, word in fwd.items():
        if random.random() < 0.25:
            text = re.sub(r"\b" + re.escape(digit) + r"\b", word, text)
    for word, digit in rev.items():
        if random.random() < 0.25:
            text = re.sub(r"\b" + re.escape(word) + r"\b", digit, text)
    return text


def _swap_set_notation(text: str, lang: str) -> str:
    """Swap numeric set×rep notation to textual form and vice-versa."""
    pool = _NOTATION_IT if lang == "it" else _NOTATION_EN

    def _repl_cross(m: re.Match) -> str:
        n, r = m.group(1), m.group(2)
        t = random.choice(pool)
        return t.format(n=n, r=r)

    def _repl_sets_of(m: re.Match) -> str:
        n, r = m.group(1), m.group(2)
        return f"{n}x{r}"

    # "NxM" or "N x M"
    text = re.sub(r"\b(\d+)\s*[xX×]\s*(\d+)\b", _repl_cross, text)
    # "N sets of M" → "NxM"
    text = re.sub(r"\b(\d+)\s+sets?\s+of\s+(\d+)\b", _repl_sets_of, text)
    text = re.sub(r"\b(\d+)\s+serie\s+da\s+(\d+)\b", _repl_sets_of, text)
    return text


def _swap_weight_format(text: str, lang: str) -> str:
    """Randomly swap weight expression style."""
    pool = _WEIGHT_IT if lang == "it" else _WEIGHT_EN

    def _repl(m: re.Match) -> str:
        w = m.group(1)
        t = random.choice(pool)
        return t.format(w=w)

    # Match patterns like "80kg", "80 kg", "a 80 chili", "con 80kg"
    text = re.sub(r"(?:da|a|con|at|with)?\s*(\d+(?:\.\d+)?)\s*(?:kg|chili|kilo)\b", _repl, text)
    return text


def _strip_disfluency(text: str, lang: str) -> str:
    """Remove leading disfluency if present."""
    pool = _DISFLUENCIES_IT if lang == "it" else _DISFLUENCIES_EN
    for p in pool:
        if p and text.startswith(p):
            return text[len(p):].strip()
    return text


_ALL_OPS = [
    "add_disfluency",
    "strip_disfluency",
    "swap_number_words",
    "swap_set_notation",
    "swap_weight_format",
]


def augment_text(text: str, lang: str) -> str:
    # Pick 1-3 random non-conflicting ops
    k = random.randint(1, 3)
    ops = random.sample(_ALL_OPS, k=min(k, len(_ALL_OPS)))

    # Avoid add+strip together
    if "add_disfluency" in ops and "strip_disfluency" in ops:
        ops.remove("strip_disfluency")

    for op in ops:
        if op == "add_disfluency":
            text = _add_disfluency(text, lang)
        elif op == "strip_disfluency":
            text = _strip_disfluency(text, lang)
        elif op == "swap_number_words":
            text = _swap_number_words(text, lang)
        elif op == "swap_set_notation":
            text = _swap_set_notation(text, lang)
        elif op == "swap_weight_format":
            text = _swap_weight_format(text, lang)

    return re.sub(r"\s+", " ", text).strip()


def augment_record(record: Dict) -> Dict:
    lang = record.get("lang", "en")
    new_text = augment_text(record["text"], lang)

    new = dict(record)
    new["id"] = record["id"] + "_aug"
    new["text"] = new_text
    new["messages"] = [
        record["messages"][0],                                   # system
        {"role": "user", "content": new_text},                   # user (new)
        record["messages"][2],                                   # assistant (same label)
    ]
    return new


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,       help="Input JSONL file")
    parser.add_argument("--output", required=True,       help="Output JSONL file")
    parser.add_argument("--factor", type=float, default=1.5,
                        help="Output/input ratio. 1.5 = original + 50%% extra (default 1.5)")
    parser.add_argument("--seed",   type=int,   default=123)
    parser.add_argument("--include_original", action="store_true", default=True,
                        help="Include original samples in output (default True)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load
    samples: List[Dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    n_aug = int(len(samples) * (args.factor - 1))
    to_augment = random.choices(samples, k=n_aug)
    augmented = [augment_record(s) for s in to_augment]

    # Deduplicate augmented by text (keep original if clash)
    existing_texts = {s["text"] for s in samples}
    deduped_aug: List[Dict] = []
    for a in augmented:
        if a["text"] not in existing_texts:
            existing_texts.add(a["text"])
            deduped_aug.append(a)

    if args.include_original:
        output = samples + deduped_aug
    else:
        output = deduped_aug

    random.shuffle(output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in output:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Input:          {len(samples):,} samples")
    print(f"Augmented added:{len(deduped_aug):,} samples")
    print(f"Output total:   {len(output):,} samples -> {args.output}")


if __name__ == "__main__":
    main()
