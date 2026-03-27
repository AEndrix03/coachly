#!/usr/bin/env python3
"""
dataset_creator_v2.py — Large-scale Coachly NLU dataset generator

Key principle: label["exercise"] == exact STT surface name used in text.
The model learns structural/action patterns, NOT exercise vocabulary.

Usage:
  python dataset_creator_v2.py --total 80000 --output_dir data_v2
  python dataset_creator_v2.py --total 80000 --output_dir data_v2 --md_dir exercises
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── System prompt (unchanged schema) ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Coachly NLU. Convert workout speech-to-text into strict JSON.\n"
    "Return ONLY valid JSON, no markdown.\nSchema:\n"
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

ACTION_VALUES = ["ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE", "UNKNOWN"]

# ─── Exercise loading & alias generation ──────────────────────────────────────

def _parse_md_exercises(md_dir: Path) -> List[Dict]:
    """Parse all exercise .md files, deduplicate by EN name."""
    exercises: List[Dict] = []
    seen: set = set()
    for md_file in sorted(md_dir.glob("*.md")):
        with open(md_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("|") or "---" in line:
                    continue
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) < 2:
                    continue
                en = parts[0].strip().strip("\ufeff")
                it = parts[1].strip() if len(parts) > 1 else ""
                disc = parts[2].strip() if len(parts) > 2 else ""
                if en.lower() in ("exercise name (en)", "exercise name", "en", "name"):
                    continue
                if not en or en == "-":
                    continue
                key = en.lower()
                if key not in seen:
                    seen.add(key)
                    exercises.append({"en": en, "it": it or en, "discipline": disc})
    return exercises


# Word-level substitution table for generating EN surface aliases
_EN_SUBS: List[Tuple[str, List[str]]] = [
    (r"\bbarbell\b",      ["barbell", "bb", "bar", ""]),
    (r"\bdumbbell\b",     ["dumbbell", "db", "dumbbells", ""]),
    (r"\bdbbell\b",       ["db", ""]),
    (r"\bcable\b",        ["cable", ""]),
    (r"\bromanian\b",     ["romanian", "rdl"]),
    (r"\boverhead\b",     ["overhead", "over head", "ohp", ""]),
    (r"\bpull[-\s]?down\b", ["pulldown", "pull down", "lat pull"]),
    (r"\bpull[-\s]?up\b",   ["pull up", "pullup", "chin up"]),
    (r"\bpush[-\s]?up\b",   ["push up", "pushup"]),
    (r"\bdeadlift\b",     ["deadlift", "dead lift", "dl"]),
    (r"\bincline\b",      ["incline", ""]),
    (r"\bdecline\b",      ["decline", ""]),
    (r"\bseated\b",       ["seated", ""]),
    (r"\bstanding\b",     ["standing", ""]),
    (r"\breverse\b",      ["reverse", ""]),
    (r"\bclose[-\s]grip\b", ["close grip", ""]),
    (r"\bwide[-\s]grip\b",  ["wide grip", ""]),
    (r"\bsingle[-\s]arm\b", ["single arm", ""]),
    (r"\bsingle[-\s]leg\b", ["single leg", ""]),
    (r"\bsumo\b",         ["sumo", ""]),
    (r"\bconventional\b", ["conventional", "conv", ""]),
]

# Word-level substitution table for generating IT surface aliases
_IT_SUBS: List[Tuple[str, List[str]]] = [
    (r"\bbilanciere\b",   ["bilanciere", "bb", ""]),
    (r"\bmanubri\b",      ["manubri", "db", ""]),
    (r"\bmanubrio\b",     ["manubrio", "db", ""]),
    (r"\bai cavi\b",      ["ai cavi", "cavo", ""]),
    (r"\balle spalle\b",  ["alle spalle", ""]),
    (r"\bal petto\b",     ["al petto", ""]),
    (r"\bseduto\b",       ["seduto", ""]),
    (r"\bin piedi\b",     ["in piedi", ""]),
]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _generate_aliases(en: str, it: str) -> Dict[str, List[str]]:
    """Return EN and IT surface aliases for one exercise."""
    en_l = en.lower()
    it_l = it.lower() if it and it.lower() != en_l else ""

    en_set: set = {en_l}
    it_set: set = {it_l} if it_l else set()
    it_set.add(en_l)  # English name used even in Italian speech

    # Apply EN substitutions
    for pattern, replacements in _EN_SUBS:
        if re.search(pattern, en_l):
            for repl in replacements:
                alias = _clean(re.sub(pattern, repl, en_l))
                if alias and len(alias) >= 2:
                    en_set.add(alias)

    # Apply IT substitutions
    if it_l:
        for pattern, replacements in _IT_SUBS:
            if re.search(pattern, it_l):
                for repl in replacements:
                    alias = _clean(re.sub(pattern, repl, it_l))
                    if alias and len(alias) >= 2:
                        it_set.add(alias)

    # Word-drop aliases (EN)
    words = en_l.split()
    if len(words) >= 2:
        en_set.add(_clean(" ".join(words[1:])))   # drop first word
        en_set.add(_clean(" ".join(words[:-1])))  # drop last word
        en_set.add(words[0])                       # first word only
        # Acronym (e.g. "rdl", "ohp")
        acronym = "".join(w[0] for w in words if len(w) > 2)
        if 2 <= len(acronym) <= 5:
            en_set.add(acronym)

    # Word-drop aliases (IT)
    if it_l:
        it_words = it_l.split()
        if len(it_words) >= 2:
            it_set.add(it_words[0])
            it_set.add(_clean(" ".join(it_words[:2])))

    en_list = sorted({a for a in en_set if 2 <= len(a) <= 50})
    it_list = sorted({a for a in it_set if 2 <= len(a) <= 50})
    return {
        "en": en_list or [en_l],
        "it": it_list or [it_l or en_l],
    }


def build_exercise_pool(md_dir: Path) -> List[Dict]:
    raw = _parse_md_exercises(md_dir)
    pool = []
    for ex in raw:
        aliases = _generate_aliases(ex["en"], ex["it"])
        if aliases["en"] and aliases["it"]:
            pool.append({
                "en": ex["en"],
                "it": ex["it"],
                "discipline": ex["discipline"],
                "aliases": aliases,
            })
    return pool


# ─── Vocabulary tables ──────────────────────────────────────────────────────────

MODIFIERS: Dict[str, Dict[str, List[str]]] = {
    "to_failure": {"it": ["a cedimento", "fino a cedimento", "cedimento", "al massimo"], "en": ["to failure", "failure", "till failure", "max reps"]},
    "dropset":    {"it": ["dropset", "drop set", "drop"],    "en": ["dropset", "drop set", "drop"]},
    "superset":   {"it": ["superset", "super set"],          "en": ["superset", "super set"]},
    "amrap":      {"it": ["amrap", "più che puoi"],          "en": ["amrap", "as many reps as possible", "as many as possible"]},
    "pause":      {"it": ["con pausa", "pausa"],             "en": ["paused", "with pause", "pause reps"]},
}

_NUMS_IT = {1:"uno",2:"due",3:"tre",4:"quattro",5:"cinque",6:"sei",7:"sette",8:"otto",9:"nove",10:"dieci",12:"dodici",15:"quindici",20:"venti",25:"venticinque",30:"trenta"}
_NUMS_EN = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",12:"twelve",15:"fifteen",20:"twenty",25:"twenty five",30:"thirty"}

def _nstr(n: int, lang: str) -> str:
    if random.random() < 0.12:
        d = _NUMS_IT if lang == "it" else _NUMS_EN
        return d.get(n, str(n))
    return str(n)

DISFLUENCIES: Dict[str, List[str]] = {
    "it": ["", "", "", "uhm ", "allora ", "ok ", "dunque ", "ehm ", "quindi ", "tipo ", "cioè "],
    "en": ["", "", "", "uhm ", "okay ", "well ", "so ", "like ", "alright ", "right "],
}

CORRECTIONS: Dict[str, List[str]] = {
    "it": ["", "", "", "no aspetta ", "anzi ", "correggo ", "aspetta ", "no no "],
    "en": ["", "", "", "wait ", "no actually ", "actually ", "correction ", "hold on "],
}

CONJUNCTIONS: Dict[str, List[str]] = {
    "it": [" e ", " più ", ", ", " ed ", " e anche ", " poi "],
    "en": [" and ", " plus ", ", ", " & ", " and also ", " then "],
}

ADD_VERBS: Dict[str, List[str]] = {
    "it": ["aggiungi", "metti", "inserisci", "aggiungimi", "voglio fare", "devo fare", "faccio", "segna", "aggiungi al programma", "voglio aggiungere"],
    "en": ["add", "put", "insert", "i want to add", "i need to do", "include", "track", "add in", "schedule"],
}

LOG_VERBS: Dict[str, List[str]] = {
    "it": ["ho fatto", "ho completato", "registra", "loggami", "segna", "metti", "finito", "appena fatto", "completato"],
    "en": ["i did", "i completed", "log", "track", "mark", "done", "just did", "finished", "record", "i just did"],
}

UPDATE_VERBS: Dict[str, List[str]] = {
    "it": ["aggiorna", "cambia", "modifica", "correggi", "porta", "sistema", "metti", "imposta"],
    "en": ["update", "change", "modify", "fix", "set", "correct", "adjust", "edit"],
}

DELETE_VERBS: Dict[str, List[str]] = {
    "it": ["togli", "rimuovi", "cancella", "elimina", "leva", "togliemi", "togli dal programma"],
    "en": ["remove", "delete", "take out", "get rid of", "cancel", "drop", "clear"],
}

SETS_TMPL: Dict[str, List[str]] = {
    "it": [
        "{s} serie da {r}", "{s} serie da {r} reps", "{s}x{r}", "{s} set da {r}",
        "{s} serie per {r} rep", "{s}x{r} reps", "{s} serie {r} ripetizioni",
        "{s} x {r}", "{s}x{r} ripetizioni",
    ],
    "en": [
        "{s} sets of {r}", "{s} sets of {r} reps", "{s}x{r}", "{s} x {r}",
        "{s} sets x {r} reps", "{s} set of {r}", "{s}x{r} reps",
        "{s} sets {r} reps",
    ],
}

SETS_ONLY_TMPL: Dict[str, List[str]] = {
    "it": ["{s} serie", "{s} set", "{s} volte"],
    "en": ["{s} sets", "{s} set"],
}

WEIGHT_TMPL: Dict[str, List[str]] = {
    "it": ["da {w}{u}", "a {w} chili", "con {w}{u}", "{w}{u}", "da {w} {u}", "da {w} kilo"],
    "en": ["at {w}{u}", "with {w}{u}", "{w}{u}", "at {w} {u}", "{w} {u}"],
}

UNKNOWN_PHRASES: Dict[str, List[str]] = {
    "it": [
        "quante calorie ho bruciato", "fammi una scheda per le gambe", "crea un programma di allenamento",
        "quanti kg ho sollevato questa settimana", "mostrami i miei progressi", "qual è il mio massimale",
        "quante sessioni ho fatto questo mese", "dimmi quanto ho migliorato", "crea un programma per la massa",
        "qual è il miglior esercizio per le spalle", "mi fai una scheda petto tricipiti", "come si fa lo squat",
        "ho finito il workout", "inizia la sessione", "termina la sessione", "manda la sessione al trainer",
        "mostrami la storia dei miei allenamenti", "aggiungi una nota alla sessione", "modifica il mio profilo",
        "fammi un workout di 20 minuti", "avvia il timer", "ferma il timer", "calcola il mio volume totale",
        "ho un dolore al ginocchio", "suggeriscimi un alternativo alla panca", "voglio perdere peso",
        "voglio mettere massa", "dimmi la mia 1 rep max", "sync con l'apple watch", "esporta i dati",
        "qual è la mia frequenza cardiaca", "quante proteine devo mangiare", "quando mi alleno domani",
        "mostrami le statistiche del mese", "registra il peso corporeo", "imposta un obiettivo di forza",
        "analizza i miei progressi", "aiutami con la programmazione", "ho saltato l'allenamento",
        "cambia la lingua dell'app", "quanto devo recuperare tra le serie", "che muscoli allena il deadlift",
        "metti un timer da 90 secondi", "quanto dura un deload", "come migliorare la tecnica",
        "fammi vedere il workout di ieri", "qual è la mia percentuale di grasso", "crea una scheda per dimagrire",
        "qual è il miglior integratore", "pianifica il prossimo allenamento", "mostrami il calendario",
        "crea una sfida mensile", "condividi il mio workout", "connettiti con il mio trainer",
    ],
    "en": [
        "how many calories did i burn", "create a leg workout plan", "build a training program",
        "how much weight have i lifted this week", "show me my progress", "what is my one rep max",
        "how many sessions have i done", "tell me how much i've improved", "create a muscle gain program",
        "what is the best exercise for shoulders", "make me a chest and triceps routine", "how do i squat",
        "i'm done with my workout", "start the session", "end the session", "send the session to my trainer",
        "show my workout history", "add a note to the session", "edit my profile",
        "give me a 20 minute workout", "start the timer", "stop the timer", "calculate my total volume",
        "i have knee pain", "suggest an alternative to bench press", "i want to lose weight",
        "i want to build muscle", "tell me my one rep max", "sync with apple watch", "export my data",
        "what is my heart rate", "how much protein should i eat", "when do i train tomorrow",
        "show me this month's stats", "log my body weight", "set a strength goal",
        "analyze my progress", "help me with programming", "i missed my workout",
        "change app language", "how long should i rest between sets", "what muscles does deadlift train",
        "set a timer for 90 seconds", "how long is a deload", "how to improve my technique",
        "show me yesterday's workout", "what is my body fat percentage", "create a fat loss plan",
        "what is the best supplement", "plan my next workout", "show me the calendar",
        "create a monthly challenge", "share my workout", "connect with my trainer",
    ],
}

# ─── Random helpers ────────────────────────────────────────────────────────────

def _rand_sets() -> int:
    return random.choices([1, 2, 3, 4, 5, 6], weights=[5, 15, 30, 25, 15, 10])[0]

def _rand_reps() -> int:
    return random.choice([3, 4, 5, 6, 8, 10, 12, 15, 20, 25])

def _rand_weight() -> Tuple[float, str]:
    if random.random() < 0.7:
        w = random.choice([10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 60.5, 80.5, 100.5])
        return w, "kg"
    w = random.choice([45, 65, 95, 115, 135, 155, 185, 205, 225])
    return w, "lbs"

def _rand_modifier() -> Optional[str]:
    return None if random.random() < 0.8 else random.choice(list(MODIFIERS))

def _disf(lang: str) -> str:
    return random.choice(DISFLUENCIES[lang])

def _corr(lang: str) -> str:
    return random.choice(CORRECTIONS[lang])

def _conj(lang: str) -> str:
    return random.choice(CONJUNCTIONS[lang])


# ─── Item phrase renderer ──────────────────────────────────────────────────────

def _render_item_phrase(
    name: str,
    sets_v: Optional[int],
    reps_v: Optional[int],
    weight_v: Optional[float],
    weight_u: Optional[str],
    modifier: Optional[str],
    lang: str,
) -> str:
    parts = [name]

    if sets_v is not None and reps_v is not None:
        t = random.choice(SETS_TMPL[lang])
        parts.append(t.format(s=_nstr(sets_v, lang), r=_nstr(reps_v, lang)))
    elif sets_v is not None:
        t = random.choice(SETS_ONLY_TMPL[lang])
        parts.append(t.format(s=_nstr(sets_v, lang)))
    elif reps_v is not None:
        parts.append(f"{_nstr(reps_v, lang)} reps")

    if weight_v is not None and weight_u is not None:
        w_s = str(int(weight_v)) if weight_v == int(weight_v) else str(weight_v)
        t = random.choice(WEIGHT_TMPL[lang])
        parts.append(t.format(w=w_s, u=weight_u))

    if modifier is not None:
        parts.append(random.choice(MODIFIERS[modifier][lang]))

    return " ".join(parts)


# ─── Action renderers ──────────────────────────────────────────────────────────

def render_add(pool: List[Dict], lang: str) -> Tuple[str, Dict]:
    n = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[15, 22, 20, 16, 10, 7, 5, 5])[0]
    selected = random.sample(pool, min(n, len(pool)))

    # Chain-only mode: no sets/reps/weight — pure exercise list
    chain_only = n >= 3 and random.random() < 0.25

    items: List[Dict] = []
    phrases: List[str] = []

    for ex in selected:
        name = random.choice(ex["aliases"][lang])
        if chain_only:
            s, r, w, u, m = None, None, None, None, None
        else:
            s = _rand_sets() if random.random() < 0.65 else None
            r = _rand_reps() if (s is not None and random.random() < 0.65) else (
                _rand_reps() if random.random() < 0.15 else None
            )
            w, u = _rand_weight() if random.random() < 0.35 else (None, None)
            m = _rand_modifier()

        phrases.append(_render_item_phrase(name, s, r, w, u, m, lang))
        items.append({"exercise": name, "sets": s, "reps": r, "weight": w, "unit": u, "modifier": m})

    conj = _conj(lang)
    # Chain: space-separated (no conjunction) for 4+ items with random chance
    if n >= 4 and chain_only and random.random() < 0.5:
        ex_str = " ".join(p.split()[0] for p in phrases)  # names only, space separated
        # Rebuild items with bare names
        items = [{"exercise": p.split()[0], "sets": None, "reps": None, "weight": None, "unit": None, "modifier": None} for p in phrases]
    else:
        ex_str = conj.join(phrases)

    verb = random.choice(ADD_VERBS[lang])
    # Occasionally bare (no verb) for very short commands
    if n == 1 and random.random() < 0.1:
        text = f"{_disf(lang)}{phrases[0]}"
    else:
        text = f"{_disf(lang)}{_corr(lang)}{verb} {ex_str}"

    return _clean(text), {"action": "ADD_EXERCISE", "items": items}


def render_log(pool: List[Dict], lang: str) -> Tuple[str, Dict]:
    n = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    selected = random.sample(pool, min(n, len(pool)))

    items: List[Dict] = []
    phrases: List[str] = []

    for ex in selected:
        name = random.choice(ex["aliases"][lang])
        s = _rand_sets() if random.random() < 0.50 else None
        r = _rand_reps()
        w, u = _rand_weight() if random.random() < 0.55 else (None, None)
        m = _rand_modifier()

        phrases.append(_render_item_phrase(name, s, r, w, u, m, lang))
        items.append({"exercise": name, "sets": s, "reps": r, "weight": w, "unit": u, "modifier": m})

    conj = _conj(lang)
    verb = random.choice(LOG_VERBS[lang])
    text = f"{_disf(lang)}{verb} {conj.join(phrases)}"
    return _clean(text), {"action": "LOG_SET", "items": items}


def render_update(pool: List[Dict], lang: str) -> Tuple[str, Dict]:
    ex = random.choice(pool)
    name = random.choice(ex["aliases"][lang])
    s = _rand_sets() if random.random() < 0.35 else None
    r = _rand_reps() if random.random() < 0.55 else None
    w, u = _rand_weight() if random.random() < 0.50 else (None, None)

    phrase = _render_item_phrase(name, s, r, w, u, None, lang)
    verb = random.choice(UPDATE_VERBS[lang])
    text = f"{_disf(lang)}{verb} {phrase}"
    return _clean(text), {
        "action": "UPDATE_SET",
        "items": [{"exercise": name, "sets": s, "reps": r, "weight": w, "unit": u, "modifier": None}],
    }


def render_delete(pool: List[Dict], lang: str) -> Tuple[str, Dict]:
    n = random.choices([1, 2, 3, 4], weights=[45, 30, 15, 10])[0]
    selected = random.sample(pool, min(n, len(pool)))

    items: List[Dict] = []
    names: List[str] = []
    for ex in selected:
        name = random.choice(ex["aliases"][lang])
        names.append(name)
        items.append({"exercise": name, "sets": None, "reps": None, "weight": None, "unit": None, "modifier": None})

    conj = _conj(lang)
    verb = random.choice(DELETE_VERBS[lang])
    text = f"{_disf(lang)}{verb} {conj.join(names)}"
    return _clean(text), {"action": "DELETE_EXERCISE", "items": items}


def render_unknown(lang: str) -> Tuple[str, Dict]:
    text = random.choice(UNKNOWN_PHRASES[lang])
    text = (_disf(lang) + text).strip()
    return _clean(text), {"action": "UNKNOWN", "items": []}


# ─── Sample builder ────────────────────────────────────────────────────────────

_RENDERERS = {
    "ADD_EXERCISE":    render_add,
    "LOG_SET":         render_log,
    "UPDATE_SET":      render_update,
    "DELETE_EXERCISE": render_delete,
}

# ADD is the most important use case
_ACTION_WEIGHTS = [0.40, 0.25, 0.15, 0.15, 0.05]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def _build_record(idx: int, lang: str, text: str, label: Dict) -> Dict:
    text = _normalize(text)
    return {
        "id": f"{lang}_{idx:07d}",
        "lang": lang,
        "action": label["action"],
        "text": text,
        "label": label,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def generate_dataset(pool: List[Dict], total: int, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    records: List[Dict] = []
    seen_texts: set = set()

    # Max attempts before giving up on uniqueness (prevents infinite loop)
    max_attempts = total * 5
    attempts = 0
    idx = 0

    while len(records) < total and attempts < max_attempts:
        attempts += 1
        action = random.choices(ACTION_VALUES, weights=_ACTION_WEIGHTS)[0]
        lang = random.choices(["it", "en"], weights=[0.55, 0.45])[0]

        if action == "UNKNOWN":
            text, label = render_unknown(lang)
        else:
            text, label = _RENDERERS[action](pool, lang)

        text = _normalize(text)
        if text in seen_texts:
            continue

        seen_texts.add(text)
        records.append(_build_record(idx, lang, text, label))
        idx += 1

        if idx % 10000 == 0:
            print(f"  {idx:,} / {total:,} generated...")

    if len(records) < total:
        print(f"  Warning: only {len(records):,} unique samples generated (target {total:,})")
    return records


# ─── Split & write ─────────────────────────────────────────────────────────────

def _stratified_split(
    records: List[Dict], seed: int, train_r: float = 0.90, val_r: float = 0.05
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split by (lang, action)."""
    bucket: Dict = defaultdict(list)
    for r in records:
        bucket[(r["lang"], r["action"])].append(r)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for key in sorted(bucket):
        chunk = bucket[key]
        rng.shuffle(chunk)
        n = len(chunk)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        train.extend(chunk[:n_train])
        val.extend(chunk[n_train: n_train + n_val])
        test.extend(chunk[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stats(rows: List[Dict]) -> Dict:
    return {
        "action": dict(Counter(r["action"] for r in rows)),
        "lang":   dict(Counter(r["lang"] for r in rows)),
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md_dir",     default="./exercises",  help="Directory with exercise .md files")
    parser.add_argument("--output_dir", default="./data_v2")
    parser.add_argument("--total",      type=int, default=80000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    out_dir = Path(args.output_dir)

    print(f"Loading exercises from {md_dir} ...")
    pool = build_exercise_pool(md_dir)
    print(f"  {len(pool)} unique exercises loaded")
    total_aliases = sum(len(e["aliases"]["en"]) + len(e["aliases"]["it"]) for e in pool)
    print(f"  {total_aliases} total surface aliases")

    print(f"\nGenerating {args.total:,} samples (seed={args.seed}) ...")
    records = generate_dataset(pool, args.total, args.seed)

    print(f"\nSplitting and writing to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = _stratified_split(records, args.seed)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl",   val)
    _write_jsonl(out_dir / "test.jsonl",  test)

    metadata = {
        "seed": args.seed,
        "exercise_pool_size": len(pool),
        "total_aliases": total_aliases,
        "system_prompt": SYSTEM_PROMPT,
        "sizes": {"total": len(records), "train": len(train), "val": len(val), "test": len(test)},
        "stats": {"all": _stats(records), "train": _stats(train), "val": _stats(val), "test": _stats(test)},
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n  train: {len(train):,}")
    print(f"  val:   {len(val):,}")
    print(f"  test:  {len(test):,}")
    print(f"\nAction dist: {metadata['stats']['all']['action']}")
    print(f"Lang dist:   {metadata['stats']['all']['lang']}")
    print("Done.")


if __name__ == "__main__":
    main()
