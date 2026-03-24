"""
Coachly NLU Dataset Generator - v2
Genera dataset sintetico multilingue per intent classification + slot filling (NER)
Output: data/train.json, data/val.json, data/test.json, data/label_maps.json

Intents: ADD_EXERCISE, LOG_SET, UPDATE_SET, DELETE_EXERCISE, UNKNOWN
NER tags (BIO): O, B-EXE, I-EXE, B-SET, B-REP, B-WGT, B-UNT, B-MOD, I-MOD

Novità v2:
- Tag B-MOD/I-MOD per tecniche avanzate (cedimento, dropset, cluster, super serie, ...)
- Nomi esercizi inglesi in frasi italiane (mixed language, ~40%)
- Molti più esempi di splitting (2-4 esercizi per frase)
- Template più ricchi e naturali per tutte le lingue
- ~15.000 esempi totali
"""

import json
import random
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

random.seed(42)

# ─── LABEL MAPS ────────────────────────────────────────────────────────────────

INTENTS = ["ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE", "UNKNOWN"]
NER_TAGS = ["O", "B-EXE", "I-EXE", "B-SET", "B-REP", "B-WGT", "B-UNT", "B-MOD", "I-MOD"]

INTENT2ID = {k: i for i, k in enumerate(INTENTS)}
TAG2ID    = {k: i for i, k in enumerate(NER_TAGS)}

# ─── EXERCISE POOL ─────────────────────────────────────────────────────────────

EXERCISES = [
    # CHEST
    {"it": ["panca piana", "distensioni su panca"], "en": ["bench press", "flat bench"],
     "fr": ["développé couché", "bench press"], "de": ["bankdrücken", "flachbank"],
     "es": ["press banca", "press plano"]},
    {"it": ["panca inclinata"], "en": ["incline bench press", "incline bench"],
     "fr": ["développé incliné", "incliné"], "de": ["schrägbankdrücken"],
     "es": ["press inclinado", "banca inclinada"]},
    {"it": ["chest press", "pec deck", "croci ai cavi"], "en": ["chest press", "pec deck", "chest fly", "cable fly"],
     "fr": ["écarté poulie", "pec deck"], "de": ["brustpresse", "butterfly"],
     "es": ["aperturas pecho", "press pectoral"]},
    {"it": ["dip", "dips alle parallele"], "en": ["dips", "parallel bar dips", "chest dips"],
     "fr": ["dips", "tractions aux barres parallèles"], "de": ["dips", "parallelbarren dips"],
     "es": ["fondos", "dips en paralelas"]},
    {"it": ["croci con manubri", "fly con manubri"], "en": ["dumbbell fly", "dumbbell flyes"],
     "fr": ["écarté haltères", "fly haltères"], "de": ["kurzhantel fliegen"],
     "es": ["aperturas mancuernas", "fly mancuernas"]},
    # BACK
    {"it": ["trazioni", "pull up", "trazioni alla sbarra", "pullup"], "en": ["pull ups", "pullups", "chin ups"],
     "fr": ["tractions", "pull ups"], "de": ["klimmzüge", "pull ups"],
     "es": ["dominadas", "pull ups"]},
    {"it": ["lat machine", "lat pulldown", "tirata al petto"], "en": ["lat pulldown", "lat machine"],
     "fr": ["tirage vertical", "lat pulldown"], "de": ["latziehen", "lat maschine"],
     "es": ["jalón al pecho", "polea alta"]},
    {"it": ["rematore", "row con bilanciere", "bent over row"], "en": ["barbell row", "bent over row", "rows"],
     "fr": ["tirage barre", "rowing barre"], "de": ["rudern", "langhantelrudern"],
     "es": ["remo con barra", "remo inclinado"]},
    {"it": ["deadlift", "stacco da terra", "stacco"], "en": ["deadlift", "conventional deadlift"],
     "fr": ["soulevé de terre", "deadlift"], "de": ["kreuzheben", "deadlift"],
     "es": ["peso muerto", "deadlift"]},
    {"it": ["rematore ai cavi", "cable row"], "en": ["cable row", "seated cable row"],
     "fr": ["tirage poulie basse", "rowing poulie"], "de": ["kabelrudern", "sitzrudern"],
     "es": ["remo en polea", "remo cable"]},
    {"it": ["pullover"], "en": ["pullover", "dumbbell pullover"],
     "fr": ["pullover"], "de": ["pullover"],
     "es": ["pullover"]},
    # LEGS
    {"it": ["squat", "squat con bilanciere", "back squat"], "en": ["squat", "back squat", "barbell squat"],
     "fr": ["squat", "squat barre"], "de": ["kniebeugen", "squat"],
     "es": ["sentadilla", "squat"]},
    {"it": ["leg press", "pressa"], "en": ["leg press"],
     "fr": ["leg press", "presse à cuisses"], "de": ["beinpresse", "leg press"],
     "es": ["prensa de piernas", "leg press"]},
    {"it": ["affondi", "lunges", "affondi camminate"], "en": ["lunges", "walking lunges", "reverse lunges"],
     "fr": ["fentes", "fentes avant"], "de": ["ausfallschritte", "lunges"],
     "es": ["zancadas", "lunges"]},
    {"it": ["leg extension", "estensioni quadricipiti"], "en": ["leg extension", "quad extension"],
     "fr": ["leg extension", "extension quadriceps"], "de": ["beinstrecker", "leg extension"],
     "es": ["extensión de pierna", "leg extension"]},
    {"it": ["leg curl", "curl femorali"], "en": ["leg curl", "hamstring curl"],
     "fr": ["leg curl", "curl ischio-jambiers"], "de": ["leg curl", "beinbeuger"],
     "es": ["curl femoral", "leg curl"]},
    {"it": ["romanian deadlift", "rdl", "stacco rumeno"], "en": ["romanian deadlift", "rdl"],
     "fr": ["soulevé de terre roumain", "rdl"], "de": ["rumänisches kreuzheben", "rdl"],
     "es": ["peso muerto rumano", "rdl"]},
    {"it": ["calf raise", "alzate sui polpacci"], "en": ["calf raise", "standing calf raise"],
     "fr": ["élévation mollets", "calf raise"], "de": ["wadenheben", "calf raise"],
     "es": ["elevación de talones", "calf raise"]},
    {"it": ["front squat", "squat frontale"], "en": ["front squat"],
     "fr": ["squat avant", "front squat"], "de": ["frontkniebeugen", "front squat"],
     "es": ["sentadilla frontal", "front squat"]},
    {"it": ["hack squat"], "en": ["hack squat"],
     "fr": ["hack squat"], "de": ["hack kniebeugen"],
     "es": ["hack squat"]},
    {"it": ["hip thrust"], "en": ["hip thrust", "barbell hip thrust"],
     "fr": ["hip thrust", "poussée de hanche"], "de": ["hüftheben", "hip thrust"],
     "es": ["empuje de cadera", "hip thrust"]},
    # SHOULDERS
    {"it": ["military press", "lento avanti", "shoulder press", "overhead press"],
     "en": ["overhead press", "military press", "shoulder press", "ohp"],
     "fr": ["développé militaire", "press épaules"], "de": ["schulterdrücken", "militärpresse"],
     "es": ["press militar", "overhead press"]},
    {"it": ["alzate laterali", "lateral raise"], "en": ["lateral raises", "side raises", "lateral raise"],
     "fr": ["élévations latérales"], "de": ["seitheben"],
     "es": ["elevaciones laterales", "lateral raise"]},
    {"it": ["face pull"], "en": ["face pulls", "face pull"],
     "fr": ["tirage visage", "face pull"], "de": ["face pull"],
     "es": ["face pull"]},
    {"it": ["alzate frontali", "front raise"], "en": ["front raises", "front raise"],
     "fr": ["élévations frontales", "front raise"], "de": ["frontheben"],
     "es": ["elevaciones frontales", "front raise"]},
    {"it": ["arnold press"], "en": ["arnold press", "arnold dumbbell press"],
     "fr": ["arnold press"], "de": ["arnold press"],
     "es": ["press arnold"]},
    # ARMS
    {"it": ["curl con bilanciere", "bicep curl", "curl", "curl bilanciere"],
     "en": ["barbell curl", "bicep curl", "curls"],
     "fr": ["curl barre", "biceps curl"], "de": ["bizepscurl", "langhantelcurl"],
     "es": ["curl bíceps", "curl con barra"]},
    {"it": ["curl con manubri", "dumbbell curl"], "en": ["dumbbell curls", "db curl"],
     "fr": ["curl haltères"], "de": ["kurzhantelcurl"],
     "es": ["curl mancuernas"]},
    {"it": ["hammer curl"], "en": ["hammer curls", "hammer curl"],
     "fr": ["curl marteau", "hammer curl"], "de": ["hammer curl"],
     "es": ["curl martillo", "hammer curl"]},
    {"it": ["french press", "skull crusher", "estensioni tricipiti"],
     "en": ["skull crushers", "french press", "tricep extension"],
     "fr": ["barre au front", "skull crusher"], "de": ["trizepsdrücken", "skull crusher"],
     "es": ["press francés", "skull crusher"]},
    {"it": ["pushdown ai cavi", "tricep pushdown", "pushdown"],
     "en": ["tricep pushdown", "cable pushdown"],
     "fr": ["tirage corde triceps", "pushdown"], "de": ["trizepsdrücken kabel", "pushdown"],
     "es": ["jalón tríceps", "pushdown tríceps"]},
    {"it": ["piegamenti", "push up", "flessioni"], "en": ["push ups", "pushups"],
     "fr": ["pompes", "push-ups"], "de": ["liegestütze", "push ups"],
     "es": ["flexiones", "push ups"]},
    {"it": ["curl ai cavi", "cable curl"], "en": ["cable curl", "cable curls"],
     "fr": ["curl poulie", "cable curl"], "de": ["kabelcurl"],
     "es": ["curl en polea", "cable curl"]},
    {"it": ["tricep dip", "dip panca"], "en": ["tricep dips", "bench dips"],
     "fr": ["dips triceps", "dips banc"], "de": ["trizepsdips"],
     "es": ["dips tríceps", "fondos banco"]},
    # CORE
    {"it": ["crunch", "addominali", "sit up"], "en": ["crunches", "abs crunches", "sit ups"],
     "fr": ["crunch", "abdominaux"], "de": ["crunches", "bauchübungen"],
     "es": ["crunches", "abdominales"]},
    {"it": ["plank"], "en": ["plank", "planks"],
     "fr": ["planche", "plank"], "de": ["planke", "plank"],
     "es": ["plancha", "plank"]},
    {"it": ["leg raise", "sollevamenti gambe"], "en": ["leg raises", "hanging leg raises"],
     "fr": ["relevés de jambes", "leg raise"], "de": ["beinheben", "leg raise"],
     "es": ["elevación de piernas", "leg raise"]},
    {"it": ["russian twist"], "en": ["russian twists", "russian twist"],
     "fr": ["rotation russe", "russian twist"], "de": ["russian twist"],
     "es": ["giro ruso", "russian twist"]},
    {"it": ["cable crunch", "crunch ai cavi"], "en": ["cable crunch", "cable crunches"],
     "fr": ["crunch poulie", "cable crunch"], "de": ["kabelcrunch"],
     "es": ["crunch polea", "cable crunch"]},
    {"it": ["mountain climber", "scalatore"], "en": ["mountain climbers", "mountain climber"],
     "fr": ["montée de genou", "mountain climber"], "de": ["bergsteiger", "mountain climber"],
     "es": ["escalador", "mountain climber"]},
    {"it": ["hyperextension", "back extension"], "en": ["hyperextensions", "back extensions"],
     "fr": ["hyperextension", "extension dos"], "de": ["hyperextension"],
     "es": ["hiperextensión", "extensión de espalda"]},
    {"it": ["ab wheel", "ruota addominali"], "en": ["ab wheel", "ab roller"],
     "fr": ["roue abdominale", "ab wheel"], "de": ["bauchrad", "ab wheel"],
     "es": ["rueda abdominal", "ab wheel"]},
    {"it": ["burpees", "burpee"], "en": ["burpees", "burpee"],
     "fr": ["burpees"], "de": ["burpees"],
     "es": ["burpees"]},
]

# Nomi inglesi comuni nelle palestre italiane
EN_IN_IT_POOL = [
    "bench press", "deadlift", "squat", "pull up", "push up", "lat pulldown",
    "leg press", "leg extension", "leg curl", "calf raise", "shoulder press",
    "overhead press", "face pull", "hammer curl", "french press", "skull crusher",
    "lat machine", "dip", "dips", "plank", "crunch", "russian twist",
    "front squat", "hack squat", "rdl", "hip thrust", "burpees",
    "cable row", "cable curl", "tricep pushdown", "arnold press", "pullover",
]

# ─── NUMBER WORDS ───────────────────────────────────────────────────────────────

NUM_WORDS = {
    "it": {1:"uno",2:"due",3:"tre",4:"quattro",5:"cinque",6:"sei",7:"sette",8:"otto",
           9:"nove",10:"dieci",12:"dodici",15:"quindici",20:"venti"},
    "en": {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",
           9:"nine",10:"ten",12:"twelve",15:"fifteen",20:"twenty"},
    "fr": {1:"un",2:"deux",3:"trois",4:"quatre",5:"cinq",6:"six",7:"sept",8:"huit",
           9:"neuf",10:"dix",12:"douze",15:"quinze",20:"vingt"},
    "de": {1:"ein",2:"zwei",3:"drei",4:"vier",5:"fünf",6:"sechs",7:"sieben",8:"acht",
           9:"neun",10:"zehn",12:"zwölf",15:"fünfzehn",20:"zwanzig"},
    "es": {1:"uno",2:"dos",3:"tres",4:"cuatro",5:"cinco",6:"seis",7:"siete",8:"ocho",
           9:"nueve",10:"diez",12:"doce",15:"quince",20:"veinte"},
}

UNITS = {
    "it": ["kg","chili","chilogrammi","kilo"],
    "en": ["kg","kilos","kilograms","lbs","pounds"],
    "fr": ["kg","kilos","kilogrammes","livres"],
    "de": ["kg","kilo","kilogramm","pfund"],
    "es": ["kg","kilos","kilogramos","libras"],
}

# ─── TRAINING MODIFIERS (BIO tagged tuples) ─────────────────────────────────────

MODIFIERS_IT = [
    [("a", "B-MOD"), ("cedimento", "I-MOD")],
    [("cedimento", "B-MOD")],
    [("fino", "B-MOD"), ("al", "I-MOD"), ("cedimento", "I-MOD")],
    [("a", "B-MOD"), ("esaurimento", "I-MOD")],
    [("dropset", "B-MOD")],
    [("drop", "B-MOD"), ("set", "I-MOD")],
    [("cluster", "B-MOD")],
    [("cluster", "B-MOD"), ("set", "I-MOD")],
    [("super", "B-MOD"), ("serie", "I-MOD")],
    [("superset", "B-MOD")],
    [("giant", "B-MOD"), ("set", "I-MOD")],
    [("amrap", "B-MOD")],
    [("negativa", "B-MOD")],
    [("negativa", "B-MOD"), ("lenta", "I-MOD")],
    [("con", "O"), ("pausa", "B-MOD")],
    [("rest", "B-MOD"), ("pause", "I-MOD")],
    [("myo", "B-MOD"), ("rep", "I-MOD")],
    [("parziali", "B-MOD")],
    [("isometrica", "B-MOD")],
    [("21", "B-MOD")],
    [("occlusion", "B-MOD")],
    [("lenta", "B-MOD")],
]

MODIFIERS_EN = [
    [("to", "B-MOD"), ("failure", "I-MOD")],
    [("failure", "B-MOD")],
    [("to", "B-MOD"), ("exhaustion", "I-MOD")],
    [("dropset", "B-MOD")],
    [("drop", "B-MOD"), ("set", "I-MOD")],
    [("cluster", "B-MOD")],
    [("cluster", "B-MOD"), ("set", "I-MOD")],
    [("superset", "B-MOD")],
    [("super", "B-MOD"), ("set", "I-MOD")],
    [("giant", "B-MOD"), ("set", "I-MOD")],
    [("amrap", "B-MOD")],
    [("slow", "B-MOD"), ("negative", "I-MOD")],
    [("rest", "B-MOD"), ("pause", "I-MOD")],
    [("myo", "B-MOD"), ("rep", "I-MOD")],
    [("isometric", "B-MOD")],
    [("partial", "B-MOD"), ("reps", "I-MOD")],
    [("21s", "B-MOD")],
    [("tempo", "B-MOD")],
]

MODIFIERS_FR = [
    [("à", "B-MOD"), ("l'échec", "I-MOD")],
    [("dropset", "B-MOD")],
    [("cluster", "B-MOD")],
    [("superset", "B-MOD")],
    [("amrap", "B-MOD")],
    [("négatif", "B-MOD"), ("lent", "I-MOD")],
]

MODIFIERS_DE = [
    [("bis", "B-MOD"), ("zum", "I-MOD"), ("versagen", "I-MOD")],
    [("dropset", "B-MOD")],
    [("cluster", "B-MOD")],
    [("superset", "B-MOD")],
    [("amrap", "B-MOD")],
    [("langsame", "B-MOD"), ("negative", "I-MOD")],
]

MODIFIERS_ES = [
    [("al", "B-MOD"), ("fallo", "I-MOD")],
    [("hasta", "B-MOD"), ("el", "I-MOD"), ("fallo", "I-MOD")],
    [("dropset", "B-MOD")],
    [("cluster", "B-MOD")],
    [("superset", "B-MOD")],
    [("amrap", "B-MOD")],
    [("negativo", "B-MOD"), ("lento", "I-MOD")],
]

MODIFIERS = {
    "it": MODIFIERS_IT,
    "en": MODIFIERS_EN,
    "fr": MODIFIERS_FR,
    "de": MODIFIERS_DE,
    "es": MODIFIERS_ES,
}

# ─── DATACLASS ─────────────────────────────────────────────────────────────────

@dataclass
class Example:
    id: str
    lang: str
    text: str
    intent: str
    words: List[str]
    ner_tags: List[str]

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def pick_num(lang: str, value: int) -> str:
    if random.random() < 0.2 and value in NUM_WORDS[lang]:
        return NUM_WORDS[lang][value]
    return str(value)

def pick_unit(lang: str) -> str:
    return random.choice(UNITS[lang])

def pick_exercise(lang: str) -> Tuple[List[str], str]:
    pool = random.choice(EXERCISES)
    name = random.choice(pool[lang])
    return name.split(), name

def pick_exercise_mixed_it() -> Tuple[List[str], str]:
    """40% chance di usare nome inglese in frase italiana"""
    if random.random() < 0.4:
        name = random.choice(EN_IN_IT_POOL)
        return name.split(), name
    return pick_exercise("it")

def pick_modifier(lang: str) -> List[Tuple[str, str]]:
    return random.choice(MODIFIERS[lang])

def make_sets() -> int: return random.choice([2, 3, 4, 5, 6])
def make_reps() -> int: return random.choice([4, 5, 6, 8, 10, 12, 15, 20])
def make_weight() -> float:
    bases = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    w = random.choice(bases)
    if random.random() < 0.3:
        w += random.choice([2.5, 5])
    return w

def tag_words(wt: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    return [w for w, _ in wt], [t for _, t in wt]

def ex_tags(words: List[str]) -> List[Tuple[str, str]]:
    return [(w, "B-EXE" if i == 0 else "I-EXE") for i, w in enumerate(words)]

# ─── ADD_EXERCISE — ITALIANO ────────────────────────────────────────────────────

def gen_add_it(idx: int) -> Example:
    ex_words, _ = pick_exercise_mixed_it()
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("it")
    ss = pick_num("it", sets); rs = pick_num("it", reps)
    mod = pick_modifier("it") if random.random() < 0.3 else []
    ex = ex_tags(ex_words)
    wp = [(str(weight), "B-WGT"), (unit, "B-UNT")] if weight else []

    templates = [
        lambda: [("aggiungi","O")] + ex + [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP"),("ripetizioni","O")] + wp + mod,
        lambda: [("aggiungi","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("set","O"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("voglio","O"),("fare","O")] + ex + [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("metti","O")] + ex + [(rs,"B-REP"),("ripetizioni","O"),("per","O"),(ss,"B-SET"),("serie","O")] + wp + mod,
        lambda: [("inserisci","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(rs,"B-REP"),("rep","O"),(ss,"B-SET"),("serie","O")] + wp + mod,
        lambda: [("aggiungi","O")] + ex + wp + [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP")] + mod,
        lambda: [("aggiungi","O"),(ss,"B-SET"),("serie","O"),("di","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [(ss,"B-SET"),("serie","O"),("di","O")] + ex + [(rs,"B-REP"),("ripetizioni","O")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O")] + ex + [(rs,"B-REP")] + wp + mod,
        lambda: [("segna","O")] + ex + [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("devo","O"),("fare","O")] + ex + [(ss,"B-SET"),("serie","O"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("metti","O"),("nel","O"),("workout","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("aggiungi","O"),("al","O"),("piano","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("programma","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("serie","O"),(rs,"B-REP"),("rip","O")] + wp + mod,
        lambda: [("aggiungimi","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("certe","O"),("volte","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"it_add_{idx}", "it", " ".join(words), "ADD_EXERCISE", words, tags)


# ─── ADD_EXERCISE — INGLESE ─────────────────────────────────────────────────────

def gen_add_en(idx: int) -> Example:
    ex_words, _ = pick_exercise("en")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("en")
    ss = pick_num("en", sets); rs = pick_num("en", reps)
    mod = pick_modifier("en") if random.random() < 0.3 else []
    ex = ex_tags(ex_words)
    wp = [(str(weight), "B-WGT"), (unit, "B-UNT")] if weight else []

    templates = [
        lambda: [("add","O")] + ex + [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("add","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("do","O")] + ex + [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("insert","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("i","O"),("want","O"),("to","O"),("do","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: ex + wp + [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP"),("repetitions","O")] + mod,
        lambda: [("log","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [(ss,"B-SET"),("sets","O"),("of","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + ex + wp + mod,
        lambda: [("put","O"),("in","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("schedule","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("add","O")] + ex + wp + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + mod,
        lambda: [("let","O"),("me","O"),("do","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("sets","O"),("at","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"en_add_{idx}", "en", " ".join(words), "ADD_EXERCISE", words, tags)


# ─── ADD_EXERCISE — FRANCESE ────────────────────────────────────────────────────

def gen_add_fr(idx: int) -> Example:
    ex_words, _ = pick_exercise("fr")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("fr")
    ss = pick_num("fr", sets); rs = pick_num("fr", reps)
    mod = pick_modifier("fr") if random.random() < 0.2 else []
    ex = ex_tags(ex_words)
    wp = [(str(weight), "B-WGT"), (unit, "B-UNT")] if weight else []

    templates = [
        lambda: [("ajoute","O")] + ex + [(ss,"B-SET"),("séries","O"),("de","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("ajouter","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("séries","O"),(rs,"B-REP"),("répétitions","O")] + wp + mod,
        lambda: [("insère","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("je","O"),("veux","O"),("faire","O")] + ex + [(ss,"B-SET"),("séries","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("note","O")] + ex + [(ss,"B-SET"),("séries","O"),("de","O"),(rs,"B-REP"),("répétitions","O")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + ex + wp + mod,
        lambda: [("planifie","O")] + ex + [(ss,"B-SET"),("séries","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"fr_add_{idx}", "fr", " ".join(words), "ADD_EXERCISE", words, tags)


# ─── ADD_EXERCISE — TEDESCO ─────────────────────────────────────────────────────

def gen_add_de(idx: int) -> Example:
    ex_words, _ = pick_exercise("de")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("de")
    ss = pick_num("de", sets); rs = pick_num("de", reps)
    mod = pick_modifier("de") if random.random() < 0.2 else []
    ex = ex_tags(ex_words)
    wp = [(str(weight), "B-WGT"), (unit, "B-UNT")] if weight else []

    templates = [
        lambda: [("füge","O"),("hinzu","O")] + ex + [(ss,"B-SET"),("sätze","O"),(rs,"B-REP"),("wiederholungen","O")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("trainiere","O")] + ex + [(ss,"B-SET"),("sätze","O"),("à","O"),(rs,"B-REP"),("wdh","O")] + wp + mod,
        lambda: [("mach","O")] + ex + [(ss,"B-SET"),("sätze","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("sätze","O"),(rs,"B-REP"),("wiederholungen","O")] + wp + mod,
        lambda: [("hinzufügen","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + ex + wp + mod,
        lambda: [("plane","O")] + ex + [(ss,"B-SET"),("sätze","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"de_add_{idx}", "de", " ".join(words), "ADD_EXERCISE", words, tags)


# ─── ADD_EXERCISE — SPAGNOLO ────────────────────────────────────────────────────

def gen_add_es(idx: int) -> Example:
    ex_words, _ = pick_exercise("es")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("es")
    ss = pick_num("es", sets); rs = pick_num("es", reps)
    mod = pick_modifier("es") if random.random() < 0.2 else []
    ex = ex_tags(ex_words)
    wp = [(str(weight), "B-WGT"), (unit, "B-UNT")] if weight else []

    templates = [
        lambda: [("agrega","O")] + ex + [(ss,"B-SET"),("series","O"),("de","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("añade","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("series","O"),(rs,"B-REP"),("repeticiones","O")] + wp + mod,
        lambda: [("quiero","O"),("hacer","O")] + ex + [(ss,"B-SET"),("series","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("pon","O")] + ex + [(ss,"B-SET"),("series","O"),("de","O"),(rs,"B-REP"),("repeticiones","O")] + wp + mod,
        lambda: [("insertar","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + ex + wp + mod,
        lambda: [("planifica","O")] + ex + [(ss,"B-SET"),("series","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"es_add_{idx}", "es", " ".join(words), "ADD_EXERCISE", words, tags)


# ─── MULTI-EXERCISE (SPLITTING) ─────────────────────────────────────────────────

def _block_it(ex_words, sets, reps, weight=None, unit=None, mod=None):
    ss = pick_num("it", sets); rs = pick_num("it", reps)
    block = ex_tags(ex_words)
    fmt = random.choice(["SxR", "SerieR", "SetR"])
    if fmt == "SxR":
        block += [(ss,"B-SET"),("x","O"),(rs,"B-REP")]
    elif fmt == "SerieR":
        block += [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP")]
    else:
        block += [(ss,"B-SET"),("set","O"),("x","O"),(rs,"B-REP")]
    if weight and unit:
        block += [(str(weight),"B-WGT"),(unit,"B-UNT")]
    if mod:
        block += mod
    return block


def _block_en(ex_words, sets, reps, weight=None, unit=None, mod=None):
    ss = pick_num("en", sets); rs = pick_num("en", reps)
    block = ex_tags(ex_words)
    fmt = random.choice(["SxR", "SetsReps", "SetOf"])
    if fmt == "SxR":
        block += [(ss,"B-SET"),("x","O"),(rs,"B-REP")]
    elif fmt == "SetsReps":
        block += [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")]
    else:
        block += [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP")]
    if weight and unit:
        block += [(str(weight),"B-WGT"),(unit,"B-UNT")]
    if mod:
        block += mod
    return block


def gen_multi_it(idx: int) -> Example:
    n = random.randint(2, 4)
    items = []
    for _ in range(n):
        ew, _ = pick_exercise_mixed_it()
        s = make_sets(); r = make_reps()
        w = make_weight() if random.random() < 0.25 else None
        u = pick_unit("it") if w else None
        m = pick_modifier("it") if random.random() < 0.3 else None
        items.append((ew, s, r, w, u, m))

    seps = [[("e","O")],[("poi","O")],[(",","O")],[("e","O"),("poi","O")],[("più","O")]]

    p = random.random()
    if p < 0.3:   wt = [("aggiungi","O")]
    elif p < 0.5: wt = [("metti","O")]
    elif p < 0.65: wt = [("inserisci","O")]
    elif p < 0.8:  wt = [("aggiungi","O"),("al","O"),("workout","O")]
    else: wt = []

    for i, (ew, s, r, w, u, m) in enumerate(items):
        wt += _block_it(ew, s, r, w, u, m)
        if i < n - 1:
            wt += random.choice(seps)

    words, tags = tag_words(wt)
    return Example(f"it_multi_{idx}", "it", " ".join(words), "ADD_EXERCISE", words, tags)


def gen_multi_en(idx: int) -> Example:
    n = random.randint(2, 4)
    items = []
    for _ in range(n):
        ew, _ = pick_exercise("en")
        s = make_sets(); r = make_reps()
        w = make_weight() if random.random() < 0.25 else None
        u = pick_unit("en") if w else None
        m = pick_modifier("en") if random.random() < 0.3 else None
        items.append((ew, s, r, w, u, m))

    seps = [[("and","O")],[("then","O")],[(",","O")],[("plus","O")],
            [("followed","O"),("by","O")]]

    p = random.random()
    if p < 0.4:   wt = [("add","O")]
    elif p < 0.6: wt = [("log","O")]
    elif p < 0.75: wt = [("do","O")]
    else: wt = []

    for i, (ew, s, r, w, u, m) in enumerate(items):
        wt += _block_en(ew, s, r, w, u, m)
        if i < n - 1:
            wt += random.choice(seps)

    words, tags = tag_words(wt)
    return Example(f"en_multi_{idx}", "en", " ".join(words), "ADD_EXERCISE", words, tags)


def gen_multi_fr(idx: int) -> Example:
    n = random.randint(2, 3)
    wt = [("ajoute","O")]
    for i in range(n):
        ew, _ = pick_exercise("fr")
        ss = pick_num("fr", make_sets()); rs = pick_num("fr", make_reps())
        wt += ex_tags(ew) + [(ss,"B-SET"),("x","O"),(rs,"B-REP")]
        if i < n - 1:
            wt += [random.choice([("et","O"),("puis","O"),(",","O")])]
    words, tags = tag_words(wt)
    return Example(f"fr_multi_{idx}", "fr", " ".join(words), "ADD_EXERCISE", words, tags)


def gen_multi_de(idx: int) -> Example:
    n = random.randint(2, 3)
    wt = [("füge","O"),("hinzu","O")]
    for i in range(n):
        ew, _ = pick_exercise("de")
        ss = pick_num("de", make_sets()); rs = pick_num("de", make_reps())
        wt += ex_tags(ew) + [(ss,"B-SET"),("x","O"),(rs,"B-REP")]
        if i < n - 1:
            wt += [random.choice([("und","O"),("dann","O"),(",","O")])]
    words, tags = tag_words(wt)
    return Example(f"de_multi_{idx}", "de", " ".join(words), "ADD_EXERCISE", words, tags)


def gen_multi_es(idx: int) -> Example:
    n = random.randint(2, 3)
    wt = [("agrega","O")]
    for i in range(n):
        ew, _ = pick_exercise("es")
        ss = pick_num("es", make_sets()); rs = pick_num("es", make_reps())
        wt += ex_tags(ew) + [(ss,"B-SET"),("x","O"),(rs,"B-REP")]
        if i < n - 1:
            wt += [random.choice([("y","O"),("luego","O"),(",","O")])]
    words, tags = tag_words(wt)
    return Example(f"es_multi_{idx}", "es", " ".join(words), "ADD_EXERCISE", words, tags)


# ─── LOG_SET ────────────────────────────────────────────────────────────────────

def gen_log_it(idx: int) -> Example:
    ew, _ = pick_exercise_mixed_it()
    reps = make_reps()
    weight = make_weight() if random.random() < 0.5 else None
    unit = pick_unit("it")
    rs = pick_num("it", reps)
    mod = pick_modifier("it") if random.random() < 0.2 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []

    templates = [
        lambda: [("fatto","O")] + ex + [(rs,"B-REP"),("ripetizioni","O")] + wp + mod,
        lambda: [("completato","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("ho","O"),("fatto","O")] + ex + [(rs,"B-REP"),("rep","O")] + wp + mod,
        lambda: ex + [("fatto","O"),(rs,"B-REP"),("volte","O")] + wp + mod,
        lambda: [("fatto","O"),(rs,"B-REP")] + ex + wp + mod,
        lambda: [("ok","O"),("fatto","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("eseguito","O")] + ex + [(rs,"B-REP"),("ripetizioni","O")] + wp + mod,
        lambda: [("serie","O"),("completata","O")] + ex + [(rs,"B-REP")] + wp + mod,
        lambda: [("finito","O")] + ex + [(rs,"B-REP"),("rep","O")] + wp + mod,
        lambda: ex + [(rs,"B-REP"),("rips","O")] + wp + mod,
        lambda: [("registra","O")] + ex + [(rs,"B-REP"),("ripetizioni","O")] + wp + mod,
        lambda: [("nota","O")] + ex + [(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"it_log_{idx}", "it", " ".join(words), "LOG_SET", words, tags)


def gen_log_en(idx: int) -> Example:
    ew, _ = pick_exercise("en")
    reps = make_reps()
    weight = make_weight() if random.random() < 0.5 else None
    unit = pick_unit("en")
    rs = pick_num("en", reps)
    mod = pick_modifier("en") if random.random() < 0.2 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []

    templates = [
        lambda: [("done","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("completed","O")] + ex + [(rs,"B-REP"),("repetitions","O")] + wp + mod,
        lambda: [("just","O"),("did","O")] + ex + [(rs,"B-REP"),("times","O")] + wp + mod,
        lambda: ex + [("done","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("finished","O")] + ex + [(rs,"B-REP")] + wp + mod,
        lambda: [("logged","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("got","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("checked","O"),("off","O")] + ex + [(rs,"B-REP")] + wp + mod,
        lambda: [("banged","O"),("out","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: ex + [(rs,"B-REP")] + wp + [("complete","O")] + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"en_log_{idx}", "en", " ".join(words), "LOG_SET", words, tags)


def gen_log_implicit(lang: str, idx: int) -> Example:
    reps = make_reps()
    weight = make_weight() if random.random() < 0.4 else None
    unit = pick_unit(lang)
    rs = pick_num(lang, reps)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []

    choices = {
        "it": [
            [("fatto","O"),(rs,"B-REP"),("reps","O")] + wp,
            [("completato","O"),(rs,"B-REP"),("ripetizioni","O")] + wp,
            [("ok","O"),(rs,"B-REP")] + wp,
            [("fatte","O"),(rs,"B-REP"),("volte","O")] + wp,
            [("ho","O"),("fatto","O"),(rs,"B-REP")] + wp,
            [("serie","O"),("fatta","O"),(rs,"B-REP"),("rips","O")] + wp,
            [("ci","O"),("siamo","O"),(rs,"B-REP")] + wp,
        ],
        "en": [
            [("done","O"),(rs,"B-REP"),("reps","O")] + wp,
            [("finished","O"),(rs,"B-REP"),("repetitions","O")] + wp,
            [("got","O"),(rs,"B-REP")] + wp,
            [("set","O"),("complete","O"),(rs,"B-REP"),("reps","O")] + wp,
            [("that","O"),("was","O"),(rs,"B-REP")] + wp,
        ],
        "fr": [
            [("fait","O"),(rs,"B-REP"),("reps","O")] + wp,
            [("terminé","O"),(rs,"B-REP"),("répétitions","O")] + wp,
        ],
        "de": [
            [("gemacht","O"),(rs,"B-REP"),("wiederholungen","O")] + wp,
            [("fertig","O"),(rs,"B-REP"),("reps","O")] + wp,
        ],
        "es": [
            [("listo","O"),(rs,"B-REP"),("repeticiones","O")] + wp,
            [("hecho","O"),(rs,"B-REP"),("reps","O")] + wp,
        ],
    }
    wt = random.choice(choices[lang])
    words, tags = tag_words(wt)
    return Example(f"{lang}_log_impl_{idx}", lang, " ".join(words), "LOG_SET", words, tags)


# ─── UPDATE_SET ─────────────────────────────────────────────────────────────────

def gen_update(lang: str, idx: int) -> Example:
    ew, _ = pick_exercise_mixed_it() if lang == "it" else pick_exercise(lang)
    reps = make_reps(); sets = make_sets()
    weight = make_weight() if random.random() < 0.5 else None
    unit = pick_unit(lang)
    rs = pick_num(lang, reps); ss = pick_num(lang, sets)
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []

    pfx = {
        "it": [("modifica","O"),("correggi","O"),("cambia","O"),("aggiorna","O"),("sistema","O"),("rettifica","O")],
        "en": [("update","O"),("change","O"),("correct","O"),("edit","O"),("modify","O"),("fix","O")],
        "fr": [("modifie","O"),("change","O"),("corrige","O"),("actualise","O")],
        "de": [("ändere","O"),("aktualisiere","O"),("korrigiere","O"),("bearbeite","O")],
        "es": [("modifica","O"),("cambia","O"),("corrige","O"),("actualiza","O")],
    }
    p = random.choice(pfx[lang])
    patterns = [
        [p] + ex + [(rs,"B-REP"),("reps","O")] + wp,
        [p] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp,
        [p] + ex + wp,
        [p] + ex + [(ss,"B-SET"),("serie","O"),(rs,"B-REP")] + wp,
    ]
    wt = random.choice(patterns)
    words, tags = tag_words(wt)
    return Example(f"{lang}_update_{idx}", lang, " ".join(words), "UPDATE_SET", words, tags)


# ─── DELETE_EXERCISE ────────────────────────────────────────────────────────────

def gen_delete(lang: str, idx: int) -> Example:
    ew, _ = pick_exercise_mixed_it() if lang == "it" else pick_exercise(lang)
    ex = ex_tags(ew)
    pfx = {
        "it": [("rimuovi","O"),("elimina","O"),("togli","O"),("cancella","O"),("leva","O"),("salta","O"),("rimuovere","O"),("toglierei","O")],
        "en": [("remove","O"),("delete","O"),("drop","O"),("skip","O"),("cancel","O"),("take out","O")],
        "fr": [("supprime","O"),("enlève","O"),("retire","O"),("efface","O")],
        "de": [("entferne","O"),("lösche","O"),("streiche","O"),("überspringe","O")],
        "es": [("elimina","O"),("borra","O"),("quita","O"),("saca","O"),("salta","O")],
    }
    p = random.choice(pfx[lang])
    wt = [p] + ex
    words, tags = tag_words(wt)
    return Example(f"{lang}_delete_{idx}", lang, " ".join(words), "DELETE_EXERCISE", words, tags)


# ─── UNKNOWN ────────────────────────────────────────────────────────────────────

def gen_unknown(lang: str, idx: int) -> Example:
    samples = {
        "it": [
            "quanto manca alla fine","pausa","timer cinque minuti","come si fa questo esercizio",
            "musica più alta","quante calorie ho bruciato","buongiorno","che ora è",
            "quanto peso devo usare","aiuto","non capisco","esercizio sbagliato",
            "quanto riposo tra le serie","come si chiama questo esercizio","ottimo lavoro",
            "prossima sessione quando","qual è il mio record personale","sono stanco",
            "quanta acqua bevo","stretching","salva il workout","esporta i dati",
            "qual è il mio massimale","mostra le statistiche","resetta il timer",
            "basta per oggi","finito l'allenamento","ottima sessione","troppo pesante",
            "musica più bassa","abbassa il volume","quanto manca alla pausa",
        ],
        "en": [
            "how much time left","pause workout","set timer five minutes","how do I do this",
            "turn up the music","how many calories burned","hello","what time is it",
            "what weight should I use","help","I don't understand","wrong exercise",
            "how long should I rest","what's this exercise called","great job",
            "when's my next session","what's my personal record","I'm tired",
            "how much water should I drink","stretch","save workout","export data",
            "what's my one rep max","show statistics","reset timer",
            "that's enough for today","workout complete","too heavy","lower the volume",
        ],
        "fr": [
            "combien de temps","pause","minuterie cinq minutes","comment faire cet exercice",
            "augmente la musique","combien de calories","bonjour","quelle heure est-il",
            "quel poids utiliser","aide","je ne comprends pas","mauvais exercice",
            "enregistre l'entraînement","montre mes statistiques","trop lourd",
        ],
        "de": [
            "wie viel zeit","pause","timer fünf minuten","wie mache ich diese übung",
            "musik lauter","wie viele kalorien","guten morgen","wie spät ist es",
            "welches gewicht","hilfe","ich verstehe nicht","falsche übung",
            "training speichern","statistiken anzeigen","zu schwer",
        ],
        "es": [
            "cuánto tiempo queda","pausa","temporizador cinco minutos","cómo hago este ejercicio",
            "sube la música","cuántas calorías","hola","qué hora es",
            "qué peso usar","ayuda","no entiendo","ejercicio equivocado",
            "guarda el entrenamiento","mostrar estadísticas","demasiado pesado",
        ],
    }
    text = random.choice(samples[lang])
    words = text.split()
    return Example(f"{lang}_unk_{idx}", lang, text, "UNKNOWN", words, ["O"] * len(words))


# ─── MAIN GENERATION ───────────────────────────────────────────────────────────

def generate_all() -> List[Example]:
    examples = []
    langs = ["it", "en", "fr", "de", "es"]

    # ADD_EXERCISE — single
    for lang, gen in [("it", gen_add_it), ("en", gen_add_en), ("fr", gen_add_fr),
                      ("de", gen_add_de), ("es", gen_add_es)]:
        n = 700 if lang in ["it", "en"] else 450
        for i in range(n):
            examples.append(gen(i))

    # ADD_EXERCISE — multi (splitting)
    for i in range(400): examples.append(gen_multi_it(i))
    for i in range(400): examples.append(gen_multi_en(i))
    for i in range(180): examples.append(gen_multi_fr(i))
    for i in range(180): examples.append(gen_multi_de(i))
    for i in range(180): examples.append(gen_multi_es(i))

    # LOG_SET
    for i in range(300): examples.append(gen_log_it(i))
    for i in range(300): examples.append(gen_log_en(i))
    for lang in langs:
        for i in range(120): examples.append(gen_log_implicit(lang, i))

    # UPDATE_SET
    for lang in langs:
        n = 180 if lang in ["it", "en"] else 120
        for i in range(n): examples.append(gen_update(lang, i))

    # DELETE_EXERCISE
    for lang in langs:
        for i in range(120): examples.append(gen_delete(lang, i))

    # UNKNOWN
    for lang in langs:
        for i in range(100): examples.append(gen_unknown(lang, i))

    return examples


def split_dataset(examples: List[Example], train_ratio=0.8, val_ratio=0.1):
    random.shuffle(examples)
    n = len(examples)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    return examples[:t], examples[t:v], examples[v:]


def save_json(examples: List[Example], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in examples], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples):5d} examples -> {path}")


if __name__ == "__main__":
    print("Generating dataset v2...")
    all_ex = generate_all()
    print(f"Total: {len(all_ex)}")

    from collections import Counter
    print(f"Intents: {dict(Counter(e.intent for e in all_ex))}")
    print(f"Langs:   {dict(Counter(e.lang   for e in all_ex))}")

    train, val, test = split_dataset(all_ex)
    save_json(train, "data/train.json")
    save_json(val,   "data/val.json")
    save_json(test,  "data/test.json")

    label_maps = {
        "intent2id": INTENT2ID,
        "id2intent": {str(v): k for k, v in INTENT2ID.items()},
        "tag2id":    TAG2ID,
        "id2tag":    {str(v): k for k, v in TAG2ID.items()},
    }
    with open("data/label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)
    print("Label maps -> data/label_maps.json")
    print("Done!")
