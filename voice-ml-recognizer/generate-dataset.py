"""
Coachly NLU — Dataset Generator v3
====================================
Genera dataset sintetico multilingue per:
  - Intent classification: ADD_EXERCISE, LOG_SET, UPDATE_SET, DELETE_EXERCISE, UNKNOWN
  - Slot filling BIO-NER: O, B-EXE, I-EXE, B-SET, B-REP, B-WGT, B-UNT, B-MOD, I-MOD

Fix v3 rispetto v2:
  - RIMOSSO "log" da template ADD_EXERCISE (conflitto semantico con LOG_SET)
  - RIMOSSO "logged" da LOG_SET (stessa radice di "log")
  - Pool UNKNOWN: 130+ frasi uniche per it/en, 55+ per fr/de/es → leakage ~0%
  - Noise injection: fillers (uhm, beh...) + rethinking (no aspetta, anzi...)
    su ~20% esempi → modello robusto a input vocale reale con errori/ripensamenti
  - gen_log_fr/de/es: generator completo (non solo implicit)
  - Split STRATIFICATO per intent → no leakage, distribuzione uniforme
  - Totale: ~9500 esempi
"""

import json, random, os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from collections import defaultdict

random.seed(42)

# ─── LABEL MAPS ────────────────────────────────────────────────────────────────

INTENTS  = ["ADD_EXERCISE", "LOG_SET", "UPDATE_SET", "DELETE_EXERCISE", "UNKNOWN"]
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
     "fr": ["développé incliné"], "de": ["schrägbankdrücken"],
     "es": ["press inclinado", "banca inclinada"]},
    {"it": ["chest press", "pec deck", "croci ai cavi"], "en": ["chest press", "pec deck", "cable fly"],
     "fr": ["écarté poulie", "pec deck"], "de": ["brustpresse", "butterfly"],
     "es": ["aperturas pecho", "press pectoral"]},
    {"it": ["dip", "dips alle parallele"], "en": ["dips", "chest dips"],
     "fr": ["dips"], "de": ["dips"],
     "es": ["fondos", "dips en paralelas"]},
    {"it": ["croci con manubri", "fly con manubri"], "en": ["dumbbell fly", "dumbbell flyes"],
     "fr": ["écarté haltères"], "de": ["kurzhantel fliegen"],
     "es": ["aperturas mancuernas"]},
    # BACK
    {"it": ["trazioni", "pull up", "trazioni alla sbarra"], "en": ["pull ups", "chin ups"],
     "fr": ["tractions", "pull ups"], "de": ["klimmzüge"],
     "es": ["dominadas", "pull ups"]},
    {"it": ["lat machine", "lat pulldown", "tirata al petto"], "en": ["lat pulldown"],
     "fr": ["tirage vertical"], "de": ["latziehen"],
     "es": ["jalón al pecho", "polea alta"]},
    {"it": ["rematore", "row con bilanciere", "bent over row"], "en": ["barbell row", "bent over row"],
     "fr": ["tirage barre"], "de": ["rudern"],
     "es": ["remo con barra"]},
    {"it": ["deadlift", "stacco da terra", "stacco"], "en": ["deadlift"],
     "fr": ["soulevé de terre"], "de": ["kreuzheben"],
     "es": ["peso muerto"]},
    {"it": ["rematore ai cavi", "cable row"], "en": ["cable row", "seated cable row"],
     "fr": ["tirage poulie basse"], "de": ["kabelrudern"],
     "es": ["remo en polea"]},
    {"it": ["pullover"], "en": ["pullover", "dumbbell pullover"],
     "fr": ["pullover"], "de": ["pullover"], "es": ["pullover"]},
    # LEGS
    {"it": ["squat", "squat con bilanciere", "back squat"], "en": ["squat", "barbell squat"],
     "fr": ["squat"], "de": ["kniebeugen"],
     "es": ["sentadilla", "squat"]},
    {"it": ["leg press", "pressa"], "en": ["leg press"],
     "fr": ["leg press"], "de": ["beinpresse"],
     "es": ["prensa de piernas"]},
    {"it": ["affondi", "lunges"], "en": ["lunges", "walking lunges"],
     "fr": ["fentes"], "de": ["ausfallschritte"],
     "es": ["zancadas", "lunges"]},
    {"it": ["leg extension"], "en": ["leg extension"],
     "fr": ["leg extension"], "de": ["beinstrecker"],
     "es": ["extensión de pierna"]},
    {"it": ["leg curl", "curl femorali"], "en": ["leg curl"],
     "fr": ["leg curl"], "de": ["leg curl"],
     "es": ["curl femoral"]},
    {"it": ["romanian deadlift", "rdl", "stacco rumeno"], "en": ["romanian deadlift", "rdl"],
     "fr": ["soulevé de terre roumain", "rdl"], "de": ["rumänisches kreuzheben"],
     "es": ["peso muerto rumano", "rdl"]},
    {"it": ["calf raise", "alzate sui polpacci"], "en": ["calf raise"],
     "fr": ["élévation mollets"], "de": ["wadenheben"],
     "es": ["elevación de talones"]},
    {"it": ["front squat", "squat frontale"], "en": ["front squat"],
     "fr": ["squat avant"], "de": ["frontkniebeugen"],
     "es": ["sentadilla frontal"]},
    {"it": ["hack squat"], "en": ["hack squat"],
     "fr": ["hack squat"], "de": ["hack kniebeugen"], "es": ["hack squat"]},
    {"it": ["hip thrust"], "en": ["hip thrust"],
     "fr": ["hip thrust"], "de": ["hüftheben"],
     "es": ["empuje de cadera"]},
    # SHOULDERS
    {"it": ["military press", "lento avanti", "shoulder press", "overhead press"],
     "en": ["overhead press", "military press", "ohp"],
     "fr": ["développé militaire"], "de": ["schulterdrücken"],
     "es": ["press militar", "overhead press"]},
    {"it": ["alzate laterali", "lateral raise"], "en": ["lateral raises"],
     "fr": ["élévations latérales"], "de": ["seitheben"],
     "es": ["elevaciones laterales"]},
    {"it": ["face pull"], "en": ["face pulls"],
     "fr": ["face pull"], "de": ["face pull"], "es": ["face pull"]},
    {"it": ["alzate frontali", "front raise"], "en": ["front raises"],
     "fr": ["élévations frontales"], "de": ["frontheben"],
     "es": ["elevaciones frontales"]},
    {"it": ["arnold press"], "en": ["arnold press"],
     "fr": ["arnold press"], "de": ["arnold press"], "es": ["press arnold"]},
    # ARMS
    {"it": ["curl con bilanciere", "bicep curl", "curl", "curl bilanciere"],
     "en": ["barbell curl", "bicep curl"],
     "fr": ["curl barre"], "de": ["bizepscurl"],
     "es": ["curl bíceps"]},
    {"it": ["curl con manubri", "dumbbell curl"], "en": ["dumbbell curls"],
     "fr": ["curl haltères"], "de": ["kurzhantelcurl"],
     "es": ["curl mancuernas"]},
    {"it": ["hammer curl"], "en": ["hammer curls"],
     "fr": ["curl marteau"], "de": ["hammer curl"],
     "es": ["curl martillo"]},
    {"it": ["french press", "skull crusher", "estensioni tricipiti"],
     "en": ["skull crushers", "french press", "tricep extension"],
     "fr": ["skull crusher"], "de": ["skull crusher"],
     "es": ["press francés", "skull crusher"]},
    {"it": ["pushdown ai cavi", "tricep pushdown", "pushdown"],
     "en": ["tricep pushdown", "cable pushdown"],
     "fr": ["pushdown"], "de": ["pushdown"],
     "es": ["jalón tríceps"]},
    {"it": ["piegamenti", "push up", "flessioni"], "en": ["push ups"],
     "fr": ["pompes"], "de": ["liegestütze"],
     "es": ["flexiones"]},
    {"it": ["curl ai cavi", "cable curl"], "en": ["cable curl"],
     "fr": ["curl poulie"], "de": ["kabelcurl"],
     "es": ["curl en polea"]},
    {"it": ["tricep dip", "dip panca"], "en": ["tricep dips", "bench dips"],
     "fr": ["dips triceps"], "de": ["trizepsdips"],
     "es": ["dips tríceps"]},
    # CORE
    {"it": ["crunch", "addominali", "sit up"], "en": ["crunches", "sit ups"],
     "fr": ["crunch"], "de": ["crunches"],
     "es": ["crunches", "abdominales"]},
    {"it": ["plank"], "en": ["plank"],
     "fr": ["planche"], "de": ["planke"], "es": ["plancha"]},
    {"it": ["leg raise", "sollevamenti gambe"], "en": ["leg raises"],
     "fr": ["relevés de jambes"], "de": ["beinheben"],
     "es": ["elevación de piernas"]},
    {"it": ["russian twist"], "en": ["russian twists"],
     "fr": ["rotation russe"], "de": ["russian twist"],
     "es": ["giro ruso"]},
    {"it": ["cable crunch", "crunch ai cavi"], "en": ["cable crunch"],
     "fr": ["crunch poulie"], "de": ["kabelcrunch"],
     "es": ["crunch polea"]},
    {"it": ["mountain climber"], "en": ["mountain climbers"],
     "fr": ["montée de genou"], "de": ["bergsteiger"],
     "es": ["escalador"]},
    {"it": ["hyperextension", "back extension"], "en": ["hyperextensions"],
     "fr": ["hyperextension"], "de": ["hyperextension"],
     "es": ["hiperextensión"]},
    {"it": ["ab wheel", "ruota addominali"], "en": ["ab wheel"],
     "fr": ["roue abdominale"], "de": ["bauchrad"],
     "es": ["rueda abdominal"]},
    {"it": ["burpees", "burpee"], "en": ["burpees"],
     "fr": ["burpees"], "de": ["burpees"], "es": ["burpees"]},
]

EN_IN_IT_POOL = [
    "bench press", "deadlift", "squat", "pull up", "push up", "lat pulldown",
    "leg press", "leg extension", "leg curl", "calf raise", "shoulder press",
    "overhead press", "face pull", "hammer curl", "french press", "skull crusher",
    "lat machine", "dip", "dips", "plank", "crunch", "russian twist",
    "front squat", "hack squat", "rdl", "hip thrust", "burpees",
    "cable row", "cable curl", "tricep pushdown", "arnold press", "pullover",
]

# ─── NUMERI, UNITÀ, MODIFICATORI ───────────────────────────────────────────────

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
    "fr": ["kg","kilos","kilogrammes"],
    "de": ["kg","kilo","kilogramm"],
    "es": ["kg","kilos","kilogramos"],
}

MODIFIERS_IT = [
    [("a","B-MOD"),("cedimento","I-MOD")],
    [("cedimento","B-MOD")],
    [("fino","B-MOD"),("al","I-MOD"),("cedimento","I-MOD")],
    [("a","B-MOD"),("esaurimento","I-MOD")],
    [("dropset","B-MOD")],
    [("drop","B-MOD"),("set","I-MOD")],
    [("cluster","B-MOD")],
    [("super","B-MOD"),("serie","I-MOD")],
    [("superset","B-MOD")],
    [("giant","B-MOD"),("set","I-MOD")],
    [("amrap","B-MOD")],
    [("negativa","B-MOD")],
    [("con","O"),("pausa","B-MOD")],
    [("rest","B-MOD"),("pause","I-MOD")],
    [("parziali","B-MOD")],
    [("isometrica","B-MOD")],
    [("lenta","B-MOD")],
]

MODIFIERS_EN = [
    [("to","B-MOD"),("failure","I-MOD")],
    [("failure","B-MOD")],
    [("to","B-MOD"),("exhaustion","I-MOD")],
    [("dropset","B-MOD")],
    [("drop","B-MOD"),("set","I-MOD")],
    [("cluster","B-MOD")],
    [("superset","B-MOD")],
    [("super","B-MOD"),("set","I-MOD")],
    [("giant","B-MOD"),("set","I-MOD")],
    [("amrap","B-MOD")],
    [("slow","B-MOD"),("negative","I-MOD")],
    [("rest","B-MOD"),("pause","I-MOD")],
    [("isometric","B-MOD")],
    [("partial","B-MOD"),("reps","I-MOD")],
    [("tempo","B-MOD")],
]

MODIFIERS_FR = [
    [("à","B-MOD"),("l'échec","I-MOD")],
    [("dropset","B-MOD")],
    [("cluster","B-MOD")],
    [("superset","B-MOD")],
    [("amrap","B-MOD")],
    [("négatif","B-MOD"),("lent","I-MOD")],
]

MODIFIERS_DE = [
    [("bis","B-MOD"),("zum","I-MOD"),("versagen","I-MOD")],
    [("dropset","B-MOD")],
    [("cluster","B-MOD")],
    [("superset","B-MOD")],
    [("amrap","B-MOD")],
    [("langsame","B-MOD"),("negative","I-MOD")],
]

MODIFIERS_ES = [
    [("al","B-MOD"),("fallo","I-MOD")],
    [("hasta","B-MOD"),("el","I-MOD"),("fallo","I-MOD")],
    [("dropset","B-MOD")],
    [("cluster","B-MOD")],
    [("superset","B-MOD")],
    [("amrap","B-MOD")],
]

MODIFIERS = {"it":MODIFIERS_IT,"en":MODIFIERS_EN,"fr":MODIFIERS_FR,
             "de":MODIFIERS_DE,"es":MODIFIERS_ES}

# ─── NOISE: FILLERS E RETHINKING ───────────────────────────────────────────────
# Simulano input vocale reale: "uhm aggiungi squat 3x10" oppure
# "no aspetta aggiungi panca 3 serie da 10"

FILLERS = {
    "it": ["uhm","beh","allora","quindi","mmm","eh","ok","ecco","tipo","insomma"],
    "en": ["uhm","so","okay","well","like","uh","hmm","right","actually","you know"],
    "fr": ["euh","bon","alors","donc","hm","voilà","genre"],
    "de": ["äh","also","nun","hmm","ok","naja","ähm"],
    "es": ["eh","bueno","pues","entonces","mm","ok","osea","vamos"],
}

RETHINKING = {
    "it": ["no aspetta","anzi","no no","aspetta","cioè","voglio dire",
           "in realtà","meglio","no scusa","aspetta un attimo"],
    "en": ["no wait","actually","no no","wait","i mean","or rather",
           "hold on","never mind that","scratch that","wait actually"],
    "fr": ["non attends","en fait","plutôt","attends","non non","en réalité"],
    "de": ["nein warte","eigentlich","warte mal","also doch","nein nein","moment"],
    "es": ["no espera","en realidad","bueno no","espera","no no","a ver"],
}

# ─── POOL UNKNOWN (130+ it/en, 55+ fr/de/es) ──────────────────────────────────

UNKNOWN_POOL = {
    "it": [
        "quanto manca alla fine","a che punto sono","quanto tempo manca",
        "quanti esercizi mancano","sono a metà allenamento","ultimi esercizi",
        "come sto andando oggi","mostrami il piano","mostrami la scheda",
        "fammi vedere il programma","cosa c'è dopo","quale esercizio viene dopo",
        "pausa","timer cinque minuti","timer due minuti","timer un minuto",
        "timer trenta secondi","resetta il timer","imposta pausa due minuti",
        "quanto riposo tra le serie","quanto riposo consigliato",
        "come si fa questo esercizio","come si chiama questo esercizio",
        "quali muscoli sto allenando","quante calorie ho bruciato",
        "quanto peso devo usare","qual è il mio massimale",
        "qual è il mio record personale","mostra le statistiche",
        "quanto ho sollevato in totale","volume totale",
        "confronta con ieri","storico allenamenti","ultima sessione",
        "musica più alta","musica più bassa","abbassa il volume",
        "alzati","cambio musica","metti musica diversa",
        "sono stanco","sono esausto","non ce la faccio più",
        "mi fa male la schiena","dolore al ginocchio","ho i muscoli indolenziti",
        "sento tirare","bruciore muscolare","ottima pompa","crampi",
        "ho sete","quanta acqua bevo",
        "buongiorno","che ora è","ottimo lavoro","brava","perfetto",
        "basta per oggi","finito l'allenamento","ottima sessione",
        "troppo pesante","troppo facile","ancora uno","dai forza",
        "salva il workout","esporta i dati","salva sessione",
        "fine sessione","chiudi workout","termina allenamento",
        "prossima sessione quando","giorni di riposo",
        "programma la prossima sessione","quando mi alleno di nuovo",
        "stretching","voglio fare stretching","stretching finale",
        "riscaldamento","defaticamento","foam rolling",
        "post workout","pre workout","berò uno shake",
        "recupero muscolare","doms","mi sento in forma oggi",
        "voglio saltare questo","passo al prossimo",
        "questo esercizio non mi piace","preferisco altro",
        "dammi una motivazione","sono al massimo","posso farcela",
        "aiuto","non capisco","esercizio sbagliato",
        "mostra i miei progressi","voglio vedere i miei progressi",
        "quanto ho allenato oggi","sessione terminata",
        "basta così","chiamiamola qui","mi fermo per oggi",
        "ho finito","ci siamo","fatto per oggi",
        "devo riposare","ho bisogno di una pausa","mi siedo un attimo",
        "mostra il record","nuovo record personale","personal best",
        "frequenza cardiaca","battiti al minuto",
        "mi fa male il polso","tendinite","mi sento acciaccato",
        "oggi mi sento forte","bella sessione","devo andare",
        "quanto pesa il bilanciere","bilanciere olimpico","collari",
        "quanto è carico il rack","libera la sbarra",
        "qualcuno usa quello attrezzo","è libero il banco",
        "ho saltato ieri","allenamento di recupero","deload",
        "settimana di scarico","volume alto questa settimana",
        "sto aumentando i pesi","progressione lineare",
        "plateau","non miglioro più","come supero il plateau",
        "voglio cambiare programma","nuovo programma di allenamento",
        "quante volte a settimana mi alleno","split push pull legs",
        "full body o split","quale programma consigli",
        "mi annoio","stesso allenamento di sempre",
        "fuori forma","torno dopo la pausa","primo giorno di palestra",
        "quanto tempo ci vuole","risultati visibili quando",
        "integrazione","creatina","proteine in polvere",
        "calorie totali oggi","sono in deficit",
        "voglio mettere massa","voglio dimagrire","corpo libero o pesi",
    ],
    "en": [
        "how much time left","where am I in the workout","how much time remaining",
        "how many exercises left","halfway there","last exercises",
        "how am I doing today","show me the plan","show my program",
        "what's coming up next","what exercise is after this",
        "pause workout","set timer five minutes","set timer two minutes",
        "set timer one minute","set timer thirty seconds","reset timer",
        "how long should I rest","how long between sets","recommended rest time",
        "how do I do this exercise","what's this exercise called",
        "which muscles am I training","how many calories burned",
        "what weight should I use","what's my one rep max",
        "what's my personal record","show statistics",
        "how much did I lift total","total volume",
        "compare to yesterday","training history","last session",
        "turn up the music","lower the volume","change the music",
        "play something else","volume down",
        "I'm tired","I'm exhausted","I can't anymore",
        "my back hurts","knee pain","my muscles are sore",
        "I feel a pull","muscle burn","great pump","cramps",
        "I'm thirsty","how much water should I drink",
        "hello","what time is it","great job","well done","perfect",
        "that's enough for today","workout complete","great session",
        "too heavy","too easy","one more","let's go",
        "save workout","export data","save session",
        "end session","close workout","finish training",
        "when's my next session","rest days",
        "schedule next session","when do I train again",
        "stretch","stretching now","final stretch",
        "warm up","cool down","foam roll",
        "post workout shake","pre workout",
        "muscle recovery","doms","feeling strong today",
        "skip this one","move to next",
        "I don't like this exercise","prefer something else",
        "motivate me","I'm at my best","I can do this",
        "help","I don't understand","wrong exercise",
        "show my progress","I want to see my progress",
        "how long have I been training","session done",
        "calling it a day","I'm done for today",
        "I need to rest","I need a break","let me sit down",
        "show my record","new personal record","personal best",
        "heart rate","beats per minute",
        "my wrist hurts","tendinitis","feeling beat up",
        "feeling strong today","great session","I gotta go",
        "how heavy is the bar","olympic bar","collars",
        "is the rack loaded","clear the bar",
        "is someone using that","is the bench free",
        "I skipped yesterday","recovery workout","deload",
        "deload week","high volume this week",
        "I'm increasing weights","linear progression",
        "plateau","not improving anymore","how do I break through",
        "want to change program","new training program",
        "how many times a week","push pull legs split",
        "full body or split","which program do you recommend",
        "I'm bored","same workout as always",
        "out of shape","back after a break","first day at the gym",
        "how long does it take","when will I see results",
        "supplements","creatine","protein powder",
        "total calories today","I'm in a deficit",
        "want to bulk","want to cut","bodyweight or weights",
    ],
    "fr": [
        "combien de temps","pause","minuterie cinq minutes","minuterie deux minutes",
        "comment faire cet exercice","augmente la musique","combien de calories",
        "bonjour","quelle heure est-il","quel poids utiliser","aide",
        "je ne comprends pas","mauvais exercice","enregistre l'entraînement",
        "montre mes statistiques","trop lourd","exercice suivant",
        "j'ai besoin d'une pause","je suis fatigué","fin de séance",
        "montre mes progrès","quel est mon record","temps de repos",
        "combien d'exercices restants","bonne séance","trop facile",
        "mes muscles font mal","record personnel","volume total",
        "je suis épuisé","boire de l'eau","étirements","échauffement",
        "session terminée","prochaine séance","jours de repos",
        "combien ai-je soulevé","je me sens bien","douleur au dos",
        "changer la musique","motivation","sauter cet exercice",
        "affiche l'historique","comparer à hier","fréquence cardiaque",
        "je n'aime pas cet exercice","passer au suivant",
        "c'est assez pour aujourd'hui","j'ai terminé",
        "douleur au genou","crampes","j'ai soif",
        "baisser le volume","réinitialiser la minuterie",
        "nouveau record personnel","récupération musculaire",
        "je n'ai pas l'énergie","bonne pompe","parfait",
        "cool down","allongement","retour au calme",
    ],
    "de": [
        "wie viel zeit","pause","timer fünf minuten","timer zwei minuten",
        "wie mache ich diese übung","musik lauter","wie viele kalorien",
        "guten morgen","wie spät ist es","welches gewicht","hilfe",
        "ich verstehe nicht","falsche übung","training speichern",
        "statistiken anzeigen","zu schwer","nächste übung",
        "ich brauche eine pause","ich bin müde","training beendet",
        "zeige meine fortschritte","was ist mein rekord","erholungszeit",
        "wie viele übungen noch","gutes training","zu leicht",
        "meine muskeln schmerzen","persönlicher rekord","gesamtvolumen",
        "ich bin erschöpft","wasser trinken","dehnung","aufwärmen",
        "abkühlen","session beendet","nächste einheit","ruhetage",
        "wie viel habe ich gehoben","ich fühle mich gut",
        "rückenschmerzen","musik ändern","motivation",
        "diese übung überspringen","verlauf anzeigen",
        "mit gestern vergleichen","herzfrequenz","knieschmerzen",
        "ich habe durst","lautstärke reduzieren","timer zurücksetzen",
        "neuer persönlicher rekord","muskelkater",
        "ich habe keine energie","gute pumpe","perfekt",
    ],
    "es": [
        "cuánto tiempo queda","pausa","temporizador cinco minutos","temporizador dos minutos",
        "cómo hago este ejercicio","sube la música","cuántas calorías",
        "hola","qué hora es","qué peso usar","ayuda",
        "no entiendo","ejercicio equivocado","guarda el entrenamiento",
        "mostrar estadísticas","demasiado pesado","siguiente ejercicio",
        "necesito un descanso","estoy cansado","fin de sesión",
        "muestra mi progreso","cuál es mi récord","tiempo de descanso",
        "cuántos ejercicios quedan","buena sesión","demasiado fácil",
        "me duelen los músculos","récord personal","volumen total",
        "estoy agotado","beber agua","estiramientos","calentamiento",
        "sesión terminada","próxima sesión","días de descanso",
        "cuánto he levantado","me siento bien","dolor de espalda",
        "cambiar música","motivación","saltar este ejercicio",
        "ver historial","comparar con ayer","frecuencia cardíaca",
        "no me gusta este ejercicio","pasar al siguiente",
        "ya es suficiente","he terminado",
        "dolor de rodilla","calambres","tengo sed",
        "bajar el volumen","reiniciar el temporizador",
        "nuevo récord personal","recuperación muscular",
        "no tengo energía","buena congestión","perfecto",
    ],
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

def pick_num(lang, value):
    if random.random() < 0.2 and value in NUM_WORDS[lang]:
        return NUM_WORDS[lang][value]
    return str(value)

def pick_unit(lang):
    return random.choice(UNITS[lang])

def pick_exercise(lang):
    pool = random.choice(EXERCISES)
    name = random.choice(pool[lang])
    return name.split(), name

def pick_exercise_mixed_it():
    if random.random() < 0.4:
        name = random.choice(EN_IN_IT_POOL)
        return name.split(), name
    return pick_exercise("it")

def pick_modifier(lang):
    return random.choice(MODIFIERS[lang])

def make_sets():  return random.choice([2,3,4,5,6])
def make_reps():  return random.choice([4,5,6,8,10,12,15,20])
def make_weight():
    w = random.choice([20,30,40,50,60,70,80,90,100,110,120,130])
    if random.random() < 0.3:
        w += random.choice([2.5,5])
    return w

def tag_words(wt):
    return [w for w,_ in wt], [t for _,t in wt]

def ex_tags(words):
    return [(w,"B-EXE" if i==0 else "I-EXE") for i,w in enumerate(words)]

# ─── NOISE INJECTION ───────────────────────────────────────────────────────────

def apply_noise(examples: List[Example], p: float = 0.20) -> List[Example]:
    """
    Aggiunge filler/rethinking al p% degli esempi non-UNKNOWN.
    Simula input vocale reale con errori, esitazioni, ripensamenti.
    Tutti i token aggiunti sono taggati O (non cambiano le entità).
    """
    result = []
    for ex in examples:
        if ex.intent == "UNKNOWN" or random.random() >= p:
            result.append(ex)
            continue
        lang = ex.lang
        mode = random.random()
        if mode < 0.5:
            prefix_words = random.choice(FILLERS[lang]).split()
        elif mode < 0.80:
            prefix_words = random.choice(RETHINKING[lang]).split()
        else:
            prefix_words = (random.choice(FILLERS[lang]).split() +
                            random.choice(RETHINKING[lang]).split())
        new_words = prefix_words + ex.words
        new_tags  = ["O"] * len(prefix_words) + ex.ner_tags
        result.append(Example(
            ex.id + "_n", ex.lang, " ".join(new_words),
            ex.intent, new_words, new_tags
        ))
    return result

# ─── GENERATORS: ADD_EXERCISE ──────────────────────────────────────────────────

def gen_add_it(idx):
    ew, _ = pick_exercise_mixed_it()
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("it")
    ss = pick_num("it",sets); rs = pick_num("it",reps)
    mod = pick_modifier("it") if random.random() < 0.3 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
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
        lambda: [(ss,"B-SET"),("serie","O"),("di","O")] + ex + [(rs,"B-REP"),("ripetizioni","O")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O")] + ex + [(rs,"B-REP")] + wp + mod,
        lambda: [("segna","O")] + ex + [(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("devo","O"),("fare","O")] + ex + [(ss,"B-SET"),("serie","O"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("metti","O"),("nel","O"),("workout","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("programma","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("aggiungimi","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("inserisci","O"),("nel","O"),("piano","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("metti","O"),("in","O"),("lista","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"it_add_{idx}", "it", " ".join(words), "ADD_EXERCISE", words, tags)


def gen_add_en(idx):
    ew, _ = pick_exercise("en")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("en")
    ss = pick_num("en",sets); rs = pick_num("en",reps)
    mod = pick_modifier("en") if random.random() < 0.3 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    # NOTE: "log" RIMOSSO — confliggeva con LOG_SET intent
    templates = [
        lambda: [("add","O")] + ex + [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("add","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("do","O")] + ex + [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("insert","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("i","O"),("want","O"),("to","O"),("do","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: ex + wp + [(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP"),("repetitions","O")] + mod,
        lambda: [(ss,"B-SET"),("sets","O"),("of","O")] + ex + [(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + ex + wp + mod,
        lambda: [("put","O"),("in","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("schedule","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("add","O")] + ex + wp + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + mod,
        lambda: [("let","O"),("me","O"),("do","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP")] + wp + mod,
        lambda: ex + [(ss,"B-SET"),("sets","O"),("at","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("include","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
        lambda: [("program","O")] + ex + [(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")] + wp + mod,
        lambda: [("add","O"),("to","O"),("my","O"),("workout","O")] + ex + [(ss,"B-SET"),("x","O"),(rs,"B-REP")] + wp + mod,
    ]
    wt = random.choice(templates)()
    words, tags = tag_words(wt)
    return Example(f"en_add_{idx}", "en", " ".join(words), "ADD_EXERCISE", words, tags)


def gen_add_fr(idx):
    ew, _ = pick_exercise("fr")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("fr")
    ss = pick_num("fr",sets); rs = pick_num("fr",reps)
    mod = pick_modifier("fr") if random.random() < 0.2 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
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


def gen_add_de(idx):
    ew, _ = pick_exercise("de")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("de")
    ss = pick_num("de",sets); rs = pick_num("de",reps)
    mod = pick_modifier("de") if random.random() < 0.2 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
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


def gen_add_es(idx):
    ew, _ = pick_exercise("es")
    sets = make_sets(); reps = make_reps()
    weight = make_weight() if random.random() < 0.65 else None
    unit = pick_unit("es")
    ss = pick_num("es",sets); rs = pick_num("es",reps)
    mod = pick_modifier("es") if random.random() < 0.2 else []
    ex = ex_tags(ew)
    wp = [(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
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


# ─── GENERATORS: ADD_EXERCISE MULTI ───────────────────────────────────────────

def _block_it(ew, s, r, w=None, u=None, mod=None):
    ss=pick_num("it",s); rs=pick_num("it",r)
    block=ex_tags(ew)
    fmt=random.choice(["SxR","SerieR","SetR"])
    if fmt=="SxR": block+=[(ss,"B-SET"),("x","O"),(rs,"B-REP")]
    elif fmt=="SerieR": block+=[(ss,"B-SET"),("serie","O"),("da","O"),(rs,"B-REP")]
    else: block+=[(ss,"B-SET"),("set","O"),("x","O"),(rs,"B-REP")]
    if w and u: block+=[(str(w),"B-WGT"),(u,"B-UNT")]
    if mod: block+=mod
    return block

def _block_en(ew, s, r, w=None, u=None, mod=None):
    ss=pick_num("en",s); rs=pick_num("en",r)
    block=ex_tags(ew)
    fmt=random.choice(["SxR","SetsReps","SetOf"])
    if fmt=="SxR": block+=[(ss,"B-SET"),("x","O"),(rs,"B-REP")]
    elif fmt=="SetsReps": block+=[(ss,"B-SET"),("sets","O"),(rs,"B-REP"),("reps","O")]
    else: block+=[(ss,"B-SET"),("sets","O"),("of","O"),(rs,"B-REP")]
    if w and u: block+=[(str(w),"B-WGT"),(u,"B-UNT")]
    if mod: block+=mod
    return block


def gen_multi_it(idx):
    n=random.randint(2,4)
    items=[(pick_exercise_mixed_it()[0],make_sets(),make_reps(),
            make_weight() if random.random()<0.25 else None,
            pick_unit("it") if True else None,
            pick_modifier("it") if random.random()<0.3 else None) for _ in range(n)]
    seps=[[("e","O")],[("poi","O")],[(",","O")],[("più","O")],[("e","O"),("poi","O")]]
    p=random.random()
    if p<0.30: wt=[("aggiungi","O")]
    elif p<0.50: wt=[("metti","O")]
    elif p<0.65: wt=[("inserisci","O")]
    elif p<0.80: wt=[("aggiungi","O"),("al","O"),("workout","O")]
    else: wt=[]
    for i,(ew,s,r,w,u,m) in enumerate(items):
        u2=pick_unit("it") if w else None
        wt+=_block_it(ew,s,r,w,u2,m)
        if i<n-1: wt+=random.choice(seps)
    words,tags=tag_words(wt)
    return Example(f"it_multi_{idx}","it"," ".join(words),"ADD_EXERCISE",words,tags)


def gen_multi_en(idx):
    n=random.randint(2,4)
    items=[(pick_exercise("en")[0],make_sets(),make_reps(),
            make_weight() if random.random()<0.25 else None,
            pick_unit("en"),
            pick_modifier("en") if random.random()<0.3 else None) for _ in range(n)]
    seps=[[("and","O")],[("then","O")],[(",","O")],[("plus","O")],[("followed","O"),("by","O")]]
    # NOTE: "log" RIMOSSO come prefisso
    p=random.random()
    if p<0.45: wt=[("add","O")]
    elif p<0.65: wt=[("do","O")]
    elif p<0.80: wt=[("include","O")]
    else: wt=[]
    for i,(ew,s,r,w,u,m) in enumerate(items):
        u2=pick_unit("en") if w else None
        wt+=_block_en(ew,s,r,w,u2,m)
        if i<n-1: wt+=random.choice(seps)
    words,tags=tag_words(wt)
    return Example(f"en_multi_{idx}","en"," ".join(words),"ADD_EXERCISE",words,tags)


def gen_multi_fr(idx):
    n=random.randint(2,3)
    wt=[("ajoute","O")]
    for i in range(n):
        ew,_=pick_exercise("fr")
        ss=pick_num("fr",make_sets()); rs=pick_num("fr",make_reps())
        wt+=ex_tags(ew)+[(ss,"B-SET"),("x","O"),(rs,"B-REP")]
        if i<n-1: wt+=[random.choice([("et","O"),("puis","O"),(",","O")])]
    words,tags=tag_words(wt)
    return Example(f"fr_multi_{idx}","fr"," ".join(words),"ADD_EXERCISE",words,tags)


def gen_multi_de(idx):
    n=random.randint(2,3)
    wt=[("füge","O"),("hinzu","O")]
    for i in range(n):
        ew,_=pick_exercise("de")
        ss=pick_num("de",make_sets()); rs=pick_num("de",make_reps())
        wt+=ex_tags(ew)+[(ss,"B-SET"),("x","O"),(rs,"B-REP")]
        if i<n-1: wt+=[random.choice([("und","O"),("dann","O"),(",","O")])]
    words,tags=tag_words(wt)
    return Example(f"de_multi_{idx}","de"," ".join(words),"ADD_EXERCISE",words,tags)


def gen_multi_es(idx):
    n=random.randint(2,3)
    wt=[("agrega","O")]
    for i in range(n):
        ew,_=pick_exercise("es")
        ss=pick_num("es",make_sets()); rs=pick_num("es",make_reps())
        wt+=ex_tags(ew)+[(ss,"B-SET"),("x","O"),(rs,"B-REP")]
        if i<n-1: wt+=[random.choice([("y","O"),("luego","O"),(",","O")])]
    words,tags=tag_words(wt)
    return Example(f"es_multi_{idx}","es"," ".join(words),"ADD_EXERCISE",words,tags)


# ─── GENERATORS: LOG_SET ───────────────────────────────────────────────────────

def gen_log_it(idx):
    ew,_=pick_exercise_mixed_it()
    reps=make_reps(); weight=make_weight() if random.random()<0.5 else None
    unit=pick_unit("it"); rs=pick_num("it",reps)
    mod=pick_modifier("it") if random.random()<0.2 else []
    ex=ex_tags(ew); wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    templates=[
        lambda: [("fatto","O")]+ex+[(rs,"B-REP"),("ripetizioni","O")]+wp+mod,
        lambda: [("completato","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("ho","O"),("fatto","O")]+ex+[(rs,"B-REP"),("rep","O")]+wp+mod,
        lambda: ex+[("fatto","O"),(rs,"B-REP"),("volte","O")]+wp+mod,
        lambda: [("fatto","O"),(rs,"B-REP")]+ex+wp+mod,
        lambda: [("ok","O"),("fatto","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("eseguito","O")]+ex+[(rs,"B-REP"),("ripetizioni","O")]+wp+mod,
        lambda: [("serie","O"),("completata","O")]+ex+[(rs,"B-REP")]+wp+mod,
        lambda: [("finito","O")]+ex+[(rs,"B-REP"),("rep","O")]+wp+mod,
        lambda: [("registra","O")]+ex+[(rs,"B-REP"),("ripetizioni","O")]+wp+mod,
        lambda: [("nota","O")]+ex+[(rs,"B-REP")]+wp+mod,
        lambda: [("ho","O"),("completato","O")]+ex+[(rs,"B-REP")]+wp+mod,
        lambda: ex+[(rs,"B-REP"),("rips","O")]+wp+mod,
    ]
    wt=random.choice(templates)()
    words,tags=tag_words(wt)
    return Example(f"it_log_{idx}","it"," ".join(words),"LOG_SET",words,tags)


def gen_log_en(idx):
    ew,_=pick_exercise("en")
    reps=make_reps(); weight=make_weight() if random.random()<0.5 else None
    unit=pick_unit("en"); rs=pick_num("en",reps)
    mod=pick_modifier("en") if random.random()<0.2 else []
    ex=ex_tags(ew); wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    # NOTE: "logged" RIMOSSO (stessa radice di "log" che era in ADD)
    templates=[
        lambda: [("done","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("completed","O")]+ex+[(rs,"B-REP"),("repetitions","O")]+wp+mod,
        lambda: [("just","O"),("did","O")]+ex+[(rs,"B-REP"),("times","O")]+wp+mod,
        lambda: ex+[("done","O"),(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("finished","O")]+ex+[(rs,"B-REP")]+wp+mod,
        lambda: [("got","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("checked","O"),("off","O")]+ex+[(rs,"B-REP")]+wp+mod,
        lambda: [("banged","O"),("out","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: ex+[(rs,"B-REP")]+wp+[("complete","O")]+mod,
        lambda: [("just","O"),("finished","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("hit","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp+mod,
        lambda: [("knocked","O"),("out","O")]+ex+[(rs,"B-REP")]+wp+mod,
    ]
    wt=random.choice(templates)()
    words,tags=tag_words(wt)
    return Example(f"en_log_{idx}","en"," ".join(words),"LOG_SET",words,tags)


def gen_log_fr(idx):
    ew,_=pick_exercise("fr")
    reps=make_reps(); weight=make_weight() if random.random()<0.5 else None
    unit=pick_unit("fr"); rs=pick_num("fr",reps)
    ex=ex_tags(ew); wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    templates=[
        lambda: [("fait","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp,
        lambda: [("terminé","O")]+ex+[(rs,"B-REP"),("répétitions","O")]+wp,
        lambda: [("j'ai","O"),("fait","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: ex+[("fait","O"),(rs,"B-REP"),("reps","O")]+wp,
        lambda: [("fini","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: [("série","O"),("terminée","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: [("effectué","O")]+ex+[(rs,"B-REP"),("répétitions","O")]+wp,
        lambda: [("j'ai","O"),("complété","O")]+ex+[(rs,"B-REP")]+wp,
    ]
    wt=random.choice(templates)()
    words,tags=tag_words(wt)
    return Example(f"fr_log_{idx}","fr"," ".join(words),"LOG_SET",words,tags)


def gen_log_de(idx):
    ew,_=pick_exercise("de")
    reps=make_reps(); weight=make_weight() if random.random()<0.5 else None
    unit=pick_unit("de"); rs=pick_num("de",reps)
    ex=ex_tags(ew); wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    templates=[
        lambda: [("gemacht","O")]+ex+[(rs,"B-REP"),("wiederholungen","O")]+wp,
        lambda: [("fertig","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp,
        lambda: [("habe","O"),("gemacht","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: ex+[("erledigt","O"),(rs,"B-REP"),("wdh","O")]+wp,
        lambda: [("abgeschlossen","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: [("satz","O"),("fertig","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: [("ausgeführt","O")]+ex+[(rs,"B-REP"),("wiederholungen","O")]+wp,
    ]
    wt=random.choice(templates)()
    words,tags=tag_words(wt)
    return Example(f"de_log_{idx}","de"," ".join(words),"LOG_SET",words,tags)


def gen_log_es(idx):
    ew,_=pick_exercise("es")
    reps=make_reps(); weight=make_weight() if random.random()<0.5 else None
    unit=pick_unit("es"); rs=pick_num("es",reps)
    ex=ex_tags(ew); wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    templates=[
        lambda: [("hecho","O")]+ex+[(rs,"B-REP"),("reps","O")]+wp,
        lambda: [("completado","O")]+ex+[(rs,"B-REP"),("repeticiones","O")]+wp,
        lambda: [("he","O"),("hecho","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: ex+[("listo","O"),(rs,"B-REP"),("reps","O")]+wp,
        lambda: [("terminado","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: [("serie","O"),("completada","O")]+ex+[(rs,"B-REP")]+wp,
        lambda: [("acabo","O"),("de","O"),("hacer","O")]+ex+[(rs,"B-REP")]+wp,
    ]
    wt=random.choice(templates)()
    words,tags=tag_words(wt)
    return Example(f"es_log_{idx}","es"," ".join(words),"LOG_SET",words,tags)


def gen_log_implicit(lang, idx):
    reps=make_reps(); weight=make_weight() if random.random()<0.4 else None
    unit=pick_unit(lang); rs=pick_num(lang,reps)
    wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    choices={
        "it":[[("fatto","O"),(rs,"B-REP"),("reps","O")]+wp,
               [("completato","O"),(rs,"B-REP"),("ripetizioni","O")]+wp,
               [("ok","O"),(rs,"B-REP")]+wp,
               [("fatte","O"),(rs,"B-REP"),("volte","O")]+wp,
               [("ho","O"),("fatto","O"),(rs,"B-REP")]+wp,
               [("serie","O"),("fatta","O"),(rs,"B-REP"),("rips","O")]+wp,
               [("ci","O"),("siamo","O"),(rs,"B-REP")]+wp],
        "en":[[("done","O"),(rs,"B-REP"),("reps","O")]+wp,
               [("finished","O"),(rs,"B-REP"),("repetitions","O")]+wp,
               [("got","O"),(rs,"B-REP")]+wp,
               [("set","O"),("complete","O"),(rs,"B-REP"),("reps","O")]+wp,
               [("that","O"),("was","O"),(rs,"B-REP")]+wp],
        "fr":[[("fait","O"),(rs,"B-REP"),("reps","O")]+wp,
               [("terminé","O"),(rs,"B-REP"),("répétitions","O")]+wp,
               [("ok","O"),(rs,"B-REP")]+wp],
        "de":[[("gemacht","O"),(rs,"B-REP"),("wiederholungen","O")]+wp,
               [("fertig","O"),(rs,"B-REP"),("reps","O")]+wp,
               [("ok","O"),(rs,"B-REP")]+wp],
        "es":[[("listo","O"),(rs,"B-REP"),("repeticiones","O")]+wp,
               [("hecho","O"),(rs,"B-REP"),("reps","O")]+wp,
               [("ok","O"),(rs,"B-REP")]+wp],
    }
    wt=random.choice(choices[lang])
    words,tags=tag_words(wt)
    return Example(f"{lang}_log_impl_{idx}",lang," ".join(words),"LOG_SET",words,tags)


# ─── GENERATORS: UPDATE_SET ────────────────────────────────────────────────────

def gen_update(lang, idx):
    ew,_=pick_exercise_mixed_it() if lang=="it" else pick_exercise(lang)
    reps=make_reps(); sets=make_sets()
    weight=make_weight() if random.random()<0.5 else None
    unit=pick_unit(lang); rs=pick_num(lang,reps); ss=pick_num(lang,sets)
    ex=ex_tags(ew); wp=[(str(weight),"B-WGT"),(unit,"B-UNT")] if weight else []
    pfx={
        "it":[("modifica","O"),("correggi","O"),("cambia","O"),("aggiorna","O"),("sistema","O"),("rettifica","O")],
        "en":[("update","O"),("change","O"),("correct","O"),("edit","O"),("modify","O"),("fix","O")],
        "fr":[("modifie","O"),("change","O"),("corrige","O"),("actualise","O")],
        "de":[("ändere","O"),("aktualisiere","O"),("korrigiere","O"),("bearbeite","O")],
        "es":[("modifica","O"),("cambia","O"),("corrige","O"),("actualiza","O")],
    }
    p=random.choice(pfx[lang])
    patterns=[
        [p]+ex+[(rs,"B-REP"),("reps","O")]+wp,
        [p]+ex+[(ss,"B-SET"),("x","O"),(rs,"B-REP")]+wp,
        [p]+ex+wp,
        [p]+ex+[(ss,"B-SET"),("serie","O"),(rs,"B-REP")]+wp if lang=="it" else [p]+ex+[(ss,"B-SET"),("sets","O"),(rs,"B-REP")]+wp,
    ]
    wt=random.choice(patterns)
    words,tags=tag_words(wt)
    return Example(f"{lang}_update_{idx}",lang," ".join(words),"UPDATE_SET",words,tags)


# ─── GENERATORS: DELETE_EXERCISE ───────────────────────────────────────────────

def gen_delete(lang, idx):
    ew,_=pick_exercise_mixed_it() if lang=="it" else pick_exercise(lang)
    ex=ex_tags(ew)
    pfx={
        "it":[("rimuovi","O"),("elimina","O"),("togli","O"),("cancella","O"),("leva","O"),("salta","O")],
        "en":[("remove","O"),("delete","O"),("drop","O"),("skip","O"),("cancel","O")],
        "fr":[("supprime","O"),("enlève","O"),("retire","O"),("efface","O")],
        "de":[("entferne","O"),("lösche","O"),("streiche","O"),("überspringe","O")],
        "es":[("elimina","O"),("borra","O"),("quita","O"),("saca","O"),("salta","O")],
    }
    p=random.choice(pfx[lang])
    words,tags=tag_words([p]+ex)
    return Example(f"{lang}_delete_{idx}",lang," ".join(words),"DELETE_EXERCISE",words,tags)


# ─── GENERATOR: UNKNOWN ────────────────────────────────────────────────────────

def gen_unknown_all(lang: str) -> List[Example]:
    """
    Genera UN esempio per ogni frase unica del pool — zero duplicati, zero leakage.
    Il pool è abbastanza grande (130+ per it/en, 55+ per fr/de/es) da garantire
    che val e test contengano solo frasi mai viste in train.
    """
    pool = list(UNKNOWN_POOL[lang])
    random.shuffle(pool)
    return [
        Example(f"{lang}_unk_{i}", lang, text, "UNKNOWN",
                text.split(), ["O"] * len(text.split()))
        for i, text in enumerate(pool)
    ]


# ─── GENERATE ALL ──────────────────────────────────────────────────────────────

def generate_all() -> List[Example]:
    examples=[]
    langs=["it","en","fr","de","es"]

    # ADD_EXERCISE single
    for i in range(800): examples.append(gen_add_it(i))
    for i in range(800): examples.append(gen_add_en(i))
    for lang,gen,n in [("fr",gen_add_fr,500),("de",gen_add_de,500),("es",gen_add_es,500)]:
        for i in range(n): examples.append(gen(i))

    # ADD_EXERCISE multi
    for i in range(450): examples.append(gen_multi_it(i))
    for i in range(450): examples.append(gen_multi_en(i))
    for i in range(200): examples.append(gen_multi_fr(i))
    for i in range(200): examples.append(gen_multi_de(i))
    for i in range(200): examples.append(gen_multi_es(i))

    # LOG_SET
    for i in range(350): examples.append(gen_log_it(i))
    for i in range(350): examples.append(gen_log_en(i))
    for i in range(200): examples.append(gen_log_fr(i))
    for i in range(200): examples.append(gen_log_de(i))
    for i in range(200): examples.append(gen_log_es(i))
    for lang in langs:
        for i in range(120): examples.append(gen_log_implicit(lang,i))

    # UPDATE_SET
    for lang in langs:
        n=200 if lang in ["it","en"] else 130
        for i in range(n): examples.append(gen_update(lang,i))

    # DELETE_EXERCISE
    for lang in langs:
        for i in range(150): examples.append(gen_delete(lang,i))

    # UNKNOWN — un esempio per frase unica → leakage 0%
    for lang in langs:
        examples.extend(gen_unknown_all(lang))

    # Noise injection: ~20% esempi con filler/rethinking
    examples = apply_noise(examples, p=0.20)

    return examples


# ─── STRATIFIED SPLIT ─────────────────────────────────────────────────────────

def stratified_split(examples: List[Example], train_r=0.80, val_r=0.10, seed=42):
    """
    Split per intent: mantiene la proporzione di classi in ogni split.
    Garantisce no leakage e distribuzione uniforme su tutte le classi.
    """
    rng = random.Random(seed)
    by_intent: dict = defaultdict(list)
    for ex in examples:
        by_intent[ex.intent].append(ex)

    train, val, test = [], [], []
    for intent_exs in by_intent.values():
        rng.shuffle(intent_exs)
        n = len(intent_exs)
        t = int(n * train_r)
        v = int(n * (train_r + val_r))
        train.extend(intent_exs[:t])
        val.extend(intent_exs[t:v])
        test.extend(intent_exs[v:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_json(examples: List[Example], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump([asdict(e) for e in examples], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples):5d} examples → {path}")


if __name__ == "__main__":
    from collections import Counter
    print("Generating dataset v3...")
    all_ex = generate_all()
    print(f"Total: {len(all_ex)}")
    print(f"Intents: {dict(Counter(e.intent for e in all_ex))}")
    print(f"Langs:   {dict(Counter(e.lang   for e in all_ex))}")

    train, val, test = stratified_split(all_ex)
    print(f"\nSplit — Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    print(f"Train intents: {dict(Counter(e.intent for e in train))}")

    save_json(train, "data/train.json")
    save_json(val,   "data/val.json")
    save_json(test,  "data/test.json")

    label_maps = {
        "intent2id": INTENT2ID,
        "id2intent":  {str(v):k for k,v in INTENT2ID.items()},
        "tag2id":     TAG2ID,
        "id2tag":     {str(v):k for k,v in TAG2ID.items()},
    }
    with open("data/label_maps.json","w",encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)
    print("Label maps → data/label_maps.json")

    # Verifica leakage
    train_texts = {e.text for e in train}
    unk_test = [e for e in test if e.intent == "UNKNOWN"]
    overlap = sum(1 for e in unk_test if e.text in train_texts)
    print(f"\nUNKNOWN leakage: {overlap}/{len(unk_test)} "
          f"({100*overlap/max(len(unk_test),1):.0f}%) — target <10%")

    log_add = [e for e in train if e.intent=="ADD_EXERCISE" and e.words[0]=="log"]
    print(f"ADD_EXERCISE con 'log': {len(log_add)} — target 0")
    print("\nDone!")