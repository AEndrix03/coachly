 ---
  1. Installa le dipendenze

  # PyTorch per AMD ROCm (RX6600)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

  # oppure per NVIDIA / CPU
  pip install torch torchvision

  # Librerie comuni
  pip install -r requirements.txt

  ---
  2. Avvia il fine tuning

  # AMD RX6600 (ROCm)
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python finetune.py

  # NVIDIA / CPU (nessun env var)
  python finetune.py

  Il training dura ~15 epoche con early stopping (patience=4). Output atteso:
  Epoch 1/15 | loss=1.234 | intent_acc=0.71 | slot_f1=0.68
  ...
  Epoch 8/15 | loss=0.091 | intent_acc=0.97 | slot_f1=0.94  <-- best
  Early stopping triggered.

  Saved: output/workout_nlu/best_model.pt
  Saved: output/workout_nlu/workout_nlu.onnx
  Saved: output/workout_nlu/workout_nlu_int8.onnx

  ---
  3. Testa il modello (dopo il training)

  python finetune.py demo

  Questo carica best_model.pt e lo testa su frasi di esempio in tutte e 5 le lingue. Output atteso:
  [it] "aggiungi squat 4 serie da 10 a 100 kg"
    Intent:   ADD_EXERCISE (conf=0.997)
    Entities: {'EXE': ['squat'], 'SET': ['4'], 'REP': ['10'], 'WGT': ['100'], 'UNT': ['kg']}

  [it] "aggiungi panca piana 3 x 8 e trazioni 4 x 6"
    Intent:   ADD_EXERCISE (conf=0.991)
    Entities: {'EXE': ['panca piana', 'trazioni'], 'SET': ['3', '4'], 'REP': ['8', '6']}

  ---
  4. Test manuale su frase custom

  Crea un file test_custom.py:

  import torch, json
  from finetune import WorkoutNLU, load_label_maps, CFG
  from transformers import XLMRobertaTokenizerFast

  # Carica
  intent2id, id2intent, tag2id, id2tag = load_label_maps("data/label_maps.json")
  tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
  ckpt = torch.load("output/workout_nlu/best_model.pt", map_location="cpu")

  model = WorkoutNLU(
      model_name="xlm-roberta-base",
      num_intent_labels=len(intent2id),
      num_slot_labels=len(tag2id),
  )
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  def predict(text: str):
      words = text.split()
      enc = tokenizer(words, is_split_into_words=True, max_length=64,
                      padding="max_length", truncation=True, return_tensors="pt")
      with torch.no_grad():
          intent_logits, slot_logits = model(enc["input_ids"], enc["attention_mask"])

      intent = id2intent[str(intent_logits[0].argmax().item())]
      conf   = torch.softmax(intent_logits, -1)[0].max().item()

      slot_preds = slot_logits.argmax(-1)[0].tolist()
      word_ids   = enc.word_ids(0)
      seen = {}
      for pos, wid in enumerate(word_ids):
          if wid is not None and wid not in seen:
              seen[wid] = id2tag[str(slot_preds[pos])]

      entities = {}
      cur_type, cur_words = None, []
      for wid in sorted(seen):
          tag = seen[wid]
          if tag.startswith("B-"):
              if cur_type: entities.setdefault(cur_type, []).append(" ".join(cur_words))
              cur_type, cur_words = tag[2:], [words[wid]]
          elif tag.startswith("I-") and cur_type:
              cur_words.append(words[wid])
          else:
              if cur_type: entities.setdefault(cur_type, []).append(" ".join(cur_words))
              cur_type, cur_words = None, []
      if cur_type: entities.setdefault(cur_type, []).append(" ".join(cur_words))

      print(f'"{text}"')
      print(f"  Intent:   {intent} ({conf:.1%})")
      print(f"  Entities: {entities}\n")

  # Prova frasi tue
  predict("aggiungi bench press 3 x 10 a cedimento")
  predict("aggiungi una serie di bench press 3 x 8 push up e 2 crunch a cedimento")
  predict("fatto deadlift 5 reps 140 kg")
  predict("rimuovi squat")

  python test_custom.py