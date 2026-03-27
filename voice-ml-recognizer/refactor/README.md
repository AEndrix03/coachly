# Coachly Refactor (FunctionGemma + Better Dataset)

Questa cartella contiene una pipeline semplice per ottenere un risultato migliore rispetto al vecchio setup intent/slot.

## Cosa c'e

- `dataset_creator.py`
  - Genera dataset sintetico IT/EN bilanciato per action.
  - Inserisce rumore realistico da STT (filler, ripensamenti, code-switching).
  - Produce target JSON rigoroso (`action` + `items`).
- `colab_functiongemma_train.py`
  - Fine-tuning QLoRA (4-bit) pensato per Colab T4.
  - Allena solo LoRA adapters, quindi meno memoria e costo.
  - Valutazione rapida su validita JSON + action accuracy.
- `colab_functiongemma_finetune.ipynb`
  - Notebook pronto da caricare in Colab.

## Perche dovrebbe funzionare meglio

- Classi bilanciate: meno rischio di collasso su `ADD_EXERCISE`.
- Hard negatives: frasi gym non operative in `UNKNOWN`.
- Output strutturato unico: il modello impara direttamente il JSON finale che ti serve.
- LoRA su base model piccolo: piu adatto a target Android (con quantizzazione in inferenza).

## Uso rapido (locale)

```bash
python refactor/dataset_creator.py --output_dir refactor/data --per_action_per_lang 340 --unknown_per_lang 170
python refactor/colab_functiongemma_train.py --data_dir refactor/data --output_dir refactor/output/functiongemma_qlora
```

## Uso in Colab

1. Carica `refactor/` su Colab (o il repo intero).
2. Apri `colab_functiongemma_finetune.ipynb`.
3. Esegui le celle in ordine.

## Note modello

- Default consigliato: `Qwen/Qwen2.5-0.5B-Instruct` (open, piccolo, stabile).
- Lo script tenta prima `google/functiongemma-270m-it` (gated).
- Se non hai accesso HF, fallback automatico su `unsloth/functiongemma-270m-it` (non gated).
- Fallback ulteriore: `Qwen/Qwen2.5-1.5B-Instruct`.
- Puoi forzare il modello con `--base_model <model_id>`.
- Se vuoi usare un modello gated Google, passa `--hf_token <token>` o imposta env `HF_TOKEN`.
