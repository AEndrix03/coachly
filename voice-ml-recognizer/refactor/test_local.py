# test_local.py  —  usage: python test_local.py "add 3 sets of bench press"
import sys, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if len(sys.argv) < 2:
    print("Usage: python test_local.py \"<prompt>\"")
    sys.exit(1)

TEXT        = " ".join(sys.argv[1:])
ADAPTER_DIR = "./adapter"
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM = (
    "You are Coachly NLU. Convert workout speech-to-text into strict JSON.\nReturn ONLY valid JSON, no markdown.\nSchema:\n{\n  \"action\": \"ADD_EXERCISE|LOG_SET|UPDATE_SET|DELETE_EXERCISE|UNKNOWN\",\n  \"items\": [\n    {\n      \"exercise\": string,\n      \"sets\": integer|null,\n      \"reps\": integer|null,\n      \"weight\": number|null,\n      \"unit\": \"kg\"|\"lbs\"|null,\n      \"modifier\": \"to_failure\"|\"dropset\"|\"superset\"|\"amrap\"|\"pause\"|null\n    }\n  ]\n}"
)

print(f"Loading on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map=DEVICE)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEXT}],
    tokenize=False, add_generation_prompt=True
)
ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

with torch.no_grad():
    out = model.generate(
        input_ids=ids,
        max_new_tokens=1500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

raw = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)

try:
    result, _ = json.JSONDecoder().raw_decode(raw)
except Exception:
    result = {"_raw": raw}

print(json.dumps(result, ensure_ascii=False, indent=2))
