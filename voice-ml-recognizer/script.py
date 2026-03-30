import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

# ---- Prompt base ----
SYSTEM_PROMPT = """Extract workout data.

Return only JSON:
{"entries":[{"action":"ADD|LOG|DELETE","exercise":"string|null","weight":number|null,"unit":"kg|lb|null","sets":number|null,"reps":number|null}]}

Rules:
- ho fatto/logga = LOG
- aggiungi = ADD
- cancella/togli = DELETE
- if unclear = LOG
- panca piana = bench press
- lat machine = lat pulldown
- missing fields = null
- if weight exists and no unit = kg
- no text outside JSON
"""

# ---- Test inputs ----
TEST_INPUTS = [
    "panca piana 80 kg 3x8",
    "ho fatto panca piana 80 chili tre serie da otto e poi lat machine 50 per 4 da 10",
    "panca 80 chili no aspetta 85 per tre da otto",
    "lat machine 12 reps poi 40 chili per 10",
    "togli panca 80 chili",
    "panca 80 3x8 lat machine 50 4x10 curl 10 2x12",
]

# ---- Simple validator ----
def validate_output(text):
    try:
        # estrai JSON (nel caso il modello sporchi output)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return False, "No JSON found"

        data = json.loads(match.group())

        if "entries" not in data:
            return False, "Missing 'entries'"

        for e in data["entries"]:
            if "action" not in e:
                return False, "Missing action"
            if e["action"] not in ["ADD", "LOG", "DELETE"]:
                return False, f"Invalid action {e['action']}"

        return True, "OK"

    except Exception as e:
        return False, str(e)

# ---- Load model ----
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu"
)

def run_test(input_text):
    prompt = SYSTEM_PROMPT + "\nInput:\n" + input_text

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=False
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # prendi solo output dopo prompt
    response = response[len(prompt):].strip()

    valid, msg = validate_output(response)

    return {
        "input": input_text,
        "output": response,
        "valid": valid,
        "message": msg
    }

# ---- Run tests ----
results = []

for i, test in enumerate(TEST_INPUTS):
    print(f"\n--- Test {i+1} ---")
    res = run_test(test)
    results.append(res)

    print("Input:", res["input"])
    print("Valid:", res["valid"], "-", res["message"])
    print("Output:", res["output"])

# ---- Summary ----
success = sum(1 for r in results if r["valid"])
print("\n====================")
print(f"SUCCESS: {success}/{len(results)}")