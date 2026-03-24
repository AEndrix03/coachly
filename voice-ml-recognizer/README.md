# 1. Installa PyTorch con ROCm 6

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# 2. Installa il resto

pip install -r requirements.txt

# 3. Genera il dataset (già fatto, ma puoi rieseguire per varianti diverse)

python generate_dataset.py

# 4. Fine-tuning (la variabile HSA è già settata dentro lo script)

HSA_OVERRIDE_GFX_VERSION=10.3.0 python finetune.py

# 5. Solo inference demo sul modello salvato

python finetune.py demo
