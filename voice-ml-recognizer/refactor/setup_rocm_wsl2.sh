#!/usr/bin/env bash
# setup_rocm_wsl2.sh — Setup ROCm 6 + PyTorch on Ubuntu WSL2 for RX 6600
#
# PREREQUISITI (PowerShell admin):
#   wsl --unregister Ubuntu
#   wsl --install -d Ubuntu
#   (poi esegui questo script dentro WSL2)
#
# USO:
#   bash setup_rocm_wsl2.sh

set -e

RX6600_GFX="10.3.0"   # gfx1032 si spaccia per gfx1030
ROCM_VER="6.4.1"
PYTHON="python3.11"

echo "=== [1/5] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    wget curl gnupg2 software-properties-common \
    python3.11 python3.11-venv python3-pip \
    libstdc++6 libncurses6

echo "=== [2/5] ROCm $ROCM_VER repo ==="
# AMD official ROCm repo for Ubuntu 22.04
wget -q https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.1.60401-1_all.deb
sudo dpkg -i amdgpu-install_6.4.1.60401-1_all.deb
sudo amdgpu-install -y --usecase=rocm --no-dkms   # --no-dkms: no kernel driver in WSL2
rm amdgpu-install_6.4.1.60401-1_all.deb

# Add current user to render/video groups
sudo usermod -aG render,video "$USER" 2>/dev/null || true

echo "=== [3/5] Python venv ==="
$PYTHON -m venv ~/coachly-env
source ~/coachly-env/bin/activate

pip install -q --upgrade pip

echo "=== [4/5] PyTorch ROCm wheel ==="
# ROCm 6.1 wheel (closest stable to 6.4 for pytorch.org releases)
pip install -q torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

echo "=== [5/5] ML deps ==="
pip install -q \
    "transformers>=4.46.0" \
    "datasets>=2.20.0" \
    "accelerate>=0.33.0" \
    "peft>=0.12.0" \
    sentencepiece protobuf huggingface_hub

echo ""
echo "=== Verifica GPU ==="
HSA_OVERRIDE_GFX_VERSION=$RX6600_GFX python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA/ROCm:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
"

echo ""
echo "========================================="
echo "Setup completato!"
echo ""
echo "Per attivare l'env ed eseguire il training:"
echo "  source ~/coachly-env/bin/activate"
echo "  cd /mnt/c/Users/redeg/Documents/Progetti/Coachly/voice-ml-recognizer/refactor"
echo "  export HSA_OVERRIDE_GFX_VERSION=$RX6600_GFX"
echo "  python train_local_rocm.py --data_dir data_v2 --train_file train_aug.jsonl"
echo "========================================="
