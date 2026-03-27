# setup_windows_gpu.ps1 — Setup GPU backend per training locale su RX 6600
# Esegui in PowerShell: .\setup_windows_gpu.ps1
#
# Tenta in ordine:
#   1) PyTorch ROCm (HIP SDK nativo, migliore)
#   2) torch-directml (fallback DirectX12, piu' lento ma funziona)

$ErrorActionPreference = "Continue"
$ROCM_PATH = "C:\Program Files\AMD\ROCm\6.4"

Write-Host "`n=== Coachly — Setup GPU (Windows + RX 6600) ===" -ForegroundColor Cyan

# Verifica Python
$pyver = python --version 2>&1
Write-Host "`nPython: $pyver"

# Verifica HIP SDK
if (Test-Path $ROCM_PATH) {
    Write-Host "HIP SDK trovato: $ROCM_PATH" -ForegroundColor Green
} else {
    Write-Host "HIP SDK non trovato in $ROCM_PATH — potrebbe mancare" -ForegroundColor Yellow
}

# ── Step 1: ML deps base ─────────────────────────────────────────────────────
Write-Host "`n[1/3] Installo dipendenze ML base..."
pip install -q --upgrade pip
pip install -q `
    "transformers>=4.46.0" `
    "datasets>=2.20.0" `
    "accelerate>=0.33.0" `
    "peft>=0.12.0" `
    sentencepiece protobuf huggingface_hub

# ── Step 2: Provo PyTorch ROCm ───────────────────────────────────────────────
Write-Host "`n[2/3] Provo PyTorch ROCm (HIP SDK)..." -ForegroundColor Yellow

# Disinstalla eventuali versioni precedenti
pip uninstall -q -y torch torchvision torchaudio 2>$null

# Tenta installazione ROCm wheel (rocm6.1 e' la build piu' stabile su pytorch.org)
pip install -q torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/rocm6.1

# Verifica se CUDA/ROCm e' disponibile con l'override per gfx1032
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"
$rocm_ok = python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" 2>&1
Write-Host "  ROCm check: $rocm_ok"

if ($rocm_ok -eq "OK") {
    Write-Host "`nROCm funziona!" -ForegroundColor Green
    $gpu_name = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    $vram     = python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1),'GB')" 2>&1
    Write-Host "  GPU:  $gpu_name"
    Write-Host "  VRAM: $vram"
} else {
    # ── Step 3: Fallback DirectML ─────────────────────────────────────────────
    Write-Host "`n[3/3] ROCm non disponibile, installo torch-directml (fallback)..." -ForegroundColor Yellow

    pip uninstall -q -y torch torchvision torchaudio 2>$null
    # DirectML usa CPU PyTorch + DirectX12 backend
    pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    pip install -q torch-directml

    $dml_ok = python -c "import torch_directml; d=torch_directml.device(); print('OK')" 2>&1
    if ($dml_ok -eq "OK") {
        $gpu_name = python -c "import torch_directml; print(torch_directml.device_name(0))" 2>&1
        Write-Host "`nDirectML funziona! GPU: $gpu_name" -ForegroundColor Green
        Write-Host "  (DirectML e' piu' lento di ROCm ma funziona su Windows senza WSL2)" -ForegroundColor Yellow
    } else {
        Write-Host "`nNeanche DirectML disponibile." -ForegroundColor Red
        Write-Host "  Controlla i driver AMD (devono essere 22.20+)"
    }
}

# ── Recap comandi training ────────────────────────────────────────────────────
Write-Host @"

=== Setup completato! ===

Per generare il dataset e avviare il training:

  # 1. Dataset (80k campioni)
  python dataset_creator_v2.py --total 80000 --output_dir data_v2 --md_dir exercises
  python augment.py --input data_v2/train.jsonl --output data_v2/train_aug.jsonl --factor 1.5

  # 2. Training locale
  `$env:HSA_OVERRIDE_GFX_VERSION="10.3.0"
  python train_local_rocm.py --data_dir data_v2 --train_file train_aug.jsonl

  # Output adapter in: output/rocm_lora/adapter/
  # (compatibile con test_local.py cambiando ADAPTER_DIR)

"@ -ForegroundColor Cyan
