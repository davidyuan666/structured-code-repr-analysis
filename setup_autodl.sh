#!/bin/bash
# AutoDL one-click setup for Structured Code Representations
# torch / CUDA are pre-installed by AutoDL

set -e

echo "============================================"
echo " Structured Code Representations — AutoDL"
echo "============================================"

echo ""
echo "[1/4] Checking environment..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable!'"
echo "  PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPU   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  OK"

echo ""
echo "[2/4] Installing Python packages (uv)..."

# Ensure uv is available (AutoDL may not have it pre-installed)
if ! command -v uv &> /dev/null; then
    echo "  uv not found, installing..."
    pip install uv -q
fi
uv pip install -r code/requirements_nogpu.txt --system --index-url https://mirrors.aliyun.com/pypi/simple
echo "  transformers sentencepiece tree-sitter sklearn peft datasets ... OK"

echo ""
echo "[3/4] Downloading dataset from ModelScope..."
python -c "
from modelscope.hub.api import HubApi
import os, zipfile, shutil
from pathlib import Path

proc = Path('code/data/processed')
proc.mkdir(parents=True, exist_ok=True)

print('  Downloading from davidyuan666/StructuredCodeRepresentations ...')
api = HubApi()
zip_path = proc / 'structured-code-repr.zip'
if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
    _ = api.dataset_download('davidyuan666/StructuredCodeRepresentations', local_dir=str(proc), zip_download=True)

zip_path = proc / 'structured-code-repr.zip'
if not zip_path.exists():
    # Try alternate path
    for f in proc.rglob('*.zip'):
        shutil.copy(f, zip_path)
        break

if not zip_path.exists():
    raise FileNotFoundError('Dataset zip not found. Download manually from https://www.modelscope.cn/datasets/davidyuan666/StructuredCodeRepresentations')

print(f'  Extracting {zip_path.stat().st_size/1024/1024:.0f} MB ...')
with zipfile.ZipFile(str(zip_path), 'r') as zf:
    zf.extractall(str(proc))
print('  Done')
" || echo "  WARNING: auto-download failed. Download zip from https://www.modelscope.cn/datasets/davidyuan666/StructuredCodeRepresentations and unzip into code/data/processed/"

echo ""
echo "[4/4] Verifying data..."
python -c "
from pathlib import Path
d = Path('code/data/processed')
checks = {
    'OJClone raw train': d / 'ojclone' / 'raw' / 'train.jsonl',
    'OJClone AST train': d / 'ojclone' / 'ast_seq' / 'train.jsonl',
    'Devign raw train':  d / 'devign' / 'raw' / 'train.jsonl',
    'BCBench raw train': d / 'bcbench' / 'raw' / 'train.jsonl',
    'SPM shared model':  d / 'spm_shared_50k.model',
    'SPM raw model':     d / 'spm_raw_50k.model',
    'SPM struct model':  d / 'spm_struct_50k.model',
}
ok = all(p.exists() for p in checks.values())
for name, path in checks.items():
    print(f'  {\"OK\" if path.exists() else \"MISSING\"}: {name}')
if not ok:
    print('\n  Some files missing. If auto-download failed, manually place structured-code-repr.zip in code/data/processed/')
    exit(1)
print('  All data verified')

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Run experiments:"
echo ""
echo "  # RQ1a: shared vocabulary (primary)"
echo "  python code/experiments/run.py --config code/experiments/configs/rq1.yaml"
echo ""
echo "  # RQ1b: independent tokenizers (robustness)"
echo "  python code/experiments/run.py --config code/experiments/configs/rq1b.yaml"
echo ""
echo "  # RQ2:  full fine-tuning vocab gap"
echo "  python code/experiments/run.py --config code/experiments/configs/rq2.yaml"
echo ""
echo "  # RQ3:  LoRA vocabulary gap"
echo "  python code/experiments/run.py --config code/experiments/configs/rq3.yaml"
echo ""
echo "  # RQ4:  traversal sensitivity"
echo "  python code/experiments/run.py --config code/experiments/configs/rq4.yaml"
echo ""
echo "  # Ablation: identifier contribution"
echo "  python code/experiments/run.py --config code/experiments/configs/ablation.yaml"
