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
uv pip install -r code/requirements_autodl.txt --system --index-url https://mirrors.aliyun.com/pypi/simple
echo "  transformers sentencepiece tree-sitter sklearn peft datasets ... OK"

echo ""
echo "[3/4] Downloading dataset from ModelScope..."

ZIP_PATH="code/data/processed/structured-code-repr.zip"
if [ -f "$ZIP_PATH" ] && [ $(stat -c%s "$ZIP_PATH" 2>/dev/null || echo 0) -gt 100000000 ]; then
    echo "  Found existing zip ($(du -h "$ZIP_PATH" | cut -f1)), skip download"
elif [ -f "code/data/processed.zip" ] && [ $(stat -c%s "code/data/processed.zip" 2>/dev/null || echo 0) -gt 100000000 ]; then
    echo "  Found processed.zip, using it"
    ZIP_PATH="code/data/processed.zip"
else
    echo "  Auto-download not reliable on AutoDL. Please manually upload:"
    echo "    structured-code-repr.zip (2.5 GB) from your local machine"
    echo "    to ~/structured-code-repr-analysis/code/data/processed/"
    echo "  Then re-run: bash setup_autodl.sh"
    exit 1
fi

echo "  Extracting ..."
unzip -o "$ZIP_PATH" -d code/data/processed/ 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ERROR: unzip failed. The file may be corrupted."
    echo "  Please re-upload structured-code-repr.zip manually."
    exit 1
fi
echo "  Done"

echo ""
echo "[4/4] Verifying data..."
PROC_DIR="$(pwd)/code/data/processed"
echo "  Looking in: $PROC_DIR"
python -c "
from pathlib import Path
import os
d = Path(os.environ.get('PROC_DIR', 'code/data/processed'))
checks = {
    'OJClone raw train': d / 'ojclone' / 'raw' / 'train.jsonl',
    'OJClone AST train': d / 'ojclone' / 'ast_seq' / 'train.jsonl',
    'Devign raw train':  d / 'devign' / 'raw' / 'train.jsonl',
    'Devign AST train':  d / 'devign' / 'ast_seq' / 'train.jsonl',
    'BCBench raw train': d / 'bcbench' / 'raw' / 'train.jsonl',
    'BCBench AST train': d / 'bcbench' / 'ast_seq' / 'train.jsonl',
    'SPM shared model':  d / 'spm_shared_50k.model',
    'SPM raw model':     d / 'spm_raw_50k.model',
    'SPM struct model':  d / 'spm_struct_50k.model',
}
ok = all(p.exists() for p in checks.values())
for name, path in checks.items():
    print(f'  {\"OK\" if path.exists() else \"MISSING\"}: {name}')
if not ok:
    print('\n  Some files missing. Re-upload structured-code-repr.zip to code/data/processed/')
    exit(1)
print('  All data verified')
"

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
