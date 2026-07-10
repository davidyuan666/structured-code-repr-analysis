#!/bin/bash
# AutoDL / GPU server setup for Structured Code Representations
# torch is assumed pre-installed by the platform

set -e

echo "=== Installing dependencies (uv) ==="
uv pip install -r code/requirements_nogpu.txt

echo ""
echo "=== Checking installation ==="
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers {transformers.__version__}')
import sentencepiece; import sklearn; import yaml; import tqdm; import datasets; import peft
print('All packages OK')
"

echo ""
echo "=== Downloading datasets from ModelScope ==="
echo "Please manually download structured-code-repr.zip from:"
echo "  https://www.modelscope.cn/datasets/davidyuan666/StructuredCodeRepresentations"
echo "Then unzip into code/data/processed/"
echo ""
echo "=== Ready to run ==="
echo "python code/experiments/run.py --config code/experiments/configs/rq1.yaml"
