# Structured Code Representations

Preprocessed structured code representations for clone detection and vulnerability detection. Three benchmark datasets, each with four representation schemes (raw code, AST sequence, CFG sequence, IR sequence), plus shared and independent SentencePiece BPE tokenizers (50K vocabulary).

| Dataset | Language | Task | #Samples | #Unique Sources |
|---------|----------|------|----------|-----------------|
| BigCloneBench (BCBench) | Java | Clone Detection | 1,731,860 pairs | 8,063 |
| OJClone (POJ-104) | C | Clone Detection | 106,000 pairs | 51,995 |
| Devign | C | Vulnerability Detection | 27,318 samples | 27,258 |

## Representations

| Representation | Description |
|---------------|-------------|
| `raw` | Whitespace-tokenized source code sequence |
| `ast_seq` | AST node sequence (tree-sitter, DFS pre-order) |
| `cfg_seq` | Control-flow skeleton (ENTRY→BRANCH/LOOP/JUMP/RETURN/FUNC→EXIT) |
| `ir_seq` | Statement-level instruction sequence (EXPR, DECL, CALL, IF, ...) |

## Reproducibility

Paper: *Revisiting Structured Code Representations: Serialization, Vocabulary, and Traversal Effects*

Preprocessed dataset: [ModelScope](https://www.modelscope.cn/datasets/davidyuan666/StructuredCodeRepresentations)

## Quick Start (AutoDL / GPU Server)

```bash
git clone https://github.com/davidyuan666/structured-code-repr-analysis.git
cd structured-code-repr-analysis

# One-click setup (install deps + download dataset)
bash setup_autodl.sh

# RQ1a: shared vocabulary (primary experiment)
python code/experiments/run.py --config code/experiments/configs/rq1.yaml

# RQ1b: independent tokenizers (robustness check)
python code/experiments/run.py --config code/experiments/configs/rq1b.yaml
```

## Experiment Configs

| Config | Models | Representations | Description | Conditions |
|--------|--------|-----------------|-------------|------------|
| `rq1.yaml` | BiLSTM, TextCNN | raw, ast, cfg, ir | Shared SPM vocabulary | 2×4×3×5 = 120 |
| `rq1b.yaml` | BiLSTM, TextCNN | raw, ast, cfg, ir | Independent SPM tokenizers | 2×4×3×5 = 120 |
| `rq2.yaml` | CodeBERT, CodeT5 | raw, ast, ir | Full FT vocabulary gap | 2×3×3×5 = 90 |
| `rq3.yaml` | CodeBERT, CodeT5 | raw, ast, ir | LoRA vocabulary gap | 2×3×3×5 = 90 |
| `rq4.yaml` | BiLSTM, BERT, UniXcoder | 4 AST traversals | Traversal sensitivity | 3×4×3×5 = 180 |
| `ablation.yaml` | BiLSTM, BERT, UniXcoder | ast_id, ast_noid | Identifier contribution | 3×2×3×5 = 90 |

## Results Collection

```bash
# Aggregate results from all seeds
python code/scripts/analyze/collect_results.py --rq rq1 --output analysis/rq1_summary.json

# Statistical tests (Friedman, Nemenyi, Cohen's d)
python code/scripts/analyze/stat_tests.py --summary analysis/rq1_summary.json --rq rq1
```

## Repository Structure

```
code/
├── src/                  # Core library
│   ├── config.py         # Global configuration
│   ├── models/           # BiLSTM, TextCNN, PretrainedClassifier
│   ├── data/             # Dataset loaders (OJClone, Devign, BCBench)
│   ├── training/         # Trainer (full FT + LoRA)
│   └── eval/             # Metrics (F1, Δ, identifier contribution)
├── experiments/
│   ├── run.py            # Experiment runner
│   └── configs/          # YAML configs (rq1-4, ablation, rq1b)
├── scripts/
│   ├── preprocess/       # Data extraction (raw, AST, CFG, IR) + splits
│   ├── tokenize/         # SPM training, fragmentation analysis
│   └── analyze/          # Result collection, statistical tests
├── data/                 # [local only] Raw + processed datasets
└── results/              # [local only] Experiment outputs
```

## Data Pipeline

```
raw source files (.c/.java)
    ↓ extract_raw / extract_ast / extract_cfg / extract_ir (tree-sitter)
full.jsonl
    ↓ build_dataset_splits (8:1:1)
train.jsonl / val.jsonl / test.jsonl
    ↓ train SPM tokenizers (50K BPE)
spm_*.model
    ↓ RQ1 experiments
```

## License

Apache 2.0
