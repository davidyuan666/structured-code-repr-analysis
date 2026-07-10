# Structured Code Representations

Preprocessed structured code representations for clone detection and vulnerability detection. Three benchmark datasets, each with four representation schemes (raw code, AST sequence, CFG sequence, IR sequence), plus shared and independent SentencePiece BPE tokenizers (50K vocabulary).

| Dataset | Language | Task | #Samples | #Unique Sources |
|---------|----------|------|----------|-----------------|
| BigCloneBench (BCBench) | Java | Clone Detection | 1,731,860 pairs | 8,063 |
| OJClone (POJ-104) | C | Clone Detection | 106,000 pairs | 51,995 |
| Devign | C | Vulnerability Detection | 27,318 samples | 27,258 |

## Representations

Each dataset contains the following representation variants under `code/data/processed/<dataset>/`:

| Directory | Description | Used In |
|-----------|-------------|---------|
| `raw` | Whitespace-tokenized source code (baseline) | RQ1, RQ2, RQ3 |
| `ast_seq` | AST DFS pre-order, identifiers preserved (= `ast_id`) | RQ1, RQ2, RQ3, Ablation |
| `ast_seq_bfs` | AST breadth-first traversal | RQ4 |
| `ast_seq_dfs_post` | AST DFS post-order (children before parent) | RQ4 |
| `ast_seq_random` | AST with random node ordering (control) | RQ4 |
| `ast_noid` | AST with identifiers/literals abstracted to `ID`/`LIT` | Ablation |
| `cfg_seq` | Control-flow skeleton: ENTRY→BRANCH/LOOP/JUMP/RETURN/FUNC→EXIT | RQ1 |
| `ir_seq` | Statement-level instruction sequence: EXPR/DECL/CALL/IF/WHILE/... | RQ1, RQ2, RQ3 |

Each directory contains `train.jsonl` / `val.jsonl` / `test.jsonl` (8:1:1 split). Clone detection tasks (BCBench, OJClone) use paired format `{"code_a": "...", "code_b": "...", "label": 0/1}`; vulnerability detection (Devign) uses single format `{"code": "...", "label": 0/1}`.

### Per-Dataset Structure

```
code/data/processed/
├── bcbench/   (Java, clone detection, 1.7M pairs, 8,063 unique sources)
│   ├── raw/ train.jsonl val.jsonl test.jsonl
│   ├── ast_seq/ ast_seq_bfs/ ast_seq_dfs_post/ ast_seq_random/
│   └── ast_noid/ cfg_seq/ ir_seq/
├── devign/    (C, vulnerability detection, 27K samples, 27K unique sources)
│   └── (same 8 directories)
└── ojclone/   (C, clone detection, 106K pairs, 52K unique sources)
    └── (same 8 directories)
```

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
