"""Global configuration constants shared across all experiments."""

import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("STRUCT_REPR_ROOT", Path(__file__).resolve().parent.parent.parent))
CODE_DIR = ROOT_DIR / "code"

DATA_DIR = CODE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = CODE_DIR / "results"

# Safe path for SPM (avoids Unicode issues in C++ backends on Windows)
_SPM_CANDIDATE = PROCESSED_DIR / "spm_shared_50k.model"
if not _SPM_CANDIDATE.exists():
    # Try searching recursively (handle nested extraction from zip)
    _found = list(PROCESSED_DIR.rglob("spm_shared_50k.model"))
    if _found:
        _SPM_CANDIDATE = _found[0]

SPM_MODEL_PATH = str(_SPM_CANDIDATE.resolve()) if _SPM_CANDIDATE.exists() else ""

# Only use tempdir fallback when path contains non-ASCII chars (Windows Chinese path workaround)
if SPM_MODEL_PATH:
    try:
        SPM_MODEL_PATH.encode("ascii")
    except UnicodeEncodeError:
        import tempfile, shutil
        _tmp_dir = Path(tempfile.gettempdir()) / "opencode_spm"
        _tmp_dir.mkdir(exist_ok=True)
        _tmp_model = _tmp_dir / "spm_shared_50k.model"
        if not _tmp_model.exists():
            shutil.copy(_SPM_CANDIDATE, _tmp_model)
            _vocab_src = _SPM_CANDIDATE.with_suffix(".vocab")
            if _vocab_src.exists():
                shutil.copy(_vocab_src, _tmp_dir / "spm_shared_50k.vocab")
        SPM_MODEL_PATH = str(_tmp_model.resolve())

DATASETS = ["bcbench", "ojclone", "devign"]

REPRESENTATIONS_RQ1 = ["raw", "ast_seq", "cfg_seq", "ir_seq"]
REPRESENTATIONS_RQ2 = ["raw", "ast_seq", "ir_seq"]

TRAVERSALS = ["bfs", "dfs_pre", "dfs_post", "random"]
AST_VARIANTS = ["ast_id", "ast_noid"]

SEEDS = [42, 123, 456, 789, 1024]
SEARCH_SEED = 42

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test

MAX_SEQ_LEN = 512
SPM_VOCAB_SIZE = 50_000

MODEL_ARCHITECTURES = {
    "bilstm": {
        "hidden_dim": 256,
        "num_layers": 2,
    },
    "textcnn": {
        "filter_sizes": [2, 3, 4, 5],
        "num_filters": 128,
    },
}

HF_MODEL_NAMES = {
    "codebert": "microsoft/codebert-base",
    "codet5": "Salesforce/codet5-base",
    "bert_base": "google-bert/bert-base-uncased",
    "unixcoder": "microsoft/unixcoder-base",
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"],
}

RQ1_LR_CANDIDATES = [1e-4, 5e-4, 1e-3]
RQ1_BATCH_SIZE = 32
RQ1_MAX_EPOCHS = 50
RQ1_EARLY_STOP_PATIENCE = 5

RQ2_LR_CANDIDATES = [1e-5, 3e-5, 5e-5]
RQ2_BATCH_SIZE_CANDIDATES = [16, 32]
RQ2_CODET5_BATCH_SIZE_CANDIDATES = [8, 16]
RQ2_MAX_EPOCHS = 10
RQ2_WARMUP_RATIO = 0.10

RQ3_LR_CANDIDATES = [5e-6, 1e-5, 3e-5, 5e-5]
RQ3_BATCH_SIZE_CANDIDATES = [16, 32]
RQ3_CODET5_BATCH_SIZE_CANDIDATES = [8, 16]
RQ3_MAX_EPOCHS = 10
RQ3_WARMUP_RATIO = 0.20

COHEN_D_THRESHOLDS = {
    "trivial": 0.2,
    "small": 0.5,
    "medium": 0.8,
}
