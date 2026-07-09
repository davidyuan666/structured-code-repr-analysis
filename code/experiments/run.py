"""Experiment runner — reads a YAML config and executes all training conditions.

Usage:
    python experiments/run.py --config experiments/configs/rq1.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import HF_MODEL_NAMES, SEEDS, SEARCH_SEED, ROOT_DIR, CODE_DIR, DATA_DIR, RESULTS_DIR, SPM_MODEL_PATH
from src.data.bcbench import build_bcbench_dataset
from src.data.ojclone import build_ojclone_dataset
from src.data.devign import build_devign_dataset
from src.data.base_dataset import collate_fn
from src.models.bilstm import BiLSTMAttention
from src.models.textcnn import TextCNN
from src.models.pretrained import PretrainedCodeClassifier
from src.training.trainer import Trainer


DATASET_BUILDERS = {
    "bcbench": build_bcbench_dataset,
    "ojclone": build_ojclone_dataset,
    "devign": build_devign_dataset,
}


def load_tokenizer(tokenizer_type: str, tokenizer_path: str):
    if tokenizer_type == "spm":
        import sentencepiece as spm
        return spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
    elif tokenizer_type == "hf":
        from transformers import AutoTokenizer
        model_name = HF_MODEL_NAMES.get(tokenizer_path, tokenizer_path)
        return AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def build_model(model_type: str, vocab_size: int):
    if model_type == "bilstm":
        return BiLSTMAttention(vocab_size=vocab_size)
    elif model_type == "textcnn":
        return TextCNN(vocab_size=vocab_size)
    elif model_type == "codet5":
        model_name = HF_MODEL_NAMES["codet5"]
        return PretrainedCodeClassifier(model_name, encoder_only=False)
    elif model_type in ("codebert", "bert_base", "unixcoder"):
        model_name = HF_MODEL_NAMES[model_type]
        return PretrainedCodeClassifier(model_name, encoder_only=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_one_condition(
    config: Dict[str, Any],
    model_type: str,
    repr_name: str,
    ds_name: str,
    seed: int,
    result_dir: Path,
    is_search: bool = False,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = config.get("model_configs", {}).get(model_type, {})
    tok_type = model_cfg.get("tokenizer_type", config["tokenizer_type"])
    tok_path = model_cfg.get("tokenizer_path", config["tokenizer_path"])
    vocab_size = model_cfg.get("vocab_size", config.get("vocab_size", 30000))
    encoder_only = model_cfg.get("encoder_only", config.get("encoder_only", True))
    batch_size = model_cfg.get("batch_size", config["batch_size"])
    lr_candidates = model_cfg.get("lr_candidates", config.get("lr_candidates", [config["lr"]]))
    batch_size_candidates = model_cfg.get("batch_size_candidates", config.get("batch_size_candidates", []))
    max_epochs = model_cfg.get("max_epochs", config["max_epochs"])
    patience = model_cfg.get("patience", config.get("patience", 0))
    warmup_ratio = model_cfg.get("warmup_ratio", config.get("warmup_ratio", 0.0))

    ds_processed_dir = CODE_DIR / "data" / "processed" / ds_name

    if is_search:
        best_score = -1
        best_lr = config["lr"]
        best_bs = batch_size

        search_lrs = lr_candidates if config.get("search_lr", False) else [config["lr"]]
        search_bss = batch_size_candidates if config.get("search_batch", False) and batch_size_candidates else [batch_size]

        for lr in search_lrs:
            for bs in search_bss:
                tokenizer = load_tokenizer(tok_type, tok_path)
                model = build_model(model_type, vocab_size)
                if config.get("lora", False):
                    model = _apply_lora(model)

                train_set = DATASET_BUILDERS[ds_name](ds_processed_dir, repr_name, "train", tokenizer)
                val_set = DATASET_BUILDERS[ds_name](ds_processed_dir, repr_name, "val", tokenizer)
                test_set = DATASET_BUILDERS[ds_name](ds_processed_dir, repr_name, "test", tokenizer)

                train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_set, batch_size=bs * 2, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_set, batch_size=bs * 2, shuffle=False, collate_fn=collate_fn)

                out_dir = result_dir / f"search_lr{lr}_bs{bs}"
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    lr=lr, max_epochs=max_epochs,
                    warmup_ratio=warmup_ratio,
                    patience=patience,
                    device=device, output_dir=out_dir,
                )
                result = trainer.train()
                if result["best_val_f1"] > best_score:
                    best_score = result["best_val_f1"]
                    best_lr = lr
                    best_bs = bs

        return {"best_lr": best_lr, "best_batch_size": best_bs, "best_val_f1": best_score}
    else:
        tokenizer = load_tokenizer(tok_type, tok_path)
        model = build_model(model_type, vocab_size)
        if config.get("lora", False):
            model = _apply_lora(model)

        train_set = DATASET_BUILDERS[ds_name](ds_processed_dir, repr_name, "train", tokenizer)
        val_set = DATASET_BUILDERS[ds_name](ds_processed_dir, repr_name, "val", tokenizer)
        test_set = DATASET_BUILDERS[ds_name](ds_processed_dir, repr_name, "test", tokenizer)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            lr=config.get("best_lr", config["lr"]),
            max_epochs=max_epochs,
            warmup_ratio=warmup_ratio,
            patience=patience,
            device=device, output_dir=result_dir,
        )
        return trainer.train()


def _apply_lora(model):
    from peft import LoraConfig, get_peft_model, TaskType
    from src.config import LORA_CONFIG

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
    )
    if hasattr(model, "encoder") and hasattr(model.encoder, "base_model"):
        return model
    if hasattr(model, "encoder"):
        model.encoder = get_peft_model(model.encoder, lora_cfg)
    else:
        model.encoder = get_peft_model(model, lora_cfg)
    return model


def run_experiment(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    rq_name = config["rq"]
    models = config["models"]
    representations = config["representations"]
    datasets = config["datasets"]

    print(f"=== {rq_name} ===")
    print(f"Models: {models}")
    print(f"Reprs:  {representations}")
    print(f"Data:   {datasets}")

    for model_type in models:
        for repr_name in representations:
            for ds_name in datasets:
                base_dir = RESULTS_DIR / rq_name / model_type / repr_name / ds_name
                search_cache_file = base_dir / "search_result.json"

                if config.get("search_lr", False) or config.get("search_batch", False):
                    if search_cache_file.exists():
                        print(f"\n  [search-cached] model={model_type} repr={repr_name} ds={ds_name}")
                        with open(search_cache_file, "r") as f:
                            best = json.load(f)
                        print(f"    Best: lr={best['best_lr']} bs={best['best_batch_size']} val_f1={best['best_val_f1']:.4f}")
                        config["_search_cache"] = config.get("_search_cache", {})
                        config["_search_cache"][(model_type, repr_name, ds_name)] = best
                    else:
                        print(f"\n  [search] model={model_type} repr={repr_name} ds={ds_name}")
                        try:
                            best = run_one_condition(config, model_type, repr_name, ds_name, SEARCH_SEED, base_dir, is_search=True)
                            print(f"    Best: lr={best['best_lr']} bs={best['best_batch_size']} val_f1={best['best_val_f1']:.4f}")
                            base_dir.mkdir(parents=True, exist_ok=True)
                            with open(search_cache_file, "w") as f:
                                json.dump(best, f)
                            config["_search_cache"] = config.get("_search_cache", {})
                            config["_search_cache"][(model_type, repr_name, ds_name)] = best
                        except Exception as e:
                            print(f"    SEARCH FAILED: {e}")
                            import traceback
                            traceback.print_exc()
                            best = {"best_lr": config["lr"], "best_batch_size": config.get("batch_size", 16), "best_val_f1": 0.0}
                            base_dir.mkdir(parents=True, exist_ok=True)
                            with open(search_cache_file, "w") as f:
                                json.dump(best, f)
                            config["_search_cache"] = config.get("_search_cache", {})
                            config["_search_cache"][(model_type, repr_name, ds_name)] = best

                cached = config.get("_search_cache", {}).get((model_type, repr_name, ds_name), {})
                config["best_lr"] = cached.get("best_lr", config["lr"])
                config["batch_size"] = cached.get("best_batch_size", config["batch_size"])

                for seed in SEEDS:
                    result_dir = base_dir / f"seed_{seed}"
                    if (result_dir / "result.json").exists():
                        print(f"  [skip] model={model_type} repr={repr_name} ds={ds_name} seed={seed}")
                        continue
                    print(f"  [run]  model={model_type} repr={repr_name} ds={ds_name} seed={seed}")
                    try:
                        run_one_condition(config, model_type, repr_name, ds_name, seed, result_dir, is_search=False)
                    except Exception as e:
                        print(f"  [FAIL] {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n=== {rq_name} complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args.config)
