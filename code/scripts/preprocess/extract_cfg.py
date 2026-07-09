"""Extract CFG block sequences from C/C++ source code using tree-sitter.

For C/C++: analyzes the AST to collect control-flow relevant nodes
(if, while, for, switch, goto, break, continue, return, etc.).

Usage:
    python scripts/preprocess/extract_cfg.py \
        --input data/raw/ojclone/code/ \
        --output data/processed/ojclone/cfg_seq/ \
        --language c
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

try:
    import tree_sitter
except ImportError:
    print("tree-sitter not installed. Install with: pip install tree-sitter tree-sitter-c tree-sitter-cpp")
    sys.exit(1)


LANGUAGE_PARSERS = {}

CFG_NODE_TYPES = {
    "if_statement", "while_statement", "do_statement", "for_statement",
    "switch_statement", "case_statement", "default_statement",
    "goto_statement", "labeled_statement",
    "break_statement", "continue_statement",
    "return_statement",
    "function_definition", "function_declarator",
    "preproc_if", "preproc_ifdef", "preproc_else", "preproc_elif",
    "try_statement", "catch_clause",  # C++
}


def get_parser(language: str):
    lang_modules = {
        "java": "tree_sitter_java",
        "cpp": "tree_sitter_cpp",
        "c": "tree_sitter_c",
    }
    if language not in lang_modules:
        raise ValueError(f"Unsupported language: {language}")
    if language not in LANGUAGE_PARSERS:
        module = __import__(lang_modules[language], fromlist=["language"])
        parser = tree_sitter.Parser(tree_sitter.Language(module.language()))
        LANGUAGE_PARSERS[language] = parser
    return LANGUAGE_PARSERS[language]


def extract_cfg_blocks(source: str, language: str) -> str:
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    blocks = ["ENTRY"]

    def collect_cfg(node):
        if node.type in CFG_NODE_TYPES:
            kind = node.type
            if kind in ("if_statement", "switch_statement"):
                kind = "BRANCH"
            elif kind in ("while_statement", "do_statement", "for_statement"):
                kind = "LOOP"
            elif kind in ("goto_statement", "break_statement", "continue_statement"):
                kind = "JUMP"
            elif kind == "return_statement":
                kind = "RETURN"
            elif kind in ("function_definition", "function_declarator"):
                kind = "FUNC"
            blocks.append(kind)
        for child in node.children:
            collect_cfg(child)

    collect_cfg(root)
    blocks.append("EXIT")
    return " ".join(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="c")
    parser.add_argument("--ext", default=".c", help="File extension filter")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = list(input_dir.glob(f"**/*{args.ext}"))
    print(f"Found {len(source_files)} {args.ext} files")

    count = 0
    with open(output_dir / "full.jsonl", "w", encoding="utf-8") as out:
        for fpath in source_files:
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
                seq = extract_cfg_blocks(source, args.language)
                if seq.strip():
                    out.write(json.dumps({
                        "file": str(fpath.relative_to(input_dir)),
                        "seq": seq,
                    }, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                print(f"  [skip] {fpath}: {e}")

    print(f"Output: {count} records -> {output_dir / 'full.jsonl'}")


if __name__ == "__main__":
    main()
