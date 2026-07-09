"""Extract AST node sequences from source code using tree-sitter.

Usage:
    python scripts/preprocess/extract_ast.py \
        --input data/raw/bcbench/functions/ \
        --output data/processed/bcbench/ast_seq/ \
        --language java \
        --traversal dfs_pre \
        --identifier_mode keep

Supports traversal strategies: dfs_pre, dfs_post, bfs, random
Supports identifier modes: keep (preserve original names), abstract (ID/LIT placeholders)
"""
import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

try:
    import tree_sitter
except ImportError:
    print("tree-sitter not installed. Install with: pip install tree-sitter")
    sys.exit(1)


LANGUAGE_PARSERS = {}


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


def traverse_bfs(root, label_fn: Callable) -> List[str]:
    nodes = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        nodes.append(label_fn(node))
        for child in node.children:
            queue.append(child)
    return nodes


def traverse_dfs_pre(root, label_fn: Callable) -> List[str]:
    nodes = [label_fn(root)]
    for child in root.children:
        nodes.extend(traverse_dfs_pre(child, label_fn))
    return nodes


def traverse_dfs_post(root, label_fn: Callable) -> List[str]:
    nodes = []
    for child in root.children:
        nodes.extend(traverse_dfs_post(child, label_fn))
    nodes.append(label_fn(root))
    return nodes


def traverse_random(root, label_fn: Callable) -> List[str]:
    all_nodes = []

    def collect(node):
        all_nodes.append(label_fn(node))
        for child in node.children:
            collect(child)

    collect(root)
    rng = random.Random(id(root))
    rng.shuffle(all_nodes)
    return all_nodes


TRAVERSALS = {
    "dfs_pre": traverse_dfs_pre,
    "dfs_post": traverse_dfs_post,
    "bfs": traverse_bfs,
    "random": traverse_random,
}


def extract_ast_nodes(source: str, language: str, traversal: str, identifier_mode: str) -> str:
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    def node_label(node):
        if node.child_count == 0:
            if identifier_mode == "abstract":
                if node.type == "identifier":
                    return "ID"
                elif node.type in ("decimal_integer_literal", "string_literal",
                                   "number_literal", "string_content",
                                   "system_lib_string", "character_literal",
                                   "preproc_arg"):
                    return "LIT"
            text = source[node.start_byte:node.end_byte]
            return text
        return node.type

    if traversal not in TRAVERSALS:
        raise ValueError(f"Unknown traversal: {traversal}")

    nodes = TRAVERSALS[traversal](root, node_label)
    return " ".join(nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of source files")
    parser.add_argument("--output", required=True, help="Output directory for processed jsonl")
    parser.add_argument("--language", default="java")
    parser.add_argument("--traversal", default="dfs_pre")
    parser.add_argument("--identifier_mode", default="keep", choices=["keep", "abstract"])
    parser.add_argument("--ext", default=".java", help="File extension to process")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ext == "*":
        source_files = [f for f in input_dir.rglob("*") if f.is_file()]
    else:
        source_files = list(input_dir.glob(f"**/*{args.ext}"))
    print(f"Found {len(source_files)} files matching *{args.ext}")

    with open(output_dir / "full.jsonl", "w", encoding="utf-8") as out:
        for fpath in source_files:
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
                seq = extract_ast_nodes(source, args.language, args.traversal, args.identifier_mode)
                out.write(json.dumps({"file": str(fpath.relative_to(input_dir)), "seq": seq}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  [skip] {fpath}: {e}")

    print(f"Output: {output_dir / 'full.jsonl'}")


if __name__ == "__main__":
    main()
