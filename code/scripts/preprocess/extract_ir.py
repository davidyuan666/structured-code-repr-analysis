"""Extract IR-like instruction sequences from C/C++ source using tree-sitter.

Produces a simplified instruction sequence by traversing statement-level AST nodes
and mapping them to abbreviated operation names.

Usage:
    python scripts/preprocess/extract_ir.py \
        --input data/raw/ojclone/code/ \
        --output data/processed/ojclone/ir_seq/ \
        --language c
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import tree_sitter
except ImportError:
    print("tree-sitter not installed. Install with: pip install tree-sitter tree-sitter-c tree-sitter-cpp")
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


IR_STATEMENT_TYPES = {
    "expression_statement": "EXPR",
    "declaration": "DECL",
    "return_statement": "RET",
    "if_statement": "IF",
    "while_statement": "WHILE",
    "do_statement": "DO",
    "for_statement": "FOR",
    "switch_statement": "SWITCH",
    "break_statement": "BREAK",
    "continue_statement": "CONT",
    "goto_statement": "GOTO",
    "labeled_statement": "LABEL",
    "compound_statement": "BLOCK",
    "assignment_expression": "ASSIGN",
    "call_expression": "CALL",
    "binary_expression": "BINOP",
    "unary_expression": "UNOP",
    "update_expression": "UPDATE",
    "init_declarator": "INIT",
    "pointer_declarator": "PTR",
    "array_declarator": "ARR",
    "function_definition": "FUNC",
    "preproc_include": "INCLUDE",
    "preproc_def": "DEFINE",
}


def extract_ir(source: str, language: str) -> str:
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    instructions = []

    def visit(node):
        node_type = node.type
        if node_type in IR_STATEMENT_TYPES:
            instructions.append(IR_STATEMENT_TYPES[node_type])
        for child in node.children:
            visit(child)

    visit(root)
    return " ".join(instructions)


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
                seq = extract_ir(source, args.language)
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
