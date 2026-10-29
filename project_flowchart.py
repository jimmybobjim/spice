#!/usr/bin/env python3
"""Generate one Graphviz flowchart of a Python project's classes and call relationships.

Usage:
    python project_flowchart.py path/to/main.py
    python project_flowchart.py path/to/main.py --format pdf
    python project_flowchart.py path/to/main.py -o architecture.svg

Install:
    python -m pip install graphviz
    Also install the Graphviz system application from https://graphviz.org/download/
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    "build", "dist", "node_modules",
}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return None


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


@dataclass
class Definition:
    key: str
    name: str
    kind: str
    file: Path
    line: int
    calls: set[str] = field(default_factory=set)
    bases: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)


class Analyzer:
    def __init__(self, main_file: Path):
        self.main_file = main_file.resolve()
        self.root = self.main_file.parent
        self.definitions: dict[str, Definition] = {}
        self.short_names: dict[str, set[str]] = defaultdict(set)
        self.imports: dict[Path, dict[str, str]] = defaultdict(dict)
        self.class_keys: set[str] = set()

    def files(self) -> list[Path]:
        found = []
        for p in self.root.rglob("*.py"):
            rel = p.relative_to(self.root)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            found.append(p.resolve())
        if self.main_file.exists() and self.main_file not in found:
            found.append(self.main_file)
        return sorted(set(found))

    def module_name(self, path: Path) -> str:
        parts = list(path.relative_to(self.root).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts) or path.stem

    def parse(self, path: Path) -> ast.Module | None:
        try:
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            print(f"Warning: cannot parse {path}: {exc}", file=sys.stderr)
            return None

    def collect_imports(self, tree: ast.Module, path: Path) -> None:
        module = self.module_name(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[path][alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = module.split(".")[:-1]
                    for _ in range(max(0, node.level - 1)):
                        if parts:
                            parts.pop()
                    base = ".".join(parts + ([base] if base else []))
                for alias in node.names:
                    if alias.name != "*":
                        self.imports[path][alias.asname or alias.name] = f"{base}.{alias.name}".strip(".")

    def collect(self, tree: ast.Module, path: Path) -> None:
        module = self.module_name(path)

        def collect_calls(node: ast.AST, owner: str) -> None:
            # Walk this callable, but do not attribute nested callable bodies to it.
            for child in ast.walk(node):
                if child is not node and isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue
                if isinstance(child, ast.Call):
                    name = dotted_name(child.func)
                    if name:
                        self.definitions[owner].calls.add(name)

        def visit(body: list[ast.stmt], scope: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    full = ".".join([module, *scope, node.name])
                    key = "class:" + full
                    bases = {name for base in node.bases if (name := dotted_name(base))}
                    self.definitions[key] = Definition(key, full, "class", path, node.lineno, bases=bases)
                    self.short_names[node.name].add(key)
                    self.class_keys.add(key)
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_full = ".".join([module, *scope, node.name, child.name])
                            method_key = "method:" + method_full
                            self.definitions[method_key] = Definition(
                                method_key, method_full, "method", path, child.lineno
                            )
                            self.short_names[child.name].add(method_key)
                            self.definitions[key].methods.add(method_key)
                            collect_calls(child, method_key)
                        elif isinstance(child, ast.ClassDef):
                            visit([child], [*scope, node.name])
                        elif isinstance(child, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
                            visit(child.body, [*scope, node.name])
                            for attr in ("orelse", "finalbody"):
                                branch = getattr(child, attr, [])
                                if branch:
                                    visit(branch, [*scope, node.name])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    full = ".".join([module, *scope, node.name])
                    key = "function:" + full
                    self.definitions[key] = Definition(key, full, "function", path, node.lineno)
                    self.short_names[node.name].add(key)
                    collect_calls(node, key)
                elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
                    visit(node.body, scope)
                    for attr in ("orelse", "finalbody"):
                        branch = getattr(node, attr, [])
                        if branch:
                            visit(branch, scope)

        visit(tree.body, [])

    def resolve(self, caller: Definition, call: str) -> set[str]:
        result: set[str] = set()
        imports = self.imports.get(caller.file, {})
        first, dot, rest = call.partition(".")
        if first in imports:
            expanded = imports[first] + (("." + rest) if dot else "")
            result |= self.match_qualified(expanded)
        result |= self.match_qualified(call)
        if not result:
            matches = self.short_names.get(call.rsplit(".", 1)[-1], set())
            if len(matches) == 1:
                result |= matches
            else:
                methods = {key for key in matches if self.definitions[key].kind == "method"}
                if len(methods) == 1:
                    result |= methods
        return result

    def match_qualified(self, name: str) -> set[str]:
        return {
            key for key, d in self.definitions.items()
            if d.name == name or d.name.endswith("." + name)
        }

    def analyze(self) -> None:
        trees: dict[Path, ast.Module] = {}
        for file in self.files():
            tree = self.parse(file)
            if tree is not None:
                trees[file] = tree
                self.collect_imports(tree, file)
        for file, tree in trees.items():
            self.collect(tree, file)

    def to_dot(self, show_files: bool = False, include_external: bool = False) -> str:
        out = [
            "digraph PythonProject {",
            '  graph [rankdir="LR", splines=true, overlap=false, pad="0.25", nodesep="0.25", ranksep="0.8", label="Python Project Flowchart", labelloc="t", fontname="Helvetica"];',
            '  node [fontname="Helvetica", fontsize="9", style="filled", color="#555555"];',
            '  edge [fontname="Helvetica", fontsize="8", color="#666666", arrowsize="0.65"];',
        ]
        for key, d in sorted(self.definitions.items()):
            label = d.name
            if show_files:
                label += f"\n{d.file.relative_to(self.root)}:{d.line}"
            if d.kind == "class":
                attrs = 'shape="box", fillcolor="#DCEBFA", penwidth="1.4"'
            elif d.kind == "method":
                attrs = 'shape="ellipse", fillcolor="#E5F4E3"'
            else:
                attrs = 'shape="component", fillcolor="#FFF0CC"'
            out.append(f"  {quote(key)} [{attrs}, label={quote(label)}];")

        edges: set[tuple[str, str, str]] = set()
        for key in self.class_keys:
            cls = self.definitions[key]
            for method in cls.methods:
                edges.add((key, method, "contains"))
            for base in cls.bases:
                parents = self.match_qualified(base)
                if not parents:
                    parents = self.short_names.get(base.rsplit(".", 1)[-1], set())
                for parent in parents:
                    if self.definitions[parent].kind == "class":
                        edges.add((parent, key, "inherits"))

        external: dict[str, str] = {}
        for caller in self.definitions.values():
            for call in caller.calls:
                targets = self.resolve(caller, call)
                for target in targets:
                    if target != caller.key:
                        edges.add((caller.key, target, "calls"))
                if include_external and not targets:
                    external["external:" + call] = call

        for key, label in sorted(external.items()):
            out.append(f"  {quote(key)} [shape=plaintext, fontcolor=\"#777777\", label={quote(label)}];")
        for source, target, label in sorted(edges):
            if label == "inherits":
                attrs = 'style="dashed", color="#5B79A6", arrowhead="empty"'
            elif label == "contains":
                attrs = 'style="dotted", color="#8A8A8A", arrowhead="none"'
            else:
                attrs = 'color="#444444", arrowhead="normal"'
            out.append(f"  {quote(source)} -> {quote(target)} [label={quote(label)}, {attrs}];")
        if include_external:
            for caller in self.definitions.values():
                for call in caller.calls:
                    if not self.resolve(caller, call):
                        out.append(
                            f"  {quote(caller.key)} -> {quote('external:' + call)} "
                            '[label="external call", style="dashed", color="#AAAAAA"];'
                        )
        out.append("}")
        return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one Graphviz flowchart for a Python project.")
    parser.add_argument("main_module", type=Path, help="Path to the project's main .py file")
    parser.add_argument("-o", "--output", type=Path, help="Output file (.svg, .pdf, .png, or .dot)")
    parser.add_argument("--format", choices=("svg", "pdf", "png", "dot"), default=None)
    parser.add_argument("--show-files", action="store_true", help="Include relative source paths and line numbers")
    parser.add_argument("--include-external", action="store_true", help="Show unresolved/library calls (can clutter the graph)")
    args = parser.parse_args()

    main_file = args.main_module.expanduser().resolve()
    if not main_file.is_file() or main_file.suffix != ".py":
        parser.error(f"expected an existing Python file: {main_file}")

    output_format = args.format
    if args.output and output_format is None:
        suffix = args.output.suffix.lower().lstrip(".")
        output_format = suffix if suffix in {"svg", "pdf", "png", "dot"} else "svg"
    output_format = output_format or "svg"
    output = (args.output or main_file.with_name(f"{main_file.stem}_flowchart.{output_format}")).expanduser().resolve()

    analyzer = Analyzer(main_file)
    analyzer.analyze()
    if not analyzer.definitions:
        print("No classes, functions, or methods found.", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    dot = analyzer.to_dot(args.show_files, args.include_external)
    if output_format == "dot":
        output.write_text(dot, encoding="utf-8")
    else:
        try:
            import graphviz
        except ImportError:
            print("Install the Python package with: python -m pip install graphviz", file=sys.stderr)
            print("Also install Graphviz itself: https://graphviz.org/download/", file=sys.stderr)
            return 2
        try:
            source = graphviz.Source(dot, filename=str(output.with_suffix("")), format=output_format)
            rendered = Path(source.render(cleanup=True))
            if rendered.resolve() != output.resolve():
                rendered.replace(output)
        except graphviz.backend.ExecutableNotFound:
            print("Graphviz 'dot' executable not found. Install Graphviz and ensure it is on PATH.", file=sys.stderr)
            return 2
        except graphviz.backend.CalledProcessError as exc:
            print(f"Graphviz failed: {exc}", file=sys.stderr)
            return 2

    print(f"Analysed {len(analyzer.definitions)} definitions.")
    print(f"Flowchart written to: {output}")
    if output_format == "svg":
        print("Open the SVG in a browser; zoom in to inspect dense areas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
