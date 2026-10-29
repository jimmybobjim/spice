#!/usr/bin/env python3
"""
Project inventory for Python source code.

Scans the directory containing this script (and all subdirectories) by default,
then reports functions, methods, classes and their likely types, variables,
imports, and other useful source-code metrics.

Usage:
    python project_inventory.py
    python project_inventory.py "path/to/project"
    python project_inventory.py . --json inventory.json

This is static analysis using Python's AST. It does not execute project code,
so class-type detection is best-effort and cannot resolve every alias or
dynamically-created object.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "venv", ".venv", "env", "node_modules", "site-packages",
    "build", "dist",
}


def dotted_name(node: ast.AST) -> str:
    """Return a dotted name for a Name/Attribute expression when possible."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)
    return ""


def decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> set[str]:
    return {dotted_name(d).split(".")[-1] for d in node.decorator_list}


def base_names(node: ast.ClassDef) -> set[str]:
    return {dotted_name(base).split(".")[-1] for base in node.bases}


def class_kinds(node: ast.ClassDef) -> list[str]:
    """Infer common class categories from bases and decorators."""
    bases = base_names(node)
    decorators = decorator_names(node)
    kinds: list[str] = []

    if "dataclass" in decorators:
        kinds.append("dataclass")
    if "Enum" in bases or any(x in bases for x in {"IntEnum", "StrEnum", "Flag", "IntFlag"}):
        kinds.append("enum")
    if "NamedTuple" in bases:
        kinds.append("namedtuple")
    if "TypedDict" in bases:
        kinds.append("typeddict")
    if "Protocol" in bases:
        kinds.append("protocol")
    if "ABC" in bases or "ABCMeta" in bases:
        kinds.append("abstract-base-class")
    if any(x in bases for x in {"BaseException", "Exception", "RuntimeError",
                                "ValueError", "TypeError", "OSError", "Warning"}):
        kinds.append("exception")
    if any(x in bases for x in {"BaseModel", "RootModel"}):
        kinds.append("pydantic-model")
    if any(x in bases for x in {"Model", "models.Model"}):
        kinds.append("model-like")
    if not kinds:
        kinds.append("regular")
    return kinds


@dataclass
class FileReport:
    file: str
    lines: int = 0
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    variables: Counter = field(default_factory=Counter)
    imports: Counter = field(default_factory=Counter)
    other: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "lines": self.lines,
            "class_count": len(self.classes),
            "classes": self.classes,
            "function_count": sum(1 for f in self.functions if f["kind"] == "function"),
            "method_count": sum(1 for f in self.functions if f["kind"] != "function"),
            "functions_and_methods": self.functions,
            "variables": dict(self.variables),
            "imports": dict(self.imports),
            "other": dict(self.other),
            "errors": self.errors,
        }


class InventoryVisitor(ast.NodeVisitor):
    def __init__(self, report: FileReport):
        self.report = report
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.function_kind_stack: list[str] = []
        self._seen_attributes: set[tuple[int, str, str]] = set()

    def _scope(self) -> str:
        if self.function_stack:
            return "local"
        if self.class_stack:
            return "class"
        return "module"

    def _record_target(self, target: ast.AST, category: str) -> None:
        if isinstance(target, ast.Name):
            self.report.variables[f"{category} ({self._scope()})"] += 1
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._record_target(element, category)
        elif isinstance(target, ast.Attribute):
            # Count self.x / cls.x / ClassName.x as attributes, without counting
            # arbitrary attributes such as some_object.x as project variables.
            owner = dotted_name(target.value)
            if owner in {"self", "cls"} or (
                self.class_stack and owner == self.class_stack[-1]
            ):
                key = (id(target), target.attr, category)
                if key not in self._seen_attributes:
                    self._seen_attributes.add(key)
                    self.report.variables[f"attribute assignment ({owner})"] += 1

    def _record_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = node.args
        count = (
            len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
            + (1 if args.vararg else 0) + (1 if args.kwarg else 0)
        )
        self.report.variables["parameters"] += count
        self.report.variables["annotated parameters"] += sum(
            1 for arg in args.posonlyargs + args.args + args.kwonlyargs if arg.annotation
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, async_=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, async_=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, async_: bool) -> None:
        decorators = decorator_names(node)
        if self.class_stack:
            if "staticmethod" in decorators:
                kind = "static method"
            elif "classmethod" in decorators:
                kind = "class method"
            elif any(x in decorators for x in {"property", "setter", "deleter"}):
                kind = "property/accessor"
            else:
                kind = "instance method"
        else:
            kind = "function"

        record = {
            "name": ".".join(self.class_stack + [node.name]),
            "kind": kind,
            "async": async_,
            "decorators": sorted(decorators),
            "line": node.lineno,
            "parameters": len(node.args.posonlyargs) + len(node.args.args)
                          + len(node.args.kwonlyargs)
                          + (1 if node.args.vararg else 0)
                          + (1 if node.args.kwarg else 0),
        }
        self.report.functions.append(record)
        self._record_parameters(node)
        self.function_stack.append(node.name)
        self.function_kind_stack.append(kind)
        for statement in node.body:
            self.visit(statement)
        self.function_kind_stack.pop()
        self.function_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        kinds = class_kinds(node)
        self.report.classes.append({
            "name": ".".join(self.class_stack + [node.name]),
            "types": kinds,
            "bases": sorted(base_names(node)),
            "decorators": sorted(decorator_names(node)),
            "line": node.lineno,
            "methods": sum(
                1 for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
        })
        self.class_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.class_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_target(target, "assignments")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_target(node.target, "annotated assignments")
        self.report.variables["type annotations"] += 1
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._record_target(node.target, "assignment expressions")
        self.generic_visit(node)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self._record_target(node.target, "loop targets")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars:
                self._record_target(item.optional_vars, "context-manager targets")
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.report.imports[alias.name] += 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.report.imports[f"{module}.{alias.name}".strip(".")] += 1

    def visit_Global(self, node: ast.Global) -> None:
        self.report.other["global declarations"] += len(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.report.other["nonlocal declarations"] += len(node.names)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.report.other["raise statements"] += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.report.other["try statements"] += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.report.other["assert statements"] += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.report.other["lambda expressions"] += 1
        args = node.args
        self.report.variables["lambda parameters"] += (
            len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
            + (1 if args.vararg else 0) + (1 if args.kwarg else 0)
        )
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.report.other["list comprehensions"] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.report.other["set comprehensions"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.report.other["dict comprehensions"] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.report.other["generator expressions"] += 1
        self.generic_visit(node)


def iter_python_files(root: Path, output_path: Path | None = None):
    output_resolved = output_path.resolve() if output_path else None
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = Path(current) / filename
            if output_resolved and path.resolve() == output_resolved:
                continue
            yield path


def analyze_file(path: Path, root: Path) -> FileReport:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")
    report = FileReport(file=str(path.relative_to(root)), lines=len(source.splitlines()))
    try:
        tree = ast.parse(source, filename=str(path))
        InventoryVisitor(report).visit(tree)
    except SyntaxError as exc:
        report.errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
    except Exception as exc:
        report.errors.append(f"{type(exc).__name__}: {exc}")
    return report


def build_summary(reports: list[FileReport]) -> dict[str, Any]:
    summary: Counter = Counter()
    class_types: Counter = Counter()
    function_kinds: Counter = Counter()
    variables: Counter = Counter()
    imports: Counter = Counter()
    other: Counter = Counter()

    for report in reports:
        summary["files scanned"] += 1
        summary["lines of Python"] += report.lines
        summary["classes"] += len(report.classes)
        for cls in report.classes:
            for kind in cls["types"]:
                class_types[kind] += 1
        for function in report.functions:
            function_kinds[function["kind"]] += 1
            summary["async functions/methods"] += int(function["async"])
        for key, value in report.variables.items():
            variables[key] += value
        for key, value in report.imports.items():
            imports[key] += value
        for key, value in report.other.items():
            other[key] += value
        summary["files with parse/read errors"] += bool(report.errors)

    summary["functions"] = function_kinds["function"]
    summary["methods"] = sum(
        value for key, value in function_kinds.items() if key != "function"
    )
    summary["functions and methods total"] = sum(function_kinds.values())

    return {
        "summary": dict(summary),
        "class types": dict(class_types),
        "function/method kinds": dict(function_kinds),
        "variable metrics": dict(variables),
        "imported names (occurrences)": dict(imports),
        "other constructs": dict(other),
    }


def print_report(reports: list[FileReport], summary: dict[str, Any], root: Path) -> None:
    print("=" * 72)
    print(f"PYTHON PROJECT INVENTORY: {root.resolve()}")
    print("=" * 72)

    for section in ("summary", "class types", "function/method kinds", "variable metrics", "other constructs"):
        print(f"\n{section.upper()}")
        print("-" * len(section))
        values = summary.get(section, {})
        if not values:
            print("  (none)")
        else:
            for name, value in values.items():
                print(f"  {name}: {value}")

    print("\nPER-FILE DETAILS")
    print("-" * 16)
    for report in reports:
        data = report.to_dict()
        print(f"\n{data['file']}  ({data['lines']} lines)")
        print(f"  Classes: {data['class_count']}; functions: {data['function_count']}; methods: {data['method_count']}")
        if report.classes:
            for cls in report.classes:
                print(f"    class {cls['name']} [{', '.join(cls['types'])}] — line {cls['line']}")
        if report.errors:
            for error in report.errors:
                print(f"  ERROR: {error}")

    print("\nNotes:")
    print("  • Variable metrics count declarations/assignments, not unique variable names.")
    print("  • Attribute assignments are limited to self.x, cls.x, and ClassName.x patterns.")
    print("  • Class types are inferred from names of bases/decorators; aliases may be missed.")
    print("  • Third-party/generated folders and hidden directories are skipped by default.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Count Python constructs across a project tree.")
    parser.add_argument(
        "root", nargs="?", default=None,
        help="project directory to scan (default: directory containing this script)",
    )
    parser.add_argument(
        "--json", dest="json_path", metavar="FILE",
        help="also write the full report as JSON",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).expanduser().resolve() if args.root else script_dir
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    json_path = Path(args.json_path).expanduser().resolve() if args.json_path else None
    reports = [analyze_file(path, root) for path in iter_python_files(root, json_path)]
    summary = build_summary(reports)
    print_report(reports, summary, root)

    if json_path:
        payload = {
            "root": str(root),
            **summary,
            "files": [report.to_dict() for report in reports],
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
