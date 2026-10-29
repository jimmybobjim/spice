"""Export a project to a single printable HTML page.

Usage:
    python export_project.py path/to/project

The page starts with the project's README.md (rendered), followed by a folder
tree and every file that .gitignore does not exclude, one file per page.
Open project.html in a browser, then Print -> Save as PDF (tick
"Background graphics" so the colours show).

Install:     pip install pygments markdown pathspec
Type check:  mypy --strict export_project.py   (needs Python 3.12+)
"""

from __future__ import annotations

import base64
import html
import re
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import pathspec

# pygments and markdown are loaded dynamically and typed as Any, so strict
# checkers don't depend on whether their type stubs are installed.
_pygments: Any = import_module("pygments")
_lexers: Any = import_module("pygments.lexers")
_formatters: Any = import_module("pygments.formatters")
_util: Any = import_module("pygments.util")
_markdown: Any = import_module("markdown")

ALWAYS_SKIP: frozenset[str] = frozenset({".git"})
MAX_FILE_BYTES: int = 300_000  # larger files are listed but not printed
OUTPUT_FILE: str = "project.html"
RESULTS_DIR: str = "results"  # folder of .png images, shown after the code

# A .gitignore's rules apply relative to the folder that contains it.
type IgnoreRule = tuple[Path, pathspec.GitIgnoreSpec]

STYLE: str = """
@page { margin: 14mm; }
html { font-size: 16px; }
body { font-family: sans-serif; line-height: 1.45; }
/* Code is 0.8x the regular text size. rem (not em) so that nested elements
   such as pre inside .highlight don't compound to 0.64x. */
pre, code, .highlight { font-family: Consolas, monospace; font-size: 0.8rem; }
pre, .highlight { white-space: pre-wrap; }
.pg { break-before: page; font-size: 14px; }
.note { color: #666; font-style: italic; }
.result { break-inside: avoid; margin: 0 0 18px; }
.result figcaption { font-family: Consolas, monospace; font-size: 11px; margin-bottom: 4px; }
.result img { max-width: 100%; border: 1px solid #ccc; }
.readme img { max-width: 100%; }
.readme table { border-collapse: collapse; }
.readme th, .readme td { border: 1px solid #bbb; padding: 4px 8px; }
.readme blockquote { border-left: 4px solid #ccc; margin-left: 0; padding-left: 12px; color: #555; }
.readme code { background: #f2f2f2; padding: 1px 4px; }
.readme pre code { background: none; padding: 0; }
"""


def load_rules(directory: Path) -> list[IgnoreRule]:
    gitignore: Path = directory / ".gitignore"
    if not gitignore.is_file():
        return []
    lines: list[str] = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return [(directory, pathspec.GitIgnoreSpec.from_lines(lines))]


def is_ignored(path: Path, is_dir: bool, rules: Sequence[IgnoreRule]) -> bool:
    for base, spec in rules:
        relative: str = path.relative_to(base).as_posix()
        if is_dir:
            relative += "/"  # lets patterns like "build/" match folders
        if spec.match_file(relative):
            return True
    return False


def walk(
    directory: Path,
    inherited: Sequence[IgnoreRule],
    skip: frozenset[Path],
    prefix: str = "",
) -> tuple[list[str], list[Path]]:
    """Return the tree lines and the included files, in tree order."""
    rules: list[IgnoreRule] = [*inherited, *load_rules(directory)]
    try:
        entries: list[Path] = sorted(
            directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError:
        return [], []

    visible: list[Path] = [
        entry
        for entry in entries
        if entry.name not in ALWAYS_SKIP
        and not (entry.is_symlink() and entry.is_dir())  # avoid link loops
        and not is_ignored(entry, entry.is_dir(), rules)
        and entry.resolve() not in skip
    ]

    tree_lines: list[str] = []
    files: list[Path] = []
    for index, entry in enumerate(visible):
        is_last: bool = index == len(visible) - 1
        connector: str = "└── " if is_last else "├── "
        if entry.is_dir():
            tree_lines.append(f"{prefix}{connector}{entry.name}/")
            child_prefix: str = prefix + ("    " if is_last else "│   ")
            child_lines, child_files = walk(entry, rules, skip, child_prefix)
            tree_lines.extend(child_lines)
            files.extend(child_files)
        else:
            tree_lines.append(f"{prefix}{connector}{entry.name}")
            files.append(entry)
    return tree_lines, files


def pick_lexer(name: str, source: str) -> Any:
    try:
        return _lexers.get_lexer_for_filename(name, source)
    except _util.ClassNotFound:
        return _lexers.TextLexer()


def note(text: str) -> str:
    return f'<p class="note">[{html.escape(text)}]</p>'


def render_file(path: Path, root: Path, formatter: Any) -> str:
    relative: str = path.relative_to(root).as_posix()
    heading: str = f'<h2 class="pg">{html.escape(relative)}</h2>'

    size: int = path.stat().st_size
    if size == 0:
        return heading + note("empty file")
    if size > MAX_FILE_BYTES:
        return heading + note(f"file omitted: {size:,} bytes")

    data: bytes = path.read_bytes()
    if b"\0" in data:
        return heading + note(f"binary file omitted: {size:,} bytes")

    source: str = data.decode("utf-8-sig", errors="replace")
    lexer: Any = pick_lexer(path.name, source)
    return heading + str(_pygments.highlight(source, lexer, formatter))


def find_readme(root: Path) -> Path | None:
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.name.lower() == "readme.md":
            return entry
    return None


def render_readme(path: Path) -> str:
    text: str = path.read_text(encoding="utf-8-sig", errors="replace")
    body: str = str(
        _markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "sane_lists", "codehilite"],
            extension_configs={
                "codehilite": {"css_class": "highlight", "guess_lang": False}
            },
        )
    )
    return f'<section class="readme">{body}</section>'


def natural_key(path: Path) -> list[str | int]:
    """Sort plot2.png before plot10.png."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name.lower())]


def find_results(root: Path) -> list[Path]:
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.lower() == RESULTS_DIR:
            pngs: list[Path] = [
                p for p in entry.iterdir() if p.is_file() and p.suffix.lower() == ".png"
            ]
            return sorted(pngs, key=natural_key)
    return []


def render_results(images: Sequence[Path], root: Path) -> str:
    parts: list[str] = ['<h2 class="pg">Results</h2>']
    for image in images:
        encoded: str = base64.b64encode(image.read_bytes()).decode("ascii")
        label: str = html.escape(image.relative_to(root).as_posix())
        parts.append(
            f'<figure class="result"><figcaption>{label}</figcaption>'
            f'<img src="data:image/png;base64,{encoded}" alt="{label}"></figure>'
        )
    return "".join(parts)


def build_page(root: Path, skip: frozenset[Path]) -> str:
    tree_lines, files = walk(root, [], skip)
    readme: Path | None = find_readme(root)
    formatter: Any = _formatters.HtmlFormatter(linenos="inline", style="friendly")

    parts: list[str] = []
    if readme is not None:
        parts.append(render_readme(readme))
        files = [f for f in files if f != readme]  # already shown above, rendered
    else:
        parts.append(f"<h1>{html.escape(root.name)}</h1>")

    tree_text: str = "\n".join([f"{root.name}/", *tree_lines])
    tree_attr: str = ' class="pg"' if readme is not None else ""
    parts.append(f"<h2{tree_attr}>Project structure</h2><pre>{html.escape(tree_text)}</pre>")

    results: list[Path] = find_results(root)
    files = [f for f in files if f not in results]  # shown as images instead
    for file in files:
        parts.append(render_file(file, root, formatter))

    if results:
        parts.append(render_results(results, root))

    style_defs: str = str(formatter.get_style_defs(".highlight"))
    base: str = html.escape(root.as_uri() + "/")  # so relative README images resolve
    return (
        f'<html><head><meta charset="utf-8"><base href="{base}">'
        f"<style>{STYLE}{style_defs}</style></head><body>{''.join(parts)}</body></html>"
    )


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("Usage: python export_project.py path/to/project", file=sys.stderr)
        return 1

    root: Path = Path(argv[1]).resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}", file=sys.stderr)
        return 1

    output: Path = Path(OUTPUT_FILE).resolve()
    # Never include the output file itself if it sits inside the project.
    output.write_text(build_page(root, frozenset({output})), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))