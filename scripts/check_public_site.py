#!/usr/bin/env python3
"""Check the public RePert site and frozen figure values without network access."""
from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

class Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.refs: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.add(str(attrs["id"]))
        for name in ("href", "src"):
            if attrs.get(name):
                self.refs.append((tag, str(attrs[name])))

def check_local_html_links(content: str) -> list[str]:
    doc = Document()
    doc.feed(content)
    errors: list[str] = []
    for tag, value in doc.refs:
        parsed = urlsplit(value)
        if parsed.scheme in ("https", "http", "mailto", "data") or value.startswith("//"):
            continue
        if parsed.scheme or parsed.netloc:
            errors.append(f"Unsupported local link ({tag}): {value}")
            continue
        if parsed.path:
            target = (ROOT / unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append(f"Missing local link ({tag}): {value}")
        if parsed.fragment and not parsed.path and unquote(parsed.fragment) not in doc.ids:
            errors.append(f"Missing in-page target ({tag}): {value}")
    return errors

def check_readme_figures(name: str) -> list[str]:
    content = (ROOT / name).read_text(encoding="utf-8")
    errors: list[str] = []
    for src in re.findall(r'<img\s+[^>]*src="([^"]+)"', content, re.IGNORECASE):
        if src.startswith(("http://", "https://")):
            continue
        target = ROOT / src
        if not target.is_file():
            errors.append(f"{name}: image not found: {src}")
    if "assets/repert_figure1.png" not in content:
        errors.append(f"{name}: original authors' Figure 1 PNG missing from introduction")
    return errors

def check_frozen_e_values(html: str) -> list[str]:
    file = ROOT / "figures/data/figure_2/figure2_panelE.csv"
    with file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    blocks = html.split('<div class="result-row">')[1:4]
    errors: list[str] = []
    if len(blocks) != 3:
        return ["Expected three frozen-result visual rows"]
    for budget, block in enumerate(blocks, 1):
        if f"{budget} repeat" not in block:
            errors.append(f"Unexpected visual row order at budget {budget}")
        sources = {}
        for method in ("Mean", "CFRA"):
            matches = [r for r in rows if int(r["budget"]) == budget and r["method"] == method and r["metric"] == "E"]
            if len(matches) != 1:
                errors.append(f"Missing unique {budget}/{method} frozen E")
            else:
                sources[method] = matches[0]
        values = [float(x) for x in re.findall(r'class="bar-value">([0-9.]+)',block)[:2]]
        if len(values) != 2:
            errors.append(f"Missing E bar labels for budget {budget}")
            continue
        for observed, method in zip(values, ("Mean", "CFRA")):
            if method in sources and abs(observed - float(sources[method]["point"])) >= 0.000051:
                errors.append(f"Figure 2 E mismatch for budget {budget}/{method}: {observed}")
        if "CFRA" in sources:
            n = sources["CFRA"]["n_compounds"]
            if f"{int(n):,} compounds" not in block:
                errors.append(f"Figure 2 n_compounds mismatch for budget {budget}")
    return errors

def main() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    errors = check_local_html_links(html)
    errors += check_readme_figures("README.md")
    errors += check_readme_figures("README_zh-CN.md")
    errors += check_frozen_e_values(html)
    figure = ROOT / "assets/repert_figure1.png"
    if not figure.is_file() or figure.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        errors.append("Missing or invalid authors' Figure 1 PNG")
    if not (ROOT / "figures/data/figure_1/figure1_authors.pdf").is_file():
        errors.append("Original Figure 1 PDF missing")
    print(f"Site links, README images, Figure 1 and frozen Figure 2 values: {'PASS' if not errors else 'FAIL'}")
    for error in errors:
        print(" -", error)
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
