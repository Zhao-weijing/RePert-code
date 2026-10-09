#!/usr/bin/env python3
"""Check public-page navigation, evidence numbers and README image integrity.

All numeric assertions use frozen Figure 2 CSV files in this repository.
This checks static publication integrity; it does not rerun scientific models.
"""
from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "index.html"


class Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.budgets: list[dict[str, str | None]] = []
        self.copy_buttons: list[str] = []
        self.images: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attr = dict(attributes)
        if attr.get("id"):
            self.ids.append(str(attr["id"]))
        for key in ("href", "src"):
            if attr.get(key):
                self.links.append((tag, str(attr[key])))
        if tag == "button" and "budget-button" in str(attr.get("class", "")).split():
            self.budgets.append(attr)
        if tag == "button" and attr.get("data-copy"):
            self.copy_buttons.append(str(attr["data-copy"]))
        if tag == "img":
            self.images.append(attr)


def check_links(document: Document) -> list[str]:
    errors: list[str] = []
    ids = set(document.ids)
    if len(ids) != len(document.ids):
        errors.append("Duplicate element ids")
    for tag, value in document.links:
        parsed = urlsplit(value)
        if parsed.scheme in ("https", "http", "mailto", "data") or value.startswith("//"):
            continue
        if parsed.scheme or parsed.netloc:
            errors.append(f"Unsupported local URL in {tag}: {value}")
            continue
        if parsed.path:
            target = (ROOT / unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append(f"Broken local {tag} reference: {value}")
        if parsed.fragment and not parsed.path and unquote(parsed.fragment) not in ids:
            errors.append(f"Broken navigation target: {value}")
    for name in document.copy_buttons:
        if name not in ids:
            errors.append(f"Copy control has missing target: {name}")
    for image in document.images:
        if not image.get("alt"):
            errors.append(f"Image is missing alt text: {image.get('src')}")
    for identifier in ("main", "figure1", "problem", "approach", "evidence", "data", "start", "resources"):
        if identifier not in ids:
            errors.append(f"Missing major page section: {identifier}")
    return errors


def load_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def check_frozen_values(document: Document, html: str) -> list[str]:
    errors: list[str] = []
    rows_e = load_rows("figures/data/figure_2/figure2_panelE.csv")
    rows_map = load_rows("figures/data/figure_2/figure2_panelF.csv")
    buttons = document.budgets
    if len(buttons) != 3:
        return [f"Expected 3 support-budget controls, found {len(buttons)}"]
    if [item.get("data-budget") for item in buttons] != ["1", "2", "3"]:
        errors.append("Support budgets must be 1, 2 and 3 in order")

    if [item.get("aria-pressed") for item in buttons] != ["true", "false", "false"]:
        errors.append("Exactly the first support budget must be selected by default")

    for button in buttons:
        try:
            budget = int(str(button["data-budget"]))
            observed = {
                "Mean": float(str(button["data-raw"])),
                "CFRA": float(str(button["data-reca"])),
            }
            n = int(str(button["data-count"]))
            map_observed = {
                "RawMean": float(str(button["data-map-raw"])),
                "CFRA": float(str(button["data-map-reca"])),
            }
        except (TypeError, KeyError, ValueError) as exception:
            errors.append(f"Invalid budget data: {exception}")
            continue
        for method, actual in observed.items():
            candidates = [
                row for row in rows_e
                if row.get("method") == method and row.get("metric") == "E"
                and int(row["budget"]) == budget
            ]
            if len(candidates) != 1:
                errors.append(f"No unique Figure 2 E record for {budget}/{method}")
                continue
            ref = candidates[0]
            if abs(float(ref["point"]) - actual) > 1e-10:
                errors.append(f"Incorrect E value for {budget}/{method}: {actual}")
            if int(ref["n_compounds"]) != n:
                errors.append(f"Incorrect eligible compound count for budget {budget}")
        for method, actual in map_observed.items():
            candidates = [
                row for row in rows_map
                if row.get("arm") == method and row.get("metric") == "mAP_at_33"
                and int(row["budget"]) == budget
            ]
            if len(candidates) != 1 or abs(float(candidates[0]["point"]) - actual) > 1e-10:
                errors.append(f"Incorrect retrieval mAP for budget {budget}/{method}")

    def initial_text(element_id: str, expected: str) -> None:
        # Static HTML should display the selected budget correctly even without JS.
        match = re.search(r'id="' + re.escape(element_id) + r'">([^<]+)', html)
        if not match or match.group(1).strip() != expected:
            errors.append(f"Incorrect initial {element_id}: expected {expected}")

    if buttons:
        try:
            first = buttons[0]
            raw = float(str(first["data-raw"]))
            reca = float(str(first["data-reca"]))
            n = int(str(first["data-count"]))
            initial_text("raw-score", f"{raw:.4f}")
            initial_text("reca-score", f"{reca:.4f}")
            initial_text("gain-score", f"{reca - raw:+.4f}")
            initial_text("compound-count", f"{n:,}")
            for identifier, value in (("raw-bar", raw), ("reca-bar", reca)):
                match = re.search(r'id="' + identifier + r'"\s+style="width:([0-9.]+)%"', html)
                if not match or abs(float(match.group(1)) - 100 * value / .30) > .11:
                    errors.append(f"Initial width of {identifier} not consistent with E data")
        except (TypeError, KeyError, ValueError) as exception:
            errors.append(f"Invalid selected default budget data: {exception}")
    return errors


def check_readmes() -> list[str]:
    errors: list[str] = []
    for name in ("README.md", "README_zh-CN.md"):
        content = (ROOT / name).read_text(encoding="utf-8")
        for src in re.findall(r'<img\s+[^>]*src="([^"]+)"', content, re.IGNORECASE):
            if src.startswith(("https://", "http://")):
                continue
            if not (ROOT / src).is_file():
                errors.append(f"{name} missing image {src}")
        if "assets/repert_figure1.png" not in content:
            errors.append(f"{name} is missing Figure 1")
    return errors


def check_assets() -> list[str]:
    errors: list[str] = []
    figure = ROOT / "assets/repert_figure1.png"
    if not figure.is_file() or figure.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        errors.append("Original Figure 1 raster is missing or invalid")
    if not (ROOT / "figures/data/figure_1/figure1_authors.pdf").is_file():
        errors.append("Authors' original Figure 1 PDF missing")
    css = (ROOT / "assets/site/repert.css").read_text(encoding="utf-8")
    js = (ROOT / "assets/site/repert.js").read_text(encoding="utf-8")
    for required in ("@media (max-width: 810px)", "@media (max-width: 550px)", "prefers-reduced-motion"):
        if required not in css:
            errors.append(f"Responsive/accessibility style missing: {required}")
    for required in ("renderBudget", "aria-pressed", "data-copy"):
        if required not in js:
            errors.append(f"JavaScript behavior missing: {required}")
    return errors


def main() -> None:
    html = PAGE.read_text(encoding="utf-8")
    document = Document()
    document.feed(html)
    errors = (
        check_links(document)
        + check_frozen_values(document, html)
        + check_readmes()
        + check_assets()
    )
    outcome = "PASS" if not errors else "FAIL"
    print(f"Public website navigation, source values, initial state and assets: {outcome}")
    print(f"Checked {len(document.budgets)} budget buttons and {len(document.ids)} element ids.")
    for error in errors:
        print(" -", error)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
