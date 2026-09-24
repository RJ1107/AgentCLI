#!/usr/bin/env python3
"""PDF toolkit for the pdf-tools skill. Output is Markdown or plain text for the model to read.

Usage:
  pdf_tool.py info FILE                          pages, metadata, whether there is a text layer
  pdf_tool.py text FILE [--pages 1-3,7]          text of each page, with page markers
  pdf_tool.py tables FILE [--pages 2]            tables as Markdown (text-based PDFs only)
  pdf_tool.py search FILE KEYWORD [--pages ...]  matching lines with page numbers
  pdf_tool.py merge OUT IN1 IN2 [...]            join PDFs in order into a new file
  pdf_tool.py split FILE --pages 1-3 OUT         copy the chosen pages into a new file

Page numbers start at 1. Writing commands never overwrite an input or an existing file
unless --force is given.

Needs: pip install pypdf pdfplumber
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

# Piped output on Windows defaults to the ANSI code page; the agent reads UTF-8, and a
# character outside that code page would otherwise crash the script mid-document.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# pdfminer warns once per glyph about harmless font metadata; dozens of those lines would
# land in the agent's context for nothing.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Many PDF producers (Chrome's print-to-PDF among them) draw Chinese characters with CJK
# radical code points: text reads "项⽬" and "⻛险" instead of "项目" and "风险", so searches
# miss. NFKC maps the Kangxi radicals back; these simplified-Chinese radical forms have no
# NFKC mapping and are listed by hand (component-only forms such as 讠 are left alone).
_RADICALS = str.maketrans(
    {
        "⺠": "民", "⻅": "见", "⻆": "角", "⻉": "贝", "⻋": "车", "⻓": "长", "⻔": "门",
        "⻘": "青", "⻙": "韦", "⻚": "页", "⻛": "风", "⻜": "飞", "⻝": "食", "⻢": "马",
        "⻤": "鬼", "⻥": "鱼", "⻦": "鸟", "⻧": "卤", "⻨": "麦", "⻩": "黄", "⻪": "黾",
        "⻬": "齐", "⻮": "齿", "⻰": "龙",
    }
)  # fmt: skip


def _clean(text: str) -> str:
    # Only the radical blocks: NFKC on the whole text would also turn Chinese full-width
    # punctuation (，（）) into ASCII.
    return "".join(
        unicodedata.normalize("NFKC", char) if 0x2E80 <= ord(char) <= 0x2FDF else char
        for char in text
    ).translate(_RADICALS)


def _require(module: str, package: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"Missing Python package '{package}'. Install it with: pip install {package}")


def _page_numbers(spec: str | None, total: int) -> list[int]:
    """'1-3,7' -> [1, 2, 3, 7]; None -> every page. Out-of-range pages are an error."""

    if not spec:
        return list(range(1, total + 1))
    pages: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        start, _, end = part.partition("-")
        first, last = int(start), int(end or start)
        if first < 1 or last > total or first > last:
            sys.exit(f"page range {part} is outside 1-{total}")
        pages.extend(range(first, last + 1))
    return pages


def _check_output(out: Path, inputs: list[Path], force: bool) -> None:
    resolved = out.resolve()
    if any(resolved == item.resolve() for item in inputs):
        sys.exit("refusing to overwrite an input file; choose another output path")
    if out.exists() and not force:
        sys.exit(f"{out} already exists; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)


def cmd_info(args) -> None:
    pypdf = _require("pypdf", "pypdf")
    reader = pypdf.PdfReader(args.file)
    if reader.is_encrypted:
        print(f"{args.file}: encrypted; open it with the password first.")
        return
    meta = reader.metadata or {}
    sample = "".join((page.extract_text() or "") for page in reader.pages[:3]).strip()
    print(f"File: {args.file}")
    print(f"Pages: {len(reader.pages)}")
    for key in ("/Title", "/Author", "/Subject", "/Creator", "/CreationDate"):
        if meta.get(key):
            print(f"{key[1:]}: {meta.get(key)}")
    if sample:
        print("Text layer: yes (text can be extracted)")
    else:
        print(
            "Text layer: none found on the first pages; this is probably a scanned image PDF, "
            "which needs OCR. Say so instead of guessing its content."
        )


def cmd_text(args) -> None:
    pdfplumber = _require("pdfplumber", "pdfplumber")
    with pdfplumber.open(args.file) as pdf:
        for number in _page_numbers(args.pages, len(pdf.pages)):
            text = _clean(pdf.pages[number - 1].extract_text() or "")
            print(f"\n===== Page {number} =====")
            print(text.strip() or "(no text on this page)")


def cmd_tables(args) -> None:
    pdfplumber = _require("pdfplumber", "pdfplumber")
    found = 0
    with pdfplumber.open(args.file) as pdf:
        for number in _page_numbers(args.pages, len(pdf.pages)):
            for index, table in enumerate(pdf.pages[number - 1].extract_tables(), start=1):
                rows = [
                    [_clean(cell or "").replace("\n", " ").strip() for cell in row] for row in table
                ]
                if not rows:
                    continue
                found += 1
                width = max(len(row) for row in rows)
                rows = [row + [""] * (width - len(row)) for row in rows]
                print(f"\n### Page {number}, table {index}\n")
                print("| " + " | ".join(rows[0]) + " |")
                print("|" + "---|" * width)
                for row in rows[1:]:
                    print("| " + " | ".join(row) + " |")
    if not found:
        print("No tables detected. Tables drawn without ruling lines may need `text` instead.")


def cmd_search(args) -> None:
    pdfplumber = _require("pdfplumber", "pdfplumber")
    keyword = _clean(args.keyword).lower()
    hits = 0
    with pdfplumber.open(args.file) as pdf:
        for number in _page_numbers(args.pages, len(pdf.pages)):
            for line in _clean(pdf.pages[number - 1].extract_text() or "").splitlines():
                if keyword in line.lower():
                    hits += 1
                    print(f"p.{number}: {line.strip()}")
    print(f"\n{hits} matching line(s).")


def cmd_merge(args) -> None:
    pypdf = _require("pypdf", "pypdf")
    inputs = [Path(item) for item in args.inputs]
    out = Path(args.out)
    _check_output(out, inputs, args.force)
    writer = pypdf.PdfWriter()
    for item in inputs:
        writer.append(str(item))
    with out.open("wb") as handle:
        writer.write(handle)
    print(f"Wrote {out} ({len(writer.pages)} pages from {len(inputs)} files).")


def cmd_split(args) -> None:
    pypdf = _require("pypdf", "pypdf")
    reader = pypdf.PdfReader(args.file)
    out = Path(args.out)
    _check_output(out, [Path(args.file)], args.force)
    writer = pypdf.PdfWriter()
    pages = _page_numbers(args.pages, len(reader.pages))
    for number in pages:
        writer.add_page(reader.pages[number - 1])
    with out.open("wb") as handle:
        writer.write(handle)
    print(f"Wrote {out} with pages {args.pages} ({len(pages)} pages).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("info", cmd_info), ("text", cmd_text), ("tables", cmd_tables)):
        p = sub.add_parser(name)
        p.add_argument("file")
        if name != "info":
            p.add_argument("--pages")
        p.set_defaults(handler=handler)

    p = sub.add_parser("search")
    p.add_argument("file")
    p.add_argument("keyword")
    p.add_argument("--pages")
    p.set_defaults(handler=cmd_search)

    p = sub.add_parser("merge")
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--force", action="store_true")
    p.set_defaults(handler=cmd_merge)

    p = sub.add_parser("split")
    p.add_argument("file")
    p.add_argument("--pages", required=True)
    p.add_argument("out")
    p.add_argument("--force", action="store_true")
    p.set_defaults(handler=cmd_split)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
