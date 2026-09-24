from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / "examples" / "skills"


def _script(relative: str):
    path = SKILLS / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_pdf(path: Path, pages: list[str]) -> None:
    """A minimal real PDF with one line of Helvetica text per page."""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
            + f"] /Count {len(pages)} >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(pages):
        stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>"
            ).encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


def test_pdf_text_search_split_and_merge(tmp_path, capsys):
    pdf = _script("pdf-tools/scripts/pdf_tool.py")
    report = tmp_path / "report.pdf"
    _make_pdf(report, ["Revenue grew 25 percent", "Risk factors and outlook"])

    pdf.main(["info", str(report)])
    pdf.main(["text", str(report), "--pages", "2"])
    pdf.main(["search", str(report), "risk"])
    out = capsys.readouterr().out
    assert "Pages: 2" in out and "Text layer: yes" in out
    assert "===== Page 2 =====" in out and "Risk factors" in out
    assert "p.2: Risk factors and outlook" in out

    part = tmp_path / "part.pdf"
    pdf.main(["split", str(report), "--pages", "2", str(part)])
    both = tmp_path / "both.pdf"
    pdf.main(["merge", str(both), str(report), str(part)])
    pdf.main(["info", str(both)])
    assert "Pages: 3" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="overwrite an input"):
        pdf.main(["merge", str(report), str(report), str(part)])


def test_pdf_text_turns_radical_code_points_back_into_characters():
    clean = _script("pdf-tools/scripts/pdf_tool.py")._clean

    # Chrome-style PDFs write 目/入/风 as Kangxi and simplified-radical code points.
    assert clean("项⽬ 营业收⼊ ⻛险") == "项目 营业收入 风险"
    # Chinese full-width punctuation is not touched.
    assert clean("同比增长 25%，（注）") == "同比增长 25%，（注）"


def test_docx_create_read_and_replace(tmp_path, capsys):
    docx_tool = _script("office-docs/scripts/docx_tool.py")
    notes = tmp_path / "weekly.md"
    notes.write_text(
        "# 周报\n\n本周完成了 **缓存优化**。\n\n- 命中率 63%\n\n"
        "| 指标 | 数值 |\n|---|---|\n| 测试 | 158 |\n",
        encoding="utf-8",
    )
    weekly = tmp_path / "weekly.docx"

    docx_tool.main(["create", str(weekly), "--from", str(notes)])
    docx_tool.main(["read", str(weekly)])
    text = capsys.readouterr().out
    assert "# 周报" in text and "- 命中率 63%" in text
    assert "| 指标 | 数值 |" in text and "| 测试 | 158 |" in text

    revised = tmp_path / "weekly-v2.docx"
    docx_tool.main(["replace", str(weekly), str(revised), "--find", "本周", "--replace", "上周"])
    docx_tool.main(["read", str(revised)])
    assert "上周完成了" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="already exists"):
        docx_tool.main(["create", str(weekly), "--from", str(notes)])


def test_xlsx_from_csv_set_formula_and_read(tmp_path, capsys):
    xlsx = _script("office-docs/scripts/xlsx_tool.py")
    data = tmp_path / "sales.csv"
    data.write_text("季度,收入,成本\nQ1,100,60\nQ2,120,70\n", encoding="utf-8")
    book, revised = tmp_path / "sales.xlsx", tmp_path / "sales-v2.xlsx"

    xlsx.main(["from-csv", str(book), "--csv", str(data), "--sheet", "数据"])
    xlsx.main(["set", str(book), str(revised), "--cell", "D1=毛利", "--cell", "D2==B2-C2"])
    capsys.readouterr()
    xlsx.main(["read", str(revised), "--formulas"])
    out = capsys.readouterr().out

    assert "| 季度 | 收入 | 成本 | 毛利 |" in out
    assert "| Q1 | 100 | 60 | =B2-C2 |" in out


def test_pptx_outline_round_trip(tmp_path, capsys):
    pptx_tool = _script("office-docs/scripts/pptx_tool.py")
    outline = tmp_path / "deck.md"
    outline.write_text(
        "# AgentCLI 介绍\n面试演示\n\n# 核心能力\n- 分层压缩\n  - 先清理工具结果\n> 讲缓存命中率\n",
        encoding="utf-8",
    )
    deck = tmp_path / "deck.pptx"

    pptx_tool.main(["create", str(deck), "--from", str(outline)])
    pptx_tool.main(["read", str(deck)])
    out = capsys.readouterr().out

    assert "## Slide 1: AgentCLI 介绍" in out and "- 面试演示" in out
    assert "## Slide 2: 核心能力" in out and "  - 先清理工具结果" in out
    assert "> Notes: 讲缓存命中率" in out
