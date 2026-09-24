---
name: office-docs
description: |
  Read and create Microsoft Office files: Word .docx, Excel .xlsx, PowerPoint .pptx (Word文档、Excel表格、PPT演示文稿、读取、生成、修改、替换文字、单元格、幻灯片). Use whenever a task involves one of these files.
version: "1.0.0"
author: AgentCLI examples
tags: [docx, xlsx, pptx, office, word, excel, powerpoint, 文档]
---

# Office documents

`read_file` cannot read Office files: they are zipped XML. Use the scripts in this skill, with
the full paths shown in the skill folder note. Each script prints its full usage with `--help`.

## Word (.docx) — `scripts/docx_tool.py`

```
python <skill folder>/scripts/docx_tool.py read contract.docx
python <skill folder>/scripts/docx_tool.py create report.docx --from report.md
python <skill folder>/scripts/docx_tool.py replace contract.docx contract-v2.docx --find 甲方 --replace 委托方
```

To write a document, first write its content as Markdown in a `.md` file (headings, `- `
bullets, `1. ` numbered items, `| a | b |` tables, `**bold**`), then run `create`.

## Excel (.xlsx) — `scripts/xlsx_tool.py`

```
python <skill folder>/scripts/xlsx_tool.py info sales.xlsx
python <skill folder>/scripts/xlsx_tool.py read sales.xlsx --sheet Q3 --range A1:F40
python <skill folder>/scripts/xlsx_tool.py from-csv sales.xlsx --csv sales.csv --sheet Data
python <skill folder>/scripts/xlsx_tool.py set sales.xlsx sales-v2.xlsx --cell F2==SUM(B2:E2)
```

`read` shows values Excel last calculated. A workbook created by a script has no calculated
values until it is opened in Excel; use `--formulas` to see the formulas, and do sums or
averages yourself with a short `python -c` if you need the numbers now.

## PowerPoint (.pptx) — `scripts/pptx_tool.py`

```
python <skill folder>/scripts/pptx_tool.py read deck.pptx
python <skill folder>/scripts/pptx_tool.py create deck.pptx --from outline.md
```

Outline: each `# ` line is a new slide title, `- ` lines are bullets (indent two spaces for a
sub-bullet), `> ` lines are speaker notes. A first slide with no bullets becomes a cover and
its plain lines become the subtitle.

## Rules

- Never overwrite the user's original. Write changes to a new file (for example
  `name-v2.docx`) unless the user asked to replace it; `--force` is only for that.
- Reading output is Markdown; quote from it rather than paraphrasing numbers.
- Layout is basic: default templates, no images or charts. Say so if the user expects a
  designed document.
- Needs `pip install python-docx openpyxl python-pptx`; each script names the missing package.
