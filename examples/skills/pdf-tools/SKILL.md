---
name: pdf-tools
description: |
  Read and work with PDF files: extract text and tables, search inside, merge several PDFs, split out pages (PDF、读取PDF、提取文字、提取表格、合并PDF、拆分PDF、年报PDF、论文PDF). Use whenever a task involves a .pdf file, local or downloaded.
version: "1.0.0"
author: AgentCLI examples
tags: [pdf, documents, 文档, 表格]
---

# PDF tools

`read_file` cannot read a PDF: it is binary. Use the script in this skill instead, with the
full path shown in the skill folder note:

```
python <skill folder>/scripts/pdf_tool.py info report.pdf
python <skill folder>/scripts/pdf_tool.py text report.pdf --pages 1-3
python <skill folder>/scripts/pdf_tool.py tables report.pdf --pages 12
python <skill folder>/scripts/pdf_tool.py search report.pdf 营业收入
python <skill folder>/scripts/pdf_tool.py merge merged.pdf a.pdf b.pdf
python <skill folder>/scripts/pdf_tool.py split report.pdf --pages 5-8 chapter2.pdf
```

## Workflow

1. Start with `info`: page count, and whether the PDF has a text layer.
2. For a long document, `search` for the terms the question is about, then read only those
   pages with `text --pages`. Do not dump a 200-page report into the conversation.
3. Use `tables` for tabular data such as financial statements; if nothing is detected, fall
   back to `text` for that page.
4. When quoting, give the page number: "（第 12 页）".

## A PDF from the web

`web_fetch` reports PDFs as binary files. Download first, then use this skill:

```
python -c "import urllib.request; urllib.request.urlretrieve('https://example.com/a.pdf', 'a.pdf')"
```

## Limits

- Scanned PDFs (images of pages) have no text layer; `info` says so. They need OCR, which
  this skill does not do. Tell the user rather than guessing the content.
- Encrypted PDFs need their password first.
- Writing commands create new files and never overwrite an input; pass `--force` only when the
  user asked to replace an existing output.
- Needs `pip install pypdf pdfplumber`; the script names the missing package if not.
