---
name: finance-qa
description: |
  Answer questions about listed companies, stocks, financial statements and metrics (营收、利润、增长率、毛利率、市盈率、估值、财报、年报、股票、公司基本面). Finds primary sources, computes numbers with a script instead of mental arithmetic, and cites every figure. Not for personal investment advice.
version: "1.0.0"
author: AgentCLI examples
tags: [finance, stocks, 金融, 股票, 财报]
---

# Finance Q&A

Answer the way a careful analyst writes a research note: sourced numbers, correct arithmetic,
clear limits.

## Workflow

1. Pin down the question: which company (full name and ticker, e.g. 贵州茅台 600519.SH,
   Apple AAPL), which metric, which period (FY2024, 2025Q2, TTM).
2. Get primary figures. Search with `web_search`, then read the filing or official release
   with `web_fetch`. Prefer annual reports and exchange filings over news articles. The
   sources to try are listed in `references/metrics.md`.
3. Compute with the script, never in your head. Run it with bash, using the full path from
   the skill folder shown when this skill loads:

   ```
   python <skill folder>/scripts/fin_calc.py growth 2022=100 2023=120 2024=150
   python <skill folder>/scripts/fin_calc.py cagr 100 150 2
   python <skill folder>/scripts/fin_calc.py margins revenue=500 gross_profit=200 net_income=60
   python <skill folder>/scripts/fin_calc.py valuation price=30 eps=2 bvps=12 dps=0.6
   ```

   These numbers only show the syntax. Use the figures you found, never these.

   Keep units consistent (亿元 with 亿元) and say the unit in the answer.
4. Read `references/metrics.md` only when you need a definition or a caveat (for example
   归母净利润 versus 净利润, or TTM versus static P/E).

## Answer format

- Lead with the direct answer in one or two sentences.
- Then a small table of the figures used, each with its period and source.
- Then the calculation result from the script, and one or two sentences of interpretation
  (what drove it, what to compare it with).
- End with the data date, for example "数据截至 2025 年报（2026-03 披露）".

## Limits

- Facts and analysis only. Do not tell the user to buy, sell, or hold, and do not give price
  targets or portfolio allocations for them; if asked, say you are not a licensed advisor and
  offer the facts that would inform the decision.
- If a figure cannot be found in a primary source, say so rather than estimating silently.
  Mark any estimate as an estimate and say how it was made.
- Market prices move: state when a price was observed.
