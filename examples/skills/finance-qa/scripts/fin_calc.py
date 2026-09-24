#!/usr/bin/env python3
"""Deterministic financial arithmetic for the finance-qa skill.

Language models are unreliable at multi-step arithmetic, so the skill sends every number it
reports through this script. Standard library only; prints Markdown tables.

Usage:
  fin_calc.py growth 2021=100 2022=120 2023=150        year-over-year growth and CAGR
  fin_calc.py cagr START END YEARS                      compound annual growth rate
  fin_calc.py margins revenue=500 gross_profit=200 operating_income=80 net_income=60
  fin_calc.py valuation price=30 eps=2 bvps=12 dps=0.6  P/E, P/B, dividend yield, payout
"""

from __future__ import annotations

import sys


def _num(text: str) -> float:
    return float(text.replace(",", "").replace("_", ""))


def _pairs(args: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for arg in args:
        key, sep, raw = arg.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {arg!r}")
        values[key.strip()] = _num(raw)
    return values


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def cagr(start: float, end: float, years: float) -> float | None:
    if start <= 0 or end <= 0 or years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def growth(args: list[str]) -> str:
    series = sorted(_pairs(args).items(), key=lambda item: item[0])
    if len(series) < 2:
        raise SystemExit("growth needs at least two period=value pairs")
    lines = ["| Period | Value | YoY |", "|---|---:|---:|"]
    previous = None
    for period, value in series:
        yoy = None if previous is None else _ratio(value - previous, abs(previous))
        lines.append(f"| {period} | {value:,.2f} | {'-' if previous is None else _pct(yoy)} |")
        previous = value
    years = len(series) - 1
    rate = cagr(series[0][1], series[-1][1], years)
    lines.append("")
    lines.append(f"CAGR over {years} period(s) ({series[0][0]} to {series[-1][0]}): {_pct(rate)}")
    if rate is None:
        lines.append("(CAGR is undefined when the start or end value is zero or negative.)")
    return "\n".join(lines)


def margins(args: list[str]) -> str:
    values = _pairs(args)
    revenue = values.pop("revenue", None)
    if not revenue:
        raise SystemExit("margins needs revenue=... and at least one profit line")
    lines = ["| Line | Amount | Margin |", "|---|---:|---:|"]
    lines.append(f"| revenue | {revenue:,.2f} | 100.00% |")
    for name, amount in values.items():
        lines.append(f"| {name} | {amount:,.2f} | {_pct(_ratio(amount, revenue))} |")
    return "\n".join(lines)


def valuation(args: list[str]) -> str:
    v = _pairs(args)
    price = v.get("price")
    if price is None:
        raise SystemExit("valuation needs price=...")
    rows = []
    if "eps" in v:
        pe = _ratio(price, v["eps"])
        rows.append(("P/E", "n/a (EPS <= 0)" if pe is None or v["eps"] <= 0 else f"{pe:.2f}x"))
    if "bvps" in v:
        pb = _ratio(price, v["bvps"])
        rows.append(("P/B", "n/a" if pb is None else f"{pb:.2f}x"))
    if "dps" in v:
        rows.append(("Dividend yield", _pct(_ratio(v["dps"], price))))
        if "eps" in v and v["eps"] > 0:
            rows.append(("Payout ratio", _pct(_ratio(v["dps"], v["eps"]))))
    if not rows:
        raise SystemExit("valuation needs eps=, bvps=, or dps= alongside price=")
    return "\n".join(["| Metric | Value |", "|---|---:|", *(f"| {k} | {x} |" for k, x in rows)])


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    command, args = argv[1], argv[2:]
    if command == "growth":
        print(growth(args))
    elif command == "cagr":
        if len(args) != 3:
            raise SystemExit("cagr needs START END YEARS")
        print(f"CAGR: {_pct(cagr(_num(args[0]), _num(args[1]), _num(args[2])))}")
    elif command == "margins":
        print(margins(args))
    elif command == "valuation":
        print(valuation(args))
    else:
        raise SystemExit(f"unknown command {command!r}; run with --help")


if __name__ == "__main__":
    main(sys.argv)
