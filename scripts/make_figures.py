#!/usr/bin/env python3
"""Create paper-facing SVG summaries from regenerated result tables."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def authentic_chart(rows: list[dict[str, str]]) -> str:
    values = {row["condition"]: int(row["correct"]) / int(row["n"])
              for row in rows if row["experiment"] == "authentic_repository_evaluation"}
    width, height = 720, 300
    bars = []
    colors = {"CLEAN": "#3569a8", "PRESERVE": "#6f91bd", "DISRUPT": "#c16a52"}
    for index, condition in enumerate(("CLEAN", "PRESERVE", "DISRUPT")):
        x, bar_width = 130 + index * 185, 90
        y, bar_height = 220 - values[condition] * 155, values[condition] * 155
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[condition]}"/>')
        bars.append(f'<text x="{x + bar_width/2}" y="{y - 10:.1f}" text-anchor="middle">{round(values[condition]*32)}/32</text>')
        bars.append(f'<text x="{x + bar_width/2}" y="245" text-anchor="middle">{condition}</text>')
    return _svg(width, height, "Authentic repository evaluation", "Accuracy", "".join(bars),
                '<line x1="90" y1="220" x2="660" y2="220" stroke="#78818c"/>')


def effects_chart(rows: list[dict[str, str]]) -> str:
    rows = sorted(rows, key=lambda r: (r["model"], r["interface"], r["condition"]))
    width, height = 920, 510
    x0, x1 = 400, 850
    xmin, xmax = -40.0, 10.0

    def x(value: float) -> float:
        return x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)

    shapes = [f'<line x1="{x(0):.1f}" y1="55" x2="{x(0):.1f}" y2="460" stroke="#737b86"/>']
    for index, row in enumerate(rows):
        y = 76 + index * 32
        low, high = map(float, row["bootstrap_ci_pp"].strip("[]").split(","))
        effect = float(row["effect_pp"])
        label = f'{row["model"]}  {row["interface"]}  {row["condition"]}'
        shapes.append(f'<text x="22" y="{y+4}" class="label">{label}</text>')
        shapes.append(f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="#3569a8" stroke-width="3"/>')
        shapes.append(f'<circle cx="{x(effect):.1f}" cy="{y}" r="4.5" fill="#c16a52"/>')
    ticks = []
    for tick in (-40, -30, -20, -10, 0, 10):
        ticks.append(f'<line x1="{x(tick):.1f}" y1="460" x2="{x(tick):.1f}" y2="466" stroke="#737b86"/>')
        ticks.append(f'<text x="{x(tick):.1f}" y="485" text-anchor="middle">{tick}</text>')
    return _svg(width, height, "180-repository procedural-minus-neutral effects", "Effect (percentage points)",
                "".join(shapes + ticks), '<line x1="400" y1="460" x2="850" y2="460" stroke="#737b86"/>',
                extra='<style>.label{font-size:13px}</style>')


def _svg(width: int, height: int, title: str, axis: str, body: str, extras: str = "", extra: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="#fff"/>'
            f'<text x="24" y="34" font-family="Arial,sans-serif" font-size="20" font-weight="600">{title}</text>'
            f'<text x="24" y="72" font-family="Arial,sans-serif" font-size="13" fill="#555">{axis}</text>'
            f'<g font-family="Arial,sans-serif" font-size="13" fill="#24303d">{extras}{body}</g>{extra}</svg>')


def main() -> None:
    counts = read_csv(ROOT / "results" / "arm_counts.csv")
    effects = read_csv(ROOT / "results" / "large_repository_results.csv")
    output = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    (output / "authentic_accuracy.svg").write_text(authentic_chart(counts), encoding="utf-8")
    (output / "large_repository_effects.svg").write_text(effects_chart(effects), encoding="utf-8")
    print(f"wrote 2 figures to {output}")


if __name__ == "__main__":
    main()
