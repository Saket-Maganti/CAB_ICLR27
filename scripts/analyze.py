#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cab.analysis import analyze

parser = argparse.ArgumentParser(description="Recompute offline manuscript analyses from compact released outcomes.")
parser.add_argument("--output", type=Path, help="output directory (defaults to this repository's results/)")
args = parser.parse_args()
report = analyze(args.output)
print(f"{report['status']}: {sum(x['status']=='PASS' for x in report['checks'])}/{len(report['checks'])} release verification checks passed")
for item in report['checks']:
    if item['status'] != 'PASS':
        print(f"{item['status']}: {item['name']}")
raise SystemExit(1 if report['status'] == 'FAIL' else 0)
