#!/usr/bin/env python3
"""Compute EX-L1 and EX-Linf results from generated approximate games."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation import generation_kfold_approx_wl1_wlinf

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--model", choices=["lr", "rf"], default="lr")
p.add_argument("--repeat-start", "--fold-start", dest="repeat_start", type=int, default=0)
p.add_argument("--repeat-end", "--fold-end", dest="repeat_end", type=int, default=10)
args = p.parse_args()

generation_kfold_approx_wl1_wlinf(
    ds_name=args.dataset,
    model_string=args.model,
    fold_start=args.repeat_start,
    fold_end=args.repeat_end,
)
