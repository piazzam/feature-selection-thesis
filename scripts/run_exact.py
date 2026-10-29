#!/usr/bin/env python3
"""Run the exact-regime experiments for one dataset/model pair."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation import generation_kfold

EXACT_DATASETS = [
    "iris", "banknote", "monks1", "monks3", "raisin", "breastcancer",
    "magic", "australian", "seismic", "heart"
]

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=EXACT_DATASETS, required=True)
p.add_argument("--model", choices=["lr", "rf"], default="lr")
args = p.parse_args()

generation_kfold(ds_name=args.dataset, model_string=args.model, do_exact=True)
