#!/usr/bin/env python3
"""Run the approximate-regime experiments for one dataset/model pair."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation import generation_kfold_approx

APPROX_DATASETS = [
    "sonar", "spambase", "chess", "valley_without_noise", "valley_with_noise",
    "parkinson", "hepatitis", "nursery", "dermatology", "satimage", "mushrooms",
    "splice", "robot", "qsar", "indian_patient", "image", "tic_tac_toe", "libras",
    "gas_sensor", "musk1", "musk2", "coil", "myocardial", "optical", "cancer",
    "darwin", "gait", "naticus", "scene", "drug", "voice"
]

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=APPROX_DATASETS, required=True)
p.add_argument("--model", choices=["lr", "rf"], default="lr")
p.add_argument("--repeat-start", "--fold-start", dest="repeat_start", type=int, default=0,
               help="First repeated-holdout repetition to run (inclusive).")
p.add_argument("--repeat-end", "--fold-end", dest="repeat_end", type=int, default=10,
               help="Last repeated-holdout repetition to run (exclusive).")
args = p.parse_args()

generation_kfold_approx(
    ds_name=args.dataset,
    model_string=args.model,
    fold_start=args.repeat_start,
    fold_end=args.repeat_end,
)
