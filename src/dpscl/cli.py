"""Command-line interface for DP-SCL experiments."""

import argparse

from .config import SEED_LIST


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run DP-SCL experiments with one or more seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-indir", type=str, default=".", help="input directory")
    parser.add_argument("-outdir", type=str, default=".", help="output directory")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEED_LIST, help="random seeds to run")
    parser.add_argument("--split", nargs=3, type=float, default=[0.60, 0.10, 0.30], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lambda-con", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from .experiment import run_experiment

    run_experiment(args)
