"""Command line entry point: `depower run`."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depower", description="German day-ahead price forecasting.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the full offline experiment on data/raw.")
    run.add_argument("--root", type=Path, default=None, help="Project root containing data/raw (default: auto).")
    run.add_argument("--out", type=Path, default=None, help="Output folder (default: data/processed/pipeline).")
    run.add_argument("--quick", action="store_true", help="Small smoke-test run (one test week).")
    run.add_argument("--skip-extended", action="store_true", help="Only audit, EDA and the four-model comparison.")
    run.add_argument("--skip-intervals", action="store_true", help="Skip quantile interval models.")
    run.add_argument("--trials", type=int, default=None, help="Number of sampled LightGBM configurations.")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    from .config import find_root
    from .pipeline import RunConfig, run_pipeline

    if args.command == "run":
        cfg = RunConfig.quick() if args.quick else RunConfig()
        cfg.run_extended = not args.skip_extended
        cfg.run_intervals = cfg.run_intervals and not args.skip_intervals
        if args.trials is not None:
            cfg.n_trials = args.trials
        root = args.root or find_root()
        run_pipeline(root, cfg, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
