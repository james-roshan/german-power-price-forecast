"""CLI wiring (offline, mocked) and an opt-in end-to-end smoke run on the real data files."""
import os
from pathlib import Path

import pytest

from depower import pipeline
from depower.cli import main
from depower.config import find_root


def test_cli_run_builds_the_requested_config(monkeypatch, tmp_path):
    seen = {}

    def fake_run(root, cfg, out):
        seen.update(root=root, cfg=cfg, out=out)

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    code = main(["run", "--root", str(tmp_path), "--out", str(tmp_path / "o"), "--quick", "--skip-intervals",
                 "--skip-extended", "--trials", "3"])
    assert code == 0 and seen["root"] == tmp_path and seen["out"] == tmp_path / "o"
    cfg = seen["cfg"]
    assert not cfg.run_extended and not cfg.run_intervals and cfg.n_trials == 3
    assert cfg.test_end == "2025-10-01"  # --quick shortens the test period


def test_default_config_is_the_full_experiment():
    cfg = pipeline.RunConfig()
    assert (cfg.test_start, cfg.test_end) == ("2025-09-25", "2026-09-24") and cfg.n_trials == 7
    assert len(cfg.fold_starts) == 4 and cfg.run_extended and cfg.include_competitors


def test_runner_requires_input_files(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing input files"):
        pipeline.Runner(tmp_path, out_dir=tmp_path / "out")


def _real_root():
    try:
        root = find_root(Path(__file__).parent)
    except FileNotFoundError:
        return None
    return root if (root / "data/raw/weather_forecast.parquet").exists() else None


@pytest.mark.skipif(_real_root() is None or not os.environ.get("DEPOWER_SMOKE"),
                    reason="set DEPOWER_SMOKE=1 and provide data/raw to run the end-to-end smoke test")
def test_quick_pipeline_end_to_end(tmp_path):
    import matplotlib
    matplotlib.use("Agg")  # Headless execution on CI and minimal Windows Python installs.
    res = pipeline.run_pipeline(_real_root(), pipeline.RunConfig.quick(), tmp_path, verbose=False)
    assert (res.checks.result == "PASS").all() and len(res.checks) >= 10
    assert len(res.predictions) == 7 * 24
    for name in ["model_results", "cv_leaderboard", "extended_model_results", "interval_summary",
                 "validation_checks"]:
        assert (tmp_path / f"{name}.csv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 18
