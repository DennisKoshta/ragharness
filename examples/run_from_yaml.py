"""Run a sweep from a YAML config — the README quickstart as a runnable script.

Equivalent to:
    rag_eval_kit run examples/basic_config.yaml --no-confirm

Usage:
    python examples/run_from_yaml.py
"""

from __future__ import annotations

from pathlib import Path

from rag_eval_kit.config import load_config
from rag_eval_kit.orchestrator import run_sweep
from rag_eval_kit.reporters import write_charts, write_csv


def main() -> None:
    config_path = Path(__file__).parent / "basic_config.yaml"
    config = load_config(config_path)

    result = run_sweep(config, no_confirm=True)

    out_dir = Path("./results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(result, out_dir)
    write_charts(result, out_dir / "charts")

    print(f"\nWrote {len(result.runs)} run(s) to {csv_path}")
    for run in result.runs:
        label = ", ".join(f"{k}={v}" for k, v in sorted(run.config_params.items())) or "(baseline)"
        print(f"  {label}: {run.aggregate_scores}")


if __name__ == "__main__":
    main()
