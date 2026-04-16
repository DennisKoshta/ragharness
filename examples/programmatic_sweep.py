"""Construct a RagHarnessConfig in Python — no YAML — and run a sweep.

Useful when the config is computed (e.g. sweeping over N model checkpoints
discovered at runtime, or wiring ragharness into a notebook / pipeline).

Usage:
    python examples/programmatic_sweep.py
"""

from __future__ import annotations

from pathlib import Path

from ragharness.config import DatasetConfig, RagHarnessConfig, SystemConfig
from ragharness.orchestrator import run_sweep
from ragharness.reporters import write_csv


def main() -> None:
    dataset_path = Path(__file__).parent / "sample_dataset.jsonl"

    config = RagHarnessConfig(
        dataset=DatasetConfig(source="jsonl", path=str(dataset_path), limit=1),
        system=SystemConfig(
            adapter="raw",
            adapter_config={
                "llm_provider": "anthropic",
                "llm_model": "claude-sonnet-4-20250514",
            },
        ),
        sweep={
            "top_k": [3, 5],
            "temperature": [0.0, 0.3],
        },
        metrics=[
            "exact_match",
            "latency_p50",
            {
                "token_cost": {
                    "pricing": {
                        "claude-sonnet-4-20250514": {
                            "input_per_1k": 0.003,
                            "output_per_1k": 0.015,
                        }
                    }
                }
            },
        ],
    )

    result = run_sweep(config, no_confirm=True)

    out_dir = Path("./results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(result, out_dir)

    print(f"\nWrote {len(result.runs)} run(s) to {csv_path}")
    for run in result.runs:
        label = ", ".join(f"{k}={v}" for k, v in sorted(run.config_params.items()))
        print(f"  {label}: {run.aggregate_scores}")


if __name__ == "__main__":
    main()
