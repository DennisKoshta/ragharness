"""JSONL checkpoint writer/loader for resumable sweeps.

Each row captures a completed ``(config_idx, item_idx)`` pair with its
reconstructed :class:`RAGResult` fields plus per-question scores. The
format is append-only and flushed per-write so a killed process leaves
a usable file behind.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from rag_eval_kit.protocol import RAGResult

logger = logging.getLogger(__name__)

CheckpointMap = dict[tuple[int, int], dict[str, Any]]


class CheckpointWriter:
    """Thread-safe append-only JSONL writer for sweep checkpoints."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def write(
        self,
        *,
        config_idx: int,
        item_idx: int,
        config_params: dict[str, Any],
        result: RAGResult,
        scores: dict[str, float],
    ) -> None:
        row = {
            "config_idx": config_idx,
            "item_idx": item_idx,
            "config_params": config_params,
            "answer": result.answer,
            "retrieved_docs": result.retrieved_docs,
            "metadata": result.metadata,
            "scores": scores,
        }
        line = json.dumps(row, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> CheckpointWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_checkpoint(path: str | Path) -> CheckpointMap:
    """Load a checkpoint file into a ``(config_idx, item_idx) -> row`` map.

    Returns an empty map if the file is missing. Malformed lines are
    logged and skipped so a partial / corrupt tail does not doom the
    resume.
    """
    p = Path(path)
    if not p.exists():
        return {}

    out: CheckpointMap = {}
    with p.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed checkpoint line %d: %s", line_num, e)
                continue
            try:
                key = (int(row["config_idx"]), int(row["item_idx"]))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    "Skipping checkpoint row with bad index on line %d: %s", line_num, e
                )
                continue
            out[key] = row
    return out


def row_to_result(row: dict[str, Any]) -> RAGResult:
    """Reconstruct a :class:`RAGResult` from a checkpoint row."""
    return RAGResult(
        answer=str(row.get("answer", "")),
        retrieved_docs=list(row.get("retrieved_docs", []) or []),
        metadata=dict(row.get("metadata", {}) or {}),
    )
