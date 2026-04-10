from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalItem:
    """A single evaluation question with expected answer and optional metadata."""

    question: str
    expected_answer: str
    expected_docs: list[str] | None = None
    tags: dict | None = None


class EvalDataset:
    """Loads and iterates over evaluation items from JSONL or CSV files."""

    def __init__(self, items: list[EvalItem]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[EvalItem]:
        return iter(self._items)

    def __getitem__(self, idx: int) -> EvalItem:
        return self._items[idx]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalDataset:
        """Load evaluation items from a JSONL file.

        Each line must be a JSON object with at least ``question`` and
        ``expected_answer`` keys.  Optional keys: ``expected_docs`` (list
        of strings) and ``tags`` (dict).
        """
        path = Path(path)
        items: list[EvalItem] = []
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {path}: {e}"
                    ) from e
                items.append(
                    EvalItem(
                        question=obj["question"],
                        expected_answer=obj["expected_answer"],
                        expected_docs=obj.get("expected_docs"),
                        tags=obj.get("tags"),
                    )
                )
        return cls(items)

    @classmethod
    def from_csv(cls, path: str | Path) -> EvalDataset:
        """Load evaluation items from a CSV file.

        Required columns: ``question``, ``expected_answer``.
        Optional columns: ``expected_docs`` (JSON-encoded list),
        ``tags`` (JSON-encoded dict).
        """
        path = Path(path)
        items: list[EvalItem] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                expected_docs = None
                if row.get("expected_docs"):
                    expected_docs = json.loads(row["expected_docs"])
                tags = None
                if row.get("tags"):
                    tags = json.loads(row["tags"])
                items.append(
                    EvalItem(
                        question=row["question"],
                        expected_answer=row["expected_answer"],
                        expected_docs=expected_docs,
                        tags=tags,
                    )
                )
        return cls(items)

    @classmethod
    def from_huggingface(
        cls, name: str, split: str = "validation", limit: int | None = None
    ) -> EvalDataset:
        """Load from a HuggingFace dataset. Not yet implemented."""
        raise NotImplementedError(
            "HuggingFace dataset loading is not yet implemented. "
            "Use from_jsonl() or from_csv() instead."
        )
