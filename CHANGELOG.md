# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-16

Initial public release.

### Added
- `RAGSystem` protocol and `RAGResult` dataclass for zero-inheritance custom systems.
- `EvalDataset` / `EvalItem` with JSONL, CSV, and HuggingFace loaders (dotted-path field access).
- Pydantic-backed YAML config (`dataset`, `system`, `sweep`, `metrics`, `output`) with validation.
- Sweep orchestrator with Cartesian product expansion, per-question and aggregate metrics, tqdm progress.
- Adapters: `raw` (OpenAI/Anthropic), `langchain`, `llamaindex`, `r2r`, `haystack`.
- Metrics: `exact_match`, `precision_at_k`, `llm_judge`, `latency_p50`, `latency_p95`, `token_cost`.
- Reporters: per-question and summary CSV, matplotlib charts (accuracy bars, latency box plots, cost vs accuracy scatter, per-metric bars).
- CLI: `ragharness run`, `ragharness validate`, `ragharness report`.
- Auth helper with `.env` loading and user-friendly `MissingAPIKeyError`.
- Strict mypy, ruff, and a pytest suite with adapter/metric/reporter/config coverage.

[Unreleased]: https://github.com/DennisKoshta/ragharness/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DennisKoshta/ragharness/releases/tag/v0.1.0
