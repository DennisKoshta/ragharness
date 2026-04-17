# Writing a custom adapter

An adapter is the bridge between a RAG framework (LangChain, LlamaIndex, R2R, …) or a bespoke stack and rag_eval_kit's evaluation loop. The contract is deliberately tiny: anything with a `query(self, question: str) -> RAGResult` method satisfies the [`RAGSystem`](../rag_eval_kit/protocol.py) protocol. There is no base class, no registration step at import time, and no required `__init__` signature.

This guide walks through adding a first-class adapter (shipped with the package and selectable via `system.adapter: <name>` in YAML). If you only need a one-off for your own codebase, skip ahead to [Ad-hoc custom systems](#ad-hoc-custom-systems).

## 1. Write the adapter class

Create `rag_eval_kit/adapters/<name>.py`. The class takes everything it needs as keyword arguments — the orchestrator merges sweep parameters on top of `adapter_config` before instantiation, so any keyword that appears in the YAML `sweep:` block is available here.

```python
# rag_eval_kit/adapters/myvendor.py
from __future__ import annotations
import time
from typing import Any
from rag_eval_kit.protocol import RAGResult


class MyVendorRAGSystem:
    def __init__(
        self,
        api_url: str,
        top_k: int = 5,
        temperature: float = 0.0,
        **kwargs: Any,  # soak up future sweep params you haven't declared yet
    ) -> None:
        try:
            import myvendor_sdk
        except ImportError:
            raise ImportError(
                "myvendor-sdk required. Install with: pip install rag-eval-kit[myvendor]"
            ) from None

        self.client = myvendor_sdk.Client(api_url)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

    def query(self, question: str) -> RAGResult:
        start = time.perf_counter()
        resp = self.client.ask(question, top_k=self.top_k, temperature=self.temperature)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=resp.answer,
            retrieved_docs=[d.text for d in resp.sources],
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "model": resp.model,
                "top_k": self.top_k,
            },
        )
```

### What to put in `metadata`

The metrics registry reads these keys — populate what you can:

| Key | Type | Consumers |
|---|---|---|
| `latency_ms` | float | `latency_p50`, `latency_p95` (auto-filled by orchestrator if missing) |
| `prompt_tokens` | int | `token_cost` |
| `completion_tokens` | int | `token_cost` |
| `model` | str | `token_cost` (for pricing lookup) |
| `top_k` | int | reporters (appears in detail CSV) |

Anything extra you drop here is passed through to the CSV writer untouched, so it is a fine place to stash adapter-specific debugging info (cache hits, retrieval IDs, etc.).

## 2. Register it in the factory

Edit [rag_eval_kit/adapters/__init__.py](../rag_eval_kit/adapters/__init__.py) and add a branch:

```python
elif adapter_type == "myvendor":
    from rag_eval_kit.adapters.myvendor import MyVendorRAGSystem
    return MyVendorRAGSystem(**merged)
```

Keep the import inside the branch — this keeps `rag-eval-kit` importable when the optional dep is missing.

## 3. Whitelist the name in the config validator

Open [rag_eval_kit/config.py](../rag_eval_kit/config.py) and add `"myvendor"` to the adapter allowlist in `SystemConfig`. Configs with unknown adapter names fail `rag-eval-kit validate` with a clear error.

## 4. Declare the optional dependency

In [pyproject.toml](../pyproject.toml):

```toml
[project.optional-dependencies]
myvendor = ["myvendor-sdk>=1.2"]
```

And add `myvendor` to the `all` extra so `pip install rag-eval-kit[all]` pulls it in.

## 5. Add tests

`tests/test_adapters/test_myvendor.py`. Mock the SDK — don't hit the network:

```python
from unittest.mock import MagicMock
from rag_eval_kit.adapters.myvendor import MyVendorRAGSystem
from rag_eval_kit.protocol import RAGSystem


def test_protocol_conformance():
    adapter = MyVendorRAGSystem(api_url="http://x")
    assert isinstance(adapter, RAGSystem)


def test_query_populates_metadata():
    adapter = MyVendorRAGSystem(api_url="http://x")
    adapter.client = MagicMock()
    adapter.client.ask.return_value = ...  # fake response
    result = adapter.query("hi")
    assert result.metadata["model"] == "expected-model"
```

## 6. Ship examples

- `examples/myvendor_config.yaml` — minimal YAML config
- `examples/myvendor_python.py` — equivalent programmatic invocation

## Thread-safety when `concurrency > 1`

When the sweep runs with `concurrency > 1` (either via `concurrency:` in YAML or `--concurrency N` on the CLI), multiple threads call `query()` on the same adapter instance concurrently. The adapter must be thread-safe.

The bundled adapters are all safe:

- `raw` and `r2r` build their SDK clients lazily behind a `threading.Lock` (double-checked init in `_get_client`).
- `langchain`, `llamaindex`, and `haystack` construct their clients eagerly in `__init__`, before any thread can call `query()`.
- The OpenAI, Anthropic, R2R, LangChain, LlamaIndex, and Haystack Python SDKs themselves document thread-safe client use.

If you're writing a custom adapter that will run at `concurrency > 1`:

1. **Avoid mutable shared state in `query()`.** Read-only `self.foo` attributes set in `__init__` are fine. Don't mutate `self.<something>` from inside `query()`.
2. **If you lazy-init an SDK client, guard it with a lock** (same pattern as `raw.py`):

   ```python
   import threading

   class MyAdapter:
       def __init__(self, ...):
           self._client = None
           self._client_lock = threading.Lock()

       def _get_client(self):
           if self._client is not None:
               return self._client
           with self._client_lock:
               if self._client is not None:
                   return self._client
               self._client = ExpensiveSDK.Client(...)
               return self._client
   ```

3. **Check your SDK's thread-safety docs.** Most modern HTTP-based LLM SDKs are fine, but some have connection-pool caveats worth reading.

Adapters used only at `concurrency == 1` (the default) need no special handling.

## Ad-hoc custom systems

If you don't need a PR to this repo, skip steps 2–6. Any class with a `query` method works with the orchestrator's Python API:

```python
from rag_eval_kit import RAGResult
from rag_eval_kit.config import load_config
from rag_eval_kit.orchestrator import run_sweep

class MySystem:
    def query(self, question):
        ...
        return RAGResult(answer="...", retrieved_docs=[...], metadata={...})

# Instead of loading a YAML, build the RagEvalKitConfig in Python and
# supply an adapter factory via the 'raw' adapter's retriever argument,
# or call the metric functions directly on your own loop. See
# examples/programmatic_sweep.py and examples/custom_rag_system.py.
```
