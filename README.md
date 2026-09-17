# llm-router

**Route to the cheapest capable model.** llm-router is a small, dependency-free
Python library that picks the cheapest LLM able to handle each request —
configurable model catalogs (cost, quality, context window, capability flags),
routing policies (`cheapest-first`, `quality-first`, `budget-capped`),
automatic fallback chains on backend failure, and per-request cost/quality
tracking. Ships with a CLI demo.

## Why

Most LLM apps send every request to one flagship model and overpay for easy
work: classification, extraction, rewrites, summaries. Routing each request
to the cheapest model that meets its requirements (quality floor, features,
context, budget) cuts spend with no code changes to prompts — and the ledger
proves the savings.

## Install

```bash
pip install git+https://github.com/anushamukka9/llm-router.git
# or, for local development:
git clone https://github.com/anushamukka9/llm-router.git
cd llm-router
pip install -e ".[dev]"
```

Optional YAML catalog support: `pip install "llm-router[yaml]"` (or add `pyyaml`).

## Quickstart

```python
from llm_router import (
    MockBackend, ModelCatalog, Policy, Router, RoutingConstraints, UsageLedger,
)
from llm_router.defaults import DEFAULT_MODELS

catalog = ModelCatalog.from_dict({"models": DEFAULT_MODELS})
router = Router(catalog)
for spec in catalog:
    router.register_backend(MockBackend(spec))  # swap for OpenAICompatibleBackend

completion, decision, attempted = router.execute(
    "Classify this ticket: 'duplicate charge on my invoice'",
    policy=Policy.CHEAPEST_FIRST,
    constraints=RoutingConstraints(min_quality=55),
)
print(decision.model.name)   # nano-1
print(completion.cost_usd)   # actual tracked cost
```

Run the runnable example: `python examples/quickstart.py`.
See [docs/usage.md](docs/usage.md) for the full guide.

## CLI demo

```bash
llm-router demo
# five requests, each routed under a different policy; ends with a
# cost/quality summary and savings vs. the all-flagship baseline

llm-router route "Summarize this log" --policy cheapest-first --min-quality 60
llm-router route "Proofread this" --policy budget-capped --budget 0.02
llm-router compare "Draft a release note" --catalog catalog.yaml
```

## API

| Symbol | Role |
|---|---|
| `ModelSpec` | Cost, quality, context window, feature flags for one model; cost estimation. |
| `ModelCatalog` | Fleet of specs; load from YAML/JSON; filter by features, quality, cost, context, provider. |
| `ModelBackend` | Pluggable execution interface (`MockBackend` for offline, `OpenAICompatibleBackend` for any OpenAI-compatible endpoint). |
| `Router` | `route()` picks a model under a policy + hard constraints; `execute()` serves it with fallback chains; `compare()` prices every model for a prompt. |
| `Policy` | `cheapest-first`, `quality-first`, `budget-capped`. |
| `RoutingConstraints` | Hard gates: required features, min quality, min context, provider, max cost. |
| `UsageLedger` | Per-request log; summaries: total cost, tokens, avg quality, fallback rate, per-model aggregates, savings vs. baseline. |

## Architecture

```
src/llm_router/
├── __init__.py      # public API
├── models.py        # ModelSpec, backends (Mock, OpenAI-compatible)
├── catalog.py       # ModelCatalog: load/filter/manage the fleet
├── router.py        # Policies, constraints, fallback execution
├── tracking.py      # UsageLedger: cost/quality accounting
├── defaults.py      # demo fleet (illustrative sample pricing)
└── cli.py           # llm-router CLI: route / compare / demo
examples/quickstart.py   # runnable three-request walkthrough
docs/usage.md            # full usage guide
```

**Flow:** prompt → catalog filtering (hard constraints) → policy ranking
(cost or quality) → decision + ordered fallback chain → backend execution
(failures walk the chain) → completion logged to the ledger.

## Tests

```bash
pytest            # 18 tests: routing policies, constraints, fallbacks, ledger, backends
```

## License

MIT — Copyright (c) 2026 Anusha Mukka. See [LICENSE](LICENSE).

Author: Anusha Mukka · [anushamukka.com](https://anushamukka.com)
