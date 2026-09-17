# Usage guide

This guide walks through everyday use of **llm-router**: defining your model
fleet, choosing a routing policy, serving requests, and reading the cost
ledger.

## 1. Describe your fleet

A catalog is a list of models with cost, quality, context window, and
capability flags. Keep one in YAML (or JSON) next to your app:

```yaml
models:
  - name: nano-1
    provider: acme
    cost_in_per_1k: 0.02
    cost_out_per_1k: 0.06
    quality: 62.0
    context_window: 16384
    features: [json, tools]
  - name: pro-1
    provider: acme
    cost_in_per_1k: 1.25
    cost_out_per_1k: 5.00
    quality: 91.0
    context_window: 131072
    features: [json, tools, vision]
```

> Prices in the bundled demo fleet are illustrative samples. Fill in your own
> numbers from provider pricing pages before using cost figures for real
> spend decisions.

Quality scores are yours to define: run your own evals (accuracy, win-rate,
human preference) and record the results. The router treats them as an
ordering, so they only need to be consistent across your fleet.

## 2. Pick a policy

| Policy | Behaviour |
|---|---|
| `cheapest-first` | Lowest estimated cost among models meeting all constraints. |
| `quality-first` | Highest quality score among eligible models. |
| `budget-capped` | Highest quality whose estimated cost stays under `max_cost_usd`. |

## 3. Set hard constraints

`RoutingConstraints` are non-negotiable: a model that violates any of them
is never chosen.

- `required_features` — e.g. `("vision",)` for image inputs, `("tools",)` for function calling.
- `min_quality` — floor on your quality score.
- `min_context` — prompts longer than a model's window exclude it.
- `max_cost_usd` — ceiling on estimated input cost; required for `budget-capped`.
- `provider` — restrict to one provider.

```python
from llm_router import ModelCatalog, Policy, Router, RoutingConstraints, MockBackend

catalog = ModelCatalog.from_file("catalog.yaml")
router = Router(catalog)
for spec in catalog:
    router.register_backend(MockBackend(spec))  # swap for OpenAICompatibleBackend in prod

constraints = RoutingConstraints(required_features=("vision",), min_quality=65)
completion, decision, attempted = router.execute(
    "Describe this chart.", policy=Policy.CHEAPEST_FIRST, constraints=constraints
)
print(decision.model.name, f"${completion.cost_usd:.6f}")
```

## 4. Fallbacks

Every decision carries an ordered fallback chain (next-best models under the
same policy). `execute()` walks the chain when a backend raises
`BackendError` — a flaky or down model never takes the request down with it.
`attempted` tells you which models were tried, and the ledger records it.

## 5. Read the ledger

```python
from llm_router import UsageLedger

ledger = UsageLedger()
# after each request:
ledger.log(model=decision.model.name, policy=decision.policy.value,
           input_tokens=completion.input_tokens,
           output_tokens=completion.output_tokens,
           cost_usd=completion.cost_usd,
           quality=decision.model.quality, attempted=attempted)

print(ledger.summary())
# {'total_requests': N, 'total_cost_usd': ..., 'total_tokens': ...,
#  'avg_quality': ..., 'fallback_rate': ..., 'per_model': {...}}
```

`savings_vs(baseline_cost_per_request)` answers the question leadership asks:
"how much did routing save versus sending everything to the flagship model?"

## 6. CLI

```bash
llm-router route "Summarize this log line" --policy cheapest-first --min-quality 60
llm-router route "Proofread this" --policy budget-capped --budget 0.02
llm-router compare "Draft a release note"   # price every model for one prompt
llm-router demo                             # five-request demo with summary
llm-router route "..." --catalog catalog.yaml --require vision
```

## 7. Using real models

`OpenAICompatibleBackend` speaks the OpenAI `/v1/chat/completions` wire
format, so it works against OpenAI, Azure OpenAI, Ollama, vLLM, and any
other compatible endpoint:

```python
from llm_router import OpenAICompatibleBackend

router.register_backend(OpenAICompatibleBackend(
    catalog.get("pro-1"),
    base_url="https://api.openai.com/v1",
    api_key="sk-...",          # or read from env in your own code
))
```
