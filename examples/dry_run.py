"""Dry-run routing: preview decisions without executing anything.

Run from the repo root::

    pip install -e ".[dev]"
    python examples/dry_run.py

Shows the balanced cost-vs-quality policy next to cheapest-first and
quality-first on the same prompt, then persists a small ledger to JSONL
and loads it back.
"""

from llm_router import MockBackend, ModelCatalog, Policy, Router, RoutingConstraints, UsageLedger
from llm_router.defaults import DEFAULT_MODELS


def main() -> None:
    catalog = ModelCatalog.from_dict({"models": DEFAULT_MODELS})
    router = Router(catalog)
    for spec in catalog:
        router.register_backend(MockBackend(spec))

    prompt = "Draft a two-paragraph launch note for the new search API"
    constraints = RoutingConstraints(min_quality=60)

    for policy in (Policy.CHEAPEST_FIRST, Policy.BALANCED, Policy.QUALITY_FIRST):
        plan = router.dry_run(prompt, policy=policy, constraints=constraints)
        print(f"[{policy.value}]")
        print(f"  chosen : {plan['chosen_model']} "
              f"(est. ${plan['estimated_cost_usd']:.6f})")
        print(f"  reason : {plan['reason']}")
        print(f"  fallbacks: {', '.join(plan['fallbacks']) or 'none'}")

    # Ledger persistence round-trip.
    ledger = UsageLedger()
    ledger.log(model="flash-2", policy="balanced", input_tokens=14,
               output_tokens=120, cost_usd=0.0374, quality=78.0)
    ledger.save_jsonl("/tmp/dry-run-ledger.jsonl")
    restored = UsageLedger.load_jsonl("/tmp/dry-run-ledger.jsonl")
    print("\nrestored ledger:", restored.summary())


if __name__ == "__main__":
    main()
