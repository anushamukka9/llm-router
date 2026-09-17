"""Quickstart: route three requests with different policies and print decisions."""

from llm_router import (
    MockBackend,
    ModelCatalog,
    Policy,
    Router,
    RoutingConstraints,
    UsageLedger,
)
from llm_router.defaults import DEFAULT_MODELS


def main() -> None:
    catalog = ModelCatalog.from_dict({"models": DEFAULT_MODELS})
    router = Router(catalog)
    for spec in catalog:
        router.register_backend(MockBackend(spec))

    ledger = UsageLedger()

    jobs = [
        ("Classify this ticket: 'My invoice shows a duplicate charge'",
         Policy.CHEAPEST_FIRST, RoutingConstraints(min_quality=55)),
        ("Write a launch announcement for our new API gateway",
         Policy.CHEAPEST_FIRST, RoutingConstraints(min_quality=70)),
        ("Prove the CAP theorem informally for a systems design doc",
         Policy.QUALITY_FIRST, RoutingConstraints(min_quality=85)),
    ]

    for prompt, policy, constraints in jobs:
        completion, decision, attempted = router.execute(
            prompt, policy=policy, constraints=constraints
        )
        ledger.log(
            model=decision.model.name,
            policy=policy.value,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            quality=decision.model.quality,
            attempted=attempted,
        )
        print(f"prompt : {prompt[:60]}")
        print(f"  -> {decision.model.name:12} quality={decision.model.quality:<5} "
              f"cost=${completion.cost_usd:.6f} ({policy.value}: {decision.reason})")

    print("\nsession:", ledger.summary())


if __name__ == "__main__":
    main()
