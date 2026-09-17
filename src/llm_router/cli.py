"""Command-line interface for llm-router."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog import ModelCatalog
from .defaults import DEFAULT_MODELS
from .models import MockBackend
from .router import NoEligibleModelError, Policy, Router, RoutingConstraints
from .tracking import UsageLedger


def _parse_policy(value: str) -> Policy:
    try:
        return Policy(value)
    except ValueError:
        valid = ", ".join(p.value for p in Policy)
        raise argparse.ArgumentTypeError(f"unknown policy {value!r}; choose from: {valid}")


def _load_catalog(path: str | None) -> ModelCatalog:
    if path:
        return ModelCatalog.from_file(path)
    return ModelCatalog.from_dict({"models": DEFAULT_MODELS})


def _build_router(catalog: ModelCatalog) -> Router:
    router = Router(catalog)
    for spec in catalog:
        router.register_backend(MockBackend(spec))
    return router


def _print_decision(decision, completion=None) -> None:
    spec = decision.model
    print(f"policy        : {decision.policy.value}")
    print(f"chosen model  : {spec.name} (provider={spec.provider}, quality={spec.quality})")
    print(f"reason        : {decision.reason}")
    print(f"est. cost     : ${decision.estimated_cost_usd:.6f}")
    if decision.fallbacks:
        print("fallbacks     : " + ", ".join(s.name for s in decision.fallbacks))
    if completion is not None:
        print(f"actual cost   : ${completion.cost_usd:.6f} "
              f"({completion.input_tokens} in / {completion.output_tokens} out tokens)")
        print(f"response      : {completion.text}")


def _cmd_route(args) -> int:
    catalog = _load_catalog(args.catalog)
    router = _build_router(catalog)
    constraints = RoutingConstraints(
        required_features=tuple(args.require or []),
        min_quality=args.min_quality,
        min_context=args.min_context,
        max_cost_usd=args.budget,
    )
    try:
        completion, decision, attempted = router.execute(
            args.prompt,
            policy=args.policy,
            constraints=constraints,
            max_tokens=args.max_tokens,
        )
    except (NoEligibleModelError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_decision(decision, completion)
    if len(attempted) > 1:
        print(f"note: fallbacks used, attempts: {' -> '.join(attempted)}")
    return 0


def _cmd_compare(args) -> int:
    catalog = _load_catalog(args.catalog)
    router = _build_router(catalog)
    rows = router.compare(args.prompt, max_tokens=args.max_tokens)
    print(f"{'MODEL':<14}{'PROVIDER':<10}{'QUALITY':>8}{'CONTEXT':>10}{'EST.COST':>12}  FEATURES")
    for r in rows:
        print(
            f"{r['model']:<14}{r['provider']:<10}{r['quality']:>8.1f}"
            f"{r['context']:>10}${r['est_cost_usd']:>11.6f}  {', '.join(r['features'])}"
        )
    return 0


_DEMO_PROMPTS = [
    ("Summarize this error log in one line: ConnectionTimeout after 30s on db-replica-2",
     {"policy": "cheapest-first", "min_quality": 55}),
    ("Draft a customer apology email for a 2-hour outage, warm but professional.",
     {"policy": "cheapest-first", "min_quality": 70}),
    ("Prove that every even integer greater than 2 is the sum of two primes.",
     {"policy": "quality-first", "min_quality": 85}),
    ("Describe the attached architecture diagram for the incident report.",
     {"policy": "cheapest-first", "min_quality": 60, "require": ["vision"]}),
    ("Summarize this 400k-token codebase for the onboarding doc.",
     {"policy": "budget-capped", "budget": 1.50, "min_quality": 60,
      "require": ["long-context"]}),
]


def _cmd_demo(args) -> int:
    catalog = _load_catalog(args.catalog)
    router = _build_router(catalog)
    ledger = UsageLedger()
    print("=== llm-router demo: one router, five requests ===\n")
    for i, (prompt, opts) in enumerate(_DEMO_PROMPTS, 1):
        constraints = RoutingConstraints(
            required_features=tuple(opts.get("require", [])),
            min_quality=opts.get("min_quality", 0.0),
            max_cost_usd=opts.get("budget"),
        )
        policy = _parse_policy(opts.get("policy", "cheapest-first"))
        try:
            completion, decision, attempted = router.execute(
                prompt, policy=policy, constraints=constraints
            )
        except (NoEligibleModelError, ValueError) as exc:
            print(f"[{i}] {prompt[:60]}...\n    skipped: {exc}\n")
            continue
        ledger.log(
            model=decision.model.name,
            policy=policy.value,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            quality=decision.model.quality,
            attempted=attempted,
        )
        print(f"[{i}] {prompt[:72]}")
        print(f"    -> {decision.model.name} "
              f"(quality {decision.model.quality}, ${completion.cost_usd:.6f}) "
              f"[{policy.value}: {decision.reason}]\n")

    s = ledger.summary()
    print("=== session summary ===")
    print(f"requests : {s['total_requests']}")
    print(f"spend    : ${s['total_cost_usd']:.6f} for {s['total_tokens']} tokens")
    print(f"avg qual : {s['avg_quality']}")
    flagship = max(catalog, key=lambda m: m.quality)
    baseline = flagship.estimate_cost(400, 512)
    saved = ledger.savings_vs(baseline)
    print(f"vs all-{flagship.name}: saved ${saved:.4f} "
          f"({saved / (baseline * s['total_requests']) * 100:.1f}% cheaper)")
    print("per model:")
    for name, agg in sorted(s["per_model"].items(), key=lambda kv: -kv[1]["requests"]):
        print(f"  {name:<12} {agg['requests']:>2} req  ${agg['cost_usd']:.6f}  "
              f"avg quality {agg['avg_quality']:.1f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="Route each request to the cheapest model capable of handling it.",
    )
    parser.add_argument("--catalog", default=None,
                        help="path to a YAML/JSON model catalog (default: built-in demo fleet)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route", help="route and serve a single prompt")
    p_route.add_argument("prompt", help="the user prompt to serve")
    p_route.add_argument("--policy", type=_parse_policy, default=Policy.CHEAPEST_FIRST,
                         help="cheapest-first | quality-first | budget-capped")
    p_route.add_argument("--min-quality", type=float, default=0.0)
    p_route.add_argument("--min-context", type=int, default=0)
    p_route.add_argument("--budget", type=float, default=None,
                         help="max USD estimated input cost (required for budget-capped)")
    p_route.add_argument("--require", action="append", default=[],
                         help="required feature flag (repeatable)")
    p_route.add_argument("--max-tokens", type=int, default=512)
    p_route.set_defaults(func=_cmd_route)

    p_cmp = sub.add_parser("compare", help="rank every model by estimated cost for a prompt")
    p_cmp.add_argument("prompt", help="the user prompt to price out")
    p_cmp.add_argument("--max-tokens", type=int, default=512)
    p_cmp.set_defaults(func=_cmd_compare)

    p_demo = sub.add_parser("demo", help="run the five-request routing demo with a summary")
    p_demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
