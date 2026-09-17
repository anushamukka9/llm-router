"""Tests for llm-router."""

import pytest

from llm_router import (
    MockBackend,
    ModelCatalog,
    ModelSpec,
    NoEligibleModelError,
    OpenAICompatibleBackend,
    Policy,
    Router,
    RoutingConstraints,
    UsageLedger,
    BackendError,
)
from llm_router.defaults import DEFAULT_MODELS


def make_catalog():
    return ModelCatalog.from_dict({"models": DEFAULT_MODELS})


def make_router(catalog=None):
    catalog = catalog or make_catalog()
    router = Router(catalog)
    for spec in catalog:
        router.register_backend(MockBackend(spec))
    return router


# ---- ModelSpec --------------------------------------------------------


def test_spec_supports_features():
    spec = ModelSpec(
        name="m", provider="p", cost_in_per_1k=1.0, cost_out_per_1k=2.0,
        quality=80.0, context_window=8192, features=frozenset({"json", "vision"}),
    )
    assert spec.supports(["json"])
    assert not spec.supports(["tools"])


def test_spec_cost_estimate():
    spec = ModelSpec(
        name="m", provider="p", cost_in_per_1k=1.0, cost_out_per_1k=4.0,
        quality=80.0, context_window=8192,
    )
    # 2000 in + 1000 out tokens: 2*1.0 + 1*4.0 = 6.0
    assert spec.estimate_cost(2000, 1000) == pytest.approx(6.0)


# ---- catalog ----------------------------------------------------------


def test_catalog_load_and_roundtrip():
    catalog = make_catalog()
    assert len(catalog) == len(DEFAULT_MODELS)
    assert catalog.get("pro-1").quality == 91.0
    with pytest.raises(Exception):
        catalog.add(catalog.get("pro-1"))  # duplicate name rejected
    with pytest.raises(Exception):
        catalog.get("nope")  # unknown name rejected


def test_catalog_feature_filtering():
    catalog = make_catalog()
    vision_models = {s.name for s in catalog.candidates(required_features=["vision"])}
    assert vision_models == {"pro-1", "vision-lite"}


def test_catalog_duplicate_rejected():
    catalog = ModelCatalog()
    spec = ModelSpec(name="m", provider="p", cost_in_per_1k=0, cost_out_per_1k=0,
                     quality=50, context_window=1000)
    catalog.add(spec)
    with pytest.raises(Exception):
        catalog.add(spec)


# ---- routing ----------------------------------------------------------


def test_cheapest_first_picks_cheapest_eligible():
    router = make_router()
    decision = router.route("hello", constraints=RoutingConstraints(min_quality=55))
    assert decision.model.name == "nano-1"
    assert decision.policy == Policy.CHEAPEST_FIRST


def test_quality_first_picks_best_eligible():
    router = make_router()
    decision = router.route(
        "hello",
        policy=Policy.QUALITY_FIRST,
        constraints=RoutingConstraints(min_quality=60),
    )
    assert decision.model.name == "pro-1"


def test_budget_capped_stays_under_cap():
    router = make_router()
    decision = router.route(
        "hello",
        policy=Policy.BUDGET_CAPPED,
        constraints=RoutingConstraints(max_cost_usd=0.05),
    )
    assert decision.estimated_cost_usd <= 0.05
    assert decision.model.name in {"nano-1", "flash-2"}


def test_budget_capped_requires_cap():
    router = make_router()
    with pytest.raises(ValueError):
        router.route("hello", policy=Policy.BUDGET_CAPPED)


def test_no_eligible_model_raises():
    router = make_router()
    with pytest.raises(NoEligibleModelError):
        router.route("hello", constraints=RoutingConstraints(min_quality=99.9))


def test_vision_requirement_routes_to_vision_model():
    router = make_router()
    decision = router.route(
        "describe this image",
        constraints=RoutingConstraints(required_features=["vision"], min_quality=60),
    )
    assert decision.model.name == "vision-lite"  # cheapest with vision


# ---- execution & fallbacks --------------------------------------------


def test_execute_serves_and_returns_completion():
    router = make_router()
    completion, decision, attempted = router.execute("hello")
    assert completion.model == decision.model.name
    assert completion.text.startswith(f"[{decision.model.name}]")
    assert attempted == [decision.model.name]
    assert completion.cost_usd > 0


def test_execute_falls_back_on_backend_failure():
    catalog = make_catalog()
    router = Router(catalog)
    for spec in catalog:
        router.register_backend(
            MockBackend(spec, fail=(spec.name == "nano-1"))
        )
    completion, decision, attempted = router.execute("hello")
    assert decision.model.name == "nano-1"      # routing still picks it
    assert attempted[0] == "nano-1"             # first attempt fails
    assert completion.model != "nano-1"         # fallback served it
    assert len(attempted) == 2


def test_execute_all_fail_raises():
    catalog = make_catalog()
    router = Router(catalog)
    for spec in catalog:
        router.register_backend(MockBackend(spec, fail=True))
    with pytest.raises(BackendError):
        router.execute("hello")


def test_compare_orders_by_cost():
    router = make_router()
    rows = router.compare("hello world, price me out")
    costs = [r["est_cost_usd"] for r in rows]
    assert costs == sorted(costs)
    assert rows[0]["model"] == "nano-1"


# ---- tracking ---------------------------------------------------------


def test_ledger_summary_math():
    ledger = UsageLedger()
    ledger.log(model="nano-1", policy="cheapest-first", input_tokens=100,
               output_tokens=50, cost_usd=0.01, quality=62.0)
    ledger.log(model="pro-1", policy="quality-first", input_tokens=200,
               output_tokens=100, cost_usd=0.50, quality=91.0,
               attempted=["pro-1", "flash-2"])
    s = ledger.summary()
    assert s["total_requests"] == 2
    assert s["total_cost_usd"] == pytest.approx(0.51)
    assert s["total_tokens"] == 450
    assert s["avg_quality"] == pytest.approx(76.5)
    assert s["fallback_rate"] == pytest.approx(0.5)
    assert s["per_model"]["nano-1"]["requests"] == 1
    assert s["per_model"]["pro-1"]["avg_quality"] == 91.0


def test_savings_vs_baseline():
    ledger = UsageLedger()
    ledger.log(model="nano-1", policy="cheapest-first", input_tokens=100,
               output_tokens=50, cost_usd=0.01, quality=62.0)
    assert ledger.savings_vs(1.00) == pytest.approx(0.99)


# ---- backends ---------------------------------------------------------


def test_openai_compatible_backend_wraps_failures():
    spec = ModelSpec(name="m", provider="p", cost_in_per_1k=1.0,
                     cost_out_per_1k=1.0, quality=80, context_window=1000)
    backend = OpenAICompatibleBackend(
        spec, base_url="http://127.0.0.1:1", timeout=1.0  # nothing listens here
    )
    assert backend.model_name == "m"
    with pytest.raises(BackendError):
        backend.complete([{"role": "user", "content": "hi"}])
