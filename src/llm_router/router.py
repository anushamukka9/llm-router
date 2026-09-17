"""Routing policies: pick the cheapest *capable* model for each request."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from .catalog import ModelCatalog
from .models import BackendError, Completion, ModelBackend, ModelSpec


class NoEligibleModelError(ValueError):
    """Raised when no catalog model satisfies the routing constraints."""


class Policy(str, Enum):
    CHEAPEST_FIRST = "cheapest-first"
    QUALITY_FIRST = "quality-first"
    BUDGET_CAPPED = "budget-capped"


@dataclass(frozen=True)
class RoutingConstraints:
    """Hard constraints every eligible model must satisfy."""

    required_features: tuple = ()
    min_quality: float = 0.0
    min_context: int = 0
    provider: Optional[str] = None
    # Budget cap in USD for the estimated *input* cost of this request.
    max_cost_usd: Optional[float] = None


@dataclass
class RoutingDecision:
    """Outcome of one routing decision."""

    model: ModelSpec
    policy: Policy
    reason: str
    estimated_cost_usd: float
    # Ordered fallback chain: models to try if the primary backend fails.
    fallbacks: List[ModelSpec] = field(default_factory=list)
    constraints: Optional[RoutingConstraints] = None


def _estimated_input_tokens(prompt: str) -> int:
    return max(1, (len(prompt) + 3) // 4)


def _input_cost_ok(spec: ModelSpec, constraints: RoutingConstraints, prompt: str) -> bool:
    if constraints.max_cost_usd is None:
        return True
    return spec.estimate_cost(_estimated_input_tokens(prompt)) <= constraints.max_cost_usd


class Router:
    """Routes requests to models from a catalog according to a policy."""

    def __init__(self, catalog: ModelCatalog, backends: Optional[Dict[str, ModelBackend]] = None):
        self.catalog = catalog
        self.backends: Dict[str, ModelBackend] = dict(backends or {})

    # ---- backend registration -----------------------------------------
    def register_backend(self, backend: ModelBackend) -> None:
        self.backends[backend.model_name] = backend

    # ---- decision -----------------------------------------------------
    def route(
        self,
        prompt: str,
        *,
        policy: Policy = Policy.CHEAPEST_FIRST,
        constraints: Optional[RoutingConstraints] = None,
        max_tokens: int = 512,
        fallback_depth: int = 2,
    ) -> RoutingDecision:
        constraints = constraints or RoutingConstraints()
        pool = [
            s
            for s in self.catalog.candidates(
                required_features=constraints.required_features,
                min_quality=constraints.min_quality,
                min_context=constraints.min_context,
                provider=constraints.provider,
            )
            if _input_cost_ok(s, constraints, prompt)
        ]
        if not pool:
            raise NoEligibleModelError(
                "no model satisfies the routing constraints "
                f"(features={list(constraints.required_features)}, "
                f"min_quality={constraints.min_quality}, "
                f"min_context={constraints.min_context}, "
                f"max_cost_usd={constraints.max_cost_usd})"
            )

        in_tokens = _estimated_input_tokens(prompt)
        cost = {s.name: s.estimate_cost(in_tokens, max_tokens) for s in pool}

        if policy == Policy.BUDGET_CAPPED:
            if constraints.max_cost_usd is None:
                raise ValueError("budget-capped policy requires max_cost_usd constraint")
            pool = [s for s in pool if cost[s.name] <= constraints.max_cost_usd]
            if not pool:
                raise NoEligibleModelError(
                    f"no model fits the budget cap ${constraints.max_cost_usd:.4f}"
                )
            ordered = sorted(pool, key=lambda s: (-s.quality, cost[s.name]))
            reason = (
                f"highest quality within budget cap ${constraints.max_cost_usd:.4f}"
            )
        elif policy == Policy.CHEAPEST_FIRST:
            ordered = sorted(pool, key=lambda s: (cost[s.name], -s.quality))
            reason = "lowest estimated cost among eligible models"
        elif policy == Policy.QUALITY_FIRST:
            ordered = sorted(pool, key=lambda s: (-s.quality, cost[s.name]))
            reason = "highest quality among eligible models"
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown policy: {policy!r}")

        primary = ordered[0]
        fallbacks = ordered[1 : 1 + fallback_depth]
        return RoutingDecision(
            model=primary,
            policy=policy,
            reason=reason,
            estimated_cost_usd=cost[primary.name],
            fallbacks=fallbacks,
            constraints=constraints,
        )

    # ---- execution with fallbacks -------------------------------------
    def execute(
        self,
        prompt: str,
        *,
        policy: Policy = Policy.CHEAPEST_FIRST,
        constraints: Optional[RoutingConstraints] = None,
        max_tokens: int = 512,
        fallback_depth: int = 2,
    ) -> tuple[Completion, RoutingDecision, List[str]]:
        """Route a prompt and execute it, walking the fallback chain on failure.

        Returns (completion, decision, attempted_model_names). Raises
        BackendError if every model in the chain fails.
        """
        decision = self.route(
            prompt,
            policy=policy,
            constraints=constraints,
            max_tokens=max_tokens,
            fallback_depth=fallback_depth,
        )
        attempted: List[str] = []
        last_error: Optional[BackendError] = None
        for spec in [decision.model, *decision.fallbacks]:
            attempted.append(spec.name)
            backend = self.backends.get(spec.name)
            if backend is None:
                last_error = BackendError(f"no backend registered for {spec.name!r}")
                continue
            try:
                completion = backend.complete(
                    [{"role": "user", "content": prompt}], max_tokens=max_tokens
                )
            except BackendError as exc:
                last_error = exc
                continue
            return completion, decision, attempted
        raise BackendError(
            f"all {len(attempted)} models in the chain failed; last error: {last_error}"
        )

    # ---- comparison table ----------------------------------------------
    def compare(self, prompt: str, max_tokens: int = 512) -> List[dict]:
        """Rank every catalog model for a prompt by estimated cost.

        Useful for the CLI demo: shows what the router considered.
        """
        in_tokens = _estimated_input_tokens(prompt)
        rows = []
        for spec in self.catalog:
            rows.append(
                {
                    "model": spec.name,
                    "provider": spec.provider,
                    "quality": spec.quality,
                    "context": spec.context_window,
                    "features": sorted(spec.features),
                    "est_cost_usd": round(spec.estimate_cost(in_tokens, max_tokens), 6),
                }
            )
        return sorted(rows, key=lambda r: r["est_cost_usd"])
