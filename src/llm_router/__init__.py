"""llm-router: route to the cheapest capable model."""

from .catalog import ModelCatalog
from .models import (
    BackendError,
    Completion,
    MockBackend,
    ModelBackend,
    ModelSpec,
    OpenAICompatibleBackend,
)
from .router import NoEligibleModelError, Policy, Router, RoutingConstraints, RoutingDecision
from .tracking import RequestRecord, UsageLedger

__all__ = [
    "ModelSpec",
    "ModelCatalog",
    "ModelBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "BackendError",
    "Completion",
    "Policy",
    "RoutingConstraints",
    "RoutingDecision",
    "NoEligibleModelError",
    "Router",
    "UsageLedger",
    "RequestRecord",
]

__version__ = "0.1.0"
__author__ = "Anusha Mukka"
