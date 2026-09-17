"""Model specs and backend adapters for llm-router."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ModelSpec:
    """Static capability/cost description of a model.

    Attributes:
        name: Identifier, e.g. ``"gpt-4o-mini"``.
        provider: Provider label, e.g. ``"openai"``.
        cost_in_per_1k: USD per 1k input tokens.
        cost_out_per_1k: USD per 1k output tokens.
        quality: Quality score in [0, 100] (your own eval benchmark numbers).
        context_window: Max input tokens.
        features: Capability flags the model supports,
            e.g. {"json", "vision", "tools", "long-context"}.
        notes: Free-form notes about the model.
    """

    name: str
    provider: str
    cost_in_per_1k: float
    cost_out_per_1k: float
    quality: float
    context_window: int
    features: frozenset = field(default_factory=frozenset)
    notes: str = ""

    def supports(self, required: Sequence[str]) -> bool:
        """True when this model supports every required feature flag."""
        return all(flag in self.features for flag in required)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int = 0
    ) -> float:
        """Estimated USD cost for a request with this many tokens."""
        return (
            self.cost_in_per_1k * input_tokens / 1000.0
            + self.cost_out_per_1k * output_tokens / 1000.0
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "cost_in_per_1k": self.cost_in_per_1k,
            "cost_out_per_1k": self.cost_out_per_1k,
            "quality": self.quality,
            "context_window": self.context_window,
            "features": sorted(self.features),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            name=data["name"],
            provider=data.get("provider", "unknown"),
            cost_in_per_1k=float(data.get("cost_in_per_1k", 0.0)),
            cost_out_per_1k=float(data.get("cost_out_per_1k", 0.0)),
            quality=float(data.get("quality", 0.0)),
            context_window=int(data.get("context_window", 4096)),
            features=frozenset(data.get("features", [])),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class Completion:
    """Result of a backend chat completion."""

    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    finish_reason: str = "stop"


class BackendError(RuntimeError):
    """Raised when a model backend fails to serve a request."""


class ModelBackend(ABC):
    """Pluggable backend that executes completions against one model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model this backend serves."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Completion:
        """Run a chat completion, raising BackendError on failure."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token). Deterministic and offline."""
        return max(1, (len(text) + 3) // 4)


class MockBackend(ModelBackend):
    """Deterministic offline stub backend. Useful for demos, tests, CI.

    It never calls the network; it echoes a canned response prefixed with the
    model name so routing behaviour is visible in the CLI demo.
    """

    def __init__(
        self,
        spec: ModelSpec,
        response_text: str = "This is a stub response from the mock backend.",
        fail: bool = False,
    ):
        self._spec = spec
        self._response_text = response_text
        self._fail = fail

    @property
    def model_name(self) -> str:
        return self._spec.name

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Completion:
        if self._fail:
            raise BackendError(f"mock backend for {self._spec.name} is failing")
        prompt = " ".join(m.get("content", "") for m in messages)
        in_tokens = self.estimate_tokens(prompt)
        out_text = f"[{self._spec.name}] {self._response_text}"
        out_tokens = self.estimate_tokens(out_text)
        return Completion(
            model=self._spec.name,
            text=out_text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=self._spec.estimate_cost(in_tokens, out_tokens),
        )


class OpenAICompatibleBackend(ModelBackend):
    """Adapter for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    ``base_url`` points at the API root (e.g. ``https://api.openai.com/v1`` or
    any local OpenAI-compatible server such as Ollama or vLLM), and
    ``api_key`` is used if the endpoint requires one.
    """

    def __init__(
        self,
        spec: ModelSpec,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self._spec = spec
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._spec.name

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Completion:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self._spec.name,
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # network/HTTP/JSON errors all map to BackendError
            raise BackendError(
                f"request to {self._base_url} for model {self._spec.name} failed: {exc}"
            ) from exc
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason", "stop")
            usage = body.get("usage", {})
            in_tokens = int(usage.get("prompt_tokens", self.estimate_tokens(text)))
            out_tokens = int(usage.get("completion_tokens", self.estimate_tokens(text)))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendError(f"unexpected response shape: {exc}") from exc
        return Completion(
            model=self._spec.name,
            text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=self._spec.estimate_cost(in_tokens, out_tokens),
            finish_reason=finish,
        )
