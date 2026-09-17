"""Model catalog: loading, querying, and managing the model fleet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

try:  # PyYAML is optional; JSON catalogs always work.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .models import ModelSpec


class CatalogError(ValueError):
    """Raised for invalid catalog data or operations."""


class ModelCatalog:
    """An in-memory fleet of :class:`ModelSpec` with filtering helpers."""

    def __init__(self, specs: Iterable[ModelSpec] = ()):
        self._specs: List[ModelSpec] = []
        self._by_name: dict = {}
        for spec in specs:
            self.add(spec)

    # ---- mutation -----------------------------------------------------
    def add(self, spec: ModelSpec) -> None:
        if spec.name in self._by_name:
            raise CatalogError(f"duplicate model name: {spec.name!r}")
        self._specs.append(spec)
        self._by_name[spec.name] = spec

    def remove(self, name: str) -> None:
        spec = self.get(name)  # raises if missing
        self._specs.remove(spec)
        del self._by_name[name]

    # ---- lookup -------------------------------------------------------
    def get(self, name: str) -> ModelSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise CatalogError(f"unknown model: {name!r}") from None

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    # ---- filtering ----------------------------------------------------
    def candidates(
        self,
        *,
        required_features: Sequence[str] = (),
        min_quality: float = 0.0,
        max_cost_per_1k_in: Optional[float] = None,
        min_context: int = 0,
        provider: Optional[str] = None,
    ) -> List[ModelSpec]:
        """Return specs satisfying every constraint, unsorted."""
        out = []
        for spec in self._specs:
            if not spec.supports(required_features):
                continue
            if spec.quality < min_quality:
                continue
            if max_cost_per_1k_in is not None and spec.cost_in_per_1k > max_cost_per_1k_in:
                continue
            if spec.context_window < min_context:
                continue
            if provider is not None and spec.provider != provider:
                continue
            out.append(spec)
        return out

    # ---- (de)serialization --------------------------------------------
    def to_dict(self) -> dict:
        return {"models": [s.to_dict() for s in self._specs]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelCatalog":
        models = data.get("models", [])
        if not models:
            raise CatalogError("catalog contains no models")
        return cls(ModelSpec.from_dict(m) for m in models)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelCatalog":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise CatalogError(
                    "PyYAML is required to load YAML catalogs (pip install pyyaml)"
                )
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise CatalogError(f"unsupported catalog format: {suffix!r}")
        return cls.from_dict(data)

    def to_file(self, path: str | Path, format: str = "yaml") -> None:
        path = Path(path)
        data = self.to_dict()
        if format == "yaml":
            if yaml is None:
                raise CatalogError("PyYAML is required to write YAML catalogs")
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        elif format == "json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            raise CatalogError(f"unsupported catalog format: {format!r}")
