"""Cost/quality tracking: per-request ledger and fleet summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RequestRecord:
    """One served request."""

    timestamp: str
    model: str
    policy: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality: float  # quality score of the model that served the request
    attempted: List[str] = field(default_factory=list)  # fallback chain walked


class UsageLedger:
    """Append-only ledger of served requests with aggregate reports."""

    def __init__(self):
        self.records: List[RequestRecord] = []

    # ---- recording ----------------------------------------------------
    def log(
        self,
        *,
        model: str,
        policy: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        quality: float,
        attempted: Optional[List[str]] = None,
    ) -> RequestRecord:
        record = RequestRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model=model,
            policy=policy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            quality=quality,
            attempted=list(attempted or [model]),
        )
        self.records.append(record)
        return record

    # ---- aggregates ---------------------------------------------------
    @property
    def total_requests(self) -> int:
        return len(self.records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    @property
    def avg_quality(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.quality for r in self.records) / len(self.records)

    def per_model(self) -> Dict[str, dict]:
        agg: Dict[str, dict] = {}
        for r in self.records:
            entry = agg.setdefault(
                r.model,
                {"requests": 0, "cost_usd": 0.0, "tokens": 0, "quality_sum": 0.0},
            )
            entry["requests"] += 1
            entry["cost_usd"] += r.cost_usd
            entry["tokens"] += r.input_tokens + r.output_tokens
            entry["quality_sum"] += r.quality
        for entry in agg.values():
            entry["avg_quality"] = entry["quality_sum"] / entry["requests"]
            del entry["quality_sum"]
            entry["cost_usd"] = round(entry["cost_usd"], 6)
        return agg

    def fallback_rate(self) -> float:
        """Fraction of requests that needed more than one attempt."""
        if not self.records:
            return 0.0
        used = sum(1 for r in self.records if len(r.attempted) > 1)
        return used / len(self.records)

    def savings_vs(self, baseline_model_cost_per_request: float) -> float:
        """USD saved vs. routing every request to a flat-rate baseline model."""
        return baseline_model_cost_per_request * len(self.records) - self.total_cost

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "avg_quality": round(self.avg_quality, 2),
            "fallback_rate": round(self.fallback_rate(), 4),
            "per_model": self.per_model(),
        }

    # ---- persistence ----------------------------------------------------
    def save_jsonl(self, path: str | Path) -> int:
        """Append-style durable request log: one JSON object per line.

        Returns the number of records written. The file can be tailed,
        grepped, or loaded back with :meth:`load_jsonl`.
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(asdict(record)) + "\n")
        return len(self.records)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "UsageLedger":
        """Rebuild a ledger from a JSONL file written by :meth:`save_jsonl`."""
        ledger = cls()
        with open(Path(path), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                ledger.records.append(
                    RequestRecord(
                        timestamp=data["timestamp"],
                        model=data["model"],
                        policy=data["policy"],
                        input_tokens=int(data["input_tokens"]),
                        output_tokens=int(data["output_tokens"]),
                        cost_usd=float(data["cost_usd"]),
                        quality=float(data["quality"]),
                        attempted=list(data.get("attempted", [data["model"]])),
                    )
                )
        return ledger
