"""Default fleet used by the CLI demo and quickstart examples.

Prices are illustrative sample values in the shape of real provider pricing;
edit them (or point the CLI at your own catalog file) before relying on
any cost numbers for real spend decisions.
"""

DEFAULT_MODELS = [
    {
        "name": "nano-1",
        "provider": "acme",
        "cost_in_per_1k": 0.02,
        "cost_out_per_1k": 0.06,
        "quality": 62.0,
        "context_window": 16384,
        "features": ["json", "tools"],
        "notes": "Tiny workhorse: classification, extraction, simple rewrites.",
    },
    {
        "name": "flash-2",
        "provider": "acme",
        "cost_in_per_1k": 0.10,
        "cost_out_per_1k": 0.30,
        "quality": 78.0,
        "context_window": 65536,
        "features": ["json", "tools"],
        "notes": "Balanced mid-tier for everyday chat and RAG answers.",
    },
    {
        "name": "pro-1",
        "provider": "acme",
        "cost_in_per_1k": 1.25,
        "cost_out_per_1k": 5.00,
        "quality": 91.0,
        "context_window": 131072,
        "features": ["json", "tools", "vision"],
        "notes": "Flagship quality; use only when it earns its keep.",
    },
    {
        "name": "vision-lite",
        "provider": "beta",
        "cost_in_per_1k": 0.35,
        "cost_out_per_1k": 0.70,
        "quality": 71.0,
        "context_window": 32768,
        "features": ["json", "vision"],
        "notes": "Cheapest model in the fleet that can see images.",
    },
    {
        "name": "longctx-1",
        "provider": "beta",
        "cost_in_per_1k": 0.50,
        "cost_out_per_1k": 1.50,
        "quality": 74.0,
        "context_window": 1048576,
        "features": ["json", "long-context"],
        "notes": "1M-token context for whole-repo summarization jobs.",
    },
]
