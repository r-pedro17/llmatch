import json
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "hf_models.json")
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/AlexsJones/llmfit/main/data/hf_models.json"
)
CACHE_TTL_DAYS = 7


@dataclass
class Model:
    name: str
    provider: str
    parameter_count: str       # human-readable: "7B", "70B"
    parameters_raw: int        # raw parameter count
    min_ram_gb: float
    recommended_ram_gb: float
    min_vram_gb: float
    quantization: str          # default/recommended quant
    context_length: int
    use_case: str
    capabilities: List[str]
    architecture: str
    hf_downloads: int
    hf_likes: int
    release_date: str
    is_moe: bool = False
    num_experts: int = 0
    active_experts: int = 0
    active_parameters: int = 0
    pipeline_tag: str = "text-generation"
    gguf_sources: List[str] = field(default_factory=list)


def _is_cache_stale() -> bool:
    """Return True if cached data file is older than CACHE_TTL_DAYS."""
    if not os.path.exists(DATA_FILE):
        return True
    age_days = (time.time() - os.path.getmtime(DATA_FILE)) / 86400
    return age_days > CACHE_TTL_DAYS


def _fetch_and_cache() -> bool:
    """Fetch model database from upstream. Returns True on success."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    sys.stderr.write("Fetching model database from upstream...\n")
    try:
        with urllib.request.urlopen(UPSTREAM_URL, timeout=15) as resp:
            data = resp.read()
        models = json.loads(data)
        with open(DATA_FILE, "wb") as f:
            f.write(data)
        sys.stderr.write(f"Cached {len(models)} models to {DATA_FILE}\n")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        sys.stderr.write(f"Warning: failed to fetch model database: {e}\n")
        return False


def load(refresh: bool = False) -> List[Model]:
    if refresh or not os.path.exists(DATA_FILE):
        if not _fetch_and_cache() and not os.path.exists(DATA_FILE):
            print("Error: no cached model database and fetch failed.", file=sys.stderr)
            sys.exit(1)
    elif _is_cache_stale():
        if not _fetch_and_cache():
            print("Using stale cache (fetch failed).", file=sys.stderr)

    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: corrupted model database: {e}", file=sys.stderr)
        print("Try running with --refresh to re-download.", file=sys.stderr)
        sys.exit(1)

    models = []
    for entry in raw:
        models.append(Model(
            name=entry.get("name", ""),
            provider=entry.get("provider", ""),
            parameter_count=entry.get("parameter_count", ""),
            parameters_raw=entry.get("parameters_raw", 0),
            min_ram_gb=entry.get("min_ram_gb", 0.0),
            recommended_ram_gb=entry.get("recommended_ram_gb", 0.0),
            min_vram_gb=entry.get("min_vram_gb", 0.0),
            quantization=entry.get("quantization", "Q4_K_M"),
            context_length=entry.get("context_length", 0),
            use_case=entry.get("use_case", ""),
            capabilities=entry.get("capabilities", []),
            architecture=entry.get("architecture", ""),
            hf_downloads=entry.get("hf_downloads", 0),
            hf_likes=entry.get("hf_likes", 0),
            release_date=entry.get("release_date", ""),
            is_moe=entry.get("is_moe", False),
            num_experts=entry.get("num_experts", 0),
            active_experts=entry.get("active_experts", 0),
            active_parameters=entry.get("active_parameters", 0),
            pipeline_tag=entry.get("pipeline_tag", "text-generation"),
            gguf_sources=entry.get("gguf_sources", []),
        ))
    return models
