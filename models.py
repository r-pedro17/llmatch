import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "hf_models.json")
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/AlexsJones/llmfit/main/data/hf_models.json"
)


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


def _fetch_and_cache() -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    print(f"Fetching model database from upstream...")
    with urllib.request.urlopen(UPSTREAM_URL, timeout=15) as resp:
        data = resp.read()
    with open(DATA_FILE, "wb") as f:
        f.write(data)
    print(f"Cached {len(json.loads(data))} models to {DATA_FILE}")


def load(refresh: bool = False) -> List[Model]:
    if refresh or not os.path.exists(DATA_FILE):
        _fetch_and_cache()

    with open(DATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)

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
