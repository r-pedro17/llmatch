import math
from dataclasses import dataclass
from typing import List, Optional

from hardware import HardwareProfile
from models import Model

# Bits per parameter for each quantization level
QUANT_BITS = {
    "Q8_0":   8.0,
    "Q6_K":   6.6,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.8,
    "Q3_K_M": 3.9,
    "Q2_K":   2.6,
}
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

# Quality multiplier per quantization (relative to Q8_0 = 1.0)
QUANT_QUALITY = {
    "Q8_0":   1.00,
    "Q6_K":   0.97,
    "Q5_K_M": 0.94,
    "Q4_K_M": 0.88,
    "Q3_K_M": 0.78,
    "Q2_K":   0.60,
}

# Approximate memory bandwidth (GB/s) by backend for speed estimation
BACKEND_BANDWIDTH = {
    "cuda":  700,
    "rocm":  500,
    "metal": 300,
    "cpu":    50,
}


@dataclass
class ScoredModel:
    model: Model
    best_quant: str
    est_vram_gb: float
    fit_level: str      # "perfect", "good", "marginal", "too_tight"
    run_mode: str       # "gpu", "cpu+gpu", "cpu", "moe"
    est_tps: float      # estimated tokens/second
    score_quality: float
    score_speed: float
    score_fit: float
    score_context: float
    score_total: float


def _model_memory_gb(model: Model, quant: str) -> float:
    """Estimate VRAM/RAM needed for a model at a given quantization."""
    bits = QUANT_BITS.get(quant, 4.8)
    # For MoE, active parameter memory + small fraction for inactive experts
    if model.is_moe and model.active_parameters > 0:
        active_params = model.active_parameters
        # inactive experts still loaded to RAM but not VRAM-active
        total_params = model.parameters_raw
        inactive_params = total_params - active_params
        active_gb = active_params * bits / 8 / 1e9
        inactive_gb = inactive_params * 2.0 / 8 / 1e9  # Q2 for inactive
        return active_gb + inactive_gb
    return model.parameters_raw * bits / 8 / 1e9


def _best_quant(model: Model, available_gb: float) -> Optional[str]:
    """Return the best (highest quality) quant that fits in available_gb."""
    for quant in QUANT_ORDER:
        mem = _model_memory_gb(model, quant)
        if mem <= available_gb:
            return quant
    return None


def _fit_level(util: float) -> str:
    """Classify memory utilization into fit level."""
    if util > 1.0:
        return "too_tight"
    if util >= 0.5:
        return "perfect"
    if util >= 0.25:
        return "good"
    return "marginal"


def _score_quality(model: Model, quant: str) -> float:
    """0–100. Based on log-scaled parameter count and quant quality."""
    if model.parameters_raw <= 0:
        return 0.0
    # Log scale: 1B → ~30, 7B → ~53, 70B → ~77, 700B → ~100
    param_score = min(100.0, math.log10(model.parameters_raw / 1e6) / math.log10(1e6) * 100)
    param_score = max(0.0, param_score)
    quality_mult = QUANT_QUALITY.get(quant, 0.88)
    return round(param_score * quality_mult, 1)


def _score_speed(model: Model, quant: str, hw: HardwareProfile) -> float:
    """0–100. Rough tok/s estimation normalized to a reference."""
    bits = QUANT_BITS.get(quant, 4.8)
    param_gb = model.parameters_raw * bits / 8 / 1e9
    if param_gb <= 0:
        return 0.0

    bandwidth = BACKEND_BANDWIDTH.get(hw.backend, 50)
    est_tps = bandwidth / param_gb  # very rough: bandwidth / model_size

    # Reference: 30 tok/s is a comfortable speed → score 70
    #            100 tok/s → score ~100, 5 tok/s → score ~30
    score = min(100.0, math.log10(max(est_tps, 0.1) + 1) / math.log10(101) * 100)
    return round(score, 1)


def _est_tps(model: Model, quant: str, hw: HardwareProfile) -> float:
    bits = QUANT_BITS.get(quant, 4.8)
    param_gb = model.parameters_raw * bits / 8 / 1e9
    if param_gb <= 0:
        return 0.0
    bandwidth = BACKEND_BANDWIDTH.get(hw.backend, 50)
    return round(bandwidth / param_gb, 1)


def _score_fit(util: float) -> float:
    """0–100. Sweet spot is 50–80% utilization."""
    if util > 1.0:
        return 0.0
    if 0.5 <= util <= 0.8:
        return 100.0
    if util > 0.8:
        # Too tight — linearly decrease from 100 at 0.8 to 40 at 1.0
        return round(100.0 - (util - 0.8) / 0.2 * 60, 1)
    if util >= 0.25:
        # Underutilized but usable
        return round(util / 0.5 * 80, 1)
    return round(util / 0.25 * 30, 1)


def _score_context(model: Model) -> float:
    """0–100. More context is generally better, diminishing returns."""
    ctx = model.context_length
    if ctx <= 0:
        return 0.0
    # 4K → ~60, 32K → ~85, 128K → ~95, 1M+ → ~100
    score = min(100.0, math.log2(max(ctx, 1)) / math.log2(2**20) * 100)
    return round(score, 1)


def _run_mode(model: Model, hw: HardwareProfile) -> str:
    if model.is_moe:
        return "moe"
    if hw.gpu_vram_gb > 0:
        return "gpu"
    if hw.ram_gb > 0:
        return "cpu"
    return "cpu"


def score_all(
    models: List[Model],
    hw: HardwareProfile,
    max_context: Optional[int] = None,
    weights: Optional[dict] = None,
) -> List[ScoredModel]:
    if weights is None:
        weights = {"quality": 0.30, "speed": 0.25, "fit": 0.30, "context": 0.15}

    available_gb = hw.effective_memory_gb()
    scored = []

    for model in models:
        # Apply context cap
        ctx = model.context_length
        if max_context and ctx > max_context:
            ctx = max_context

        quant = _best_quant(model, available_gb)
        if quant is None:
            # Model doesn't fit even at lowest quant — skip
            continue

        mem_gb = _model_memory_gb(model, quant)
        util = mem_gb / available_gb if available_gb > 0 else 1.0
        fit = _fit_level(util)

        sq = _score_quality(model, quant)
        ss = _score_speed(model, quant, hw)
        sf = _score_fit(util)
        sc = _score_context(model)
        total = round(
            sq * weights["quality"]
            + ss * weights["speed"]
            + sf * weights["fit"]
            + sc * weights["context"],
            1,
        )
        tps = _est_tps(model, quant, hw)

        scored.append(ScoredModel(
            model=model,
            best_quant=quant,
            est_vram_gb=round(mem_gb, 2),
            fit_level=fit,
            run_mode=_run_mode(model, hw),
            est_tps=tps,
            score_quality=sq,
            score_speed=ss,
            score_fit=sf,
            score_context=sc,
            score_total=total,
        ))

    scored.sort(key=lambda x: x.score_total, reverse=True)
    return scored
