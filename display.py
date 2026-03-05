import json
import sys
from typing import List, Optional

from hardware import HardwareProfile
from scorer import ScoredModel

# ANSI color codes — disabled on non-TTY
_COLORS = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "red":     "\033[31m",
    "cyan":    "\033[36m",
    "dim":     "\033[2m",
}

FIT_COLORS = {
    "perfect":   "green",
    "good":      "cyan",
    "marginal":  "yellow",
    "too_tight": "red",
}


def _c(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return _COLORS.get(color, "") + text + _COLORS["reset"]


def _bold(text: str) -> str:
    return _c(text, "bold")


def print_hardware(hw: HardwareProfile) -> None:
    print(_bold("System Hardware"))
    print(f"  RAM        : {hw.ram_gb:.1f} GB")
    print(f"  CPU cores  : {hw.cpu_cores}")
    if hw.gpu_vram_gb > 0:
        gpu_label = hw.gpu_name or "GPU"
        print(f"  GPU        : {gpu_label}")
        print(f"  VRAM       : {hw.gpu_vram_gb:.1f} GB")
    else:
        print(f"  GPU        : none detected")
    print(f"  Backend    : {hw.backend}")
    eff = hw.effective_memory_gb()
    print(f"  Eff. memory: {eff:.1f} GB (used for scoring)")


def print_table(scored: List[ScoredModel], top: Optional[int] = None) -> None:
    if top:
        scored = scored[:top]
    if not scored:
        print("No models fit your hardware.")
        return

    col_w = {
        "rank":   4,
        "name":   38,
        "score":  6,
        "fit":    9,
        "quant":  8,
        "tps":    8,
        "vram":   6,
        "use":    28,
    }

    header = (
        f"{'#':>{col_w['rank']}}  "
        f"{'Model':<{col_w['name']}}  "
        f"{'Score':>{col_w['score']}}  "
        f"{'Fit':<{col_w['fit']}}  "
        f"{'Quant':<{col_w['quant']}}  "
        f"{'tok/s':>{col_w['tps']}}  "
        f"{'Mem GB':>{col_w['vram']}}  "
        f"{'Use Case':<{col_w['use']}}"
    )
    sep = "-" * len(header)
    print(_bold(header))
    print(sep)

    for i, sm in enumerate(scored, 1):
        name = sm.model.name
        if len(name) > col_w["name"]:
            name = name[: col_w["name"] - 1] + "…"
        use = sm.model.use_case
        if len(use) > col_w["use"]:
            use = use[: col_w["use"] - 1] + "…"

        fit_color = FIT_COLORS.get(sm.fit_level, "reset")
        fit_str = _c(f"{sm.fit_level:<{col_w['fit']}}", fit_color)

        tps_str = f"{sm.est_tps:>{col_w['tps']}.1f}" if sm.est_tps < 10000 else f"{'>9999':>{col_w['tps']}}"

        line = (
            f"{i:>{col_w['rank']}}  "
            f"{name:<{col_w['name']}}  "
            f"{sm.score_total:>{col_w['score']}.1f}  "
            f"{fit_str}  "
            f"{sm.best_quant:<{col_w['quant']}}  "
            f"{tps_str}  "
            f"{sm.est_vram_gb:>{col_w['vram']}.1f}  "
            f"{use:<{col_w['use']}}"
        )
        print(line)


def print_json(scored: List[ScoredModel], top: Optional[int] = None) -> None:
    if top:
        scored = scored[:top]
    out = []
    for sm in scored:
        out.append({
            "name": sm.model.name,
            "provider": sm.model.provider,
            "score": sm.score_total,
            "fit_level": sm.fit_level,
            "run_mode": sm.run_mode,
            "best_quant": sm.best_quant,
            "est_tps": sm.est_tps,
            "est_vram_gb": sm.est_vram_gb,
            "use_case": sm.model.use_case,
            "parameters": sm.model.parameter_count,
            "context_length": sm.model.context_length,
            "architecture": sm.model.architecture,
            "scores": {
                "quality": sm.score_quality,
                "speed": sm.score_speed,
                "fit": sm.score_fit,
                "context": sm.score_context,
            },
        })
    print(json.dumps(out, indent=2))


def print_model_info(sm: ScoredModel) -> None:
    m = sm.model
    print(_bold(f"\n{m.name}"))
    print(f"  Provider     : {m.provider}")
    print(f"  Architecture : {m.architecture}")
    print(f"  Parameters   : {m.parameter_count}")
    print(f"  Context      : {m.context_length:,} tokens")
    print(f"  Use case     : {m.use_case}")
    if m.capabilities:
        print(f"  Capabilities : {', '.join(m.capabilities)}")
    print(f"  Released     : {m.release_date}")
    print(f"  HF downloads : {m.hf_downloads:,}")
    if m.is_moe:
        print(f"  MoE experts  : {m.num_experts} total / {m.active_experts} active")
    print()
    print(_bold("  Fit for your hardware"))
    fit_color = FIT_COLORS.get(sm.fit_level, "reset")
    print(f"  Fit level    : {_c(sm.fit_level, fit_color)}")
    print(f"  Run mode     : {sm.run_mode}")
    print(f"  Best quant   : {sm.best_quant}")
    print(f"  Est. memory  : {sm.est_vram_gb:.2f} GB")
    print(f"  Est. tok/s   : {sm.est_tps:.1f}")
    print()
    print(_bold("  Scores"))
    print(f"  Quality      : {sm.score_quality:.1f}")
    print(f"  Speed        : {sm.score_speed:.1f}")
    print(f"  Fit          : {sm.score_fit:.1f}")
    print(f"  Context      : {sm.score_context:.1f}")
    print(f"  Total        : {_bold(f'{sm.score_total:.1f}')}")
