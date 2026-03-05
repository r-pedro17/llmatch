# Plan

## Phase 1 — Hardware detection
- [ ] Detect total RAM (Windows: ctypes, Linux: /proc/meminfo, macOS: sysctl)
- [ ] Detect CPU core count
- [ ] Detect GPU VRAM via nvidia-smi (NVIDIA), rocm-smi (AMD), system_profiler (Apple)
- [ ] Infer acceleration backend (CUDA, ROCm, Metal, CPU x86, CPU ARM)
- [ ] Return `HardwareProfile` dataclass
- [ ] `python main.py system` prints detected specs

## Phase 2 — Model database
- [ ] Fetch `hf_models.json` from llmfit GitHub repo on first run
- [ ] Cache to `data/hf_models.json`
- [ ] Parse into `Model` dataclasses
- [ ] `--refresh` flag to re-fetch

## Phase 3 — Scoring engine
- [ ] Quantization memory estimation per model
- [ ] Dynamic quant selection (best quant that fits available memory)
- [ ] MoE active-expert memory reduction
- [ ] Fit level classification (perfect / good / marginal / too_tight)
- [ ] Run mode detection (gpu / cpu+gpu / cpu / moe)
- [ ] Quality score (params, quant penalty, task alignment)
- [ ] Speed estimate (bandwidth-based for known GPUs, fallback constants)
- [ ] Fit score (memory utilization efficiency)
- [ ] Context score (context window vs use-case target)
- [ ] Composite weighted score

## Phase 4 — Display
- [ ] Ranked table output (model name, score, fit, quant, est tok/s, VRAM, use case)
- [ ] JSON output (`--json`)
- [ ] Filter flags (`--top N`, `--perfect`, `--use-case coding`)
- [ ] Color output on supported terminals

## Phase 5 — Polish
- [ ] `--memory` override for manual VRAM specification
- [ ] `--max-context` cap for memory estimation
- [ ] Search by name (`python main.py search "llama 8b"`)
- [ ] Detail view for a single model (`python main.py info "Mistral-7B"`)
