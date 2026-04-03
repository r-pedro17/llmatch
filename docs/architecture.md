# Architecture

## Overview

llmatch is a single-process CLI tool. On each run it detects hardware, loads the model database, scores every model, and prints a ranked table.

## Data flow

```
main.py
  |
  |-- hardware.py      --> HardwareProfile (ram_gb, cpu_cores, gpu_vram_gb, backend)
  |-- models.py        --> list[Model] (from data/hf_models.json)
  |-- scorer.py        --> list[ScoredModel] (score, fit_level, best_quant, est_tps)
  |-- display.py       --> stdout table or JSON
```

## Modules

### hardware.py
Detects system specs and returns a `HardwareProfile` dataclass.
- RAM: Linux via `/proc/meminfo`, Windows via `ctypes` / `GlobalMemoryStatusEx`, macOS via `sysctl`
- CPU cores: `os.cpu_count()`
- GPU VRAM: subprocess call to `nvidia-smi`, `rocm-smi`, or `system_profiler` (macOS)
- Backend: inferred from available GPU tooling (CUDA, ROCm, Metal, CPU)

### models.py
Loads `data/hf_models.json`. On first run, fetches the file from the llmfit GitHub repo and caches it locally. Returns a list of `Model` dataclasses.

### scorer.py
Scores each model against the detected hardware across four dimensions (0-100 each):

| Dimension | Description |
|---|---|
| Quality | Param count, quantization penalty, task alignment |
| Speed | Estimated tok/s from backend + params + quant |
| Fit | Memory utilization efficiency (sweet spot 50-80%) |
| Context | Context window vs use-case target |

Fit levels: `perfect`, `good`, `marginal`, `too_tight`
Run modes: `gpu`, `cpu+gpu`, `cpu`, `moe`

Quantization hierarchy (best to worst): Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K

### display.py
Renders the scored model list as a formatted table to stdout, or as JSON with `--json`.

## Caching
`data/hf_models.json` is fetched on first run and auto-refreshed every 7 days. Pass `--refresh` to force re-fetch.
