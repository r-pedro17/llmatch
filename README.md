# llmatch

Detect your hardware. Rank every local LLM by how well it fits.

Python reimplementation of [llmfit](https://github.com/AlexsJones/llmfit) — no Rust, no dependencies, pure stdlib.

```
$ python main.py
   #  Model                                   Score  Fit        Quant     tok/s  Mem GB  Use Case
----------------------------------------------------------------------------------------------------
   1  meta-llama/Llama-3.1-8B-Instruct        74.2  perfect    Q4_K_M    87.5     4.30  chat
   2  Qwen/Qwen2.5-7B-Instruct                73.8  perfect    Q4_K_M    93.2     3.90  chat
   3  mistralai/Mistral-7B-Instruct-v0.3      71.1  perfect    Q4_K_M    96.0     3.80  chat
  ...
```

## Requirements

- Python 3.12+
- No pip installs

## Usage

```bash
python main.py                        # ranked table (top 20)
python main.py --top 5                # top 5 only
python main.py --perfect              # only models with perfect fit
python main.py --use-case coding      # filter by use case
python main.py --memory 24            # override GPU VRAM (GB)
python main.py --max-context 8192     # cap context length for scoring
python main.py --json                 # JSON output

python main.py system                 # show detected hardware
python main.py list                   # list all models in database
python main.py search "llama 8b"      # search by name / provider / use case
python main.py info "Llama-3.1-8B"    # detailed view for one model

python main.py --refresh              # re-fetch model database from upstream
```

## Scoring

Each model is scored across four dimensions (0–100):

| Dimension | What it measures |
|-----------|-----------------|
| Quality   | Parameter count + quantization penalty |
| Speed     | Estimated tok/s based on backend bandwidth |
| Fit       | Memory utilization efficiency (sweet spot 50–80%) |
| Context   | Context window size |

Weights: Quality 30% · Speed 25% · Fit 30% · Context 15%

Fit levels: `perfect` · `good` · `marginal` · `too_tight`

Quantization hierarchy (best → smallest): Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K

The tool picks the best quantization that fits your available memory.

## Hardware detection

| Backend | Detection method |
|---------|-----------------|
| CUDA    | `nvidia-smi` |
| ROCm    | `rocm-smi` |
| Metal   | `system_profiler` (macOS) |
| CPU     | fallback, uses 60% of RAM |

MoE models (e.g. Mixtral, DeepSeek, Qwen3-MoE) use active-expert memory estimation.

## Model database

963+ models from HuggingFace, sourced from the llmfit repo. Automatically synced weekly via GitHub Actions. Run `--refresh` to update manually at any time.

## License

MIT
