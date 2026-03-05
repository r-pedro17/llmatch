# CLAUDE.md

llmatch

## Goal

Python reimplementation of [llmfit](https://github.com/AlexsJones/llmfit).
Detects local hardware and ranks LLM models by how well they fit your system using python.

## Settings
No admin priviledges
tools location: C:\Users\RPedro\tools

## Tools
GH CLI

## Stack
- Python 3.12, stdlib only (no pip dependencies)
- Model database: `data/hf_models.json` sourced from llmfit repo

## Project structure
```
main.py           # CLI entry point
hardware.py       # RAM, CPU, GPU detection
models.py         # Load and parse hf_models.json
scorer.py         # Fit/quality/speed/context scoring
display.py        # Table output
data/
  hf_models.json  # Model database (fetched from llmfit repo)
docs/
  architecture.md
  plan.md
  features.json
```

## How to run
```bash
python main.py           # ranked table
python main.py --json    # JSON output
python main.py --top 5   # top N models
```

## Key conventions
- No external dependencies — stdlib only
- GPU detection via nvidia-smi subprocess call
- Model data fetched from llmfit GitHub on first run, cached locally
- Scoring logic mirrors llmfit: Quality, Speed, Fit, Context dimensions