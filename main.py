#!/usr/bin/env python3
"""llmatch — rank LLM models by how well they fit your hardware."""

import argparse
import json
import sys

import hardware
import models as model_db
import scorer
import display


def cmd_system(args):
    hw = hardware.detect()
    display.print_hardware(hw)


def cmd_rank(args):
    hw = hardware.detect()
    if args.memory:
        hw.gpu_vram_gb = args.memory

    all_models = model_db.load(refresh=args.refresh)

    scored = scorer.score_all(
        all_models,
        hw,
        max_context=args.max_context,
    )

    if args.perfect:
        scored = [s for s in scored if s.fit_level == "perfect"]
    if args.use_case:
        q = args.use_case.lower()
        scored = [s for s in scored if q in s.model.use_case.lower()]

    if args.json:
        display.print_json(scored, top=args.top)
    else:
        display.print_table(scored, top=args.top)


def cmd_search(args):
    hw = hardware.detect()
    all_models = model_db.load(refresh=args.refresh)
    scored = scorer.score_all(all_models, hw)

    query = args.query.lower()
    results = [
        s for s in scored
        if query in s.model.name.lower()
        or query in s.model.provider.lower()
        or query in s.model.architecture.lower()
        or query in s.model.use_case.lower()
    ]

    if not results:
        print(f"No models matching '{args.query}'.")
        sys.exit(1)

    if args.json:
        display.print_json(results, top=args.top)
    else:
        display.print_table(results, top=args.top)


def cmd_info(args):
    hw = hardware.detect()
    all_models = model_db.load(refresh=args.refresh)
    scored = scorer.score_all(all_models, hw)

    query = args.model_name.lower()
    matches = [s for s in scored if query in s.model.name.lower()]

    if not matches:
        print(f"No model matching '{args.model_name}'.")
        sys.exit(1)

    # Pick closest (shortest name that contains the query)
    best = min(matches, key=lambda s: len(s.model.name))
    display.print_model_info(best)


def cmd_list(args):
    all_models = model_db.load(refresh=args.refresh)
    if args.json:
        out = [{"name": m.name, "provider": m.provider, "parameters": m.parameter_count,
                "use_case": m.use_case, "architecture": m.architecture} for m in all_models]
        print(json.dumps(out, indent=2))
    else:
        print(f"{'Name':<50}  {'Params':>8}  {'Architecture':<18}  Use Case")
        print("-" * 110)
        for m in all_models:
            name = m.name if len(m.name) <= 50 else m.name[:49] + "…"
            print(f"{name:<50}  {m.parameter_count:>8}  {m.architecture:<18}  {m.use_case}")
        print(f"\n{len(all_models)} models total.")


def main():
    parser = argparse.ArgumentParser(
        prog="llmatch",
        description="Rank LLM models by how well they fit your hardware.",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-fetch model database")

    subparsers = parser.add_subparsers(dest="command")

    # system
    subparsers.add_parser("system", help="Show detected hardware specs")

    # list
    p_list = subparsers.add_parser("list", help="List all models in database")
    p_list.add_argument("--json", action="store_true")

    # search
    p_search = subparsers.add_parser("search", help="Search models by name/use-case")
    p_search.add_argument("query")
    p_search.add_argument("--top", type=int, default=None)
    p_search.add_argument("--json", action="store_true")

    # info
    p_info = subparsers.add_parser("info", help="Detailed view for a single model")
    p_info.add_argument("model_name")

    # default rank (no subcommand)
    parser.add_argument("--top", type=int, default=20, help="Show top N results (default 20)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--perfect", action="store_true", help="Show only perfect-fit models")
    parser.add_argument("--use-case", dest="use_case", metavar="CATEGORY",
                        help="Filter by use case keyword (e.g. coding, chat)")
    parser.add_argument("--memory", type=float, metavar="GB",
                        help="Override GPU VRAM detection (GB)")
    parser.add_argument("--max-context", dest="max_context", type=int, metavar="N",
                        help="Cap context length used for memory estimation")

    args = parser.parse_args()

    if args.command == "system":
        cmd_system(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        cmd_rank(args)


if __name__ == "__main__":
    main()
