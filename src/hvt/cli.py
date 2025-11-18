"""Command line interface for Hybrid Verifier Toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from . import HybridVerifier
from .builtins import register_builtin_tasks
from .eval import evaluate_dataset, load_jsonl_dataset
from .synlogic import default_tasks, synthesize_dataset, export_jsonl
from .types import provenance_to_dict


def _load_metadata(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_json(obj: dict) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _build_verifier(task_name: str, cache_dir: str | None = None) -> HybridVerifier:
    return HybridVerifier(task_name=task_name, cache_dir=cache_dir)


def _handle_verify(args: argparse.Namespace) -> int:
    if args.use_builtins:
        register_builtin_tasks()
    metadata = _load_metadata(args.metadata_file)
    verifier = _build_verifier(args.task, args.cache_dir)
    result = verifier.verify(prompt=args.prompt, candidate_answer=args.candidate, metadata=metadata)
    _print_json(
        {
            "verdict": result.verdict.name,
            "score": result.score,
            "provenance": provenance_to_dict(result.provenance),
        }
    )
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    if args.use_builtins:
        register_builtin_tasks()
    verifier = _build_verifier(args.task, args.cache_dir)
    dataset = load_jsonl_dataset(args.dataset)
    metrics = evaluate_dataset(verifier, dataset)
    _print_json(metrics.to_dict())
    return 0


def _handle_synthesize(args: argparse.Namespace) -> int:
    register_builtin_tasks()
    task_map = {task.name: task for task in default_tasks()}
    if args.tasks:
        missing = set(args.tasks) - task_map.keys()
        if missing:
            raise SystemExit(f"Unknown SynLogic task(s): {', '.join(sorted(missing))}")
        selected = [task_map[name] for name in args.tasks]
    else:
        selected = list(task_map.values())

    verifier_names = {"gsm8k_builtin", "logic_sat_builtin", "math_expr_builtin", "code_exec_builtin"}
    lookup = {name: _build_verifier(name, args.cache_dir) for name in verifier_names}
    dataset = synthesize_dataset(selected, lookup, per_task=args.count, seed=args.seed)
    output_path = export_jsonl(dataset, args.output)
    _print_json({"output": str(output_path), "num_examples": len(dataset)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hvt", description="Hybrid Verifier Toolkit CLI")
    parser.add_argument("--cache-dir", help="Override cache directory", default=None)
    parser.add_argument(
        "--use-builtins",
        action="store_true",
        help="Register built-in demo tasks before running the command",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify a single candidate")
    verify_parser.add_argument("--task", required=True)
    verify_parser.add_argument("--prompt", required=True)
    verify_parser.add_argument("--candidate", required=True)
    verify_parser.add_argument("--metadata-file")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation on a dataset")
    eval_parser.add_argument("--task", required=True)
    eval_parser.add_argument("--dataset", required=True)

    synth_parser = subparsers.add_parser("synthesize", help="Generate SynLogic-lite dataset")
    synth_parser.add_argument("--output", required=True)
    synth_parser.add_argument("--count", type=int, default=5)
    synth_parser.add_argument("--seed", type=int, default=42)
    synth_parser.add_argument(
        "--tasks",
        nargs="*",
        help="Subset of SynLogic tasks to run",
    )

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        return _handle_verify(args)
    if args.command == "eval":
        return _handle_eval(args)
    if args.command == "synthesize":
        return _handle_synthesize(args)
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
