"""Dataset synthesis helpers for SynLogic-lite tasks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Mapping

from ..orchestrator import HybridVerifier
from .tasks import SynLogicExample, SynLogicTask


def synthesize_dataset(
    tasks: Iterable[SynLogicTask],
    verifier_lookup: Mapping[str, HybridVerifier],
    *,
    per_task: int = 10,
    seed: int = 42,
) -> List[SynLogicExample]:
    import random

    rng = random.Random(seed)
    dataset: List[SynLogicExample] = []
    for task in tasks:
        for idx in range(per_task):
            example = task.generate(rng)
            verifier = verifier_lookup.get(example.verifier_task)
            if verifier is None:
                raise KeyError(f"Verifier '{example.verifier_task}' not found in lookup")
            result = verifier.verify(
                prompt=example.prompt,
                candidate_answer=example.canonical_answer,
                metadata=example.metadata,
            )
            example.reward = 1.0 if result.verdict.name == "PASS" else 0.0
            example.extra = {
                "verdict": result.verdict.name,
                "rule": result.provenance.rule_name,
            }
            dataset.append(example)
    return dataset


def export_jsonl(examples: Iterable[SynLogicExample], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            payload = asdict(example)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def load_synlogic_dataset(path: str | Path) -> List[SynLogicExample]:
    data: List[SynLogicExample] = []
    from .tasks import SynLogicExample as Example

    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            data.append(Example(**payload))
    return data
