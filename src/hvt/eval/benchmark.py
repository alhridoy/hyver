"""Evaluation harness for verifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

from ..orchestrator import HybridVerifier
from ..types import Verdict


@dataclass(slots=True)
class EvalExample:
    prompt: str
    candidate: str
    metadata: Mapping[str, object]
    label: bool


@dataclass(slots=True)
class EvalMetrics:
    total: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        prec = self.precision
        rec = self.recall
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


def load_jsonl_dataset(path: str | Path) -> List[EvalExample]:
    path = Path(path)
    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                EvalExample(
                    prompt=payload["prompt"],
                    candidate=payload["candidate"],
                    metadata=payload.get("metadata", {}),
                    label=bool(payload.get("label", True)),
                )
            )
    return examples


def evaluate_dataset(verifier: HybridVerifier, dataset: Sequence[EvalExample]) -> EvalMetrics:
    tp = fp = tn = fn = 0
    for example in dataset:
        result = verifier.verify(
            prompt=example.prompt,
            candidate_answer=example.candidate,
            metadata=example.metadata,
        )
        predicted_positive = result.verdict is Verdict.PASS
        if example.label and predicted_positive:
            tp += 1
        elif example.label and not predicted_positive:
            fn += 1
        elif not example.label and predicted_positive:
            fp += 1
        else:
            tn += 1
    return EvalMetrics(
        total=len(dataset),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )
