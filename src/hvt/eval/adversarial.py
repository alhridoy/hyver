"""Adversarial verifier suite."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..orchestrator import HybridVerifier
from ..types import Verdict


def run_false_positive_suite(
    verifier: HybridVerifier,
    *,
    prompt: str,
    adversarial_samples: Sequence[str],
    metadata: Mapping[str, object],
) -> dict:
    """Return statistics on false positives for adversarial candidates."""

    total = len(adversarial_samples)
    fp = 0
    for sample in adversarial_samples:
        result = verifier.verify(prompt=prompt, candidate_answer=sample, metadata=metadata)
        if result.verdict is Verdict.PASS:
            fp += 1
    return {"false_positive_rate": fp / total if total else 0.0, "total": total}
