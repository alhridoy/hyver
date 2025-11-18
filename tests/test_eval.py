from __future__ import annotations

import json

import pytest

from hvt import HybridVerifier
from hvt.eval import EvalExample, evaluate_dataset, load_jsonl_dataset
from hvt.registry import clear_registry, register_task
from hvt.rules.math import gsm8k_exact_match


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_evaluation_metrics(tmp_path):
    register_task(
        name="gsm8k_eval",
        rule_fn=gsm8k_exact_match,
    )
    verifier = HybridVerifier(task_name="gsm8k_eval")
    dataset = [
        EvalExample(prompt="Q", candidate="12", metadata={"reference_answer": "12"}, label=True),
        EvalExample(prompt="Q", candidate="11", metadata={"reference_answer": "12"}, label=True),
        EvalExample(prompt="Q", candidate="3", metadata={"reference_answer": "9"}, label=False),
    ]
    metrics = evaluate_dataset(verifier, dataset)
    assert metrics.true_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 0

    dataset_path = tmp_path / "data.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for example in dataset:
            fh.write(
                json.dumps(
                    {
                        "prompt": example.prompt,
                        "candidate": example.candidate,
                        "metadata": example.metadata,
                        "label": example.label,
                    }
                )
                + "\n"
            )

    loaded = load_jsonl_dataset(dataset_path)
    assert len(loaded) == len(dataset)
    assert loaded[0].metadata["reference_answer"] == "12"
