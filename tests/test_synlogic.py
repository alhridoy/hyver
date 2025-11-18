from __future__ import annotations

import pytest

from hvt import HybridVerifier
from hvt.registry import clear_registry, register_task
from hvt.rules import LogicSATRule, gsm8k_exact_match, sympy_equivalence
from hvt.synlogic import default_tasks, export_jsonl, load_synlogic_dataset, synthesize_dataset


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_synlogic_export(tmp_path):
    register_task(name="logic_sat_builtin", rule_fn=LogicSATRule())
    register_task(name="gsm8k_builtin", rule_fn=gsm8k_exact_match)
    register_task(name="math_expr_builtin", rule_fn=sympy_equivalence)

    lookup = {
        name: HybridVerifier(task_name=name)
        for name in ("logic_sat_builtin", "gsm8k_builtin", "math_expr_builtin")
    }
    dataset = synthesize_dataset(default_tasks(), lookup, per_task=2, seed=7)
    output_path = tmp_path / "synlogic.jsonl"
    export_jsonl(dataset, output_path)
    loaded = load_synlogic_dataset(output_path)
    assert len(loaded) == len(dataset)
    assert all(example.reward in (0.0, 1.0) for example in loaded)
