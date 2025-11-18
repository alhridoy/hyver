from __future__ import annotations

import pytest

from hvt import HybridVerifier
from hvt.integrations import VerifierRewardAdapter, build_trl_reward_fn
from hvt.registry import clear_registry, register_task
from hvt.rules.math import gsm8k_exact_match


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_trl_adapter():
    register_task(
        name="trl_gsm8k",
        rule_fn=gsm8k_exact_match,
    )
    verifier = HybridVerifier(task_name="trl_gsm8k")
    adapter = VerifierRewardAdapter(verifier)
    records = adapter(["Q"], ["12"], [{"reference_answer": "12"}])
    assert records[0].reward == 1.0

    reward_fn = build_trl_reward_fn(adapter)
    rewards = reward_fn([
        {
            "prompt": "Q",
            "completion": "11",
            "metadata": {"reference_answer": "12"},
        }
    ])
    assert rewards == [0.0]
