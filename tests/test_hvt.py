from __future__ import annotations

import pytest

from hvt import HybridVerifier, register_task
from hvt.registry import clear_registry
from hvt.rules.math import gsm8k_exact_match
from hvt.rules.code import PythonUnitTestRule
from hvt.rules.logic import LogicSATRule
from hvt.model_verifiers import StaticJudge


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_rule_only_pass(tmp_path):
    cache_dir = tmp_path / "cache"
    register_task(name="gsm8k", rule_fn=gsm8k_exact_match, cache_dir=str(cache_dir))
    verifier = HybridVerifier(task_name="gsm8k")
    result = verifier.verify(
        prompt="A child...",
        candidate_answer="12 apples",
        metadata={"reference_answer": "12"},
    )
    assert result.verdict.name == "PASS"
    assert result.score == 1.0
    assert result.provenance.model_invoked is False


def test_model_fallback(tmp_path):
    cache_dir = tmp_path / "cache"
    register_task(
        name="gsm8k",
        rule_fn=gsm8k_exact_match,
        model_verifier=StaticJudge(confidence=0.91, verdict=True),
        thresholds={"judge_min": 0.9},
        cache_dir=str(cache_dir),
    )
    verifier = HybridVerifier(task_name="gsm8k")
    result = verifier.verify(
        prompt="A child...",
        candidate_answer="13",
        metadata={"reference_answer": "12"},
    )
    assert result.verdict.name == "PASS"
    assert result.score >= 0.9
    assert result.provenance.model_invoked is True


def test_cache_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    register_task(
        name="gsm8k",
        rule_fn=gsm8k_exact_match,
        model_verifier=StaticJudge(confidence=0.8, verdict=False),
        cache_dir=str(cache_dir),
    )
    verifier = HybridVerifier(task_name="gsm8k")
    verifier.verify(
        prompt="Q",
        candidate_answer="0",
        metadata={"reference_answer": "1"},
    )
    second = verifier.verify(
        prompt="Q",
        candidate_answer="0",
        metadata={"reference_answer": "1"},
    )
    assert second.provenance.cache_hit is True


def test_python_unit_rule(tmp_path):
    cache_dir = tmp_path / "cache"
    rule = PythonUnitTestRule(timeout=5.0)
    register_task(name="code", rule_fn=rule, cache_dir=str(cache_dir))
    verifier = HybridVerifier(task_name="code")
    candidate = """
def add(a, b):
    return a + b
"""
    tests_code = """
import unittest


class AddTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""
    result = verifier.verify(
        prompt="Write add",
        candidate_answer=candidate,
        metadata={"tests_code": tests_code},
    )
    assert result.verdict.name == "PASS"


def test_logic_rule(tmp_path):
    cache_dir = tmp_path / "cache"
    rule = LogicSATRule()
    register_task(name="logic", rule_fn=rule, cache_dir=str(cache_dir))
    verifier = HybridVerifier(task_name="logic")
    res = verifier.verify(
        prompt="Assign",
        candidate_answer="x=True y=False",
        metadata={"constraints": ["x", "Not(y)"]},
    )
    assert res.verdict.name == "PASS"
