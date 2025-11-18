"""Registration helpers for built-in demo tasks."""

from __future__ import annotations

from typing import Optional

from .registry import get_task_config, register_task
from .rules import LogicSATRule, PythonUnitTestRule, gsm8k_exact_match, sympy_equivalence

_REGISTERED = False
_TASK_NAMES = [
    "gsm8k_builtin",
    "math_expr_builtin",
    "logic_sat_builtin",
    "code_exec_builtin",
]


def _task_exists(name: str) -> bool:
    try:
        get_task_config(name)
        return True
    except KeyError:
        return False


def register_builtin_tasks(cache_dir: Optional[str] = None) -> None:
    global _REGISTERED
    if _REGISTERED and all(_task_exists(name) for name in _TASK_NAMES):
        return

    register_task(
        name="gsm8k_builtin",
        rule_fn=gsm8k_exact_match,
        cache_dir=cache_dir,
    )
    register_task(
        name="math_expr_builtin",
        rule_fn=sympy_equivalence,
        cache_dir=cache_dir,
    )
    register_task(
        name="logic_sat_builtin",
        rule_fn=LogicSATRule(),
        cache_dir=cache_dir,
    )
    register_task(
        name="code_exec_builtin",
        rule_fn=PythonUnitTestRule(),
        cache_dir=cache_dir,
    )

    _REGISTERED = True
