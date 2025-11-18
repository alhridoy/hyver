"""Task registry for rule/model verifier configs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .types import TaskConfig, RuleFn, ModelVerifier, QuantitativeJudgeRegressor

_TASK_REGISTRY: Dict[str, TaskConfig] = {}


def register_task(
    name: str,
    *,
    rule_fn: RuleFn,
    model_verifier: Optional[ModelVerifier] = None,
    calibrator: Optional[QuantitativeJudgeRegressor] = None,
    thresholds: Optional[Dict[str, float]] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """Register a task configuration for later lookup."""

    if not name:
        raise ValueError("Task name must be non-empty")

    resolved_cache = Path(cache_dir).expanduser().resolve() if cache_dir else None

    _TASK_REGISTRY[name] = TaskConfig(
        name=name,
        rule_fn=rule_fn,
        model_verifier=model_verifier,
        calibrator=calibrator,
        thresholds=thresholds or {"judge_min": 0.8},
        cache_dir=str(resolved_cache) if resolved_cache else None,
    )


def get_task_config(name: str) -> TaskConfig:
    try:
        return _TASK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Task '{name}' is not registered") from exc


def clear_registry() -> None:
    _TASK_REGISTRY.clear()
