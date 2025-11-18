"""Public API for the Hybrid Verifier Toolkit."""

from .registry import register_task, get_task_config
from .orchestrator import HybridVerifier
from .types import VerificationResult, Provenance
from .builtins import register_builtin_tasks
from .eval import evaluate_dataset, load_jsonl_dataset, EvalMetrics, EvalExample
from .synlogic import (
    SynLogicExample,
    SynLogicTask,
    default_tasks,
    synthesize_dataset,
    export_jsonl,
)
from .integrations import VerifierRewardAdapter, build_trl_reward_fn

__all__ = [
    "register_task",
    "get_task_config",
    "HybridVerifier",
    "VerificationResult",
    "Provenance",
    "register_builtin_tasks",
    "evaluate_dataset",
    "load_jsonl_dataset",
    "EvalMetrics",
    "EvalExample",
    "SynLogicExample",
    "SynLogicTask",
    "default_tasks",
    "synthesize_dataset",
    "export_jsonl",
    "VerifierRewardAdapter",
    "build_trl_reward_fn",
]
