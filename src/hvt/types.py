"""Shared dataclasses and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Mapping, Optional


class Verdict(Enum):
    """Verification verdict categories."""

    PASS = auto()
    FAIL = auto()
    UNKNOWN = auto()


@dataclass(slots=True)
class Provenance:
    """Trace metadata for every verification decision."""

    task_name: str
    rule_name: str
    rule_passed: bool
    model_name: Optional[str]
    model_invoked: bool
    model_confidence: Optional[float]
    cache_hit: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    """Full verification output."""

    verdict: Verdict
    score: float
    provenance: Provenance
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def provenance_to_dict(provenance: Provenance) -> Dict[str, Any]:
    return {
        "task_name": provenance.task_name,
        "rule_name": provenance.rule_name,
        "rule_passed": provenance.rule_passed,
        "model_name": provenance.model_name,
        "model_invoked": provenance.model_invoked,
        "model_confidence": provenance.model_confidence,
        "cache_hit": provenance.cache_hit,
        "timestamp": provenance.timestamp.isoformat(),
        "extra": provenance.extra,
    }


def provenance_from_dict(payload: Mapping[str, Any]) -> Provenance:
    return Provenance(
        task_name=str(payload["task_name"]),
        rule_name=str(payload["rule_name"]),
        rule_passed=bool(payload["rule_passed"]),
        model_name=payload.get("model_name"),
        model_invoked=bool(payload["model_invoked"]),
        model_confidence=payload.get("model_confidence"),
        cache_hit=bool(payload.get("cache_hit", False)),
        timestamp=datetime.fromisoformat(payload["timestamp"]),
        extra=dict(payload.get("extra", {})),
    )


RuleFn = Callable[[str, Mapping[str, Any]], tuple[bool, Dict[str, Any]]]


@dataclass(slots=True)
class TaskConfig:
    """Configuration for a registered verification task."""

    name: str
    rule_fn: RuleFn
    model_verifier: Optional["ModelVerifier"] = None
    calibrator: Optional["QuantitativeJudgeRegressor"] = None
    thresholds: Dict[str, float] = field(default_factory=lambda: {"judge_min": 0.0})
    cache_dir: Optional[str] = None


class ModelVerifier:
    """Protocol-like base class for LLM judges."""

    model_name: str = "unknown"

    def score(self, prompt: str, candidate: str, metadata: Optional[Mapping[str, Any]] = None) -> float:
        raise NotImplementedError


class QuantitativeJudgeRegressor:
    """Protocol for calibration models."""

    def predict(self, judge_score: float, features: Mapping[str, Any]) -> float:
        raise NotImplementedError


CalibratorFactory = Callable[[Mapping[str, Any]], QuantitativeJudgeRegressor]
