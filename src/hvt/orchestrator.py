"""Hybrid verifier orchestrator."""

from __future__ import annotations

from typing import Mapping, Optional

from .cache import VerificationCache
from .registry import get_task_config
from .types import Provenance, VerificationResult, Verdict


class HybridVerifier:
    def __init__(self, *, task_name: str, cache_dir: Optional[str] = None) -> None:
        self.config = get_task_config(task_name)
        cache_location = cache_dir or self.config.cache_dir
        self.cache = VerificationCache(cache_location)

    def verify(
        self,
        *,
        prompt: str,
        candidate_answer: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> VerificationResult:
        metadata = metadata or {}
        cached = self.cache.get(self.config.name, prompt, candidate_answer)
        if cached:
            cached.provenance.cache_hit = True
            return cached

        rule_passed, rule_diag = self.config.rule_fn(candidate_answer, metadata)
        rule_name = getattr(self.config.rule_fn, "__name__", self.config.rule_fn.__class__.__name__)
        provenance = Provenance(
            task_name=self.config.name,
            rule_name=rule_name,
            rule_passed=rule_passed,
            model_name=self.config.model_verifier.model_name if self.config.model_verifier else None,
            model_invoked=False,
            model_confidence=None,
            cache_hit=False,
        )

        if rule_passed:
            result = VerificationResult(
                verdict=Verdict.PASS,
                score=1.0,
                provenance=provenance,
                diagnostics={"rule": rule_diag},
            )
            self.cache.set(self.config.name, prompt, candidate_answer, result)
            return result

        if not self.config.model_verifier:
            result = VerificationResult(
                verdict=Verdict.FAIL,
                score=0.0,
                provenance=provenance,
                diagnostics={"rule": rule_diag},
            )
            self.cache.set(self.config.name, prompt, candidate_answer, result)
            return result

        judge_score = self.config.model_verifier.score(prompt, candidate_answer, metadata)
        provenance.model_invoked = True
        provenance.model_confidence = judge_score
        calibrated_score = self._calibrate(judge_score, prompt, candidate_answer, metadata)
        verdict = Verdict.PASS if calibrated_score >= self.config.thresholds.get("judge_min", 0.8) else Verdict.FAIL

        result = VerificationResult(
            verdict=verdict,
            score=calibrated_score,
            provenance=provenance,
            diagnostics={"rule": rule_diag, "judge_score": judge_score},
        )
        self.cache.set(self.config.name, prompt, candidate_answer, result)
        return result

    def _calibrate(
        self,
        raw_score: float,
        prompt: str,
        candidate: str,
        metadata: Mapping[str, object],
    ) -> float:
        if not self.config.calibrator:
            return raw_score
        features = {
            "prompt_length": len(prompt.split()),
            "candidate_length": len(candidate.split()),
        }
        calibrated = self.config.calibrator.predict(raw_score, features)
        return max(0.0, min(1.0, calibrated))
