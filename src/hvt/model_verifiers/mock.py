"""Offline-friendly verifier mocks."""

from __future__ import annotations

import re
from typing import Mapping, Optional

from ..types import ModelVerifier


class StaticJudge(ModelVerifier):
    def __init__(self, confidence: float = 1.0, verdict: bool = True, model_name: str = "static-judge") -> None:
        self._confidence = confidence
        self._verdict = verdict
        self.model_name = model_name

    def score(self, prompt: str, candidate: str, metadata: Optional[Mapping[str, object]] = None) -> float:
        return self._confidence if self._verdict else 1.0 - self._confidence


class RegexJudge(ModelVerifier):
    """Judge answers correct if they match a regex."""

    def __init__(self, pattern: str, weight: float = 0.95, model_name: str = "regex-judge") -> None:
        self._pattern = re.compile(pattern)
        self._weight = weight
        self.model_name = model_name

    def score(self, prompt: str, candidate: str, metadata: Optional[Mapping[str, object]] = None) -> float:
        return self._weight if self._pattern.fullmatch(candidate.strip()) else 0.0
