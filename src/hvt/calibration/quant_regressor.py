"""Quantitative judge regression calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

import numpy as np

from ..types import QuantitativeJudgeRegressor


@dataclass(frozen=True)
class CalibrationExample:
    judge_score: float
    human_score: float
    features: Mapping[str, float]


class QuantitativeJudgeRegressorImpl(QuantitativeJudgeRegressor):
    def __init__(self, weights: np.ndarray, bias: float, feature_order: List[str]):
        self.weights = weights
        self.bias = bias
        self.feature_order = feature_order

    @classmethod
    def train(
        cls,
        examples: Iterable[CalibrationExample],
        *,
        l2: float = 0.1,
    ) -> "QuantitativeJudgeRegressorImpl":
        data = list(examples)
        if not data:
            raise ValueError("Need at least one calibration example")

        feature_names = sorted({name for ex in data for name in ex.features.keys()})
        X = []
        y = []
        for ex in data:
            X.append([ex.features.get(name, 0.0) for name in feature_names])
            y.append(ex.human_score)

        X_mat = np.array(X, dtype=float)
        y_vec = np.array(y, dtype=float)
        gram = X_mat.T @ X_mat + l2 * np.eye(len(feature_names))
        weights = np.linalg.solve(gram, X_mat.T @ y_vec)
        bias = float(np.mean(y_vec - X_mat @ weights))
        return cls(weights=weights, bias=bias, feature_order=feature_names)

    def predict(self, judge_score: float, features: Mapping[str, float]) -> float:
        vector = np.array([features.get(name, 0.0) for name in self.feature_order])
        return float(vector @ self.weights + self.bias)
