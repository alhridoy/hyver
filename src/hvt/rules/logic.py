"""Simple logical satisfiability checker."""

from __future__ import annotations

from typing import Mapping, Tuple

import sympy as sp


class LogicSATRule:
    def __init__(self) -> None:
        self._symbols_cache: dict[str, sp.Symbol] = {}

    def __call__(self, candidate: str, metadata: Mapping[str, object]) -> Tuple[bool, dict]:
        constraints = metadata.get("constraints")
        if not isinstance(constraints, list) or not constraints:
            raise ValueError("LogicSATRule requires a non-empty list of 'constraints'")
        try:
            assignment = self._parse_assignment(candidate)
            exprs = [sp.sympify(expr, locals=self._symbols_cache) for expr in constraints]
            result = all(bool(expr.subs(assignment)) for expr in exprs)
            serializable_assignment = {str(k): bool(v) for k, v in assignment.items()}
            return result, {"assignment": serializable_assignment}
        except Exception as exc:  # pragma: no cover - sympy parsing edge cases
            return False, {"error": str(exc)}

    def _parse_assignment(self, text: str) -> dict[sp.Symbol, bool]:
        assignment: dict[sp.Symbol, bool] = {}
        for token in text.replace(",", " ").split():
            if "=" not in token:
                continue
            name, value = token.split("=", 1)
            sym = self._symbols_cache.setdefault(name.strip(), sp.Symbol(name.strip()))
            assignment[sym] = value.strip().lower() in {"true", "1", "t"}
        if not assignment:
            raise ValueError("No variable assignments found in candidate")
        return assignment
