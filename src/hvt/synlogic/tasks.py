"""SynLogic-lite task generators."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import sympy as sp


@dataclass(slots=True)
class SynLogicExample:
    task_name: str
    prompt: str
    canonical_answer: str
    metadata: Dict[str, object]
    verifier_task: str
    difficulty: str
    reward: float = 1.0
    extra: Dict[str, object] = field(default_factory=dict)


class SynLogicTask(Protocol):
    name: str

    def generate(self, rng: random.Random, difficulty: str = "medium") -> SynLogicExample:
        ...


class BooleanConstraintTask:
    """Generate logic assignments that satisfy boolean constraints."""

    name = "boolean_constraints"

    def generate(self, rng: random.Random, difficulty: str = "medium") -> SynLogicExample:
        var_count = 3 if difficulty == "easy" else 4
        variables = rng.sample(list("xyzuvw"), var_count)
        assignment = {var: bool(rng.randint(0, 1)) for var in variables}
        constraints = []
        for var, value in assignment.items():
            constraints.append(var if value else f"Not({var})")
        # Add one composite constraint for diversity
        a, b = rng.sample(variables, 2)
        constraints.append(f"Or({a}, {b})")
        prompt = (
            "Assign boolean values to the variables so that all constraints hold. "
            "Respond as 'x=True y=False ...'."
        )
        answer = " ".join(f"{var}={'True' if val else 'False'}" for var, val in assignment.items())
        metadata = {"constraints": constraints}
        return SynLogicExample(
            task_name=self.name,
            prompt=prompt,
            canonical_answer=answer,
            metadata=metadata,
            verifier_task="logic_sat_builtin",
            difficulty=difficulty,
        )


class WordEquationTask:
    """Small arithmetic story problems tested via GSM8K-style rule."""

    name = "word_equations"

    def generate(self, rng: random.Random, difficulty: str = "medium") -> SynLogicExample:
        base = rng.randint(3, 10)
        gain = rng.randint(2, 6)
        give = rng.randint(1, 3)
        prompt = (
            f"Alex has {base} marbles, wins {gain} more in a game, and gives {give} to a friend. "
            "How many marbles does Alex have now?"
        )
        answer = str(base + gain - give)
        metadata = {"reference_answer": answer}
        return SynLogicExample(
            task_name=self.name,
            prompt=prompt,
            canonical_answer=answer,
            metadata=metadata,
            verifier_task="gsm8k_builtin",
            difficulty=difficulty,
        )


class ExpressionSimplifyTask:
    """Symbolic equivalence puzzles checked with SymPy."""

    name = "expression_simplify"

    def generate(self, rng: random.Random, difficulty: str = "medium") -> SynLogicExample:
        x = sp.symbols("x")
        coeff = rng.randint(2, 5)
        expr = coeff * x + coeff * x
        extra = rng.randint(1, 4)
        combined = expr + extra - extra
        simplified = sp.simplify(combined)
        prompt = f"Simplify the expression: {sp.sstr(combined)}"
        answer = sp.sstr(simplified)
        metadata = {"reference_expression": answer}
        return SynLogicExample(
            task_name=self.name,
            prompt=prompt,
            canonical_answer=answer,
            metadata=metadata,
            verifier_task="math_expr_builtin",
            difficulty=difficulty,
        )


def default_tasks() -> List[SynLogicTask]:
    return [BooleanConstraintTask(), WordEquationTask(), ExpressionSimplifyTask()]
