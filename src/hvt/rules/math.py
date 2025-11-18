"""Math-oriented rule verifiers."""

from __future__ import annotations

import re
from typing import Mapping, Tuple

import sympy as sp


NUMERIC_RE = re.compile(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?")


def _extract_number(text: str) -> str:
    match = NUMERIC_RE.search(text.replace(",", ""))
    return match.group(0) if match else text.strip()


def gsm8k_exact_match(candidate: str, metadata: Mapping[str, object]) -> Tuple[bool, dict]:
    reference = str(metadata.get("reference_answer", "")).strip()
    if not reference:
        raise ValueError("GSM8K rule requires 'reference_answer' in metadata")
    cand = _extract_number(candidate)
    ref = _extract_number(reference)
    result = cand == ref
    return result, {"candidate_normalized": cand, "reference_normalized": ref}


def sympy_equivalence(candidate: str, metadata: Mapping[str, object]) -> Tuple[bool, dict]:
    reference = str(metadata.get("reference_expression", "")).strip()
    if not reference:
        raise ValueError("SymPy equivalence rule requires 'reference_expression'")
    try:
        cand_expr = sp.simplify(candidate)
        ref_expr = sp.simplify(reference)
        diff = sp.simplify(cand_expr - ref_expr)
        result = diff == 0
    except Exception as exc:  # pragma: no cover - sympy edge cases
        return False, {"error": str(exc)}
    return result, {
        "candidate_simplified": sp.sstr(cand_expr),
        "reference_simplified": sp.sstr(ref_expr),
        "difference": sp.sstr(diff),
    }
