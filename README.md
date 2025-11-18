 # Hybrid Verifier Toolkit (HVT)

Hybrid Verifier Toolkit provides a reusable, auditable rule-first verification stack with lazy LLM fallbacks, designed for synthetic data generation and verifiable RL. The MVP ships with math/code/logic rule checkers, pluggable model-judge adapters, provenance logging, caching, and adversarial stress tests.

## Key modules

- `hvt.registry`: task registration API
- `hvt.rules`: ready-made rule verifiers (GSM8K normalization, SymPy math equivalence, code sandbox, logic SAT)
- `hvt.model_verifiers`: adapters for external LLM judges (LiteLLM/OpenAI/local) plus offline mocks
- `hvt.orchestrator`: rule-first → lazy LLM fallback, caching, provenance, metrics
- `hvt.calibration`: quantitative judge regressors trained on small human calibration sets
- `hvt.eval`: adversarial suites and precision/recall dashboards

## Quick usage

```python
from hvt import HybridVerifier, register_task
from hvt.rules.math import gsm8k_exact_match
from hvt.model_verifiers.mock import StaticJudge

register_task(
    name="gsm8k",
    rule_fn=gsm8k_exact_match,
    model_verifier=StaticJudge(confidence=0.95),
)

verifier = HybridVerifier(task_name="gsm8k")
result = verifier.verify(prompt="A child has 10 apples...", candidate_answer="12")
print(result.verdict, result.provenance.rule_passed, result.score)
```

Run `pytest` for the unit tests and `ruff check .` for linting.
