"""LiteLLM-powered verifier."""

from __future__ import annotations

from typing import Mapping, Optional

from ..types import ModelVerifier


try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    litellm = None  # type: ignore


class LiteLLMJudge(ModelVerifier):
    def __init__(
        self,
        model: str,
        *,
        system_prompt: str = "You are a strict verifier that outputs a score between 0 and 1.",
        user_template: str | None = None,
    ) -> None:
        if litellm is None:
            raise RuntimeError("litellm is not installed; install with `pip install litellm`.")
        self.model_name = model
        self.system_prompt = system_prompt
        self.user_template = user_template or "Prompt: {prompt}\nCandidate: {candidate}\nScore between 0 and 1:"

    def score(self, prompt: str, candidate: str, metadata: Optional[Mapping[str, object]] = None) -> float:
        completion = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_template.format(prompt=prompt, candidate=candidate),
                },
            ],
        )
        text = completion.choices[0].message["content"].strip()
        try:
            return max(0.0, min(1.0, float(text.split()[0])))
        except (ValueError, IndexError):
            return 0.0
