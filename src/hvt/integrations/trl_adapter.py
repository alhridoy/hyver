"""TRL/GRPO adapter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

from ..orchestrator import HybridVerifier
from ..types import provenance_to_dict


@dataclass(slots=True)
class RewardRecord:
    reward: float
    metadata: Mapping[str, object]


class VerifierRewardAdapter:
    """Callable adapter turning verifier outputs into scalar rewards."""

    def __init__(
        self,
        verifier: HybridVerifier,
        *,
        reward_pass: float = 1.0,
        reward_fail: float = 0.0,
    ) -> None:
        self.verifier = verifier
        self.reward_pass = reward_pass
        self.reward_fail = reward_fail

    def __call__(
        self,
        prompts: Sequence[str],
        candidates: Sequence[str],
        metadata_list: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> List[RewardRecord]:
        if metadata_list is None:
            metadata_list = [{} for _ in prompts]
        records: List[RewardRecord] = []
        for prompt, candidate, metadata in zip(prompts, candidates, metadata_list):
            result = self.verifier.verify(prompt=prompt, candidate_answer=candidate, metadata=metadata)
            reward = self.reward_pass if result.verdict.name == "PASS" else self.reward_fail
            records.append(
                RewardRecord(
                    reward=reward,
                    metadata={
                        "score": result.score,
                        "verdict": result.verdict.name,
                        "provenance": provenance_to_dict(result.provenance),
                    },
                )
            )
        return records


def build_trl_reward_fn(adapter: VerifierRewardAdapter):
    """Return a function compatible with TRL's reward interface."""

    def reward_fn(samples: Iterable[Mapping[str, object]], **kwargs) -> List[float]:
        prompts = [sample.get("prompt", "") for sample in samples]
        responses = [sample.get("completion") or sample.get("response", "") for sample in samples]
        metadata = [sample.get("metadata", {}) for sample in samples]
        records = adapter(prompts, responses, metadata)
        return [record.reward for record in records]

    return reward_fn
