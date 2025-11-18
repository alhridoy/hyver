"""Simple file-based cache for verification results."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Optional

from .types import VerificationResult, Verdict, provenance_from_dict, provenance_to_dict


class VerificationCache:
    def __init__(self, cache_dir: Optional[str]) -> None:
        self.enabled = bool(cache_dir)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, task_name: str, prompt: str, candidate: str) -> str:
        h = sha256()
        h.update(task_name.encode())
        h.update(b"\x00")
        h.update(prompt.strip().encode())
        h.update(b"\x00")
        h.update(candidate.strip().encode())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"

    def get(self, task_name: str, prompt: str, candidate: str) -> Optional[VerificationResult]:
        if not self.enabled:
            return None
        path = self._path(self._key(task_name, prompt, candidate))
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        provenance = provenance_from_dict(payload["provenance"])
        return VerificationResult(
            verdict=Verdict[payload["verdict"]],
            score=float(payload["score"]),
            provenance=provenance,
            diagnostics=payload.get("diagnostics", {}),
        )

    def set(self, task_name: str, prompt: str, candidate: str, result: VerificationResult) -> None:
        if not self.enabled:
            return
        path = self._path(self._key(task_name, prompt, candidate))
        payload = {
            "verdict": result.verdict.name,
            "score": result.score,
            "provenance": provenance_to_dict(result.provenance),
            "diagnostics": result.diagnostics,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False))
