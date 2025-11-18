from __future__ import annotations

import json
from pathlib import Path

import pytest

from hvt.cli import main
from hvt.registry import clear_registry


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_cli_verify_and_eval(tmp_path, capsys):
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"reference_answer": "12"}))

    dataset_path = tmp_path / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "prompt": "Q",
                    "candidate": "12",
                    "metadata": {"reference_answer": "12"},
                    "label": True,
                }
            )
            + "\n"
        )

    assert (
        main(
            [
                "--use-builtins",
                "verify",
                "--task",
                "gsm8k_builtin",
                "--prompt",
                "Q",
                "--candidate",
                "12",
                "--metadata-file",
                str(metadata),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out.strip())
    assert output["verdict"] == "PASS"

    assert (
        main(
            [
                "--use-builtins",
                "eval",
                "--task",
                "gsm8k_builtin",
                "--dataset",
                str(dataset_path),
            ]
        )
        == 0
    )
    eval_output = json.loads(capsys.readouterr().out.strip())
    assert eval_output["precision"] == 1.0


def test_cli_synthesize(tmp_path, capsys):
    output = tmp_path / "syn.jsonl"
    assert (
        main(
            [
                "synthesize",
                "--output",
                str(output),
                "--tasks",
                "boolean_constraints",
                "word_equations",
                "--count",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out.strip())
    assert Path(payload["output"]).exists()
