#!/usr/bin/env python3
"""Compare the native tokenizer with the qualified Transformers tokenizer."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any


TEXT_CASES = [
    "",
    "hello",
    " hello",
    "Hello, world!",
    "I'm testing contractions; WE'LL preserve them.",
    "  leading and trailing whitespace  ",
    "tabs\tand\nnewlines\r\n",
    "你好，世界。AMD395 上的 Qwen 推理。",
    "日本語と한국어를 함께 테스트합니다.",
    "العربية हिन्दी Русский Ελληνικά",
    "e\u0301 == é; A\u030a == Å",
    "emoji: 😀 🚀 👩\u200d💻 🏳️\u200d🌈",
    "def f(x: int) -> str:\n    return f\"value={x:#x}\"",
    "<|im_start|>system\nexact<|im_end|>\n",
    "<think>reason</think><tool_call></tool_call>",
    "1234567890 3.1415926535 -0xDEADBEEF",
    "\n\n\nparagraph one\n\nparagraph two\n",
]

CHAT_CASES = [
    {
        "system": "Answer directly. Preserve exact identifiers, numbers, and code. Do not include analysis.",
        "user": "Hello",
        "disable_thinking": False,
    },
    {
        "system": "",
        "user": "Explain prefix caching in one sentence.",
        "disable_thinking": False,
    },
    {
        "system": "  Keep identifiers exact.  ",
        "user": "  What is gfx1151?  ",
        "disable_thinking": True,
    },
    {
        "system": "请直接回答。",
        "user": "解释 AMD395 上的前缀缓存。",
        "disable_thinking": True,
    },
]


def generated_text_cases() -> list[str]:
    randomizer = random.Random(20260721)
    alphabet = [
        "a", "Z", "é", "e\u0301", "中", "界", "日", "한", "ع", "ह", "Ж",
        "😀", "👩\u200d💻", " ", "  ", "\t", "\n", "\r\n", "'s", "'RE",
        "123", "_", "-", ".", ",", "!", "?", "<think>", "<|im_end|>",
    ]
    cases = []
    for _ in range(64):
        cases.append(
            "".join(
                randomizer.choice(alphabet)
                for _ in range(randomizer.randint(1, 24))
            )
        )
    return cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_native(binary: Path, arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"native tokenizer failed ({result.returncode}): {result.stderr.strip()}"
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or value.get("complete") is not True:
        raise RuntimeError("native tokenizer returned an incomplete report")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.native_binary.resolve()
    model_dir = args.model_dir.resolve()
    if not binary.is_file() or not model_dir.is_dir():
        raise SystemExit("native binary or model directory is missing")

    try:
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("qualified Transformers/tokenizers reference is required") from error

    reference = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=True
    )
    rows: list[dict[str, Any]] = []
    text_cases = [*TEXT_CASES, *generated_text_cases()]
    for index, text in enumerate(text_cases):
        expected_ids = [int(item) for item in reference.encode(text, add_special_tokens=False)]
        expected_decoded = reference.decode(expected_ids, skip_special_tokens=False)
        native = run_native(
            binary,
            ["tokenizer-probe", "--model-dir", str(model_dir), "--text", text],
        )
        ids_equal = native["token_ids"] == expected_ids
        decoded_equal = native["decoded"] == expected_decoded
        rows.append(
            {
                "kind": "text",
                "index": index,
                "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_count": len(expected_ids),
                "ids_equal": ids_equal,
                "decoded_equal": decoded_equal,
                "native_load_ms": native["load_ms"],
                "native_encode_ms": native["encode_ms"],
            }
        )
        if not ids_equal or not decoded_equal:
            raise RuntimeError(
                f"text case {index} mismatch: expected={expected_ids} native={native['token_ids']}"
            )

    for index, case in enumerate(CHAT_CASES):
        messages = []
        if case["system"]:
            messages.append({"role": "system", "content": case["system"]})
        messages.append({"role": "user", "content": case["user"]})
        expected_text = reference.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=not case["disable_thinking"],
        )
        expected_ids = [
            int(item)
            for item in reference.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=not case["disable_thinking"],
            )
        ]
        command = [
            "chat-template-probe",
            "--model-dir",
            str(model_dir),
            "--user",
            case["user"],
        ]
        if case["system"]:
            command.extend(["--system", case["system"]])
        if case["disable_thinking"]:
            command.append("--disable-thinking")
        native = run_native(binary, command)
        text_equal = native["prompt_text"] == expected_text
        ids_equal = native["token_ids"] == expected_ids
        rows.append(
            {
                "kind": "chat_template",
                "index": index,
                "disable_thinking": case["disable_thinking"],
                "prompt_sha256": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
                "token_count": len(expected_ids),
                "text_equal": text_equal,
                "ids_equal": ids_equal,
                "native_load_ms": native["load_ms"],
                "native_encode_ms": native["encode_ms"],
            }
        )
        if not text_equal or not ids_equal:
            raise RuntimeError(
                f"chat case {index} mismatch: text_equal={text_equal} ids_equal={ids_equal}"
            )

    report = {
        "schema": "aima-amd395-qwen36/native-tokenizer-parity/v1",
        "complete": True,
        "reference": {
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(
                model_dir / "tokenizer_config.json"
            ),
        },
        "native_binary_sha256": sha256_file(binary),
        "case_count": len(rows),
        "text_case_count": len(text_cases),
        "chat_case_count": len(CHAT_CASES),
        "all_token_ids_equal": all(row["ids_equal"] for row in rows),
        "all_text_decodes_or_templates_equal": all(
            row.get("decoded_equal", row.get("text_equal", False)) for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "complete", "case_count", "all_token_ids_equal",
        "all_text_decodes_or_templates_equal", "native_binary_sha256"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
