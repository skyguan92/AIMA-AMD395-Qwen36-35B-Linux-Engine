"""Frozen image/video task-quality contracts and deterministic scoring."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence

from aima_engine.vl_reference import (
    canonical_json_sha256,
    sha256_file,
    verify_manifest_integrity,
)


REFERENCE_SCHEMA = "aima-amd395-qwen36/vl-task-quality-reference/v1"
NATIVE_SCHEMA = "aima-amd395-qwen36/native-vl-task-quality/v1"
FIXTURE_SCHEMA = "aima-amd395-qwen36/vl-task-quality-fixtures/v1"
SERVED_MODEL_SENTINEL = "${AIMA_SERVED_MODEL}"
MAX_TOKENS = 192
EOS_TOKEN_ID = 248_046
MIN_REFERENCE_CASE_SCORE_MILLIONTHS = 500_000
MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS = 850_000


TASK_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "image_central_red_circle",
        "modality": "image",
        "fixture": "image-central-red-circle.png",
        "prompt": (
            "Identify the large central shape and its color. Answer in one "
            "complete sentence and mention both facts."
        ),
        "rubric": (
            {"id": "red", "any_of": ("red",)},
            {"id": "circle", "any_of": ("circle", "round")},
        ),
    },
    {
        "case_id": "image_blue_square_left_yellow_triangle",
        "modality": "image",
        "fixture": "image-spatial-shapes.png",
        "prompt": (
            "Name both colored shapes and state which one is on the left and "
            "which one is on the right."
        ),
        "rubric": (
            {"id": "blue", "any_of": ("blue",)},
            {"id": "square", "any_of": ("square",)},
            {"id": "yellow", "any_of": ("yellow",)},
            {"id": "triangle", "any_of": ("triangle",)},
            {
                "id": "left_relation",
                "ordered_any_of": (("blue square", "square"), ("left",)),
            },
        ),
    },
    {
        "case_id": "image_count_green_circles_red_squares",
        "modality": "image",
        "fixture": "image-count-shapes.png",
        "prompt": (
            "Count the green circles and the red squares separately. State "
            "both counts clearly."
        ),
        "rubric": (
            {
                "id": "three_green_circles",
                "all_of": (
                    ("three", "3"),
                    ("green",),
                    ("circle", "circles"),
                ),
            },
            {
                "id": "two_red_squares",
                "all_of": (
                    ("two", "2"),
                    ("red",),
                    ("square", "squares"),
                ),
            },
        ),
    },
    {
        "case_id": "image_red_circle_top_right_quadrant",
        "modality": "image",
        "fixture": "image-quadrant-red-circle.png",
        "prompt": (
            "Which quadrant contains the red circle? Answer with one of the "
            "four standard quadrant names."
        ),
        "rubric": (
            {
                "id": "top_right",
                "any_of": (
                    "top-right",
                    "top right",
                    "upper-right",
                    "upper right",
                ),
            },
        ),
    },
    {
        "case_id": "image_orange_top_purple_bottom",
        "modality": "image",
        "fixture": "image-two-color-bands.png",
        "prompt": (
            "What color is the top band and what color is the bottom band? "
            "Give them in top-to-bottom order."
        ),
        "rubric": (
            {"id": "orange", "any_of": ("orange",)},
            {"id": "purple", "any_of": ("purple", "violet")},
            {
                "id": "top_to_bottom_order",
                "ordered_any_of": (("orange",), ("purple", "violet")),
            },
        ),
    },
    {
        "case_id": "image_arrow_points_right",
        "modality": "image",
        "fixture": "image-arrow-right.png",
        "prompt": (
            "Which direction does the large black arrow point? Explain the "
            "direction in one short sentence."
        ),
        "rubric": ({"id": "right", "any_of": ("right", "rightward")},),
    },
    {
        "case_id": "video_red_circle_moves_right",
        "modality": "video",
        "fixture": "video-red-circle-right.mp4",
        "prompt": (
            "Describe the red circle's motion from the beginning to the end. "
            "Name the object, color, and direction."
        ),
        "rubric": (
            {"id": "red", "any_of": ("red",)},
            {"id": "circle", "any_of": ("circle", "ball")},
            {"id": "right", "any_of": ("right", "rightward")},
        ),
    },
    {
        "case_id": "video_blue_square_moves_down",
        "modality": "video",
        "fixture": "video-blue-square-down.mp4",
        "prompt": (
            "Describe the blue square's motion from the beginning to the end. "
            "Name the object, color, and direction."
        ),
        "rubric": (
            {"id": "blue", "any_of": ("blue",)},
            {"id": "square", "any_of": ("square", "box")},
            {"id": "down", "any_of": ("down", "downward", "bottom")},
        ),
    },
    {
        "case_id": "video_circle_blue_then_green",
        "modality": "video",
        "fixture": "video-circle-blue-green.mp4",
        "prompt": (
            "How does the circle's color change over time? State the initial "
            "color and the final color in order."
        ),
        "rubric": (
            {"id": "blue", "any_of": ("blue",)},
            {"id": "green", "any_of": ("green",)},
            {
                "id": "blue_then_green",
                "ordered_any_of": (("blue",), ("green",)),
            },
        ),
    },
    {
        "case_id": "video_count_one_to_four",
        "modality": "video",
        "fixture": "video-count-one-to-four.mp4",
        "prompt": (
            "How many yellow circles are visible at the beginning and at the "
            "end, and does the count increase or decrease?"
        ),
        "rubric": (
            {
                "id": "one_at_start",
                "any_of": ("one", "1"),
            },
            {
                "id": "four_at_end",
                "any_of": ("four", "4"),
            },
            {"id": "increases", "any_of": ("increase", "increases", "grows")},
        ),
    },
    {
        "case_id": "video_shape_order_circle_square_triangle",
        "modality": "video",
        "fixture": "video-shape-order.mp4",
        "prompt": (
            "List the three colored shapes in the order they appear from "
            "first to last. Include each shape's color."
        ),
        "rubric": (
            {"id": "red_circle", "any_of": ("red circle",)},
            {"id": "blue_square", "any_of": ("blue square",)},
            {"id": "green_triangle", "any_of": ("green triangle",)},
            {
                "id": "shape_order",
                "ordered_any_of": (
                    ("red circle",),
                    ("blue square",),
                    ("green triangle",),
                ),
            },
        ),
    },
    {
        "case_id": "video_arrow_left_then_up",
        "modality": "video",
        "fixture": "video-arrow-left-up.mp4",
        "prompt": (
            "Describe how the arrow's pointing direction changes. State the "
            "first direction and the later direction in order."
        ),
        "rubric": (
            {"id": "left", "any_of": ("left", "leftward")},
            {"id": "up", "any_of": ("up", "upward", "top")},
            {
                "id": "left_then_up",
                "ordered_any_of": (("left", "leftward"), ("up", "upward")),
            },
        ),
    },
)

CASE_ORDER = tuple(case["case_id"] for case in TASK_CASES)
IMAGE_CASES = tuple(
    case["case_id"] for case in TASK_CASES if case["modality"] == "image"
)
VIDEO_CASES = tuple(
    case["case_id"] for case in TASK_CASES if case["modality"] == "video"
)


def text_part(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def rubric_contract(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the JSON representation used in sealed manifests."""

    return json.loads(json.dumps(case["rubric"]))


def build_cases(fixtures: Any, model: str) -> list[dict[str, Any]]:
    """Build the exact data-URI requests shared by reference and native runs."""

    specs: list[dict[str, Any]] = []
    for contract in TASK_CASES:
        replacements: dict[str, Any] = {}
        media = fixtures.part(
            contract["modality"],
            contract["fixture"],
            "data",
            replacements,
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [media, text_part(contract["prompt"])],
                }
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        specs.append(
            {
                "case_id": contract["case_id"],
                "modality": contract["modality"],
                "surfaces": [contract["modality"], "generation", "task_quality"],
                "expected_accept": True,
                "payload": payload,
                "replacements": replacements,
                "require_tool_call": False,
                "rubric": rubric_contract(contract),
            }
        )
    if tuple(spec["case_id"] for spec in specs) != CASE_ORDER:
        raise RuntimeError("task-quality case order changed")
    return specs


def normalize_contract_request(request: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(request))
    normalized["model"] = SERVED_MODEL_SENTINEL
    return normalized


def response_content(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return ""
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    value = choice.get("finish_reason") if isinstance(choice, dict) else None
    return value if isinstance(value, str) else None


def usage_signature(response: Any) -> tuple[int, int, int] | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    values = tuple(
        usage.get(name)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return None
    return values  # type: ignore[return-value]


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _phrase_span(
    text: str, phrase: str, start: int = 0
) -> tuple[int, int] | None:
    normalized = _normalized_text(phrase)
    if not normalized:
        return None
    pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    if normalized[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if normalized[-1].isalnum():
        pattern += r"(?!\w)"
    match = re.search(pattern, text[start:])
    if match is None:
        return None
    return start + match.start(), start + match.end()


def _criterion_passed(text: str, criterion: Mapping[str, Any]) -> bool:
    alternatives = criterion.get("any_of")
    if isinstance(alternatives, Sequence) and not isinstance(alternatives, str):
        return any(
            isinstance(phrase, str) and _phrase_span(text, phrase) is not None
            for phrase in alternatives
        )
    required_groups = criterion.get("all_of")
    if isinstance(required_groups, Sequence) and not isinstance(
        required_groups, str
    ):
        return bool(required_groups) and all(
            isinstance(group, Sequence)
            and not isinstance(group, str)
            and bool(group)
            and any(
                isinstance(phrase, str)
                and _phrase_span(text, phrase) is not None
                for phrase in group
            )
            for group in required_groups
        )
    ordered = criterion.get("ordered_any_of")
    if not isinstance(ordered, Sequence) or isinstance(ordered, str):
        return False
    if not ordered:
        return False
    cursor = 0
    for group in ordered:
        if not isinstance(group, Sequence) or isinstance(group, str):
            return False
        spans = [
            span
            for phrase in group
            if isinstance(phrase, str)
            for span in [_phrase_span(text, phrase, cursor)]
            if span is not None
        ]
        if not spans:
            return False
        cursor = min(spans, key=lambda span: span[0])[1]
    return True


def score_text(content: str, rubric: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score a response without an LLM judge or post-hoc fuzzy threshold."""

    normalized = _normalized_text(content)
    criteria: list[dict[str, Any]] = []
    for criterion in rubric:
        criterion_id = criterion.get("id")
        if not isinstance(criterion_id, str) or not criterion_id:
            raise ValueError("task-quality criterion requires an id")
        matcher_count = sum(
            name in criterion
            for name in ("any_of", "all_of", "ordered_any_of")
        )
        if matcher_count != 1:
            raise ValueError("task-quality criterion requires exactly one matcher")
        criteria.append(
            {
                "id": criterion_id,
                "passed": _criterion_passed(normalized, criterion),
            }
        )
    if not criteria:
        raise ValueError("task-quality rubric cannot be empty")
    earned = sum(item["passed"] for item in criteria)
    total = len(criteria)
    return {
        "earned_points": earned,
        "total_points": total,
        "score_millionths": earned * 1_000_000 // total,
        "criteria": criteria,
    }


def aggregate_scores(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for modality in ("image", "video"):
        scores = [
            case.get("score")
            for case in cases
            if case.get("modality") == modality
        ]
        if not scores or any(not isinstance(score, Mapping) for score in scores):
            raise ValueError(f"task-quality {modality} scores are incomplete")
        earned = sum(int(score["earned_points"]) for score in scores)
        total = sum(int(score["total_points"]) for score in scores)
        result[modality] = {
            "cases": len(scores),
            "earned_points": earned,
            "total_points": total,
            "score_millionths": earned * 1_000_000 // total,
        }
    return result


def score_not_below(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return (
        int(candidate.get("earned_points", -1))
        * int(reference.get("total_points", 0))
        >= int(reference.get("earned_points", 0))
        * int(candidate.get("total_points", 0))
        and int(candidate.get("total_points", 0)) > 0
        and int(reference.get("total_points", 0)) > 0
    )


def output_token_ids_sha256(token_ids: Sequence[int]) -> str:
    return canonical_json_sha256(list(token_ids))


def complete_output_token_ids(
    tokenized_content: Sequence[int],
    *,
    completion_tokens: int,
    finish: str | None,
) -> tuple[list[int], bool]:
    """Recover the exact generated vector from decoded content fail-closed.

    vLLM counts the terminal EOS token but omits it from decoded response text.
    All other tokenization discrepancies are rejected instead of being treated
    as token-level evidence.
    """

    if (
        not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens <= 0
        or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in tokenized_content
        )
    ):
        raise ValueError("task-quality output-token accounting is invalid")
    token_ids = list(tokenized_content)
    if len(token_ids) == completion_tokens:
        return token_ids, False
    if finish == "stop" and len(token_ids) + 1 == completion_tokens:
        token_ids.append(EOS_TOKEN_ID)
        return token_ids, True
    raise ValueError(
        "decoded output does not reconstruct the generated token count"
    )


def validate_fixture_manifest(
    payload: Mapping[str, Any], fixture_root: Path | None = None
) -> list[str]:
    errors = verify_manifest_integrity(dict(payload))
    if payload.get("schema") != FIXTURE_SCHEMA:
        errors.append("task-quality fixture schema changed")
    if payload.get("complete") is not True:
        errors.append("task-quality fixture manifest is incomplete")
    fixtures = payload.get("fixtures")
    expected_names = tuple(case["fixture"] for case in TASK_CASES)
    if not isinstance(fixtures, list) or tuple(
        record.get("path")
        for record in fixtures
        if isinstance(record, Mapping)
    ) != expected_names:
        errors.append("task-quality fixture set changed")
        return errors
    for record in fixtures:
        if not isinstance(record, Mapping):
            errors.append("task-quality fixture record is malformed")
            continue
        name = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            errors.append(f"task-quality fixture identity is invalid: {name}")
            continue
        if fixture_root is None:
            continue
        path = fixture_root / name
        if not path.is_file():
            errors.append(f"task-quality fixture is missing: {name}")
        elif path.stat().st_size != size or sha256_file(path) != digest:
            errors.append(f"task-quality fixture changed: {name}")
    return errors


def _validate_case_request(
    request: Any, contract: Mapping[str, Any], case_id: str
) -> list[str]:
    if not isinstance(request, Mapping):
        return [f"task-quality request is missing: {case_id}"]
    errors: list[str] = []
    for key, expected in (
        ("model", SERVED_MODEL_SENTINEL),
        ("temperature", 0),
        ("max_tokens", MAX_TOKENS),
        ("stream", False),
    ):
        if request.get(key) != expected:
            errors.append(f"task-quality request {key} changed: {case_id}")
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        errors.append(f"task-quality request messages changed: {case_id}")
        return errors
    message = messages[0]
    content = message.get("content") if isinstance(message, Mapping) else None
    if (
        not isinstance(message, Mapping)
        or message.get("role") != "user"
        or not isinstance(content, list)
        or len(content) != 2
    ):
        errors.append(f"task-quality request content changed: {case_id}")
        return errors
    media = content[0]
    text = content[1]
    modality = contract["modality"]
    media_value = (
        media.get(f"{modality}_url")
        if isinstance(media, Mapping)
        else None
    )
    identity = (
        media_value.get("url")
        if isinstance(media_value, Mapping)
        else None
    )
    if (
        not isinstance(media, Mapping)
        or media.get("type") != f"{modality}_url"
        or not isinstance(identity, Mapping)
        or identity.get("fixture") != contract["fixture"]
        or identity.get("transport") != "data"
    ):
        errors.append(f"task-quality request media changed: {case_id}")
    if text != text_part(contract["prompt"]):
        errors.append(f"task-quality request prompt changed: {case_id}")
    return errors


def validate_reference_manifest(payload: Mapping[str, Any]) -> list[str]:
    errors = verify_manifest_integrity(dict(payload))
    if payload.get("schema") != REFERENCE_SCHEMA:
        errors.append("task-quality reference schema changed")
    if payload.get("complete") is not True:
        errors.append("task-quality reference is incomplete")
    if payload.get("qualified_for_native_replay") is not True:
        errors.append("task-quality reference is not qualified")
    cases = payload.get("cases")
    if not isinstance(cases, list) or tuple(
        case.get("case_id") for case in cases if isinstance(case, dict)
    ) != CASE_ORDER:
        errors.append("task-quality reference case order changed")
        return errors
    contracts = {case["case_id"]: case for case in TASK_CASES}
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case["case_id"]
        contract = contracts[case_id]
        if case.get("modality") != contract["modality"]:
            errors.append(f"task-quality modality changed: {case_id}")
        if case.get("rubric") != rubric_contract(contract):
            errors.append(f"task-quality rubric changed: {case_id}")
        errors.extend(_validate_case_request(case.get("request"), contract, case_id))
        checks = case.get("qualification_checks")
        if (
            case.get("qualified") is not True
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            errors.append(f"task-quality reference case failed: {case_id}")
        if (
            case.get("status_code") != 200
            or case.get("accepted") is not True
            or case.get("passed") is not True
        ):
            errors.append(f"task-quality reference HTTP result changed: {case_id}")
        content = case.get("output_text")
        score = case.get("score")
        if not isinstance(content, str) or not content:
            errors.append(f"task-quality output is missing: {case_id}")
        elif not isinstance(score, dict) or score != score_text(
            content, contract["rubric"]
        ):
            errors.append(f"task-quality score changed: {case_id}")
        token_ids = case.get("output_token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in token_ids
        ):
            errors.append(f"task-quality output tokens are missing: {case_id}")
        elif case.get("output_token_ids_sha256") != output_token_ids_sha256(
            token_ids
        ):
            errors.append(f"task-quality output token hash changed: {case_id}")
        response = case.get("response")
        finish = finish_reason(response)
        usage = usage_signature(response)
        render = case.get("render")
        reconstruction = case.get("output_token_reconstruction")
        if content != response_content(response):
            errors.append(f"task-quality response content changed: {case_id}")
        if finish not in {"stop", "length"}:
            errors.append(f"task-quality finish reason changed: {case_id}")
        if (
            usage is None
            or not isinstance(token_ids, list)
            or usage[1] != len(token_ids)
            or usage[2] != usage[0] + usage[1]
        ):
            errors.append(f"task-quality usage changed: {case_id}")
        if (
            not isinstance(render, dict)
            or not isinstance(render.get("prompt_token_ids"), list)
            or not render["prompt_token_ids"]
            or render.get("prompt_tokens") != len(render["prompt_token_ids"])
            or render.get("prompt_token_ids_sha256")
            != canonical_json_sha256(render["prompt_token_ids"])
            or render.get("max_tokens") != MAX_TOKENS
            or usage is None
            or render.get("prompt_tokens") != usage[0]
        ):
            errors.append(f"task-quality render vector changed: {case_id}")
        appended = (
            reconstruction.get("eos_appended")
            if isinstance(reconstruction, dict)
            else None
        )
        retokenized = (
            reconstruction.get("retokenized_tokens")
            if isinstance(reconstruction, dict)
            else None
        )
        if (
            not isinstance(appended, bool)
            or not isinstance(retokenized, int)
            or isinstance(retokenized, bool)
            or reconstruction.get("method")
            != "vllm-tokenize-decoded-content-with-terminal-eos-recovery"
            or not isinstance(token_ids, list)
            or retokenized + int(appended) != len(token_ids)
            or (
                appended
                and (finish != "stop" or token_ids[-1:] != [EOS_TOKEN_ID])
            )
        ):
            errors.append(
                f"task-quality output reconstruction changed: {case_id}"
            )
        if isinstance(score, dict) and score.get("score_millionths", 0) < (
            MIN_REFERENCE_CASE_SCORE_MILLIONTHS
        ):
            errors.append(f"task-quality reference case is too weak: {case_id}")
    try:
        expected_aggregate = aggregate_scores(cases)
    except (KeyError, TypeError, ValueError):
        errors.append("task-quality reference aggregate is malformed")
    else:
        if payload.get("aggregate") != expected_aggregate:
            errors.append("task-quality reference aggregate changed")
        for modality, score in expected_aggregate.items():
            if score["score_millionths"] < MIN_REFERENCE_MODALITY_SCORE_MILLIONTHS:
                errors.append(f"task-quality {modality} reference floor failed")
    return errors
