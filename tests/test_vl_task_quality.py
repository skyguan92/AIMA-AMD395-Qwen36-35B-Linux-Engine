from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

from aima_engine.vl_reference import seal_manifest
from aima_engine.vl_task_quality import (
    CASE_ORDER,
    EOS_TOKEN_ID,
    IMAGE_CASES,
    MAX_TOKENS,
    REFERENCE_SCHEMA,
    TASK_CASES,
    VIDEO_CASES,
    aggregate_scores,
    build_cases,
    complete_output_token_ids,
    normalize_contract_request,
    output_token_ids_sha256,
    rubric_contract,
    score_not_below,
    score_text,
    validate_reference_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate-vl-task-quality-fixtures.py"


class FakeFixtures:
    def part(self, modality, fixture, transport, replacements):
        url = f"data:{modality}/{fixture}"
        replacements[url] = {"fixture": fixture, "transport": transport}
        return {
            "type": f"{modality}_url",
            f"{modality}_url": {"url": url},
        }


def perfect_content(contract: dict) -> str:
    phrases: list[str] = []
    for criterion in contract["rubric"]:
        if "any_of" in criterion:
            phrases.append(criterion["any_of"][0])
        elif "all_of" in criterion:
            phrases.extend(group[0] for group in criterion["all_of"])
        else:
            phrases.extend(group[0] for group in criterion["ordered_any_of"])
    return ". ".join(phrases)


class VlTaskQualityTest(unittest.TestCase):
    def test_frozen_case_surface_is_balanced_and_longer_greedy(self) -> None:
        self.assertEqual(len(CASE_ORDER), 12)
        self.assertEqual(len(IMAGE_CASES), 6)
        self.assertEqual(len(VIDEO_CASES), 6)
        specs = build_cases(FakeFixtures(), "test-model")
        self.assertEqual(tuple(spec["case_id"] for spec in specs), CASE_ORDER)
        self.assertTrue(
            all(spec["payload"]["max_tokens"] == MAX_TOKENS for spec in specs)
        )
        self.assertTrue(
            all(spec["payload"]["temperature"] == 0 for spec in specs)
        )
        self.assertTrue(
            all(
                next(iter(spec["replacements"].values()))["transport"] == "data"
                for spec in specs
            )
        )

    def test_scorer_uses_word_boundaries_and_order(self) -> None:
        rubric = (
            {"id": "right", "any_of": ("right",)},
            {
                "id": "blue_then_green",
                "ordered_any_of": (("blue",), ("green",)),
            },
            {
                "id": "three_green_circles",
                "all_of": (("three", "3"), ("green",), ("circles",)),
            },
        )
        score = score_text(
            "A bright blue circle turns green on the right; three green circles remain.",
            rubric,
        )
        self.assertEqual(score["earned_points"], 3)
        boundary_score = score_text(
            "A bright blue circle turns green; three green circles remain.", rubric
        )
        self.assertEqual(boundary_score["earned_points"], 2)
        reversed_score = score_text(
            "The green circle was blue earlier and moves left.", rubric
        )
        self.assertEqual(reversed_score["earned_points"], 0)

    def test_exact_rational_comparison_does_not_use_float_tolerance(self) -> None:
        reference = {"earned_points": 5, "total_points": 6}
        self.assertTrue(
            score_not_below(
                {"earned_points": 10, "total_points": 12}, reference
            )
        )
        self.assertFalse(
            score_not_below(
                {"earned_points": 9, "total_points": 12}, reference
            )
        )

    def test_output_token_reconstruction_accepts_only_exact_or_terminal_eos(
        self,
    ) -> None:
        self.assertEqual(
            complete_output_token_ids(
                [7, 8], completion_tokens=2, finish="length"
            ),
            ([7, 8], False),
        )
        self.assertEqual(
            complete_output_token_ids(
                [7, 8], completion_tokens=3, finish="stop"
            ),
            ([7, 8, EOS_TOKEN_ID], True),
        )
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            complete_output_token_ids(
                [7], completion_tokens=3, finish="stop"
            )
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            complete_output_token_ids(
                [7, 8], completion_tokens=3, finish="length"
            )

    def test_reference_validator_recomputes_every_score_and_hash(self) -> None:
        specs = build_cases(FakeFixtures(), "test-model")
        cases = []
        for index, (contract, spec) in enumerate(zip(TASK_CASES, specs)):
            content = perfect_content(contract)
            token_ids = [index + 1, EOS_TOKEN_ID]
            prompt_ids = [100, index + 1]
            request = copy.deepcopy(spec["payload"])
            modality = contract["modality"]
            media = request["messages"][0]["content"][0]
            media[f"{modality}_url"]["url"] = next(
                iter(spec["replacements"].values())
            )
            request = normalize_contract_request(request)
            cases.append(
                {
                    "case_id": contract["case_id"],
                    "modality": contract["modality"],
                    "accepted": True,
                    "passed": True,
                    "status_code": 200,
                    "request": request,
                    "response": {
                        "choices": [
                            {
                                "message": {"content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": len(token_ids),
                            "total_tokens": len(prompt_ids) + len(token_ids),
                        },
                    },
                    "rubric": rubric_contract(contract),
                    "output_text": content,
                    "output_token_ids": token_ids,
                    "output_token_ids_sha256": output_token_ids_sha256(
                        token_ids
                    ),
                    "output_token_reconstruction": {
                        "method": (
                            "vllm-tokenize-decoded-content-with-terminal-"
                            "eos-recovery"
                        ),
                        "retokenized_tokens": len(token_ids) - 1,
                        "eos_appended": True,
                    },
                    "render": {
                        "prompt_tokens": len(prompt_ids),
                        "prompt_token_ids": prompt_ids,
                        "prompt_token_ids_sha256": output_token_ids_sha256(
                            prompt_ids
                        ),
                        "max_tokens": MAX_TOKENS,
                    },
                    "score": score_text(content, contract["rubric"]),
                    "qualification_checks": {"synthetic_complete": True},
                    "qualified": True,
                }
            )
        payload = seal_manifest(
            {
                "schema": REFERENCE_SCHEMA,
                "complete": True,
                "qualified_for_native_replay": True,
                "cases": cases,
                "aggregate": aggregate_scores(cases),
            }
        )
        self.assertEqual(validate_reference_manifest(payload), [])
        payload["cases"][0]["score"]["earned_points"] -= 1
        self.assertIn(
            "task-quality score changed: image_central_red_circle",
            validate_reference_manifest(payload),
        )

    def test_fixture_generator_owns_exact_contract_names(self) -> None:
        source = (
            ROOT / "scripts/generate-vl-task-quality-fixtures.py"
        ).read_text(encoding="utf-8")
        for contract in TASK_CASES:
            self.assertIn(contract["fixture"], source)

    def test_video_up_arrow_stays_inside_non_square_canvas(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "vl_task_quality_fixture_generator_test", GENERATOR
        )
        if spec is None or spec.loader is None:
            self.fail("cannot import task-quality fixture generator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        points = module.arrow_points(module.VIDEO_SIZE, "up")
        width, height = module.VIDEO_SIZE
        self.assertTrue(all(0 <= x < width and 0 <= y < height for x, y in points))
        self.assertLess(min(y for _, y in points), height // 4)


if __name__ == "__main__":
    unittest.main()
