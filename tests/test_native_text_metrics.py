from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from native_text_metrics import text_path_idle_checks, text_path_is_idle


def text_metrics() -> dict[str, object]:
    return {
        "mrope": {
            "enabled": False,
            "position_delta": 0,
            "position_upload_bytes": 0,
            "full_attention_launches": 0,
            "fmha_launches": 10,
            "unified_attention_launches": 0,
            "decode_steps": 0,
        },
        "vl": {
            "enabled": False,
            "media_count": 0,
            "image_count": 0,
            "video_count": 0,
            "source_bytes": 0,
            "vision_patches": 0,
            "visual_tokens": 0,
            "media_cache_hits": 0,
            "media_cache_misses": 0,
            "media_cache_entries": 0,
            "media_cache_resident_bytes": 0,
            "vision_batch_count": 0,
            "vision_max_batch_patches": 0,
            "vision_max_batch_tokens": 0,
            "vision_plan_cache_hit": False,
            "vision_plan_cache_entries": 0,
            "host_to_device_bytes": 0,
            "media_load_decode_wall_ms": 0.0,
            "media_load_wall_ms": 0.0,
            "media_decode_wall_ms": 0.0,
            "processor_wall_ms": 0.0,
            "vision_plan_build_wall_ms": 0.0,
            "vision_input_upload_wall_ms": 0.0,
            "vision_encode_wall_ms": 0.0,
            "embedding_injection_wall_ms": 0.0,
        },
    }


class NativeTextMetricsTest(unittest.TestCase):
    def test_text_request_allows_language_fmha_only(self) -> None:
        metrics = text_metrics()
        self.assertTrue(text_path_is_idle(metrics))
        self.assertTrue(all(text_path_idle_checks(metrics).values()))

    def test_any_media_processor_or_vision_work_fails(self) -> None:
        for section, field, value in (
            ("vl", "enabled", True),
            ("vl", "processor_wall_ms", 0.01),
            ("vl", "vision_batch_count", 1),
            ("vl", "host_to_device_bytes", 2),
            ("mrope", "enabled", True),
            ("mrope", "unified_attention_launches", 1),
        ):
            with self.subTest(section=section, field=field):
                metrics = text_metrics()
                metrics[section][field] = value
                self.assertFalse(text_path_is_idle(metrics))

    def test_missing_nested_metrics_fail_closed(self) -> None:
        self.assertEqual(
            text_path_idle_checks({}), {"metrics_shape_complete": False}
        )
        self.assertFalse(text_path_is_idle({}))
        self.assertEqual(
            text_path_idle_checks(None), {"metrics_shape_complete": False}
        )
        self.assertFalse(text_path_is_idle([]))


if __name__ == "__main__":
    unittest.main()
