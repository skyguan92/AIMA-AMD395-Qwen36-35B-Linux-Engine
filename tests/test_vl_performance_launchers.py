from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / "scripts/run-vllm-vl-performance-reference.sh"
NATIVE = ROOT / "scripts/run-native-vl-performance-candidate.sh"
DIAGNOSTIC = ROOT / "scripts/run-vl-performance-diagnostic-pair.sh"
VLLM_ENTRYPOINT = ROOT / "scripts/aima_vllm_vl_performance_server.py"
NATIVE_HTTP = ROOT / "native/src/native_http_server.cpp"
MATRIX_RUNNER = ROOT / "scripts/run-vl-performance-matrix-pair.sh"
DIAGNOSTIC_REQUEST = (
    ROOT
    / "benchmarks/fixtures/vl-performance-v0.1.0/requests/"
    / "diagnostic-image-typical-output1.json"
)


class VlPerformanceLauncherTest(unittest.TestCase):
    def test_reference_uses_frozen_surface_and_stage_instrumentation(self) -> None:
        source = VLLM.read_text(encoding="utf-8")
        self.assertIn("aima_vllm_vl_performance_server", source)
        self.assertIn("--max-model-len 262144", source)
        self.assertNotIn("--enable-mm-processor-stats", source)
        self.assertIn(
            "vllm_vl_benchmark_middleware.VlBenchmarkMetricsMiddleware",
            source,
        )
        self.assertIn("--no-enable-prefix-caching", source)
        self.assertIn('--mm-processor-cache-gb "${cache_gb}"', source)
        self.assertNotIn("\n  --enable-prefix-caching", source)

        entrypoint = VLLM_ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn(
            "from vllm.entrypoints.openai.api_server import run_server",
            entrypoint,
        )
        self.assertIn("parser = make_arg_parser(parser)", entrypoint)
        self.assertIn(
            "parser.set_defaults(enable_mm_processor_stats=True)",
            entrypoint,
        )
        self.assertIn("validate_parsed_serve_args(args)", entrypoint)

    def test_candidate_uses_window_and_automatic_provider_policy(self) -> None:
        source = NATIVE.read_text(encoding="utf-8")
        self.assertIn("AIMA_VL_CONTEXT_TOKENS:-262143", source)
        self.assertIn("AIMA_VL_CACHE_CAPACITY:-262144", source)
        self.assertNotIn("--fmha-provider", source)
        for provider in (
            "libaima-fmha-aotriton.so",
            "libaima-fmha-ck.so",
            "libaima-fmha-q16384-hybrid.so",
        ):
            self.assertIn(provider, source)
        self.assertIn("libaotriton_v2.so.0.11.1", source)
        self.assertIn(
            "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2", source
        )
        self.assertIn("AOTRITON_RUNTIME_SHA256", source)
        self.assertIn("AOTRITON_IMAGE_SHA256", source)
        self.assertIn("aotriton_images", source)
        self.assertIn('--vision-attention-image "${VISION_IMAGE}"', source)
        self.assertIn("--disable-media-cache", source)
        self.assertIn("--request-timeout-ms 600000", source)
        self.assertIn("AIMA_VL_PREFIX_CACHE_MODE", source)

        server = NATIVE_HTTP.read_text(encoding="utf-8")
        self.assertIn('argument == "--disable-prefix-cache"', server)
        self.assertIn("options.engine.prefix_cache_enabled = false", server)
        self.assertIn(
            "native_request.disable_prefix_cache = parsed.disable_prefix_cache",
            server,
        )

    def test_launchers_are_fail_closed_and_loopback_only(self) -> None:
        for path in (VLLM, NATIVE):
            source = path.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", source)
            self.assertIn("nohup setsid env -i", source)
            self.assertIn('HOST="127.0.0.1"', source)
            self.assertIn("ss -ltn", source)
            self.assertNotIn("0.0.0.0", source)

    def test_diagnostic_uses_fresh_processes_and_alternating_order(self) -> None:
        source = DIAGNOSTIC.read_text(encoding="utf-8")
        self.assertIn("PAIR_INDEX % 2", source)
        self.assertIn("capture_reference", source)
        self.assertIn("capture_candidate", source)
        self.assertIn("stop_active", source)
        self.assertIn("warm_text_path", source)
        self.assertIn("symmetric warmup", source)
        self.assertIn('pgrep -g "${active_pid}"', source)
        self.assertIn('kill -TERM -- "-${active_pid}"', source)
        self.assertIn("--prometheus", source)
        self.assertIn("AIMA_VL_CELL_ID", source)
        self.assertIn("AIMA_VL_TEXT_PADDING_TOKENS", source)
        self.assertIn("AIMA_VL_EXPECTED_COMPLETION_TOKENS", source)
        self.assertIn("AIMA_VL_TIMEOUT_SECONDS", source)
        self.assertIn('--prompt-nonce "${CELL_ID}"', source)
        self.assertIn('--timeout-seconds "${TIMEOUT_SECONDS}"', source)
        self.assertIn("vllm-vl-stages.jsonl", source)
        self.assertIn("set -euo pipefail", source)
        self.assertNotIn("0.0.0.0", source)

    def test_diagnostic_uses_the_frozen_typical_portrait_envelope(self) -> None:
        source = DIAGNOSTIC_REQUEST.read_text(encoding="utf-8")
        self.assertIn(
            "vl-envelope-v0.1.0/image-portrait-256x1024.png", source
        )
        self.assertNotIn("vl-capability-v0.1.0/image-rgb-256.png", source)

    def test_matrix_runner_balances_cells_and_disables_prefix_cache(self) -> None:
        source = MATRIX_RUNNER.read_text(encoding="utf-8")
        self.assertIn("PAIR_INDEX % 2", source)
        self.assertIn("balanced_orders", source)
        self.assertIn("AIMA_VL_PREFIX_CACHE_MODE=disabled", source)
        self.assertIn(".prompt_nonce", source)
        self.assertIn('--prompt-nonce "${prompt_nonce}"', source)
        self.assertIn("--expected-completion-tokens", source)
        self.assertIn("summarize-vl-performance-matrix-pair.py", source)


if __name__ == "__main__":
    unittest.main()
