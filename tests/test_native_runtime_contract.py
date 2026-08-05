from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import unittest

from aima_engine.release_evidence import verify_release_evidence


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_CONTRACT = ROOT / "native/product-contract.json"
PRODUCT_CONTRACT_V150 = ROOT / "native/product-contract-v1.5.0.json"
PRODUCT_CONTRACT_V141 = ROOT / "native/product-contract-v1.4.1.json"
PRODUCT_CONTRACT_V140 = ROOT / "native/product-contract-v1.4.0.json"
PRODUCT_CONTRACT_V130 = ROOT / "native/product-contract-v1.3.0.json"
NATIVE_RESULT = ROOT / "benchmarks/results/native-foundation-v0.1.0.json"
TOKENIZER_RESULT = ROOT / "benchmarks/results/native-tokenizer-v0.1.0.json"
AOT_MANIFEST = ROOT / "native/aot/gfx1151/q8192-output2/manifest.json"
AOT_RESULT = ROOT / "benchmarks/results/native-aot-q8192-v0.1.0.json"
GEMM_RESULT = ROOT / "benchmarks/results/native-bf16-gemm-v0.1.0.json"
WVSPLITK_RESULT = ROOT / "benchmarks/results/native-wvsplitk-v0.1.0.json"
DERIVED_RESULT = ROOT / "benchmarks/results/native-derived-weights-v0.1.0.json"
DERIVED_V2_RESULT = ROOT / "benchmarks/results/native-derived-weights-v0.2.0.json"
PORTABLE_BUNDLE_RESULT = ROOT / "benchmarks/results/native-portable-bundle-v0.1.0.json"
PORTABLE_BUNDLE_V130_RESULT = (
    ROOT / "benchmarks/results/native-portable-bundle-v1.3.0.json"
)
PORTABLE_BUNDLE_V140_RESULT = (
    ROOT / "benchmarks/results/native-portable-bundle-v1.4.0.json"
)
PORTABLE_BUNDLE_V141_RESULT = (
    ROOT / "benchmarks/results/native-portable-bundle-v1.4.1.json"
)
PORTABLE_BUNDLE_V150_RESULT = (
    ROOT / "benchmarks/results/native-portable-bundle-v1.5.0.json"
)
PORTABLE_PRODUCT_RESULT = ROOT / "benchmarks/results/native-portable-product-v1.5.0.json"
PORTABLE_PRODUCT_V141_RESULT = (
    ROOT / "benchmarks/results/native-portable-product-v1.4.1.json"
)
PORTABLE_PRODUCT_V140_RESULT = (
    ROOT / "benchmarks/results/native-portable-product-v1.4.0.json"
)
PORTABLE_PRODUCT_V130_RESULT = (
    ROOT / "benchmarks/results/native-portable-product-v1.3.0.json"
)
RELEASE_PROVENANCE_V130_RESULT = (
    ROOT / "benchmarks/results/native-release-provenance-v1.3.0.json"
)
RELEASE_PROVENANCE_V140_RESULT = (
    ROOT / "benchmarks/results/native-release-provenance-v1.4.0.json"
)
RELEASE_PROVENANCE_V141_RESULT = (
    ROOT / "benchmarks/results/native-release-provenance-v1.4.1.json"
)
RELEASE_PROVENANCE_V150_RESULT = (
    ROOT / "benchmarks/results/native-release-provenance-v1.5.0.json"
)
DECODE_SCHEDULE = ROOT / "native/aot/gfx1151/q8192-output2/decode-schedule.json"
DECODE_SCHEDULE_RESULT = ROOT / "benchmarks/results/native-decode-schedule-v0.1.0.json"
DECODE_BINDINGS_RESULT = ROOT / "benchmarks/results/native-decode-bindings-v0.1.0.json"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root must be an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class NativeRuntimeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_json(PRODUCT_CONTRACT)
        cls.contract_v150 = load_json(PRODUCT_CONTRACT_V150)
        cls.contract_v141 = load_json(PRODUCT_CONTRACT_V141)
        cls.contract_v140 = load_json(PRODUCT_CONTRACT_V140)
        cls.contract_v130 = load_json(PRODUCT_CONTRACT_V130)
        cls.inference_baseline = load_json(ROOT / "benchmarks/results/v1.0.0.json")
        cls.startup_baseline = load_json(ROOT / "benchmarks/results/v1.1.0.json")
        cls.result = load_json(NATIVE_RESULT)
        cls.tokenizer_result = load_json(TOKENIZER_RESULT)
        cls.aot_manifest = load_json(AOT_MANIFEST)
        cls.aot_result = load_json(AOT_RESULT)
        cls.gemm_result = load_json(GEMM_RESULT)
        cls.wvsplitk_result = load_json(WVSPLITK_RESULT)
        cls.derived_result = load_json(DERIVED_RESULT)
        cls.derived_v2_result = load_json(DERIVED_V2_RESULT)
        cls.portable_bundle_result = load_json(PORTABLE_BUNDLE_RESULT)
        cls.portable_bundle_v130_result = load_json(
            PORTABLE_BUNDLE_V130_RESULT
        )
        cls.portable_bundle_v140_result = load_json(
            PORTABLE_BUNDLE_V140_RESULT
        )
        cls.portable_bundle_v141_result = load_json(
            PORTABLE_BUNDLE_V141_RESULT
        )
        cls.portable_bundle_v150_result = load_json(
            PORTABLE_BUNDLE_V150_RESULT
        )
        cls.portable_product_result = load_json(PORTABLE_PRODUCT_RESULT)
        cls.portable_product_v141_result = load_json(
            PORTABLE_PRODUCT_V141_RESULT
        )
        cls.portable_product_v140_result = load_json(
            PORTABLE_PRODUCT_V140_RESULT
        )
        cls.portable_product_v130_result = load_json(PORTABLE_PRODUCT_V130_RESULT)
        cls.release_provenance_v130 = load_json(RELEASE_PROVENANCE_V130_RESULT)
        cls.release_provenance_v140 = load_json(RELEASE_PROVENANCE_V140_RESULT)
        cls.release_provenance_v141 = load_json(RELEASE_PROVENANCE_V141_RESULT)
        cls.release_provenance_v150 = load_json(RELEASE_PROVENANCE_V150_RESULT)
        cls.decode_schedule = load_json(DECODE_SCHEDULE)
        cls.decode_schedule_result = load_json(DECODE_SCHEDULE_RESULT)
        cls.decode_bindings_result = load_json(DECODE_BINDINGS_RESULT)

    def test_product_contract_defines_the_current_release(self) -> None:
        model = self.contract["model"]
        gates = self.contract["promotion_gates"]
        self.assertEqual(self.contract, self.contract_v150)
        self.assertEqual(self.contract["release"], "1.5.0")
        self.assertEqual(
            self.contract["status"],
            "qualification_bound_portable_native_full_envelope",
        )
        self.assertEqual(model["tensor_count"], 693)
        self.assertEqual(model["source_repository"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(
            model["source_revision"],
            "995ad96eacd98c81ed38be0c5b274b04031597b0",
        )
        self.assertEqual(model["checkpoint_shards"], 26)
        self.assertEqual(model["payload_bytes"], 69_321_221_376)
        self.assertEqual(
            model["checkpoint_index_sha256"],
            self.startup_baseline["checkpoint_index_sha256"],
        )
        self.assertEqual(
            gates["startup"]["standard_safetensors_command_to_ready_median_ms_max"],
            self.startup_baseline["startup"]["median_model_load_to_api_ready_ms"],
        )
        self.assertEqual(
            gates["startup"]["optional_striped_command_to_ready_median_ms_max"],
            self.inference_baseline["startup_and_first_user"]["command_to_ready_median_ms"],
        )
        self.assertEqual(
            gates["prefix_cache"]["minimum_ttft_speedup"],
            self.inference_baseline["prefix_cache"]["median_ttft_speedup"],
        )
        self.assertEqual(
            gates["prefix_cache"]["minimum_decode_retention"],
            self.inference_baseline["prefix_cache"]["minimum_decode_retention"],
        )
        profile = self.contract["qualified_native_profile"]
        self.assertEqual(
            profile["input_tokens"],
            [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        )
        self.assertEqual(profile["output_tokens"], [512, 1024])
        self.assertEqual(
            profile["valid_window_endpoints"],
            gates["matrix"]["valid_window_endpoints"],
        )
        self.assertTrue(profile["automatic_provider_selection"])
        self.assertTrue(profile["legacy_full_context_envelope_replaced"])
        self.assertEqual(profile["qualification"], "share/aima/qualification.json")
        self.assertIn("every positive prompt", profile["variable_length_prompts"])
        self.assertIn("q1024/q2048/q4096/q8192", profile["variable_length_prompts"])
        self.assertIn("capacity-bounded", profile["prefix_cache"])
        openai = gates["openai_features"]
        self.assertTrue(openai["variable_length_cold_prompt_required"])
        self.assertTrue(openai["ordinary_multi_turn_cache_miss_required"])
        self.assertTrue(openai["post_long_short_request_isolation_required"])
        capability = gates["capability_eval"]
        self.assertEqual(capability["items"], 256)
        self.assertEqual(capability["minimum_correct"], 216)
        self.assertEqual(capability["invalid_answers_max"], 0)
        self.assertTrue(capability["frozen_gb10_score_nonregression_required"])
        self.assertEqual(
            capability["frozen_gb10_prompt_token_hash_matches_required"], 256
        )
        self.assertTrue(
            capability[
                "prompt_text_and_token_ids_must_be_excluded_from_public_scorecard"
            ]
        )

    def test_v141_contract_is_bound_at_package_time(self) -> None:
        contract = self.contract_v141
        self.assertEqual(contract["release"], "1.4.1")
        self.assertTrue(contract["artifact_integrity"]["qualification_required"])
        self.assertTrue(contract["artifact_integrity"]["clean_source_required"])
        self.assertTrue(
            contract["artifact_integrity"]["embedded_source_commit_must_match"]
        )

    def test_v150_contract_is_bound_at_package_time(self) -> None:
        contract = self.contract_v150
        self.assertEqual(contract["release"], "1.5.0")
        self.assertTrue(contract["artifact_integrity"]["qualification_required"])
        self.assertTrue(contract["artifact_integrity"]["clean_source_required"])
        self.assertTrue(
            contract["artifact_integrity"]["embedded_source_commit_must_match"]
        )

    def test_v140_contract_is_bound_at_package_time(self) -> None:
        contract = self.contract_v140
        self.assertEqual(contract["release"], "1.4.0")
        self.assertEqual(
            contract["qualified_native_profile"]["qualification"],
            "share/aima/qualification.json",
        )
        self.assertTrue(contract["artifact_integrity"]["qualification_required"])
        self.assertTrue(contract["artifact_integrity"]["clean_source_required"])
        self.assertTrue(
            contract["artifact_integrity"]["embedded_source_commit_must_match"]
        )

    def test_portable_native_product_profile_is_hash_bound_and_nonregressing(self) -> None:
        result = self.portable_product_result
        profile = self.contract_v150["qualified_native_profile"]
        self.assertTrue(result["complete"])
        self.assertTrue(result["qualified"])
        self.assertEqual(result["release"], "1.5.0")
        self.assertEqual(result["scope"]["input_tokens"], profile["input_tokens"])
        self.assertEqual(result["scope"]["output_tokens"], profile["output_tokens"])
        self.assertTrue(result["scope"]["matrix_complete_for_admitted_native_profile"])
        self.assertTrue(result["scope"]["legacy_v1_1_long_context_profile_replaced"])
        self.assertEqual(result["components"]["source"]["release_tag"], "v1.5.0")
        self.assertEqual(
            result["components"]["source"]["release_commit"],
            self.release_provenance_v150["release_commit"],
        )
        self.assertEqual(len(result["performance"]["cells"]), 19)
        self.assertTrue(result["performance"]["all_cells_pass"])
        for cell in result["performance"]["cells"]:
            self.assertGreaterEqual(cell["prefill_retention"], 0.97)
            if cell["output_tokens"] > 1:
                self.assertGreaterEqual(cell["decode_retention"], 0.97)
        self.assertTrue(result["correctness"]["all_contexts_pass"])
        self.assertEqual(len(result["correctness"]["contexts"]), 9)
        for context in result["correctness"]["contexts"]:
            self.assertLess(context["kld"], 0.005)
            self.assertTrue(context["top1_match"])
        self.assertTrue(
            result["correctness"]["exact_completion"][
                "expected_tokens_match"
            ]
        )
        self.assertTrue(result["startup"]["pass"])
        self.assertTrue(result["prefix_cache"]["q32768_output512_exact"]["pass"])
        self.assertTrue(result["http"]["resident"])
        self.assertEqual(result["http"]["model_loads"], 1)
        self.assertTrue(result["openai_features"]["pass"])
        self.assertTrue(result["openai_features"]["variable_prompts"]["pass"])
        self.assertEqual(
            result["openai_features"]["variable_prompts"][
                "ordinary_turn_status"
            ],
            200,
        )
        self.assertTrue(result["openai_features"]["streaming"]["pass"])
        self.assertTrue(
            result["openai_features"]["streaming"][
                "content_matches_nonstream"
            ]
        )
        self.assertTrue(
            result["openai_features"]["streaming"][
                "token_sha256_matches_nonstream"
            ]
        )
        self.assertTrue(result["openai_features"]["tools"]["pass"])
        self.assertTrue(
            result["openai_features"]["tools"][
                "token_sha256_matches_nonstream"
            ]
        )
        self.assertEqual(
            result["openai_features"]["tools"]["finish_reason"],
            "tool_calls",
        )
        self.assertTrue(result["openai_features"]["disconnect"]["pass"])
        capability = result["capability_eval"]
        self.assertTrue(capability["pass"])
        self.assertEqual(capability["score"]["items"], 256)
        self.assertEqual(capability["score"]["correct"], 216)
        self.assertEqual(capability["score"]["invalid_answers"], 0)
        self.assertEqual(capability["reference_comparison"]["reference_correct"], 216)
        self.assertEqual(
            capability["reference_comparison"]["prompt_token_hash_matches"], 256
        )
        self.assertTrue(
            capability["reference_comparison"]["score_nonregression_pass"]
        )
        self.assertFalse(capability["source"]["prompt_text_in_scorecard"])
        self.assertFalse(capability["source"]["prompt_token_ids_in_scorecard"])
        for dependency in ("python", "torch", "vllm", "triton", "transformers"):
            self.assertFalse(result["runtime_dependency_gate"][f"runtime_{dependency}"])
        self.assertFalse(result["runtime_dependency_gate"]["host_rocm_userspace_required"])
        self.assertTrue(result["decision"]["native_profile_performance_nonregression_pass"])
        self.assertTrue(result["decision"]["variable_length_prompts_pass"])
        self.assertTrue(result["decision"]["full_legacy_context_envelope_replacement_pass"])
        self.assertTrue(result["decision"]["http_streaming_pass"])
        self.assertTrue(result["decision"]["tool_calling_pass"])
        self.assertTrue(result["decision"]["disconnect_cancellation_pass"])
        self.assertTrue(result["decision"]["resident_prefill_dispatch_pass"])
        self.assertTrue(result["decision"]["multi_entry_prefix_lru_pass"])
        self.assertTrue(result["decision"]["frozen_capability_eval_pass"])

    def test_v130_release_archive_is_isolated_and_provider_complete(self) -> None:
        bundle = self.portable_bundle_v130_result
        self.assertEqual(bundle["release"], "1.3.0")
        self.assertTrue(bundle["complete"])
        self.assertTrue(bundle["qualified"])
        self.assertTrue(bundle["archive"]["checksum_verified"])
        self.assertEqual(bundle["archive"]["root_mode"], "0755")
        self.assertTrue(bundle["manifest"]["complete"])
        self.assertEqual(bundle["manifest"]["checked_files"], 240)
        self.assertTrue(bundle["elf_closure"]["complete"])
        self.assertTrue(bundle["elf_closure"]["launcher_static"])
        self.assertEqual(
            bundle["elf_closure"]["host_userspace_dependencies"],
            [],
        )
        self.assertEqual(
            bundle["elf_closure"]["unresolved_userspace_dependencies"],
            {},
        )
        self.assertFalse(
            bundle["isolated_environment"]["host_rocm_path"]
        )
        self.assertEqual(
            bundle["isolated_environment"]["version"],
            "aima-engine-native 1.3.0-native",
        )
        self.assertEqual(
            [smoke["context_tokens"] for smoke in bundle["provider_smokes"]],
            [1024, 16384, 65536],
        )
        self.assertTrue(
            all(smoke["qualified"] for smoke in bundle["provider_smokes"])
        )

    def test_v130_public_raw_evidence_is_present_and_hash_bound(self) -> None:
        self.assertEqual(verify_release_evidence(ROOT, release="1.3.0"), [])
        provenance = self.release_provenance_v130
        self.assertEqual(provenance["release_tag"], "v1.3.0")
        self.assertEqual(
            provenance["release_commit"],
            "032dc137992365649a47353910b76f93acb86d75",
        )
        self.assertEqual(
            provenance["native_source_commit"],
            "745930457f06629542ea996c8771ab38382fce98",
        )
        self.assertEqual(
            self.portable_product_v130_result["components"]["source_base_commit"],
            provenance["derived_from_commit"],
        )
        self.assertEqual(
            sum(
                record["file_count"]
                for record in provenance["public_evidence_trees"].values()
            ),
            92,
        )

    def test_v140_public_raw_evidence_is_preserved_and_hash_bound(self) -> None:
        self.assertEqual(verify_release_evidence(ROOT, release="1.4.0"), [])
        provenance = self.release_provenance_v140
        self.assertEqual(provenance["release_tag"], "v1.4.0")
        self.assertEqual(
            provenance["release_commit"],
            "db54224cfcb9dae60607ccf6481e412e5c3a991e",
        )
        self.assertEqual(
            self.portable_product_v140_result["components"]["source"]["native_source_commit"],
            provenance["native_source_commit"],
        )
        self.assertEqual(self.portable_bundle_v140_result["release"], "1.4.0")
        self.assertTrue(self.portable_bundle_v140_result["qualified"])
        self.assertEqual(
            sum(
                record["file_count"]
                for record in provenance["public_evidence_trees"].values()
            ),
            90,
        )

    def test_v141_public_raw_evidence_is_preserved_and_hash_bound(self) -> None:
        self.assertEqual(verify_release_evidence(ROOT, release="1.4.1"), [])
        provenance = self.release_provenance_v141
        self.assertEqual(provenance["release_tag"], "v1.4.1")
        self.assertEqual(
            provenance["release_commit"],
            "ba45639c178061f9bdadd22c86744f6924f5bf44",
        )
        self.assertEqual(
            self.portable_product_v141_result["components"]["source"][
                "native_source_commit"
            ],
            provenance["native_source_commit"],
        )
        self.assertEqual(self.portable_bundle_v141_result["release"], "1.4.1")
        self.assertTrue(self.portable_bundle_v141_result["qualified"])
        self.assertEqual(
            sum(
                record["file_count"]
                for record in provenance["public_evidence_trees"].values()
            ),
            90,
        )

    def test_v150_public_raw_evidence_is_the_default_and_hash_bound(self) -> None:
        self.assertEqual(verify_release_evidence(ROOT), [])
        provenance = self.release_provenance_v150
        self.assertEqual(provenance["release_tag"], "v1.5.0")
        self.assertEqual(
            provenance["release_commit"],
            "d82e6943bc50d821011ce79e95afee06f6b12a36",
        )
        self.assertEqual(
            self.portable_product_result["components"]["source"][
                "native_source_commit"
            ],
            provenance["native_source_commit"],
        )
        self.assertEqual(self.portable_bundle_v150_result["release"], "1.5.0")
        self.assertTrue(self.portable_bundle_v150_result["qualified"])
        self.assertEqual(
            set(provenance["public_evidence"]),
            {
                "matrix",
                "correctness",
                "surfaces",
                "openai_features",
                "capability_eval",
                "portable_bundle",
                "second_host_compat",
            },
        )
        self.assertEqual(
            sum(
                record["file_count"]
                for record in provenance["public_evidence_trees"].values()
            ),
            114,
        )
        independent_host = load_json(
            ROOT
            / provenance["public_evidence"]["second_host_compat"]["path"]
        )
        self.assertTrue(independent_host["decision"]["overall_pass"])
        self.assertGreaterEqual(
            independent_host["performance"]["cold_prefill_tps"]["retention"],
            0.97,
        )
        self.assertGreaterEqual(
            independent_host["performance"]["decode_512_tps"]["retention"],
            0.97,
        )

    def test_generated_layout_is_current_and_complete(self) -> None:
        check = subprocess.run(
            ["python3", "scripts/generate-native-layout.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        header = (ROOT / "native/generated/model_layout.h").read_text(encoding="utf-8")
        self.assertIn("std::array<TensorSpec, 693>", header)
        self.assertIn("kPayloadBytes = 69321221376ULL", header)
        self.assertIn(self.contract["model"]["config_sha256"], header)
        self.assertIn(self.contract["model"]["checkpoint_index_sha256"], header)

    def test_native_build_has_only_origin_relative_rocm_resolution(self) -> None:
        script = (ROOT / "scripts/build-native-runtime.sh").read_text(encoding="utf-8")
        self.assertIn("-fno-rtlib-add-rpath", script)
        self.assertIn("-fno-gpu-rdc", script)
        self.assertIn("-Wl,-z,origin", script)
        self.assertIn("-Wl,-rpath,'$ORIGIN/../lib'", script)
        self.assertIn("libicui18n.a", script)
        self.assertIn("libicuuc.a", script)
        self.assertIn("libicudata.a", script)
        self.assertIn("generate-native-aot-registry.py", script)
        self.assertIn("aot_kernel.hip.cpp", script)
        self.assertIn("aot_registry_probe.hip.cpp", script)
        self.assertIn("bf16_gemm.hip.cpp", script)
        self.assertIn("bf16_wvsplitk.hip.cpp", script)
        self.assertIn("native_derived_weights.hip.cpp", script)
        self.assertIn("native_decode_bindings.hip.cpp", script)
        self.assertIn("native_decode_invocation.cpp", script)
        self.assertIn("native_decode_workspace.hip.cpp", script)
        self.assertIn("native_lm_head.hip.cpp", script)
        self.assertIn("native_doctor.cpp", script)
        self.assertIn("AIMA_SOURCE_COMMIT", script)
        self.assertIn("portable_launcher.c", script)
        self.assertIn("-static", script)
        self.assertIn("-lhipblaslt", script)
        self.assertIn("objcopy", script.lower())
        self.assertNotIn("-Wl,-rpath,/opt/rocm", script)
        self.assertNotIn("patchelf", script)
        tracer = (ROOT / "scripts/native_aot_trace/sitecustomize.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("AIMA_AOT_TRACE_ALL_LAUNCHES", tracer)
        self.assertIn("AIMA_AOT_TRACE_POINTERS", tracer)
        self.assertIn("AIMA_AOT_TENSOR_REGISTRY_JSON", tracer)
        self.assertIn('frame.f_code.co_name == "run_with_torch"', tracer)

        package = (ROOT / "scripts/package-native-foundation.sh").read_text(
            encoding="utf-8"
        )
        for soname in (
            "libamdhip64.so.7",
            "libhipblaslt.so.1",
            "libhsa-runtime64.so.1",
            "librocprofiler-register.so.0",
            "libroctx64.so.4",
            "librocroller.so.1",
            "libamd_comgr.so.3",
            "libhsa-amd-aqlprofile64.so.1",
        ):
            self.assertIn(soname, package)
        for soname in (
            "ld-linux-x86-64.so.2",
            "libc.so.6",
            "libm.so.6",
            "libstdc++.so.6",
            "libgcc_s.so.1",
            "libelf.so.1",
            "libdrm.so.2",
            "libdrm_amdgpu.so.1",
            "libnuma.so.1",
            "libz.so.1",
            "libzstd.so.1",
            "liblzma.so.5",
        ):
            self.assertIn(soname, package)
        for product_payload in (
            "libaima-fmha-aotriton.so",
            "libaima-fmha-ck.so",
            "libaima-fmha-q16384-hybrid.so",
            "libaotriton_v2.so.0.11.1",
            "product-contract.json",
            "aima-engine.service",
        ):
            self.assertIn(product_payload, package)
        self.assertIn("tar --sort=name", package)
        self.assertIn("--zstd", package)
        self.assertIn("sha256sum", package)
        self.assertIn("native_bundle_closure.py", package)
        self.assertIn("AIMA_ALLOW_DIRTY_PACKAGE", package)
        self.assertIn("--source-commit", package)
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "package-native:\n\tbash scripts/package-native-foundation.sh",
            makefile,
        )
        self.assertNotIn(
            "package-native: build-native-runtime build-native", makefile
        )
        self.assertIn("verify-native-package-inputs.py", package)
        self.assertIn("product-contract-v${RELEASE_VERSION}.json", package)
        self.assertIn('release_contracts=("${ROOT}"/native/product-contract-v*.json)', package)
        for component in (
            "native_engine",
            "static_launcher",
            "aotriton_fmha_provider",
            "ck_fmha_provider",
            "q16384_hybrid_fmha_provider",
            "aotriton_runtime",
            "aotriton_gfx1151_image",
        ):
            self.assertIn(f'--component "{component}=', package)
        product_generator = (
            ROOT / "scripts/generate-native-product-result.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"variable_prompts": features["variable_prompts"]',
            product_generator,
        )
        self.assertIn('"variable_length_prompts_pass":', product_generator)
        launcher = (ROOT / "native/src/portable_launcher.c").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--inhibit-cache"', launcher)
        self.assertIn('"--library-path"', launcher)
        self.assertIn('"/libexec/aima-engine.real"', launcher)
        for license_path in (
            "share/doc/hip/LICENSE.md",
            "share/doc/hsa-rocr/LICENSE.md",
            "share/doc/rocprofiler-register/LICENSE.md",
            "share/doc/amd_comgr/LICENSE.txt",
            "share/doc/rocm-device-libs/LICENSE.TXT",
            "share/doc/hipblaslt/LICENSE.md",
            "share/doc/rocprofiler-sdk/LICENSE.md",
        ):
            self.assertIn(license_path, package)
        main = (ROOT / "native/src/main.cpp").read_text(encoding="utf-8")
        self.assertIn('setenv("ROCM_PATH"', main)
        self.assertIn('setenv("HIP_DEVICE_LIB_PATH"', main)
        self.assertIn('setenv("HIPBLASLT_TENSILE_LIBPATH"', main)
        self.assertIn('std::string(argv[1]) == "doctor"', main)
        self.assertIn('std::string(argv[1]) == "--build-info"', main)
        doctor = (ROOT / "native/src/native_doctor.cpp").read_text(
            encoding="utf-8"
        )
        for check_id in (
            "host.platform",
            "device.kfd",
            "device.render",
            "gpu.architecture",
            "memory.kernel_parameters",
            "memory.vram",
            "memory.gtt",
            "runtime.bundle",
            "model.shards",
        ):
            self.assertIn(check_id, doctor)
        server = (ROOT / "native/src/native_http_server.cpp").read_text(
            encoding="utf-8"
        )
        for option in (
            "--api-key-file",
            "--request-timeout-ms",
            "--disable-http-shutdown",
            "--allow-insecure-remote",
            "WWW-Authenticate: Bearer",
            "O_NOFOLLOW",
            "::fstat",
            "metadata.st_mode & 0027",
            "receive_before",
            "READY=1\\nSTATUS=Ready",
            "STOPPING=1\\nSTATUS=Stopping",
        ):
            self.assertIn(option, server)
        self.assertLess(server.index("::bind("), server.index("engine.load("))

    def test_variable_prompt_fallback_is_part_of_the_native_contract(self) -> None:
        server = (ROOT / "native/src/native_http_server.cpp").read_text(
            encoding="utf-8"
        )
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        planner = (ROOT / "native/include/aima/native_prompt_plan.h").read_text(
            encoding="utf-8"
        )
        qualification = (
            ROOT / "scripts/qualify-native-openai-features.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("requires a cold prompt of", server)
        self.assertNotIn("cold prefill requires the static context", resident)
        self.assertIn("plan_native_prompt_execution", resident)
        self.assertIn("cold-decode-fallback", planner)
        self.assertIn("cold-aot-plus-decode", planner)
        self.assertIn("clear_request_scratch", resident)
        self.assertIn("ordinary_turn_pass", qualification)
        self.assertIn("resident_bucket_pass", qualification)
        self.assertIn("prefix_lru_pass", qualification)
        self.assertIn("prompt_token_ids", qualification)
        product_generator = (
            ROOT / "scripts/generate-native-product-result.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--capability-eval", product_generator)
        self.assertIn("prompt_token_ids_in_scorecard", product_generator)
        self.assertIn("frozen_capability_eval_pass", product_generator)

    def test_resident_prefill_dispatch_and_prefix_lru_are_bounded(self) -> None:
        resident = (
            ROOT / "native/src/native_resident_engine.hip.cpp"
        ).read_text(encoding="utf-8")
        server = (ROOT / "native/src/native_http_server.cpp").read_text(
            encoding="utf-8"
        )
        header = (
            ROOT / "native/include/aima/native_resident_engine.h"
        ).read_text(encoding="utf-8")
        for token_count in ("1024ULL", "2048ULL", "4096ULL", "8192ULL"):
            self.assertIn(token_count, resident)
        self.assertIn("resident_prefill_buckets", resident)
        self.assertIn("auxiliary_prefill_buckets", resident)
        self.assertIn("prefix_cache_entries", resident)
        self.assertIn("kPrefixCacheEntries = 4", resident)
        self.assertIn("options.cache_capacity <= 131072 ? 2 : 1", resident)
        self.assertIn("resident_prefill_buckets", server)
        self.assertIn("prefix_cache_entries", server)
        self.assertIn("aot_prefill_tokens", server)
        self.assertIn("resident_prefill_buckets", header)
        self.assertIn("aot_prefill_tokens", header)

    def test_native_weight_foundation_has_target_measurement(self) -> None:
        self.assertTrue(self.result["complete"])
        self.assertEqual(
            self.result["scope"], "native_weight_ownership_and_bundle_only"
        )
        self.assertEqual(self.result["binary"]["sha256"],
                         "4daba873b997415bbc4755ee09c69e9c96decd5af66262c1d1328edac142da36")
        self.assertEqual(len(self.result["runs"]), 3)
        for run in self.result["runs"]:
            self.assertEqual(run["direct_io_shards"], 26)
            self.assertEqual(run["buffered_shards"], 0)
            self.assertEqual(run["unique_device_pointers"], 693)
            self.assertTrue(run["all_pointer_types_device"])
            self.assertTrue(run["all_device_pointers_match"])
            self.assertTrue(run["gpu_payload_checksum_equal"])
        self.assertGreater(
            self.result["comparison"]["native_to_v1_1_preload_speedup"], 1.0
        )
        self.assertFalse(self.result["claims"]["full_native_inference_qualified"])
        self.assertFalse(self.result["claims"]["api_ready_startup_qualified"])
        self.assertTrue(
            self.result["claims"]["native_foundation_rocm_bundle_self_contained"]
        )
        self.assertFalse(
            self.result["claims"]["full_product_rocm_bundle_qualified"]
        )

    def test_native_tokenizer_has_exact_reference_parity(self) -> None:
        result = self.tokenizer_result
        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage"]["case_count"], 85)
        self.assertTrue(result["decision"]["all_token_ids_equal"])
        self.assertTrue(result["decision"]["all_text_decodes_equal"])
        self.assertTrue(result["decision"]["all_chat_template_text_equal"])
        self.assertEqual(result["decision"]["icu_linkage"], "static")
        self.assertFalse(result["decision"]["native_runtime_python"])
        self.assertFalse(result["decision"]["native_runtime_transformers"])

    def test_bounded_aot_closure_is_complete_path_clean_and_measured(self) -> None:
        manifest = self.aot_manifest
        result = self.aot_result
        self.assertTrue(result["complete"])
        self.assertEqual(result["scope"], "bounded_q8192_output2_aot_foundation_only")
        self.assertEqual(sha256(AOT_MANIFEST), result["aot_manifest"]["sha256"])
        self.assertEqual(manifest["target"]["arch"], "gfx1151")
        self.assertEqual(manifest["kernel_count"], 25)
        self.assertEqual(manifest["kernel_symbol_count"], 22)
        self.assertEqual(manifest["launch_variant_count"], 26)
        self.assertEqual(manifest["packaged_hsaco_bytes"], 345_392)
        self.assertFalse(manifest["coverage"]["full_matrix_complete"])
        self.assertEqual(len(manifest["kernels"]), manifest["kernel_count"])

        image_bytes = 0
        for kernel in manifest["kernels"]:
            image = kernel["image"]
            path = AOT_MANIFEST.parent / image["path"]
            payload = path.read_bytes()
            self.assertEqual(len(payload), image["bytes"])
            self.assertEqual(sha256(path), image["sha256"])
            for marker in (b"/home/", b"/data/", b"site-packages"):
                self.assertNotIn(marker, payload)
            image_bytes += len(payload)
        self.assertEqual(image_bytes, manifest["packaged_hsaco_bytes"])

        probe = result["native_probe"]
        self.assertEqual(probe["loaded_count"], manifest["kernel_count"])
        self.assertEqual(probe["exact_bf16_elements"], 1025)
        self.assertEqual(probe["expected_bf16_elements"], 1025)
        for dependency in ("python", "torch", "vllm", "triton"):
            self.assertFalse(probe[f"runtime_{dependency}"])
        self.assertGreaterEqual(result["bounded_performance"]["prefill_retention"], 0.97)
        self.assertFalse(result["claims"]["full_native_inference_qualified"])

    def test_native_derived_weights_are_exact_and_within_startup_gate(self) -> None:
        result = self.derived_result
        self.assertTrue(result["complete"])
        self.assertEqual(result["layout"]["view_count"], 80)
        self.assertEqual(result["layout"]["payload_bytes"], 2_063_237_120)
        self.assertEqual(len(result["runs"]), 2)
        for run in result["runs"]:
            self.assertTrue(run["full_payload_checksum_equal"])
            self.assertEqual(run["exact_sample_elements"], 270)
            self.assertEqual(run["projection_exact_elements"], 98_816)
            self.assertEqual(run["projection_relative_l2_error"], 0.0)
        startup = result["startup_comparison"]
        self.assertLessEqual(
            startup["native_weight_load_plus_derived_median_ms"],
            self.contract["promotion_gates"]["startup"]
            ["standard_safetensors_command_to_ready_median_ms_max"],
        )
        self.assertTrue(startup["bounded_startup_gate_passed"])
        self.assertFalse(result["claims"]["api_ready_startup_qualified"])
        self.assertFalse(result["claims"]["full_context_matrix_qualified"])

    def test_all_schedule_required_decode_weight_views_are_resident(self) -> None:
        result = self.derived_v2_result
        self.assertTrue(result["complete"])
        self.assertEqual(result["layout"]["view_count"], 120)
        self.assertEqual(result["layout"]["router_transposed_views"], 40)
        self.assertEqual(result["layout"]["payload_bytes"], 2_105_180_160)
        self.assertEqual(len(result["runs"]), 2)
        self.assertLess(result["spread_percent"]["weight_load_plus_derived_ms"], 3.0)
        for run in result["runs"]:
            self.assertTrue(run["full_payload_checksum_equal"])
            self.assertEqual(run["exact_sample_elements"], 310)
            self.assertEqual(run["projection_elements"], 98_816)
            self.assertEqual(run["projection_exact_elements"], 98_816)
            self.assertEqual(run["router_transpose_elements"], 524_288)
            self.assertEqual(run["router_transpose_exact_elements"], 524_288)
        startup = result["startup_comparison"]
        self.assertLessEqual(
            startup["native_weight_load_plus_derived_median_ms"],
            self.contract["promotion_gates"]["startup"]
            ["standard_safetensors_command_to_ready_median_ms_max"],
        )
        self.assertTrue(
            result["claims"]["all_layer_local_aot_decode_weight_layouts_implemented"]
        )
        self.assertFalse(result["claims"]["global_lm_head_int8_layout_implemented"])
        self.assertFalse(result["claims"]["all_aot_decode_weight_layouts_implemented"])
        self.assertFalse(result["claims"]["api_ready_startup_qualified"])
        self.assertFalse(result["claims"]["full_native_inference_qualified"])

    def test_native_production_shape_gemm_is_exact_and_nonregressing(self) -> None:
        result = self.gemm_result
        self.assertTrue(result["complete"])
        self.assertEqual(result["shape"], {"m": 8192, "n": 12352, "k": 2048})
        native = result["native"]
        reference = result["torch_reference"]
        self.assertEqual(native["provider"], "hipBLASLt")
        self.assertFalse(native["runtime_python"])
        self.assertFalse(native["runtime_torch"])
        self.assertEqual(native["exact_bf16_elements"], 4096)
        self.assertEqual(reference["exact_bf16_elements"], 4096)
        self.assertGreaterEqual(result["comparison"]["native_throughput_ratio"], 0.97)
        self.assertTrue(result["comparison"]["bounded_nonregression_passed"])
        self.assertEqual(result["bundle"]["successful_system_rocm_path_opens"], 0)
        self.assertFalse(result["claims"]["full_native_inference_qualified"])

    def test_native_decode_wvsplitk_is_bit_exact_and_nonregressing(self) -> None:
        result = self.wvsplitk_result
        self.assertTrue(result["complete"])
        self.assertEqual(
            result["source"]["upstream"]["commit"],
            "29e5d102050669d03992a2eb863ad364ea50fab2",
        )
        self.assertEqual(
            sha256(ROOT / result["source"]["native_source"]),
            result["source"]["native_source_sha256"],
        )
        self.assertFalse(result["binary"]["runtime_python"])
        self.assertFalse(result["binary"]["runtime_torch"])
        self.assertFalse(result["binary"]["runtime_vllm"])
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(
            {(case["shape"]["m"], case["shape"]["k"]) for case in result["cases"]},
            {(2048, 512), (2048, 4096)},
        )
        for case in result["cases"]:
            self.assertTrue(case["full_output_bf16_sha256_equal_in_all_runs"])
            self.assertLessEqual(case["native_to_vllm_time_ratio"], 1.03)
            self.assertTrue(case["bounded_nonregression_passed"])
        self.assertTrue(result["decision"]["full_bf16_output_parity_passed"])
        self.assertTrue(
            result["decision"]["both_decode_projection_shapes_nonregressing"]
        )
        self.assertTrue(result["bundle"]["isolated_home_and_empty_environment_passed"])
        self.assertEqual(result["bundle"]["successful_system_rocm_path_opens"], 0)
        self.assertFalse(result["claims"]["full_native_inference_qualified"])

    def test_portable_bundle_closes_userspace_without_provider_regression(self) -> None:
        result = self.portable_bundle_result
        self.assertTrue(result["complete"])
        self.assertTrue(result["launcher"]["static_elf"])
        self.assertIsNone(result["launcher"]["elf_interpreter"])
        self.assertEqual(result["launcher"]["needed_shared_libraries"], [])
        bundle = result["bundle"]
        self.assertEqual(bundle["unresolved_userspace_elf_dependencies"], 0)
        self.assertEqual(bundle["host_userspace_elf_dependencies"], 0)
        isolation = result["dependency_isolation"]
        self.assertEqual(isolation["system_rocm_successful_opens"], 0)
        self.assertEqual(isolation["preexisting_host_shared_object_successful_opens"], 0)
        self.assertTrue(isolation["optional_host_metadata_absence_test"]["probe_complete"])
        prefill = result["provider_nonregression"]["prefill_hipblaslt"]
        self.assertLessEqual(prefill["bundle_to_direct_time_ratio"], 1.03)
        self.assertTrue(prefill["passed"])
        for case in result["provider_nonregression"]["decode_wvsplitk"]:
            self.assertTrue(case["all_output_hashes_equal"])
            self.assertLessEqual(case["bundle_to_direct_time_ratio"], 1.03)
            self.assertTrue(case["passed"])
        self.assertFalse(result["decision"]["system_installed_rocm_required"])
        self.assertFalse(result["decision"]["host_c_or_cxx_runtime_required"])
        self.assertTrue(result["decision"]["kernel_amdgpu_driver_required"])
        self.assertFalse(result["claims"]["full_native_inference_qualified"])

    def test_decode_schedule_is_complete_pointer_free_and_not_overclaimed(self) -> None:
        schedule = self.decode_schedule
        result = self.decode_schedule_result
        self.assertTrue(result["complete"])
        self.assertEqual(
            result["scope"],
            "qualified_q8192_single_decode_token_launch_schedule_only",
        )
        self.assertEqual(sha256(DECODE_SCHEDULE), result["schedule"]["sha256"])
        self.assertEqual(schedule["status"],
                         "qualified_schedule_contract_native_executor_wiring_pending")
        closure = schedule["closure"]
        self.assertEqual(closure["launch_count"], 402)
        self.assertEqual(closure["layer_launch_count"], 400)
        self.assertEqual(closure["final_logit_launch_count"], 2)
        self.assertEqual(closure["linear_layer_count"], 30)
        self.assertEqual(closure["full_attention_layer_count"], 10)
        self.assertEqual(closure["launches_per_layer"], 10)
        self.assertGreater(closure["registry_mapping_fraction"], 0.9)
        self.assertFalse(closure["raw_device_pointers_retained"])

        payload = DECODE_SCHEDULE.read_text(encoding="utf-8")
        for forbidden in ("data_ptr", "storage_data_ptr", "/home/", "/data/"):
            self.assertNotIn(forbidden, payload)
        manifest_hashes = {
            item["kernel_hash"] for item in self.aot_manifest["kernels"]
        }
        self.assertEqual(len(schedule["schedule"]), 402)
        self.assertTrue(
            all(item["kernel_hash"] in manifest_hashes for item in schedule["schedule"])
        )
        for layer_index in range(40):
            actual = [
                item["symbol"]
                for item in schedule["schedule"]
                if item["layer_index"] == layer_index
            ]
            expected = schedule["layer_templates"][
                "full_attention" if layer_index % 4 == 3 else "linear_attention"
            ]
            self.assertEqual(actual, expected)
        self.assertEqual(
            [item["symbol"] for item in schedule["schedule"][-2:]],
            schedule["layer_templates"]["final_logits"],
        )
        registry = result["compiled_native_registry"]
        self.assertEqual(registry["launch_count"], 402)
        self.assertEqual(registry["tensor_argument_count"], 1777)
        self.assertEqual(registry["scalar_argument_count"], 150)
        self.assertEqual(registry["embedded_kernel_matches"], 402)
        for dependency in ("python", "torch", "vllm", "triton"):
            self.assertFalse(registry[f"runtime_{dependency}"])
        generator = ROOT / registry["generator"]
        self.assertRegex(registry["generator_sha256"], r"^[0-9a-f]{64}$")
        check = subprocess.run(
            [
                "python3",
                str(generator),
                "--check",
                "--schedule",
                str(DECODE_SCHEDULE),
                "--aot-manifest",
                str(AOT_MANIFEST),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertFalse(result["claims"]["full_native_decode_executor_qualified"])
        self.assertFalse(result["claims"]["full_native_inference_qualified"])
        self.assertFalse(result["claims"]["performance_qualified"])

    def test_all_native_decode_weight_bindings_are_resident_and_exact(self) -> None:
        result = self.decode_bindings_result
        self.assertTrue(result["complete"])
        closure = result["binding_closure"]
        self.assertEqual(closure["schedule_weight_arguments"], 423)
        self.assertEqual(closure["unique_bindings"], 423)
        self.assertEqual(closure["raw_weight_bindings"], 301)
        self.assertEqual(closure["layer_derived_bindings"], 120)
        self.assertEqual(closure["lm_head_derived_bindings"], 2)
        self.assertEqual(closure["device_pointer_checks"], 423)
        self.assertEqual(closure["exact_payload_byte_checks"], 423)
        self.assertEqual(closure["unresolved_bindings"], 0)
        lm_head = result["lm_head"]
        self.assertEqual(
            lm_head["qualified_q_weight_sha256"], lm_head["native_q_weight_sha256"]
        )
        self.assertEqual(
            lm_head["qualified_scales_sha256"], lm_head["native_scales_sha256"]
        )
        residual = lm_head["residual_validation"]
        self.assertEqual(residual["rows"], 248_320)
        self.assertEqual(residual["native_below_qualified_rows"], 0)
        self.assertLess(residual["maximum_relative_inflation"], 2e-6)
        self.assertEqual(len(result["runs"]), 3)
        self.assertLessEqual(
            result["median"]["total_wall_ms"],
            self.contract["promotion_gates"]["startup"]
            ["standard_safetensors_command_to_ready_median_ms_max"],
        )
        self.assertTrue(result["claims"]["all_schedule_weight_bindings_resolved"])
        self.assertFalse(result["claims"]["full_native_decode_executor_qualified"])
        self.assertFalse(result["claims"]["api_ready_startup_qualified"])


if __name__ == "__main__":
    unittest.main()
