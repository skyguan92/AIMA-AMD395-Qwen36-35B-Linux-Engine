.PHONY: check check-cpu check-native-syntax check-python-package security-scan verify-evidence test verify native-layout-check native-chat-template-parity build-native build-direct-loader build-native-runtime package-native package-native-foundation package-evidence

check: check-cpu
	@if test -f /opt/rocm/include/hip/hip_runtime_api.h && \
	    test -f /usr/include/nlohmann/json.hpp; then \
		$(MAKE) --no-print-directory check-native-syntax; \
	else \
		echo "native HIP/protocol syntax: SKIP (builder headers unavailable)"; \
	fi

check-cpu:
	python3 -m compileall -q aima_engine tools benchmarks/shape-lab tests
	python3 -m py_compile scripts/capture-native-lm-head-reference.py scripts/check-native-wvsplitk-parity.py scripts/check-public-tree.py scripts/export-native-aot-closure.py scripts/generate-native-aot-registry.py scripts/generate-native-bundle-manifest.py scripts/generate-native-decode-registry.py scripts/generate-native-decode-schedule.py scripts/generate-native-product-result.py scripts/native_bundle_closure.py scripts/package-release-evidence.py scripts/qualify-native-correctness.py scripts/qualify-native-eval.py scripts/qualify-native-full-matrix.py scripts/qualify-native-openai-features.py scripts/qualify-native-portable-bundle.py scripts/qualify-native-surfaces.py scripts/verify-native-package-inputs.py scripts/verify-release-evidence.py scripts/native_aot_trace/sitecustomize.py
	python3 scripts/check-public-tree.py
	python3 scripts/verify-release-evidence.py
	python3 scripts/generate-native-decode-registry.py --check --schedule native/aot/gfx1151/q8192-output2/decode-schedule.json --aot-manifest native/aot/gfx1151/q8192-output2/manifest.json
	python3 scripts/generate-native-decode-registry.py --phase prefill --check --schedule native/aot/gfx1151/q8192-output2/prefill-schedule.json --aot-manifest native/aot/gfx1151/q8192-output2/manifest.json
	g++ -std=c++17 -pthread -fsyntax-only benchmarks/shape-lab/native/src/striped_image_builder.cpp
	python3 scripts/generate-native-layout.py --check
	gcc -std=c11 -fsyntax-only native/src/portable_launcher.c
	python3 -m unittest discover -s tests -p 'test_*.py'
	./aima-engine verify
	$(MAKE) check-python-package

check-python-package:
	python3 -m pip wheel --no-deps --wheel-dir build/wheel .

check-native-syntax:
	mkdir -p build && g++ -std=c++17 -O2 -I native/include tests/native_chat_protocol_test.cpp native/src/native_chat_protocol.cpp -o build/native_chat_protocol_test
	./build/native_chat_protocol_test
	g++ -std=c++17 -O2 -I native/include tests/native_prompt_plan_test.cpp -o build/native_prompt_plan_test
	./build/native_prompt_plan_test
	g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -DU_STATIC_IMPLEMENTATION -I /opt/rocm/include -I native/include -I native/generated -fsyntax-only native/src/main.cpp native/src/decode_schedule_probe.cpp native/src/sha256.cpp native/src/native_tokenizer.cpp native/src/native_chat_protocol.cpp native/src/native_doctor.cpp native/src/native_http_server.cpp

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

verify:
	./aima-engine verify

security-scan:
	python3 scripts/check-public-tree.py

verify-evidence:
	python3 scripts/verify-release-evidence.py

native-layout-check:
	python3 scripts/generate-native-layout.py --check

native-chat-template-parity:
	mkdir -p build && g++ -std=c++17 -O2 -I native/include -I native/generated tests/native_chat_template_parity.cpp native/src/native_chat_protocol.cpp native/src/native_tokenizer.cpp native/src/sha256.cpp $$(pkg-config --cflags --libs icu-i18n) -o build/native_chat_template_parity
	./build/native_chat_template_parity "$${AIMA_MODEL_DIR:?set AIMA_MODEL_DIR}"

build-native:
	bash scripts/build-native.sh

build-direct-loader:
	bash scripts/build-direct-loader.sh

build-native-runtime:
	bash scripts/build-native-runtime.sh

package-native: build-native-runtime build-native
	bash scripts/package-native-foundation.sh

package-native-foundation: package-native

package-evidence:
	python3 scripts/package-release-evidence.py
