.PHONY: check test verify native-layout-check build-native build-direct-loader build-native-runtime package-native package-native-foundation

check:
	python3 -m compileall -q aima_engine tools benchmarks/shape-lab tests
	python3 -m py_compile scripts/capture-native-lm-head-reference.py scripts/check-native-wvsplitk-parity.py scripts/export-native-aot-closure.py scripts/generate-native-aot-registry.py scripts/generate-native-bundle-manifest.py scripts/generate-native-decode-registry.py scripts/generate-native-decode-schedule.py scripts/generate-native-product-result.py scripts/native_bundle_closure.py scripts/qualify-native-correctness.py scripts/qualify-native-full-matrix.py scripts/qualify-native-portable-bundle.py scripts/qualify-native-surfaces.py scripts/native_aot_trace/sitecustomize.py
	python3 scripts/generate-native-decode-registry.py --check --schedule native/aot/gfx1151/q8192-output2/decode-schedule.json --aot-manifest native/aot/gfx1151/q8192-output2/manifest.json
	python3 scripts/generate-native-decode-registry.py --phase prefill --check --schedule native/aot/gfx1151/q8192-output2/prefill-schedule.json --aot-manifest native/aot/gfx1151/q8192-output2/manifest.json
	g++ -std=c++17 -pthread -fsyntax-only benchmarks/shape-lab/native/src/striped_image_builder.cpp
	python3 scripts/generate-native-layout.py --check
	g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -DU_STATIC_IMPLEMENTATION -I /opt/rocm/include -I native/include -I native/generated -fsyntax-only native/src/main.cpp native/src/decode_schedule_probe.cpp native/src/sha256.cpp native/src/native_tokenizer.cpp
	gcc -std=c11 -fsyntax-only native/src/portable_launcher.c
	python3 -m unittest discover -s tests -p 'test_*.py'
	./aima-engine verify

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

verify:
	./aima-engine verify

native-layout-check:
	python3 scripts/generate-native-layout.py --check

build-native:
	bash scripts/build-native.sh

build-direct-loader:
	bash scripts/build-direct-loader.sh

build-native-runtime:
	bash scripts/build-native-runtime.sh

package-native: build-native-runtime build-native
	bash scripts/package-native-foundation.sh

package-native-foundation: package-native
