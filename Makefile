.PHONY: check test verify build-native

check:
	python3 -m compileall -q aima_engine tools benchmarks/shape-lab tests
	g++ -std=c++17 -pthread -fsyntax-only benchmarks/shape-lab/native/src/striped_image_builder.cpp
	python3 -m unittest discover -s tests -p 'test_*.py'
	./aima-engine verify

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

verify:
	./aima-engine verify

build-native:
	bash scripts/build-native.sh
