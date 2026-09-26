PY ?= .venv/bin/python

.PHONY: setup micpipe test test-unit live fixtures clean

setup: .venv micpipe          ## create venv, install package with Qwen + whisper backends, build mic helper
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[qwen,whisper,dev]"

.venv:
	python3 -m venv .venv

micpipe: tools/micpipe        ## build the microphone helper with swiftc
tools/micpipe: tools/micpipe.swift
	swiftc -O -o $@ $<

test:                         ## unit + integration tests; integration needs NeMo-Speech.cpp and cached models
	$(PY) -m pytest -q

test-unit:                    ## fast tests only
	$(PY) -m pytest -q -m "not integration"

live:                         ## start live transcription from the microphone; run from a real terminal
	./live.sh

fixtures:                     ## regenerate the synthetic 3-speaker test audio with macOS `say`
	$(PY) tools/make_test_audio.py

clean:
	rm -rf build *.egg-info .pytest_cache tests/fixtures/_work
