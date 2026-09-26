import importlib.util
import os

import pytest

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture(scope="session")
def fixtures_dir() -> str:
    return FIXTURES


@pytest.fixture(scope="session")
def meeting_wav(fixtures_dir) -> str:
    return os.path.join(fixtures_dir, "meeting_ja_3spk.wav")


def diarizer_available() -> bool:
    from livediar.diarizer import LIBRARY, DiarizerUnavailable, find_model

    if not os.path.exists(LIBRARY):
        return False
    try:
        find_model()
    except DiarizerUnavailable:
        return False
    return True


requires_diarizer = pytest.mark.skipif(not diarizer_available(), reason="NeMo-Speech.cpp library or model not installed")
requires_qwen = pytest.mark.skipif(not has_module("mlx_audio"), reason="mlx-audio not installed")
requires_whisper = pytest.mark.skipif(not has_module("mlx_whisper"), reason="mlx-whisper not installed")
