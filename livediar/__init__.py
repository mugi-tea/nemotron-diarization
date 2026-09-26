"""livediar: real-time "who said what" for Japanese on Apple Silicon.

Speaker diarization runs on NVIDIA Nemotron 3 Diarization through the NeMo-Speech.cpp
C API; each finalized speaker turn is transcribed separately by a pluggable ASR backend
(Qwen3-ASR or whisper on MLX by default).
"""

__version__ = "0.1.0"
