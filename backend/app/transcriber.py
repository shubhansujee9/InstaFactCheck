"""Extract audio from video via ffmpeg and transcribe with OpenAI Whisper API."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import openai


def _find_executable(name: str) -> str:
    """Find an executable, checking the current Python env's bin dir first."""
    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"'{name}' not found in venv or system PATH")

logger = logging.getLogger(__name__)

# Max audio duration to send to Whisper (seconds).  Reels are usually <90s,
# but we cap just in case to avoid huge API bills.
MAX_AUDIO_DURATION_S = 300


def extract_audio(video_path: Path) -> Path:
    """Extract audio from a video file into a WAV file using ffmpeg.

    Returns the path to the extracted audio file (same dir, .wav extension).
    """
    audio_path = video_path.with_suffix(".wav")
    ffmpeg_bin = _find_executable("ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",                   # overwrite if exists
        "-i", str(video_path),
        "-vn",                  # drop video stream
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16 kHz mono — Whisper optimal
        "-ac", "1",
        "-t", str(MAX_AUDIO_DURATION_S),
        str(audio_path),
    ]
    logger.info("Extracting audio: %s → %s", video_path.name, audio_path.name)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg audio extraction timed out") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FileNotFoundError("ffmpeg produced no audio output")

    logger.info("Audio extracted: %s (%.1f KB)", audio_path.name, audio_path.stat().st_size / 1024)
    return audio_path


_whisper_model_instance = None


def _get_local_whisper_model():
    """Lazy-load faster-whisper model singleton."""
    global _whisper_model_instance
    if _whisper_model_instance is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Initializing local faster-whisper model (tiny, int8)...")
            _whisper_model_instance = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception as exc:
            logger.warning("Could not initialize local faster-whisper (%s)", exc)
            return None
    return _whisper_model_instance


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe an audio file using local faster-whisper with OpenAI API fallback.

    Returns the transcript text.
    """
    # 1. Try local faster-whisper first (free, fast, no external API dependency)
    model = _get_local_whisper_model()
    if model is not None:
        try:
            logger.info("Transcribing audio with local faster-whisper...")
            segments, info = model.transcribe(str(audio_path), beam_size=5)
            transcript = " ".join([seg.text for seg in segments]).strip()
            if transcript:
                logger.info("Local Whisper transcription complete (%s, %d chars): %s", info.language, len(transcript), transcript[:100])
                return transcript
        except Exception as exc:
            logger.warning("Local faster-whisper failed (%s), trying cloud API...", exc)

    # 2. Fallback to OpenAI Whisper API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""

    base_url = os.environ.get("OPENAI_BASE_URL")
    # If using Google or other providers that do not support /audio/transcriptions, skip
    if base_url and "generativelanguage.googleapis.com" in base_url:
        return ""

    try:
        client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
        whisper_model = os.environ.get("WHISPER_MODEL", "whisper-1")
        logger.info("Transcribing with cloud Whisper model=%s ...", whisper_model)

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=whisper_model,
                file=audio_file,
                response_format="text",
            )
        return response.strip() if isinstance(response, str) else str(response).strip()
    except Exception as exc:
        logger.warning("Cloud transcription failed (%s)", exc)
        return ""


def get_transcript(video_path: Path) -> str:
    """High-level helper: extract audio then transcribe.

    Returns transcript text. Cleans up the intermediate audio file.
    """
    audio_path = extract_audio(video_path)
    try:
        return transcribe_audio(audio_path)
    finally:
        # Clean up the intermediate WAV file
        try:
            audio_path.unlink()
        except OSError:
            pass
