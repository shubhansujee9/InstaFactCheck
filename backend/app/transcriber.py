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


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe an audio file using OpenAI Whisper API.

    Returns the transcript text.  Requires OPENAI_API_KEY env var.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("WHISPER_MODEL", "whisper-1")

    logger.info("Transcribing with Whisper model=%s ...", model)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
        )

    transcript = response.strip() if isinstance(response, str) else str(response).strip()
    logger.info("Transcript received: %d chars", len(transcript))
    return transcript


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
