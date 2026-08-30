"""Multimodal video forensic analyzer: extracts keyframes and analyzes visual + synthetic AI anomalies."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import openai
from app.schemas import MediaForensics

logger = logging.getLogger(__name__)


def _find_executable(name: str) -> str:
    """Find an executable in venv or system PATH."""
    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"'{name}' not found in venv or system PATH")


def extract_keyframes(video_path: Path, max_frames: int = 3) -> list[Path]:
    """Extract representative keyframe images from the video using ffmpeg."""
    frame_paths: list[Path] = []
    ffmpeg_bin = _find_executable("ffmpeg")

    # Extract frames at 1s, 4s, 8s (or proportional timestamps)
    timestamps = ["00:00:01", "00:00:04", "00:00:08"][:max_frames]

    for idx, ts in enumerate(timestamps, 1):
        frame_file = video_path.parent / f"keyframe_{idx}.jpg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", ts,
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "3",
            str(frame_file),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and frame_file.exists() and frame_file.stat().st_size > 0:
                frame_paths.append(frame_file)
        except Exception as exc:
            logger.warning("Failed to extract keyframe at %s: %s", ts, exc)

    logger.info("Extracted %d keyframes from %s", len(frame_paths), video_path.name)
    return frame_paths


def _encode_image_base64(image_path: Path) -> str:
    """Encode an image file to base64 data URL."""
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


FORENSIC_PROMPT = """\
You are an expert digital media forensics and video analysis AI.
You are inspecting representative video frames from a social media reel/post.

Analyze the visual evidence and return ONLY a JSON object:
1. `visual_summary`: 1-2 sentence description of what is visually shown (who is in the video, location, actions).
2. `on_screen_text`: List of all text overlays, subtitles, captions, or watermarks visible on screen.
3. `is_likely_ai_generated`: Boolean (true if visual anomalies indicate deepfake, AI voice-clone video, Sora/Kling/Midjourney generation, or digital face swap).
4. `ai_confidence`: "high" | "medium" | "low" | "none".
5. `ai_indicators`: List of specific visual cues (e.g. "unnatural skin smoothing", "warping fingers/teeth", "synthetic background geometry", "face boundary blending", "no anomalies found").
6. `content_genre`: Media type (e.g. "news_broadcast", "political_speech", "movie_clip", "satire_comedy", "personal_vlog", "educational").
7. `is_selectively_clipped`: Boolean (true if this looks like a clipped excerpt of a larger speech or event).
8. `original_context_note`: Brief note on the suspected origin or setting of the video.

Schema:
{
  "visual_summary": "string",
  "on_screen_text": ["string"],
  "is_likely_ai_generated": false,
  "ai_confidence": "none",
  "ai_indicators": [],
  "content_genre": "string",
  "is_selectively_clipped": false,
  "original_context_note": "string or null"
}
"""


def analyze_video_frames(video_path: Path) -> MediaForensics:
    """Extract keyframes and perform multimodal forensic & visual inspection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return MediaForensics()

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("OPENAI_MODEL", "gemini-flash-latest")

    keyframes = extract_keyframes(video_path, max_frames=3)
    if not keyframes:
        return MediaForensics()

    try:
        # Build multimodal image message
        content_parts: list[dict] = [{"type": "text", "text": FORENSIC_PROMPT}]
        for kf in keyframes:
            b64_url = _encode_image_base64(kf)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": b64_url},
            })

        logger.info("Running visual forensic analysis with %s on %d keyframes...", model, len(keyframes))

        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content_parts}],
        )
        raw = resp.choices[0].message.content or "{}"
        data = _parse_json_safely(raw)

        return MediaForensics(
            is_likely_ai_generated=bool(data.get("is_likely_ai_generated", False)),
            ai_confidence=str(data.get("ai_confidence", "none")),
            ai_indicators=data.get("ai_indicators", []),
            visual_summary=data.get("visual_summary"),
            on_screen_text=data.get("on_screen_text", []),
            content_genre=data.get("content_genre"),
            is_selectively_clipped=bool(data.get("is_selectively_clipped", False)),
            original_context_note=data.get("original_context_note"),
        )
    except Exception as exc:
        logger.warning("Visual forensic analysis failed (%s) — continuing with audio/caption", exc)
        return MediaForensics()
    finally:
        # Clean up temporary keyframe image files
        for kf in keyframes:
            try:
                kf.unlink()
            except OSError:
                pass


def _parse_json_safely(raw: str) -> dict:
    """Safely parse JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}
