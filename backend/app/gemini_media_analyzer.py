"""Native Gemini multimodal analysis: transcribe speech and understand video content."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MEDIA_ANALYSIS_PROMPT = """\
You are an expert multilingual media forensic analyst and fact-checker specializing in Indian political and news content.

Watch and listen to this Instagram video carefully. The poster's caption is:
<POST_CAPTION>
{caption}
</POST_CAPTION>

Your job is to perform a comprehensive multimodal analysis in ONE SINGLE PASS:
1. AUDIO & SPEECH: Transcribe ALL spoken words verbatim (Hindi/English/mixed), identify speakers, and provide an English summary.
2. VISUALS: Describe visual scenes, actions, on-screen text overlays, genre, and whether the clip is selectively clipped or satire.
3. CONTEXT & CONSISTENCY:
   - `caption_summary`: 1 sentence summarizing what the caption asserts (or note 'Generic hashtags/emojis' if no factual claim).
   - `video_actual_context`: 2-3 sentences explaining what the video ACTUALLY shows and what was SAID.
   - `original_full_video_title`: Name of the original full-length interview, speech, press conference, or video (or null if amateur/unknown).
   - `original_full_video_url`: YouTube search query URL or direct URL to watch the full uncut video (e.g. 'https://www.youtube.com/results?search_query=...').
   - `has_caption_video_mismatch`: true ONLY if the caption makes an active false or contradictory claim about the video (false for generic emojis/hashtags).
   - `mismatch_summary`: Description of discrepancy if present, or null.
4. CLAIM EXTRACTION:
   - Extract 2 to 4 of the most important, discrete, verifiable factual assertions made in the video.
   - For each claim, identify its `source_origin` ('audio_transcript', 'visual_overlay', 'caption', 'both').

Return ONLY a valid JSON object with this schema:
{
  "transcript": "Verbatim transcript of ALL spoken words in the video. Preserve the original language.",
  "transcript_english": "English translation/summary of what was spoken.",
  "speakers": ["Names or descriptions of who is speaking, e.g. 'PM Narendra Modi'"],
  "speech_summary": "2-3 sentences summarizing the main points actually spoken in the video.",
  "visual_summary": "1-2 sentences describing what is visually shown (who, where, actions).",
  "on_screen_text": ["All text overlays, subtitles, banners visible on screen"],
  "content_genre": "e.g. political_speech, news_clip, interview, satire, meme",
  "is_selectively_clipped": false,
  "original_context_note": "Brief note on the event/setting if identifiable, or null",
  "caption_summary": "1 sentence summarizing caption",
  "video_actual_context": "2-3 sentences explaining actual video context",
  "original_full_video_title": "string or null",
  "original_full_video_url": "string or null",
  "has_caption_video_mismatch": false,
  "mismatch_summary": null,
  "claims": [
    {
      "claim_text": "string — discrete verifiable factual claim",
      "source_origin": "audio_transcript | visual_overlay | caption | both",
      "mismatch_warning": "string or null"
    }
  ]
}
"""


@dataclass
class GeminiMediaAnalysis:
    """Structured output from Gemini native video/audio analysis."""

    transcript: str = ""
    transcript_english: str = ""
    speakers: list[str] = field(default_factory=list)
    speech_summary: str = ""
    visual_summary: str = ""
    on_screen_text: list[str] = field(default_factory=list)
    content_genre: str | None = None
    is_selectively_clipped: bool = False
    original_context_note: str | None = None
    caption_summary: str | None = None
    video_actual_context: str | None = None
    original_full_video_title: str | None = None
    original_full_video_url: str | None = None
    has_caption_video_mismatch: bool = False
    mismatch_summary: str | None = None
    claims: list[dict] = field(default_factory=list)


def _get_api_key() -> str | None:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def _get_model_name() -> str:
    return os.environ.get("GEMINI_MEDIA_MODEL", "gemini-2.5-flash")


def analyze_video_with_gemini(video_path: Path, caption: str = "") -> GeminiMediaAnalysis | None:
    """Upload video to Gemini and extract transcript + claims in a single unified pass.

    Returns None if Gemini is unavailable or analysis fails.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("No Gemini/Google API key — skipping native video analysis")
        return None

    if not video_path.exists() or video_path.stat().st_size == 0:
        logger.warning("Video file missing or empty: %s", video_path)
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai not installed — skipping native video analysis")
        return None

    uploaded = None
    try:
        genai.configure(api_key=api_key)
        model_name = _get_model_name()
        logger.info("Uploading video to Gemini (%s) for unified multimodal analysis...", model_name)

        uploaded = genai.upload_file(path=str(video_path))

        deadline = time.monotonic() + 120
        while uploaded.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                logger.warning("Gemini file processing timed out")
                return None
            time.sleep(0.8)
            uploaded = genai.get_file(uploaded.name)

        if uploaded.state.name != "ACTIVE":
            logger.warning("Gemini file not active (state=%s)", uploaded.state.name)
            return None

        prompt = MEDIA_ANALYSIS_PROMPT.replace("{caption}", caption.strip() or "(No caption provided)")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            [uploaded, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw = response.text or "{}"
        data = _parse_json_safely(raw)

        transcript = (data.get("transcript") or "").strip()
        transcript_en = (data.get("transcript_english") or "").strip()

        # Prefer verbatim transcript; append English summary for downstream LLM context
        full_transcript = transcript
        if transcript_en and transcript_en.lower() != transcript.lower():
            full_transcript = f"{transcript}\n\n[English summary: {transcript_en}]"

        result = GeminiMediaAnalysis(
            transcript=full_transcript,
            transcript_english=transcript_en,
            speakers=data.get("speakers") or [],
            speech_summary=(data.get("speech_summary") or "").strip(),
            visual_summary=(data.get("visual_summary") or "").strip(),
            on_screen_text=data.get("on_screen_text") or [],
            content_genre=data.get("content_genre"),
            is_selectively_clipped=bool(data.get("is_selectively_clipped", False)),
            original_context_note=data.get("original_context_note"),
            caption_summary=data.get("caption_summary"),
            video_actual_context=data.get("video_actual_context"),
            original_full_video_title=data.get("original_full_video_title"),
            original_full_video_url=data.get("original_full_video_url"),
            has_caption_video_mismatch=bool(data.get("has_caption_video_mismatch", False)),
            mismatch_summary=data.get("mismatch_summary"),
            claims=data.get("claims") or [],
        )

        logger.info(
            "Gemini unified analysis complete — %d claims extracted, transcript=%d chars, speakers=%s",
            len(result.claims),
            len(result.transcript),
            result.speakers,
        )
        return result

        logger.info(
            "Gemini media analysis complete — speakers=%s, transcript=%d chars, speech_summary=%s",
            result.speakers,
            len(result.transcript),
            result.speech_summary[:80] if result.speech_summary else "(none)",
        )
        return result

    except Exception as exc:
        logger.warning("Gemini native video analysis failed (%s)", exc)
        return None
    finally:
        if uploaded is not None:
            try:
                import google.generativeai as genai
                genai.delete_file(uploaded.name)
            except Exception:
                pass


def _parse_json_safely(raw: str) -> dict:
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
