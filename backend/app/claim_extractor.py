"""Use GPT-4o to extract discrete factual claims from video content."""

from __future__ import annotations

import json
import logging
import os
import re

import openai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert fact-checking and media forensic analyst. You will receive:
1. The post caption/text written by the user (SECONDARY — may be unrelated clickbait).
2. The audio transcript spoken in the video (PRIMARY source of truth).
3. The visual scene description and on-screen text overlays (PRIMARY).

CRITICAL PRIORITY RULE:
- The VIDEO (audio transcript + visuals) is the PRIMARY source of truth.
- The caption is often unrelated commentary, hashtags, or misleading framing added by the poster.
- When caption content differs from or is unrelated to the video, IGNORE caption claims for fact-checking.
- Only fact-check what is actually SAID or SHOWN in the video.
- Set `has_caption_video_mismatch` to true when caption misrepresents, falsely frames, or is unrelated to video content.
- When mismatch is true, extract claims ONLY from audio_transcript and visual_overlay — NOT from caption.

Your job:
1. Context Breakdown:
1. Context Breakdown & Source Identification:
   - `caption_summary`: 1 sentence summarizing what the caption/poster is asserting.
   - `video_actual_context`: 2-3 sentences explaining what the video ACTUALLY shows and what was SAID.
   - `original_full_video_title`: The name of the original full-length interview, speech, press conference, or video (e.g., 'Narendra Modi with Akshay Kumar Interview (2019)', 'Vogue World 2024 Livestream', or null if amateur/unknown).
   - `original_full_video_url`: A direct link or YouTube search URL to watch the full uncut video (e.g., 'https://www.youtube.com/results?search_query=Narendra+Modi+Akshay+Kumar+Interview+Canvas+Shoes' or direct YouTube URL, or null if unknown).

2. Cross-Modal Consistency Check:
   - Carefully check if the caption MATCHES or CONTRADICTS what is actually in the video.
   - Set `has_caption_video_mismatch` to true if the caption misrepresents, falsely frames, or is unrelated to the video.
   - Provide `mismatch_summary` explaining the discrepancy.

3. Claim Extraction:
   - Extract discrete, verifiable factual claims FROM THE VIDEO (speech and on-screen text).
   - For each claim, identify its `source_origin` ('audio_transcript', 'visual_overlay', 'caption', 'both').

Return a JSON object with this schema:
{
  "caption_summary": "string",
  "video_actual_context": "string",
  "original_full_video_title": "string or null",
  "original_full_video_url": "string or null",
  "has_caption_video_mismatch": false,
  "mismatch_summary": "string or null",
  "claims": [
    {
      "claim_text": "string — the factual claim",
      "source_origin": "caption | audio_transcript | visual_overlay | both",
      "mismatch_warning": "string or null"
    }
  ]
}
"""


def extract_claims(
    caption: str,
    transcript: str,
    visual_summary: str | None = None,
    on_screen_text: list[str] | None = None,
    speech_summary: str | None = None,
    speakers: list[str] | None = None,
) -> dict:
    """Extract factual claims, context breakdown, and cross-modal consistency analysis.

    Returns a dict containing:
      - claims: list of claim dicts
      - caption_summary: str | None
      - video_actual_context: str | None
      - has_caption_video_mismatch: bool
      - mismatch_summary: str | None
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("OPENAI_MODEL", "gemini-flash-latest")

    user_content = _build_user_message(
        caption, transcript, visual_summary, on_screen_text, speech_summary, speakers
    )
    if not user_content.strip():
        logger.warning("No content provided — returning empty claims")
        return {
            "claims": [],
            "caption_summary": None,
            "video_actual_context": None,
            "has_caption_video_mismatch": False,
            "mismatch_summary": None,
        }

    logger.info("Extracting claims & context with %s (input ~%d chars) ...", model, len(user_content))

    raw = ""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
    except Exception as exc:
        logger.warning("Call with json_object failed (%s), retrying standard prompt...", exc)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"

    data = _parse_json_safely(raw)
    claims = data.get("claims", [])
    has_mismatch = bool(data.get("has_caption_video_mismatch", False))
    mismatch_summary = data.get("mismatch_summary")
    caption_summary = data.get("caption_summary")
    video_actual_context = data.get("video_actual_context")
    
    logger.info("Extracted %d claims (mismatch detected: %s)", len(claims), has_mismatch)
    caption_summary = data.get("caption_summary")
    video_actual_context = data.get("video_actual_context")
    original_full_video_title = data.get("original_full_video_title")
    original_full_video_url = data.get("original_full_video_url")
    
    logger.info("Extracted %d claims (mismatch detected: %s, full video: %s)", len(claims), has_mismatch, original_full_video_title)
    return {
        "claims": claims,
        "caption_summary": caption_summary,
        "video_actual_context": video_actual_context,
        "original_full_video_title": original_full_video_title,
        "original_full_video_url": original_full_video_url,
        "has_caption_video_mismatch": has_mismatch,
        "mismatch_summary": mismatch_summary,
    }


def _parse_json_safely(raw: str) -> dict:
    """Strip code fences or extract JSON substring if present."""
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
        logger.error("Failed to parse JSON from LLM response: %s", raw[:300])
        return {}


def _build_user_message(
    caption: str,
    transcript: str,
    visual_summary: str | None = None,
    on_screen_text: list[str] | None = None,
    speech_summary: str | None = None,
    speakers: list[str] | None = None,
) -> str:
    """Build the user message combining caption, transcript, and visual scene analysis."""
    parts: list[str] = []

    # Video content first (primary)
    if speakers:
        parts.append(f"=== IDENTIFIED SPEAKERS ===\n{', '.join(speakers)}")
    if speech_summary and speech_summary.strip():
        parts.append(f"=== SPEECH SUMMARY (what was actually said) ===\n{speech_summary.strip()}")
    if transcript.strip():
        parts.append(f"=== AUDIO TRANSCRIPT (PRIMARY — verbatim speech) ===\n{transcript.strip()}")
    if visual_summary and visual_summary.strip():
        parts.append(f"=== VISUAL SCENE ===\n{visual_summary.strip()}")
    if on_screen_text:
        parts.append(f"=== ON-SCREEN TEXT OVERLAYS ===\n" + "\n".join(f"- {t}" for t in on_screen_text if t))

    # Caption last (secondary — may be misleading)
    if caption.strip():
        parts.append(
            f"=== POST CAPTION (SECONDARY — may be unrelated to video) ===\n{caption.strip()}"
        )
    return "\n\n".join(parts)
