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
1. The post caption/text written by the user.
2. The audio transcript spoken in the video (if available).
3. The visual scene description and on-screen text overlays (if available).

Your job:
1. Context Breakdown:
   - `caption_summary`: 1 sentence summarizing what the caption/poster is asserting.
   - `video_actual_context`: 1-2 sentences explaining what the video actually shows and discusses in reality.

2. Cross-Modal Consistency Check:
   - Carefully check if the caption MATCHES or CONTRADICTS what is actually in the video (audio or visuals).
   - Detect false context / clickbait (e.g. caption says "War breaks out" but video is an old drill or video game; or caption misattributes speech).
   - Set `has_caption_video_mismatch` to true if the caption misrepresents, falsely frames, or contradicts what is in the video.
   - Provide `mismatch_summary` explaining the discrepancy.

3. Claim Extraction:
   - Extract discrete, verifiable factual claims.
   - For each claim, identify its `source_origin`:
     - "caption": asserted only in the caption/text
     - "audio_transcript": spoken in the video audio
     - "visual_overlay": shown in on-screen video text/graphics
     - "both": mentioned in both caption and video
   - If a claim from the caption is contradicted by the video or falsely attributed to the video, add a `mismatch_warning`.

Rules:
- Only extract claims that can be verified against facts (statistics, named events, quotes, historical/scientific claims).
- Do NOT extract opinions or subjective banter.
- Return ONLY valid JSON.

Return a JSON object with this schema:
{
  "caption_summary": "string",
  "video_actual_context": "string",
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

    user_content = _build_user_message(caption, transcript, visual_summary, on_screen_text)
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
    return {
        "claims": claims,
        "caption_summary": caption_summary,
        "video_actual_context": video_actual_context,
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
) -> str:
    """Build the user message combining caption, transcript, and visual scene analysis."""
    parts: list[str] = []
    if caption.strip():
        parts.append(f"=== VIDEO CAPTION ===\n{caption.strip()}")
    if transcript.strip():
        parts.append(f"=== AUDIO TRANSCRIPT ===\n{transcript.strip()}")
    if visual_summary and visual_summary.strip():
        parts.append(f"=== VISUAL SCENE FORENSICS ===\n{visual_summary.strip()}")
    if on_screen_text:
        parts.append(f"=== ON-SCREEN TEXT OVERLAYS ===\n" + "\n".join(f"- {t}" for t in on_screen_text if t))
    return "\n\n".join(parts)
