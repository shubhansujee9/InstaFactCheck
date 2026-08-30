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

Your job:
1. Cross-Modal Consistency Check:
   - Carefully check if the caption MATCHES or CONTRADICTS the actual video audio content.
   - Detect false context / clickbait (e.g., caption says "Breaking war footage!" but audio is an old movie clip or drill; or caption claims a politician said something they never uttered).
   - Set `has_caption_video_mismatch` to true if the caption misrepresents, falsely frames, or contradicts what is in the audio.
   - Provide a `mismatch_summary` describing the discrepancy if present.

2. Claim Extraction:
   - Extract discrete, verifiable factual claims.
   - For each claim, identify its `source_origin`:
     - "caption": asserted only in the caption/text
     - "audio_transcript": spoken in the video audio
     - "both": mentioned in both caption and audio
   - If a claim from the caption is contradicted by the video audio or falsely attributed to the video, add a `mismatch_warning` explaining the contradiction.

Rules:
- Only extract claims that can be verified against facts (statistics, named events, quotes, historical/scientific claims).
- Do NOT extract opinions or subjective banter.
- Return ONLY valid JSON.

Return a JSON object with this schema:
{
  "has_caption_video_mismatch": false,
  "mismatch_summary": "string or null",
  "claims": [
    {
      "claim_text": "string — the factual claim",
      "source_origin": "caption | audio_transcript | both",
      "mismatch_warning": "string or null"
    }
  ]
}
"""


def extract_claims(caption: str, transcript: str) -> dict:
    """Extract factual claims and cross-modal consistency analysis using LLM.

    Returns a dict containing:
      - claims: list of claim dicts (with claim_text, source_origin, mismatch_warning)
      - has_caption_video_mismatch: bool
      - mismatch_summary: str | None
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("OPENAI_MODEL", "gemini-flash-latest")

    user_content = _build_user_message(caption, transcript)
    if not user_content.strip():
        logger.warning("No caption or transcript provided — returning empty claims")
        return {"claims": [], "has_caption_video_mismatch": False, "mismatch_summary": None}

    logger.info("Extracting claims & checking consistency with %s (input ~%d chars) ...", model, len(user_content))

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
    
    logger.info("Extracted %d claims (mismatch detected: %s)", len(claims), has_mismatch)
    return {
        "claims": claims,
        "has_caption_video_mismatch": has_mismatch,
        "mismatch_summary": mismatch_summary,
    }


def _parse_json_safely(raw: str) -> dict:
    """Strip code fences or extract JSON substring if present."""
    text = raw.strip()
    # Strip markdown code fences like ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding outermost { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.error("Failed to parse JSON from LLM response: %s", raw[:300])
        return {}


def _build_user_message(caption: str, transcript: str) -> str:
    """Build the user message combining caption and transcript."""
    parts: list[str] = []
    if caption.strip():
        parts.append(f"=== VIDEO CAPTION ===\n{caption.strip()}")
    if transcript.strip():
        parts.append(f"=== AUDIO TRANSCRIPT ===\n{transcript.strip()}")
    return "\n\n".join(parts)
