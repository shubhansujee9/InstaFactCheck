"""Use GPT-4o to extract discrete factual claims from video content."""

from __future__ import annotations

import json
import logging
import os
import re

import openai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a fact-checking analyst. You will receive the caption and audio transcript
of a social media video. Your job is to extract every discrete, verifiable factual
claim made in the content.

Rules:
- Only extract claims that can be verified against external evidence (statistics,
  historical facts, scientific claims, named events, quotes attributed to people, etc.).
- Do NOT extract opinions, subjective statements, or humour.
- Phrase each claim as a clear, standalone assertion.
- If the content contains no verifiable claims, return an empty list.
- Return ONLY valid JSON — no markdown fences, no commentary.

Return a JSON object with this schema:
{
  "claims": [
    {"claim_text": "string — the factual claim"}
  ]
}
"""


def extract_claims(caption: str, transcript: str) -> list[dict]:
    """Extract factual claims from caption + transcript using GPT-4o.

    Returns a list of dicts, each with at least a ``claim_text`` key.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("OPENAI_MODEL", "auto/best-chat")

    user_content = _build_user_message(caption, transcript)
    if not user_content.strip():
        logger.warning("No caption or transcript provided — returning empty claims")
        return []

    logger.info("Extracting claims with %s (input ~%d chars) ...", model, len(user_content))

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
    logger.info("Extracted %d claims", len(claims))
    return claims


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
