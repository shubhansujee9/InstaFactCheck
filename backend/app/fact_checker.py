import json
import logging
import os
import re

import openai
from tavily import TavilyClient

from .schemas import Claim, Source, Verdict

logger = logging.getLogger(__name__)

VERDICT_PROMPT = """\
You are a rigorous fact-checker. You will receive:
1. A factual claim.
2. Web search results gathered as evidence (if available) or evaluate using established consensus facts.

Your job:
- Evaluate whether the claim is true, mostly_true, mixed, misleading, false,
  or unverifiable based on facts and evidence.
- Write a concise 1-3 sentence explanation of your reasoning.
- List the most relevant sources or domains (title + URL) if available.

Return a JSON object with this schema:
{
  "verdict": "true | mostly_true | mixed | misleading | false | unverifiable",
  "explanation": "string",
  "sources": [
    {"title": "string", "url": "string"}
  ]
}
"""


def search_evidence(claim_text: str) -> list[dict]:
    """Search the web for live evidence about a claim.

    Uses Tavily if TAVILY_API_KEY is present; otherwise falls back to free DuckDuckGo search.
    Returns a list of search result dicts with ``title``, ``url``, ``content``.
    """
    # 1. Try Tavily if configured
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            client = TavilyClient(api_key=tavily_key)
            logger.info("Searching Tavily for claim: %s", claim_text[:80])
            response = client.search(
                query=f"fact check: {claim_text}",
                search_depth="advanced",
                max_results=5,
            )
            results = response.get("results", [])
            logger.info("Tavily returned %d search results", len(results))
            return results
        except Exception as exc:
            logger.warning("Tavily search failed (%s), falling back to DuckDuckGo...", exc)

    # 2. Free live web search via DuckDuckGo (0 API keys needed)
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        query = f"{claim_text}"
        logger.info("Performing free live web search (DDGS) for claim: %s", claim_text[:80])
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=5))
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
            }
            for r in raw_results
            if r.get("href")
        ]
        logger.info("Free live web search found %d results", len(results))
        return results
    except Exception as exc:
        logger.warning("Live web search unavailable (%s) — using LLM knowledge", exc)
        return []


def evaluate_claim(
    claim_text: str,
    evidence: list[dict],
    source_origin: str = "content",
    mismatch_warning: str | None = None,
) -> Claim:
    """Use configured LLM to evaluate a claim against search evidence and contextual facts.

    Returns a fully populated Claim object.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("OPENAI_MODEL", "gemini-flash-latest")

    # Format evidence for the prompt
    evidence_text = _format_evidence(evidence)

    parts = [f"=== CLAIM ===\n{claim_text}\n(Claim Origin: {source_origin})"]
    if mismatch_warning:
        parts.append(f"=== CONTEXTUAL DISCREPANCY WARNING ===\n{mismatch_warning}")
    parts.append(f"=== EVIDENCE ===\n{evidence_text}")
    user_content = "\n\n".join(parts)

    logger.info("Evaluating claim with %s ...", model)

    raw = ""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VERDICT_PROMPT},
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
                {"role": "system", "content": VERDICT_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"

    data = _parse_json_safely(raw)

    # Parse verdict
    verdict_str = data.get("verdict", "unverifiable").lower().strip()
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        logger.warning("Unknown verdict '%s', defaulting to unverifiable", verdict_str)
        verdict = Verdict.UNVERIFIABLE

    # Parse sources
    sources = [
        Source(title=s.get("title", ""), url=s.get("url", ""))
        for s in data.get("sources", [])
        if s.get("url")
    ]

    return Claim(
        claim_text=claim_text,
        verdict=verdict,
        explanation=data.get("explanation", "Could not determine verdict."),
        source_origin=source_origin,
        mismatch_warning=mismatch_warning,
        sources=sources,
    )


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
        return {}


async def search_evidence_async(claim_text: str) -> list[dict]:
    """Search for live web evidence asynchronously."""
    import asyncio
    try:
        return await asyncio.to_thread(search_evidence, claim_text)
    except Exception as exc:
        logger.warning("Async search error for '%s': %s", claim_text[:50], exc)
        return []


async def evaluate_claim_async(
    claim_text: str,
    evidence: list[dict],
    source_origin: str = "content",
    mismatch_warning: str | None = None,
) -> Claim:
    """Evaluate a claim asynchronously using AsyncOpenAI / Gemini."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    async_client = openai.AsyncOpenAI(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))
    model = os.environ.get("OPENAI_MODEL", "gemini-flash-latest")

    evidence_text = _format_evidence(evidence)
    parts = [f"=== CLAIM ===\n{claim_text}\n(Claim Origin: {source_origin})"]
    if mismatch_warning:
        parts.append(f"=== CONTEXTUAL DISCREPANCY WARNING ===\n{mismatch_warning}")
    parts.append(f"=== EVIDENCE ===\n{evidence_text}")
    user_content = "\n\n".join(parts)

    logger.info("Evaluating claim asynchronously with %s: %s", model, claim_text[:60])

    raw = ""
    try:
        response = await async_client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VERDICT_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
    except Exception as exc:
        logger.warning("Async call with json_object failed (%s), retrying standard...", exc)
        response = await async_client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": VERDICT_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"

    data = _parse_json_safely(raw)

    verdict_str = data.get("verdict", "unverifiable").lower().strip()
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.UNVERIFIABLE

    sources = [
        Source(title=s.get("title", ""), url=s.get("url", ""))
        for s in data.get("sources", [])
        if s.get("url")
    ]

    return Claim(
        claim_text=claim_text,
        verdict=verdict,
        explanation=data.get("explanation", "Could not determine verdict."),
        source_origin=source_origin,
        mismatch_warning=mismatch_warning,
        sources=sources,
    )


async def check_claim_async(
    claim_text: str,
    source_origin: str = "content",
    mismatch_warning: str | None = None,
) -> Claim:
    """Concurrent async pipeline for a single claim: search → evaluate."""
    evidence = await search_evidence_async(claim_text)
    return await evaluate_claim_async(
        claim_text=claim_text,
        evidence=evidence,
        source_origin=source_origin,
        mismatch_warning=mismatch_warning,
    )


def check_claim(
    claim_text: str,
    source_origin: str = "content",
    mismatch_warning: str | None = None,
) -> Claim:
    """Full pipeline for a single claim: search → evaluate → Claim object."""
    evidence = search_evidence(claim_text)
    return evaluate_claim(
        claim_text=claim_text,
        evidence=evidence,
        source_origin=source_origin,
        mismatch_warning=mismatch_warning,
    )


def _format_evidence(results: list[dict]) -> str:
    """Format search results into a readable text block for the LLM."""
    if not results:
        return "No evidence found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")[:500]
        parts.append(f"[{i}] {title}\n    URL: {url}\n    {content}")
    return "\n\n".join(parts)
