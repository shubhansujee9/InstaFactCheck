"""Pydantic models for the InstaFactCheck API request/response contract."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Body of POST /analyze."""

    url: str | None = Field(
        None,
        description="Instagram post or reel URL to fact-check",
        examples=["https://www.instagram.com/reel/ABC123/"],
    )
    text: str | None = Field(
        None,
        description="Direct claim or caption text to fact-check",
        examples=["Eating carrots gives you night vision."],
    )


# ---------------------------------------------------------------------------
# Response building blocks
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """Possible verdicts for an overall report or individual claim."""

    TRUE = "true"
    MOSTLY_TRUE = "mostly_true"
    MIXED = "mixed"
    MISLEADING = "misleading"
    FALSE = "false"
    UNVERIFIABLE = "unverifiable"


class Source(BaseModel):
    """A single evidence source backing a claim verdict."""

    title: str = Field(..., description="Title or headline of the source")
    url: str = Field(..., description="URL of the source")


class Claim(BaseModel):
    """A single factual claim extracted from the video or post, with its verdict."""

    claim_text: str = Field(..., description="The factual claim as stated or implied")
    verdict: Verdict = Field(..., description="Fact-check verdict for this claim")
    explanation: str = Field(
        ..., description="Brief explanation of why this verdict was reached"
    )
    source_origin: str = Field(
        default="content",
        description="Where the claim originated: 'caption', 'audio_transcript', or 'both'",
    )
    mismatch_warning: str | None = Field(
        default=None,
        description="Warning if this claim is contradicted by the video audio or falsely attributed",
    )
    sources: list[Source] = Field(
        default_factory=list, description="Evidence sources for this claim"
    )


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Full fact-check report returned by POST /analyze."""

    overall_summary: str = Field(
        ..., description="2-3 sentence plain-language summary of the video"
    )
    overall_verdict: Verdict = Field(
        ..., description="Aggregate verdict across all claims"
    )
    has_caption_video_mismatch: bool = Field(
        default=False,
        description="True if the post caption contradicts or misrepresents the actual video audio content",
    )
    mismatch_summary: str | None = Field(
        default=None,
        description="Explanation of the discrepancy between caption and video content if any",
    )
    claims: list[Claim] = Field(
        default_factory=list, description="Individual claim verdicts with sources"
    )
    video_title: str | None = Field(
        None, description="Title/caption of the reel if available"
    )
    transcript_snippet: str | None = Field(
        None,
        description="First ~200 chars of transcript for preview",
    )
