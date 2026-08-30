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


class MediaForensics(BaseModel):
    """Forensic and multimodal video analysis metadata."""

    is_likely_ai_generated: bool = Field(
        default=False,
        description="True if visual/audio indicators suggest synthetic AI generation or deepfake",
    )
    ai_confidence: str = Field(
        default="none",
        description="Confidence in AI detection: 'high', 'medium', 'low', or 'none'",
    )
    ai_indicators: list[str] = Field(
        default_factory=list,
        description="Specific forensic anomalies found (e.g. skin morphing, unnatural speech cadence)",
    )
    visual_summary: str | None = Field(
        default=None,
        description="Description of what is visually seen in the video frames",
    )
    on_screen_text: list[str] = Field(
        default_factory=list,
        description="Text overlays, banners, or subtitles detected in video frames",
    )
    content_genre: str | None = Field(
        default=None,
        description="Detected media genre: e.g. 'news_report', 'satire_comedy', 'speech_clip', 'meme'",
    )
    is_selectively_clipped: bool = Field(
        default=False,
        description="True if video appears trimmed or cut from a larger speech/event altering its meaning",
    )
    original_context_note: str | None = Field(
        default=None,
        description="Context regarding the original full-length event or source if identified",
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
    caption_summary: str | None = Field(
        default=None,
        description="Summary of what the caption / poster claims",
    )
    video_actual_context: str | None = Field(
        default=None,
        description="What the video actually shows and discusses in reality",
    )
    forensics: MediaForensics | None = Field(
        default=None,
        description="Forensic, deepfake, and visual analysis details",
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
    original_full_video_url: str | None = Field(
        default=None,
        description="Direct link to watch the full original source video/speech/interview on YouTube or official site",
    )
    original_full_video_title: str | None = Field(
        default=None,
        description="Title or event name of the full original source video",
    )
    instagram_url: str | None = Field(
        default=None,
        description="Original Instagram reel/post URL",
    )
    video_title: str | None = Field(
        None, description="Title/caption of the reel if available"
    )
    transcript_snippet: str | None = Field(
        None,
        description="First ~200 chars of transcript for preview",
    )
