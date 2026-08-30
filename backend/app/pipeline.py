"""Orchestrator: download → transcribe → extract claims → fact-check → report."""

from __future__ import annotations

import logging
import shutil
from collections import Counter

from .claim_extractor import extract_claims
from .downloader import download_reel
from .fact_checker import check_claim
from .schemas import AnalyzeResponse, MediaForensics, Verdict
from .transcriber import get_transcript

logger = logging.getLogger(__name__)


async def analyze_content(url: str | None = None, text: str | None = None) -> AnalyzeResponse:
    """Run fact-checking on an Instagram URL or direct claim text."""
    if url:
        return await analyze_reel(url)
    elif text:
        return await analyze_text(text)
    raise ValueError("Either 'url' or 'text' must be provided.")


async def analyze_text(text: str) -> AnalyzeResponse:
    """Fact-check raw claim or caption text directly."""
    logger.info("Extracting claims directly from text (%d chars)...", len(text))
    raw_claims = extract_claims(caption=text, transcript="")
    
    if not raw_claims:
        return AnalyzeResponse(
            overall_summary=_build_no_claims_summary(text, ""),
            overall_verdict=Verdict.UNVERIFIABLE,
            claims=[],
            transcript_snippet=text[:200],
        )

    logger.info("Fact-checking %d extracted claims in parallel...", len(raw_claims))
    valid_claims = [r.get("claim_text", "") for r in raw_claims if r.get("claim_text", "")]
    checked_claims = []
    if valid_claims:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_claims), 6)) as executor:
            checked_claims = list(executor.map(check_claim, valid_claims))

    overall_verdict = _aggregate_verdict(checked_claims)
    overall_summary = _build_summary(checked_claims, text, "")

    return AnalyzeResponse(
        overall_summary=overall_summary,
        overall_verdict=overall_verdict,
        claims=checked_claims,
        transcript_snippet=text[:200],
    )


async def analyze_reel(url: str) -> AnalyzeResponse:
    """Run the full fact-checking pipeline for an Instagram reel or post URL."""
    temp_dir: str | None = None

    try:
        # ── Step 1: Download ─────────────────────────────────────────
        logger.info("Step 1/5: Downloading reel from %s", url)
        dl = download_reel(url)
        temp_dir = dl.temp_dir

        # ── Step 2: Multimodal Video & Audio Analysis ────────────────
        transcript = ""
        speech_summary = ""
        speakers: list[str] = []
        forensics = MediaForensics()
        if dl.video_path and dl.video_path.exists():
            logger.info("Step 2/5: Analyzing video with Gemini native audio/video model...")

            # Primary: Gemini native multimodal (hears speech, sees visuals)
            gemini_analysis = None
            try:
                from app.gemini_media_analyzer import analyze_video_with_gemini
                gemini_analysis = analyze_video_with_gemini(dl.video_path)
            except Exception as exc:
                logger.warning("Gemini media analysis error (%s)", exc)

            if gemini_analysis and gemini_analysis.transcript:
                transcript = gemini_analysis.transcript
                speech_summary = gemini_analysis.speech_summary
                speakers = gemini_analysis.speakers
                forensics = MediaForensics(
                    visual_summary=gemini_analysis.visual_summary,
                    on_screen_text=gemini_analysis.on_screen_text,
                    content_genre=gemini_analysis.content_genre,
                    is_selectively_clipped=gemini_analysis.is_selectively_clipped,
                    original_context_note=gemini_analysis.original_context_note,
                )
                logger.info(
                    "Using Gemini transcript (%d chars), speakers: %s",
                    len(transcript),
                    speakers,
                )
            else:
                # Fallback: local Whisper + keyframe analysis
                logger.info("Gemini unavailable — falling back to Whisper + keyframes...")
                try:
                    transcript = get_transcript(dl.video_path)
                except Exception as exc:
                    logger.warning("Transcription failed (%s). Continuing with caption metadata.", exc)
                    transcript = ""

                try:
                    from app.video_analyzer import analyze_video_frames
                    forensics = analyze_video_frames(dl.video_path)
                except Exception as exc:
                    logger.warning("Visual forensic analysis error (%s)", exc)
                    forensics = MediaForensics()
        else:
            logger.info("Step 2/5: No video stream (photo/carousel/text post). Using post caption.")

        # ── Step 3: Extract claims & check consistency ───────────────
        logger.info("Step 3/5: Extracting factual claims & checking consistency...")
        extraction_res = extract_claims(
            caption=dl.caption,
            transcript=transcript,
            visual_summary=forensics.visual_summary,
            on_screen_text=forensics.on_screen_text,
            speech_summary=speech_summary or None,
            speakers=speakers or None,
        )
        raw_claims = extraction_res.get("claims", [])
        has_mismatch = extraction_res.get("has_caption_video_mismatch", False)
        mismatch_summary = extraction_res.get("mismatch_summary")
        caption_summary = extraction_res.get("caption_summary")
        video_actual_context = extraction_res.get("video_actual_context")

        # When caption mismatches video, drop caption-only claims
        if has_mismatch:
            before = len(raw_claims)
            raw_claims = [
                c for c in raw_claims
                if c.get("source_origin") != "caption"
            ]
            if before != len(raw_claims):
                logger.info(
                    "Dropped %d caption-only claims due to caption/video mismatch",
                    before - len(raw_claims),
                )

        original_full_video_title = extraction_res.get("original_full_video_title")
        original_full_video_url = extraction_res.get("original_full_video_url")

        if not raw_claims:
            logger.info("No verifiable claims found — returning summary-only report")
            return AnalyzeResponse(
                overall_summary=_build_no_claims_summary(dl.caption, transcript, speech_summary),
                overall_verdict=Verdict.UNVERIFIABLE,
                caption_summary=caption_summary,
                video_actual_context=video_actual_context,
                forensics=forensics,
                has_caption_video_mismatch=has_mismatch,
                mismatch_summary=mismatch_summary,
                original_full_video_url=original_full_video_url,
                original_full_video_title=original_full_video_title,
                instagram_url=url,
                claims=[],
                video_title=dl.title or None,
                transcript_snippet=transcript[:200] if transcript else None,
            )

        # ── Step 4: Fact-check each claim concurrently ────────────────
        logger.info("Step 4/5: Fact-checking %d claims in parallel...", len(raw_claims))
        
        def _check_wrapper(raw_item: dict):
            return check_claim(
                claim_text=raw_item.get("claim_text", ""),
                source_origin=raw_item.get("source_origin", "content"),
                mismatch_warning=raw_item.get("mismatch_warning"),
            )

        checked_claims = []
        valid_items = [r for r in raw_claims if r.get("claim_text")]
        if valid_items:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_items), 6)) as executor:
                checked_claims = list(executor.map(_check_wrapper, valid_items))

        # ── Step 5: Aggregate ────────────────────────────────────────
        logger.info("Step 5/5: Aggregating report ...")
        overall_verdict = _aggregate_verdict(checked_claims, has_mismatch)
        overall_summary = _build_summary(checked_claims, dl.caption, transcript, mismatch_summary)

        return AnalyzeResponse(
            overall_summary=overall_summary,
            overall_verdict=overall_verdict,
            caption_summary=caption_summary,
            video_actual_context=video_actual_context,
            forensics=forensics,
            has_caption_video_mismatch=has_mismatch,
            mismatch_summary=mismatch_summary,
            original_full_video_url=original_full_video_url,
            original_full_video_title=original_full_video_title,
            instagram_url=url,
            claims=checked_claims,
            video_title=dl.title or None,
            transcript_snippet=transcript[:200] if transcript else None,
        )

    finally:
        # Always clean up temp files
        if temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def _aggregate_verdict(claims: list, has_mismatch: bool = False) -> Verdict:
    """Determine an overall verdict from individual claim verdicts.

    If the caption deliberately misrepresents or contradicts the video (has_mismatch),
    the overall verdict reflects the false framing unless all claims are verified true.
    """
    if not claims:
        return Verdict.UNVERIFIABLE

    counts = Counter(c.verdict for c in claims)
    total = len(claims)

    # 1. If any claim is false
    if counts.get(Verdict.FALSE, 0) > 0:
        if counts[Verdict.FALSE] > total / 2:
            return Verdict.FALSE
        return Verdict.MISLEADING

    # 2. If all claims are true and verified
    if counts.get(Verdict.TRUE, 0) == total:
        return Verdict.TRUE

    # 3. If there is a major discrepancy between caption framing and video reality
    if has_mismatch:
        return Verdict.MISLEADING

    # 4. If mostly true
    true_ish = counts.get(Verdict.TRUE, 0) + counts.get(Verdict.MOSTLY_TRUE, 0)
    if true_ish > total / 2:
        return Verdict.MOSTLY_TRUE

    # 5. If mostly unverifiable
    if counts.get(Verdict.UNVERIFIABLE, 0) > total / 2:
        return Verdict.UNVERIFIABLE

    return Verdict.MIXED


def _build_summary(
    claims: list, caption: str, transcript: str, mismatch_summary: str | None = None
) -> str:
    """Build a brief overall summary from the checked claims and mismatch notes."""
    total = len(claims)
    counts = Counter(c.verdict.value for c in claims)

    parts = []
    if mismatch_summary:
        parts.append(f"⚠️ Context Discrepancy Detected: {mismatch_summary}")

    parts.append(f"This content contained {total} verifiable claim{'s' if total != 1 else ''}.")

    verdict_descriptions = []
    for v, label in [
        ("true", "verified as true"),
        ("mostly_true", "mostly true"),
        ("mixed", "mixed"),
        ("misleading", "misleading"),
        ("false", "false"),
        ("unverifiable", "unverifiable"),
    ]:
        count = counts.get(v, 0)
        if count > 0:
            verdict_descriptions.append(f"{count} {label}")

    if verdict_descriptions:
        parts.append("Of these, " + ", ".join(verdict_descriptions) + ".")

    return " ".join(parts)


def _build_no_claims_summary(caption: str, transcript: str, speech_summary: str = "") -> str:
    """Build a summary when no verifiable claims were found."""
    if speech_summary:
        return (
            f"The video shows: {speech_summary} "
            "No specific verifiable factual claims were identified in the spoken content. "
            "The content may be primarily opinion-based, ceremonial, or personal narrative."
        )
    content_source = "caption and transcript" if caption and transcript else (
        "caption" if caption else "transcript" if transcript else "content"
    )
    return (
        f"After analyzing the {content_source} of this video, no specific "
        "verifiable factual claims were identified. The content may be primarily "
        "opinion-based, entertainment, or personal narrative."
    )
