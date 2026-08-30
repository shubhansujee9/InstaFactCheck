"""FastAPI application — InstaFactCheck backend.

Exposes POST /analyze that accepts an Instagram reel URL and returns
a structured fact-check report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import analyze_content, analyze_reel
from .schemas import AnalyzeRequest, AnalyzeResponse

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InstaFactCheck",
    description="Fact-check Instagram reels by analysing video content, "
    "extracting claims, and verifying them against web sources.",
    version="0.1.0",
)

# Allow the Flutter app (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timeout (seconds).  The pipeline can take 15-60s for a real reel.
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))


from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Static directory path
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the Web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "ok", "message": "InstaFactCheck API is running. See /docs"}


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Fact-check an Instagram reel, video post, photo/carousel post, or claim text.

    Accepts a reel/post URL or text, extracts factual claims,
    and returns a structured report with per-claim verdicts and explanations.
    """
    if not request.url and not request.text:
        raise HTTPException(
            status_code=422,
            detail="Please provide either an Instagram URL ('url') or text/claim ('text').",
        )

    target_desc = request.url or f"text: {request.text[:40]}..."
    logger.info("──── Analyze request: %s ────", target_desc)
    start = time.monotonic()

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_pipeline_sync, request.url, request.text),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Pipeline timed out after %ds", REQUEST_TIMEOUT)
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timed out after {REQUEST_TIMEOUT}s. "
            "The content may be too long or the service is under heavy load.",
        )
    except PermissionError as exc:
        logger.warning("Permission error: %s", exc)
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        logger.error("Download error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to retrieve the Instagram post. It may have been deleted or "
            "Instagram may be blocking requests. You can also paste the post text directly.",
        )
    except Exception as exc:
        logger.exception("Unexpected pipeline error")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {type(exc).__name__}: {exc}",
        )

    elapsed = time.monotonic() - start
    logger.info(
        "──── Analysis complete in %.1fs — verdict=%s, claims=%d ────",
        elapsed,
        result.overall_verdict.value,
        len(result.claims),
    )
    return result


def _run_pipeline_sync(url: str | None, text: str | None) -> AnalyzeResponse:
    """Wrapper to call the async pipeline from a sync executor context."""
    import asyncio as _asyncio

    return _asyncio.run(analyze_content(url=url, text=text))
