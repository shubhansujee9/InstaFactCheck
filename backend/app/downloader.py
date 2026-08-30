"""Download Instagram reels using yt-dlp and extract caption metadata."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_executable(name: str) -> str:
    """Find an executable, checking the current Python env's bin dir first."""
    # Check the venv/bin directory (where pip-installed scripts like yt-dlp live)
    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)
    # Fall back to system PATH
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"'{name}' not found in venv or system PATH")



@dataclass
class DownloadResult:
    """Artifact produced by downloading an Instagram reel or post."""

    video_path: Path | None = None
    image_paths: list[Path] = field(default_factory=list)
    caption: str = ""
    title: str = ""
    temp_dir: str = ""  # caller should clean up


def validate_instagram_url(url: str) -> str:
    """Validate and normalise an Instagram URL (reel, post, tv, etc.).

    Returns the cleaned URL or raises ValueError.
    """
    clean_url = url.split("?")[0].rstrip("/")
    # Handle formats like:
    # https://www.instagram.com/reel/XYZ
    # https://www.instagram.com/p/XYZ
    # https://www.instagram.com/reels/XYZ
    # https://www.instagram.com/tv/XYZ
    # https://instagram.com/share/p/XYZ
    pattern = r"https?://(?:www\.)?instagram\.com/(?:reel|reels|p|tv|share/(?:p|reel))/[\w-]+"
    match = re.search(pattern, clean_url)
    if not match:
        # If it contains instagram.com and a shortcode, accept it
        general_match = re.search(r"https?://(?:www\.)?instagram\.com/[^/]+/(?:p|reel)/[\w-]+", clean_url)
        if general_match:
            return general_match.group(0)
        raise ValueError(
            f"Not a recognized Instagram URL: {url}. Supported formats: /reel/..., /p/..., /tv/..."
        )
    return match.group(0)


def download_reel(url: str) -> DownloadResult:
    """Download an Instagram reel/post video or extract caption/title.

    Supports:
    - Video reels and video posts (downloads MP4 + audio + caption)
    - Photo and carousel posts (extracts caption, metadata, and images)
    
    Returns a DownloadResult with video_path (if video exists), caption, and title.
    The caller is responsible for cleaning up ``result.temp_dir``.
    """
    url = validate_instagram_url(url)
    tmp = tempfile.mkdtemp(prefix="instafact_")
    output_template = str(Path(tmp) / "%(id)s.%(ext)s")

    ytdlp_bin = _find_executable("yt-dlp")
    cmd = [
        ytdlp_bin,
        "--no-warnings",
        "--no-playlist",
        "--write-info-json",
        "--write-thumbnail",
        "--output", output_template,
        url,
    ]
    logger.info("Running yt-dlp: %s", " ".join(cmd))

    is_no_video_post = False
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("yt-dlp download timed out after 120s") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        combined_err = f"{stdout}\n{stderr}"
        
        # Check if yt-dlp simply noted that this is a photo/text post without video
        if "There is no video in this post" in combined_err or "no video" in combined_err.lower():
            logger.info("Post contains no video (photo/carousel/text post). Extracting metadata...")
            is_no_video_post = True
        elif "Private" in combined_err or "login" in combined_err.lower():
            raise PermissionError(
                "This Instagram post appears to be private or requires login."
            )
        else:
            logger.warning("yt-dlp non-zero exit (%d): %s", result.returncode, stderr[:200])

    tmp_path = Path(tmp)
    
    # ── Step 1: Look for downloaded video files ──
    video_files = [
        f for f in tmp_path.iterdir()
        if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov") and not f.name.endswith(".info.json")
    ]
    video_path = video_files[0] if video_files else None

    # ── Step 2: Look for downloaded image files ──
    image_files = [
        f for f in tmp_path.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]

    # ── Step 3: Extract caption and title from info JSON ──
    caption = ""
    title = ""
    info_files = list(tmp_path.glob("*.info.json"))
    if info_files:
        try:
            with open(info_files[0], encoding="utf-8") as fh:
                info = json.load(fh)
            caption = info.get("description", "") or info.get("caption", "") or ""
            title = info.get("title", "") or info.get("fulltitle", "") or ""
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse info JSON: %s", exc)

    # ── Step 4: Fallback video & caption extraction via parth-dl and instaloader ──
    if not video_path:
        logger.info("yt-dlp produced no video. Attempting parth-dl direct resolver for %s...", url)
        video_path, fb_caption, fb_title = _download_video_parth_dl(url, tmp_path)
        if fb_caption and not caption:
            caption = fb_caption
        if fb_title and not title:
            title = fb_title

    if not video_path:
        shortcode_match = re.search(r"/(?:p|reel|tv|share/(?:p|reel))/([A-Za-z0-9_-]+)", url)
        if shortcode_match:
            shortcode = shortcode_match.group(1)
            logger.info("Attempting instaloader direct CDN video download for %s...", shortcode)
            video_path, fb_caption, fb_title = _download_video_instaloader_fallback(shortcode, tmp_path)
            if fb_caption and not caption:
                caption = fb_caption
            if fb_title and not title:
                title = fb_title

    # ── Step 5: Final metadata scrape if caption/title still missing ──
    if not caption and not title:
        shortcode_match = re.search(r"/(?:p|reel|tv|share/(?:p|reel))/([A-Za-z0-9_-]+)", url)
        if shortcode_match:
            shortcode = shortcode_match.group(1)
            caption, title = _extract_caption_fallback(shortcode, url)

    logger.info(
        "Processed Instagram URL: is_video=%s caption_len=%d title=%s",
        bool(video_path),
        len(caption),
        title[:80],
    )

    if not video_path and not caption and not title:
        raise RuntimeError(
            "Could not extract content from this Instagram post. "
            "It may require login or has been restricted by Instagram. "
            "Tip: You can also copy the post text directly."
        )

    return DownloadResult(
        video_path=video_path,
        image_paths=image_files,
        caption=caption,
        title=title,
        temp_dir=tmp,
    )


def _download_video_parth_dl(url: str, tmp_path: Path) -> tuple[Path | None, str, str]:
    """Download video using parth-dl GraphQL resolver."""
    try:
        import parth_dl
        import httpx

        info = parth_dl.get_info(url)
        if not isinstance(info, dict):
            return None, "", ""

        caption = info.get("title") or info.get("caption") or ""
        uploader = info.get("uploader") or ""
        title = f"Video by @{uploader}" if uploader else "Instagram Reel"

        formats = info.get("formats", [])
        if not formats and info.get("entries"):
            entries = info["entries"]
            if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
                formats = entries[0].get("formats", [])

        if formats and formats[0].get("url"):
            direct_url = formats[0]["url"]
            target_file = tmp_path / "reel_video.mp4"
            logger.info("parth-dl found video stream: %s. Downloading...", direct_url[:80])
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(direct_url)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    target_file.write_bytes(resp.content)
                    logger.info("Successfully downloaded video via parth-dl (%d bytes) -> %s", len(resp.content), target_file)
                    return target_file, caption, title

        return None, caption, title
    except Exception as exc:
        logger.warning("parth-dl video download failed: %s", exc)
        return None, "", ""


def _download_video_instaloader_fallback(shortcode: str, tmp_path: Path) -> tuple[Path | None, str, str]:
    """Fallback video and metadata extraction using instaloader direct CDN stream."""
    try:
        import instaloader
        import httpx

        L = instaloader.Instaloader()
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        caption = post.caption or ""
        title = post.title or f"Video by @{post.owner_username}"

        if post.is_video and post.video_url:
            target_file = tmp_path / f"{shortcode}.mp4"
            logger.info("Downloading video stream directly from Instagram CDN: %s...", post.video_url[:80])
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(post.video_url)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    target_file.write_bytes(resp.content)
                    logger.info("Successfully downloaded video via CDN fallback (%d bytes) -> %s", len(resp.content), target_file)
                    return target_file, caption, title

        return None, caption, title
    except Exception as exc:
        logger.warning("Instaloader video fallback failed for %s: %s", shortcode, exc)
        return None, "", ""


def _extract_caption_fallback(shortcode: str, url: str) -> tuple[str, str]:
    """Fallback extraction using instaloader or direct meta scraping."""
    caption = ""
    title = ""

    # Try instaloader
    try:
        import instaloader
        L = instaloader.Instaloader()
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        if post.caption:
            caption = post.caption
            title = post.title or f"Post by @{post.owner_username}"
            logger.info("Extracted caption via instaloader (%d chars)", len(caption))
            return caption, title
    except Exception as e:
        logger.debug("Instaloader fallback did not resolve: %s", e)

    # Try open graph scraper with social bot user-agent
    try:
        import httpx
        resp = httpx.get(
            url,
            headers={"User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"},
            follow_redirects=True,
            timeout=10,
        )
        if resp.status_code == 200:
            desc_match = re.search(r"<meta\s+(?:property|name)=\"(?:og:description|description)\"\s+content=\"([^\"]*)\"", resp.text)
            if desc_match:
                import html
                caption = html.unescape(desc_match.group(1))
            title_match = re.search(r"<meta\s+(?:property|name)=\"(?:og:title|title)\"\s+content=\"([^\"]*)\"", resp.text)
            if title_match:
                import html
                title = html.unescape(title_match.group(1))
    except Exception as e:
        logger.debug("Meta scrape fallback did not resolve: %s", e)

    return caption, title

