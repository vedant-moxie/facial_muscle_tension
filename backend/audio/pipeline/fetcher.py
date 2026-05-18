"""
Reel acquisition.

Three input modes, in descending order of robustness:

  1. **Direct file paths / uploads** (most reliable) — the frontend can
     accept multi-file drops and the backend just consumes them in place.
     This is the default path during demos and tests.

  2. **List of Reel URLs** — uses instaloader to download each individual
     shortcode. Works without login for public posts but is rate-limited.

  3. **Instagram profile URL** — uses instaloader to enumerate a user's
     recent posts and download the top-N video/clip posts. Requires
     instaloader to be installed and usually requires being logged in
     (otherwise Instagram blocks the GraphQL queries it relies on).

If instaloader is not installed, modes (2) and (3) raise a helpful
ImportError pointing at the install command. Mode (1) always works.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from audio.config import MAX_REELS

log = logging.getLogger(__name__)

try:
    import instaloader                              # type: ignore
    _HAS_INSTALOADER = True
except Exception:                                   # noqa: BLE001
    _HAS_INSTALOADER = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PROFILE_RE = re.compile(r"instagram\.com/(?:p/)?([A-Za-z0-9_.]+)/?$")
_SHORTCODE_RE = re.compile(r"instagram\.com/(?:reel|reels|p)/([A-Za-z0-9_-]+)")


def _ensure_instaloader() -> None:
    if not _HAS_INSTALOADER:
        raise ImportError(
            "instaloader is required for Instagram URL-based fetching. "
            "Install with `pip install instaloader>=4.10` or pass file uploads instead."
        )


def _new_session(target_dir: Path):
    _ensure_instaloader()
    return instaloader.Instaloader(
        download_video_thumbnails=False,
        download_comments=False,
        download_geotags=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        filename_pattern="{profile}_{shortcode}",
        dirname_pattern=str(target_dir),
        quiet=True,
    )


def _collect_mp4s(directory: Path, expected_shortcode: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() != ".mp4":
            continue
        if expected_shortcode and expected_shortcode not in f.name:
            continue
        out.append(str(f))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_from_files(file_paths: List[str]) -> List[str]:
    """
    Pass-through for direct uploads. Verifies every path exists and is an
    accepted video extension; returns the same list (deduped).
    """
    seen = set()
    out: List[str] = []
    for p in file_paths[:MAX_REELS]:
        p = str(p)
        if p in seen:
            continue
        seen.add(p)
        if not os.path.exists(p):
            log.warning("fetch_from_files: missing file %s — skipping", p)
            continue
        ext = Path(p).suffix.lower()
        if ext not in {
            ".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi",
            ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
        }:
            log.warning("fetch_from_files: unsupported extension %s — skipping", p)
            continue
        out.append(p)
    return out


def fetch_from_urls(
    reel_urls: List[str],
    output_dir: Optional[str] = None,
) -> List[str]:
    """Download an explicit list of Reel URLs."""
    _ensure_instaloader()
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="sr_reels_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    L = _new_session(out_dir)

    paths: List[str] = []
    for url in reel_urls[:MAX_REELS]:
        m = _SHORTCODE_RE.search(url)
        if not m:
            log.warning("fetch_from_urls: couldn't parse shortcode from %s", url)
            continue
        shortcode = m.group(1)
        try:
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            L.download_post(post, target=str(out_dir))
            paths.extend(_collect_mp4s(out_dir, expected_shortcode=shortcode))
        except Exception as exc:                    # noqa: BLE001
            log.warning("fetch_from_urls: %s failed — %s", shortcode, exc)

    # dedup while preserving order
    return list(dict.fromkeys(paths))


def fetch_reels(
    profile_url: str,
    output_dir: Optional[str] = None,
    max_reels: int = MAX_REELS,
) -> List[str]:
    """Fetch up to `max_reels` Reels from a public Instagram profile."""
    _ensure_instaloader()
    username = profile_url.rstrip("/").split("/")[-1].lstrip("@")
    if not username:
        raise ValueError(f"Could not parse username from {profile_url!r}")

    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="sr_reels_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    L = _new_session(out_dir)

    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except Exception as exc:                        # noqa: BLE001
        raise RuntimeError(
            f"Couldn't open Instagram profile @{username}: {exc}. "
            "Public profiles work without login; private ones need an authenticated "
            "instaloader session."
        ) from exc

    paths: List[str] = []
    for post in profile.get_posts():
        if not post.is_video:
            continue
        # `product_type` reliably distinguishes reels from feed videos.
        is_reel = getattr(post, "product_type", "") in {"clips", "reels"}
        if not is_reel:
            continue
        try:
            L.download_post(post, target=str(out_dir))
            paths.extend(_collect_mp4s(out_dir, expected_shortcode=post.shortcode))
        except Exception as exc:                    # noqa: BLE001
            log.warning("fetch_reels: %s failed — %s", post.shortcode, exc)
        if len(paths) >= max_reels:
            break

    return list(dict.fromkeys(paths))[:max_reels]


def cleanup_files(paths: List[str]) -> None:
    """Best-effort delete of temporary downloads (called from main)."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    # If they all lived in one tempdir, drop that too.
    parents = {str(Path(p).parent) for p in paths}
    for parent in parents:
        if parent.startswith("/tmp/sr_reels_") and os.path.isdir(parent):
            try:
                shutil.rmtree(parent, ignore_errors=True)
            except OSError:
                pass
