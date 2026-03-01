"""Tools for downloading video snippets, extracting frames, and analyzing visual references."""

import base64
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tarfile
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.171:11434")
_VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:32b")
_VISION_TIMEOUT = 300


class _RefMediaContext:
    """Resolve the workspace root from runtime context."""

    def __init__(self, ctx: Dict[str, Any] | None):
        if ctx is None:
            raise ValueError("_context is required for reference_media tools")
        raw = ctx.get("python_workspace_root")
        if raw is None:
            raise ValueError("python_workspace_root missing from _context")
        self.workspace_root = Path(raw).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)


_NON_VIDEO_YT_PATTERN = re.compile(
    r"youtube\.com/(@|channel/|c/|user/|feed|trending|playlist\?)",
    re.IGNORECASE,
)


def download_video_snippet(
    url: str,
    *,
    duration_seconds: int = 60,
    filename: str = "reference_video.mp4",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Download the first N seconds of a video from a URL (YouTube, etc.).

    Uses yt-dlp to download a short snippet at 720p max resolution.
    The video is saved to the workspace as an MP4 file.
    Rejects YouTube channel/user/playlist URLs that are not direct video links.
    """
    if _NON_VIDEO_YT_PATTERN.search(url):
        logger.warning("download_video_snippet: rejected non-video URL: %s", url)
        return {"error": f"URL is a YouTube channel/user/playlist page, not a video: {url}"}

    ctx = _RefMediaContext(_context)
    target = ctx.workspace_root / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("download_video_snippet: url=%s duration=%ds target=%s", url, duration_seconds, target)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--download-sections", f"*0-{duration_seconds}",
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", str(target),
        "--force-overwrites",
        "--no-warnings",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error("download_video_snippet: yt-dlp failed: %s", result.stderr[:500])
        return {"error": f"yt-dlp failed (exit {result.returncode}): {result.stderr[:300]}"}

    if not target.exists():
        return {"error": "yt-dlp completed but output file not found"}

    size = target.stat().st_size
    logger.info("download_video_snippet: saved %d bytes to %s", size, target)

    return {
        "path": filename,
        "absolute_path": str(target),
        "size": size,
        "duration_requested": duration_seconds,
    }


def extract_key_frames(
    video_path: str,
    *,
    max_frames: int = 6,
    output_dir: str = "reference_frames",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extract evenly-spaced key frames from a video file.

    Uses ffprobe to get the duration and ffmpeg to extract frames.
    Frames are saved as JPEG files in the output directory.
    """
    ctx = _RefMediaContext(_context)
    video = ctx.workspace_root / video_path

    if not video.exists():
        return {"error": f"video not found: {video_path}"}

    out = ctx.workspace_root / output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, TypeError):
        logger.warning("extract_key_frames: could not probe duration, using 1fps fallback")
        duration = 0

    if duration > 0 and max_frames > 0:
        interval = max(1, duration / max_frames)
        fps_filter = f"fps=1/{interval:.2f}"
    else:
        fps_filter = "fps=1"

    logger.info("extract_key_frames: duration=%.1fs filter=%s max=%d", duration, fps_filter, max_frames)

    cmd = [
        "ffmpeg", "-i", str(video),
        "-vf", fps_filter,
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(out / "frame_%03d.jpg"),
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        logger.error("extract_key_frames: ffmpeg failed: %s", result.stderr[:500])
        return {"error": f"ffmpeg failed: {result.stderr[:300]}"}

    frames = sorted(str(p.relative_to(ctx.workspace_root)) for p in out.glob("frame_*.jpg"))
    logger.info("extract_key_frames: extracted %d frames", len(frames))

    return {"frames": frames, "count": len(frames), "output_dir": output_dir}


def analyze_visual_reference(
    image_path: str,
    *,
    analysis_focus: str = "",
    model: str = "",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Analyze an image using a vision model to extract art direction insights.

    Sends the image to the OLLAMA vision model with a detailed prompt
    about color palette, art style, composition, mood, and game-relevant details.
    """
    ctx = _RefMediaContext(_context)
    use_model = model or _VISION_MODEL
    target = ctx.workspace_root / image_path

    if not target.exists():
        return {"error": f"image not found: {image_path}"}

    logger.info("analyze_visual_reference: model=%s image=%s", use_model, target)

    img_b64 = base64.b64encode(target.read_bytes()).decode()

    prompt = (
        "Analyze this image as a visual reference for game art direction. Describe:\n"
        "1. COLOR PALETTE: Dominant and accent colors, overall tone. List approximate HEX values.\n"
        "2. ART STYLE: Pixel art, painterly, minimalist, realistic, cartoon, etc.?\n"
        "3. COMPOSITION: Layout, perspective, focal points, use of space.\n"
        "4. MOOD & ATMOSPHERE: Emotion evoked, visual techniques creating that mood.\n"
        "5. KEY VISUAL ELEMENTS: Characters, objects, environments, effects, lighting.\n"
        "6. GAME-RELEVANT DETAILS: HUD style, entity design, animation feel (if applicable).\n"
    )
    if analysis_focus:
        prompt += f"\nAdditional focus: {analysis_focus}\n"

    payload = json.dumps({
        "model": use_model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = json.loads(urllib.request.urlopen(req, timeout=_VISION_TIMEOUT).read())
    content = resp.get("message", {}).get("content", "")
    logger.info("analyze_visual_reference: got %d chars", len(content))

    return {"analysis": content, "model": use_model, "image_path": image_path}


def download_image(
    url: str,
    *,
    filename: str = "",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Download an image from a URL and save it to the workspace.

    If filename is not provided, it is derived from the URL.
    """
    ctx = _RefMediaContext(_context)

    if not filename:
        # Extract filename from URL, fallback to reference_image.jpg
        url_path = url.split("?")[0].split("#")[0]
        basename = url_path.rstrip("/").rsplit("/", 1)[-1]
        if not re.match(r".+\.(png|jpg|jpeg|gif|webp|bmp)$", basename, re.IGNORECASE):
            basename = "reference_image.jpg"
        filename = f"reference_frames/{basename}"

    target = ctx.workspace_root / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("download_image: url=%s target=%s", url[:100], target)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            target.write_bytes(resp.read())
    except Exception as e:
        logger.error("download_image: failed: %s", e)
        return {"error": f"download failed: {e}"}

    size = target.stat().st_size
    logger.info("download_image: saved %d bytes", size)

    return {"path": filename, "absolute_path": str(target), "size": size}


def create_placeholder_image(
    filename: str,
    *,
    width: int = 64,
    height: int = 64,
    color: str = "#FF00FF",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a solid-color PNG placeholder image using only struct/zlib (no PIL).

    Developers use these as compile-time placeholders for //go:embed.
    The art pipeline replaces them with real generated art later.
    """
    ctx = _RefMediaContext(_context)
    target = ctx.workspace_root / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    # Parse hex color
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    # Build minimal PNG: IHDR + single IDAT + IEND
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes([r, g, b]) * width  # filter byte + pixel data
    raw = b"".join(row for _ in range(height))
    idat_data = zlib.compress(raw)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr_data)
    png += _png_chunk(b"IDAT", idat_data)
    png += _png_chunk(b"IEND", b"")

    target.write_bytes(png)
    size = target.stat().st_size
    logger.info("create_placeholder_image: %s %dx%d color=%s (%d bytes)", filename, width, height, color, size)

    return {"path": filename, "absolute_path": str(target), "size": size, "width": width, "height": height, "color": color}


def create_project_archive(
    *,
    output_filename: str = "game_project.tar.gz",
    include_patterns: List[str] | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a .tar.gz archive of the project (source, assets, WASM, HTML).

    Collects files matching the include patterns from the workspace root
    and bundles them into a compressed archive for download.
    """
    ctx = _RefMediaContext(_context)

    target = ctx.workspace_root / output_filename

    if include_patterns is None:
        # Include all project files by default
        matched_files = [
            f for f in ctx.workspace_root.rglob("*")
            if f.is_file() and f != target
        ]
    else:
        matched_files = []
        for pattern in include_patterns:
            matched_files.extend(ctx.workspace_root.glob(pattern))

    # Deduplicate and sort, skip directories
    matched_files = sorted({f for f in matched_files if f.is_file()})

    logger.info("create_project_archive: %d files matching %s", len(matched_files), include_patterns)

    with tarfile.open(target, "w:gz") as tar:
        for f in matched_files:
            arcname = str(f.relative_to(ctx.workspace_root))
            tar.add(str(f), arcname=arcname)

    size = target.stat().st_size
    logger.info("create_project_archive: %s (%d bytes, %d files)", target, size, len(matched_files))

    return {
        "path": output_filename,
        "absolute_path": str(target),
        "size": size,
        "file_count": len(matched_files),
    }
