"""Tools for AI image generation and vision-based critique via OLLAMA."""

import base64
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.171:11434")
_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "x/z-image-turbo:latest")
_VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:32b")
_TIMEOUT = 300


class _ImageContext:
    """Resolve the workspace root from runtime context."""

    def __init__(self, ctx: Dict[str, Any] | None):
        if ctx is None:
            raise ValueError("_context is required for image tools")
        raw = ctx.get("python_workspace_root")
        if raw is None:
            raise ValueError("python_workspace_root missing from _context")
        self.workspace_root = Path(raw).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)


def generate_image(
    prompt: str,
    *,
    filename: str = "generated.png",
    model: str = "",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate an image using OLLAMA and save it to the workspace.

    Calls the OLLAMA image generation API with the given prompt,
    decodes the base64 response, and writes the PNG to *filename*
    (relative to workspace root).
    """
    ctx = _ImageContext(_context)
    use_model = model or _IMAGE_MODEL

    target = ctx.workspace_root / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("generate_image: model=%s prompt=%.80s target=%s", use_model, prompt, target)

    payload = json.dumps({
        "model": use_model,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = json.loads(urllib.request.urlopen(req, timeout=_TIMEOUT).read())

    img_b64 = resp.get("image", "")
    if not img_b64:
        logger.error("generate_image: no image in response, keys=%s", list(resp.keys()))
        return {"error": "no image in response", "response_keys": list(resp.keys())}

    target.write_bytes(base64.b64decode(img_b64))
    size = target.stat().st_size
    logger.info("generate_image: saved %d bytes to %s", size, target)

    return {
        "path": filename,
        "absolute_path": str(target),
        "size": size,
        "model": use_model,
    }


def critique_image(
    image_path: str,
    criteria: str,
    *,
    model: str = "",
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Send an image to a vision model for quality critique.

    Reads the image from *image_path* (relative to workspace),
    base64-encodes it, and sends it to the OLLAMA vision model
    along with the evaluation *criteria*.
    """
    ctx = _ImageContext(_context)
    use_model = model or _VISION_MODEL

    target = ctx.workspace_root / image_path
    if not target.exists():
        return {"error": f"image not found: {image_path}"}

    logger.info("critique_image: model=%s image=%s", use_model, target)

    img_b64 = base64.b64encode(target.read_bytes()).decode()

    payload = json.dumps({
        "model": use_model,
        "messages": [{
            "role": "user",
            "content": criteria,
            "images": [img_b64],
        }],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = json.loads(urllib.request.urlopen(req, timeout=_TIMEOUT).read())

    content = resp.get("message", {}).get("content", "")
    logger.info("critique_image: got %d chars of critique", len(content))

    return {
        "critique": content,
        "model": use_model,
        "image_path": image_path,
    }
