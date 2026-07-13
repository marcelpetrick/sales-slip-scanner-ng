"""Shared receipt-image preparation and Ollama response handling."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Optional

import ollama
from PIL import Image

MAX_SIDE_PX = 1500

PROMPT = (
    "What is the sum to pay in the given sales slip? "
    "It is a German sales slip for groceries or gas. "
    "Look for 'Summe', 'Gesamt' or 'zu zahlen'. "
    "Reply with ONLY the amount in the format 'Euro,Cent' (e.g. '79,49'). "
    "No currency symbol, no extra text. If not found, reply 'NaN'."
)


def encode_image(path: Path) -> str:
    """Resize an image when necessary and return base64-encoded JPEG data."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, MAX_SIDE_PX / max(width, height))
        if scale < 1.0:
            size = (int(width * scale), int(height * scale))
            img = img.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()


def query_model(model_id: str, image_b64: str) -> str:
    """Submit one encoded receipt to an Ollama vision model."""
    response = ollama.chat(
        model=model_id,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_b64],
        }],
        think=False,
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 64},
    )
    return response.message.content.strip()


def parse_price(raw_text: str) -> Optional[int]:
    """Parse an exact decimal-only model response as integer euro-cents.

    The prompt requires a single amount, so responses containing labels,
    currency symbols, or additional numbers are rejected instead of guessing
    which number is the receipt total.
    """
    match = re.fullmatch(r"(\d+)[,.](\d{2})", raw_text.strip())
    if not match:
        return None
    return int(match.group(1)) * 100 + int(match.group(2))


def format_price(price_cents: int) -> str:
    """Format integer euro-cents as the model's canonical ``Euro,Cent`` form."""
    return f"{price_cents // 100},{price_cents % 100:02d}"


def model_id_is_available(requested: str, available: list[str] | set[str]) -> bool:
    """Return whether the exact requested model tag is locally available."""
    if ":" in requested:
        return requested in available
    return requested in available or f"{requested}:latest" in available
