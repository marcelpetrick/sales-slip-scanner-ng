#!/usr/bin/env python3
"""
Sales Slip Scanner
==================
OCR German grocery / gas receipts using a local ollama vision model.

Usage
-----
    python salesSlipScanner.py [--model MODEL_NAME]

Drop image files (JPEG, PNG, GIF) into the ``input/`` directory next to this
script, then run it.  Each file whose total is successfully extracted gets
renamed to include the detected price before its extension::

    receipt.jpg  →  receipt_7949.jpg   (79,49 €)

A human-readable summary is printed at the end showing every file and the
grand total of all detected expenses.

Model selection
---------------
The default model is the best-performing compact model from the local
benchmark (see localVisionModelTest/results.pdf).  Pass ``--model`` to
override:

    python salesSlipScanner.py --model llama3.2-vision:11b

If the chosen model is not installed in the local ollama instance the script
exits immediately with an instructive error message including the install
command.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import ollama

from receipt_ocr import encode_image, parse_price, query_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Best accuracy/size balance from the local benchmark (100 %, 3.3 GB).
DEFAULT_MODEL: str = "qwen3-vl:4b"

#: Directory where users drop files to be processed.
INPUT_DIR: Path = Path(__file__).parent / "input"

#: Image file extensions that will be picked up.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif"})

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def list_local_models() -> list[str]:
    """Return all model IDs currently installed in the local ollama instance.

    Raises:
        RuntimeError: If the ollama server cannot be reached.
    """
    try:
        return [m.model for m in ollama.list().models]
    except Exception as exc:
        raise RuntimeError(f"Cannot reach ollama server: {exc}") from exc


def ensure_model_available(model_id: str) -> None:
    """Assert that *model_id* is installed locally; exit with an error if not.

    Performs a prefix match so ``"qwen3-vl:4b"`` matches entries like
    ``"qwen3-vl:4b-fp16"``.

    Args:
        model_id: The ollama model identifier to verify.

    Raises:
        SystemExit: With exit code 1 when the model is absent.
        RuntimeError: When the ollama server is unreachable.
    """
    available = list_local_models()
    base = model_id.split(":")[0]
    tag  = model_id.split(":")[1] if ":" in model_id else ""
    found = any(
        m == model_id or (tag and m.startswith(f"{base}:{tag}"))
        for m in available
    )
    if not found:
        print(f"Error: model '{model_id}' is not installed in ollama.", file=sys.stderr)
        print(
            f"  Available: {', '.join(available) if available else '(none)'}",
            file=sys.stderr,
        )
        print(f"  Install:   ollama pull {model_id}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def load_and_encode_image(path: Path) -> str:
    """Open an image, resize it, and return a base64-encoded JPEG string.

    The image is resized so its longest side is at most *MAX_SIDE_PX* pixels
    (aspect ratio preserved).  Smaller images pass through unchanged.

    Args:
        path: Path to the source image file.

    Returns:
        Base64-encoded JPEG data (no ``data:`` URI prefix).
    """
    return encode_image(path)


def is_already_processed(path: Path) -> bool:
    """Return ``True`` when the filename already carries a price suffix.

    A price suffix is ``_NNN…`` (3–6 decimal digits) immediately before the
    file extension, e.g. ``receipt_7949.jpg``.  Files matching this pattern
    are skipped so re-runs are idempotent.

    Args:
        path: File path to inspect (only the name is examined).
    """
    return bool(re.search(r"_\d{3,6}\.", path.name))


def collect_input_files(directory: Path) -> list[Path]:
    """Collect all unprocessed, supported image files in *directory*.

    Subdirectories are not traversed.  Files whose names already contain a
    price suffix (see :func:`is_already_processed`) are silently skipped.

    Args:
        directory: The directory to scan (must exist).

    Returns:
        Sorted list of ``Path`` objects ready for processing.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not is_already_processed(p)
    )


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def query_ollama(model_id: str, image_b64: str) -> str:
    """Submit the base64 image to the vision model and return the raw response.

    Args:
        model_id: Local ollama model to use.
        image_b64: Base64-encoded JPEG image data.

    Returns:
        Stripped text response from the model.
    """
    return query_model(model_id, image_b64)


# ---------------------------------------------------------------------------
# File renaming
# ---------------------------------------------------------------------------


def build_renamed_path(original: Path, price_cents: int) -> Path:
    """Return the target path with the price suffix inserted before the extension.

    Example::

        Path("input/receipt.jpg"), 7949  →  Path("input/receipt_7949.jpg")

    Args:
        original: Original file path.
        price_cents: Detected price in integer euro-cents.

    Returns:
        New ``Path`` in the same parent directory as *original*.
    """
    return original.parent / f"{original.stem}_{price_cents}{original.suffix}"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process_file(path: Path, model_id: str) -> Optional[int]:
    """OCR a single sales slip and rename it with its detected price.

    On success the original file is renamed in-place and the price in
    euro-cents is returned.  If no price is detected, or any error occurs,
    the file is left untouched and ``None`` is returned.

    Args:
        path: Path to the image file.
        model_id: Local ollama model to use for OCR.

    Returns:
        Detected price in euro-cents, or ``None``.
    """
    print(f"  {path.name}", end=" … ", flush=True)
    try:
        b64         = load_and_encode_image(path)
        raw         = query_ollama(model_id, b64)
        price_cents = parse_price(raw)

        if price_cents is None:
            print(f"SKIP  (response: {raw!r})")
            return None

        target = build_renamed_path(path, price_cents)
        path.rename(target)
        euro = price_cents // 100
        cent = price_cents % 100
        print(f"OK  →  {target.name}  ({euro},{cent:02d} €)")
        return price_cents

    except Exception as exc:
        print(f"ERROR  ({exc})")
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    model_id: str = DEFAULT_MODEL,
    input_dir: Path = INPUT_DIR,
) -> dict:
    """Scan *input_dir*, OCR each slip, rename files, and print a summary.

    Args:
        model_id: Local ollama model to use.  Must be installed.
        input_dir: Directory containing the slip images to process.

    Returns:
        Summary dictionary::

            {
                "processed":   int,   # files successfully OCR-ed and renamed
                "skipped":     int,   # files where no price was found
                "total_cents": int,   # sum of all detected prices in cents
                "results":     list,  # per-file dicts with "file" and "price_cents"
            }

    Raises:
        SystemExit(1): If *model_id* is not installed in ollama.
        FileNotFoundError: If *input_dir* does not exist.
    """
    ensure_model_available(model_id)

    files = collect_input_files(input_dir)
    if not files:
        print(f"No unprocessed image files found in: {input_dir}/")
        return {"processed": 0, "skipped": 0, "total_cents": 0, "results": []}

    print(f"Found {len(files)} file(s)  [model: {model_id}]\n")

    results: list[dict] = []
    for f in files:
        price = process_file(f, model_id)
        results.append({"file": f.name, "price_cents": price})

    ok     = [r for r in results if r["price_cents"] is not None]
    failed = [r for r in results if r["price_cents"] is None]
    total  = sum(r["price_cents"] for r in ok)

    print(f"\n{'─' * 50}")
    print(f"  Processed : {len(ok)}")
    print(f"  Skipped   : {len(failed)}")
    print(f"  Total     : {total // 100},{total % 100:02d} €")
    print(f"{'─' * 50}")

    return {
        "processed":   len(ok),
        "skipped":     len(failed),
        "total_cents": total,
        "results":     results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Build and return parsed CLI arguments.

    Args:
        argv: Explicit argument list; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="OCR German grocery/gas sales slips with a local ollama vision model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Default model : {DEFAULT_MODEL}\n"
            "Workflow      : drop images into input/ → run this script\n"
            "Rename format : receipt.jpg → receipt_7949.jpg  (79,49 €)"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL_NAME",
        help=(
            "ollama model to use for OCR (default: %(default)s). "
            "Must be installed locally; see 'ollama list'."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    run(model_id=_args.model)
