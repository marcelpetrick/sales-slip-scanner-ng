#!/usr/bin/env python3
"""Process receipt images from a local hot folder with an Ollama vision model.

Images are never renamed, moved, or deleted. Successful receipt fingerprints
are recorded in an atomic state file, and a Markdown report contains every
detected amount plus the cumulative grand total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ollama

from receipt_ocr import (
    encode_image,
    format_price,
    model_id_is_available,
    parse_price,
    query_model,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3.5:4b"
HOT_FOLDER = ROOT / "hot_folder"
REPORT_NAME = "receipt-report.md"
STATE_NAME = ".sales-slip-scanner.json"
STATE_VERSION = 2
KEEP_ALIVE = "10m"
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def list_local_models() -> list[str]:
    """Return exact model IDs installed on the reachable Ollama server."""
    try:
        return [model.model for model in ollama.list().models]
    except Exception as exc:
        raise RuntimeError(f"Cannot reach Ollama server: {exc}") from exc


def ensure_model_available(model_id: str) -> None:
    """Exit with installation guidance when the requested model is absent."""
    available = list_local_models()
    if model_id_is_available(model_id, available):
        return
    print(f"Error: model '{model_id}' is not installed in Ollama.", file=sys.stderr)
    print(
        f"  Available: {', '.join(available) if available else '(none)'}",
        file=sys.stderr,
    )
    print(f"  Install:   ollama pull {model_id}", file=sys.stderr)
    raise SystemExit(1)


def warm_model(model_id: str, keep_alive: str) -> None:
    """Load the model before the first receipt and retain it after the batch."""
    try:
        ollama.generate(
            model=model_id,
            prompt="",
            keep_alive=keep_alive,
            options={"temperature": 0, "num_ctx": 4096, "num_predict": 1},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not warm Ollama model '{model_id}': {exc}") from exc


def file_sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_state() -> dict:
    """Return a new scanner state document."""
    return {"version": STATE_VERSION, "receipts": [], "last_failures": []}


def load_state(directory: Path) -> dict:
    """Load and strictly validate the hot folder's scanner state."""
    path = directory / STATE_NAME
    if not path.exists():
        return empty_state()
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    receipts = state.get("receipts")
    failures = state.get("last_failures")
    receipts_valid = isinstance(receipts, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("file"), str)
        and isinstance(item.get("sha256"), str)
        and isinstance(item.get("price_cents"), int)
        for item in receipts
    )
    failures_valid = isinstance(failures, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("file"), str)
        and isinstance(item.get("error"), str)
        for item in failures
    )
    if (
        state.get("version") != STATE_VERSION
        or not receipts_valid
        or not failures_valid
    ):
        raise ValueError(
            f"Invalid or obsolete scanner state: {path}. "
            "Move it aside before starting the hot-folder workflow."
        )
    return state


def atomic_write(path: Path, content: str) -> None:
    """Durably replace a UTF-8 text file without exposing partial content."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def save_state(directory: Path, state: dict) -> None:
    """Atomically persist scanner state after each successful receipt."""
    atomic_write(
        directory / STATE_NAME,
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
    )


def collect_pending(
    directory: Path, processed_hashes: set[str]
) -> list[tuple[Path, str]]:
    """Return sorted supported images whose content has not succeeded before."""
    pending = []
    seen_hashes = set(processed_hashes)
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        digest = file_sha256(path)
        if digest not in seen_hashes:
            pending.append((path, digest))
            seen_hashes.add(digest)
    return pending


def markdown_escape(value: str) -> str:
    """Escape table delimiters in a Markdown cell."""
    return value.replace("\\", "\\\\").replace("|", "\\|")


def render_report(state: dict, model_id: str) -> str:
    """Render all successful receipts and the latest failures as Markdown."""
    receipts = state["receipts"]
    total = sum(item["price_cents"] for item in receipts)
    lines = [
        "# Receipt Report",
        "",
        f"Generated: `{utc_now()}`  ",
        f"Local model: `{model_id}`  ",
        f"Successfully processed receipts: **{len(receipts)}**",
        "",
        "| Receipt | Amount | Processed | SHA-256 |",
        "|---|---:|---|---|",
    ]
    for item in receipts:
        lines.append(
            f"| {markdown_escape(item['file'])} | {format_price(item['price_cents'])} € "
            f"| {item['processed_at']} | `{item['sha256'][:12]}` |"
        )
    if not receipts:
        lines.append("| _No successful receipts yet_ | — | — | — |")
    lines += ["", f"## Grand total: {format_price(total)} €"]
    failures = state["last_failures"]
    if failures:
        lines += ["", "## Latest failed attempts", ""]
        for item in failures:
            lines.append(
                f"- **{markdown_escape(item['file'])}:** {markdown_escape(item['error'])}"
            )
    lines += [
        "",
        "---",
        "",
        "Source images remain unchanged in the hot folder. A receipt is counted once by its SHA-256 content fingerprint.",
    ]
    return "\n".join(lines) + "\n"


def write_report(path: Path, state: dict, model_id: str) -> None:
    """Atomically publish the cumulative Markdown report."""
    atomic_write(path, render_report(state, model_id))


def process_file(path: Path, digest: str, model_id: str, keep_alive: str) -> dict:
    """Extract one receipt total while leaving the source image untouched."""
    print(f"  {path.name}", end=" … ", flush=True)
    try:
        response = query_model(model_id, encode_image(path), keep_alive=keep_alive)
        price_cents = parse_price(response)
        if price_cents is None:
            error = f"unparseable model response {response!r}"
            print(f"FAIL ({error})")
            return {"file": path.name, "sha256": digest, "error": error}
        print(f"OK ({format_price(price_cents)} €)")
        return {
            "file": path.name,
            "sha256": digest,
            "price_cents": price_cents,
            "model": model_id,
            "response": response,
            "processed_at": utc_now(),
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR ({error})")
        return {"file": path.name, "sha256": digest, "error": error}


def run(
    model_id: str = DEFAULT_MODEL,
    input_dir: Path = HOT_FOLDER,
    report_path: Path | None = None,
    keep_alive: str = KEEP_ALIVE,
) -> dict:
    """Process all new receipt content and publish a cumulative report."""
    input_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_path or input_dir / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_model_available(model_id)
    state = load_state(input_dir)
    processed_hashes = {item["sha256"] for item in state["receipts"]}
    pending = collect_pending(input_dir, processed_hashes)
    state["last_failures"] = []

    if not pending:
        save_state(input_dir, state)
        write_report(report_path, state, model_id)
        print(f"No new receipt images found in: {input_dir}")
        print(f"Report: {report_path}")
        return summary(state, [], report_path)

    print(f"Found {len(pending)} new receipt(s) [model: {model_id}]")
    print(f"Warming model and keeping it loaded for {keep_alive} …")
    warm_model(model_id, keep_alive)

    current_results = []
    for path, digest in pending:
        result = process_file(path, digest, model_id, keep_alive)
        current_results.append(result)
        if "price_cents" in result:
            state["receipts"].append(result)
            processed_hashes.add(digest)
        else:
            state["last_failures"].append(result)
        save_state(input_dir, state)
        write_report(report_path, state, model_id)

    result_summary = summary(state, current_results, report_path)
    print("\n========== Receipt Summary ==========")
    print(f"Processed this run : {result_summary['processed']}")
    print(f"Failed this run    : {result_summary['failed']}")
    print(f"All receipts      : {result_summary['receipt_count']}")
    print(f"Grand total       : {format_price(result_summary['total_cents'])} €")
    print(f"Report            : {report_path}")
    print("=====================================")
    return result_summary


def summary(state: dict, current_results: list[dict], report_path: Path) -> dict:
    """Build the public run summary from durable state and current attempts."""
    successful = [item for item in current_results if "price_cents" in item]
    failed = [item for item in current_results if "error" in item]
    return {
        "processed": len(successful),
        "failed": len(failed),
        "receipt_count": len(state["receipts"]),
        "total_cents": sum(item["price_cents"] for item in state["receipts"]),
        "results": current_results,
        "report": str(report_path),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the hot-folder command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hot-folder", type=Path, default=HOT_FOLDER)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep-alive", default=KEEP_ALIVE)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the scanner and fail when any current receipt could not be read."""
    args = parse_args(argv)
    result = run(args.model, args.hot_folder, args.report, args.keep_alive)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
