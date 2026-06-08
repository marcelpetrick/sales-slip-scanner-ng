#!/usr/bin/env python3
"""
Vision model benchmark on the local ollama server.
Tests each model on German grocery sales slip images and measures accuracy + latency.
GPU target: NVIDIA RTX A2000 8GB Laptop GPU (~7.8 GB free VRAM).
"""

import base64
import io
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = REPO_ROOT / "test_images"
OUTPUT_PDF = Path(__file__).resolve().parent / "results.pdf"

# Vision-capable models that fit in 8 GB VRAM.
# Sizes are approximate on-disk / VRAM footprints at listed quantisation.
# Current run: limited to models already on disk due to available disk space.
# See localVisionModelTest/modelsToTest.md for the full candidate list.
MODELS = [
    {"id": "moondream",  "display": "Moondream 2",   "size": "~1.7 GB"},
    {"id": "llava-phi3", "display": "LLaVA-Phi3",    "size": "~2.9 GB"},
    {"id": "gemma3:4b",  "display": "Gemma 3 4B",    "size": "~3.3 GB"},
    {"id": "llava:7b",   "display": "LLaVA 1.5 7B",  "size": "~4.7 GB"},
]

PROMPT = (
    "What is the sum to pay in the given sales slip? "
    "It is a German sales slip for groceries or gas. "
    "Look for 'Summe', 'Gesamt' or 'zu zahlen'. "
    "Reply with ONLY the amount in the format 'Euro,Cent' (e.g. '79,49'). "
    "No currency symbol, no extra text. If not found, reply 'NaN'."
)

MAX_SIDE_PX = 1500  # resize before sending

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    filename: str
    expected: str       # e.g. "79,49"
    response: str
    correct: bool
    latency_s: float
    error: Optional[str] = None


@dataclass
class ModelResult:
    model_id: str
    display_name: str
    size_label: str
    pull_time_s: float
    skipped: bool = False
    skip_reason: str = ""
    image_results: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        valid = [r for r in self.image_results if r.error is None]
        if not valid:
            return 0.0
        return sum(r.correct for r in valid) / len(valid)

    @property
    def avg_latency_s(self) -> float:
        valid = [r for r in self.image_results if r.error is None]
        if not valid:
            return float("nan")
        return sum(r.latency_s for r in valid) / len(valid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def installed_model_ids() -> set:
    try:
        resp = ollama.list()
        return {m.model for m in resp.models}
    except Exception:
        return set()


def pull_model(model_id: str) -> float:
    """Pull model if absent; return elapsed seconds."""
    present = installed_model_ids()
    # match both "llava:7b" and "llava:7b-..." variants
    already = any(mid == model_id or mid.startswith(model_id.split(":")[0] + ":" + (model_id.split(":")[1] if ":" in model_id else "")) for mid in present)
    if already:
        print(f"  [cache] {model_id} already installed")
        return 0.0
    print(f"  [pull]  downloading {model_id} …", flush=True)
    t0 = time.monotonic()
    # Stream pull progress to stdout so the user sees progress
    for chunk in ollama.pull(model_id, stream=True):
        status = getattr(chunk, "status", "") or ""
        if "pulling" in status or "success" in status:
            print(f"\r         {status[:80]}    ", end="", flush=True)
    print()
    return time.monotonic() - t0


def encode_image(path: Path) -> str:
    """Resize to MAX_SIDE_PX and return base64-encoded JPEG string."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, MAX_SIDE_PX / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


def extract_expected(filename: str) -> str:
    """slip0_7949.jpg → '79,49'"""
    m = re.search(r"_(\d+)\.", filename)
    if not m:
        return "?"
    cents = int(m.group(1))
    return f"{cents // 100},{cents % 100:02d}"


def parse_response(text: str) -> str:
    """Return a normalised 'Euro,Cent' string or 'NaN'."""
    text = text.strip()
    # Accept "79,49", "79.49", "€79,49", "79,49 €", etc.
    m = re.search(r"(\d+)[,.](\d{2})", text)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    return "NaN"


def run_model_on_image(model_id: str, image_path: Path) -> tuple[str, float]:
    """Return (raw_response_text, latency_s)."""
    b64 = encode_image(image_path)
    t0 = time.monotonic()
    resp = ollama.chat(
        model=model_id,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [b64],
        }],
        options={"temperature": 0},
    )
    latency = time.monotonic() - t0
    return resp.message.content.strip(), latency


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark() -> list:
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not test_images:
        sys.exit(f"No JPEG images found in {TEST_IMAGES_DIR}")

    results: list[ModelResult] = []

    for spec in MODELS:
        mid = spec["id"]
        print(f"\n{'='*60}")
        print(f"Model: {spec['display']} ({mid})")
        print(f"{'='*60}")

        # --- pull ---
        try:
            pull_time = pull_model(mid)
        except Exception as e:
            print(f"  [skip] pull failed: {e}")
            mr = ModelResult(mid, spec["display"], spec["size"], 0.0,
                             skipped=True, skip_reason=f"pull failed: {e}")
            results.append(mr)
            continue

        mr = ModelResult(mid, spec["display"], spec["size"], pull_time)

        # --- inference ---
        for img_path in test_images:
            expected = extract_expected(img_path.name)
            print(f"  image: {img_path.name}  expected={expected}", end=" … ", flush=True)
            try:
                raw, latency = run_model_on_image(mid, img_path)
                parsed = parse_response(raw)
                correct = (parsed == expected)
                print(f"got='{parsed}'  {'OK' if correct else 'WRONG'}  {latency:.1f}s")
                mr.image_results.append(ImageResult(
                    filename=img_path.name,
                    expected=expected,
                    response=raw[:120],  # truncate for PDF readability
                    correct=correct,
                    latency_s=latency,
                ))
            except Exception as e:
                print(f"ERROR: {e}")
                mr.image_results.append(ImageResult(
                    filename=img_path.name,
                    expected=expected,
                    response="",
                    correct=False,
                    latency_s=0.0,
                    error=str(e)[:120],
                ))

        results.append(mr)
        # unload model from VRAM before loading the next
        try:
            ollama.chat(model=mid, messages=[{"role": "user", "content": "."}],
                        options={"num_predict": 1, "keep_alive": "0"})
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def build_pdf(results: list, gpu_info: str):
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    h1 = styles["h1"]
    h2 = styles["h2"]
    body = styles["BodyText"]
    body.fontSize = 8
    body.leading = 11

    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=7, leading=10)
    small = ParagraphStyle("small", parent=body, fontSize=7, leading=10)
    ok_style = ParagraphStyle("ok", parent=small, textColor=colors.darkgreen)
    bad_style = ParagraphStyle("bad", parent=small, textColor=colors.red)

    story = []

    # Title
    story.append(Paragraph("Local Vision Model Benchmark", h1))
    story.append(Paragraph(
        f"Sales-slip OCR · {datetime.now().strftime('%Y-%m-%d %H:%M')} · {gpu_info}",
        small
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3*cm))

    # Task description
    story.append(Paragraph(
        "Task: extract the total amount from German grocery/gas sales slips. "
        "Each model receives the raw JPEG (resized to ≤1500 px) and a fixed prompt. "
        "Accuracy = fraction of the 3 test images where the extracted value matches "
        "the ground truth encoded in the filename (e.g. <i>slip0_7949.jpg</i> → 79,49 €).",
        body
    ))
    story.append(Spacer(1, 0.3*cm))

    # --- Summary table ---
    story.append(Paragraph("Summary", h2))
    header = ["Model", "VRAM", "Pull (s)", "Accuracy", "Avg latency (s)", "Status"]
    rows = [header]
    for mr in results:
        if mr.skipped:
            status = f"skipped: {mr.skip_reason}"
            rows.append([mr.display_name, mr.size_label, "-", "-", "-", status])
        else:
            acc = f"{mr.accuracy*100:.0f}% ({sum(r.correct for r in mr.image_results)}/{len(mr.image_results)})"
            lat = f"{mr.avg_latency_s:.1f}" if mr.avg_latency_s == mr.avg_latency_s else "N/A"
            pull = f"{mr.pull_time_s:.0f}" if mr.pull_time_s else "cached"
            rows.append([mr.display_name, mr.size_label, pull, acc, lat, "ok"])

    col_w = [4.0*cm, 1.8*cm, 1.8*cm, 2.5*cm, 3.2*cm, 2.5*cm]
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.grey),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",(0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    # --- Per-model detail ---
    story.append(Paragraph("Per-model responses", h2))
    for mr in results:
        story.append(Paragraph(f"<b>{mr.display_name}</b> ({mr.model_id})", body))
        if mr.skipped:
            story.append(Paragraph(f"Skipped — {mr.skip_reason}", bad_style))
            story.append(Spacer(1, 0.2*cm))
            continue

        det_rows = [["Image", "Expected", "Response (truncated)", "Parsed", "Match", "Latency"]]
        for ir in mr.image_results:
            parsed = parse_response(ir.response) if not ir.error else "ERR"
            match_label = "✓" if ir.correct else "✗"
            lat_label = f"{ir.latency_s:.1f}s" if not ir.error else "—"
            raw_cell = ir.error if ir.error else ir.response.replace("\n", " ")[:80]
            det_rows.append([ir.filename, ir.expected, raw_cell, parsed, match_label, lat_label])

        dt = Table(det_rows, colWidths=[3.0*cm, 1.8*cm, 5.5*cm, 1.5*cm, 1.0*cm, 2.0*cm], repeatRows=1)
        dt.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#546e7a")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 6.5),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#eceff1")]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.grey),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 3),
            ("RIGHTPADDING",(0,0), (-1,-1), 3),
        ]))
        story.append(dt)
        story.append(Spacer(1, 0.25*cm))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "Prompt used: \"" + PROMPT + "\"",
        ParagraphStyle("footer", parent=body, fontSize=6, textColor=colors.grey)
    ))

    doc.build(story)
    print(f"\n[pdf] written → {OUTPUT_PDF}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
        return out
    except Exception:
        return "GPU info unavailable"


if __name__ == "__main__":
    print("=== Local Vision Model Benchmark ===")
    print(f"GPU: {gpu_info()}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Models to test: {len(MODELS)}")

    results = benchmark()

    print("\n=== Building PDF report ===")
    build_pdf(results, gpu_info())

    # Print quick ASCII summary
    print("\n=== Results summary ===")
    print(f"{'Model':<28} {'Acc':>6}  {'Avg lat':>9}")
    print("-" * 48)
    for mr in results:
        if mr.skipped:
            print(f"{mr.display_name:<28}   skip  {mr.skip_reason[:20]}")
        else:
            acc = f"{mr.accuracy*100:.0f}%"
            lat = f"{mr.avg_latency_s:.1f}s" if mr.avg_latency_s == mr.avg_latency_s else "N/A"
            print(f"{mr.display_name:<28} {acc:>6}  {lat:>9}")
