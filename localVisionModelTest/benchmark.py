#!/usr/bin/env python3
"""
Vision model benchmark on the local ollama server.
Runs a receipt-OCR smoke test and measures exact-match rate plus latency.
GPU target: NVIDIA RTX A2000 8 GB Laptop GPU.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
    KeepTogether,
)

from receipt_ocr import (
    MAX_SIDE_PX,
    PROMPT,
    encode_image,
    format_price,
    model_id_is_available,
    parse_price,
    query_model,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = REPO_ROOT / "test_images"
OUTPUT_PDF = Path(__file__).resolve().parent / "results.pdf"
MIN_FREE_GB = 2.0

# ---------------------------------------------------------------------------
# Models available to benchmark. No models run unless selected on the CLI.
# ---------------------------------------------------------------------------

MODELS: list[dict] = [
    {"id": "moondream", "display": "Moondream 2", "size": "~1.7 GB", "size_gb": 1.7},
    {"id": "llava-phi3", "display": "LLaVA-Phi3", "size": "~2.9 GB", "size_gb": 2.9},
    {"id": "gemma3:4b", "display": "Gemma 3 4B", "size": "~3.3 GB", "size_gb": 3.3},
    {"id": "llava:7b", "display": "LLaVA 1.5 7B", "size": "~4.7 GB", "size_gb": 4.7},
    {"id": "Keyvan/german-ocr-3", "display": "German-OCR-3", "size": "~2.7 GB", "size_gb": 2.7},
    {
        "id": "richardyoung/smolvlm2-2.2b-instruct",
        "display": "SmolVLM2 2.2B",
        "size": "~1.1 GB",
        "size_gb": 1.1,
    },
    {"id": "minicpm-o:4b", "display": "MiniCPM-o 4B", "size": "~5.0 GB", "size_gb": 5.0},
    {"id": "qwen3-vl:4b", "display": "Qwen3-VL 4B", "size": "~3.3 GB", "size_gb": 3.3},
    {"id": "qwen2.5-vl:7b", "display": "Qwen2.5-VL 7B", "size": "~5.5 GB", "size_gb": 5.5},
    {"id": "qwen2-vl:7b", "display": "Qwen2-VL 7B", "size": "~5.5 GB", "size_gb": 5.5},
    {"id": "minicpm-v", "display": "MiniCPM-V 2.6", "size": "~5.5 GB", "size_gb": 5.5},
    {"id": "bakllava", "display": "BakLLaVA 7B", "size": "~4.7 GB", "size_gb": 4.7},
    {
        "id": "llama3.2-vision:11b",
        "display": "Llama 3.2-Vision 11B",
        "size": "~7.9 GB",
        "size_gb": 7.9,
    },
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    filename: str
    expected: str
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
    archived: bool = False   # True → benchmarked in a prior run, no longer on disk
    image_results: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.image_results:
            return 0.0
        return self.correct_count / len(self.image_results)

    @property
    def correct_count(self) -> int:
        return sum(r.correct for r in self.image_results)

    @property
    def avg_latency_s(self) -> float:
        valid = [r for r in self.image_results if r.error is None]
        if not valid:
            return float("nan")
        return sum(r.latency_s for r in valid) / len(valid)


# ---------------------------------------------------------------------------
# Archived results from prior runs (models removed from disk after testing)
# ---------------------------------------------------------------------------

ARCHIVED_RESULTS: list[ModelResult] = [
    # ── 100 % ───────────────────────────────────────────────────────────────
    ModelResult("llama3.2-vision:11b",     "Llama 3.2-Vision 11B", "~7.9 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "79,49", True,  18.4),
        ImageResult("slip1_2841.jpg", "28,41", "28,41", True,   8.3),
        ImageResult("slip2_1093.jpg", "10,93", "10,93", True,   7.9),
    ]),
    ModelResult("qwen3-vl:4b",             "Qwen3-VL 4B",          "~3.3 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "79,49", True,  29.1),
        ImageResult("slip1_2841.jpg", "28,41", "28,41", True,   5.7),
        ImageResult("slip2_1093.jpg", "10,93", "10,93", True,  11.0),
    ]),
    # ── 67 % ────────────────────────────────────────────────────────────────
    ModelResult("moondream",               "Moondream 2",          "~1.7 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "79,49", True,  0.3),
        ImageResult("slip1_2841.jpg", "28,41", "28,41", True,  0.3),
        ImageResult("slip2_1093.jpg", "10,93", "79,49", False, 0.3),
    ]),
    # MiniCPM-V errored in the automated benchmark run (VRAM flush timing);
    # result below is from a verified standalone retest immediately after.
    ModelResult("minicpm-v",               "MiniCPM-V 2.6",        "~5.5 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "66,80", False, 8.4),
        ImageResult("slip1_2841.jpg", "28,41", "28,41", True,  3.5),
        ImageResult("slip2_1093.jpg", "10,93", "10,93", True,  3.6),
    ]),
    # ── 33 % ────────────────────────────────────────────────────────────────
    ModelResult("bakllava",                "BakLLaVA 7B",          "~4.7 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "79,49", True,  2.6),
        ImageResult("slip1_2841.jpg", "28,41", "79,49", False, 0.9),
        ImageResult("slip2_1093.jpg", "10,93", "79,49", False, 0.9),
    ]),
    ModelResult("llava-phi3",              "LLaVA-Phi3",           "~2.9 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "85,01", False, 4.0),
        ImageResult("slip1_2841.jpg", "28,41", "28,41", True,  0.6),
        ImageResult("slip2_1093.jpg", "10,93", "11,03", False, 0.6),
    ]),
    ModelResult("gemma3:4b",               "Gemma 3 4B",           "~3.3 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "12,69", False, 5.6),
        ImageResult("slip1_2841.jpg", "28,41", "NaN",   False, 2.1),
        ImageResult("slip2_1093.jpg", "10,93", "10,93", True,  2.1),
    ]),
    ModelResult("Keyvan/german-ocr-3",     "German-OCR-3",         "~2.7 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "NaN",   False, 93.8),
        ImageResult("slip1_2841.jpg", "28,41", "28,41", True,  27.7),
        ImageResult("slip2_1093.jpg", "10,93", "NaN",   False, 98.7),
    ]),
    # ── 0 % ─────────────────────────────────────────────────────────────────
    ModelResult("llava:7b",                "LLaVA 1.5 7B",         "~4.7 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "NaN",   False, 2.7),
        ImageResult("slip1_2841.jpg", "28,41", "NaN",   False, 0.9),
        ImageResult("slip2_1093.jpg", "10,93", "NaN",   False, 0.9),
    ]),
    ModelResult("richardyoung/smolvlm2-2.2b-instruct", "SmolVLM2 2.2B", "~1.1 GB", 0, archived=True, image_results=[
        ImageResult("slip0_7949.jpg", "79,49", "123,45", False, 1.4),
        ImageResult("slip1_2841.jpg", "28,41", "79,49",  False, 0.2),
        ImageResult("slip2_1093.jpg", "10,93", "79,49",  False, 0.2),
    ]),
    # ── not available in ollama registry ────────────────────────────────────
    ModelResult("minicpm-o:4b",  "MiniCPM-o 4B",   "~5.0 GB", 0,
                skipped=True, skip_reason="model not found in ollama registry"),
    ModelResult("qwen2.5-vl:7b", "Qwen2.5-VL 7B",  "~5.5 GB", 0,
                skipped=True, skip_reason="pull failed (disk / registry)"),
    ModelResult("qwen2-vl:7b",   "Qwen2-VL 7B",    "~5.5 GB", 0,
                skipped=True, skip_reason="pull failed (disk / registry)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def installed_model_ids() -> set:
    try:
        return {m.model for m in ollama.list().models}
    except Exception:
        return set()


def _is_installed(model_id: str) -> bool:
    return model_id_is_available(model_id, installed_model_ids())


def model_storage_path() -> Path:
    """Return the configured Ollama model-storage location."""
    return Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def free_disk_gb(path: Path) -> float:
    """Return free space for the filesystem containing *path*."""
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing).free / 1024 ** 3


def pull_model(spec: dict, allow_downloads: bool) -> float:
    """Use an installed model or explicitly and capacity-safely download it."""
    model_id = spec["id"]
    if _is_installed(model_id):
        print(f"  [cache] {model_id} already installed")
        return 0.0
    if not allow_downloads:
        raise RuntimeError("model is not installed; use --allow-downloads to pull it")

    storage = model_storage_path()
    free = free_disk_gb(storage)
    required = float(spec["size_gb"]) + MIN_FREE_GB
    if free < required:
        raise RuntimeError(
            f"{free:.1f} GB free in {storage}; {required:.1f} GB required "
            f"for the estimated model size plus {MIN_FREE_GB:.0f} GB reserve"
        )
    print(f"  [pull]  downloading {model_id} … ({free:.1f} GB free)", flush=True)
    t0 = time.monotonic()
    for chunk in ollama.pull(model_id, stream=True):
        status = getattr(chunk, "status", "") or ""
        if "pulling" in status or "success" in status:
            print(f"\r         {status[:80]}    ", end="", flush=True)
    print()
    return time.monotonic() - t0


def extract_expected(filename: str) -> str:
    m = re.search(r"_(\d+)\.[^.]+$", filename)
    if not m:
        return "?"
    cents = int(m.group(1))
    return f"{cents // 100},{cents % 100:02d}"


def parse_response(text: str) -> str:
    price_cents = parse_price(text)
    return format_price(price_cents) if price_cents is not None else "NaN"


def run_model_on_image(model_id: str, image_path: Path) -> tuple[str, float]:
    b64 = encode_image(image_path)
    t0 = time.monotonic()
    response = query_model(model_id, b64)
    return response, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(model_specs: list[dict], allow_downloads: bool = False) -> list[ModelResult]:
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not test_images:
        sys.exit(f"No JPEG images found in {TEST_IMAGES_DIR}")

    results: list[ModelResult] = []
    for spec in model_specs:
        mid = spec["id"]
        print(f"\n{'='*60}\nModel: {spec['display']} ({mid})\n{'='*60}")

        try:
            pull_time = pull_model(spec, allow_downloads)
        except Exception as e:
            print(f"  [skip] {e}")
            results.append(ModelResult(mid, spec["display"], spec["size"], 0.0,
                                       skipped=True, skip_reason=str(e)))
            continue

        mr = ModelResult(mid, spec["display"], spec["size"], pull_time)

        for img_path in test_images:
            expected = extract_expected(img_path.name)
            print(f"  image: {img_path.name}  expected={expected}", end=" … ", flush=True)
            try:
                raw, latency = run_model_on_image(mid, img_path)
                parsed = parse_response(raw)
                correct = (parsed == expected)
                print(f"got='{parsed}'  {'OK' if correct else 'WRONG'}  {latency:.1f}s")
                mr.image_results.append(ImageResult(img_path.name, expected,
                                                     raw[:120], correct, latency))
            except Exception as e:
                print(f"ERROR: {e}")
                mr.image_results.append(ImageResult(img_path.name, expected,
                                                     "", False, 0.0, str(e)[:120]))

        results.append(mr)

        # Evict from VRAM
        try:
            ollama.chat(model=mid,
                        messages=[{"role": "user", "content": "."}],
                        options={"num_predict": 1, "keep_alive": "0"})
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

# Accuracy colour bands
def acc_color(pct: float) -> colors.Color:
    if pct >= 1.0:
        return colors.HexColor("#1e8449")  # green
    if pct >= 0.67:
        return colors.HexColor("#d4ac0d")  # gold
    if pct >= 0.33:
        return colors.HexColor("#e67e22")  # orange
    return colors.HexColor("#c0392b")  # red


def build_pdf(live_results: list[ModelResult], gpu_info: str):
    # A live result supersedes an archived result for the same model.
    live_ids = {result.model_id for result in live_results}
    all_results = live_results + [
        result for result in ARCHIVED_RESULTS if result.model_id not in live_ids
    ]
    ranked = sorted(
        [r for r in all_results if not r.skipped],
        key=lambda r: (-r.accuracy, r.avg_latency_s if r.avg_latency_s == r.avg_latency_s else 9999)
    )
    skipped = [r for r in all_results if r.skipped]

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )
    styles = getSampleStyleSheet()

    def style(name, **kw):
        s = ParagraphStyle(name, parent=styles["Normal"], **kw)
        return s

    title_s  = style("T",  fontSize=18, fontName="Helvetica-Bold", spaceAfter=2)
    sub_s    = style("S",  fontSize=8,  textColor=colors.HexColor("#555555"), spaceAfter=6)
    h2_s     = style("H2", fontSize=10, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    body_s   = style("B",  fontSize=8,  leading=11)
    note_s   = style("N",  fontSize=6,  textColor=colors.grey, leading=8)

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Vision Model Benchmark", title_s))
    story.append(Paragraph(
        f"German Sales-Slip OCR &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d')} "
        f"&nbsp;·&nbsp; {gpu_info}",
        sub_s
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2c3e50")))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph(
        "<b>Task:</b> Extract the total amount from German grocery / gas sales slips. "
        "Each model receives the raw JPEG (resized ≤ 1500 px longest side) and a fixed "
        "prompt asking for the sum in <i>Euro,Cent</i> format. "
        "<b>Smoke-test exact-match rate</b> = share of 3 annotated test images where the extracted value "
        "matches the ground truth encoded in the filename "
        "(<i>slip0_7949.jpg</i> → 79,49 €). "
        "This tiny convenience sample is not an estimate of production accuracy. "
        "Models marked <i>archived</i> were tested in an earlier run and removed from "
        "disk afterwards.",
        body_s
    ))
    story.append(Spacer(1, 0.35*cm))

    # ── Leaderboard ─────────────────────────────────────────────────────────
    story.append(Paragraph("Leaderboard", h2_s))

    hdr = ["#", "Model", "Size", "Accuracy", "Avg latency", "On disk"]
    rows = [hdr]
    row_colors = []

    for rank, mr in enumerate(ranked, 1):
        n_total = len(mr.image_results)
        acc_str = f"{mr.correct_count}/{n_total}  ({mr.accuracy*100:.0f} %)"
        lat = f"{mr.avg_latency_s:.1f} s" if mr.avg_latency_s == mr.avg_latency_s else "—"
        on_disk = "archived" if mr.archived else "✓"
        rows.append([str(rank), mr.display_name, mr.size_label, acc_str, lat, on_disk])
        row_colors.append(mr.accuracy)

    cw = [0.7*cm, 4.5*cm, 1.8*cm, 3.2*cm, 2.5*cm, 2.0*cm]
    tbl = Table(rows, colWidths=cw, repeatRows=1)

    ts = [
        # Header
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("ALIGN",        (0,0), (0,-1), "CENTER"),   # rank col
        ("ALIGN",        (3,0), (4,-1), "RIGHT"),    # acc + latency
        ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),  # rank bold
    ]
    # Colour-code the accuracy cell per row
    for i, pct in enumerate(row_colors, 1):
        c = acc_color(pct)
        ts.append(("TEXTCOLOR", (3, i), (3, i), c))
        ts.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))

    tbl.setStyle(TableStyle(ts))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Per-model detail ────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbbbbb")))
    story.append(Paragraph("Per-model responses", h2_s))

    for mr in ranked + skipped:
        label = mr.display_name
        if mr.archived:
            label += "  <font color='#888888' size='6'>(archived)</font>"
        if mr.skipped:
            label += "  <font color='#c0392b' size='6'>(skipped)</font>"

        block = [Paragraph(f"<b>{label}</b>  <font size='6.5' color='#666666'>{mr.model_id}</font>", body_s)]

        if mr.skipped:
            block.append(Paragraph(f"Not tested — {mr.skip_reason}", note_s))
        else:
            det_hdr = ["Image", "Expected", "Model response", "Parsed", "✓/✗", "Time"]
            det_rows = [det_hdr]
            for ir in mr.image_results:
                parsed = parse_response(ir.response) if not ir.error else "ERR"
                raw_cell = (ir.error or ir.response.replace("\n", " "))[:90]
                det_rows.append([
                    ir.filename, ir.expected, raw_cell, parsed,
                    "✓" if ir.correct else "✗",
                    f"{ir.latency_s:.1f} s" if not ir.error else "—"
                ])

            dcw = [2.5*cm, 1.7*cm, 6.2*cm, 1.5*cm, 0.8*cm, 1.8*cm]
            dt = Table(det_rows, colWidths=dcw, repeatRows=1)
            dts = [
                ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#455a64")),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 6.5),
                ("LEADING",       (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#eceff1")]),
                ("GRID",          (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ("LEFTPADDING",   (0,0), (-1,-1), 4),
                ("RIGHTPADDING",  (0,0), (-1,-1), 4),
                ("ALIGN",         (4,1), (5,-1), "CENTER"),
            ]
            # Colour ✓/✗ per row
            for ri, ir in enumerate(mr.image_results, 1):
                c = colors.HexColor("#1e8449") if ir.correct else colors.HexColor("#c0392b")
                dts.append(("TEXTCOLOR", (4, ri), (4, ri), c))
                dts.append(("FONTNAME",  (4, ri), (4, ri), "Helvetica-Bold"))
            dt.setStyle(TableStyle(dts))
            block.append(dt)

        block.append(Spacer(1, 0.3*cm))
        story.append(KeepTogether(block))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(f"<b>Prompt:</b> {PROMPT}", note_s))
    story.append(Paragraph(
        f"Test images: {TEST_IMAGES_DIR} &nbsp;·&nbsp; "
        f"Max image side: {MAX_SIDE_PX} px &nbsp;·&nbsp; temperature=0",
        note_s
    ))

    doc.build(story)
    print(f"\n[pdf] written → {OUTPUT_PDF}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse explicit model selections for a benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=[spec["id"] for spec in MODELS],
        help="model to run; repeat to select more than one",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every configured model",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="allow selected missing models to be downloaded after a capacity check",
    )
    args = parser.parse_args(argv)
    if args.all and args.model:
        parser.error("--all and --model cannot be combined")
    return args


def selected_models(args: argparse.Namespace) -> list[dict]:
    """Resolve command-line selections to model specifications."""
    if args.all:
        return MODELS
    selected_ids = set(args.model or [])
    return [spec for spec in MODELS if spec["id"] in selected_ids]

def gpu_info() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        return "GPU info unavailable"


def main(argv: Optional[list[str]] = None) -> int:
    """Run selected models and rebuild the archived/live smoke-test report."""
    args = parse_args(argv)
    model_specs = selected_models(args)
    print("=== Local Vision Model Benchmark ===")
    print(f"GPU: {gpu_info()}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Models to run: {len(model_specs)}  (+{len(ARCHIVED_RESULTS)} archived)")

    live = benchmark(model_specs, allow_downloads=args.allow_downloads)

    print("\n=== Building PDF report ===")
    build_pdf(live, gpu_info())

    all_res = live + ARCHIVED_RESULTS
    ranked = sorted([r for r in all_res if not r.skipped],
                    key=lambda r: (-r.accuracy, r.avg_latency_s
                                  if r.avg_latency_s == r.avg_latency_s else 9999))
    print(f"\n{'#':<3} {'Model':<28} {'Acc':>6}  {'Avg lat':>9}  {'Disk'}")
    print("-" * 58)
    for i, mr in enumerate(ranked, 1):
        lat = f"{mr.avg_latency_s:.1f}s" if mr.avg_latency_s == mr.avg_latency_s else "N/A"
        flag = "archived" if mr.archived else "on disk"
        print(f"{i:<3} {mr.display_name:<28} {mr.accuracy*100:>5.0f}%  {lat:>9}  {flag}")
    for mr in [r for r in all_res if r.skipped]:
        print(f"{'—':<3} {mr.display_name:<28}  skip   —          skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
