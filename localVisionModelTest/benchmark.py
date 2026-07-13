#!/usr/bin/env python3
"""
Vision model benchmark on the local ollama server.
Tests each model on German grocery sales slip images and measures accuracy + latency.
GPU target: NVIDIA RTX A2000 8 GB Laptop GPU.
"""

import os
import re
import subprocess
import sys
import threading
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

from receipt_ocr import MAX_SIDE_PX, PROMPT, encode_image, format_price, parse_price, query_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = REPO_ROOT / "test_images"
OUTPUT_PDF = Path(__file__).resolve().parent / "results.pdf"

# ---------------------------------------------------------------------------
# Models to benchmark
# keep=True  → always retain on disk
# keep=False → retain unless free disk < 2 GB (emergency eviction only)
# ---------------------------------------------------------------------------

# All known candidates have been benchmarked. Add new model IDs here for
# future benchmark runs; they will be downloaded, tested, and kept on disk
# unless the 2 GB emergency eviction threshold is hit.
MODELS: list[dict] = []

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
        valid = [r for r in self.image_results if r.error is None]
        if not valid:
            return 0.0
        return sum(r.correct for r in valid) / len(valid)

    @property
    def correct_count(self) -> int:
        return sum(r.correct for r in self.image_results if r.error is None)

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
    present = installed_model_ids()
    base = model_id.split(":")[0]
    tag  = model_id.split(":")[1] if ":" in model_id else ""
    return any(
        mid == model_id or (tag and mid.startswith(base + ":" + tag))
        for mid in present
    )


def free_disk_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1024 ** 3


def pull_model(model_id: str, silent: bool = False) -> float:
    if _is_installed(model_id):
        if not silent:
            print(f"  [cache] {model_id} already installed")
        return 0.0
    free = free_disk_gb()
    if free < 2.0:
        raise RuntimeError(f"only {free:.1f} GB free — aborting pull to protect disk")
    if not silent:
        print(f"  [pull]  downloading {model_id} … ({free:.1f} GB free)", flush=True)
    t0 = time.monotonic()
    for chunk in ollama.pull(model_id, stream=True):
        if not silent:
            status = getattr(chunk, "status", "") or ""
            if "pulling" in status or "success" in status:
                print(f"\r         {status[:80]}    ", end="", flush=True)
    if not silent:
        print()
    return time.monotonic() - t0


def remove_model(model_id: str) -> None:
    try:
        ollama.delete(model_id)
        print(f"  [cleanup] removed {model_id} from disk")
    except Exception as e:
        print(f"  [cleanup] could not remove {model_id}: {e}")


def prefetch_model(model_id: str) -> Optional[threading.Thread]:
    if _is_installed(model_id):
        return None
    print(f"  [prefetch] background download of {model_id} …", flush=True)
    t = threading.Thread(target=pull_model, args=(model_id, True), daemon=True)
    t.start()
    return t


def extract_expected(filename: str) -> str:
    m = re.search(r"_(\d+)\.", filename)
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

def benchmark() -> list[ModelResult]:
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not test_images:
        sys.exit(f"No JPEG images found in {TEST_IMAGES_DIR}")

    results: list[ModelResult] = []
    prefetch_thread: Optional[threading.Thread] = None

    for i, spec in enumerate(MODELS):
        mid = spec["id"]
        print(f"\n{'='*60}\nModel: {spec['display']} ({mid})\n{'='*60}")

        if prefetch_thread is not None:
            if prefetch_thread.is_alive():
                print("  [prefetch] waiting for download …", flush=True)
                prefetch_thread.join()
            prefetch_thread = None

        try:
            pull_time = pull_model(mid)
        except Exception as e:
            print(f"  [skip] {e}")
            results.append(ModelResult(mid, spec["display"], spec["size"], 0.0,
                                       skipped=True, skip_reason=str(e)))
            continue

        mr = ModelResult(mid, spec["display"], spec["size"], pull_time)

        # Prefetch next keep=True model while running inference
        next_spec = MODELS[i + 1] if i + 1 < len(MODELS) else None
        if next_spec and next_spec.get("keep", True):
            prefetch_thread = prefetch_model(next_spec["id"])

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

        # Emergency eviction from disk only
        if free_disk_gb() < 2.0 and not spec.get("keep", True):
            print(f"  [disk] emergency eviction of {mid}")
            remove_model(mid)

    if prefetch_thread and prefetch_thread.is_alive():
        prefetch_thread.join()

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
    # Merge live + archived, sort by accuracy desc then latency asc
    all_results = live_results + ARCHIVED_RESULTS
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
        "<b>Accuracy</b> = share of 3 annotated test images where the extracted value "
        "matches the ground truth encoded in the filename "
        "(<i>slip0_7949.jpg</i> → 79,49 €). "
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

def gpu_info() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        return "GPU info unavailable"


if __name__ == "__main__":
    print("=== Local Vision Model Benchmark ===")
    print(f"GPU: {gpu_info()}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Models to run: {len(MODELS)}  (+{len(ARCHIVED_RESULTS)} archived)")

    live = benchmark()

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
