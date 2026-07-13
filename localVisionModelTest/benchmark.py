#!/usr/bin/env python3
"""Resumable local receipt-OCR benchmark for compact Ollama vision models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ollama
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

from receipt_ocr import MAX_SIDE_PX, PROMPT, encode_image, parse_price

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "test_images"
RESULT_JSON = Path(__file__).with_name("results.json")
RESULT_MD = Path(__file__).with_name("results.md")
RESULT_PDF = Path(__file__).with_name("results.pdf")
SCHEMA = 3


@dataclass(frozen=True)
class ModelSpec:
    model: str
    name: str
    size_gb: float
    source: str


MODELS = [
    ModelSpec(
        "minicpm-v4.6:1b",
        "MiniCPM-V 4.6 1B",
        1.6,
        "https://ollama.com/library/minicpm-v4.6",
    ),
    ModelSpec(
        "qwen3.5:0.8b", "Qwen 3.5 0.8B", 1.0, "https://ollama.com/library/qwen3.5"
    ),
    ModelSpec("qwen3.5:2b", "Qwen 3.5 2B", 2.7, "https://ollama.com/library/qwen3.5"),
    ModelSpec("qwen3.5:4b", "Qwen 3.5 4B", 3.4, "https://ollama.com/library/qwen3.5"),
    ModelSpec("glm-ocr:latest", "GLM-OCR", 2.2, "https://ollama.com/library/glm-ocr"),
    ModelSpec("qwen3-vl:2b", "Qwen3-VL 2B", 1.9, "https://ollama.com/library/qwen3-vl"),
    ModelSpec("qwen3-vl:4b", "Qwen3-VL 4B", 3.3, "https://ollama.com/library/qwen3-vl"),
    ModelSpec(
        "granite3.2-vision:2b",
        "Granite 3.2 Vision 2B",
        2.4,
        "https://ollama.com/library/granite3.2-vision",
    ),
    ModelSpec(
        "ministral-3:3b",
        "Ministral 3 3B",
        3.0,
        "https://ollama.com/library/ministral-3",
    ),
    ModelSpec("gemma3:4b", "Gemma 3 4B", 3.3, "https://ollama.com/library/gemma3"),
    ModelSpec(
        "minicpm-v4.5:8b",
        "MiniCPM-V 4.5 8B",
        6.1,
        "https://ollama.com/library/minicpm-v4.5",
    ),
    ModelSpec(
        "deepseek-ocr:3b",
        "DeepSeek-OCR 3B",
        6.7,
        "https://ollama.com/library/deepseek-ocr",
    ),
    ModelSpec(
        "minicpm-v:8b", "MiniCPM-V 2.6 8B", 5.5, "https://ollama.com/library/minicpm-v"
    ),
    ModelSpec(
        "llama3.2-vision:11b",
        "Llama 3.2 Vision 11B",
        7.8,
        "https://ollama.com/library/llama3.2-vision",
    ),
    ModelSpec(
        "moondream:1.8b",
        "Moondream 2 1.8B",
        1.7,
        "https://ollama.com/library/moondream",
    ),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except Exception:
        Path(temp).unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cents(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def installed() -> dict[str, dict]:
    return {m.model: {"digest": m.digest, "size": m.size} for m in ollama.list().models}


def storage_path() -> Path:
    configured = os.environ.get("OLLAMA_MODELS")
    if configured:
        return Path(configured)
    service_path = Path("/usr/share/ollama/.ollama/models")
    return service_path if service_path.exists() else Path.home() / ".ollama/models"


def service_configuration() -> str:
    """Return the latest Ollama startup configuration without tail truncation."""
    try:
        return subprocess.check_output(
            [
                "journalctl",
                "-u",
                "ollama",
                "-b",
                "--no-pager",
                "-g",
                "OLLAMA_FLASH_ATTENTION",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def free_gb(path: Path) -> float:
    current = path
    while not current.exists():
        current = current.parent
    return shutil.disk_usage(current).free / 1024**3


def pull(spec: ModelSpec, allow: bool, reserve: float) -> float:
    if spec.model in installed():
        return 0.0
    if not allow:
        raise RuntimeError("missing model; use --allow-downloads")
    free = free_gb(storage_path())
    required = spec.size_gb + reserve
    if free < required:
        raise RuntimeError(f"{free:.1f} GB free, {required:.1f} GB required")
    start = time.monotonic()
    for chunk in ollama.pull(spec.model, stream=True):
        status = getattr(chunk, "status", "")
        if status:
            print(f"\r  pull: {status[:80]:80}", end="", flush=True)
    print()
    return time.monotonic() - start


def environment() -> dict:
    def command(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, text=True, stderr=subprocess.STDOUT
            ).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    journal = service_configuration()
    configured_storage = re.findall(r"OLLAMA_MODELS:([^ ]+)", journal)
    model_storage = os.environ.get("OLLAMA_MODELS") or (
        configured_storage[-1] if configured_storage else str(storage_path())
    )
    return {
        "ollama": command(["ollama", "--version"]),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "flash_attention": "true"
        if "OLLAMA_FLASH_ATTENTION:true" in journal
        else "false",
        "kv_cache": "f16",
        "model_storage": model_storage,
        "free_gb_start": round(free_gb(storage_path()), 2),
    }


def new_results(specs: list[ModelSpec], runs: int) -> dict:
    images = sorted(IMAGES.glob("*.jpg"))
    return {
        "schema": SCHEMA,
        "started_at": now(),
        "completed_at": None,
        "active_seconds": 0.0,
        "runs_per_image": runs,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "images": {
            p.name: {"sha256": sha256(p), "expected_cents": expected_cents(p)}
            for p in images
        },
        "catalog": [asdict(s) for s in specs],
        "environment": environment(),
        "installed_before": installed(),
        "models": {
            s.model: {
                "status": "pending",
                "pull_seconds": 0.0,
                "trials": [],
                "error": None,
            }
            for s in specs
        },
        "storage_actions": [],
    }


def validate(data: dict, specs: list[ModelSpec], runs: int) -> None:
    expected = new_results(specs, runs)
    for key in ("schema", "runs_per_image", "prompt_sha256", "images", "catalog"):
        if data.get(key) != expected[key]:
            raise RuntimeError(f"cannot resume: {key} changed; use --fresh")
    current = environment()
    for key in ("ollama", "gpu", "flash_attention", "kv_cache"):
        if data["environment"].get(key) != current.get(key):
            raise RuntimeError(f"cannot resume: environment {key} changed; use --fresh")


def summary(spec: ModelSpec, result: dict) -> dict:
    trials = result["trials"]
    latencies = [t["wall_seconds"] for t in trials if not t["error"]]
    correct = sum(t["correct"] for t in trials)
    stable = sum(
        all(t["correct"] for t in trials if t["image"] == image)
        for image in {t["image"] for t in trials}
    )
    return {
        "model": spec.model,
        "name": spec.name,
        "size_gb": spec.size_gb,
        "correct": correct,
        "total": len(trials),
        "accuracy": correct / len(trials) if trials else 0,
        "stable": stable,
        "median": statistics.median(latencies) if latencies else float("inf"),
        "mean": statistics.mean(latencies) if latencies else float("inf"),
        "cold": latencies[0] if latencies else float("inf"),
        "warm": statistics.median(latencies[1:])
        if len(latencies) > 1
        else float("inf"),
        "errors": sum(bool(t["error"]) for t in trials),
    }


def ranked(data: dict, specs: list[ModelSpec]) -> list[dict]:
    rows = [summary(s, data["models"][s.model]) for s in specs]
    return sorted(
        rows, key=lambda r: (-r["accuracy"], -r["stable"], r["warm"], r["size_gb"])
    )


def recommend(rows: list[dict]) -> dict | None:
    perfect = [
        r for r in rows if r["total"] and r["accuracy"] == 1 and r["stable"] == 3
    ]
    if not perfect:
        return None
    fastest = min(r["warm"] for r in perfect)
    eligible = [r for r in perfect if r["warm"] <= fastest * 1.5]
    return min(eligible, key=lambda r: (r["size_gb"], r["warm"]))


def write_markdown(data: dict, specs: list[ModelSpec]) -> None:
    rows, winner = ranked(data, specs), recommend(ranked(data, specs))
    lines = [
        "# Compact Vision Model Receipt Benchmark",
        "",
        "> Three receipts × three runs is a smoke test, not production accuracy.",
        "",
        f"Environment: {data['environment']['gpu']}; {data['environment']['ollama']}; Flash Attention `{data['environment']['flash_attention']}`; KV cache `{data['environment']['kv_cache']}`.",
        "",
        f"Total active benchmark time: **{data['active_seconds'] / 60:.1f} minutes**. Recommendation: **{winner['name'] if winner else 'none'}**.",
        "",
        "| # | Model | Exact | Stable | Warm median | Cold | Size | Errors |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        warm = f"{r['warm']:.2f}s" if r["warm"] != float("inf") else "—"
        cold = f"{r['cold']:.2f}s" if r["cold"] != float("inf") else "—"
        lines.append(
            f"| {i} | {r['name']} | {r['correct']}/{r['total']} | {r['stable']}/3 | {warm} | {cold} | {r['size_gb']:.1f} GB | {r['errors']} |"
        )
    lines += ["", "## Storage actions", ""] + (
        [f"- `{a['model']}` removed: {a['reason']}" for a in data["storage_actions"]]
        or ["- None."]
    )
    lines += [
        "",
        "## Method",
        "",
        f"Images were resized to at most {MAX_SIDE_PX}px, thinking was disabled, temperature was zero, and exact `Euro,Cent` responses were scored. Failed trials remain in the denominator.",
    ]
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pdf(data: dict, specs: list[ModelSpec]) -> None:
    rows, winner = ranked(data, specs), recommend(ranked(data, specs))
    width, height = landscape(A4)
    c = canvas.Canvas(str(RESULT_PDF), pagesize=(width, height), pageCompression=1)
    navy, teal, pale = (
        colors.HexColor("#152238"),
        colors.HexColor("#13A89E"),
        colors.HexColor("#EEF4F7"),
    )
    c.setFillColor(navy)
    c.rect(0, height - 82, width, 82, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(28, height - 38, "Local Vision OCR Benchmark")
    c.setFont("Helvetica", 8)
    c.drawString(
        28,
        height - 58,
        f"3 receipts × 3 runs · {data['environment']['gpu']} · Flash Attention {data['environment']['flash_attention']} · f16 KV",
    )
    cards = [
        ("MODELS", str(len(rows))),
        ("TRIALS", str(sum(r["total"] for r in rows))),
        ("TOTAL TIME", f"{data['active_seconds'] / 60:.1f} min"),
        ("RECOMMENDATION", winner["name"] if winner else "None"),
    ]
    x = 28
    for label, value in cards:
        w = 188 if label == "RECOMMENDATION" else 110
        c.setFillColor(pale)
        c.roundRect(x, height - 135, w, 40, 7, fill=1, stroke=0)
        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 7)
        c.drawString(x + 9, height - 108, label)
        c.setFont("Helvetica-Bold", 12 if label != "RECOMMENDATION" else 9)
        c.drawString(x + 9, height - 126, value[:30])
        x += w + 9
    y = height - 160
    columns = [
        (28, "#"),
        (48, "Model"),
        (270, "Exact"),
        (320, "Stable"),
        (375, "Warm"),
        (430, "Cold"),
        (485, "Size"),
        (535, "Err"),
    ]
    c.setFillColor(navy)
    c.rect(24, y - 14, width - 48, 22, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8)
    for x, label in columns:
        c.drawString(x, y - 7, label)
    y -= 24
    for i, r in enumerate(rows, 1):
        if winner and r["model"] == winner["model"]:
            c.setFillColor(colors.HexColor("#DDF6F2"))
            c.rect(24, y - 5, width - 48, 20, fill=1, stroke=0)
        elif i % 2 == 0:
            c.setFillColor(colors.HexColor("#F5F7F9"))
            c.rect(24, y - 5, width - 48, 20, fill=1, stroke=0)
        c.setFillColor(teal if winner and r["model"] == winner["model"] else navy)
        c.setFont("Helvetica-Bold" if i <= 3 else "Helvetica", 8)
        values = [
            str(i),
            r["name"],
            f"{r['correct']}/{r['total']}",
            f"{r['stable']}/3",
            f"{r['warm']:.2f}s" if r["warm"] != float("inf") else "—",
            f"{r['cold']:.2f}s" if r["cold"] != float("inf") else "—",
            f"{r['size_gb']:.1f}G",
            str(r["errors"]),
        ]
        for (x, _), value in zip(columns, values):
            c.drawString(x, y, value[:34])
        y -= 20
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(28, 54, "Key finding")
    c.setFont("Helvetica", 8)
    finding = (
        f"{winner['name']} is the recommended smallest near-fastest perfect model."
        if winner
        else "No model was perfect across all nine trials."
    )
    c.drawString(28, 40, finding)
    c.setFillColor(colors.HexColor("#607080"))
    c.setFont("Helvetica", 7)
    c.drawString(
        28,
        23,
        "Smoke test only: three annotated German receipts; errors count as incorrect. Warm median excludes the first model request.",
    )
    c.showPage()
    c.save()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[s.model for s in MODELS])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=10)
    parser.add_argument("--evict-poor", action="store_true")
    parser.add_argument("--poor-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not args.all and not args.model:
        parser.error("select --all or at least one --model")
    if args.all and args.model:
        parser.error("--all and --model are mutually exclusive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    specs = MODELS if args.all else [s for s in MODELS if s.model in args.model]
    if args.fresh:
        RESULT_JSON.unlink(missing_ok=True)
    data = (
        json.loads(RESULT_JSON.read_text())
        if RESULT_JSON.exists()
        else new_results(specs, args.runs)
    )
    if RESULT_JSON.exists():
        validate(data, specs, args.runs)
    session = time.monotonic()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    encoded = {p.name: encode_image(p) for p in sorted(IMAGES.glob("*.jpg"))}
    try:
        for spec in specs:
            result = data["models"][spec.model]
            print(f"\n=== {spec.name} ({spec.model}) ===", flush=True)
            try:
                result["pull_seconds"] += pull(
                    spec, args.allow_downloads, args.min_free_gb
                )
                info = installed().get(spec.model, {})
                result.update(
                    {
                        "digest": info.get("digest"),
                        "actual_size": info.get("size"),
                        "status": "running",
                    }
                )
                atomic_json(RESULT_JSON, data)
                done = {(t["image"], t["run"]) for t in result["trials"]}
                for image, meta in data["images"].items():
                    for run in range(1, args.runs + 1):
                        if (image, run) in done:
                            continue
                        start = time.monotonic()
                        trial = {
                            "image": image,
                            "run": run,
                            "expected_cents": meta["expected_cents"],
                            "response": "",
                            "parsed_cents": None,
                            "correct": False,
                            "wall_seconds": 0.0,
                            "api_total_seconds": None,
                            "load_seconds": None,
                            "prompt_eval_seconds": None,
                            "eval_seconds": None,
                            "error": None,
                            "at": now(),
                        }
                        try:
                            response = ollama.chat(
                                model=spec.model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": PROMPT,
                                        "images": [encoded[image]],
                                    }
                                ],
                                think=False,
                                options={
                                    "temperature": 0,
                                    "num_ctx": 4096,
                                    "num_predict": 64,
                                },
                                keep_alive="5m",
                            )
                            trial["response"] = response.message.content.strip()
                            trial["parsed_cents"] = parse_price(trial["response"])
                            trial["correct"] = (
                                trial["parsed_cents"] == meta["expected_cents"]
                            )
                            for field, target in (
                                ("total_duration", "api_total_seconds"),
                                ("load_duration", "load_seconds"),
                                ("prompt_eval_duration", "prompt_eval_seconds"),
                                ("eval_duration", "eval_seconds"),
                            ):
                                value = getattr(response, field, None)
                                trial[target] = (
                                    value / 1e9 if value is not None else None
                                )
                        except Exception as exc:
                            trial["error"] = f"{type(exc).__name__}: {exc}"[:500]
                        trial["wall_seconds"] = round(time.monotonic() - start, 4)
                        result["trials"].append(trial)
                        atomic_json(RESULT_JSON, data)
                        print(
                            f"  {image} run {run}: {'OK' if trial['correct'] else 'FAIL'} {trial['wall_seconds']:.2f}s {trial['response'][:50]!r}",
                            flush=True,
                        )
                result["status"] = "complete"
                ollama.generate(model=spec.model, prompt="", keep_alive=0)
                atomic_json(RESULT_JSON, data)
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                atomic_json(RESULT_JSON, data)
                print(f"  MODEL FAILED: {exc}", flush=True)
        if args.evict_poor:
            before = set(data["installed_before"])
            before_digests = {
                value.get("digest") for value in data["installed_before"].values()
            }
            winner = recommend(ranked(data, specs))
            keep = winner["model"] if winner else None
            for spec in specs:
                row = summary(spec, data["models"][spec.model])
                digest = data["models"][spec.model].get("digest")
                if (
                    spec.model not in before
                    and digest not in before_digests
                    and spec.model != keep
                    and row["total"] == args.runs * len(data["images"])
                    and row["accuracy"] < args.poor_threshold
                ):
                    ollama.delete(spec.model)
                    data["storage_actions"].append(
                        {
                            "model": spec.model,
                            "at": now(),
                            "reason": f"accuracy {row['accuracy']:.0%} below {args.poor_threshold:.0%}",
                        }
                    )
                    atomic_json(RESULT_JSON, data)
        data["active_seconds"] += time.monotonic() - session
        data["completed_at"] = now()
        data["environment"]["free_gb_end"] = round(free_gb(storage_path()), 2)
        atomic_json(RESULT_JSON, data)
        write_markdown(data, specs)
        write_pdf(data, specs)
        return 0
    except KeyboardInterrupt:
        data["active_seconds"] += time.monotonic() - session
        atomic_json(RESULT_JSON, data)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
