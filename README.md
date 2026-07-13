# Sales Slip Scanner NG

> **Scan a receipt. Get the total. Done.**  
> No cloud. No API key. No fees. Just a local vision model doing the work.

From the 2024-version - not right anymore - we run a local vision-model!
![](meme.png)

---

**Author: Marcel Petrick <mail@marcelpetrick.it>**

**Note: project is generated with AI.**

**License: GPLv3 or later. See `LICENSE`.**

---

## The story

Two years ago I built [a quick proof-of-concept](https://github.com/marcelpetrick/codingWithGPT/tree/master/SalesSlipScanner) that extracted totals from German grocery receipts by sending them to the OpenAI API. It worked — and cost me exactly **15 cents** in API credits for the test run. I wrote it in about an hour and moved on.

Fast forward to 2026: local vision models have caught up. In a fresh smoke test of **15 models** and 135 total trials, **Qwen 3.5 4B** (3.4 GB) extracted every receipt total correctly in all three repetitions — completely offline, with no per-inference fee and no receipt sent to a remote service.

This repository is the next generation: the same idea, fully local, tested, and backed by a repeatable local quality pipeline.

---

## What it does

Drop receipt images into `input/`, run the script, and each file is renamed to include its detected total:

```
receipt.jpg  →  receipt_7949.jpg   (79,49 €)
```

A summary with the grand total is printed at the end. Successful operations are recorded in `input/.sales-slip-scanner.json`, and an existing destination is never overwritten.

---

## Live output

```
❯ python salesSlipScanner.py
No unprocessed image files found in: .../input/

❯ cp test_images/slip*.jpg input/

❯ python salesSlipScanner.py
Found 3 file(s)  [model: qwen3.5:4b]

  slip0.jpg … OK  →  slip0_7949.jpg  (79,49 €)
  slip1.jpg … OK  →  slip1_2841.jpg  (28,41 €)
  slip2.jpg … OK  →  slip2_1093.jpg  (10,93 €)

──────────────────────────────────────────────────
  Processed : 3
  Skipped   : 0
  Total     : 118,83 €
──────────────────────────────────────────────────
```

---

## Benchmark

Fifteen vision models were tested on three annotated German grocery/gas receipts, with three runs per image. Full trial data and compatibility errors are in [`results.md`](localVisionModelTest/results.md), [`results.json`](localVisionModelTest/results.json), and the [one-page PDF](localVisionModelTest/results.pdf).

| # | Model | Exact | Stable receipts | Warm median |
|---:|---|---:|---:|---:|
| 1 | **Qwen 3.5 4B** ← default | **9/9** | **3/3** | 0.46 s |
| 2 | Ministral 3 3B | 6/9 | 2/3 | **0.32 s** |
| 3 | Qwen 3.5 2B | 6/9 | 2/3 | 0.40 s |
| 4 | Gemma 3 4B | 6/9 | 2/3 | 0.53 s |
| 5 | MiniCPM-V 2.6 8B | 6/9 | 2/3 | 0.68 s |
| 6 | MiniCPM-V 4.5 8B | 6/9 | 2/3 | 0.70 s |
| 7–15 | MiniCPM-V 4.6, Qwen 3.5 0.8B, Moondream, GLM-OCR, Qwen3-VL 2B/4B, Granite, DeepSeek-OCR, Llama Vision | 0–3/9 | 0–1/3 | see report |

The measured run used Ollama 0.31.2, its default Flash Attention setting (`false`), f16 KV cache, and an NVIDIA RTX A2000 8 GB Laptop GPU. Three examples are a compatibility smoke test, not a production accuracy estimate.

---

## Prerequisites

- Python 3.14.6
- [ollama](https://ollama.com) running locally
- The default model pulled: `ollama pull qwen3.5:4b`

---

## Setup

```bash
git clone https://github.com/marcelpetrick/sales-slip-scanner-ng.git
cd sales-slip-scanner-ng
pip install -r requirements.txt
ollama pull qwen3.5:4b
```

---

## Usage

```bash
# Default model (qwen3.5:4b)
python salesSlipScanner.py

# Override model
python salesSlipScanner.py --model llama3.2-vision:11b
```

If the requested model is not installed, the script prints the exact
`ollama pull` command needed and exits with a nonzero status. Ambiguous model
responses and per-file failures also produce a nonzero status.

Run selected benchmark models explicitly; missing models are never downloaded
unless `--allow-downloads` is supplied, and downloads require estimated model
space plus a 10 GB reserve on the configured Ollama storage filesystem:

```bash
python localVisionModelTest/benchmark.py --model qwen3.5:4b
python localVisionModelTest/benchmark.py --all --runs 3 --allow-downloads --fresh
```

---

## Project layout

```
input/                        ← drop receipt images here
test_images/                  ← annotated reference images
salesSlipScanner.py           ← main script
receipt_ocr.py                 ← shared image, prompt, parsing, and Ollama logic
localVisionModelTest/
  benchmark.py                ← explicit, capacity-checked smoke-test harness
  modelsToTest.md             ← full candidate list with VRAM / disk notes
  results.json / .md / .pdf  ← trial data, analysis, and one-page results
documents/
  agents.md                   ← working agreement (commit style, pipeline gate)
tests/
  test_sales_slip_scanner.py  ← scanner and shared OCR tests
  test_benchmark.py           ← offline benchmark scoring and safety tests
localPipeline.sh              ← Python/dependencies/lint/tests/coverage with summary
pyproject.toml                ← Python constraint and exact dependency pins
requirements.txt              ← compatibility installer for project + dev dependencies
```

---

## Development

Run the full quality pipeline before committing:

```bash
./localPipeline.sh
```

Stages: **Python 3.14.6 check → isolated dependency install → full-tree Ruff lint → pytest with coverage**. Every run ends with a stage-by-stage PASS/FAIL/SKIP summary, including test count and total coverage.
