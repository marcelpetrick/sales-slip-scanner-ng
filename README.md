# Sales Slip Scanner NG

> Drop receipts into a hot folder and get one cumulative Markdown report.

The application runs entirely against a local Ollama server. It sends no receipt data to a cloud service, requires no API key, and uses `qwen3.5:4b`, the winner of the repository's Ollama 0.31.2 benchmark.

**Author:** Marcel Petrick <mail@marcelpetrick.it><br>
**License:** GPLv3 or later; see `LICENSE`.

## Project history

- **2024:** the first proof of concept sent receipt images to a remote vision API and renamed files with the detected amount.
- **Early 2026:** the application moved fully to local Ollama vision models, gained strict price parsing, safe state, and an offline quality pipeline.
- **July 2026:** a fresh Ollama 0.31.2 sweep benchmarked 15 compact models over 135 trials. Qwen 3.5 4B was the only 9/9 model.
- **Current:** the rename workflow is gone. The application is a non-destructive hot-folder processor that produces a cumulative Markdown expense report.

## Features

- Fully local receipt processing through Ollama; no cloud runtime or API key.
- Qwen 3.5 4B selected from measured Ollama 0.31.2 results.
- Sequential processing with one model warm-up and configurable keep-alive.
- Original images are never renamed, moved, modified, or deleted.
- Content-based SHA-256 deduplication, independent of filenames.
- Failed receipts remain retryable on the next run.
- Atomic state and Markdown report updates after each receipt.
- Per-receipt amounts and a cumulative grand total.
- Exact dependency pins, offline tests, linting, and coverage enforcement.

## Workflow

1. Copy JPEG, PNG, GIF, or WebP receipts into `hot_folder/`.
2. Run `./scanHotFolder.sh`.
3. Read `hot_folder/receipt-report.md`.

The scanner processes new receipt content sequentially and leaves every source image unchanged. It records successful SHA-256 fingerprints in `hot_folder/.sales-slip-scanner.json`, so unchanged files are never counted twice even when copied under another name.

Failed images remain eligible for the next run. State and Markdown output are replaced atomically after every attempted receipt, and the Ollama model is warmed once and retained for ten minutes by default.

## Build and verify

Requirements are Python 3.14.6 and a running local [Ollama](https://ollama.com) server.

```bash
git clone https://github.com/marcelpetrick/sales-slip-scanner-ng.git
cd sales-slip-scanner-ng
ollama pull qwen3.5:4b
./localPipeline.sh
```

`pyproject.toml` is the sole dependency and build definition. The pipeline creates or reuses `.venv`, installs the project with its pinned development dependencies, and runs every quality gate; no separate `requirements.txt` is needed. As its final local step, it processes the existing `test_images/slip2_1093.jpg` fixture through the real Ollama server and verifies the expected `10,93 €` result.

GitHub Actions cannot access the local Ollama server or GPU, so that one live stage is reported as `SKIP` when `GITHUB_ACTIONS=true`; linting and offline tests still run normally. A successful local run ends with:

```text
========== Local Pipeline Summary ==========
Python           : PASS 3.14.6
Virtualenv       : PASS .venv is available
Dependencies     : PASS pyproject.toml installed
Ruff             : PASS 0 violations
Tests+Coverage   : PASS 38 passed in 0.31s; 97% coverage
Ollama smoke     : PASS slip2_1093.jpg -> 10,93 €
============================================
```

## Run the hot folder

```bash
mkdir -p hot_folder
cp ~/receipts/*.jpg hot_folder/
./scanHotFolder.sh
```

Example output:

```text
Hot folder : /repo/sales-slip-scanner-ng/hot_folder
Report     : /repo/sales-slip-scanner-ng/hot_folder/receipt-report.md
Model      : qwen3.5:4b
Found 3 new receipt(s) [model: qwen3.5:4b]
Warming model and keeping it loaded for 10m …
  fuel.jpg … OK (79,49 €)
  groceries-a.jpg … OK (28,41 €)
  groceries-b.jpg … OK (10,93 €)

========== Receipt Summary ==========
Processed this run : 3
Failed this run    : 0
All receipts      : 3
Grand total       : 118,83 €
Report            : .../hot_folder/receipt-report.md
=====================================
```

The generated report resembles:

```markdown
# Receipt Report

| Receipt | Amount | Processed | SHA-256 |
|---|---:|---|---|
| fuel.jpg | 79,49 € | ... | `abc123...` |
| groceries-a.jpg | 28,41 € | ... | `def456...` |
| groceries-b.jpg | 10,93 € | ... | `789abc...` |

## Grand total: 118,83 €
```

## Configuration

The shell wrapper validates Python, Ollama connectivity, and exact model availability before starting. Paths and warm duration can be configured either with arguments or environment variables:

```bash
./scanHotFolder.sh --hot-folder /data/receipts --report /data/receipt-total.md
./scanHotFolder.sh --keep-alive 30m

RECEIPT_HOT_FOLDER=/data/receipts \
RECEIPT_REPORT=/data/receipt-total.md \
RECEIPT_KEEP_ALIVE=30m \
./scanHotFolder.sh
```

`RECEIPT_MODEL` or `--model` can select another locally installed vision model. The default remains `qwen3.5:4b`; missing models are never downloaded implicitly.

The Python entry point exposes the same options:

```bash
.venv/bin/python salesSlipScanner.py --help
```

## Ollama 0.31.2 benchmark

Only the fresh Ollama 0.31.2 sweep is used for current model selection. Fifteen models were tested on three annotated receipts with three runs per image, producing 135 trials.

| # | Model | Exact | Stable receipts | Warm median |
|---:|---|---:|---:|---:|
| 1 | **Qwen 3.5 4B** ← default | **9/9** | **3/3** | 0.46 s |
| 2 | Ministral 3 3B | 6/9 | 2/3 | 0.32 s |
| 3 | Qwen 3.5 2B | 6/9 | 2/3 | 0.40 s |
| 4 | Gemma 3 4B | 6/9 | 2/3 | 0.53 s |
| 5 | MiniCPM-V 2.6 8B | 6/9 | 2/3 | 0.68 s |
| 6 | MiniCPM-V 4.5 8B | 6/9 | 2/3 | 0.70 s |

The run used Flash Attention disabled, f16 KV cache, and an NVIDIA RTX A2000 8 GB Laptop GPU. Three source images are a compatibility smoke test, not a production accuracy estimate; full data remains in [`results.md`](localVisionModelTest/results.md), [`results.json`](localVisionModelTest/results.json), and the [one-page PDF](localVisionModelTest/results.pdf).

## Project layout

```text
hot_folder/                   runtime receipts, state, and Markdown report
scanHotFolder.sh              robust operator entry point
salesSlipScanner.py           hot-folder orchestration and reporting
receipt_ocr.py                local image preparation, prompt, and parsing
test_images/                  annotated smoke-test receipts
localVisionModelTest/         benchmark harness and Ollama 0.31.2 results
tests/                        offline scanner and benchmark tests
localPipeline.sh              dependency, lint, test, and coverage gate
```

## Development

```bash
./localPipeline.sh
```

The pipeline verifies Python 3.14.6, exact dependency pins, repository-wide Ruff checks, pytest, coverage, and one real local Ollama extraction. It always finishes with the stage-by-stage summary shown above.
