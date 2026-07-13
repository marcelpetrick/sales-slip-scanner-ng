# Receipt Vision-Model Benchmark Plan

## Status at pause

Planning and market research are complete. The benchmark harness has not been
rewritten, the multi-model sweep has not started, no models have been pulled,
and no models have been deleted.

One direct smoke call was made with the already-installed `qwen3-vl:4b` model
and `test_images/slip2_1093.jpg`. It returned the correct `10,93`, took 60.62
seconds including cold loading, and `ollama ps` reported `100% GPU`; the model
was explicitly stopped after the test.

The session was paused because `nvidia-smi` failed with an NVML
driver/library-version mismatch. Restart the machine before implementing or
running the benchmark so GPU identification, memory reporting, and timing are
reliable.

## Machine inventory before restart

- Python: 3.14.6
- Ollama server: 0.24.0
- Ollama Python client: 0.6.2
- Expected GPU: NVIDIA RTX A2000 8 GB Laptop GPU
- `nvidia-smi`: failed to initialize NVML (`610.43` library mismatch)
- Root filesystem: 98 GB total, about 3.1 GB free, 97% used
- Home filesystem: 807 GB total, about 168 GB free
- Ollama model store: `/home/mpetrick/.ollama/models` on the home filesystem
- Model-store usage: about 9.5 GB, with blob sharing between tags

Installed Ollama models at the pause:

| Model | Reported size | Role |
|---|---:|---|
| `qwen3-vl:4b` | 3.3 GB | Current scanner default and benchmark baseline |
| `llama3.2-vision:11b` | 7.8 GB | Larger accuracy baseline |
| `minicpm-v:latest` | 5.5 GB | Older OCR-oriented baseline |
| `moondream:latest` | 1.7 GB | Older speed baseline |
| `qwen3.5:4b-ctx32k` | 3.4 GB | Locally customized Qwen 3.5 tag |
| `qwen3.5:4b-ctx54k` | 3.4 GB | Locally customized Qwen 3.5 tag |

The root filesystem must never be used to decide whether a model download is
safe. Capacity checks must target `OLLAMA_MODELS` when set, otherwise the
actual filesystem containing `~/.ollama/models`.

## Market review

The old benchmark is stale: it predates several compact multimodal and
OCR-specific model families now available through Ollama. The refreshed set
should favor official Ollama library entries with image input, a footprint of
roughly 8 GB or less, known OCR/document strengths, or enough adoption to make
them useful baselines.

Official sources reviewed:

- MiniCPM-V 4.6: <https://ollama.com/library/minicpm-v4.6>
- Qwen 3.5: <https://ollama.com/library/qwen3.5>
- GLM-OCR: <https://ollama.com/library/glm-ocr>
- Qwen3-VL: <https://ollama.com/library/qwen3-vl>
- Ministral 3: <https://ollama.com/library/ministral-3>
- Granite 3.2 Vision: <https://ollama.com/library/granite3.2-vision>
- Gemma 3: <https://ollama.com/library/gemma3>
- MiniCPM-V 4.5: <https://ollama.com/library/minicpm-v4.5>
- DeepSeek-OCR: <https://ollama.com/library/deepseek-ocr>
- MiniCPM-V 2.6: <https://ollama.com/library/minicpm-v>
- Llama 3.2 Vision: <https://ollama.com/library/llama3.2-vision>
- Moondream 2: <https://ollama.com/library/moondream>

## Proposed benchmark candidates

The final run should include at least the following 15 candidates. This gives
11 recent or currently relevant candidates and four established controls.

| # | Model | Approx. size | Why include |
|---:|---|---:|---|
| 1 | `minicpm-v4.6:1b` | 1.6 GB | New, ultra-compact multimodal model with OCRBench claims |
| 2 | `qwen3.5:0.8b` | 1.0 GB | New smallest Qwen multimodal model |
| 3 | `qwen3.5:2b` | 2.7 GB | New compact Qwen multimodal tier |
| 4 | `qwen3.5:4b` | 3.4 GB | New likely quality/size contender |
| 5 | `glm-ocr:latest` | 2.2 GB | New 0.9B document/OCR specialist |
| 6 | `qwen3-vl:2b` | 1.9 GB | Smaller version of the current winning family |
| 7 | `qwen3-vl:4b` | 3.3 GB | Current scanner default |
| 8 | `granite3.2-vision:2b` | 2.4 GB | Compact document-understanding specialist |
| 9 | `ministral-3:3b` | 3.0 GB | Recent edge-focused multilingual vision model |
| 10 | `gemma3:4b` | 3.3 GB | Well-known multilingual compact baseline |
| 11 | `minicpm-v4.5:8b` | 6.1 GB | Recent high-quality MiniCPM control |
| 12 | `deepseek-ocr:3b` | 6.7 GB | Recent OCR-specific architecture |
| 13 | `minicpm-v:8b` | 5.5 GB | Previously tested OCR-oriented baseline |
| 14 | `llama3.2-vision:11b` | 7.8 GB | Previously accurate upper-size baseline |
| 15 | `moondream:1.8b` | 1.7 GB | Previously fastest low-size baseline |

Optional additions, if post-restart GPU memory and elapsed time permit:

- `qwen3-vl:8b` (6.1 GB)
- `ministral-3:8b` (6.0 GB)

Do not include `gemma3n` because its current Ollama tags report text-only
input. Do not include cloud-only models because the scanner is explicitly
local-first. Older LLaVA/BakLLaVA variants should be removed from the active
candidate list because prior results were poor and newer compact families now
cover the same comparison points.

## Harness implementation requirements

Rewrite `localVisionModelTest/benchmark.py` as a reproducible, resumable
benchmark rather than extending the hard-coded archived-result script.

The harness must:

1. Run every selected model against all three annotated receipt images.
2. Run each image exactly three times per model, producing nine measured
   trials per model.
3. Use the same image normalization, prompt, strict amount parser, and Ollama
   request behavior as the production scanner.
4. Disable model thinking where supported and use deterministic inference
   settings (`temperature=0`).
5. Record wall-clock latency, Ollama total duration, load duration, prompt
   evaluation duration, generation duration, raw response, parsed cents,
   expected cents, correctness, error text, model digest, and actual size.
6. Preserve cold-load timing instead of hiding the first request, while also
   reporting warm median latency separately.
7. Record total active benchmark duration, including downloads, model loads,
   all inference trials, report generation, and safe cleanup.
8. Checkpoint an atomic JSON results file after every trial.
9. Resume completed trials after interruption without duplicating them.
10. Validate the prompt, image hashes, run count, and candidate catalog before
    resuming an existing result file.
11. Require explicit model selection (`--model` or `--all`).
12. Require `--allow-downloads` before pulling a missing model.
13. Check free space on the real Ollama model-store filesystem against the
    estimated model size plus a configurable reserve (default at least 10 GB
    for this sweep).
14. Never delete a model that was installed before this benchmark session.
15. Optionally remove only models downloaded by this run that fall below a
    documented accuracy threshold, after their results are safely persisted.
16. Unload each model from GPU memory after its trials.
17. Continue to the next model after a model-specific pull or inference
    failure, while counting failed trials as incorrect.
18. Handle `SIGINT`/`SIGTERM` without losing completed measurements.
19. Include offline unit tests for scoring, resume validation, capacity
    checks, candidate-only eviction, ranking, and report generation.

Suggested command after implementation:

```bash
./localPipeline.sh
python localVisionModelTest/benchmark.py \
  --all \
  --runs 3 \
  --allow-downloads \
  --evict-poor \
  --poor-threshold 0.50 \
  --min-free-gb 10
```

## Measurement and ranking policy

Accuracy is exact amount matches divided by all nine scheduled trials; errors
remain in the denominator. Also report receipt-level reliability: a receipt is
stable only when all three runs return the correct amount.

For each model report:

- exact trial accuracy (`correct / 9`)
- stable receipts (`0–3`)
- mean, median, minimum, and maximum wall latency
- cold first-request latency
- warm median latency excluding the first model request
- latency standard deviation
- download time and measured installed size
- error count and distinct incorrect responses

Rank primarily by exact accuracy, then stable receipts, warm median latency,
and installed size. The recommended scanner model must be perfect across all
nine trials; if several qualify, choose the smallest model within a reasonable
latency margin, not simply the fastest or largest.

The three-image dataset is a smoke test, not a production accuracy estimate.
Every report must state this prominently.

## Storage and deletion policy

- Preserve every model that was present before the run.
- Download sequentially; do not prefetch models in background threads.
- Before each pull, re-check the real model-store filesystem.
- Keep the eventual scanner recommendation.
- A newly downloaded model may be deleted only after all nine trials and a
  durable result checkpoint, and only when its exact accuracy is below 50%.
- Log each deletion with model ID, digest, reclaimed logical size, reason, and
  timestamp in the JSON and Markdown reports.
- Never run broad cache cleanup or delete unrelated Ollama models.

There was enough free space on `/home` to retain all proposed candidates at
the pause, so deletion is an optimization rather than a prerequisite.

## Reports to produce

1. `localVisionModelTest/results.json`: machine-readable trial-level data and
   full provenance.
2. `localVisionModelTest/results.md`: market shortlist, methodology, complete
   measured table, add/remove decisions, total elapsed time, storage actions,
   limitations, and final scanner recommendation.
3. `localVisionModelTest/results.pdf`: a polished, single-page landscape A4
   report with:
   - a strong title and environment/date subtitle
   - summary cards for model count, trial count, total benchmark time, and
     recommended model
   - a readable leaderboard containing accuracy, stable receipts, latency,
     and size
   - highlighted winner and fastest-perfect candidate
   - concise key findings, methodology, and the three-image limitation

Assert programmatically that the PDF has exactly one page.

## Post-restart checklist

Before changing code or downloading models:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
ollama --version
ollama ps
df -h / /home
findmnt -T "$HOME/.ollama/models"
ollama list
```

Then run one existing-model probe and confirm that Ollama reports GPU
execution. If GPU reporting still fails or Ollama falls back to CPU, stop and
fix the NVIDIA stack before starting the sweep; do not publish CPU timings as
GPU benchmark results.

## Completion sequence

1. Verify NVIDIA and storage after restart.
2. Implement the harness and tests.
3. Pass `./localPipeline.sh`.
4. Run the resumable benchmark to completion.
5. Review persisted trials and storage actions.
6. Generate and validate the one-page PDF.
7. Update the scanner default only if the measured recommendation clearly
   beats or materially reduces the footprint of `qwen3-vl:4b`.
8. Update README/model documentation with measured, qualified claims.
9. Run the final pipeline and commit implementation, measured data, report,
   and documentation in focused commits.
