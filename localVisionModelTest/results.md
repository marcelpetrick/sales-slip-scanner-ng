# Compact Vision Model Receipt Benchmark

> Three receipts × three runs is a smoke test, not production accuracy.

Environment: NVIDIA RTX A2000 8GB Laptop GPU, 8192 MiB, 610.43.03; ollama version is 0.31.2; Flash Attention `false`; KV cache `f16`.

Total active benchmark time: **39.0 minutes**. Recommendation: **Qwen 3.5 4B**.

| # | Model | Exact | Stable | Warm median | Cold | Size | Errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Qwen 3.5 4B | 9/9 | 3/3 | 0.46s | 6.23s | 3.4 GB | 0 |
| 2 | Ministral 3 3B | 6/9 | 2/3 | 0.32s | 5.82s | 3.0 GB | 0 |
| 3 | Qwen 3.5 2B | 6/9 | 2/3 | 0.40s | 5.18s | 2.7 GB | 0 |
| 4 | Gemma 3 4B | 6/9 | 2/3 | 0.53s | 8.73s | 3.3 GB | 0 |
| 5 | MiniCPM-V 2.6 8B | 6/9 | 2/3 | 0.68s | 31.22s | 5.5 GB | 0 |
| 6 | MiniCPM-V 4.5 8B | 6/9 | 2/3 | 0.70s | 6.53s | 6.1 GB | 0 |
| 7 | Qwen 3.5 0.8B | 3/9 | 1/3 | 0.36s | 13.61s | 1.0 GB | 0 |
| 8 | MiniCPM-V 4.6 1B | 3/9 | 1/3 | 0.88s | 52.65s | 1.6 GB | 0 |
| 9 | Moondream 2 1.8B | 0/9 | 0/3 | 0.14s | 2.80s | 1.7 GB | 0 |
| 10 | GLM-OCR | 0/9 | 0/3 | 0.75s | 29.90s | 2.2 GB | 0 |
| 11 | Qwen3-VL 2B | 0/9 | 0/3 | 0.95s | 5.22s | 1.9 GB | 0 |
| 12 | Qwen3-VL 4B | 0/9 | 0/3 | 1.69s | 7.02s | 3.3 GB | 0 |
| 13 | Granite 3.2 Vision 2B | 0/9 | 0/3 | — | — | 2.4 GB | 9 |
| 14 | DeepSeek-OCR 3B | 0/9 | 0/3 | — | — | 6.7 GB | 9 |
| 15 | Llama 3.2 Vision 11B | 0/9 | 0/3 | — | — | 7.8 GB | 9 |

## Storage actions

- `minicpm-v4.6:1b` removed (1.53 GiB, digest `e95583acac77`): accuracy 33% below 50%
- `qwen3.5:0.8b` removed (0.96 GiB, digest `f3817196d142`): accuracy 33% below 50%
- `glm-ocr:latest` removed (2.07 GiB, digest `6effedd0dc8a`): accuracy 0% below 50%
- `qwen3-vl:2b` removed (1.76 GiB, digest `0635d9d857d4`): accuracy 0% below 50%
- `qwen3-vl:4b` removed (3.07 GiB, digest `1343d82ebee3`): accuracy 0% below 50%
- `granite3.2-vision:2b` removed (2.27 GiB, digest `3be41a661804`): accuracy 0% below 50%
- `deepseek-ocr:3b` removed (6.23 GiB, digest `0e7b018b8a22`): accuracy 0% below 50%
- `llama3.2-vision:11b` removed (7.28 GiB, digest `6f2f9757ae97`): accuracy 0% below 50%
- `moondream:1.8b` removed (1.62 GiB, digest `55fc3abd3867`): accuracy 0% below 50%

## Compatibility findings

- Granite 3.2 Vision exceeded the project's 4096-token context with these images.
- DeepSeek-OCR failed in Ollama with `unexpected EOF`; Llama 3.2 Vision failed to load its `mllama` architecture.
- Qwen3-VL returned no final content with thinking disabled; GLM-OCR returned full OCR text instead of one parseable amount.

## Candidate sources

- [MiniCPM-V 4.6 1B](https://ollama.com/library/minicpm-v4.6) — `minicpm-v4.6:1b`
- [Qwen 3.5 0.8B](https://ollama.com/library/qwen3.5) — `qwen3.5:0.8b`
- [Qwen 3.5 2B](https://ollama.com/library/qwen3.5) — `qwen3.5:2b`
- [Qwen 3.5 4B](https://ollama.com/library/qwen3.5) — `qwen3.5:4b`
- [GLM-OCR](https://ollama.com/library/glm-ocr) — `glm-ocr:latest`
- [Qwen3-VL 2B](https://ollama.com/library/qwen3-vl) — `qwen3-vl:2b`
- [Qwen3-VL 4B](https://ollama.com/library/qwen3-vl) — `qwen3-vl:4b`
- [Granite 3.2 Vision 2B](https://ollama.com/library/granite3.2-vision) — `granite3.2-vision:2b`
- [Ministral 3 3B](https://ollama.com/library/ministral-3) — `ministral-3:3b`
- [Gemma 3 4B](https://ollama.com/library/gemma3) — `gemma3:4b`
- [MiniCPM-V 4.5 8B](https://ollama.com/library/minicpm-v4.5) — `minicpm-v4.5:8b`
- [DeepSeek-OCR 3B](https://ollama.com/library/deepseek-ocr) — `deepseek-ocr:3b`
- [MiniCPM-V 2.6 8B](https://ollama.com/library/minicpm-v) — `minicpm-v:8b`
- [Llama 3.2 Vision 11B](https://ollama.com/library/llama3.2-vision) — `llama3.2-vision:11b`
- [Moondream 2 1.8B](https://ollama.com/library/moondream) — `moondream:1.8b`

## Method

Images were resized to at most 1500px, thinking was disabled, temperature was zero, and exact `Euro,Cent` responses were scored. Failed trials remain in the denominator.

## Limitation

This is a deterministic compatibility smoke test on three annotated German receipts, not a production accuracy estimate; a larger and more varied held-out set is required before deployment decisions.
