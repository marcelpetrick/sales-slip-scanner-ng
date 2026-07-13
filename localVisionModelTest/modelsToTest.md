# Compact Vision Model Market Shortlist

This is the July 2026 shortlist used by the reproducible receipt benchmark. It favors current, well-known local vision and OCR families that fit or can be attempted on an 8 GB NVIDIA GPU; model sizes are the tested Ollama artifacts rather than parameter-memory estimates.

| Model | Size | Why included | Measured result | Disposition |
|---|---:|---|---:|---|
| [Qwen 3.5 4B](https://ollama.com/library/qwen3.5) | 3.4 GB | Recent compact native vision model | 9/9 | Keep; scanner default |
| [Ministral 3 3B](https://ollama.com/library/ministral-3) | 3.0 GB | Recent multilingual edge model | 6/9 | Keep for comparison |
| [Qwen 3.5 2B](https://ollama.com/library/qwen3.5) | 2.7 GB | Smaller current Qwen tier | 6/9 | Keep for comparison |
| [Gemma 3 4B](https://ollama.com/library/gemma3) | 3.3 GB | Well-known compact multimodal baseline | 6/9 | Keep for comparison |
| [MiniCPM-V 2.6 8B](https://ollama.com/library/minicpm-v) | 5.5 GB | Established OCR-oriented vision model | 6/9 | Keep for comparison |
| [MiniCPM-V 4.5 8B](https://ollama.com/library/minicpm-v4.5) | 6.1 GB | Newer MiniCPM vision generation | 6/9 | Keep for comparison |
| [MiniCPM-V 4.6 1B](https://ollama.com/library/minicpm-v4.6) | 1.6 GB | Recent ultra-compact OCR candidate | 3/9 | Remove; below 50% |
| [Qwen 3.5 0.8B](https://ollama.com/library/qwen3.5) | 1.0 GB | Smallest current Qwen vision tier | 3/9 | Remove; below 50% |
| [GLM-OCR](https://ollama.com/library/glm-ocr) | 2.2 GB | Recent specialist document OCR model | 0/9 | Remove; output incompatible |
| [Qwen3-VL 2B](https://ollama.com/library/qwen3-vl) | 1.9 GB | Well-known small vision-language model | 0/9 | Remove; no final content |
| [Qwen3-VL 4B](https://ollama.com/library/qwen3-vl) | 3.3 GB | Previous project default and baseline | 0/9 | Remove; no final content |
| [Granite 3.2 Vision 2B](https://ollama.com/library/granite3.2-vision) | 2.4 GB | IBM document-vision baseline | 0/9 | Remove; exceeds 4096 context |
| [DeepSeek-OCR 3B](https://ollama.com/library/deepseek-ocr) | 6.7 GB | Recent OCR specialist | 0/9 | Remove; Ollama runtime error |
| [Llama 3.2 Vision 11B](https://ollama.com/library/llama3.2-vision) | 7.8 GB | Widely used larger vision baseline | 0/9 | Remove; architecture load error |
| [Moondream 2 1.8B](https://ollama.com/library/moondream) | 1.7 GB | Well-known tiny and fast baseline | 0/9 | Remove; format/read errors |

## Selection decision

Qwen 3.5 4B is the only model that returned all three exact totals in all three repetitions. It is also compact enough for the RTX A2000 and its 0.46-second warm median is close to the fastest tested candidates, so larger or less accurate models do not offer a useful tradeoff for this scanner.

## Models removed from the old queue

The former list contained older LLaVA, BakLLaVA, Qwen2-VL, Qwen2.5-VL, SmolVLM2, German-OCR-3, MiniCPM-o, and olmOCR community tags. They were removed from this sweep because the refreshed official Ollama shortlist already spans smaller, newer general vision models and current OCR specialists while keeping the run bounded and reproducible.

The full methodology, trial responses, timings, errors, digests, and deletion log are in [`results.md`](results.md) and [`results.json`](results.json). The three-receipt set is a smoke test only and must not be presented as production accuracy.
