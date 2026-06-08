# Vision Models to Test — Sales Slip Benchmark

GPU target: NVIDIA RTX A2000 8GB Laptop GPU (~7.8 GB free VRAM).
Disk: /dev/nvme0n1p1 98 GB (space-constrained — see notes below).

## Tier 1 — Best Accuracy for Sales Slips (fit in 8 GB VRAM)

| # | Model | Params | VRAM (Q4) | OCR Quality | Ollama command | Status |
|---|-------|--------|-----------|-------------|----------------|--------|
| 1 | MiniCPM-V 2.6 | 8B | ~6 GB | Excellent (5/5 invoice test) | `ollama pull minicpm-v` | queued |
| 2 | Qwen2-VL 7B | 7B | ~6 GB | Excellent — best multilingual OCR | `ollama pull qwen2-vl:7b` | queued |
| 3 | Llama 3.2 Vision 11B | 11B | ~8 GB | Excellent — best overall OCR | `ollama pull llama3.2-vision:11b` | queued |
| 4 | Qwen3-VL 4B | 4B | ~6 GB | Very good (94.2% DocVQA) | `ollama pull qwen3-vl:4b` | queued |
| 5 | olmOCR-2 7B | 7B | ~8.8 GB | SOTA documents (82.4 olmOCR-Bench) | `ollama run hf.co/richardyoung/olmOCR-2-7B-1025-GGUF` | queued (HF) |
| 6 | Qwen2.5-VL 7B | 7B | ~6 GB | Very good for invoices | `ollama pull qwen2.5-vl:7b` | queued |
| 7 | BakLLaVA | 7B | ~8 GB | Good for OCR / text reading | `ollama pull bakllava` | queued |

## Tier 2 — Already Downloaded (run immediately)

| # | Model | Params | VRAM | Notes | Ollama command |
|---|-------|--------|------|-------|----------------|
| 1 | LLaVA 1.5 7B | 7B | ~6 GB | Good, well-tested baseline | `ollama pull llava:7b` |
| 2 | Gemma 3 4B (vision) | 4B | ~3 GB | Multilingual, 35+ languages | `ollama pull gemma3:4b` |
| 3 | LLaVA-Phi3 | ~5B | ~6 GB | Good, fast | `ollama pull llava-phi3` |
| 4 | Moondream 2 | ~1.8B | ~2 GB | Lightweight, very fast | `ollama pull moondream` |

## Tier 3 — Tiny but Capable (priority for space-constrained systems)

| # | Model | Size | VRAM | Best For | Ollama command | Status |
|---|-------|------|------|----------|----------------|--------|
| 1 | German-OCR-3 ⭐ | 2.7 GB | ~4–6 GB | German receipts/invoices (100% JSON validity, 0% hallucination) | `ollama pull Keyvan/german-ocr-3` | queued |
| 2 | SmolVLM2-2.2B | 2.2B | ~4 GB | Document OCR, fast (30+ tok/s) | `ollama pull richardyoung/smolvlm2-2.2b-instruct` | queued |
| 3 | MiniCPM-o 2.6 4B | 4B | ~5 GB | High-res images (1.8M px), excellent OCR | `ollama pull minicpm-o:4b` | queued |

## Disk / VRAM Notes

- olmOCR-2 requires 8.8 GB VRAM — may overflow the A2000's 8 GB; test last.
- Llama 3.2 Vision 11B at 8 GB is borderline — use `q4_K_M` quantisation.
- **Disk is space-constrained** (~20 GB initially, consumed quickly by Tier 2 downloads).
  To free space for Tier 1/3 models, consider removing non-vision models:
  ```
  ollama rm qwen3.6:27b-q4_K_M   # frees 17 GB
  ```
- Download models one at a time and run benchmark incrementally if disk is tight.

## Benchmark Order (recommended)

Run Tier 2 first (already present), then Tier 3 (small, high value for German text),
then Tier 1 in ascending VRAM order: Qwen3-VL 4B → Qwen2.5-VL 7B → Qwen2-VL 7B →
MiniCPM-V → BakLLaVA → Llama 3.2 Vision → olmOCR-2.
