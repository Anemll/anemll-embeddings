# Conversion plan (stub)

## Phase 1 — Text-only EmbeddingGemma 2 (~270M) → Core ML → ANE

1. **Obtain checkpoint** — Prefer official `google/embeddinggemma-2` safetensors (text backbone, ~1.53 GB). Store on **TBD network volume** (not flash USB SAN512; not internal SSD when space is tight).
2. **Inspect architecture** — Encoder / pooling / projection dims; no autoregressive LM head. Map I/O: `input_ids` (+ optional attention mask) → fixed-dim embedding vector.
3. **Export** — Torch/FX or coremltools-friendly graph; freeze for inference; FP16 baseline first.
4. **Convert** — `coremltools` → `.mlpackage` (compute units ANE/CPU+ANE). Quantize later (INT8 / LUT) if quality allows.
5. **Compile / smoke** — Load on device, measure latency vs PyTorch reference cosine similarity on a small eval set.
6. **Gaps vs anemll-forge** — Forge today is LLM-oriented (quantize → chunked Core ML/Core AI → KV cache → chat/serve). Embeddings need a thinner path: single forward (or mean-pool) encoder, no KV cache / drafter / speculative decode. Reuse forge lessons on ANE ops, chunking, and compile mode; do not expect `forge.py convert` to work unmodified.

## Phase 2 — Multimodal (later)

- Vision / audio towers only if network-volume space and ANE op support allow.
- Prefer separate Core ML packages per modality + shared text tower when possible.

## Open questions

- Confirmed network volume path for weights (replace TBD).
- Exact HF repo id / file layout for text-only vs full multimodal dump.
- Pooling strategy (CLS / mean / last) required for API parity with Sentence-Transformers / Unsloth guides.
- ANE blockers: unsupported ops in embedding models (e.g. some attention variants, large matmuls without tiling).
