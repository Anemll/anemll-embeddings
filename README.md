# anemll-embeddings

Open-source **EmbeddingGemma 2** → **Core ML / Apple Neural Engine** via an ANEMLL-forge-style workflow.

## Goal

Convert Google’s EmbeddingGemma 2 text embedding models to run efficiently on Apple Silicon (Core ML → ANE), starting with the ~270M text-only checkpoint, then extending toward multimodal (vision/audio) where feasible.

## Sources

- Official model: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (prefer this checkpoint; ~1.53 GB safetensors)
- Community / Unsloth guide & variants: [unsloth/embeddinggemma-2](https://huggingface.co/unsloth/embeddinggemma-2), Unsloth EmbeddingGemma docs
- Conversion workflow reference (read-only): [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge) (`docs/WORKFLOW.md`, `forge.py`)

## Model weights location

**Weights are not stored in this git repo.** Keep this checkout lean (code + small configs only).

Canonical models directory: **TBD — network volume** (not the flash USB `/Volumes/SAN512`, which is too slow for large downloads). Do not download checkpoints onto the internal SSD either when free space is tight.

When a network volume path is confirmed, set Hugging Face caches there as well:

```sh
export HF_HOME=<NETWORK_VOLUME>/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=<NETWORK_VOLUME>/anemll-embeddings/hf-cache
```

A local `models` symlink to that path is optional and gitignored.

## License

Apache License 2.0 — see `LICENSE.note`; EmbeddingGemma upstream terms also apply to model weights.
