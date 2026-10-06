# anemll-embeddings

Open-source **EmbeddingGemma 2** → **Core ML / Apple Neural Engine** via an ANEMLL-forge-style workflow.

## Goal

Convert Google’s EmbeddingGemma 2 text embedding models to run efficiently on Apple Silicon (Core ML → ANE), starting with the ~270M text-only checkpoint, then extending toward multimodal (vision/audio) where feasible.

## Sources

- Official model: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (prefer this checkpoint)
- Community / Unsloth guide & variants: [unsloth/embeddinggemma-2](https://huggingface.co/unsloth/embeddinggemma-2), Unsloth EmbeddingGemma docs
- Conversion workflow reference (read-only): [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge) (`docs/WORKFLOW.md`, `forge.py`)

## Model weights location

**Weights are not stored in this git repo.** On development machines they live on external storage:

```text
/Volumes/SAN512/MODELS/anemll-embeddings
```

A local `models` symlink may point at that path. See `.gitignore`. Set Hugging Face caches there as well:

```sh
export HF_HOME=/Volumes/SAN512/MODELS/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/SAN512/MODELS/anemll-embeddings/hf-cache
```

## License

Apache License 2.0 — see `LICENSE` when added; EmbeddingGemma upstream terms also apply to model weights.
