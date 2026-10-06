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

Canonical models directory (TrueNAS SMB volume **TB36**):

`/Volumes/TB36/Models/anemll-embeddings`

- Checkpoint: `/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2` (`google/embeddinggemma-2`, ~1.5 GB safetensors)
- Do **not** use flash USB `/Volumes/SAN512` for new downloads (slow). A prior incomplete copy may still exist at `/Volumes/SAN512/MODELS/anemll-embeddings` (~702M partial); leave it unless cleaning up deliberately.
- Prefer not filling the internal SSD when free space is tight.

Set Hugging Face caches on TB36:

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
```

A local `models` symlink to that path is optional and gitignored.

## License

Apache License 2.0 — see `LICENSE.note`; EmbeddingGemma upstream terms also apply to model weights.
