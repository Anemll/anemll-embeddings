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

## Reference fixtures (T1)

Fixed Sentence-Transformers text-only embeddings for parity work live under `tests/fixtures/`:

- `prompts.json` — stable SearchQuery / Document / CodeRetrieval / SentenceSimilarity / Classification / Clustering set
- `embeddings.npy` — float32 `[N, 768]` ST reference vectors (L2-normalized)
- `reference_meta.json` — load dtype/device, package versions, per-vector SHA-256 digests

Regenerate (weights on TB36; runtime venv must be **local**, not SMB — torch SIGBUSes from TB36):

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
# Optional later: Core ML / compile outputs (not TB36, not SAN512, not internal SSD)
# export ANEMLL_EMBEDDINGS_ARTIFACTS=/path/to/artifacts-disk/anemll-embeddings
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/gen_reference_fixtures.py
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_reference_fixtures.py
```

Python env: `/Volumes/Models/anemll-embeddings/.venv` (sentence-transformers≥6.1, transformers, torch, pillow, torchvision — processor import still pulls image deps even for text-only). Pip cache may stay on TB36. A TB36 `.venv` was attempted but native `torch` imports crash with SIGBUS over SMB.

## Artifacts disk (placeholder)

`.mlpackage` / `.mlmodelc` / build outputs go under **`ANEMLL_EMBEDDINGS_ARTIFACTS`** once that volume is mounted. Do **not** put compile artifacts on TB36 (models+caches only), SAN512, or the internal SSD.

## License

Apache License 2.0 — see `LICENSE.note`; EmbeddingGemma upstream terms also apply to model weights.
