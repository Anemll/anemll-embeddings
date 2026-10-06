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
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/gen_reference_fixtures.py
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_reference_fixtures.py
```

Python env: `/Volumes/Models/anemll-embeddings/.venv` (sentence-transformers≥6.1, transformers, torch, pillow, torchvision — processor import still pulls image deps even for text-only). Pip cache may stay on TB36. A TB36 `.venv` was attempted but native `torch` imports crash with SIGBUS over SMB.

## Artifacts disk

`.mlpackage` / `.mlmodelc` / TorchScript / build outputs go under:

`ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts`

Do **not** put compile artifacts on TB36 (models + HF caches only), SAN512, or the internal SSD.

## Text-only loader + wrapper (T2 / T3)

- `src/load_text_model.py` — Sentence-Transformers load with `vision_config`/`audio_config=None`, BF16/FP32 only (refuses FP16)
- `src/embed_wrapper.py` — `EmbeddingGemma2Wrapper`: mask-aware mean pool → 512→768 projection → optional L2 (pool-then-project graph ready for later `torch.jit.trace` / coremltools)

Smoke vs T1 fixtures (cosine ≥ 0.999):

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/smoke_wrapper_vs_fixtures.py
```

## Trace + Core ML convert (T4 / T5)

Fixed-S `torch.jit.trace` then public `coremltools==9.0` (see `requirements-conversion.txt`). This is **not** `forge.py convert`.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_export_utils.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/export_torchscript.py --seq-len 512
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/convert_coreml.py --seq-len 512
```

Default `S=512` matches this checkpoint’s `text_config.sliding_window` (PLAN’s 1024 note is the later ladder). Artifacts land under `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512/` (gitignored).

Core ML I/O (names are fixed):

| Name | Role | Shape | Dtype |
| --- | --- | --- | --- |
| `input_ids` | token ids (host tokenizer + task prefix) | `[1, S]` | `int32` |
| `attention_mask` | `1` = token, `0` = pad | `[1, S]` | `int32` |
| `embedding` | L2-normalized 768-d vector | `[1, 768]` | `float32` |

`CPU_AND_NE` on convert does **not** prove ANE placement (T7). First package is FP32 compute (`ct.precision.FLOAT32`); FP16 is T10.

## Parity vs T1 fixtures (T6)

Core ML **CPU** vs committed `tests/fixtures/embeddings.npy` (cosine, rel-L2, L2 norms, pairwise `cos(q,d)`). This does **not** prove ANE.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_parity_metrics.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/parity_cosine.py --seq-len 512
```

Gates (PLAN starting numbers): per-prompt cosine ≥ 0.999, rel-L2 ≤ 0.05, pairwise |Δcos| ≤ 0.01. JSON report: `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512/embeddinggemma2-text-s512.parity.json`.

## ANE smoke + placement (T7)

Same S=512 `.mlpackage`, scored with `CPU_AND_NE`. **T6 CPU parity is not ANE proof.** T7 dumps `MLComputePlan` preferred devices and fails if zero non-const ops prefer ANE.

```sh
export HF_HOME=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export HUGGINGFACE_HUB_CACHE=/Volumes/TB36/Models/anemll-embeddings/hf-cache
export ANEMLL_EMBEDDINGS_MODEL=/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2
export ANEMLL_EMBEDDINGS_ARTIFACTS=/Volumes/Models/anemll-embeddings/artifacts
/Volumes/Models/anemll-embeddings/.venv/bin/python tests/test_ane_placement.py
/Volumes/Models/anemll-embeddings/.venv/bin/python scripts/ane_smoke.py --seq-len 512
```

Report: `$ANEMLL_EMBEDDINGS_ARTIFACTS/embeddinggemma2-text-s512/embeddinggemma2-text-s512.ane.json`.

## License

Apache License 2.0 — see `LICENSE.note`; EmbeddingGemma upstream terms also apply to model weights.
